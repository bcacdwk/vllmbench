# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Linear Method

本模块是 SlideSparse FP8 的核心，通过外挂方式替换 vLLM 的 FP8 Linear 计算路径。

架构说明
========
SlideSparse 通过包装 vLLM 原有的 CompressedTensorsW8A8Fp8 scheme 实现外挂：
- create_weights: 委托给原始 scheme
- process_weights_after_loading: 委托给原始 scheme + cuSPARSELt 在线压缩
- apply_weights: 替换为 SlideSparse 的 kernel 路径

三条 Kernel 路径（通过环境变量选择）
====================================
1. CUTLASS (默认 fallback)
   - 直接调用 vLLM 的 cutlass_scaled_mm，融合 GEMM + dequant + bias
   - 权重形状: [K, N]（vLLM 转置后）
   
2. cuBLASLt (USE_CUBLASLT=1)
   - GEMM: cuBLASLt FP8 矩阵乘法（无 scale/bias 融合）
   - Dequant+Bias: 外挂 Triton kernel
   - 权重形状: [N, K]（跳过 vLLM 转置，保持原始行主序）
   
3. cuSPARSELt (USE_CUSPARSELT=1)
   - GEMM: cuSPARSELt 2:4 稀疏 FP8 矩阵乘法（无 scale/bias 融合）
   - Dequant+Bias: 外挂 Triton kernel
   - 权重形状: weight_compressed [compressed_size] uint8 1D（在线压缩后）
   - 需要配置 SPARSITY 环境变量（默认 2_8）

维度命名约定
============
GEMM: output[M, N] = input[M, K] @ weight[K, N]

cuBLASLt 路径:
- M, K, N: 算法维度（GEMM 的语义维度）
- M_pad: M 的 16 对齐版本
- K_pad: K 的 32 对齐版本

cuSPARSELt 路径:
- M, K, N: 原始算法维度
- K_slide: slide 扩展后的 K 维度（K_slide = K * expand_ratio）
- M_pad: M 的 16 对齐版本
- K_slide_pad: K_slide 的 32 对齐版本
- weight_compressed: cuSPARSELt 压缩后的 1D uint8 tensor

Padding 策略:
- M_pad: Quant kernel 内部完成 16 对齐，输出 [M_pad, K_pad] 或 [M_pad, K_slide_pad]
- K_pad/K_slide_pad: Quant kernel 内部完成 32 对齐
- GEMM 在 padded 维度上计算，Dequant 前截断回原始 M

环境变量
========
- DISABLE_SLIDESPARSE=1   : 完全禁用 SlideSparse，使用 vLLM 原生路径
- USE_CUBLASLT=1          : 使用 cuBLASLt kernel
- USE_CUSPARSELT=1        : 使用 cuSPARSELt kernel（与 USE_CUBLASLT 互斥）
- INNER_DTYPE_32=1        : GEMM 使用高精度累加（FP8→FP32）
- SPARSITY=2_L            : 稀疏格式（仅 cuSPARSELt 时生效，L=4,6,8,10,... 默认 2_8）
- SLIDESPARSE_PROFILE=1   : 启用 SlideSparse 计时诊断
"""

from typing import Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from slidesparse.utils import SlideSparseConfig, compute_output_k

from .config import (
    is_slidesparse_enabled,
    is_cublaslt_enabled,
    is_cusparselt_enabled,
    is_inner_dtype_32,
    get_sparsity_config,
)
from .profiler import ProfileTimer, profile_step
from .gemm_wrapper import (
    cublaslt_fp8_mm_op,
    cusparselt_fp8_mm_op,
    _get_gemm_extension,
    cuBLASLtGemmWrapper,
    cuSPARSELtGemmWrapper,
    get_algo_config_manager,
)
from .kernels import (
    dequant_bias_kernel,
    _load_dequant_bias_kernel,
    quant_only_fp8_kernel,
    _load_quant_only_fp8_kernel,
    quant_slide_fp8_kernel,
    _load_quant_slide_fp8_kernel,
)


logger = init_logger(__name__)


# ============================================================================
# 辅助函数：获取当前模型名
# ============================================================================

def _get_current_model_name() -> str:
    """
    从 AlgorithmConfigManager 获取当前基础模型名（不带 -SlideSparse- 后缀）
    
    返回的名字直接对应:
    - Triton kernel 文件名后缀
    - GEMM 配置 JSON 中的 model_name
    
    如果没有设置，抛出明确的错误提示
    """
    manager = get_algo_config_manager()
    model_name = manager.get_model_name()
    if model_name is None:
        raise ValueError(
            "Model name not set. Call slidesparse.init_slidesparse(model_name) first.\n"
            "Example: from slidesparse import init_slidesparse; init_slidesparse('Llama3.2-1B-FP8')"
        )
    return model_name


# ============================================================================
# FP8 Linear 函数（三条 Kernel 路径）
# ============================================================================
#
# 三个函数内部完成 quant + GEMM + dequant:
#   - cuBLASLt_FP8_linear:   quant_only + cuBLASLt dense GEMM + Triton dequant
#   - cuSPARSELt_FP8_linear: quant_slide + cuSPARSELt 2:4 sparse GEMM + Triton dequant
#   - cutlass_FP8_linear:    vLLM QuantFP8 + cutlass_scaled_mm (融合 dequant)
#
# cuBLASLt 和 cuSPARSELt 计算流程:
#   1. Quant:   qinput[M,K], scale_a = quant_only/quant_slide(input)
#   2. GEMM:    inner[M,N] = weight @ qinput
#   3. Dequant: out[M,N] = inner * scale_a * scale_b + bias
#
#   4. Padding 处理:
#   - Quant kernel 输出 padded 维度
#   - GEMM 在 padded 维度计算
#   - Dequant 前截断回原始 M（切片无数据拷贝）
# ============================================================================

def cuBLASLt_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    inner_dtype_str: str,
    model_name: str,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    cuBLASLt FP8 GEMM + Triton Dequant
    
    数据流:
        input[M, K] BF16
            ↓ quant_only_fp8_kernel
        qinput[M_pad, K_pad] FP8, scale_a[M_pad]
            ↓ cublaslt_fp8_mm
        gemm_out[M_pad, N] BF16/FP32
            ↓ 截断 [:M, :]
        gemm_out[M, N], scale_a[M]
            ↓ dequant_bias_kernel
        output[M, N] out_dtype
    """
    M = input.shape[0]
    
    # Quant: [M, K] -> [M_pad, K_pad]
    # cuBLASLt 路径始终使用 Triton quant kernel（需要 padding）
    if input.dtype != current_platform.fp8_dtype():
        with ProfileTimer("cuBLASLt.quant"):
            qinput, scale_a_pad = quant_only_fp8_kernel(input, model_name)
    else:
        # 静态量化：input 已是 FP8，但没有 padding
        # cuBLASLt GEMM wrapper 期望 padded 维度，所以不支持静态量化
        raise NotImplementedError(
            "cuBLASLt with static quantization is not supported. "
            "Use CUTLASS path or dynamic quantization."
        )
    
    # GEMM: [M_pad, K_pad] @ [N, K_pad].T -> [M_pad, N]
    with ProfileTimer("cuBLASLt.gemm"):
        gemm_out_pad = cublaslt_fp8_mm_op(weight, qinput, inner_dtype_str)
    
    # Truncate padding
    gemm_out = gemm_out_pad[:M, :]
    scale_a = scale_a_pad[:M]
    
    # Dequant + Bias
    with ProfileTimer("cuBLASLt.dequant"):
        output = dequant_bias_kernel(gemm_out, scale_a, weight_scale, bias, out_dtype, model_name)
    
    with ProfileTimer("cuBLASLt.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


def cuSPARSELt_FP8_linear(
    *,
    input: torch.Tensor,
    slide_weight_compressed: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    inner_dtype_str: str,
    model_name: str,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    slide_weight_N: Optional[int] = None,
    slide_weight_K: Optional[int] = None,
    L: int = 8,
) -> torch.Tensor:
    """
    cuSPARSELt 2:4 Sparse FP8 GEMM + Triton Dequant
    
    数据流:
        input[M, K] BF16
            ↓ quant_slide_fp8_kernel
        qinput[M_pad, K_slide_pad] FP8, scale_a[M_pad]
            ↓ cusparselt_fp8_mm
        gemm_out[M_pad, N] BF16/FP32
            ↓ 截断 [:M, :]
        gemm_out[M, N], scale_a[M]
            ↓ dequant_bias_kernel
        output[M, N] out_dtype
    
    Args:
        slide_weight_compressed: [compressed_size] uint8 1D
        slide_weight_N: 权重 N 维度
        slide_weight_K: 权重 K_slide 维度（slide 扩展后，已 32 对齐）
        L: 稀疏组大小（默认 8）
    """
    
    if slide_weight_N is None or slide_weight_K is None:
        raise ValueError(
            "cuSPARSELt requires slide_weight_N and slide_weight_K."
        )
    
    M = input.shape[0]
    
    # Quant + Slide: [M, K] -> [M_pad, K_slide_pad]
    if input.dtype != current_platform.fp8_dtype():
        with ProfileTimer("cuSPARSELt.quant_slide"):
            qinput, scale_a_pad = quant_slide_fp8_kernel(input, model_name, L)
    else:
        raise NotImplementedError(
            "cuSPARSELt with static quantization is not supported yet."
        )
    
    # 验证维度一致性：qinput 的 K 维度应与 weight 的 K 维度匹配
    K_slide_pad = qinput.shape[1]
    if K_slide_pad != slide_weight_K:
        raise ValueError(
            f"K dimension mismatch: qinput.shape[1]={K_slide_pad}, "
            f"slide_weight_K={slide_weight_K}. "
            "This may indicate L parameter mismatch between weight and activation."
        )
    
    # GEMM: [M_pad, K_slide_pad] @ compressed_weight -> [M_pad, N]
    with ProfileTimer("cuSPARSELt.gemm"):
        gemm_out_pad = cusparselt_fp8_mm_op(
            slide_weight_compressed,
            qinput,
            slide_weight_N,
            K_slide_pad,
            inner_dtype_str
        )
    
    # Truncate padding
    gemm_out = gemm_out_pad[:M, :]
    scale_a = scale_a_pad[:M]
    
    # Dequant + Bias
    with ProfileTimer("cuSPARSELt.dequant"):
        output = dequant_bias_kernel(gemm_out, scale_a, weight_scale, bias, out_dtype, model_name)
    
    with ProfileTimer("cuSPARSELt.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


def cutlass_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: QuantFP8,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    vLLM CUTLASS（融合 GEMM + Dequant + Bias）
    
    数据流:
        input[M, K] BF16
            ↓ QuantFP8
        qinput[M, K] FP8, scale_a
            ↓ cutlass_scaled_mm（融合 dequant + bias）
        output[M, N] out_dtype
    """
    # Quant（使用 vLLM 原生 QuantFP8）
    if input.dtype != current_platform.fp8_dtype():
        with ProfileTimer("CUTLASS.quant"):
            qinput, scale_a = quant_fn(input, input_scale, input_scale_ub)
    else:
        qinput, scale_a = input, input_scale
    
    # CUTLASS 融合 GEMM + Dequant + Bias
    with ProfileTimer("CUTLASS.scaled_mm"):
        output = ops.cutlass_scaled_mm(
            qinput, weight, out_dtype=out_dtype,
            scale_a=scale_a, scale_b=weight_scale, bias=bias
        )
    
    with ProfileTimer("CUTLASS.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


# ============================================================================
# SlideSparse FP8 Linear Op
# ============================================================================

class SlideSparseFp8LinearOp:
    """
    SlideSparse FP8 Linear Operation
    
    根据环境变量选择 kernel 路径：
    - USE_CUBLASLT=1: cuBLASLt_FP8_linear
    - USE_CUSPARSELT=1: cuSPARSELt_FP8_linear
    - 默认: cutlass_FP8_linear
    
    注意: Triton kernel 采用懒加载模式，首次调用 apply 时才加载，
    因为 kernel 是 model-specific 的，需要知道当前模型名。
    """
    
    def __init__(
        self,
        act_quant_static: bool = False,
        act_quant_group_shape: GroupShape = GroupShape.PER_TOKEN,
    ):
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        
        # 创建 QuantFP8 实例（CUTLASS 路径使用）
        self.quant_fp8 = QuantFP8(
            static=act_quant_static,
            group_shape=act_quant_group_shape,
            num_token_padding=None,
        )
        
        # 确定 kernel 路径（缓存环境变量判断结果）
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # 缓存 inner_dtype（环境变量在进程生命周期内不变）
        self._inner_dtype_str = "fp32" if is_inner_dtype_32() else "bf16"
        
        # 只预加载 GEMM extension（model-agnostic）
        # Triton kernel 懒加载（需要 model_name）
        if self._use_cublaslt:
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_FP8_linear
            _get_gemm_extension("cublaslt")
        elif self._use_cusparselt:
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_FP8_linear
            _get_gemm_extension("cusparselt")
        else:
            self._kernel_name = "CUTLASS"
            self._linear_fn = cutlass_FP8_linear
        
        logger.info_once(
            f"SlideSparseFp8LinearOp initialized (kernel={self._kernel_name})"
        )
    
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_scale_ub: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        slide_weight_N: int | None = None,
        slide_weight_K: int | None = None,
        L: int = 8,
    ) -> torch.Tensor:
        """
        执行 FP8 Linear 操作
        
        Args:
            input: [..., K] BF16
            weight: 权重（形状取决于 kernel 路径）
            weight_scale: [N] FP32
            out_dtype: 输出类型
            input_scale: 静态量化 scale
            input_scale_ub: 静态量化 scale 上界
            bias: [N]
            slide_weight_N: cuSPARSELt 专用，N 维度
            slide_weight_K: cuSPARSELt 专用，K_slide 维度
            L: cuSPARSELt 专用，稀疏组大小
        """
        # 获取 input 形状信息
        input_shape = input.shape
        input_ndim = input.dim()
        
        # 展平为 2D（如果已经是 2D 则直接使用，避免不必要的 view 调用）
        if input_ndim == 2:
            input_2d = input
            M = input_shape[0]
        else:
            input_2d = input.view(-1, input_shape[-1])
            M = input_2d.shape[0]
        
        # 推断输出 N 维度（根据已缓存的 kernel 路径判断，避免 weight.dim() 调用）
        # - cuSPARSELt: N 由 slide_weight_N 参数提供（weight 是压缩后的 1D）
        # - cuBLASLt:   weight [N, K]，N 在 dim=0
        # - CUTLASS:    weight [K, N]，N 在 dim=1
        if self._use_cusparselt:
            output_N = slide_weight_N  # 调用者保证 slide_weight_N 有效
        elif self._use_cublaslt:
            output_N = weight.shape[0]
        else:
            output_N = weight.shape[1]
        
        # 构建 output_shape（针对常见的 2D 输入优化，避免 list unpacking）
        if input_ndim == 2:
            output_shape = [M, output_N]
        else:
            output_shape = [*input_shape[:-1], output_N]
        
        if out_dtype is None:
            out_dtype = input.dtype
        
        # 公共参数
        common_args = dict(
            input=input_2d,
            out_dtype=out_dtype,
            weight_scale=weight_scale,
            bias=bias,
            output_shape=output_shape,
            input_scale=input_scale,
            input_scale_ub=input_scale_ub,
        )
        
        # 调用选定的 kernel 路径
        if self._use_cusparselt:
            # cuSPARSELt 需要 model_name 加载 Triton kernels
            model_name = _get_current_model_name()
            return self._linear_fn(
                **common_args,
                slide_weight_compressed=weight,
                inner_dtype_str=self._inner_dtype_str,
                model_name=model_name,
                slide_weight_N=slide_weight_N,
                slide_weight_K=slide_weight_K,
                L=L,
            )
        elif self._use_cublaslt:
            # cuBLASLt 需要 model_name 加载 Triton kernels
            model_name = _get_current_model_name()
            return self._linear_fn(
                **common_args,
                weight=weight,
                inner_dtype_str=self._inner_dtype_str,
                model_name=model_name,
            )
        else:
            # CUTLASS 路径不需要 model_name（使用 vLLM 原生 kernel）
            return self._linear_fn(
                **common_args,
                weight=weight,
                quant_fn=self.quant_fp8,
            )


# ============================================================================
# SlideSparse FP8 Linear Method
# ============================================================================

class SlideSparseFp8LinearMethod:
    """
    SlideSparse FP8 Linear Method
    
    包装 vLLM 原有的 CompressedTensorsW8A8Fp8 scheme：
    - create_weights: 委托给原始 scheme
    - process_weights_after_loading: 委托 + cuBLASLt/cuSPARSELt 后处理
    - apply_weights: 使用 SlideSparseFp8LinearOp
    
    权重形状变化:
        原始 checkpoint: [N, K] 或 [N, K_slide]（slidesparse checkpoint）
        vLLM 加载后: [N, K] 或 [N, K_slide]
        CUTLASS 路径: weight.t() -> [K, N]
        cuBLASLt 路径: 保持 [N, K]
        cuSPARSELt 路径: [N, K_slide] -> compress -> [compressed_size] uint8 1D
    """
    
    def __init__(self, original_scheme):
        self.original_scheme = original_scheme
        self.out_dtype = original_scheme.out_dtype
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.act_q_group_shape = original_scheme.act_q_group_shape
        self.strategy = original_scheme.strategy
        
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # 创建 SlideSparse Op
        self.slidesparse_fp8_linear = SlideSparseFp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape,
        )
        
        # cuSPARSELt 稀疏配置
        if self._use_cusparselt:
            Z, L, self._expand_ratio = get_sparsity_config()
            self._sparsity_config = SlideSparseConfig(Z=Z, L=L)
            logger.info_once(
                f"SlideSparseFp8LinearMethod: cuSPARSELt "
                f"sparsity={Z}:{L}, expand_ratio={self._expand_ratio:.3f}"
            )
        
        # 预加载 Triton kernels（torch.compile 兼容）
        import os
        model_name = os.environ.get("SLIDESPARSE_MODEL_NAME")
        if model_name and (self._use_cublaslt or self._use_cusparselt):
            # dequant_bias kernel 是 cuBLASLt 和 cuSPARSELt 共享的
            _load_dequant_bias_kernel(model_name)
            
            if self._use_cublaslt:
                _load_quant_only_fp8_kernel(model_name)
            elif self._use_cusparselt:
                _load_quant_slide_fp8_kernel(model_name)
            
            logger.info_once(
                f"Preloaded FP8 Triton kernels for model: {model_name}"
            )
        
        logger.info_once(
            f"SlideSparseFp8LinearMethod initialized, "
            f"kernel={self.slidesparse_fp8_linear._kernel_name}"
        )
    
    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        """
        创建权重参数
        
        cuSPARSELt 路径：扩展 input_size 以匹配 slide 后的 checkpoint 权重
        其他路径：直接委托给原始 scheme
        """
        if self._use_cusparselt:
            # cuSPARSELt: 扩展 input_size 以匹配 slide 后的 K 维度
            # checkpoint 中的权重已经是 [N, K_slide]，需要匹配
            _, input_size_per_partition_slide = compute_output_k(
                input_size_per_partition, self._sparsity_config
            )
            _, input_size_slide = compute_output_k(
                input_size, self._sparsity_config
            )
            
            return self.original_scheme.create_weights(
                layer=layer,
                input_size_per_partition=input_size_per_partition_slide,
                output_partition_sizes=output_partition_sizes,
                input_size=input_size_slide,
                output_size=output_size,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
        else:
            # CUTLASS / cuBLASLt: 直接委托
            return self.original_scheme.create_weights(
                layer=layer,
                input_size_per_partition=input_size_per_partition,
                output_partition_sizes=output_partition_sizes,
                input_size=input_size,
                output_size=output_size,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
    
    def process_weights_after_loading(self, layer: Module) -> None:
        """
        权重加载后处理
        
        处理逻辑:
        - CUTLASS: 委托给原始 scheme（执行 weight.t()）
        - cuBLASLt: 原始 scheme + 转置回 [N, K]
        - cuSPARSELt: 原始 scheme + 转置回 [N, K_slide] + 在线压缩
        """
        # 所有路径都先调用原始 scheme
        self.original_scheme.process_weights_after_loading(layer)
        
        if not self._use_cublaslt and not self._use_cusparselt:
            # CUTLASS 路径：直接返回
            return
        
        # cuBLASLt / cuSPARSELt：转置回 [N, K] 或 [N, K_slide]
        from torch.nn import Parameter
        weight_transposed = layer.weight.data.t()
        layer.weight = Parameter(weight_transposed, requires_grad=False)
        
        if self._use_cusparselt:
            self._compress_weight_online(layer)
    
    def _compress_weight_online(self, layer: Module) -> None:
        """
        cuSPARSELt 在线压缩
        
        输入: layer.weight [N, K_slide] FP8
        输出: layer.weight [compressed_size] uint8 1D
              layer.slide_weight_N: N
              layer.slide_weight_K: K_slide
        """
        from torch.nn import Parameter
        
        try:
            from slidesparse.weight_convert.compress import compress_tensor_online
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import compress_tensor_online: {e}"
            ) from e
        
        slide_weight = layer.weight.data
        N, K_slide = slide_weight.shape
        
        logger.info_once(
            f"cuSPARSELt compression: [{N}, {K_slide}] -> 1D uint8"
        )
        
        weight_compressed = compress_tensor_online(slide_weight, verbose=False)
        
        layer.weight = Parameter(weight_compressed, requires_grad=False)
        layer.slide_weight_N = N
        layer.slide_weight_K = K_slide
        
        logger.info_once(
            f"cuSPARSELt compression done: {weight_compressed.numel()} bytes"
        )
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply weights (linear transformation)"""
        input_scale = getattr(layer, "input_scale", None)
        input_scale_ub = getattr(layer, "input_scale_ub", None)
        
        if self._use_cusparselt:
            return self.slidesparse_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=self.out_dtype,
                input_scale=input_scale,
                input_scale_ub=input_scale_ub,
                bias=bias,
                slide_weight_N=layer.slide_weight_N,
                slide_weight_K=layer.slide_weight_K,
                L=self._sparsity_config.L,
            )
        else:
            return self.slidesparse_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=self.out_dtype,
                input_scale=input_scale,
                input_scale_ub=input_scale_ub,
                bias=bias,
            )


# ============================================================================
# 工厂函数
# ============================================================================

def wrap_scheme_fp8(original_scheme):
    """
    FP8 scheme 包装入口
    
    只包装 W8A8Fp8 scheme，其他 scheme 原样返回。
    内部由 SlideSparseFp8LinearOp 根据环境变量选择 kernel 路径。
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Fp8" not in scheme_name:
        logger.warning_once(
            f"SlideSparse not supported for {scheme_name}, using original"
        )
        return original_scheme
    
    if is_cublaslt_enabled():
        backend = "cuBLASLt"
    elif is_cusparselt_enabled():
        backend = "cuSPARSELt"
    else:
        backend = "CUTLASS"
    
    logger.info_once(
        f"Wrapping {scheme_name} with SlideSparse ({backend})"
    )
    return SlideSparseFp8LinearMethod(original_scheme)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Linear 函数
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    
    # Op 和 Method 类
    "SlideSparseFp8LinearOp",
    "SlideSparseFp8LinearMethod",
    
    # 工厂函数
    "wrap_scheme_fp8",
]
