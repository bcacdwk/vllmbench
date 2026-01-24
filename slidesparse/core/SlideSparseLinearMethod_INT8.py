# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse INT8 Linear Method

本模块是 SlideSparse INT8 的核心，通过外挂方式替换 vLLM 的 INT8 Linear 计算路径。

架构说明
========
SlideSparse 通过包装 vLLM 原有的 CompressedTensorsW8A8Int8 scheme 实现外挂：
- create_weights: 委托给原始 scheme
- process_weights_after_loading: 委托给原始 scheme + cuSPARSELt 在线压缩
- apply_weights: 替换为 SlideSparse 的 kernel 路径

三条 Kernel 路径（通过环境变量选择）
====================================
1. CUTLASS (默认 fallback)
   - 直接调用 vLLM 的 cutlass_scaled_mm / cutlass_scaled_mm_azp
   - 融合 GEMM + dequant + bias
   - 支持对称和非对称量化
   - 权重形状: [K, N]（vLLM 转置后）
   
2. cuBLASLt (USE_CUBLASLT=1)
   - GEMM: cuBLASLt INT8 矩阵乘法（输出固定为 INT32）
   - Dequant+Bias: 外挂 Triton kernel
   - 权重形状: [N, K]（跳过 vLLM 转置，保持原始行主序）
   - 注意：cuBLASLt INT8 只支持对称量化
   
3. cuSPARSELt (USE_CUSPARSELT=1)
   - GEMM: cuSPARSELt 2:4 稀疏 INT8 矩阵乘法
   - Dequant+Bias: 外挂 Triton kernel
   - 权重形状: weight_compressed [compressed_size] uint8 1D（在线压缩后）
   - 注意：cuSPARSELt INT8 只支持对称量化，输出可以是 BF16 或 INT32（不支持 FP32）

维度命名约定
============
GEMM: output[M, N] = input[M, K] @ weight[K, N]

cuBLASLt 路径:
- M, K, N: 算法维度（GEMM 的语义维度）
- M_pad: M 的 16 对齐版本
- K_pad: K 的 32 对齐版本
- 输出固定为 INT32

cuSPARSELt 路径:
- M, K, N: 原始算法维度
- K_slide: slide 扩展后的 K 维度
- M_pad: M 的 16 对齐版本
- K_slide_pad: K_slide 的 32 对齐版本
- 输出可以是 BF16 或 INT32

INT8 vs FP8 关键差异
====================
1. cuBLASLt INT8 输出固定为 INT32（FP8 可选 BF16/FP32）
2. cuSPARSELt INT8 不支持 FP32 输出（FP8 支持）
3. INT8 支持非对称量化（需要 azp），但 cuBLASLt/cuSPARSELt 路径不支持
4. vLLM 使用 ops.scaled_int8_quant 而非 QuantFP8

环境变量
========
- DISABLE_SLIDESPARSE=1   : 完全禁用 SlideSparse，使用 vLLM 原生路径
- USE_CUBLASLT=1          : 使用 cuBLASLt kernel
- USE_CUSPARSELT=1        : 使用 cuSPARSELt kernel（与 USE_CUBLASLT 互斥）
- INNER_DTYPE_32=1        : GEMM 使用高精度累加（INT8→INT32）
- SPARSITY=2_L            : 稀疏格式（仅 cuSPARSELt 时生效）
- SLIDESPARSE_PROFILE=1   : 启用 SlideSparse 计时诊断
"""

from typing import Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger

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
    cublaslt_int8_mm_op,
    cusparselt_int8_mm_op,
    _get_gemm_extension,
    get_algo_config_manager,
)
from .kernels import (
    dequant_bias_kernel,
    quant_only_int8_kernel,
    quant_slide_int8_kernel,
)


logger = init_logger(__name__)


# ============================================================================
# 辅助函数：获取当前模型名
# ============================================================================

def _get_current_model_name() -> str:
    """
    从 AlgorithmConfigManager 获取当前模型名
    
    如果没有设置，抛出明确的错误提示
    """
    manager = get_algo_config_manager()
    model_name = manager.get_model()
    if model_name is None:
        raise ValueError(
            "Model name not set. Call slidesparse.init_slidesparse(model_name) first.\n"
            "Example: from slidesparse import init_slidesparse; init_slidesparse('Llama3.2-1B-FP8')"
        )
    return model_name


# ============================================================================
# INT8 Linear 函数（三条 Kernel 路径）
# ============================================================================
#
# 三个函数内部完成 quant + GEMM + dequant:
#   - cuBLASLt_INT8_linear:   quant_only + cuBLASLt dense GEMM + Triton dequant
#   - cuSPARSELt_INT8_linear: quant_slide + cuSPARSELt 2:4 sparse GEMM + Triton dequant
#   - cutlass_INT8_linear:    vLLM ops.scaled_int8_quant + cutlass_scaled_mm
#
# cuBLASLt 和 cuSPARSELt 计算流程:
#   1. Quant:   qinput[M,K], scale_a = quant_only/quant_slide(input)
#   2. GEMM:    inner[M,N] = weight @ qinput
#   3. Dequant: out[M,N] = inner * scale_a * scale_b + bias
#
# CUTLASS 计算流程:
#   1. Quant:   qinput, scale_a, azp = ops.scaled_int8_quant(input, ...)
#   2. GEMM:    output = cutlass_scaled_mm[_azp](qinput, weight, scale_a, scale_b, ...)
# ============================================================================

def cuBLASLt_INT8_linear(
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
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    input_symmetric: bool = True,
) -> torch.Tensor:
    """
    cuBLASLt INT8 GEMM + Triton Dequant
    
    数据流:
        input[M, K] BF16
            ↓ quant_only_int8_kernel (对称量化)
        qinput[M_pad, K_pad] INT8, scale_a[M_pad]
            ↓ cublaslt_int8_mm (输出固定为 INT32)
        gemm_out[M_pad, N] INT32
            ↓ 截断 [:M, :]
        gemm_out[M, N], scale_a[M]
            ↓ dequant_bias_kernel
        output[M, N] out_dtype
    
    Note:
        cuBLASLt INT8 只支持对称量化。
        非对称量化请使用 CUTLASS 路径。
    """
    # cuBLASLt 只支持对称量化
    if not input_symmetric:
        raise NotImplementedError(
            "cuBLASLt INT8 does not support asymmetric quantization. "
            "Use CUTLASS path for asymmetric quantization."
        )
    
    M = input.shape[0]
    
    # Quant: [M, K] -> [M_pad, K_pad]
    # cuBLASLt 路径使用 Triton INT8 quant kernel（对称量化）
    if input.dtype != torch.int8:
        with ProfileTimer("cuBLASLt.quant"):
            qinput, scale_a_pad = quant_only_int8_kernel(input, model_name)
    else:
        # 静态量化：input 已是 INT8，但没有 padding
        raise NotImplementedError(
            "cuBLASLt with static quantization is not supported. "
            "Use CUTLASS path or dynamic quantization."
        )
    
    try:
        # GEMM: [M_pad, K_pad] @ [N, K_pad].T -> [M_pad, N] INT32
        # 注意：cuBLASLt INT8 输出固定为 INT32，inner_dtype_str 被忽略
        with ProfileTimer("cuBLASLt.gemm"):
            gemm_out_pad = cublaslt_int8_mm_op(weight, qinput, "int32")
        
        # Truncate padding
        gemm_out = gemm_out_pad[:M, :]
        scale_a = scale_a_pad[:M]
        
        # Dequant + Bias: INT32 -> out_dtype
        with ProfileTimer("cuBLASLt.dequant"):
            output = dequant_bias_kernel(gemm_out, scale_a, weight_scale, bias, out_dtype, model_name)
        
        with ProfileTimer("cuBLASLt.view"):
            result = output.view(*output_shape)
        
        profile_step()
        return result
    except Exception as e:
        raise RuntimeError(f"cuBLASLt INT8 execution failed: {e}") from e


def cuSPARSELt_INT8_linear(
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
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    input_symmetric: bool = True,
    slide_weight_N: Optional[int] = None,
    slide_weight_K: Optional[int] = None,
    L: int = 8,
) -> torch.Tensor:
    """
    cuSPARSELt 2:4 Sparse INT8 GEMM + Triton Dequant
    
    数据流:
        input[M, K] BF16
            ↓ quant_slide_int8_kernel (对称量化)
        qinput[M_pad, K_slide_pad] INT8, scale_a[M_pad]
            ↓ cusparselt_int8_mm
        gemm_out[M_pad, N] BF16/INT32
            ↓ 截断 [:M, :]
        gemm_out[M, N], scale_a[M]
            ↓ dequant_bias_kernel
        output[M, N] out_dtype
    
    Args:
        slide_weight_compressed: [compressed_size] uint8 1D
        slide_weight_N: 权重 N 维度
        slide_weight_K: 权重 K_slide 维度
        L: 稀疏组大小（默认 8）
    
    Note:
        cuSPARSELt INT8 只支持对称量化。
        cuSPARSELt INT8 不支持 FP32 输出，只能是 BF16 或 INT32。
    """
    # cuSPARSELt 只支持对称量化
    if not input_symmetric:
        raise NotImplementedError(
            "cuSPARSELt INT8 does not support asymmetric quantization. "
            "Use CUTLASS path for asymmetric quantization."
        )
    
    if slide_weight_N is None or slide_weight_K is None:
        raise ValueError(
            "cuSPARSELt requires slide_weight_N and slide_weight_K."
        )
    
    # cuSPARSELt INT8 不支持 FP32 输出
    if inner_dtype_str == "fp32":
        raise ValueError(
            "cuSPARSELt INT8 does not support FP32 output. "
            "Use 'bf16' (default) or 'int32'. "
            "Set INNER_DTYPE_32=1 to use INT32 output."
        )
    
    M = input.shape[0]
    
    # Quant + Slide: [M, K] -> [M_pad, K_slide_pad]
    if input.dtype != torch.int8:
        with ProfileTimer("cuSPARSELt.quant_slide"):
            qinput, scale_a_pad = quant_slide_int8_kernel(input, model_name, L)
    else:
        raise NotImplementedError(
            "cuSPARSELt with static quantization is not supported yet."
        )
    
    # 验证维度一致性
    K_slide_pad = qinput.shape[1]
    if K_slide_pad != slide_weight_K:
        raise ValueError(
            f"K dimension mismatch: qinput.shape[1]={K_slide_pad}, "
            f"slide_weight_K={slide_weight_K}. "
            "This may indicate L parameter mismatch between weight and activation."
        )
    
    try:
        # GEMM: [M_pad, K_slide_pad] @ compressed_weight -> [M_pad, N]
        with ProfileTimer("cuSPARSELt.gemm"):
            gemm_out_pad = cusparselt_int8_mm_op(
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
    except Exception as e:
        raise RuntimeError(f"cuSPARSELt INT8 execution failed: {e}") from e


def cutlass_INT8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    input_scale: Optional[torch.Tensor] = None,
    input_zero_point: Optional[torch.Tensor] = None,
    azp_adj: Optional[torch.Tensor] = None,
    input_symmetric: bool = True,
) -> torch.Tensor:
    """
    vLLM CUTLASS INT8（融合 GEMM + Dequant + Bias）
    
    数据流:
        input[M, K] BF16
            ↓ ops.scaled_int8_quant
        qinput[M, K] INT8, scale_a, x_zp
            ↓ cutlass_scaled_mm / cutlass_scaled_mm_azp
        output[M, N] out_dtype
    
    支持对称和非对称量化。
    """
    # Quant（使用 vLLM 原生 ops.scaled_int8_quant）
    if input.dtype != torch.int8:
        with ProfileTimer("CUTLASS.quant"):
            x_q, x_s, x_zp = ops.scaled_int8_quant(
                input.contiguous(),
                input_scale,
                input_zero_point,
                symmetric=input_symmetric
            )
    else:
        # 静态量化：input 已是 INT8
        x_q, x_s, x_zp = input, input_scale, input_zero_point
    
    # CUTLASS 融合 GEMM + Dequant + Bias
    with ProfileTimer("CUTLASS.scaled_mm"):
        if x_zp is not None:
            # 非对称量化
            # Currently, static is always per-tensor and dynamic is per-token
            static = input_zero_point is not None
            azp = None if static else x_zp
            output = ops.cutlass_scaled_mm_azp(
                x_q,
                weight,
                scale_a=x_s,
                scale_b=weight_scale,
                out_dtype=out_dtype,
                azp_adj=azp_adj,
                azp=azp,
                bias=bias,
            )
        else:
            # 对称量化
            output = ops.cutlass_scaled_mm(
                x_q, weight, out_dtype=out_dtype,
                scale_a=x_s, scale_b=weight_scale, bias=bias
            )
    
    with ProfileTimer("CUTLASS.view"):
        result = output.view(*output_shape)
    
    profile_step()
    return result


# ============================================================================
# SlideSparse INT8 Linear Op
# ============================================================================

class SlideSparseInt8LinearOp:
    """
    SlideSparse INT8 Linear Operation
    
    根据环境变量选择 kernel 路径：
    - USE_CUBLASLT=1: cuBLASLt_INT8_linear
    - USE_CUSPARSELT=1: cuSPARSELt_INT8_linear
    - 默认: cutlass_INT8_linear
    
    注意：cuBLASLt 和 cuSPARSELt 路径只支持对称量化。
    """
    
    def __init__(
        self,
        act_quant_static: bool = False,
        input_symmetric: bool = True,
    ):
        self.act_quant_static = act_quant_static
        self.input_symmetric = input_symmetric
        
        # 确定 kernel 路径（缓存环境变量判断结果）
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # 缓存 inner_dtype
        # 注意：cuBLASLt INT8 输出固定为 INT32，此设置仅对 cuSPARSELt 有效
        self._inner_dtype_str = "int32" if is_inner_dtype_32() else "bf16"
        
        # 非对称量化时，强制使用 CUTLASS
        if not input_symmetric and (self._use_cublaslt or self._use_cusparselt):
            logger.warning_once(
                "Asymmetric INT8 quantization detected. "
                "cuBLASLt/cuSPARSELt do not support asymmetric quantization. "
                "Falling back to CUTLASS path."
            )
            self._use_cublaslt = False
            self._use_cusparselt = False
        
        if self._use_cublaslt:
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_INT8_linear
            _get_gemm_extension("cublaslt")
        elif self._use_cusparselt:
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_INT8_linear
            _get_gemm_extension("cusparselt")
        else:
            self._kernel_name = "CUTLASS"
            self._linear_fn = cutlass_INT8_linear
        
        logger.info_once(
            f"SlideSparseInt8LinearOp initialized (kernel={self._kernel_name})"
        )
    
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_zero_point: torch.Tensor | None = None,
        azp_adj: torch.Tensor | None = None,
        input_symmetric: bool = True,
        bias: torch.Tensor | None = None,
        slide_weight_N: int | None = None,
        slide_weight_K: int | None = None,
        L: int = 8,
    ) -> torch.Tensor:
        """
        执行 INT8 Linear 操作
        
        Args:
            input: [..., K] BF16
            weight: 权重（形状取决于 kernel 路径）
            weight_scale: [N] FP32
            out_dtype: 输出类型
            input_scale: 静态量化 scale
            input_zero_point: 非对称量化零点
            azp_adj: AZP adjustment term
            input_symmetric: 是否对称量化
            bias: [N]
            slide_weight_N: cuSPARSELt 专用，N 维度
            slide_weight_K: cuSPARSELt 专用，K_slide 维度
            L: cuSPARSELt 专用，稀疏组大小
        """
        # 获取 input 形状信息
        input_shape = input.shape
        input_ndim = input.dim()
        
        # 展平为 2D
        if input_ndim == 2:
            input_2d = input
            M = input_shape[0]
        else:
            input_2d = input.view(-1, input_shape[-1])
            M = input_2d.shape[0]
        
        # 推断输出 N 维度
        if self._use_cusparselt:
            output_N = slide_weight_N
        elif self._use_cublaslt:
            output_N = weight.shape[0]
        else:
            output_N = weight.shape[1]
        
        # 构建 output_shape
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
            input_zero_point=input_zero_point,
            azp_adj=azp_adj,
            input_symmetric=input_symmetric,
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
            )


# ============================================================================
# SlideSparse INT8 Linear Method
# ============================================================================

class SlideSparseInt8LinearMethod:
    """
    SlideSparse INT8 Linear Method
    
    包装 vLLM 原有的 CompressedTensorsW8A8Int8 scheme：
    - create_weights: 委托给原始 scheme
    - process_weights_after_loading: 委托 + cuBLASLt/cuSPARSELt 后处理
    - apply_weights: 使用 SlideSparseInt8LinearOp
    
    权重形状变化:
        原始 checkpoint: [N, K] 或 [N, K_slide]（slidesparse checkpoint）
        vLLM 加载后: [N, K] 或 [N, K_slide]
        CUTLASS 路径: weight.t() -> [K, N]
        cuBLASLt 路径: 保持 [N, K]
        cuSPARSELt 路径: [N, K_slide] -> compress -> [compressed_size] uint8 1D
    """
    
    def __init__(self, original_scheme):
        self.original_scheme = original_scheme
        self.strategy = original_scheme.strategy
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.input_symmetric = original_scheme.input_symmetric
        
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # 创建 SlideSparse Op
        self.slidesparse_int8_linear = SlideSparseInt8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            input_symmetric=self.input_symmetric,
        )
        
        # 更新 kernel 路径（Op 可能因非对称量化而 fallback）
        self._use_cublaslt = self.slidesparse_int8_linear._use_cublaslt
        self._use_cusparselt = self.slidesparse_int8_linear._use_cusparselt
        
        # cuSPARSELt 稀疏配置
        if self._use_cusparselt:
            Z, L, self._expand_ratio = get_sparsity_config()
            self._sparsity_config = SlideSparseConfig(Z=Z, L=L)
            logger.info_once(
                f"SlideSparseInt8LinearMethod: cuSPARSELt "
                f"sparsity={Z}:{L}, expand_ratio={self._expand_ratio:.3f}"
            )
        
        logger.info_once(
            f"SlideSparseInt8LinearMethod initialized, "
            f"kernel={self.slidesparse_int8_linear._kernel_name}, "
            f"symmetric={self.input_symmetric}"
        )
    
    def create_weights(
        self,
        layer: Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        """
        创建权重参数
        
        cuSPARSELt 路径：扩展 input_size 以匹配 slide 后的 checkpoint 权重
        其他路径：直接委托给原始 scheme
        
        Note:
            INT8 scheme 的 create_weights 签名与 FP8 略有不同。
        """
        if self._use_cusparselt:
            # cuSPARSELt: 扩展 input_size 以匹配 slide 后的 K 维度
            _, input_size_per_partition_slide = compute_output_k(
                input_size_per_partition, self._sparsity_config
            )
            
            return self.original_scheme.create_weights(
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition_slide,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
        else:
            # CUTLASS / cuBLASLt: 直接委托
            return self.original_scheme.create_weights(
                layer=layer,
                output_partition_sizes=output_partition_sizes,
                input_size_per_partition=input_size_per_partition,
                params_dtype=params_dtype,
                weight_loader=weight_loader,
                **kwargs,
            )
    
    def process_weights_after_loading(self, layer: Module) -> None:
        """
        权重加载后处理
        
        处理逻辑:
        - CUTLASS: 委托给原始 scheme（执行 weight.t() 和 azp_adj 计算）
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
        
        输入: layer.weight [N, K_slide] INT8
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
            f"cuSPARSELt INT8 compression: [{N}, {K_slide}] -> 1D uint8"
        )
        
        weight_compressed = compress_tensor_online(slide_weight, verbose=False)
        
        layer.weight = Parameter(weight_compressed, requires_grad=False)
        layer.slide_weight_N = N
        layer.slide_weight_K = K_slide
        
        logger.info_once(
            f"cuSPARSELt INT8 compression done: {weight_compressed.numel()} bytes"
        )
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply weights (linear transformation)"""
        input_scale = getattr(layer, "input_scale", None)
        input_zero_point = getattr(layer, "input_zero_point", None)
        azp_adj = getattr(layer, "azp_adj", None)
        
        if self._use_cusparselt:
            return self.slidesparse_int8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=x.dtype,
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                azp_adj=azp_adj,
                input_symmetric=self.input_symmetric,
                bias=bias,
                slide_weight_N=layer.slide_weight_N,
                slide_weight_K=layer.slide_weight_K,
                L=self._sparsity_config.L,
            )
        else:
            return self.slidesparse_int8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=x.dtype,
                input_scale=input_scale,
                input_zero_point=input_zero_point,
                azp_adj=azp_adj,
                input_symmetric=self.input_symmetric,
                bias=bias,
            )


# ============================================================================
# 工厂函数
# ============================================================================

def wrap_scheme_int8(original_scheme):
    """
    INT8 scheme 包装入口
    
    只包装 W8A8Int8 scheme，其他 scheme 原样返回。
    内部由 SlideSparseInt8LinearOp 根据环境变量选择 kernel 路径。
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Int8" not in scheme_name:
        logger.warning_once(
            f"SlideSparse INT8 not supported for {scheme_name}, using original"
        )
        return original_scheme
    
    # 判断实际使用的 backend
    use_cublaslt = is_cublaslt_enabled()
    use_cusparselt = is_cusparselt_enabled()
    
    # 非对称量化时强制 CUTLASS
    if not original_scheme.input_symmetric and (use_cublaslt or use_cusparselt):
        backend = "CUTLASS (asymmetric fallback)"
    elif use_cublaslt:
        backend = "cuBLASLt"
    elif use_cusparselt:
        backend = "cuSPARSELt"
    else:
        backend = "CUTLASS"
    
    logger.info_once(
        f"Wrapping {scheme_name} with SlideSparse ({backend})"
    )
    return SlideSparseInt8LinearMethod(original_scheme)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Linear 函数
    "cuBLASLt_INT8_linear",
    "cuSPARSELt_INT8_linear",
    "cutlass_INT8_linear",
    
    # Op 和 Method 类
    "SlideSparseInt8LinearOp",
    "SlideSparseInt8LinearMethod",
    
    # 工厂函数
    "wrap_scheme_int8",
]
