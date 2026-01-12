# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Linear Method

本模块是 SlideSparse 的核心，通过外挂方式替换 vLLM 的 FP8 Linear 计算路径。

架构说明
========
SlideSparse 通过包装 vLLM 原有的 CompressedTensorsW8A8Fp8 scheme 实现外挂：
- create_weights / process_weights_after_loading: 完全委托给原始 scheme
- apply_weights: 替换为 SlideSparse 的 kernel 路径

三条 Kernel 路径（通过环境变量选择）
====================================
1. CUTLASS (默认 fallback)
   - 直接调用 vLLM 的 cutlass_scaled_mm，融合 GEMM + dequant + bias
   
2. cuBLASLt (USE_CUBLASLT=1)
   - GEMM: cuBLASLt FP8 矩阵乘法（无 scale/bias 融合）
   - Dequant+Bias: 外挂 Triton kernel
   
3. cuSPARSELt (USE_CUSPARSELT=1)
   - GEMM: cuSPARSELt 2:4 稀疏 FP8 矩阵乘法（无 scale/bias 融合）
   - Dequant+Bias: 外挂 Triton kernel

数据类型
========
- input_dtype:  输入量化精度，FP8E4M3
- inner_dtype:  GEMM 输出精度，BF16（默认）或 FP32（INNER_DTYPE_FP32=1）
- out_dtype:    最终输出精度，由 vLLM 上层指定

环境变量
========
- DISABLE_SLIDESPARSE=1   : 完全禁用 SlideSparse，使用 vLLM 原生路径
- USE_CUBLASLT=1          : 使用 cuBLASLt kernel
- USE_CUSPARSELT=1        : 使用 cuSPARSELt kernel（与 USE_CUBLASLT 互斥）
- INNER_DTYPE_FP32=1      : GEMM 输出用 FP32（仅 cuBLASLt/cuSPARSELt 生效）
"""

from .config import is_slidesparse_enabled, is_cublaslt_enabled, is_cusparselt_enabled, is_inner_dtype_fp32, get_slidesparse_status

from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

# 使用统一的 slidesparse 工具库
from slidesparse.utils import load_module, normalize_dtype

logger = init_logger(__name__)


# ============================================================================
# 环境变量配置（从 config 统一管理）
# ============================================================================

def get_inner_dtype_str() -> str:
    """获取 GEMM 输出精度字符串"""
    return "fp32" if is_inner_dtype_fp32() else "bf16"


def get_inner_dtype_torch() -> torch.dtype:
    """获取 GEMM 输出精度的 PyTorch dtype"""
    return torch.float32 if is_inner_dtype_fp32() else torch.bfloat16


# ============================================================================
# Extension 加载（cuBLASLt / cuSPARSELt 统一入口）
# ============================================================================

# CSRC 目录（用于模块加载）
_CSRC_DIR = Path(__file__).parent.parent / "csrc"


def _get_gemm_extension(backend: str):
    """
    获取指定后端的 GEMM extension（懒加载）
    
    Args:
        backend: "cublaslt" 或 "cusparselt"
        
    Returns:
        加载的 extension 模块
        
    Note:
        GEMM extension 运行时支持多种数据类型（FP8E4M3, INT8），
        文件名不包含 dtype，格式为: {backend}_gemm_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.so
    """
    if backend not in ("cublaslt", "cusparselt"):
        raise ValueError(f"Unsupported backend: {backend}")
    
    # prefix 是 cublaslt_gemm 或 cusparselt_gemm
    prefix = f"{backend}_gemm"
    build_dir = _CSRC_DIR / prefix / "build"
    
    # 使用统一 load_module，不带 dtype（因为运行时支持多种类型）
    module = load_module(prefix, search_dir=build_dir, ext=".so")
    logger.info_once(f"{backend} GEMM extension loaded")
    return module


# ============================================================================
# Dequant + Bias Kernel
# ============================================================================

_dequant_bias_fn = None  # 缓存加载的 kernel 函数


def _load_dequant_bias_kernel():
    """加载 Triton dequant+bias kernel（懒加载，仅调用一次）"""
    global _dequant_bias_fn
    if _dequant_bias_fn is not None:
        return _dequant_bias_fn
    
    # dtype_tag 对应 GEMM 输出精度
    dtype_tag = "FP32" if is_inner_dtype_fp32() else "BF16"
    build_dir = _CSRC_DIR / "fused_dequant_bias_triton" / "build"
    module = load_module("dequant_bias_tuned", dtype=dtype_tag, search_dir=build_dir, ext=".py")
    
    _dequant_bias_fn = module.dequant_bias_triton
    logger.info_once(f"Dequant+bias kernel loaded (inner_dtype={dtype_tag})")
    return _dequant_bias_fn


def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequant + Bias 操作（使用 autotuned Triton kernel）
    
    计算: output = gemm_output * scale_a[M,1] * scale_b[1,N] + bias[1,N]
    
    Args:
        gemm_output: GEMM 输出 [M, N]，inner_dtype（BF16 或 FP32）
        scale_a: 输入 scale [M, 1] 或 [1] FP32
        scale_b: 权重 scale [N, 1] 或 [1] FP32
        bias: 偏置 [N] BF16 或 None
        out_dtype: 输出数据类型
        
    Returns:
        dequant 后的输出 [M, N]，out_dtype
    """
    fn = _load_dequant_bias_kernel()
    if bias is None:
        bias = torch.zeros(gemm_output.shape[1], dtype=torch.bfloat16, device=gemm_output.device)
    return fn(gemm_output, scale_a, scale_b, bias, out_dtype)


# ============================================================================
# FP8 Linear 函数
# ============================================================================
#
# 三个函数签名完全相同，每个函数内部完成 quant + GEMM + dequant:
#   - cuBLASLt_FP8_linear:   quant + cuBLASLt dense GEMM + Triton dequant
#   - cuSPARSELt_FP8_linear: quant + cuSPARSELt 2:4 sparse GEMM + Triton dequant
#   - cutlass_FP8_linear:    quant + vLLM cutlass_scaled_mm (融合 dequant)
#
# 参数说明：
#   input:        [M, K] BF16/FP16，原始输入（未量化）
#   weight:       [K, N] FP8，vLLM 的 .t() view（物理内存是 [N,K] 行主序）
#   out_dtype:    最终输出类型
#   scale_b:      [N, 1] or [1] FP32，权重 scale
#   bias:         [N] or None
#   output_shape: 输出形状
#   quant_fn:     QuantFP8 实例，用于输入量化
#   input_scale:  输入 scale（静态量化时使用）
#   input_scale_ub: 输入 scale 上界
#
# 计算流程:
#   1. Quant:   qinput[M,K], scale_a = quant_fn(input)
#   2. GEMM:    inner[M,N] = qinput[M,K] @ weight[N,K]^T
#   3. Dequant: out[M,N] = inner * scale_a * scale_b + bias
#
# 注意: cuBLASLt/cuSPARSELt 的 quant 步骤目前使用 vLLM 原生 QuantFP8，
#       TODO: 替换为 Triton 实现的 quant kernel
# ============================================================================

def cuBLASLt_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: Optional[QuantFP8] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """cuBLASLt dense FP8 GEMM + Triton dequant"""
    ext = _get_gemm_extension("cublaslt")
    
    # TODO: 使用 Triton 实现的 quant kernel
    # 目前暂时使用 vLLM 原生的 QuantFP8
    if input.dtype != current_platform.fp8_dtype():
        assert quant_fn is not None, "quant_fn required for non-FP8 input"
        qinput, scale_a = quant_fn(input, input_scale, input_scale_ub)
    else:
        qinput, scale_a = input, input_scale
    
    # 转换 weight view: [K,N] stride=(1,K) → [N,K] stride=(K,1)
    weight_nk = weight.t()
    if not weight_nk.is_contiguous():
        logger.warning_once("weight.t() not contiguous, making copy")
        weight_nk = weight_nk.contiguous()
    
    try:
        gemm_output = ext.cublaslt_fp8_mm(weight_nk, qinput, get_inner_dtype_str())
        output = dequant_bias_kernel(gemm_output, scale_a, scale_b, bias, out_dtype)
        return output.view(*output_shape)
    except Exception as e:
        raise RuntimeError(f"cuBLASLt execution failed: {e}") from e


def cuSPARSELt_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: Optional[QuantFP8] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """cuSPARSELt 2:4 sparse FP8 GEMM + Triton dequant"""
    ext = _get_gemm_extension("cusparselt")
    
    # TODO: 使用 Triton 实现的 quant kernel
    # 目前暂时使用 vLLM 原生的 QuantFP8
    if input.dtype != current_platform.fp8_dtype():
        assert quant_fn is not None, "quant_fn required for non-FP8 input"
        qinput, scale_a = quant_fn(input, input_scale, input_scale_ub)
    else:
        qinput, scale_a = input, input_scale
    
    weight_nk = weight.t()
    if not weight_nk.is_contiguous():
        logger.warning_once("weight.t() not contiguous, making copy")
        weight_nk = weight_nk.contiguous()
    
    try:
        gemm_output = ext.cusparselt_fp8_mm(weight_nk, qinput, get_inner_dtype_str())
        output = dequant_bias_kernel(gemm_output, scale_a, scale_b, bias, out_dtype)
        return output.view(*output_shape)
    except Exception as e:
        raise RuntimeError(f"cuSPARSELt execution failed: {e}") from e


def cutlass_FP8_linear(
    *,
    input: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: Optional[QuantFP8] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """vLLM 原生 CUTLASS (融合 GEMM + dequant + bias)"""
    # Quantize input if not already FP8
    if input.dtype != current_platform.fp8_dtype():
        assert quant_fn is not None, "quant_fn required for non-FP8 input"
        qinput, scale_a = quant_fn(input, input_scale, input_scale_ub)
    else:
        qinput, scale_a = input, input_scale
    
    output = ops.cutlass_scaled_mm(
        qinput, weight, out_dtype=out_dtype,
        scale_a=scale_a, scale_b=scale_b, bias=bias
    )
    return output.view(*output_shape)


# ============================================================================
# SlideSparse FP8 Linear Op（统一入口，根据环境变量选择 kernel）
# ============================================================================

class SlideSparseFp8LinearOp:
    """
    SlideSparse FP8 Linear Operation
    
    这个类独立实现 FP8 Linear 操作，不依赖 vLLM 的 Fp8LinearOp：
    1. 自己创建 QuantFP8 实例进行输入量化
    2. 根据环境变量选择 kernel 路径：
       - USE_CUBLASLT=1: cuBLASLt_FP8_linear
       - USE_CUSPARSELT=1: cuSPARSELt_FP8_linear
       - 默认: cutlass_FP8_linear (fallback)
    
    调用链:
        SlideSparseFp8LinearOp.apply()
          ├── cuBLASLt_FP8_linear()    # USE_CUBLASLT=1
          ├── cuSPARSELt_FP8_linear()  # USE_CUSPARSELT=1
          └── cutlass_FP8_linear()     # 默认 fallback
    """
    
    def __init__(
        self,
        act_quant_static: bool = False,
        act_quant_group_shape: GroupShape = GroupShape.PER_TOKEN,
    ):
        """
        初始化 SlideSparse FP8 Linear Op
        
        Args:
            act_quant_static: 是否使用静态激活量化
            act_quant_group_shape: 激活量化的分组形状
        """
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        
        # 创建自己的 QuantFP8 实例（不依赖 Fp8LinearOp）
        self.quant_fp8 = QuantFP8(
            static=act_quant_static,
            group_shape=act_quant_group_shape,
            num_token_padding=None,
        )
        
        # 确定 kernel 路径
        if is_cublaslt_enabled():
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_FP8_linear
        elif is_cusparselt_enabled():
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_FP8_linear
        else:
            self._kernel_name = "CUTLASS"
            self._linear_fn = cutlass_FP8_linear
        
        logger.info_once(
            f"SlideSparseFp8LinearOp initialized "
            f"(kernel={self._kernel_name}, "
            f"inner_dtype={get_inner_dtype_str() if is_cublaslt_enabled() or is_cusparselt_enabled() else 'N/A'}, "
            f"static={act_quant_static}, "
            f"group_shape={act_quant_group_shape})"
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
    ) -> torch.Tensor:
        """
        执行 FP8 Linear 操作
        
        完整流程:
        1. input (BF16) -> quant -> qinput (FP8), x_scale
        2. qinput @ weight.T -> inner_output (选定的 GEMM kernel)
        3. inner_output * scale_a * scale_b + bias -> output (Dequant)
        
        Args:
            input: 输入张量 [..., K]，BF16/FP16
            weight: 权重张量 [K, N]（已转置，column-major），FP8
            weight_scale: 权重 scale [N, 1] 或 [1]
            out_dtype: 输出数据类型（由 vLLM 上层指定）
            input_scale: 输入 scale（静态量化时使用）
            input_scale_ub: 输入 scale 上界
            bias: 偏置 [N]
            
        Returns:
            输出张量 [..., N]，out_dtype
        """
        # View input as 2D matrix
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        
        if out_dtype is None:
            out_dtype = input.dtype
        
        # 调用选定的 kernel 路径（quant 在各 linear 函数内部进行）
        return self._linear_fn(
            input=input_2d,
            weight=weight,
            out_dtype=out_dtype,
            scale_b=weight_scale,
            bias=bias,
            output_shape=output_shape,
            quant_fn=self.quant_fp8,
            input_scale=input_scale,
            input_scale_ub=input_scale_ub,
        )


# ============================================================================
# SlideSparse FP8 Linear Method
# ============================================================================

class SlideSparseFp8LinearMethod:
    """
    SlideSparse FP8 Linear Method
    
    这是一个独立的 LinearMethod 实现，根据环境变量选择 kernel 路径。
    
    关键设计:
    1. 不继承任何 vllm 的类，避免依赖问题
    2. 所有方法签名与 LinearMethodBase 兼容
    3. 内部复用 CompressedTensorsW8A8Fp8 的 create_weights 和 process_weights_after_loading
    4. 通过 SlideSparseFp8LinearOp 选择 cuBLASLt/cuSPARSELt/CUTLASS 路径
    """
    
    def __init__(self, original_scheme):
        """
        初始化 SlideSparse FP8 Linear Method
        
        Args:
            original_scheme: 原始的 CompressedTensorsW8A8Fp8 scheme
        """
        self.original_scheme = original_scheme
        self.out_dtype = original_scheme.out_dtype
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.act_q_group_shape = original_scheme.act_q_group_shape
        
        # 创建 SlideSparse Op（内部根据环境变量选择 kernel）
        self.slidesparse_fp8_linear = SlideSparseFp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape,
        )
        
        logger.info_once(
            f"SlideSparseFp8LinearMethod initialized, "
            f"wrapping: {type(original_scheme).__name__}, "
            f"kernel: {self.slidesparse_fp8_linear._kernel_name}"
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
        """创建权重参数，完全委托给原始 scheme"""
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
        """权重加载后处理，完全委托给原始 scheme"""
        return self.original_scheme.process_weights_after_loading(layer)
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """应用权重（执行线性变换），使用选定的 kernel 路径"""
        return self.slidesparse_fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=getattr(layer, "input_scale", None),
            bias=bias,
        )


# ============================================================================
# 工厂函数
# ============================================================================


def wrap_scheme_fp8(original_scheme):
    """
    统一的 FP8 scheme 包装入口
    
    SlideSparse 启用后始终包装，内部由 SlideSparseFp8LinearOp 选择 kernel:
    - USE_CUBLASLT=1: cuBLASLt kernel
    - USE_CUSPARSELT=1: cuSPARSELt kernel
    - 默认: SlideSparse CUTLASS fallback
    
    注意: 只有当 is_slidesparse_enabled() 返回 True 时，这个函数才会被调用
          （由 compressed_tensors.py 中的 if 判断控制）
    
    使用示例（在 compressed_tensors.py 的 get_scheme() 中）:
        if is_slidesparse_enabled():
            scheme = wrap_scheme_fp8(scheme)
        return scheme
    
    Args:
        original_scheme: 原始的 CompressedTensorsScheme
        
    Returns:
        如果 scheme 是 W8A8Fp8，返回包装后的 scheme
        否则返回原始 scheme
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Fp8" not in scheme_name:
        # 非 W8A8Fp8 scheme，不支持
        logger.warning_once(
            f"SlideSparse wrapper not supported for {scheme_name}, "
            "using original scheme"
        )
        return original_scheme
    
    # W8A8Fp8 scheme，进行包装
    if is_cublaslt_enabled():
        backend = "cuBLASLt"
    elif is_cusparselt_enabled():
        backend = "cuSPARSELt"
    else:
        backend = "CUTLASS"
    
    logger.info_once(
        f"Wrapping {scheme_name} with SlideSparseFp8LinearMethod ({backend})"
    )
    return SlideSparseFp8LinearMethod(original_scheme)
