# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Linear Method

本模块是 SlideSparse 的核心，通过外挂方式替换 vLLM 的 FP8 Linear 计算路径。

架构说明
========
SlideSparse 通过包装 vLLM 原有的 CompressedTensorsW8A8Fp8 scheme 实现外挂：
- create_weights: 委托给原始 scheme（cuSPARSELt 需要修改 K 维度）
- process_weights_after_loading: 委托给原始 scheme + 可选在线压缩
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
   - 权重形状: slide_weight_compressed [compressed_size] uint8 1D（在线压缩后）
   - 激活形状: slide_qinput [M, K'] FP8（slide 扩展后）
   - 需要配置 SPARSITY 环境变量（默认 2_8）
   
   cuSPARSELt 命名约定:
   - slide_weight [N, K']: slide 后的 2D FP8 权重（压缩前）
   - slide_weight_compressed [bytes]: cuSPARSELt 压缩后的 1D uint8
   - slide_weight_N: 原始 N 维度（压缩前保存）
   - slide_weight_K: slide 后的 K' 维度（K' = K * expand_ratio）
   - slide_qinput [M, K']: slide + quant 后的激活

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
- SPARSITY=2_8            : 稀疏格式（仅 cuSPARSELt 时生效，默认 2_8）
"""

from .config import (
    is_slidesparse_enabled, 
    is_cublaslt_enabled, 
    is_cusparselt_enabled, 
    is_inner_dtype_fp32, 
    get_slidesparse_status,
    get_sparsity_config,
)

from pathlib import Path
from typing import Optional

import ctypes
import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

# 使用统一的 slidesparse 工具库
from slidesparse.utils import (
    load_module, 
    normalize_dtype, 
    find_file,
    SlideSparseConfig,
    compute_output_k,
)

import subprocess

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
# 自动编译/搜索函数
# ============================================================================

# kernel 类型 -> (脚本名, 额外参数)
_KERNEL_BUILD_CONFIG = {
    "cublaslt":     ("build_cublaslt.py",              ["build"]),
    "cusparselt":   ("build_cusparselt.py",            ["build"]),
    "dequant_bias": ("autotune_autogen_dequant_bias.py", ["--quick"]),
    "quant_fp8":    ("autotune_autogen_quant_only.py", ["--quick", "--dtype", "fp8"]),
    "quant_int8":   ("autotune_autogen_quant_only.py", ["--quick", "--dtype", "int8"]),
}


def _build_search_kernel(kernel_dir: Path, kernel_type: str) -> None:
    """
    在找不到 kernel 时，自动编译或进行 autotune 搜索生成 kernel
    
    Args:
        kernel_dir: kernel 源代码所在目录（build 目录的父目录）
        kernel_type: "cublaslt", "cusparselt", "dequant_bias", "quant_fp8", "quant_int8"
    """
    if kernel_type not in _KERNEL_BUILD_CONFIG:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    script_name, args = _KERNEL_BUILD_CONFIG[kernel_type]
    script_path = Path(kernel_dir) / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Build script not found: {script_path}")
    
    cmd = ["python3", str(script_path)] + args
    logger.info(f"Kernel not found, building: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(kernel_dir), capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        raise RuntimeError(f"Kernel build failed:\n{result.stderr or result.stdout}")


# ============================================================================
# Extension 加载（cuBLASLt / cuSPARSELt 统一入口）
# ============================================================================

# CSRC 目录（用于模块加载）
_CSRC_DIR = Path(__file__).parent.parent / "csrc"

# 缓存加载的 GEMM 扩展
_gemm_extensions = {}


class cuBLASLtGemmWrapper:
    """cuBLASLt GEMM ctypes 包装器（简化版）"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cublaslt_gemm_get_last_error.argtypes = []
        self._lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # GEMM 签名: int fn(W, A, D, M, N, K, inner_dtype, stream)
        # fp8_mm 和 int8_mm 签名完全相同
        gemm_sig = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_char_p, ctypes.c_void_p]
        for name in ["cublaslt_fp8_mm", "cublaslt_int8_mm"]:
            getattr(self._lib, name).argtypes = gemm_sig
            getattr(self._lib, name).restype = ctypes.c_int
    
    def _call_gemm(self, fn_name: str, W: torch.Tensor, A: torch.Tensor, inner_dtype: str) -> torch.Tensor:
        """通用 GEMM 调用: D[M,N] = W[N,K] @ A[M,K]"""
        M, K = A.shape
        N = W.shape[0]
        D = torch.empty((M, N), dtype=torch.float32 if inner_dtype == "fp32" else torch.bfloat16, device=A.device)
        
        ret = getattr(self._lib, fn_name)(
            W.data_ptr(), A.data_ptr(), D.data_ptr(), M, N, K,
            inner_dtype.encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cublaslt_gemm_get_last_error()
            raise RuntimeError(f"{fn_name} failed: {err.decode() if err else 'Unknown'}")
        return D
    
    def cublaslt_fp8_mm(self, W: torch.Tensor, A: torch.Tensor, inner_dtype: str = "bf16") -> torch.Tensor:
        return self._call_gemm("cublaslt_fp8_mm", W, A, inner_dtype)
    
    def cublaslt_int8_mm(self, W: torch.Tensor, A: torch.Tensor, inner_dtype: str = "bf16") -> torch.Tensor:
        return self._call_gemm("cublaslt_int8_mm", W, A, inner_dtype)


def _get_gemm_extension(backend: str):
    """
    获取指定后端的 GEMM extension（懒加载）
    
    Args:
        backend: "cublaslt" 或 "cusparselt"
        
    Returns:
        加载的 extension 模块（包装器对象）
        
    Note:
        GEMM extension 运行时支持多种数据类型（FP8E4M3, INT8），
        文件名不包含 dtype，格式为: {backend}_gemm_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.so
        
        如果找不到预编译的 .so 文件，会自动调用编译脚本进行编译。
    """
    global _gemm_extensions
    
    if backend in _gemm_extensions:
        return _gemm_extensions[backend]
    
    if backend not in ("cublaslt", "cusparselt"):
        raise ValueError(f"Unsupported backend: {backend}")
    
    # prefix 是 cublaslt_gemm 或 cusparselt_gemm
    prefix = f"{backend}_gemm"
    build_dir = _CSRC_DIR / prefix / "build"
    
    # 查找 .so 文件
    so_path = find_file(prefix, search_dir=build_dir, ext=".so")
    
    if so_path is None:
        # 找不到，尝试自动编译
        kernel_dir = build_dir.parent
        _build_search_kernel(kernel_dir, kernel_type=backend)
        
        # 重新查找
        so_path = find_file(prefix, search_dir=build_dir, ext=".so")
        if so_path is None:
            raise FileNotFoundError(
                f"GEMM extension not found after build: {prefix}\n"
                f"Build may have failed. Please check the logs."
            )
    
    # 根据后端创建包装器
    if backend == "cublaslt":
        wrapper = cuBLASLtGemmWrapper(so_path)
    elif backend == "cusparselt":
        # TODO: 为 cusparselt 创建类似的包装器
        wrapper = load_module(prefix, search_dir=build_dir, ext=".so")
    
    _gemm_extensions[backend] = wrapper
    logger.info_once(f"{backend} GEMM extension loaded: {so_path.name}")
    return wrapper


# ============================================================================
# Dequant + Bias Kernel
# ============================================================================

_dequant_bias_fn = None  # 缓存加载的 kernel 函数


def _load_dequant_bias_kernel():
    """
    加载 Triton dequant+bias kernel（懒加载，仅调用一次）
    
    如果找不到预生成的 kernel，会自动运行 autotune 脚本生成。
    """
    global _dequant_bias_fn
    if _dequant_bias_fn is not None:
        return _dequant_bias_fn
    
    # dequant kernel 支持 BF16 和 FP32 输入，不需要 dtype 区分
    kernel_dir = _CSRC_DIR / "fused_dequant_bias_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("dequant_bias_tuned", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        # 找不到，尝试自动 autotune
        _build_search_kernel(kernel_dir, kernel_type="dequant_bias")
        
        # 重新加载
        try:
            module = load_module("dequant_bias_tuned", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dequant+bias kernel not found after autotune.\n"
                f"Autotune may have failed. Please check the logs."
            )
    
    _dequant_bias_fn = module.dequant_bias_triton
    logger.info_once("Dequant+bias kernel loaded")
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
# Quant Only Kernel (FP8)
# ============================================================================

_quant_only_fp8_fn = None  # 缓存加载的 kernel 函数


def _load_quant_only_fp8_kernel():
    """
    加载 Triton FP8 quant kernel（懒加载，仅调用一次）
    
    如果找不到预生成的 kernel，会自动运行 autotune 脚本生成。
    """
    global _quant_only_fp8_fn
    if _quant_only_fp8_fn is not None:
        return _quant_only_fp8_fn
    
    # FP8 quant kernel，dtype 为 FP8E4M3
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("quant_only_tuned", dtype="FP8E4M3", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        # 找不到，尝试自动 autotune
        _build_search_kernel(kernel_dir, kernel_type="quant_fp8")
        
        # 重新加载
        try:
            module = load_module("quant_only_tuned", dtype="FP8E4M3", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FP8 quant kernel not found after autotune.\n"
                f"Autotune may have failed. Please check the logs."
            )
    
    _quant_only_fp8_fn = module.quant_triton
    logger.info_once("FP8 quant kernel loaded")
    return _quant_only_fp8_fn


def quant_only_fp8_kernel(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization（使用 autotuned Triton kernel）
    
    计算: qout[M,K], scale[M] = per_token_quant(input[M,K])
    
    Args:
        input: 输入张量 [M, K]，BF16/FP16/FP32，必须 contiguous
        
    Returns:
        qout: 量化输出 [M, K]，FP8E4M3
        scale: per-token scale [M]，FP32
    """
    fn = _load_quant_only_fp8_kernel()
    return fn(input)


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
# 注意: cuBLASLt 使用 Triton 实现的 quant kernel，
#       cuSPARSELt 和 CUTLASS 仍使用 vLLM 原生 QuantFP8
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
    """cuBLASLt dense FP8 GEMM + Triton quant/dequant"""
    ext = _get_gemm_extension("cublaslt")
    
    # 使用 Triton 实现的 quant kernel
    if input.dtype != current_platform.fp8_dtype():
        qinput, scale_a = quant_only_fp8_kernel(input)
    else:
        qinput, scale_a = input, input_scale
    
    # cuBLASLt 路径：权重已经是 [N, K] 行主序    
    try:
        gemm_output = ext.cublaslt_fp8_mm(weight, qinput, get_inner_dtype_str())
        output = dequant_bias_kernel(gemm_output, scale_a, scale_b, bias, out_dtype)
        return output.view(*output_shape)
    except Exception as e:
        raise RuntimeError(f"cuBLASLt execution failed: {e}") from e


def cuSPARSELt_FP8_linear(
    *,
    input: torch.Tensor,
    slide_weight_compressed: torch.Tensor,
    out_dtype: torch.dtype,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: Optional[QuantFP8] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    # cuSPARSELt 特有参数
    slide_weight_N: Optional[int] = None,
    slide_weight_K: Optional[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    cuSPARSELt 2:4 sparse FP8 GEMM + Triton dequant
    
    数据流:
    ===========
    input [M, K] BF16
        ↓ fused_quant_slide_fp8_kernel (TODO: 待实现)
    slide_qinput [M, K'] FP8, scale_a [M] FP32
        ↓ cusparselt_fp8_mm_compressed
    gemm_output [M, N]
        ↓ dequant_bias_kernel
    output [M, N]
    
    参数说明:
    - input: 原始激活 [M, K] BF16
    - slide_weight_compressed: 1D uint8 tensor（cuSPARSELt 压缩后的格式）
    - slide_weight_N: slide 后权重的 N 维度（与原始 N 相同）
    - slide_weight_K: slide 后权重的 K' 维度（K' = K * expand_ratio）
    
    注意:
    - scale_a 是 per-token 的 [M]，与 K 无关，slide 不影响
    """
    ext = _get_gemm_extension("cusparselt")
    
    # slide 操作会把 input [M, K] 扩展为 slide_qinput [M, K']
    if input.dtype != current_platform.fp8_dtype():
        # 动态量化 + slide: input [M, K] BF16 -> slide_qinput [M, K'] FP8
        slide_qinput, scale_a = fused_quant_slide_fp8_kernel(input, slide_weight_K)
    else:
        # 静态量化场景：input 已经是 FP8 且已做 slide（由上层保证）
        # TODO: 静态量化时上层需要提前做 slide，目前假设不支持静态量化
        raise NotImplementedError(
            "cuSPARSELt with static quantization is not supported yet. "
            "Input must be BF16/FP16 for dynamic quantization."
        )
    
    # 检查维度信息
    if slide_weight_N is None or slide_weight_K is None:
        raise ValueError(
            "cuSPARSELt requires slide_weight_N and slide_weight_K parameters. "
            "These should be stored during process_weights_after_loading."
        )
    
    try:
        # cuSPARSELt GEMM: slide_qinput [M, K'] @ slide_weight_compressed -> [M, N]
        gemm_output = ext.cusparselt_fp8_mm_compressed(
            slide_weight_compressed,  # 压缩后的 1D uint8 tensor
            slide_qinput,             # [M, K'] FP8（slide 后的激活）
            slide_weight_N,           # N 维度
            slide_weight_K,           # K' 维度（slide 扩展后）
            get_inner_dtype_str()
        )
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
        # cuSPARSELt 特有参数
        slide_weight_N: int | None = None,
        slide_weight_K: int | None = None,
    ) -> torch.Tensor:
        """
        执行 FP8 Linear 操作
        
        完整流程:
        1. input (BF16) -> quant -> qinput (FP8), x_scale
        2. qinput @ weight.T -> inner_output (选定的 GEMM kernel)
        3. inner_output * scale_a * scale_b + bias -> output (Dequant)
        
        Args:
            input: 输入张量 [..., K]，BF16/FP16
            weight: 权重张量，形状取决于 kernel 路径:
                    - CUTLASS: [K, N]（vLLM 转置后）
                    - cuBLASLt: [N, K]（跳过转置）
                    - cuSPARSELt: [compressed_size] uint8 1D（压缩后）
            weight_scale: 权重 scale [N, 1] 或 [1]
            out_dtype: 输出数据类型（由 vLLM 上层指定）
            input_scale: 输入 scale（静态量化时使用）
            input_scale_ub: 输入 scale 上界
            bias: 偏置 [N]
            slide_weight_N: cuSPARSELt 专用，slide 后权重的 N 维度
            slide_weight_K: cuSPARSELt 专用，slide 后权重的 K' 维度
            
        Returns:
            输出张量 [..., N]，out_dtype
        """
        # View input as 2D matrix
        input_2d = input.view(-1, input.shape[-1])
        
        # 计算 output_shape（需要知道输出的 N 维度）
        if weight.dim() == 1:
            # cuSPARSELt 压缩权重是 1D tensor，无法从 shape 推断 N
            # slide_weight_N 是压缩前保存的原始 2D 权重 [N, K'] 的 N 维度
            if slide_weight_N is None:
                raise ValueError("slide_weight_N required for cuSPARSELt compressed weight")
            output_N = slide_weight_N
        elif weight.dim() == 2:
            # CUTLASS 路径: [K, N]，N 在 dim=1
            # cuBLASLt 路径: [N, K]，N 在 dim=0
            if is_cublaslt_enabled():
                output_N = weight.shape[0]  # [N, K]
            else:
                output_N = weight.shape[1]  # [K, N]
        else:
            raise ValueError(f"Unexpected weight dimension: {weight.dim()}")
        
        output_shape = [*input.shape[:-1], output_N]
        
        if out_dtype is None:
            out_dtype = input.dtype
        
        # 调用选定的 kernel 路径（quant/slide 在各 linear 函数内部进行）
        if is_cusparselt_enabled():
            # cuSPARSELt 路径：weight 是 slide_weight_compressed (1D)
            return self._linear_fn(
                input=input_2d,
                slide_weight_compressed=weight,
                out_dtype=out_dtype,
                scale_b=weight_scale,
                bias=bias,
                output_shape=output_shape,
                quant_fn=self.quant_fp8,
                input_scale=input_scale,
                input_scale_ub=input_scale_ub,
                slide_weight_N=slide_weight_N,
                slide_weight_K=slide_weight_K,
            )
        else:
            # CUTLASS / cuBLASLt 路径：weight 是 2D tensor
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
    3. create_weights: 委托给原始 scheme
    4. process_weights_after_loading: 
       - CUTLASS 路径：委托给原始 scheme（需要转置）
       - cuBLASLt 路径：修改后的处理（跳过转置）
       - cuSPARSELt 路径：修改后的处理（跳过转置 + 在线压缩）
    5. 通过 SlideSparseFp8LinearOp 选择 cuBLASLt/cuSPARSELt/CUTLASS 路径
    
    权重形状变化:
    ==============
    原始 checkpoint 权重: [N, K]
    vLLM 加载后 (无转置): [N, K]
    
    CUTLASS 路径 (vLLM 转置): weight = weight.t() -> [K, N]
    cuBLASLt 路径 (跳过转置): weight = [N, K] 保持不变
    cuSPARSELt 路径 (跳过转置 + 压缩): 
        1. 原始: [N, K'] (K' = K * expand_ratio，来自 slidesparse checkpoint)
        2. 在线压缩: [compressed_size] uint8 1D tensor
        3. 存储原始 N, K' 维度信息供 GEMM 使用
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
        self.strategy = original_scheme.strategy
        
        # 缓存 kernel 选择结果
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        
        # 创建 SlideSparse Op（内部根据环境变量选择 kernel）
        self.slidesparse_fp8_linear = SlideSparseFp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape,
        )
        
        # 如果使用 cuSPARSELt，获取稀疏配置
        if self._use_cusparselt:
            Z, L, self._expand_ratio = get_sparsity_config()
            self._sparsity_config = SlideSparseConfig(Z=Z, L=L)
            logger.info_once(
                f"SlideSparseFp8LinearMethod using cuSPARSELt "
                f"with sparsity={Z}:{L}, expand_ratio={self._expand_ratio:.3f}"
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
        """
        创建权重参数
        
        完全委托给原始 scheme。
        
        注意：对于 cuSPARSELt 路径，用户需要传入正确的 checkpoint 路径
        （即 slidesparse 转换后的 checkpoint，K' = K * expand_ratio）。
        vLLM 的 weight_loader 会从实际的 safetensor 文件加载正确维度的权重。
        """
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
        
        调用链:
            vLLM model loading
              → layer.weight_loader(...)  # 加载权重到 layer.weight
              → layer.quant_method.process_weights_after_loading(layer)  # 后处理
        
        处理逻辑（根据 kernel 路径）:
        ================================
        1. CUTLASS 路径（默认）：
           - 完全委托给原始 scheme
           - 原始 scheme 会执行 weight = weight.t()
           - 最终 layer.weight 形状: [K, N]
        
        2. cuBLASLt 路径：
           - 先调用原始 scheme（得到转置后的 [K, N]）
           - 再转置回来得到 [N, K]
        
        3. cuSPARSELt 路径：
           - 先调用原始 scheme（得到转置后的 [K, N]）
           - 再转置回来得到 [N, K]（此时是 slide_weight [N, K']）
           - 再执行在线压缩得到 slide_weight_compressed (1D uint8)
        """
        # 第一步：所有路径都先调用原始 scheme
        self.original_scheme.process_weights_after_loading(layer)
        
        if not self._use_cublaslt and not self._use_cusparselt:
            # CUTLASS 路径：直接返回，原始 scheme 已处理完毕
            return
        
        # cuBLASLt 或 cuSPARSELt 路径：需要把权重转置回 [N, K] 或 [N, K']
        # vLLM 原始 scheme 执行了 weight.t()（只改 stride），我们再 .t() 回来，无需 .contiguous()
        from torch.nn import Parameter
        weight_transposed = layer.weight.data.t()  # [K, N] -> [N, K]，只改 stride
        layer.weight = Parameter(weight_transposed, requires_grad=False)
        
        if self._use_cusparselt:
            # cuSPARSELt 路径：额外执行在线压缩
            self._compress_weight_online(layer)
    
    def _compress_weight_online(self, layer: Module) -> None:
        """
        在线压缩权重（cuSPARSELt）
        
        将 2D FP8 slide 权重压缩为 1D uint8 tensor，并存储 slide 后的维度信息。
        
        输入:
            layer.weight: [N, K'] FP8 2D tensor（slide_weight，K' = K * expand_ratio）
        
        输出:
            layer.weight: [compressed_size] uint8 1D tensor（slide_weight_compressed）
            layer.slide_weight_N: N 维度
            layer.slide_weight_K: K' 维度（slide 后）
        
        注意:
            layer.weight 是 vLLM LinearMethod 接口要求的标准属性名，
            apply_weights 会通过 layer.weight 访问权重，所以这个名字不能改。
            虽然压缩后的数据本质上是 slide_weight_compressed，
            但为了兼容 vLLM 接口，仍然存储在 layer.weight 中。
        """
        from torch.nn import Parameter
        
        # 导入在线压缩函数
        try:
            from slidesparse.weight_convert.compress import compress_tensor_online
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import compress_tensor_online: {e}\n"
                "cuSPARSELt in-line compression requires the slidesparse weight_convert module."
            ) from e
        
        # 此时 layer.weight 是 slide_weight [N, K'] FP8
        slide_weight = layer.weight.data
        N, K_slide = slide_weight.shape  # K' = K * expand_ratio
        
        logger.info_once(
            f"cuSPARSELt online compression: slide_weight [{N}, {K_slide}] -> slide_weight_compressed (1D)"
        )
        
        # 执行在线压缩（数据保持在 GPU 上）
        slide_weight_compressed = compress_tensor_online(slide_weight, verbose=False)
        
        # 存储压缩后的权重（仍用 layer.weight 是因为 vLLM 接口要求）
        # 同时存储 slide 后的维度信息供 GEMM 使用
        layer.weight = Parameter(slide_weight_compressed, requires_grad=False)
        layer.slide_weight_N = N
        layer.slide_weight_K = K_slide
        
        logger.info_once(
            f"cuSPARSELt online compression done: compressed_size={slide_weight_compressed.numel()} bytes"
        )
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        应用权重（执行线性变换）
        
        根据 kernel 路径调用不同的处理逻辑:
        
        1. CUTLASS 路径:
           - layer.weight 形状: [K, N]
           - 直接调用 SlideSparseFp8LinearOp.apply
        
        2. cuBLASLt 路径:
           - layer.weight 形状: [N, K]
           - 直接调用 SlideSparseFp8LinearOp.apply
        
        3. cuSPARSELt 路径:
           - layer.weight (slide_weight_compressed): [compressed_size] uint8 1D
           - 额外传入 layer.slide_weight_N, layer.slide_weight_K
           - 内部会执行 fused_quant_slide_fp8_kernel 对激活做 slide
        """
        if self._use_cusparselt:
            # cuSPARSELt 路径：layer.weight 实际是 slide_weight_compressed (1D)
            # 需要额外传入 slide_weight_N, slide_weight_K 供 GEMM 使用
            return self.slidesparse_fp8_linear.apply(
                input=x,
                weight=layer.weight,  # slide_weight_compressed (1D uint8)
                weight_scale=layer.weight_scale,
                out_dtype=self.out_dtype,
                input_scale=getattr(layer, "input_scale", None),
                bias=bias,
                # cuSPARSELt 特有参数
                slide_weight_N=layer.slide_weight_N,
                slide_weight_K=layer.slide_weight_K,
            )
        else:
            # CUTLASS 或 cuBLASLt 路径
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
