# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Linear Method

本模块是 SlideSparse 的核心，通过外挂方式替换 vLLM 的 FP8 Linear 计算路径。

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
- SPARSITY=2_8            : 稀疏格式（仅 cuSPARSELt 时生效，默认 2_8）
"""

from .config import (
    is_slidesparse_enabled, 
    is_cublaslt_enabled, 
    is_cusparselt_enabled, 
    is_inner_dtype_32, 
    get_slidesparse_status,
    get_sparsity_config,
)

from pathlib import Path
from typing import Optional

import ctypes
import subprocess
import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

from slidesparse.utils import (
    load_module, 
    find_file,
    SlideSparseConfig,
    compute_output_k,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
)

logger = init_logger(__name__)


# ============================================================================
# 内部配置函数（带缓存）
# ============================================================================

# 缓存 inner_dtype 配置（进程生命周期内不变）
_inner_dtype_str_cache: str | None = None
_inner_dtype_torch_cache: torch.dtype | None = None


def get_inner_dtype_str() -> str:
    """获取 GEMM 输出精度字符串（带缓存）"""
    global _inner_dtype_str_cache
    if _inner_dtype_str_cache is None:
        _inner_dtype_str_cache = "fp32" if is_inner_dtype_32() else "bf16"
    return _inner_dtype_str_cache


def get_inner_dtype_torch() -> torch.dtype:
    """获取 GEMM 输出精度的 PyTorch dtype（带缓存）"""
    global _inner_dtype_torch_cache
    if _inner_dtype_torch_cache is None:
        _inner_dtype_torch_cache = torch.float32 if is_inner_dtype_32() else torch.bfloat16
    return _inner_dtype_torch_cache


# ============================================================================
# 自动编译/搜索函数
# ============================================================================

_KERNEL_BUILD_CONFIG = {
    "cublaslt":        ("build_cublaslt.py",                ["build"]),
    "cusparselt":      ("build_cusparselt.py",              ["build"]),
    "dequant_bias":    ("autotune_autogen_dequant_bias.py", ["--quick"]),
    "quant_fp8":       ("autotune_autogen_quant_only.py",   ["--quick", "--dtype", "fp8"]),
    "quant_slide_fp8": ("autotune_autogen_quant_slide.py",  ["--quick"]),
}


def _build_search_kernel(kernel_dir: Path, kernel_type: str) -> None:
    """在找不到 kernel 时，自动编译或进行 autotune 搜索生成 kernel"""
    if kernel_type not in _KERNEL_BUILD_CONFIG:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    script_name, extra_args = _KERNEL_BUILD_CONFIG[kernel_type]
    script_path = kernel_dir / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Build script not found: {script_path}")
    
    logger.info(f"Auto-building {kernel_type} kernel from {script_path}...")
    
    try:
        result = subprocess.run(
            ["python", str(script_path)] + extra_args,
            cwd=kernel_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"{kernel_type} kernel build completed")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to build {kernel_type} kernel:\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        ) from e


# ============================================================================
# Extension 加载（cuBLASLt / cuSPARSELt）
# ============================================================================

_CSRC_DIR = Path(__file__).parent.parent / "csrc"
_gemm_extensions = {}


class cuBLASLtGemmWrapper:
    """cuBLASLt FP8 GEMM ctypes 包装器"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cublaslt_gemm_get_last_error.argtypes = []
        self._lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # GEMM 签名: int fn(W, A, D, M, N, K, inner_dtype, stream)
        gemm_sig = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_char_p, ctypes.c_void_p]
        self._lib.cublaslt_fp8_mm.argtypes = gemm_sig
        self._lib.cublaslt_fp8_mm.restype = ctypes.c_int
        
    def cublaslt_fp8_mm(
        self,
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """
        cuBLASLt FP8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] FP8，权重（行主序，未转置）
            qinput: [M_pad, K_pad] FP8，量化后的激活
            inner_dtype: GEMM 输出精度 ("bf16" 或 "fp32")
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        ret = self._lib.cublaslt_fp8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_pad,
            inner_dtype.encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cublaslt_gemm_get_last_error()
            raise RuntimeError(f"cublaslt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
        return output


class cuSPARSELtGemmWrapper:
    """cuSPARSELt 2:4 Sparse FP8 GEMM ctypes 包装器"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cusparselt_gemm_get_last_error.argtypes = []
        self._lib.cusparselt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # GEMM 签名: int fn(W_compressed, A, D, M, N, K, inner_dtype, stream)
        gemm_sig = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_char_p, ctypes.c_void_p]
        self._lib.cusparselt_fp8_mm.argtypes = gemm_sig
        self._lib.cusparselt_fp8_mm.restype = ctypes.c_int
        
    def cusparselt_fp8_mm(
        self,
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """
        cuSPARSELt 2:4 Sparse FP8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_slide_pad] @ weight_decompressed.T
        
        Args:
            weight_compressed: [compressed_size] uint8 1D，压缩后的权重
            qinput: [M_pad, K_slide_pad] FP8，量化+slide 后的激活
            N: 权重的 N 维度
            K_slide: 权重的 K_slide 维度（slide 扩展后）
            inner_dtype: GEMM 输出精度 ("bf16" 或 "fp32")
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad = qinput.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        ret = self._lib.cusparselt_fp8_mm(
            weight_compressed.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_slide,
            inner_dtype.encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cusparselt_gemm_get_last_error()
            raise RuntimeError(f"cusparselt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
        return output


def _get_gemm_extension(backend: str):
    """
    获取 GEMM extension（懒加载）
    
    加载 ctypes 包装的 CUDA 扩展（纯 C 库，通过 ctypes.CDLL 加载）
    """
    if backend in _gemm_extensions:
        return _gemm_extensions[backend]
    
    if backend == "cublaslt":
        # 预加载系统 cuBLASLt 库（RTLD_GLOBAL 模式）
        ensure_cublaslt_loaded()
        # 目录名是 cublaslt_gemm（不是 cublaslt_fp8_gemm）
        kernel_dir = _CSRC_DIR / "cublaslt_gemm"
        build_dir = kernel_dir / "build"
        so_prefix = "cublaslt_gemm"
        wrapper_class = cuBLASLtGemmWrapper
    elif backend == "cusparselt":
        # 预加载系统 cuSPARSELt 库（0.8.1+，RTLD_GLOBAL 模式）
        # 必须在加载自定义 .so 之前完成，避免使用 PyTorch 自带的旧版本（0.7.x）
        ensure_cusparselt_loaded()
        # 目录名是 cusparselt_gemm（不是 cusparselt_fp8_gemm）
        kernel_dir = _CSRC_DIR / "cusparselt_gemm"
        build_dir = kernel_dir / "build"
        so_prefix = "cusparselt_gemm"
        wrapper_class = cuSPARSELtGemmWrapper
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # 查找 .so 文件
    so_path = find_file(so_prefix, search_dir=build_dir, ext=".so")
    
    if so_path is None:
        # 找不到，尝试自动编译
        _build_search_kernel(kernel_dir, kernel_type=backend)
        so_path = find_file(so_prefix, search_dir=build_dir, ext=".so")
        if so_path is None:
            raise FileNotFoundError(
                f"{backend} GEMM extension not found after build.\n"
                f"Build may have failed. Please check the logs."
            )
    
    # 创建 ctypes 包装器（传递 .so 路径）
    wrapper = wrapper_class(so_path)
    _gemm_extensions[backend] = wrapper
    logger.info_once(f"{backend} GEMM extension loaded: {so_path.name}")
    return wrapper


# ============================================================================
# Dequant + Bias Kernel
# ============================================================================

_dequant_bias_fn = None


def _load_dequant_bias_kernel():
    """加载 Triton dequant+bias kernel（懒加载）"""
    global _dequant_bias_fn
    if _dequant_bias_fn is not None:
        return _dequant_bias_fn
    
    kernel_dir = _CSRC_DIR / "fused_dequant_bias_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("dequant_bias_tuned", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        _build_search_kernel(kernel_dir, kernel_type="dequant_bias")
        try:
            module = load_module("dequant_bias_tuned", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dequant+bias kernel not found after autotune."
            )
    
    _dequant_bias_fn = module.dequant_bias_triton
    logger.info_once("Dequant+bias kernel loaded")
    return _dequant_bias_fn


# 缓存的零 bias tensor（按 N 维度缓存，避免重复创建）
_zero_bias_cache: dict[tuple[int, torch.device], torch.Tensor] = {}


def _get_zero_bias(N: int, device: torch.device) -> torch.Tensor:
    """获取缓存的零 bias tensor"""
    key = (N, device)
    if key not in _zero_bias_cache:
        _zero_bias_cache[key] = torch.zeros(N, dtype=torch.bfloat16, device=device)
    return _zero_bias_cache[key]


def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequant + Bias
    
    计算: output[M, N] = gemm_output[M, N] * scale_a[M] * scale_b[N] + bias[N]
    """
    fn = _load_dequant_bias_kernel()
    if bias is None:
        # 使用缓存的零 bias，避免每次调用都分配内存
        bias = _get_zero_bias(gemm_output.shape[1], gemm_output.device)
    return fn(gemm_output, scale_a, scale_b, bias, out_dtype)


# ============================================================================
# Quant Only Kernel (FP8) - cuBLASLt 专用
# ============================================================================

_quant_only_fp8_fn = None


def _load_quant_only_fp8_kernel():
    """加载 Triton FP8 quant kernel（懒加载）"""
    global _quant_only_fp8_fn
    if _quant_only_fp8_fn is not None:
        return _quant_only_fp8_fn
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("quant_only_tuned", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        _build_search_kernel(kernel_dir, kernel_type="quant_fp8")
        try:
            module = load_module("quant_only_tuned", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FP8 quant kernel not found after autotune."
            )
    
    _quant_only_fp8_fn = module.quant_only_fp8_triton
    logger.info_once("FP8 quant kernel loaded")
    return _quant_only_fp8_fn


def quant_only_fp8_kernel(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization
    
    Args:
        input: [M, K] BF16
        
    Returns:
        qinput: [M_pad, K_pad] FP8，M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    fn = _load_quant_only_fp8_kernel()
    return fn(input)


# ============================================================================
# Quant + Slide Kernel (FP8) - cuSPARSELt 专用
# ============================================================================

_quant_slide_fp8_fn = None


def _load_quant_slide_fp8_kernel():
    """加载 Triton FP8 quant+slide kernel（懒加载）"""
    global _quant_slide_fp8_fn
    if _quant_slide_fp8_fn is not None:
        return _quant_slide_fp8_fn
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("quant_slide_tuned", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        _build_search_kernel(kernel_dir, kernel_type="quant_slide_fp8")
        try:
            module = load_module("quant_slide_tuned", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"FP8 quant+slide kernel not found after autotune."
            )
    
    _quant_slide_fp8_fn = module.quant_slide_fp8_triton
    logger.info_once("FP8 quant+slide kernel loaded")
    return _quant_slide_fp8_fn


def quant_slide_fp8_kernel(
    input: torch.Tensor,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization + SlideSparse Slide
    
    Args:
        input: [M, K] BF16
        L: 稀疏组大小（默认 8）
        
    Returns:
        qinput: [M_pad, K_slide_pad] FP8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
                K_slide = num_groups * (L/2 - 1) * 4
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    fn = _load_quant_slide_fp8_kernel()
    return fn(input, L)


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
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    quant_fn: Optional[QuantFP8] = None,
    input_scale: Optional[torch.Tensor] = None,
    input_scale_ub: Optional[torch.Tensor] = None,
    **kwargs,
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
    ext = _get_gemm_extension("cublaslt")
    M = input.shape[0]
    
    # Quant: [M, K] -> [M_pad, K_pad]
    # cuBLASLt 路径始终使用 Triton quant kernel（需要 padding）
    if input.dtype != current_platform.fp8_dtype():
        qinput, scale_a_pad = quant_only_fp8_kernel(input)
    else:
        # 静态量化：input 已是 FP8，但没有 padding
        # cuBLASLt GEMM wrapper 期望 padded 维度，所以不支持静态量化
        raise NotImplementedError(
            "cuBLASLt with static quantization is not supported. "
            "Use CUTLASS path or dynamic quantization."
        )
    
    try:
        # GEMM: Weight @ Activation -> [M_pad, N]
        gemm_out_pad = ext.cublaslt_fp8_mm(weight, qinput, get_inner_dtype_str())
        
        # 截断 M_pad -> M（view 操作，无数据拷贝）
        gemm_out = gemm_out_pad[:M, :]
        scale_a = scale_a_pad[:M]
        
        # Dequant: [M, N]
        output = dequant_bias_kernel(gemm_out, scale_a, scale_b, bias, out_dtype)
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
    slide_weight_N: Optional[int] = None,
    slide_weight_K: Optional[int] = None,
    L: int = 8,
    **kwargs,
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
    ext = _get_gemm_extension("cusparselt")
    
    if slide_weight_N is None or slide_weight_K is None:
        raise ValueError(
            "cuSPARSELt requires slide_weight_N and slide_weight_K."
        )
    
    M = input.shape[0]
    
    # Quant + Slide: [M, K] -> [M_pad, K_slide_pad]
    if input.dtype != current_platform.fp8_dtype():
        qinput, scale_a_pad = quant_slide_fp8_kernel(input, L)
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
    
    try:
        # GEMM: Weight_compressed @ Activation -> [M_pad, N]
        gemm_out_pad = ext.cusparselt_fp8_mm(
            slide_weight_compressed,
            qinput,
            slide_weight_N,
            K_slide_pad,
            get_inner_dtype_str()
        )
        
        # 截断 M_pad -> M（view 操作，无数据拷贝）
        gemm_out = gemm_out_pad[:M, :]
        scale_a = scale_a_pad[:M]
        
        # Dequant: [M, N]
        output = dequant_bias_kernel(gemm_out, scale_a, scale_b, bias, out_dtype)
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
        assert quant_fn is not None, "quant_fn required for non-FP8 input"
        qinput, scale_a = quant_fn(input, input_scale, input_scale_ub)
    else:
        qinput, scale_a = input, input_scale
    
    # CUTLASS 融合 GEMM + Dequant + Bias
    output = ops.cutlass_scaled_mm(
        qinput, weight, out_dtype=out_dtype,
        scale_a=scale_a, scale_b=scale_b, bias=bias
    )
    return output.view(*output_shape)


# ============================================================================
# SlideSparse FP8 Linear Op
# ============================================================================

class SlideSparseFp8LinearOp:
    """
    SlideSparse FP8 Linear Operation
    
    根据环境变量选择 kernel 路径：
    - USE_CUBLASLT=1: cuBLASLt_FP8_linear
    - USE_CUSPARSELT=1: cuSPARSELt_FP8_linear
    - 默认: cutlass_FP8_linear (fallback)
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
        
        if self._use_cublaslt:
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_FP8_linear
        elif self._use_cusparselt:
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_FP8_linear
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
        
        # 调用选定的 kernel 路径
        if self._use_cusparselt:
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
                L=L,
            )
        else:
            # cuBLASLt 或 CUTLASS 路径
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
        """创建权重参数（委托给原始 scheme）"""
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
        """应用权重（执行线性变换）"""
        if self._use_cusparselt:
            return self.slidesparse_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                out_dtype=self.out_dtype,
                input_scale=getattr(layer, "input_scale", None),
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
                input_scale=getattr(layer, "input_scale", None),
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
