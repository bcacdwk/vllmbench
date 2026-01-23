# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Triton Kernel 加载模块

包含:
- Dequant + Bias kernel（FP8/INT8 共享）
- FP8 Quant kernels（quant_only, quant_slide）
- INT8 Quant kernels（quant_only, quant_slide）

所有 kernel 采用懒加载模式，首次调用时加载。
如果 kernel 文件不存在，会自动触发编译。
"""

import subprocess
from pathlib import Path
from typing import Optional

import torch

from vllm.logger import init_logger

from slidesparse.utils import load_module

logger = init_logger(__name__)


# ============================================================================
# 目录配置
# ============================================================================

_CSRC_DIR = Path(__file__).parent.parent / "csrc"


# ============================================================================
# Kernel 构建配置
# ============================================================================

_KERNEL_BUILD_CONFIG = {
    # GEMM 库编译
    "cublaslt":        ("build_cublaslt.py",                ["build"]),
    "cusparselt":      ("build_cusparselt.py",              ["build"]),
    
    # Dequant kernel（FP8/INT8 共享）
    "dequant_bias":    ("autotune_autogen_dequant_bias.py", ["--quick"]),
    
    # FP8 Quant kernels
    "quant_fp8":       ("autotune_autogen_quant_only.py",   ["--quick", "--dtype", "fp8"]),
    "quant_slide_fp8": ("autotune_autogen_quant_slide.py",  ["--quick"]),
    
    # INT8 Quant kernels
    "quant_int8":       ("autotune_autogen_quant_only.py",   ["--quick", "--dtype", "int8"]),
    "quant_slide_int8": ("autotune_autogen_quant_slide.py",  ["--quick", "--dtype", "int8"]),
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
        subprocess.run(
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
# Dequant + Bias Kernel（FP8/INT8 共享）
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
                f"Dequant+bias kernel not found after autotune.\n"
                f"Expected location: {build_dir}/dequant_bias_tuned_*.py"
            ) from None
    
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
    Dequant + Bias: output = gemm_output * scale_a * scale_b + bias
    
    支持 BF16、FP32、INT32 输入（自动检测）
    """
    if _dequant_bias_fn is None:
        _load_dequant_bias_kernel()
    
    if bias is None:
        bias = torch.zeros(
            gemm_output.shape[1],
            dtype=gemm_output.dtype,
            device=gemm_output.device
        )
    return _dequant_bias_fn(gemm_output, scale_a, scale_b, bias, out_dtype)


# ============================================================================
# FP8 Quant Only Kernel - cuBLASLt 专用
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
                f"FP8 quant kernel not found after autotune.\n"
                f"Expected location: {build_dir}/quant_only_tuned_*.py"
            ) from None
    
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
    if _quant_only_fp8_fn is None:
        _load_quant_only_fp8_kernel()
    return _quant_only_fp8_fn(input)


# ============================================================================
# FP8 Quant + Slide Kernel - cuSPARSELt 专用
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
                f"FP8 quant+slide kernel not found after autotune.\n"
                f"Expected location: {build_dir}/quant_slide_tuned_*.py"
            ) from None
    
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
                K_slide = K * (L - 2) / (L / 2) = K * 2 * (L - 2) / L
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    if _quant_slide_fp8_fn is None:
        _load_quant_slide_fp8_kernel()
    return _quant_slide_fp8_fn(input, L)


# ============================================================================
# INT8 Quant Only Kernel - cuBLASLt 专用
# ============================================================================

_quant_only_int8_fn = None


def _load_quant_only_int8_kernel():
    """加载 Triton INT8 quant kernel（懒加载）"""
    global _quant_only_int8_fn
    if _quant_only_int8_fn is not None:
        return _quant_only_int8_fn
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("quant_only_tuned", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        _build_search_kernel(kernel_dir, kernel_type="quant_int8")
        try:
            module = load_module("quant_only_tuned", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"INT8 quant kernel not found after autotune.\n"
                f"Expected location: {build_dir}/quant_only_tuned_*.py"
            ) from None
    
    _quant_only_int8_fn = module.quant_only_int8_triton
    logger.info_once("INT8 quant kernel loaded")
    return _quant_only_int8_fn


def quant_only_int8_kernel(
    input: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization (Symmetric)
    
    Args:
        input: [M, K] BF16
        
    Returns:
        qinput: [M_pad, K_pad] INT8，M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    if _quant_only_int8_fn is None:
        _load_quant_only_int8_kernel()
    return _quant_only_int8_fn(input)


# ============================================================================
# INT8 Quant + Slide Kernel - cuSPARSELt 专用
# ============================================================================

_quant_slide_int8_fn = None


def _load_quant_slide_int8_kernel():
    """加载 Triton INT8 quant+slide kernel（懒加载）"""
    global _quant_slide_int8_fn
    if _quant_slide_int8_fn is not None:
        return _quant_slide_int8_fn
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    build_dir = kernel_dir / "build"
    
    try:
        module = load_module("quant_slide_tuned", search_dir=build_dir, ext=".py")
    except FileNotFoundError:
        _build_search_kernel(kernel_dir, kernel_type="quant_slide_int8")
        try:
            module = load_module("quant_slide_tuned", search_dir=build_dir, ext=".py")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"INT8 quant+slide kernel not found after autotune.\n"
                f"Expected location: {build_dir}/quant_slide_tuned_*.py"
            ) from None
    
    _quant_slide_int8_fn = module.quant_slide_int8_triton
    logger.info_once("INT8 quant+slide kernel loaded")
    return _quant_slide_int8_fn


def quant_slide_int8_kernel(
    input: torch.Tensor,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization + SlideSparse Slide (Symmetric)
    
    Args:
        input: [M, K] BF16
        L: 稀疏组大小（默认 8）
        
    Returns:
        qinput: [M_pad, K_slide_pad] INT8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    if _quant_slide_int8_fn is None:
        _load_quant_slide_int8_kernel()
    return _quant_slide_int8_fn(input, L)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Build helper
    "_build_search_kernel",
    "_KERNEL_BUILD_CONFIG",
    "_CSRC_DIR",
    
    # Dequant kernel（共享）
    "_load_dequant_bias_kernel",
    "dequant_bias_kernel",
    
    # FP8 Quant kernels
    "_load_quant_only_fp8_kernel",
    "quant_only_fp8_kernel",
    "_load_quant_slide_fp8_kernel",
    "quant_slide_fp8_kernel",
    
    # INT8 Quant kernels
    "_load_quant_only_int8_kernel",
    "quant_only_int8_kernel",
    "_load_quant_slide_int8_kernel",
    "quant_slide_int8_kernel",
]
