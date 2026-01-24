# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Triton Kernel 加载模块

包含:
- Dequant + Bias kernel（FP8/INT8 共享）
- FP8 Quant kernels（quant_only, quant_slide）
- INT8 Quant kernels（quant_only, quant_slide）

所有 kernel 采用懒加载模式，首次调用时加载。
每个 model 有独立的 tuned kernel 缓存。
如果 kernel 文件不存在，会自动触发编译。

新目录结构:
    build/{hw_dir_name}/{kernel_name}_tuned_{model_name}.py
    
例如:
    build/RTX5080_cc120_py312_cu129_x86_64/dequant_bias_tuned_Llama3.2-1B-FP8.py
"""

from pathlib import Path
from typing import Callable, Dict, Optional

import torch

from vllm.logger import init_logger

from slidesparse.utils import load_tuned_module, build_hw_dir_name

logger = init_logger(__name__)


# ============================================================================
# 目录配置
# ============================================================================

_CSRC_DIR = Path(__file__).parent.parent / "csrc"


# ============================================================================
# 模型名辅助函数
# ============================================================================

def _extract_base_model_name(model_name: str) -> str:
    """
    从完整模型名中提取基础模型名
    
    例如:
        Llama3.2-1B-FP8-SlideSparse-2_8 -> Llama3.2-1B-FP8
        Qwen2.5-0.5B-INT8-SlideSparse-2_10 -> Qwen2.5-0.5B-INT8
        Llama3.2-1B-FP8 -> Llama3.2-1B-FP8 (不变)
    """
    marker = "-SlideSparse-"
    if marker in model_name:
        return model_name.split(marker)[0]
    return model_name


# ============================================================================
# Kernel 搜索配置
# ============================================================================

# Basic kernel 文件名映射
_BASIC_KERNEL_FILES = {
    "dequant_bias": "basic_dequant_bias_triton.py",
    "quant_only":   "basic_quant_only_triton.py",
    "quant_slide":  "basic_quant_slide_triton.py",
}


def _load_basic_kernel(kernel_dir: Path, kernel_type: str) -> object:
    """加载 basic kernel 模块（无 model-specific tuning）"""
    if kernel_type not in _BASIC_KERNEL_FILES:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    basic_file = kernel_dir / _BASIC_KERNEL_FILES[kernel_type]
    if not basic_file.exists():
        raise FileNotFoundError(f"Basic kernel not found: {basic_file}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(kernel_type, basic_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    logger.info_once(f"Using basic kernel: {basic_file.name}")
    return module


def _search_kernel(
    kernel_dir: Path,
    kernel_type: str,
    tuned_prefix: str,
    model_name: str,
) -> object:
    """
    搜索并加载 kernel 模块
    
    搜索顺序:
    1. 首先尝试加载 tuned kernel: build/{hw_dir}/{tuned_prefix}_{base_model}.py
    2. 如果找不到，fallback 到 basic kernel: basic_{kernel_type}_triton.py
    
    Args:
        kernel_dir: kernel 目录（如 csrc/fused_dequant_bias_triton）
        kernel_type: kernel 类型（dequant_bias, quant_only, quant_slide）
        tuned_prefix: tuned 文件前缀（如 dequant_bias_tuned）
        model_name: 模型名称（可能包含 -SlideSparse-2_L 后缀）
    
    Returns:
        加载的模块对象
    """
    build_dir = kernel_dir / "build"
    
    # 提取基础模型名用于查找 tuned kernel
    base_model = _extract_base_model_name(model_name)
    
    # 1. 尝试加载 tuned kernel
    try:
        module = load_tuned_module(tuned_prefix, base_model, build_dir)
        if base_model != model_name:
            logger.info_once(f"Loaded tuned kernel for base model: {base_model} (from {model_name})")
        else:
            logger.info_once(f"Loaded tuned kernel for model: {model_name}")
        return module
    except FileNotFoundError:
        pass
    
    # 2. Fallback 到 basic kernel
    logger.warning(f"Tuned kernel not found for {base_model}, using basic kernel")
    return _load_basic_kernel(kernel_dir, kernel_type)


# ============================================================================
# Dequant + Bias Kernel（FP8/INT8 共享）
# ============================================================================

# 缓存: model_name -> kernel function
_dequant_bias_cache: Dict[str, Callable] = {}


def _load_dequant_bias_kernel(model_name: str) -> Callable:
    """加载 Triton dequant+bias kernel（按 model 懒加载）"""
    if model_name in _dequant_bias_cache:
        return _dequant_bias_cache[model_name]
    
    kernel_dir = _CSRC_DIR / "fused_dequant_bias_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="dequant_bias",
        tuned_prefix="dequant_bias_tuned",
        model_name=model_name,
    )
    
    fn = module.dequant_bias_triton
    _dequant_bias_cache[model_name] = fn
    logger.info_once(f"Dequant+bias kernel loaded for model: {model_name}")
    return fn


def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    model_name: str,
) -> torch.Tensor:
    """
    Dequant + Bias: output = gemm_output * scale_a * scale_b + bias
    
    支持 BF16、FP32、INT32 输入（自动检测）
    
    Args:
        gemm_output: GEMM 输出
        scale_a: 激活 scale
        scale_b: 权重 scale  
        bias: 可选的 bias
        out_dtype: 输出类型
        model_name: 模型名称（用于加载对应的 tuned kernel）
    """
    fn = _load_dequant_bias_kernel(model_name)
    
    if bias is None:
        bias = torch.zeros(
            gemm_output.shape[1],
            dtype=gemm_output.dtype,
            device=gemm_output.device
        )
    return fn(gemm_output, scale_a, scale_b, bias, out_dtype)


# ============================================================================
# FP8 Quant Only Kernel - cuBLASLt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_only_fp8_cache: Dict[str, Callable] = {}


def _load_quant_only_fp8_kernel(model_name: str) -> Callable:
    """加载 Triton FP8 quant kernel（按 model 懒加载）"""
    if model_name in _quant_only_fp8_cache:
        return _quant_only_fp8_cache[model_name]
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_only",
        tuned_prefix="quant_only_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_only_fp8_triton
    _quant_only_fp8_cache[model_name] = fn
    logger.info_once(f"FP8 quant kernel loaded for model: {model_name}")
    return fn


def quant_only_fp8_kernel(
    input: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        
    Returns:
        qinput: [M_pad, K_pad] FP8，M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    fn = _load_quant_only_fp8_kernel(model_name)
    return fn(input)


# ============================================================================
# FP8 Quant + Slide Kernel - cuSPARSELt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_slide_fp8_cache: Dict[str, Callable] = {}


def _load_quant_slide_fp8_kernel(model_name: str) -> Callable:
    """加载 Triton FP8 quant+slide kernel（按 model 懒加载）"""
    if model_name in _quant_slide_fp8_cache:
        return _quant_slide_fp8_cache[model_name]
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_slide",
        tuned_prefix="quant_slide_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_slide_fp8_triton
    _quant_slide_fp8_cache[model_name] = fn
    logger.info_once(f"FP8 quant+slide kernel loaded for model: {model_name}")
    return fn


def quant_slide_fp8_kernel(
    input: torch.Tensor,
    model_name: str,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FP8 Per-token Quantization + SlideSparse Slide
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        L: 稀疏组大小（默认 8）
        
    Returns:
        qinput: [M_pad, K_slide_pad] FP8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
                K_slide = K * (L - 2) / (L / 2) = K * 2 * (L - 2) / L
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    fn = _load_quant_slide_fp8_kernel(model_name)
    return fn(input, L)


# ============================================================================
# INT8 Quant Only Kernel - cuBLASLt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_only_int8_cache: Dict[str, Callable] = {}


def _load_quant_only_int8_kernel(model_name: str) -> Callable:
    """加载 Triton INT8 quant kernel（按 model 懒加载）"""
    if model_name in _quant_only_int8_cache:
        return _quant_only_int8_cache[model_name]
    
    kernel_dir = _CSRC_DIR / "quant_only_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_only",
        tuned_prefix="quant_only_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_only_int8_triton
    _quant_only_int8_cache[model_name] = fn
    logger.info_once(f"INT8 quant kernel loaded for model: {model_name}")
    return fn


def quant_only_int8_kernel(
    input: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization (Symmetric)
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        
    Returns:
        qinput: [M_pad, K_pad] INT8，M_pad=ceil16(M), K_pad=ceil32(K)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    fn = _load_quant_only_int8_kernel(model_name)
    return fn(input)


# ============================================================================
# INT8 Quant + Slide Kernel - cuSPARSELt 专用
# ============================================================================

# 缓存: model_name -> kernel function
_quant_slide_int8_cache: Dict[str, Callable] = {}


def _load_quant_slide_int8_kernel(model_name: str) -> Callable:
    """加载 Triton INT8 quant+slide kernel（按 model 懒加载）"""
    if model_name in _quant_slide_int8_cache:
        return _quant_slide_int8_cache[model_name]
    
    kernel_dir = _CSRC_DIR / "fused_quant_slide_triton"
    module = _search_kernel(
        kernel_dir,
        kernel_type="quant_slide",
        tuned_prefix="quant_slide_tuned",
        model_name=model_name,
    )
    
    fn = module.quant_slide_int8_triton
    _quant_slide_int8_cache[model_name] = fn
    logger.info_once(f"INT8 quant+slide kernel loaded for model: {model_name}")
    return fn


def quant_slide_int8_kernel(
    input: torch.Tensor,
    model_name: str,
    L: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    INT8 Per-token Quantization + SlideSparse Slide (Symmetric)
    
    Args:
        input: [M, K] BF16
        model_name: 模型名称（用于加载对应的 tuned kernel）
        L: 稀疏组大小（默认 8）
        
    Returns:
        qinput: [M_pad, K_slide_pad] INT8
                M_pad=ceil16(M), K_slide_pad=ceil32(K_slide)
        scale_a: [M_pad] FP32，padding 区域为 1.0
    """
    fn = _load_quant_slide_int8_kernel(model_name)
    return fn(input, L)


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Search helper
    "_search_kernel",
    "_load_basic_kernel",
    "_BASIC_KERNEL_FILES",
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
