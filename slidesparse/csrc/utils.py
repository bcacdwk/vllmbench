#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Shared utilities for SlideSparse CSRC modules

Common functions for:
- Python/arch/GPU version tagging
- File naming conventions
- Extension loading
"""

import sys
import platform
from pathlib import Path

import torch


# =============================================================================
# Version & Architecture Tags
# =============================================================================

def get_python_version_tag() -> str:
    """Get Python version tag, e.g., 'py312'"""
    return f"py{sys.version_info.major}{sys.version_info.minor}"


def get_arch_tag() -> str:
    """Get system architecture tag, e.g., 'x86_64' or 'aarch64'"""
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        return "x86_64"
    elif machine in ("aarch64", "arm64"):
        return "aarch64"
    return machine.lower()


def get_gpu_cc() -> str:
    """Get GPU Compute Capability tag, e.g., 'cc90'"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    return f"cc{prop.major}{prop.minor}"


def get_gpu_name() -> str:
    """Get GPU short name, e.g., 'H100', 'A100'"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    name = prop.name.replace("NVIDIA ", "").split()[0]
    return name


# =============================================================================
# File Naming Conventions
# =============================================================================

def get_extension_name(prefix: str) -> str:
    """
    Generate extension name with version info
    
    Format: {prefix}_py312_x86_64_cc90
    
    Args:
        prefix: Extension prefix, e.g., 'cublaslt', 'cusparselt'
    """
    return f"{prefix}_{get_python_version_tag()}_{get_arch_tag()}_{get_gpu_cc()}"


def get_tuned_kernel_filename(prefix: str, dtype_tag: str) -> str:
    """
    Generate tuned kernel filename
    
    Format: {prefix}_py312_x86_64_cc90_{dtype_tag}.py
    
    Args:
        prefix: Kernel prefix, e.g., 'dequant_bias_tuned'
        dtype_tag: Data type tag, e.g., 'BF16', 'FP32'
    """
    return f"{prefix}_{get_python_version_tag()}_{get_arch_tag()}_{get_gpu_cc()}_{dtype_tag}.py"


# =============================================================================
# System Info
# =============================================================================

def print_system_info():
    """Print system and GPU information"""
    print(f"GPU:     {torch.cuda.get_device_name()}")
    print(f"CC:      {get_gpu_cc()}")
    print(f"Python:  {get_python_version_tag()}")
    print(f"Arch:    {get_arch_tag()}")
    print(f"PyTorch: {torch.__version__}")
    try:
        import triton
        print(f"Triton:  {triton.__version__}")
    except ImportError:
        pass


# =============================================================================
# Module Loading (通用模块加载器)
# =============================================================================

# 模块缓存：{(prefix, build_dir, dtype_tag): module}
_module_cache = {}


def load_tuned_module(prefix: str, build_dir: Path, dtype_tag: str = None):
    """
    加载 autotuned 模块（通用接口）
    
    根据当前环境自动构建模块名并加载：
        {prefix}_{py_tag}_{arch_tag}_{cc_tag}[_{dtype_tag}]
    
    示例:
        - load_tuned_module("dequant_bias_tuned", build_dir, "BF16")
          -> dequant_bias_tuned_py312_x86_64_cc120_BF16
        - load_tuned_module("cublaslt", build_dir)
          -> cublaslt_py312_x86_64_cc120
    
    Args:
        prefix: 模块前缀，如 'dequant_bias_tuned', 'cublaslt'
        build_dir: 构建目录的 Path 或 str
        dtype_tag: 可选的数据类型标签，如 'BF16', 'FP32'
        
    Returns:
        加载的 Python 模块
        
    Raises:
        FileNotFoundError: 模块文件不存在
        RuntimeError: CUDA 不可用
    """
    import importlib
    
    build_dir = Path(build_dir)
    cache_key = (prefix, str(build_dir), dtype_tag)
    
    # 检查缓存
    if cache_key in _module_cache:
        return _module_cache[cache_key]
    
    # 构建模块名
    if dtype_tag:
        module_name = f"{prefix}_{get_python_version_tag()}_{get_arch_tag()}_{get_gpu_cc()}_{dtype_tag}"
    else:
        module_name = f"{prefix}_{get_python_version_tag()}_{get_arch_tag()}_{get_gpu_cc()}"
    
    # 检查文件是否存在（.py 或 .so）
    py_file = build_dir / f"{module_name}.py"
    so_file = build_dir / f"{module_name}.so"
    
    if not py_file.exists() and not so_file.exists():
        raise FileNotFoundError(
            f"模块不存在: {module_name}\n"
            f"搜索路径: {build_dir}\n"
            f"期望文件: {py_file.name} 或 {so_file.name}\n"
            f"请先构建/autotune 生成对应模块。"
        )
    
    # 添加到 sys.path 并导入
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    
    module = importlib.import_module(module_name)
    _module_cache[cache_key] = module
    
    return module


# =============================================================================
# Triton Autotune Configs for Dequant+Bias Kernel
# =============================================================================

def get_dequant_bias_autotune_configs():
    """
    Get comprehensive autotune configs for dequant+bias kernel.
    
    Coverage: SM80(A100), SM89(4090), SM90(H100), SM100(B200), SM120(5080)
    
    Configs are organized by proven performance from A100 validation.
    
    Returns list of triton.Config objects.
    """
    import triton
    
    return [
        # =====================================================================
        # Tier 1: Proven Winners (A100 validated)
        # =====================================================================
        # Small M King (M=1~128): 32x32
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        # Medium M King (M=256~8192): 64x32
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        # Large M King (M=12288+): 128x64
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),

        # =====================================================================
        # Tier 2: Basic kernel heuristics (must include for fair comparison)
        # From _get_best_config() in basic_dequant_bias_triton.py
        # =====================================================================
        # Small M, N<=4096: (32, 64, 4)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # Small M, N>4096: (32, 128, 4)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        # Medium M, N<=4096: (64, 64, 4)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # Medium M, N>4096: (64, 128, 8)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        # Large M, N>4096: (128, 128, 8)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),

        # =====================================================================
        # Tier 3: Read/Write bias exploration
        # =====================================================================
        # Write Heavy (tall blocks): 128x32
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=4),
        # Read Heavy (wide blocks): 64x128 with lower warps
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        # Balanced high warp: 64x64 w=8
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        # Low warp large block
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),

        # =====================================================================
        # Tier 4: H100/Blackwell exploration (SM90/100/120)
        # =====================================================================
        # Super Wide: 256x64
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        # Super Tall: 64x256
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        # Super Square: 128x128 high warp
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=4),
        # Wide variants for large N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8, num_stages=4),
        # Extreme Wide: 256x32
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=4),

        # =====================================================================
        # Tier 5: Small M + Large N special cases (unbalanced shapes)
        # Critical for M=16~128 with N=2560~13824 (BitNet hidden sizes)
        # =====================================================================
        # 32x128 with various warps (key for M=16, N=13824)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # 32x64 with high warps (key for M=128, N=2560)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        # 32x64 with low warps (batch sizes)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=3),

        # =====================================================================
        # Tier 6: Tiny M = 16 special cases
        # =====================================================================
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2, num_stages=3),
        # Very wide for large N
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=3),
    ]


__all__ = [
    'get_python_version_tag',
    'get_arch_tag',
    'get_gpu_cc',
    'get_gpu_name',
    'get_extension_name',
    'get_tuned_kernel_filename',
    'print_system_info',
    'load_tuned_module',
    'get_dequant_bias_autotune_configs',
]
