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
        # Tier 5: Tiny M special cases
        # =====================================================================
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
    ]


__all__ = [
    'get_python_version_tag',
    'get_arch_tag',
    'get_gpu_cc',
    'get_gpu_name',
    'get_extension_name',
    'get_tuned_kernel_filename',
    'print_system_info',
    'get_dequant_bias_autotune_configs',
]
