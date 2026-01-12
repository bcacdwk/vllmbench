#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse CSRC 编译工具库

本模块提供 CUDA 扩展编译相关的通用工具函数。

注意：文件名、硬件信息等功能请使用顶层 slidesparse.utils 模块。

主要功能
========
1. NVCC 架构标志生成
2. CUDA 扩展编译器（支持 cuBLASLt, cuSPARSELt 等）
3. 编译产物清理
4. Triton Autotune 配置

使用示例
========
>>> from slidesparse.csrc.utils import build_cuda_extension, get_nvcc_arch_flags
>>>
>>> # 编译 cuBLASLt 扩展
>>> so_path = build_cuda_extension(
...     name="cublaslt_gemm",
...     source_file=Path("cublaslt_gemm.cu"),
...     build_dir=Path("build"),
...     extra_ldflags=["-lcublasLt", "-lcublas"],
... )
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional, Callable

import torch
from torch.utils.cpp_extension import load


# =============================================================================
# NVCC 架构标志
# =============================================================================

# 支持的 GPU 架构列表
SUPPORTED_ARCHITECTURES = [
    ("80", "sm_80"),   # Ampere (A100, A10, A30)
    ("86", "sm_86"),   # Ampere (RTX 30xx)
    ("89", "sm_89"),   # Ada Lovelace (RTX 40xx)
    ("90", "sm_90"),   # Hopper (H100, H200)
    ("100", "sm_100"), # Blackwell (B100, B200)
    ("120", "sm_120"), # Blackwell (RTX 50xx, GB10)
]


def get_nvcc_arch_flags(
    min_compute: int = 80,
    max_compute: int = 120,
) -> List[str]:
    """
    生成 nvcc 架构编译选项
    
    支持从 SM 80 (Ampere) 到 SM 120 (Blackwell)
    
    Args:
        min_compute: 最小支持的 compute capability (默认 80)
        max_compute: 最大支持的 compute capability (默认 120)
        
    Returns:
        nvcc -gencode 标志列表
        
    Example:
        >>> get_nvcc_arch_flags()
        ['-gencode=arch=compute_80,code=sm_80', ...]
    """
    flags = []
    for compute, sm in SUPPORTED_ARCHITECTURES:
        cc = int(compute)
        if min_compute <= cc <= max_compute:
            flags.append(f"-gencode=arch=compute_{compute},code={sm}")
    return flags


def get_current_arch_flag() -> str:
    """
    获取当前 GPU 架构的 nvcc 编译标志
    
    Returns:
        单个 -gencode 标志，针对当前 GPU
        
    Raises:
        RuntimeError: CUDA 不可用
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    prop = torch.cuda.get_device_properties(0)
    compute = f"{prop.major}{prop.minor}"
    return f"-gencode=arch=compute_{compute},code=sm_{compute}"


# =============================================================================
# CUDA 扩展编译器
# =============================================================================

# 默认编译选项
DEFAULT_CFLAGS = ['-O3', '-std=c++17']

DEFAULT_CUDA_CFLAGS = [
    '-O3',
    '-std=c++17',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
]


def should_rebuild(so_path: Path, source_paths: List[Path]) -> bool:
    """
    判断是否需要重新编译
    
    如果 .so 不存在或比任一源文件旧，返回 True
    
    Args:
        so_path: .so 文件路径
        source_paths: 源文件路径列表
        
    Returns:
        是否需要重新编译
    """
    if not so_path.exists():
        return True
    
    so_mtime = so_path.stat().st_mtime
    for src in source_paths:
        if src.exists() and src.stat().st_mtime > so_mtime:
            return True
    return False


def clean_build_artifacts(build_dir: Path, keep_extensions: List[str] = None):
    """
    清理编译中间文件
    
    默认保留 .so 和 .py 文件，删除其他所有内容。
    
    Args:
        build_dir: 构建目录
        keep_extensions: 要保留的文件扩展名列表（默认 ['.so', '.py']）
    """
    if keep_extensions is None:
        keep_extensions = ['.so', '.py']
    
    if not build_dir.exists():
        return
    
    for item in build_dir.iterdir():
        # 保留指定扩展名的文件
        if item.suffix in keep_extensions:
            continue
        
        # 删除其他文件和目录
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def build_cuda_extension(
    name: str,
    source_file: Path,
    build_dir: Path,
    *,
    extra_cflags: List[str] = None,
    extra_cuda_cflags: List[str] = None,
    extra_ldflags: List[str] = None,
    extra_include_paths: List[str] = None,
    force: bool = False,
    verbose: bool = True,
    clean_after_build: bool = True,
) -> Path:
    """
    编译 CUDA 扩展的通用函数
    
    支持编译 cuBLASLt, cuSPARSELt 等 CUDA 扩展。
    
    Args:
        name: 扩展名称（不含 .so 后缀）
        source_file: 源文件路径 (.cu 或 .cpp)
        build_dir: 构建目录
        extra_cflags: 额外的 C++ 编译标志
        extra_cuda_cflags: 额外的 CUDA 编译标志
        extra_ldflags: 额外的链接标志（如 -lcublasLt）
        extra_include_paths: 额外的头文件搜索路径
        force: 是否强制重新编译
        verbose: 是否显示详细输出
        clean_after_build: 编译后是否清理中间文件
        
    Returns:
        编译生成的 .so 文件路径
        
    Raises:
        FileNotFoundError: 源文件不存在
        RuntimeError: 编译失败
        
    Example:
        >>> so_path = build_cuda_extension(
        ...     name="cublaslt_gemm_H100_cc90_FP8E4M3_py312_cu124_x86_64",
        ...     source_file=Path("cublaslt_gemm.cu"),
        ...     build_dir=Path("build"),
        ...     extra_ldflags=["-lcublasLt", "-lcublas", "-lcuda"],
        ... )
    """
    # 验证源文件
    if not source_file.exists():
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    # 确保 build 目录存在
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找已存在的 .so
    so_pattern = f"{name}*.so"
    existing_sos = list(build_dir.glob(so_pattern))
    
    if existing_sos and not force:
        so_path = existing_sos[0]
        if not should_rebuild(so_path, [source_file]):
            if verbose:
                print(f"✓ Using existing: {so_path.name}")
            return so_path
        elif verbose:
            print(f"⚠ Source changed, rebuilding...")
    
    if verbose:
        print(f"🔨 Building {name}...")
    
    # CUDA 路径
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    
    # 合并编译选项
    cflags = DEFAULT_CFLAGS + (extra_cflags or [])
    cuda_cflags = DEFAULT_CUDA_CFLAGS + get_nvcc_arch_flags() + (extra_cuda_cflags or [])
    ldflags = extra_ldflags or []
    include_paths = [os.path.join(cuda_home, 'include')] + (extra_include_paths or [])
    
    # 编译
    try:
        load(
            name=name,
            sources=[str(source_file)],
            extra_cflags=cflags,
            extra_cuda_cflags=cuda_cflags,
            extra_ldflags=ldflags,
            extra_include_paths=include_paths,
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        raise RuntimeError(f"编译失败: {e}") from e
    
    # 查找生成的 .so
    new_sos = list(build_dir.glob(so_pattern))
    if not new_sos:
        raise RuntimeError(f"编译完成但未找到 .so 文件: {so_pattern}")
    
    so_path = new_sos[0]
    
    if verbose:
        print(f"✓ Built: {so_path.name}")
    
    # 清理中间文件
    if clean_after_build:
        if verbose:
            print(f"🧹 Cleaning build artifacts...")
        clean_build_artifacts(build_dir)
    
    return so_path


# =============================================================================
# 特定扩展的链接库配置
# =============================================================================

# cuBLASLt 扩展所需的链接库
CUBLASLT_LDFLAGS = ['-lcublasLt', '-lcublas', '-lcuda']

# cuSPARSELt 扩展所需的链接库
CUSPARSELT_LDFLAGS = ['-lcusparseLt', '-lcusparse', '-lcuda']


def get_gemm_ldflags(backend: str) -> List[str]:
    """
    获取 GEMM 后端所需的链接库标志
    
    Args:
        backend: 后端名称 ("cublaslt" 或 "cusparselt")
        
    Returns:
        链接库标志列表
        
    Raises:
        ValueError: 未知的后端
    """
    if backend.lower() == "cublaslt":
        return CUBLASLT_LDFLAGS.copy()
    elif backend.lower() == "cusparselt":
        return CUSPARSELT_LDFLAGS.copy()
    else:
        raise ValueError(f"未知的 GEMM 后端: {backend}")


# =============================================================================
# Triton Autotune 配置
# =============================================================================

def get_dequant_autotune_configs():
    """
    获取 dequant+bias kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100), SM100(B200), SM120(5080)
    
    因为 M 是灵活可变的batchsize 此处相当于是搜索 R[M,N] = A[M,K] * W[N,K] 中的 [M,N]
    
    Returns:
        triton.Config 对象列表
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
        # Tier 2: Basic kernel heuristics
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
        # Tier 5: Small M + Large N special cases
        # =====================================================================
        # 32x128 with various warps
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # 32x64 with high warps
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        # 32x64 with low warps
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


def get_quant_autotune_configs():
    """
    获取 quant (per-row quantization) kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100), SM100(B200), SM120(5080)
    
    因为 M 是灵活可变的batchsize 此处相当于是搜索 R[M,N] = A[M,K] * W[N,K] 中的 [M,K]

    - BLOCK_K 是主要的块大小参数，控制每次循环处理的元素数
    - M 虽然不在 kernel 块参数中，但影响最佳 num_warps/num_stages
    - autotune key = ['M', 'K']
    
    BLOCK_K 选择原则：
    - 必须是 2 的幂次
    - 典型值：512, 1024, 2048, 4096, 8192
    - K=2560 → BLOCK_K=2048/4096 (1-2 次循环)
    - K=6912 → BLOCK_K=4096/8192 (1-2 次循环)
    
    num_warps 选择原则：
    - 小 M（1-64）：1-4 warps
    - 中 M（64-4096）：4-8 warps
    - 大 M（4096+）：8-32 warps
    
    Returns:
        triton.Config 对象列表
    """
    import triton
    
    return [
        # =====================================================================
        # Tier 1: 小 BLOCK_K (适合小 K 或需要低寄存器压力)
        # =====================================================================
        triton.Config({'BLOCK_K': 512}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 512}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=3),
        
        # =====================================================================
        # Tier 2: 中等 BLOCK_K (K <= 2048 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 1024}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 1024}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        
        # =====================================================================
        # Tier 3: 大 BLOCK_K (K=2560-4096 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=16, num_stages=3),
        
        # =====================================================================
        # Tier 4: 超大 BLOCK_K (K=4096-8192 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=32, num_stages=3),
        
        # =====================================================================
        # Tier 5: 极大 BLOCK_K (K > 6000 的选择，如 K=6912)
        # =====================================================================
        triton.Config({'BLOCK_K': 8192}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=3),
        
        # =====================================================================
        # Tier 6: 边界探索 - 小 M 特殊优化
        # =====================================================================
        # 小 M (1-16) 需要低 num_warps
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=4),
        
        # =====================================================================
        # Tier 7: 边界探索 - 大 M 特殊优化  
        # =====================================================================
        # 大 M (16384+) 需要高 num_warps
        triton.Config({'BLOCK_K': 2048}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=4),
    ]


def get_quant_or_dequant_autotune_configs():
    """
    获取 quant/dequant 通用的 Triton autotune 配置
    
    这是旧版 2D tiled kernel 的配置（BLOCK_M x BLOCK_N/BLOCK_K）
    对于新版 per-row kernel，请使用 get_quant_autotune_configs()
    
    Returns:
        triton.Config 对象列表（等同于 get_dequant_autotune_configs）
    """
    return get_dequant_autotune_configs()



# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # NVCC 架构标志
    'SUPPORTED_ARCHITECTURES',
    'get_nvcc_arch_flags',
    'get_current_arch_flag',
    # CUDA 编译
    'DEFAULT_CFLAGS',
    'DEFAULT_CUDA_CFLAGS',
    'should_rebuild',
    'clean_build_artifacts',
    'build_cuda_extension',
    # GEMM 链接库
    'CUBLASLT_LDFLAGS',
    'CUSPARSELT_LDFLAGS',
    'get_gemm_ldflags',
    # Triton 配置
    'get_dequant_autotune_configs',
    'get_quant_autotune_configs',
    'get_quant_or_dequant_autotune_configs',
]
