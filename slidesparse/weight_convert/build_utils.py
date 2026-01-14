#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Weight Convert 编译工具库

本模块提供 CUDA 扩展编译相关的通用工具函数。
复制自 slidesparse/csrc/utils.py，用于 weight_convert 目录的独立编译。

注意：文件名、硬件信息等功能请使用顶层 slidesparse.utils 模块。

主要功能
========
1. NVCC 架构标志生成
2. CUDA 扩展编译器
3. 编译产物清理
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Optional

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
    """
    if keep_extensions is None:
        keep_extensions = ['.so', '.py']
    
    if not build_dir.exists():
        return
    
    for item in build_dir.iterdir():
        if item.suffix in keep_extensions:
            continue
        
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
    
    Args:
        name: 扩展名称（不含 .so 后缀）
        source_file: 源文件路径 (.cu 或 .cpp)
        build_dir: 构建目录
        extra_cflags: 额外的 C++ 编译标志
        extra_cuda_cflags: 额外的 CUDA 编译标志
        extra_ldflags: 额外的链接标志
        extra_include_paths: 额外的头文件搜索路径
        force: 是否强制重新编译
        verbose: 是否显示详细输出
        clean_after_build: 编译后是否清理中间文件
        
    Returns:
        编译生成的 .so 文件路径
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


def build_cuda_extension_direct(
    name: str,
    source_file: Path,
    build_dir: Path,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    直接使用 nvcc 编译 CUDA 扩展（不依赖 PyTorch）
    
    生成一个纯 C 库，可以通过 ctypes 加载。
    
    Args:
        name: 扩展名称（不含 .so 后缀）
        source_file: 源文件路径 (.cu)
        build_dir: 构建目录
        extra_cuda_cflags: 额外的 CUDA 编译标志
        extra_ldflags: 额外的链接标志
        extra_include_paths: 额外的头文件搜索路径
        force: 是否强制重新编译
        verbose: 是否显示详细输出
        
    Returns:
        编译生成的 .so 文件路径
    """
    import subprocess
    
    # 验证源文件
    if not source_file.exists():
        raise FileNotFoundError(f"源文件不存在: {source_file}")
    
    # 确保 build 目录存在
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 输出文件
    so_path = build_dir / f"{name}.so"
    
    # 检查是否需要重新编译
    if so_path.exists() and not force:
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
    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    
    # 构建编译命令
    cmd = [nvcc]
    
    # 编译选项
    cmd.extend(['-std=c++17', '-O3', '-Xcompiler', '-fPIC', '--shared'])
    
    # 架构标志
    cmd.extend(get_nvcc_arch_flags())
    
    # 额外的 CUDA 标志
    if extra_cuda_cflags:
        cmd.extend(extra_cuda_cflags)
    
    # 头文件路径
    cmd.extend(['-I', os.path.join(cuda_home, 'include')])
    if extra_include_paths:
        for inc in extra_include_paths:
            cmd.extend(['-I', inc])
    
    # 源文件
    cmd.append(str(source_file))
    
    # 链接标志
    if extra_ldflags:
        cmd.extend(extra_ldflags)
    
    # 输出
    cmd.extend(['-o', str(so_path)])
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    # 执行编译
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        raise RuntimeError(f"编译失败:\n{error_msg}")
    
    if not so_path.exists():
        raise RuntimeError(f"编译完成但未找到 .so 文件: {so_path}")
    
    if verbose:
        print(f"✓ Built: {so_path.name}")
    
    return so_path


# =============================================================================
# 特定扩展的链接库配置
# =============================================================================

# 库搜索路径
_LIB_PATH = '/usr/lib/x86_64-linux-gnu'

# cuBLASLt 扩展所需的链接库
CUBLASLT_LDFLAGS = [f'-L{_LIB_PATH}', '-lcublasLt', '-lcublas', '-lcuda']

# cuSPARSELt 扩展所需的链接库  
CUSPARSELT_LDFLAGS = [f'-L{_LIB_PATH}', '-lcusparseLt', '-lcusparse', '-lcuda']


def get_gemm_ldflags(backend: str) -> List[str]:
    """
    获取 GEMM 后端所需的链接库标志
    
    Args:
        backend: 后端名称 ("cublaslt" 或 "cusparselt")
    """
    if backend.lower() == "cublaslt":
        return CUBLASLT_LDFLAGS.copy()
    elif backend.lower() == "cusparselt":
        return CUSPARSELT_LDFLAGS.copy()
    else:
        raise ValueError(f"未知的 GEMM 后端: {backend}")


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
    'build_cuda_extension_direct',
    # GEMM 链接库
    'CUBLASLT_LDFLAGS',
    'CUSPARSELT_LDFLAGS',
    'get_gemm_ldflags',
]
