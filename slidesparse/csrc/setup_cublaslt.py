#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt GEMM Extension Setup Script

这是一个智能编译脚本，支持：
1. 自动检测 GPU 架构并生成对应的 .so 文件
2. 文件名包含 Python 版本、架构、GPU CC 信息
3. 自动复用已编译的 .so（如果存在且比源文件新）
4. 编译后自动清理中间文件

支持的 GPU 架构：
- SM 80: Ampere (A100, A10, A30)
- SM 86: Ampere (RTX 30xx)
- SM 89: Ada Lovelace (RTX 40xx)
- SM 90: Hopper (H100, H200)
- SM 100: Blackwell (B100, B200)
- SM 120: Blackwell (RTX 50xx, GB10)

使用方法：
=========
编译当前 GPU 架构的 .so：
    cd /root/vllmbench/slidesparse/csrc
    python setup_cublaslt.py build
    
强制重新编译：
    python setup_cublaslt.py build --force

查看帮助：
    python setup_cublaslt.py --help
"""

import os
import sys
import glob
import shutil
import platform
import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def get_python_version_tag() -> str:
    """获取 Python 版本标签，如 py312"""
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"py{major}{minor}"


def get_arch_tag() -> str:
    """获取系统架构标签，如 x86_64 或 aarch64"""
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        return "x86_64"
    elif machine in ("aarch64", "arm64"):
        return "aarch64"
    else:
        return machine.lower()


def get_gpu_cc() -> str:
    """获取当前 GPU 的 Compute Capability，如 cc90"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Cannot determine GPU CC.")
    
    prop = torch.cuda.get_device_properties(0)
    return f"cc{prop.major}{prop.minor}"


def get_gpu_short_name() -> str:
    """获取 GPU 简称，如 H100, A100"""
    prop = torch.cuda.get_device_properties(0)
    full_name = prop.name
    
    # 移除 "NVIDIA " 前缀
    if "NVIDIA " in full_name:
        full_name = full_name.replace("NVIDIA ", "")
    
    # 提取型号
    for sep in [" ", "-"]:
        idx = full_name.find(sep)
        if idx > 0:
            return full_name[:idx]
    
    return full_name.replace(" ", "_")


def get_extension_name() -> str:
    """
    生成带版本和架构信息的扩展名
    
    格式: slidesparse_cublaslt_py312_x86_64_cc90
    """
    py_tag = get_python_version_tag()
    arch_tag = get_arch_tag()
    cc_tag = get_gpu_cc()
    
    return f"slidesparse_cublaslt_{py_tag}_{arch_tag}_{cc_tag}"


def get_nvcc_arch_flags() -> list:
    """
    生成 nvcc 架构编译选项
    
    支持从 SM 80 (Ampere) 到 SM 120 (Blackwell)
    """
    # 支持的架构列表
    # 注：SM 100-119 之间的架构可能不存在，但加上不会出错
    architectures = [
        ("80", "sm_80"),   # Ampere (A100)
        ("86", "sm_86"),   # Ampere (RTX 30xx)
        ("89", "sm_89"),   # Ada Lovelace (RTX 40xx)
        ("90", "sm_90"),   # Hopper (H100)
        ("100", "sm_100"), # Blackwell (B100)
        ("120", "sm_120"), # Blackwell (RTX 50xx, GB10)
    ]
    
    flags = []
    for compute, sm in architectures:
        flags.append(f"-gencode=arch=compute_{compute},code={sm}")
    
    return flags


def find_existing_so(build_dir: Path, ext_name: str) -> Path | None:
    """
    查找已存在的 .so 文件
    
    返回匹配的 .so 路径，如果不存在返回 None
    """
    # 匹配模式：slidesparse_cublaslt_py312_x86_64_cc90*.so
    pattern = f"{ext_name}*.so"
    matches = list(build_dir.glob(pattern))
    
    if matches:
        return matches[0]
    return None


def should_rebuild(so_path: Path, source_path: Path) -> bool:
    """
    判断是否需要重新编译
    
    如果 .so 不存在或比源文件旧，返回 True
    """
    if not so_path.exists():
        return True
    
    so_mtime = so_path.stat().st_mtime
    src_mtime = source_path.stat().st_mtime
    
    return src_mtime > so_mtime


def clean_build_artifacts(build_dir: Path, ext_name: str):
    """
    清理编译中间文件，只保留 .so
    
    删除 build 目录下除了 .so 文件以外的所有内容
    """
    for item in build_dir.iterdir():
        # 保留 .so 文件
        if item.suffix == ".so":
            continue
        
        # 删除其他文件和目录
        if item.is_dir():
            shutil.rmtree(item)
            print(f"  Cleaned dir: {item}")
        else:
            item.unlink()
            print(f"  Cleaned file: {item.name}")


def build_extension(force: bool = False, verbose: bool = True):
    """
    编译 cuBLASLt 扩展
    
    Args:
        force: 是否强制重新编译
        verbose: 是否显示详细输出
    """
    # 路径配置
    csrc_dir = Path(__file__).parent.absolute()
    source_file = csrc_dir / "cublaslt_gemm.cu"
    build_dir = csrc_dir / "build"
    
    # 确保 build 目录存在
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取扩展名
    ext_name = get_extension_name()
    
    if verbose:
        print(f"=" * 60)
        print(f"cuBLASLt Extension Builder")
        print(f"=" * 60)
        print(f"Extension name: {ext_name}")
        print(f"Source file: {source_file}")
        print(f"Build directory: {build_dir}")
        print(f"GPU: {get_gpu_short_name()} ({get_gpu_cc()})")
        print(f"Python: {get_python_version_tag()}")
        print(f"Arch: {get_arch_tag()}")
    
    # 检查是否需要重新编译
    existing_so = find_existing_so(build_dir, ext_name)
    
    if existing_so and not force:
        if not should_rebuild(existing_so, source_file):
            if verbose:
                print(f"\n✓ Using existing .so: {existing_so.name}")
                print(f"  (use --force to rebuild)")
            return existing_so
        else:
            if verbose:
                print(f"\n⚠ Source file changed, rebuilding...")
    
    if verbose:
        print(f"\n🔨 Building extension...")
    
    # CUDA 路径
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    
    # 编译选项
    nvcc_arch_flags = get_nvcc_arch_flags()
    
    extra_cflags = ['-O3', '-std=c++17']
    extra_cuda_cflags = [
        '-O3',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
    ] + nvcc_arch_flags
    
    extra_ldflags = [
        '-lcublasLt',
        '-lcublas',
        '-lcuda',
    ]
    
    # 使用 torch.utils.cpp_extension.load 进行即时编译
    try:
        ext = load(
            name=ext_name,
            sources=[str(source_file)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=[os.path.join(cuda_home, 'include')],
            build_directory=str(build_dir),
            verbose=verbose,
        )
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        raise
    
    # 查找生成的 .so 文件
    new_so = find_existing_so(build_dir, ext_name)
    
    if new_so:
        if verbose:
            print(f"\n✓ Build successful: {new_so.name}")
        
        # 清理中间文件
        if verbose:
            print(f"\n🧹 Cleaning build artifacts...")
        clean_build_artifacts(build_dir, ext_name)
        
        return new_so
    else:
        raise RuntimeError("Build completed but .so file not found")


def main():
    parser = argparse.ArgumentParser(
        description="Build cuBLASLt GEMM extension for SlideSparse"
    )
    parser.add_argument(
        "command",
        choices=["build", "info", "clean"],
        help="Command to execute"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force rebuild even if .so exists"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_extension(force=args.force, verbose=not args.quiet)
    
    elif args.command == "info":
        print(f"Extension name: {get_extension_name()}")
        print(f"GPU: {get_gpu_short_name()} ({get_gpu_cc()})")
        print(f"Python: {get_python_version_tag()}")
        print(f"Arch: {get_arch_tag()}")
        
        build_dir = Path(__file__).parent / "build"
        existing = find_existing_so(build_dir, get_extension_name())
        if existing:
            print(f"Existing .so: {existing}")
        else:
            print(f"No existing .so found")
    
    elif args.command == "clean":
        build_dir = Path(__file__).parent / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print(f"Cleaned: {build_dir}")
        else:
            print(f"Nothing to clean")


if __name__ == "__main__":
    main()
