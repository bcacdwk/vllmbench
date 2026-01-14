#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt Compress Extension Build Script

这是一个智能编译脚本，支持：
1. 自动检测 GPU 架构并生成对应的 .so 文件
2. 文件名遵循 SlideSparse 统一命名规范
3. 自动复用已编译的 .so（如果存在且比源文件新）
4. 编译后自动清理中间文件

文件名格式
==========
{prefix}_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.so

示例:
    cusparselt_compress_H100_cc90_py312_cu124_x86_64.so
    cusparselt_compress_RTX5080_cc120_py312_cu129_x86_64.so

使用方法
========
编译当前 GPU 架构的 .so：
    python3 build_compress.py build
    
强制重新编译：
    python3 build_compress.py build --force

查看帮助：
    python3 build_compress.py --help
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# 添加项目根目录到 path
_FILE_DIR = Path(__file__).parent.absolute()
_SLIDESPARSE_DIR = _FILE_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 使用顶层 utils 获取硬件信息和文件命名
from slidesparse.utils import (
    hw_info,
    build_filename,
    find_file,
    print_system_info,
)

# 使用本地编译工具
from build_utils import (
    build_cuda_extension_direct,
    clean_build_artifacts,
    CUSPARSELT_LDFLAGS,
)


# =============================================================================
# 配置
# =============================================================================

# 扩展前缀
EXTENSION_PREFIX = "cusparselt_compress"


def get_extension_name() -> str:
    """
    生成带版本和架构信息的扩展名（不含 .so 后缀）
    
    格式: cusparselt_compress_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}
    
    Example:
        cusparselt_compress_H100_cc90_py312_cu124_x86_64
    """
    return build_filename(EXTENSION_PREFIX, ext="")


def build_extension(
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """
    编译 cuSPARSELt Compress 扩展
    
    Args:
        force: 是否强制重新编译
        verbose: 是否显示详细输出
        
    Returns:
        编译生成的 .so 文件路径
    """
    # 路径配置
    source_dir = Path(__file__).parent.absolute()
    source_file = source_dir / "cusparselt_compress.cu"
    build_dir = source_dir / "build"
    
    # 获取扩展名
    ext_name = get_extension_name()
    
    if verbose:
        print("=" * 60)
        print("cuSPARSELt Compress Extension Builder")
        print("=" * 60)
        print(f"Extension: {ext_name}")
        print(f"Source: {source_file.name}")
        print(f"Build dir: {build_dir}")
        print("-" * 60)
        print(f"GPU: {hw_info.gpu_name} ({hw_info.gpu_full_name})")
        print(f"CC: {hw_info.cc_tag} ({hw_info.sm_code})")
        print(f"Python: {hw_info.python_tag}")
        print(f"CUDA: {hw_info.cuda_tag}")
        print(f"Arch: {hw_info.arch_tag}")
        print("=" * 60)
    
    # 检查已存在的 .so
    existing_so = find_file(
        EXTENSION_PREFIX,
        search_dir=build_dir,
        ext=".so",
    )
    
    if existing_so and not force:
        # 检查是否需要重新编译
        if source_file.stat().st_mtime <= existing_so.stat().st_mtime:
            if verbose:
                print(f"\n✓ Using existing: {existing_so.name}")
                print("  (use --force to rebuild)")
            return existing_so
        elif verbose:
            print(f"\n⚠ Source file changed, rebuilding...")
    
    # 编译（使用直接编译，不依赖 PyTorch）
    so_path = build_cuda_extension_direct(
        name=ext_name,
        source_file=source_file,
        build_dir=build_dir,
        extra_ldflags=CUSPARSELT_LDFLAGS,
        force=force,
        verbose=verbose,
    )
    
    return so_path


def main():
    parser = argparse.ArgumentParser(
        description="Build cuSPARSELt Compress extension for SlideSparse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 build_compress.py build          # 编译扩展
  python3 build_compress.py build --force  # 强制重新编译
  python3 build_compress.py info           # 显示环境信息
  python3 build_compress.py clean          # 清理构建目录

说明:
  此扩展用于将 2:4 稀疏权重压缩为 cuSPARSELt 硬件加速格式。
  支持 INT8 权重，布局为 TN-CC 格式（W转置、A不转置、列主序）。
        """
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
        build_extension(
            force=args.force,
            verbose=not args.quiet,
        )
    
    elif args.command == "info":
        print("=" * 60)
        print("cuSPARSELt Compress Extension Info")
        print("=" * 60)
        print(f"Extension name: {get_extension_name()}.so")
        print(f"Input type: INT8 (2:4 sparse)")
        print(f"Layout: TN-CC (W^T * A, all column-major)")
        print("-" * 60)
        print_system_info()
        print("-" * 60)
        
        build_dir = Path(__file__).parent / "build"
        existing = find_file(
            EXTENSION_PREFIX,
            search_dir=build_dir,
            ext=".so",
        )
        if existing:
            print(f"Existing .so: {existing.name}")
        else:
            print("No existing .so found")
        print("=" * 60)
    
    elif args.command == "clean":
        build_dir = Path(__file__).parent / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print(f"✓ Cleaned: {build_dir}")
        else:
            print("Nothing to clean")


if __name__ == "__main__":
    main()
