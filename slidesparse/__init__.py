# SPDX-License-Identifier: Apache-2.0
# SlideSparse: Sparse Acceleration for LLM Inference
"""
SlideSparse 外挂模块 - vLLM 集成

此模块实现 SlideSparse 稀疏加速方法与 vLLM 的集成，包括:
- cuBLASLt Dense 基线 (Phase 3)
- cuSPARSELt Sparse 加速 (Phase 6)

目录结构:
- core/: 核心逻辑 (LinearMethod, Config)
- kernels/: Kernel 实现 (Triton, cuBLASLt, cuSPARSELt)
- offline/: 离线工具链 (prune, slide, compress)
- test/: 测试脚本
- utils.py: 统一工具库（硬件检测、文件命名、模块加载）
"""

__version__ = "0.1.0"

# 导出统一工具库的核心功能
from .utils import (
    # 硬件信息
    hw_info,
    HardwareInfo,
    
    # 文件名构建
    build_filename,
    build_stem,
    build_dir_name,
    
    # 文件查找
    find_file,
    find_files,
    find_dir,
    
    # 模块加载
    load_module,
    
    # 数据保存/加载
    save_json,
    load_json,
    save_csv,
    
    # 目录管理
    ensure_result_dir,
    
    # 便捷函数
    get_gpu_name,
    get_gpu_cc,
    get_python_version_tag,
    get_cuda_ver,
    get_arch_tag,
    get_sm_code,
    normalize_dtype,
    print_system_info,
)
