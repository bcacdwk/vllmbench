#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 1: cuBLASLt 模块导入测试

验证 slidesparse 外挂模块能否正确导入。
这是最基础的测试，确保模块结构正确。

运行方式:
    python3 slidesparse/test/test_cublaslt_import.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def test_import():
    """测试模块导入"""
    print("=" * 60)
    print("测试 1: cuBLASLt 模块导入测试")
    print("=" * 60)
    
    success = True
    
    # 测试 1.1: 导入 slidesparse 主模块
    print("\n[1.1] 导入 slidesparse 主模块...")
    try:
        import slidesparse
        print(f"    ✓ slidesparse 版本: {slidesparse.__version__}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 测试 1.2: 导入 core 模块
    print("\n[1.2] 导入 slidesparse.core 模块...")
    try:
        from slidesparse.core import (
            is_cublaslt_enabled,
            get_cublaslt_status,
            CuBLASLtFp8LinearMethod,
        )
        print(f"    ✓ is_cublaslt_enabled: {is_cublaslt_enabled}")
        print(f"    ✓ CuBLASLtFp8LinearMethod: {CuBLASLtFp8LinearMethod}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 测试 1.3: 导入配置函数
    print("\n[1.3] 导入配置函数...")
    try:
        from slidesparse.core.cublaslt_config import (
            is_cublaslt_enabled,
            get_cublaslt_status,
            VLLM_USE_CUBLASLT,
        )
        print(f"    ✓ VLLM_USE_CUBLASLT: {VLLM_USE_CUBLASLT}")
        print(f"    ✓ is_cublaslt_enabled(): {is_cublaslt_enabled()}")
        print(f"    ✓ get_cublaslt_status(): {get_cublaslt_status()}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 测试 1.4: 导入 linear method
    print("\n[1.4] 导入 linear method...")
    try:
        from slidesparse.core.cublaslt_linear_method import (
            CuBLASLtFp8LinearOp,
            wrap_scheme_with_cublaslt,
        )
        print(f"    ✓ CuBLASLtFp8LinearOp: {CuBLASLtFp8LinearOp}")
        print(f"    ✓ wrap_scheme_with_cublaslt: {wrap_scheme_with_cublaslt}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 测试 1.5: 导入 vllm 空壳转发文件
    print("\n[1.5] 导入 vllm 空壳转发文件...")
    try:
        from vllm.model_executor.layers.quantization.cublaslt import (
            is_cublaslt_enabled as vllm_is_cublaslt_enabled,
            wrap_scheme_with_cublaslt as vllm_wrap_scheme,
        )
        print(f"    ✓ vllm.cublaslt.is_cublaslt_enabled: {vllm_is_cublaslt_enabled}")
        print(f"    ✓ vllm.cublaslt.wrap_scheme_with_cublaslt: {vllm_wrap_scheme}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 总结
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有导入测试通过!")
    else:
        print("✗ 部分导入测试失败!")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
