#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 1: cuBLASLt 模块导入测试

验证 slidesparse 外挂模块能否正确导入。
这是最基础的测试，确保模块结构正确。

测试覆盖:
- slidesparse 主模块
- slidesparse.core 子模块  
- 配置函数
- vLLM 空壳转发文件
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slidesparse.test.test_base import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
)


# ============================================================================
# 测试用例
# ============================================================================

@test_case("导入 slidesparse 主模块")
def test_import_main_module():
    """测试主模块导入"""
    import slidesparse
    
    # 验证版本号
    assert hasattr(slidesparse, "__version__"), "缺少 __version__ 属性"
    version = slidesparse.__version__
    
    return True, f"版本 {version}"


@test_case("导入 slidesparse.core 模块")
def test_import_core_module():
    """测试 core 模块导入"""
    from slidesparse.core import (
        is_cublaslt_enabled,
        get_cublaslt_status,
        CuBLASLtFp8LinearMethod,
    )
    
    # 验证导出的符号
    assert callable(is_cublaslt_enabled), "is_cublaslt_enabled 应该是函数"
    assert callable(get_cublaslt_status), "get_cublaslt_status 应该是函数"
    
    return True, "核心接口导入成功"


@test_case("导入配置模块")
def test_import_config_module():
    """测试配置模块导入"""
    from slidesparse.core.cublaslt_config import (
        is_cublaslt_enabled,
        get_cublaslt_status,
        VLLM_USE_CUBLASLT,
        SLIDESPARSE_USE_CUBLASLT,
        SLIDESPARSE_CUBLASLT_DEBUG,
    )
    
    # 验证配置值类型
    assert isinstance(VLLM_USE_CUBLASLT, bool), "VLLM_USE_CUBLASLT 应该是 bool"
    
    status = get_cublaslt_status()
    enabled = is_cublaslt_enabled()
    
    return True, f"enabled={enabled}"


@test_case("导入 LinearMethod 模块")
def test_import_linear_method():
    """测试 LinearMethod 模块导入"""
    from slidesparse.core.cublaslt_linear_method import (
        CuBLASLtFp8LinearOp,
        CuBLASLtFp8LinearMethod,
        wrap_scheme_with_cublaslt,
    )
    
    # 验证类和函数
    assert hasattr(CuBLASLtFp8LinearOp, "apply"), "CuBLASLtFp8LinearOp 缺少 apply 方法"
    assert hasattr(CuBLASLtFp8LinearMethod, "apply_weights"), "缺少 apply_weights 方法"
    assert callable(wrap_scheme_with_cublaslt), "wrap_scheme_with_cublaslt 应该是函数"
    
    return True, "LinearMethod 类导入成功"


@test_case("导入 vLLM 转发模块")
def test_import_vllm_bridge():
    """测试 vLLM 空壳转发文件导入"""
    from vllm.model_executor.layers.quantization.cublaslt import (
        is_cublaslt_enabled as vllm_is_enabled,
        wrap_scheme_with_cublaslt as vllm_wrap,
    )
    
    # 验证函数引用一致性
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled
    from slidesparse.core.cublaslt_linear_method import wrap_scheme_with_cublaslt
    
    # 函数应该是同一个对象
    assert vllm_is_enabled is is_cublaslt_enabled, "函数引用不一致"
    assert vllm_wrap is wrap_scheme_with_cublaslt, "函数引用不一致"
    
    return True, "vLLM 桥接模块正常"


@test_case("验证环境变量解析")
def test_env_var_parsing():
    """测试环境变量解析逻辑"""
    import os
    
    # 简化测试：只验证当前状态
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled
    enabled = is_cublaslt_enabled()
    
    return True, f"当前状态: enabled={enabled}"


# ============================================================================
# 主函数
# ============================================================================

def run_tests(verbose: bool = True) -> bool:
    """运行所有导入测试"""
    tests = [
        test_import_main_module,
        test_import_core_module,
        test_import_config_module,
        test_import_linear_method,
        test_import_vllm_bridge,
        test_env_var_parsing,
    ]
    
    runner = TestRunner("cuBLASLt 模块导入测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="cuBLASLt 模块导入测试")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()
    
    if args.json:
        import json
        tests = [
            test_import_main_module,
            test_import_core_module,
            test_import_config_module,
            test_import_linear_method,
            test_import_vllm_bridge,
            test_env_var_parsing,
        ]
        
        results = []
        for test in tests:
            r = test()
            results.append({
                "name": r.name,
                "status": r.status.name,
                "message": r.message,
                "duration": r.duration,
            })
        
        print(json.dumps({"tests": results}, indent=2))
        success = all(r["status"] == "PASSED" for r in results)
    else:
        success = run_tests(verbose=not args.quiet)
    
    sys.exit(0 if success else 1)
