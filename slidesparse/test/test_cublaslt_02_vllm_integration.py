#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 2: vLLM 集成测试

验证 cuBLASLt 后端与 vLLM 的集成是否正确。

测试覆盖:
- CompressedTensors 量化配置加载
- Scheme 包装器功能
- CuBLASLtFp8LinearOp 实例化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slidesparse.test.test_base import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
    skip_if_no_cuda,
)


# ============================================================================
# 测试用例
# ============================================================================

@test_case("导入 CompressedTensors 配置")
def test_import_compressed_tensors():
    """测试 CompressedTensors 量化配置导入"""
    # 注意: vLLM 的模块结构是 compressed_tensors/compressed_tensors.py
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
        CompressedTensorsLinearMethod,
    )
    
    assert CompressedTensorsConfig is not None
    assert CompressedTensorsLinearMethod is not None
    
    return True, "CompressedTensors 模块导入成功"


@test_case("检查 cuBLASLt 状态")
def test_cublaslt_status():
    """测试 cuBLASLt 状态检查"""
    from slidesparse.core.cublaslt_config import (
        is_cublaslt_enabled,
        get_cublaslt_status,
    )
    
    enabled = is_cublaslt_enabled()
    status = get_cublaslt_status()
    
    return True, status


@test_case("创建 CuBLASLtFp8LinearOp 实例")
def test_create_linear_op():
    """测试 CuBLASLtFp8LinearOp 实例创建"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    # 测试不同配置
    configs = [
        {"act_quant_static": False, "act_quant_group_shape": GroupShape.PER_TOKEN},
        {"act_quant_static": True, "act_quant_group_shape": GroupShape.PER_TENSOR},
    ]
    
    for config in configs:
        op = CuBLASLtFp8LinearOp(**config)
        assert hasattr(op, "apply"), "缺少 apply 方法"
        assert hasattr(op, "_fp8_linear_op"), "缺少内部 Fp8LinearOp"
    
    return True, f"测试了 {len(configs)} 种配置"


@test_case("测试 wrap_scheme_with_cublaslt 函数")
def test_wrap_scheme_function():
    """测试 scheme 包装函数"""
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled
    from slidesparse.core.cublaslt_linear_method import (
        CuBLASLtFp8LinearMethod,
        wrap_scheme_with_cublaslt,
    )
    
    # 测试不支持的类型应该返回原始对象
    class MockScheme:
        pass
    
    mock = MockScheme()
    wrapped = wrap_scheme_with_cublaslt(mock)
    
    # mock 类型不被支持，应该返回原始对象
    assert wrapped is mock, "不支持的类型应返回原始对象"
    
    return True, "不支持的 scheme 类型正确处理"


@test_case("测试 CuBLASLtFp8LinearMethod 类结构")
def test_linear_method_structure():
    """测试 CuBLASLtFp8LinearMethod 的类结构"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearMethod
    
    # 检查类有必要的方法
    assert hasattr(CuBLASLtFp8LinearMethod, "__init__")
    assert hasattr(CuBLASLtFp8LinearMethod, "create_weights")
    assert hasattr(CuBLASLtFp8LinearMethod, "process_weights_after_loading")
    assert hasattr(CuBLASLtFp8LinearMethod, "apply_weights")
    
    return True, "CuBLASLtFp8LinearMethod 类结构正确"


@test_case("验证 CompressedTensorsConfig.get_scheme 存在")
def test_compressed_tensors_get_scheme():
    """验证 CompressedTensorsConfig 有 get_scheme 方法"""
    import inspect
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    
    # 检查 get_scheme 方法是否存在
    assert hasattr(CompressedTensorsConfig, "get_scheme"), "缺少 get_scheme 方法"
    
    # 获取源代码检查是否包含 cuBLASLt 集成
    try:
        source = inspect.getsource(CompressedTensorsConfig.get_scheme)
        
        # 检查是否有 cuBLASLt 相关代码
        has_cublaslt_code = "cublaslt" in source.lower() or "wrap_scheme" in source
        
        if has_cublaslt_code:
            return True, "cuBLASLt 集成代码存在于 get_scheme"
        else:
            return TestResult(
                name="验证 CompressedTensorsConfig.get_scheme 存在",
                status=TestStatus.WARNING,
                message="get_scheme 存在但未检测到 cuBLASLt 集成代码"
            )
    except Exception as e:
        return True, f"get_scheme 方法存在 (无法获取源码: {e})"


@test_case("验证 FP8 scheme 类可用")
def test_fp8_scheme_available():
    """验证 FP8 相关的 scheme 类可以导入"""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
    )
    
    # 检查类的必要方法
    assert hasattr(CompressedTensorsW8A8Fp8, "create_weights")
    assert hasattr(CompressedTensorsW8A8Fp8, "process_weights_after_loading")
    assert hasattr(CompressedTensorsW8A8Fp8, "apply_weights")
    
    return True, "FP8 scheme 类可用"


# ============================================================================
# 主函数
# ============================================================================

def run_tests(verbose: bool = True) -> bool:
    """运行所有 vLLM 集成测试"""
    tests = [
        test_import_compressed_tensors,
        test_cublaslt_status,
        test_create_linear_op,
        test_wrap_scheme_function,
        test_linear_method_structure,
        test_compressed_tensors_get_scheme,
        test_fp8_scheme_available,
    ]
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("vLLM 集成测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM 集成测试")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()
    
    success = run_tests(verbose=not args.quiet)
    sys.exit(0 if success else 1)
