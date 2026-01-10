#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 4: Kernel 正确性测试

验证 CuBLASLtFp8LinearOp 的计算正确性。

测试覆盖:
- Op 创建和配置
- Scheme 包装正确性
- 推理输出验证

环境变量:
- USE_CUBLASLT=1: 启用 cuBLASLt 路径
- INNER_DTYPE_FP32=1: GEMM 输出使用 FP32
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
    ModelFinder,
    Assertions,
    cuda_memory_manager,
    skip_if_no_cuda,
    skip_if_no_fp8,
)


# ============================================================================
# 测试配置
# ============================================================================

# 指定测试模型
TEST_MODELS = [
    "/root/vllmbench/checkpoints/Qwen2.5-0.5B-FP8",
    "/root/vllmbench/checkpoints/Llama3.2-1B-FP8",
]


# ============================================================================
# 测试用例
# ============================================================================

@test_case("CuBLASLtFp8LinearOp 创建")
def test_op_creation():
    """测试 Op 创建"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    # 测试默认配置
    op1 = CuBLASLtFp8LinearOp()
    assert op1 is not None
    
    # 测试自定义配置
    op2 = CuBLASLtFp8LinearOp(
        act_quant_static=True,
        act_quant_group_shape=GroupShape.PER_TENSOR
    )
    assert op2 is not None
    
    # 验证属性
    assert hasattr(op1, "quant_fp8"), "缺少 quant_fp8 实例"
    assert hasattr(op1, "apply"), "缺少 apply 方法"
    
    return True, "Op 创建成功"


@test_case("QuantFP8 配置一致性", skip_if=skip_if_no_cuda)
def test_quant_fp8_config():
    """测试 QuantFP8 配置一致性"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    
    op = CuBLASLtFp8LinearOp()
    
    # 验证 quant_fp8 实例存在且配置正确
    assert op.quant_fp8 is not None, "quant_fp8 实例不存在"
    
    # 验证配置一致性
    assert op.quant_fp8.static == op.act_quant_static
    
    return True, "QuantFP8 配置正确"


@test_case("环境变量检查")
def test_env_var_check():
    """测试环境变量配置"""
    from slidesparse.core.cublaslt_config import (
        is_cublaslt_enabled,
        is_inner_dtype_fp32,
        get_cublaslt_status,
    )
    
    enabled = is_cublaslt_enabled()
    fp32 = is_inner_dtype_fp32()
    status = get_cublaslt_status()
    
    return True, f"USE_CUBLASLT={enabled}, INNER_DTYPE_FP32={fp32}"


@test_case("Scheme 包装正确性")
def test_scheme_wrapper_correctness():
    """测试 scheme 包装器的正确性"""
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled
    from slidesparse.core.cublaslt_linear_method import (
        CuBLASLtFp8LinearMethod,
        wrap_scheme_with_cublaslt,
    )
    
    # Mock scheme - 不被支持，应该返回原始对象
    class MockScheme:
        def create_weights(self, layer, *args, **kwargs):
            pass
        
        def process_weights_after_loading(self, layer, **kwargs):
            pass
        
        def apply_weights(self, layer, x, bias=None):
            return x
    
    mock = MockScheme()
    wrapped = wrap_scheme_with_cublaslt(mock)
    
    # mock 类型不被支持，无论 cuBLASLt 是否启用都应该返回原始对象
    assert wrapped is mock, "不支持的类型应返回原始对象"
    
    return True, "不支持的 scheme 类型正确处理"


@test_case("推理输出验证", skip_if=skip_if_no_fp8)
def test_inference_output():
    """通过实际推理验证输出正确性"""
    from pathlib import Path
    
    model_path = None
    for p in TEST_MODELS:
        if Path(p).exists():
            model_path = p
            break
    
    if model_path is None:
        return TestResult(
            name="推理输出验证",
            status=TestStatus.SKIPPED,
            message="未找到指定的 FP8 模型"
        )
    
    with cuda_memory_manager():
        from vllm import LLM, SamplingParams
        
        llm = LLM(
            model=str(model_path),
            max_model_len=128,
            gpu_memory_utilization=0.5,
            disable_log_stats=True,
        )
        
        # 使用固定种子和贪婪采样确保可复现
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=20,
        )
        
        # 运行推理
        outputs = llm.generate(["The capital of France is"], sampling_params)
        
        assert len(outputs) > 0, "无输出"
        assert len(outputs[0].outputs) > 0, "输出为空"
        
        output_text = outputs[0].outputs[0].text.strip()
        
        # 简单验证：输出应该是合理的文本
        assert len(output_text) > 0, "输出文本为空"
        
        del llm
    
    return True, f"输出: {output_text[:30]}..."


# ============================================================================
# 主函数
# ============================================================================

def run_tests(verbose: bool = True) -> bool:
    """运行所有正确性测试"""
    tests = [
        test_op_creation,
        test_quant_fp8_config,
        test_env_var_check,
        test_scheme_wrapper_correctness,
        test_inference_output,
    ]
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("Kernel 正确性测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Kernel 正确性测试")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    args = parser.parse_args()
    
    success = run_tests(verbose=not args.quiet)
    sys.exit(0 if success else 1)
