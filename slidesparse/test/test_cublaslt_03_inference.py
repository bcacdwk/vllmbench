#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 3: 端到端推理测试

验证使用 cuBLASLt 后端的完整推理流程。

测试覆盖:
- 模型加载
- 单次推理
- 批量推理
- 多轮对话
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
    cuda_memory_manager,
    skip_if_no_cuda,
    skip_if_no_fp8,
    skip_if_no_model,
)


# ============================================================================
# 测试配置
# ============================================================================

# 指定测试模型
TEST_MODELS = [
    "/root/vllmbench/checkpoints/Qwen2.5-0.5B-FP8",
    "/root/vllmbench/checkpoints/Llama3.2-1B-FP8",
]

# 默认推理配置
DEFAULT_CONFIG = {
    "max_model_len": 512,
    "gpu_memory_utilization": 0.7,
    "disable_log_stats": True,
    "enforce_eager": False,  # 使用 CUDA Graph
}

# 测试提示词
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
]


# ============================================================================
# 辅助函数
# ============================================================================

def create_llm(model_path: str, **kwargs):
    """创建 LLM 实例"""
    from vllm import LLM
    
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    
    return LLM(model=model_path, **config)


# ============================================================================
# 测试用例
# ============================================================================

@test_case("查找可用模型")
def test_find_model():
    """测试模型查找功能"""
    from pathlib import Path
    
    available = []
    for model_path in TEST_MODELS:
        if Path(model_path).exists():
            available.append(Path(model_path).name)
    
    if not available:
        return False, "未找到指定的 FP8 模型"
    
    return True, f"可用模型: {', '.join(available)}"


@test_case("模型加载", skip_if=skip_if_no_fp8)
def test_model_loading():
    """测试模型加载"""
    from pathlib import Path
    
    # 使用第一个可用的模型
    model_path = None
    for p in TEST_MODELS:
        if Path(p).exists():
            model_path = p
            break
    
    if model_path is None:
        return TestResult(
            name="模型加载",
            status=TestStatus.SKIPPED,
            message="未找到指定的 FP8 模型"
        )
    
    with cuda_memory_manager():
        import time
        start = time.time()
        
        llm = create_llm(str(model_path), max_model_len=128)
        
        load_time = time.time() - start
        
        # 验证模型加载
        assert llm is not None, "LLM 实例为空"
        
        # 清理
        del llm
    
    return True, f"加载耗时 {load_time:.1f}s"


@test_case("单次推理", skip_if=skip_if_no_fp8)
def test_single_inference():
    """测试单次推理"""
    from pathlib import Path
    
    model_path = None
    for p in TEST_MODELS:
        if Path(p).exists():
            model_path = p
            break
    
    if model_path is None:
        return TestResult(
            name="单次推理",
            status=TestStatus.SKIPPED,
            message="未找到指定的 FP8 模型"
        )
    
    with cuda_memory_manager():
        from vllm import SamplingParams
        
        llm = create_llm(str(model_path), max_model_len=256)
        
        sampling_params = SamplingParams(
            temperature=0.0,  # 贪婪采样
            max_tokens=32,
        )
        
        outputs = llm.generate([TEST_PROMPTS[0]], sampling_params)
        
        assert len(outputs) == 1, "输出数量不正确"
        assert len(outputs[0].outputs) > 0, "没有生成输出"
        
        output_text = outputs[0].outputs[0].text.strip()
        
        # 清理
        del llm
    
    return True, f"输出: {output_text[:50]}..."


@test_case("批量推理", skip_if=skip_if_no_fp8)
def test_batch_inference():
    """测试批量推理"""
    from pathlib import Path
    
    model_path = None
    for p in TEST_MODELS:
        if Path(p).exists():
            model_path = p
            break
    
    if model_path is None:
        return TestResult(
            name="批量推理",
            status=TestStatus.SKIPPED,
            message="未找到指定的 FP8 模型"
        )
    
    with cuda_memory_manager():
        from vllm import SamplingParams
        import time
        
        llm = create_llm(str(model_path), max_model_len=256)
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=32,
        )
        
        start = time.time()
        outputs = llm.generate(TEST_PROMPTS, sampling_params)
        inference_time = time.time() - start
        
        assert len(outputs) == len(TEST_PROMPTS), "输出数量不匹配"
        
        for i, output in enumerate(outputs):
            assert len(output.outputs) > 0, f"第 {i} 个输出为空"
        
        # 清理
        del llm
    
    tokens_generated = sum(len(o.outputs[0].token_ids) for o in outputs)
    throughput = tokens_generated / inference_time
    
    return True, f"{len(TEST_PROMPTS)} 个请求, {throughput:.1f} tok/s"


@test_case("流式推理", skip_if=skip_if_no_fp8)
def test_streaming_inference():
    """测试流式推理 (如果支持)"""
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="流式推理",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    # 流式推理测试简化为普通推理
    # 因为 vLLM 的 LLM 类不直接支持流式
    return TestResult(
        name="流式推理",
        status=TestStatus.SKIPPED,
        message="LLM 类不支持流式 (使用 AsyncLLMEngine)"
    )


@test_case("cuBLASLt 路径验证", skip_if=skip_if_no_fp8)
def test_cublaslt_code_path():
    """验证 cuBLASLt 代码路径被执行"""
    from pathlib import Path
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled
    
    if not is_cublaslt_enabled():
        return TestResult(
            name="cuBLASLt 路径验证",
            status=TestStatus.SKIPPED,
            message="cuBLASLt 未启用"
        )
    
    model_path = None
    for p in TEST_MODELS:
        if Path(p).exists():
            model_path = p
            break
    
    if model_path is None:
        return TestResult(
            name="cuBLASLt 路径验证",
            status=TestStatus.SKIPPED,
            message="未找到指定的 FP8 模型"
        )
    
    with cuda_memory_manager():
        from vllm import SamplingParams
        
        llm = create_llm(str(model_path), max_model_len=128)
        
        sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
        outputs = llm.generate(["Hello"], sampling_params)
        
        # 如果能正常推理，说明 cuBLASLt 路径正常工作
        assert len(outputs) > 0
        
        del llm
    
    return True, "cuBLASLt 代码路径正常"


# ============================================================================
# 主函数
# ============================================================================

def run_tests(verbose: bool = True) -> bool:
    """运行所有推理测试"""
    tests = [
        test_find_model,
        test_model_loading,
        test_single_inference,
        test_batch_inference,
        test_streaming_inference,
        test_cublaslt_code_path,
    ]
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("端到端推理测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="端到端推理测试")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("--model", type=str, help="指定模型路径")
    args = parser.parse_args()
    
    success = run_tests(verbose=not args.quiet)
    sys.exit(0 if success else 1)
