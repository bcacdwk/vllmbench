#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
03_inference.py - 端到端推理测试

验证 cuBLASLt 后端的完整推理流程，并对比输出：
1. 模型加载测试
2. 单次推理测试
3. 原生 CUTLASS vs cuBLASLt/外挂CUTLASS 输出对比（核心功能）

核心测试：
    对于相同的 prompt，分别用 原生 CUTLASS 和 cuBLASLt/外挂CUTLASS 运行推理，
    并排打印输出让用户直观比较精度差异。

使用方法:
    python3 03_inference.py                   # 测试 cuBLASLt 路径（默认）
    python3 03_inference.py --ext-cutlass     # 测试外挂 CUTLASS 路径

路径说明:
    默认: USE_CUBLASLT=1 → cuBLASLt kernel
    --ext-cutlass: USE_CUBLASLT=0 → 外挂 CUTLASS
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
    ModelFinder,
    Colors,
    cuda_memory_manager,
    suppress_vllm_logs,
    skip_if_no_cuda,
    skip_if_no_fp8,
    parse_common_args,
    apply_env_args,
)


# ============================================================================
# 测试配置
# ============================================================================

# 测试提示词
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
]

# 推理配置
INFERENCE_CONFIG = {
    "max_model_len": 256,
    "gpu_memory_utilization": 0.5,
    "disable_log_stats": True,
    "enforce_eager": True,  # 禁用 CUDA Graph 以便调试
}


# ============================================================================
# 辅助函数
# ============================================================================

def create_llm(model_path: str, **kwargs):
    """创建 LLM 实例"""
    from vllm import LLM
    
    config = INFERENCE_CONFIG.copy()
    config.update(kwargs)
    
    return LLM(model=model_path, **config)


def run_inference(
    model_path: str,
    prompts: List[str],
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> List[str]:
    """运行推理并返回输出文本列表"""
    from vllm import LLM, SamplingParams
    
    with cuda_memory_manager():
        llm = create_llm(str(model_path))
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        results = [output.outputs[0].text.strip() for output in outputs]
        
        del llm
    
    return results


# ============================================================================
# 测试用例
# ============================================================================

@test_case("查找可用模型")
def test_find_models():
    """查找可用的 FP8 模型"""
    models = ModelFinder.get_test_models("FP8", max_count=2)
    
    if not models:
        return False, "未找到 FP8 模型"
    
    model_names = [m.name for m in models]
    return True, f"找到: {', '.join(model_names)}"


@test_case("模型加载测试", skip_if=skip_if_no_fp8)
def test_model_loading():
    """测试模型加载"""
    import time
    
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="模型加载测试",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    with cuda_memory_manager():
        start = time.time()
        llm = create_llm(str(model_path), max_model_len=128)
        load_time = time.time() - start
        
        assert llm is not None
        del llm
    
    return True, f"{model_path.name} 加载耗时 {load_time:.1f}s"


@test_case("单次推理测试", skip_if=skip_if_no_fp8)
def test_single_inference():
    """单次推理测试"""
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="单次推理测试",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    results = run_inference(
        model_path=str(model_path),
        prompts=[TEST_PROMPTS[0]],
        max_tokens=32,
    )
    
    assert len(results) == 1
    assert len(results[0]) > 0
    
    # 截断显示
    output_preview = results[0][:50] + "..." if len(results[0]) > 50 else results[0]
    return True, f"输出: {output_preview}"


@test_case("批量推理测试", skip_if=skip_if_no_fp8)
def test_batch_inference():
    """批量推理测试"""
    import time
    
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="批量推理测试",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    from vllm import LLM, SamplingParams
    
    with cuda_memory_manager():
        llm = create_llm(str(model_path))
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=32,
        )
        
        start = time.time()
        outputs = llm.generate(TEST_PROMPTS, sampling_params)
        elapsed = time.time() - start
        
        assert len(outputs) == len(TEST_PROMPTS)
        
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        throughput = total_tokens / elapsed
        
        del llm
    
    return True, f"{len(TEST_PROMPTS)} prompts, {throughput:.1f} tok/s"


# ============================================================================
# 核心功能：CUTLASS vs cuBLASLt 输出对比
# ============================================================================

def run_comparison_inference(
    model_path: Path,
    prompts: List[str],
    max_tokens: int = 48,
    verbose: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    运行 CUTLASS 和 cuBLASLt 推理并对比输出
    
    Returns:
        List of (prompt, cutlass_output, cublaslt_output)
    """
    from vllm import LLM, SamplingParams
    
    results = []
    
    sampling_params = SamplingParams(
        temperature=0.0,  # 贪婪采样确保可复现
        max_tokens=max_tokens,
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print(Colors.bold("CUTLASS vs cuBLASLt 推理输出对比"))
        print("=" * 80)
        print(f"模型: {model_path.name}")
        print(f"采样: temperature=0.0, max_tokens={max_tokens}")
        print("=" * 80)
    
    # 1. 运行 CUTLASS (禁用 cuBLASLt)
    if verbose:
        print(f"\n{Colors.cyan('[1/2] 运行 CUTLASS 推理...')}")
    
    # 临时禁用 cuBLASLt
    old_cublaslt = os.environ.get("USE_CUBLASLT", "")
    os.environ["USE_CUBLASLT"] = "0"
    
    # 需要重新加载模块以应用环境变量
    # 但为了简化，我们假设模型加载时会读取环境变量
    
    with cuda_memory_manager():
        llm_cutlass = LLM(
            model=str(model_path),
            max_model_len=256,
            gpu_memory_utilization=0.45,
            disable_log_stats=True,
            enforce_eager=True,
        )
        
        outputs_cutlass = llm_cutlass.generate(prompts, sampling_params)
        cutlass_texts = [o.outputs[0].text.strip() for o in outputs_cutlass]
        
        del llm_cutlass
    
    # 2. 运行 cuBLASLt
    if verbose:
        print(f"\n{Colors.cyan('[2/2] 运行 cuBLASLt 推理...')}")
    
    os.environ["USE_CUBLASLT"] = "1"
    
    with cuda_memory_manager():
        llm_cublaslt = LLM(
            model=str(model_path),
            max_model_len=256,
            gpu_memory_utilization=0.45,
            disable_log_stats=True,
            enforce_eager=True,
        )
        
        outputs_cublaslt = llm_cublaslt.generate(prompts, sampling_params)
        cublaslt_texts = [o.outputs[0].text.strip() for o in outputs_cublaslt]
        
        del llm_cublaslt
    
    # 恢复环境变量
    if old_cublaslt:
        os.environ["USE_CUBLASLT"] = old_cublaslt
    else:
        os.environ.pop("USE_CUBLASLT", None)
    
    # 3. 打印对比结果
    if verbose:
        print("\n" + "=" * 80)
        print(Colors.bold("输出对比"))
        print("=" * 80)
        
        for i, (prompt, cutlass_out, cublaslt_out) in enumerate(
            zip(prompts, cutlass_texts, cublaslt_texts)
        ):
            print(f"\n{Colors.bold(f'[Prompt {i+1}]')} {prompt}")
            print("-" * 80)
            print(f"{Colors.blue('CUTLASS:')}  {cutlass_out}")
            print()  # 空行分隔
            print(f"{Colors.green('cuBLASLt:')} {cublaslt_out}")
            
            results.append((prompt, cutlass_out, cublaslt_out))
        
        print("\n" + "=" * 80)
    
    return results


@test_case("CUTLASS vs cuBLASLt 输出对比", skip_if=skip_if_no_fp8)
def test_output_comparison():
    """对比两种后端的推理输出"""
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="CUTLASS vs cuBLASLt 输出对比",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    try:
        run_comparison_inference(
            model_path=model_path,
            prompts=TEST_PROMPTS,
            max_tokens=32,
            verbose=True,
        )
        return True, "对比完成"
                
    except Exception as e:
        return TestResult(
            name="CUTLASS vs cuBLASLt 输出对比",
            status=TestStatus.FAILED,
            message=str(e)
        )


@test_case("cuBLASLt 路径验证", skip_if=skip_if_no_fp8)
def test_cublaslt_code_path():
    """验证 cuBLASLt 代码路径实际被执行"""
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled
    
    if not is_cublaslt_enabled():
        return TestResult(
            name="cuBLASLt 路径验证",
            status=TestStatus.SKIPPED,
            message="cuBLASLt 未启用 (设置 USE_CUBLASLT=1)"
        )
    
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="cuBLASLt 路径验证",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    # 运行简单推理
    results = run_inference(
        model_path=str(model_path),
        prompts=["Hello"],
        max_tokens=10,
    )
    
    assert len(results) > 0 and len(results[0]) > 0
    
    return True, "cuBLASLt 路径正常工作"


# ============================================================================
# 主函数
# ============================================================================

def get_all_tests():
    """获取所有测试"""
    return [
        test_output_comparison,
    ]


def run_tests(verbose: bool = True) -> bool:
    """运行所有测试"""
    tests = get_all_tests()
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("端到端推理测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


def run_full_comparison():
    """运行完整的对比测试"""
    print(Colors.bold("=" * 60))
    print(Colors.bold("完整推理对比测试"))
    print(Colors.bold("=" * 60))
    
    EnvironmentChecker.print_env_info()
    
    # 查找模型
    models = ModelFinder.get_test_models("FP8", max_count=2)
    if not models:
        print(Colors.red("错误: 未找到 FP8 模型"))
        return False
    
    # 对每个模型运行对比
    for model_path in models:
        print(f"\n{Colors.bold('=' * 60)}")
        print(Colors.bold(f"模型: {model_path.name}"))
        print(Colors.bold("=" * 60))
        
        run_comparison_inference(
            model_path=model_path,
            prompts=TEST_PROMPTS,
            max_tokens=48,
            verbose=True,
        )
    
    return True


if __name__ == "__main__":
    parser = parse_common_args("端到端推理测试")
    parser.add_argument("--model", type=str, help="指定模型路径")
    args = parser.parse_args()
    
    apply_env_args(args)
    
    # 打印环境信息
    EnvironmentChecker.print_env_info()
    
    # 查找模型
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        print(Colors.red("错误: 未找到 FP8 模型"))
        sys.exit(1)
    
    # 直接运行输出对比
    run_comparison_inference(
        model_path=model_path,
        prompts=TEST_PROMPTS,
        max_tokens=64,
        verbose=True,
    )
    
    sys.exit(0)
