#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
04_throughput.py - 真实模型吞吐量测试

测试实际 LLM 推理的吞吐量，对比 原生 CUTLASS 和 cuBLASLt/外挂CUTLASS 后端。

测试指标:
- Prefill 吞吐量 (tokens/s): 首次 token 生成前处理的速度
- Decode 吞吐量 (tokens/s): 后续 token 生成速度
- 端到端吞吐量 (requests/s, tokens/s)

测试模型:
- Qwen2.5-0.5B-FP8 (最小)
- Llama3.2-1B-FP8

使用方法:
    python3 04_throughput.py                   # 基本吞吐量测试 (cuBLASLt)
    python3 04_throughput.py --ext-cutlass     # 测试外挂 CUTLASS 路径
    python3 04_throughput.py --full            # 完整对比测试

路径说明:
    默认: USE_CUBLASLT=1 → cuBLASLt kernel
    --ext-cutlass: USE_CUBLASLT=0 → 外挂 CUTLASS
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
    skip_if_no_cuda,
    skip_if_no_fp8,
    parse_common_args,
    apply_env_args,
)


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class ThroughputConfig:
    """吞吐量测试配置"""
    # Prefill 测试
    prefill_prompt_len: int = 512      # 输入 prompt 长度
    prefill_output_len: int = 1        # 只生成 1 个 token 以隔离 prefill
    prefill_num_prompts: int = 16      # 并发请求数
    
    # Decode 测试
    decode_prompt_len: int = 16        # 短 prompt
    decode_output_len: int = 128       # 生成多个 token
    decode_num_prompts: int = 8        # 并发请求数
    
    # 端到端测试
    e2e_prompt_len: int = 64
    e2e_output_len: int = 64
    e2e_num_prompts: int = 16


DEFAULT_CONFIG = ThroughputConfig()


# ============================================================================
# 辅助函数
# ============================================================================

def generate_dummy_prompt(length: int) -> str:
    """生成指定 token 长度的 dummy prompt"""
    # 简单方法：重复单词
    words = ["Hello", "world", "this", "is", "a", "test", "prompt", "for", "benchmark"]
    prompt = " ".join(words * (length // len(words) + 1))
    return prompt


def run_throughput_test(
    model_path: str,
    num_prompts: int,
    prompt_len: int,
    output_len: int,
    warmup: int = 2,
) -> Dict[str, float]:
    """
    运行吞吐量测试
    
    Returns:
        {
            "total_time": float,      # 总耗时 (s)
            "input_tokens": int,      # 输入 token 数
            "output_tokens": int,     # 输出 token 数
            "throughput_req": float,  # 请求吞吐 (req/s)
            "throughput_tok": float,  # token 吞吐 (tok/s)
        }
    """
    from vllm import LLM, SamplingParams
    
    with cuda_memory_manager():
        # 创建 LLM
        llm = LLM(
            model=model_path,
            max_model_len=prompt_len + output_len + 64,
            gpu_memory_utilization=0.8,
            disable_log_stats=True,
            enforce_eager=True,
        )
        
        # 生成 prompts
        prompts = [generate_dummy_prompt(prompt_len) for _ in range(num_prompts)]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=output_len,
            ignore_eos=True,  # 强制生成指定数量的 token
        )
        
        # Warmup
        for _ in range(warmup):
            _ = llm.generate(prompts[:2], sampling_params)
        
        # 正式测试
        import torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = llm.generate(prompts, sampling_params)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # 计算统计
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        
        del llm
    
    return {
        "total_time": elapsed,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "throughput_req": num_prompts / elapsed,
        "throughput_tok": total_output_tokens / elapsed,
    }


def run_comparison_throughput(
    model_path: Path,
    config: ThroughputConfig,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    运行 CUTLASS vs cuBLASLt 吞吐量对比
    
    Returns:
        {
            "prefill": {"cutlass": {...}, "cublaslt": {...}},
            "decode": {"cutlass": {...}, "cublaslt": {...}},
            "e2e": {"cutlass": {...}, "cublaslt": {...}},
        }
    """
    results = {}
    
    if verbose:
        print(f"\n{'=' * 80}")
        print(Colors.bold(f"模型: {model_path.name}"))
        print("=" * 80)
    
    # 测试类型
    tests = [
        ("prefill", config.prefill_num_prompts, config.prefill_prompt_len, config.prefill_output_len),
        ("decode", config.decode_num_prompts, config.decode_prompt_len, config.decode_output_len),
        ("e2e", config.e2e_num_prompts, config.e2e_prompt_len, config.e2e_output_len),
    ]
    
    for test_name, num_prompts, prompt_len, output_len in tests:
        if verbose:
            print(f"\n{Colors.cyan(f'[{test_name.upper()}]')} "
                  f"num_prompts={num_prompts}, prompt_len={prompt_len}, output_len={output_len}")
        
        results[test_name] = {}
        
        # 1. CUTLASS
        old_env = os.environ.get("USE_CUBLASLT", "")
        os.environ["USE_CUBLASLT"] = "0"
        
        try:
            cutlass_result = run_throughput_test(
                str(model_path), num_prompts, prompt_len, output_len
            )
            results[test_name]["cutlass"] = cutlass_result
            
            if verbose:
                print(f"  CUTLASS:  {cutlass_result['throughput_tok']:.1f} tok/s "
                      f"({cutlass_result['throughput_req']:.2f} req/s, "
                      f"{cutlass_result['total_time']:.2f}s)")
        except Exception as e:
            results[test_name]["cutlass"] = {"error": str(e)}
            if verbose:
                print(f"  CUTLASS:  ERROR - {e}")
        
        # 2. cuBLASLt
        os.environ["USE_CUBLASLT"] = "1"
        
        try:
            cublaslt_result = run_throughput_test(
                str(model_path), num_prompts, prompt_len, output_len
            )
            results[test_name]["cublaslt"] = cublaslt_result
            
            if verbose:
                print(f"  cuBLASLt: {cublaslt_result['throughput_tok']:.1f} tok/s "
                      f"({cublaslt_result['throughput_req']:.2f} req/s, "
                      f"{cublaslt_result['total_time']:.2f}s)")
                
                # 计算加速比
                if "throughput_tok" in results[test_name].get("cutlass", {}):
                    speedup = (cublaslt_result["throughput_tok"] / 
                               results[test_name]["cutlass"]["throughput_tok"])
                    speedup_str = f"{speedup:.2f}x"
                    if speedup > 1.05:
                        speedup_str = Colors.green(speedup_str)
                    elif speedup < 0.95:
                        speedup_str = Colors.red(speedup_str)
                    print(f"  Speedup:  {speedup_str}")
                    
        except Exception as e:
            results[test_name]["cublaslt"] = {"error": str(e)}
            if verbose:
                print(f"  cuBLASLt: ERROR - {e}")
        
        # 恢复环境变量
        if old_env:
            os.environ["USE_CUBLASLT"] = old_env
        else:
            os.environ.pop("USE_CUBLASLT", None)
    
    return results


# ============================================================================
# 测试用例
# ============================================================================

@test_case("查找测试模型")
def test_find_models():
    """查找可用模型"""
    models = ModelFinder.get_test_models("FP8", max_count=2)
    
    if not models:
        return False, "未找到 FP8 模型"
    
    return True, f"找到 {len(models)} 个模型"


@test_case("快速吞吐量测试", skip_if=skip_if_no_fp8)
def test_quick_throughput():
    """快速吞吐量测试"""
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="快速吞吐量测试",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    # 简化配置
    result = run_throughput_test(
        str(model_path),
        num_prompts=4,
        prompt_len=32,
        output_len=32,
        warmup=1,
    )
    
    return True, f"{result['throughput_tok']:.1f} tok/s"


@test_case("Prefill 吞吐量", skip_if=skip_if_no_fp8)
def test_prefill_throughput():
    """Prefill 阶段吞吐量"""
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="Prefill 吞吐量",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    result = run_throughput_test(
        str(model_path),
        num_prompts=8,
        prompt_len=256,
        output_len=1,
        warmup=2,
    )
    
    # Prefill 吞吐用输入 token 计算
    prefill_tps = result["input_tokens"] / result["total_time"]
    
    return True, f"Prefill: {prefill_tps:.1f} tok/s (input)"


@test_case("Decode 吞吐量", skip_if=skip_if_no_fp8)
def test_decode_throughput():
    """Decode 阶段吞吐量"""
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        return TestResult(
            name="Decode 吞吐量",
            status=TestStatus.SKIPPED,
            message="未找到 FP8 模型"
        )
    
    result = run_throughput_test(
        str(model_path),
        num_prompts=4,
        prompt_len=16,
        output_len=64,
        warmup=2,
    )
    
    return True, f"Decode: {result['throughput_tok']:.1f} tok/s (output)"


# ============================================================================
# 完整对比测试
# ============================================================================

def run_full_comparison(config: ThroughputConfig = None):
    """运行完整的吞吐量对比测试"""
    if config is None:
        config = DEFAULT_CONFIG
    
    print(Colors.bold("=" * 80))
    print(Colors.bold("cuBLASLt vs CUTLASS 吞吐量对比测试"))
    print(Colors.bold("=" * 80))
    
    EnvironmentChecker.print_env_info()
    
    # 查找模型
    models = ModelFinder.get_test_models("FP8", max_count=2)
    if not models:
        print(Colors.red("错误: 未找到 FP8 模型"))
        return False
    
    print(f"\n测试模型: {', '.join(m.name for m in models)}")
    print(f"测试配置:")
    print(f"  Prefill: {config.prefill_num_prompts} prompts × "
          f"{config.prefill_prompt_len} input + {config.prefill_output_len} output")
    print(f"  Decode:  {config.decode_num_prompts} prompts × "
          f"{config.decode_prompt_len} input + {config.decode_output_len} output")
    print(f"  E2E:     {config.e2e_num_prompts} prompts × "
          f"{config.e2e_prompt_len} input + {config.e2e_output_len} output")
    
    all_results = {}
    
    for model_path in models:
        results = run_comparison_throughput(model_path, config, verbose=True)
        all_results[model_path.name] = results
    
    # 打印汇总表格
    print("\n" + "=" * 80)
    print(Colors.bold("汇总"))
    print("=" * 80)
    print(f"{'Model':<20} | {'Test':<8} | {'CUTLASS':>12} | {'cuBLASLt':>12} | {'Speedup':>8}")
    print("-" * 80)
    
    for model_name, model_results in all_results.items():
        for test_name, test_results in model_results.items():
            cutlass_tps = test_results.get("cutlass", {}).get("throughput_tok", 0)
            cublaslt_tps = test_results.get("cublaslt", {}).get("throughput_tok", 0)
            
            if cutlass_tps > 0 and cublaslt_tps > 0:
                speedup = cublaslt_tps / cutlass_tps
                speedup_str = f"{speedup:.2f}x"
                if speedup > 1.05:
                    speedup_str = Colors.green(speedup_str)
                elif speedup < 0.95:
                    speedup_str = Colors.red(speedup_str)
            else:
                speedup_str = "N/A"
            
            print(f"{model_name:<20} | {test_name:<8} | "
                  f"{cutlass_tps:>10.1f} | {cublaslt_tps:>10.1f} | {speedup_str:>8}")
    
    print("=" * 80)
    
    return True


# ============================================================================
# 主函数
# ============================================================================

def get_all_tests():
    """获取所有测试"""
    return [
        test_find_models,
        test_quick_throughput,
        test_prefill_throughput,
        test_decode_throughput,
    ]


def run_tests(verbose: bool = True) -> bool:
    """运行所有测试"""
    tests = get_all_tests()
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("吞吐量测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    parser = parse_common_args("吞吐量测试")
    parser.add_argument("--full", action="store_true",
                        help="运行完整的 CUTLASS vs cuBLASLt 对比")
    parser.add_argument("--prefill-len", type=int, default=512,
                        help="Prefill 测试的 prompt 长度")
    parser.add_argument("--decode-len", type=int, default=128,
                        help="Decode 测试的输出长度")
    args = parser.parse_args()
    
    apply_env_args(args)
    
    if args.full:
        config = ThroughputConfig(
            prefill_prompt_len=args.prefill_len,
            decode_output_len=args.decode_len,
        )
        success = run_full_comparison(config)
    else:
        success = run_tests(verbose=not args.quiet)
    
    sys.exit(0 if success else 1)
