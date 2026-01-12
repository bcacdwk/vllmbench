#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
04_throughput.py - 吞吐量对比测试

对比 vLLM 原生 CUTLASS 和 slidesparse 后端的吞吐量差异。

对比路径:
=========
                        ┌─────────────────────────────────────┐
    [vLLM 原生 CUTLASS] │  USE_CUBLASLT=0, 无 slidesparse hook│  ← 基准
                        └─────────────────────────────────────┘
                              vs
                        ┌─────────────────────────────────────┐
    [slidesparse 后端]  │  根据参数选择不同 kernel            │  ← 测试
                        └─────────────────────────────────────┘

测试指标:
- Prefill 吞吐量 (tokens/s): 长输入 + 短输出
- Decode 吞吐量 (tokens/s): 短输入 + 长输出

使用方法:
    python3 04_throughput.py                # 对比: 原生 CUTLASS vs cuBLASLt
    python3 04_throughput.py --inner-fp32   # 对比: 原生 CUTLASS vs cuBLASLt(FP32累加)

    python3 04_throughput.py --ext-cutlass  # 对比: 原生 CUTLASS vs 外挂 CUTLASS (应相同)

slidesparse 后端说明:
    默认:         USE_CUBLASLT=1 → cuBLASLt kernel
    --ext-cutlass: USE_CUBLASLT=0 → 外挂 CUTLASS kernel
    --inner-fp32:  INNER_DTYPE_FP32=1 → cuBLASLt + FP32 中间累加
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    EnvironmentChecker,
    ModelFinder,
    Colors,
    cuda_memory_manager,
    parse_common_args,
)


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


# ============================================================================
# 核心功能：吞吐量对比
# ============================================================================

def get_backend_name(use_cublaslt: bool, inner_fp32: bool) -> str:
    """获取后端名称"""
    if use_cublaslt:
        if inner_fp32:
            return "cuBLASLt (FP32累加)"
        return "cuBLASLt"
    else:
        return "外挂 CUTLASS"


def run_comparison_throughput(
    model_path: Path,
    use_cublaslt: bool,
    inner_fp32: bool,
    verbose: bool = True,
) -> dict:
    """
    运行吞吐量对比测试
    
    Args:
        model_path: 模型路径
        use_cublaslt: slidesparse 后端是否使用 cuBLASLt
        inner_fp32: 是否使用 FP32 中间累加
        verbose: 是否打印详细信息
    
    Returns:
        对比结果字典
    """
    backend_name = get_backend_name(use_cublaslt, inner_fp32)
    
    if verbose:
        print("\n" + "=" * 90)
        print(Colors.bold("vLLM 原生 CUTLASS vs slidesparse 吞吐量对比"))
        print("=" * 90)
        print(f"模型: {model_path.name}")
        print(f"基准: vLLM 原生 CUTLASS")
        print(f"测试: slidesparse {backend_name}")
        print("=" * 90)
    
    results = {}
    
    # 保存原环境变量
    old_cublaslt = os.environ.get("USE_CUBLASLT")
    old_inner_fp32 = os.environ.get("INNER_DTYPE_FP32")
    
    # ========== Prefill 测试 ==========
    if verbose:
        print(f"\n{Colors.cyan('━' * 40)}")
        print(Colors.bold("Prefill 吞吐量测试"))
        print(f"配置: 8 prompts × 256 input tokens × 1 output token")
        print(f"{Colors.cyan('━' * 40)}")
    
    # 基准: vLLM 原生 CUTLASS
    if verbose:
        print(f"\n  {Colors.blue('[基准] vLLM 原生 CUTLASS...')}")
    os.environ["USE_CUBLASLT"] = "0"
    os.environ.pop("INNER_DTYPE_FP32", None)
    
    baseline_prefill = run_throughput_test(
        str(model_path),
        num_prompts=8,
        prompt_len=256,
        output_len=1,
        warmup=2,
    )
    baseline_prefill_tps = baseline_prefill["input_tokens"] / baseline_prefill["total_time"]
    
    # 测试: slidesparse 后端
    if verbose:
        print(f"  {Colors.green(f'[测试] slidesparse {backend_name}...')}")
    
    if use_cublaslt:
        os.environ["USE_CUBLASLT"] = "1"
        if inner_fp32:
            os.environ["INNER_DTYPE_FP32"] = "1"
        else:
            os.environ.pop("INNER_DTYPE_FP32", None)
    else:
        os.environ["USE_CUBLASLT"] = "0"
        os.environ.pop("INNER_DTYPE_FP32", None)
    
    test_prefill = run_throughput_test(
        str(model_path),
        num_prompts=8,
        prompt_len=256,
        output_len=1,
        warmup=2,
    )
    test_prefill_tps = test_prefill["input_tokens"] / test_prefill["total_time"]
    
    prefill_speedup = test_prefill_tps / baseline_prefill_tps if baseline_prefill_tps > 0 else 0
    results["prefill"] = {
        "baseline": baseline_prefill_tps,
        "test": test_prefill_tps,
        "speedup": prefill_speedup,
    }
    
    if verbose:
        print(f"\n  结果:")
        print(f"    原生 CUTLASS: {baseline_prefill_tps:>10.1f} tok/s")
        print(f"    {backend_name}: {test_prefill_tps:>10.1f} tok/s")
        speedup_str = f"{prefill_speedup:.3f}x"
        if prefill_speedup > 1.02:
            speedup_str = Colors.green(speedup_str + " ↑")
        elif prefill_speedup < 0.98:
            speedup_str = Colors.red(speedup_str + " ↓")
        else:
            speedup_str = Colors.yellow(speedup_str + " ≈")
        print(f"    加速比: {speedup_str}")
    
    # ========== Decode 测试 ==========
    if verbose:
        print(f"\n{Colors.cyan('━' * 40)}")
        print(Colors.bold("Decode 吞吐量测试"))
        print(f"配置: 4 prompts × 16 input tokens × 128 output tokens")
        print(f"{Colors.cyan('━' * 40)}")
    
    # 基准: vLLM 原生 CUTLASS
    if verbose:
        print(f"\n  {Colors.blue('[基准] vLLM 原生 CUTLASS...')}")
    os.environ["USE_CUBLASLT"] = "0"
    os.environ.pop("INNER_DTYPE_FP32", None)
    
    baseline_decode = run_throughput_test(
        str(model_path),
        num_prompts=4,
        prompt_len=16,
        output_len=128,
        warmup=2,
    )
    baseline_decode_tps = baseline_decode["throughput_tok"]
    
    # 测试: slidesparse 后端
    if verbose:
        print(f"  {Colors.green(f'[测试] slidesparse {backend_name}...')}")
    
    if use_cublaslt:
        os.environ["USE_CUBLASLT"] = "1"
        if inner_fp32:
            os.environ["INNER_DTYPE_FP32"] = "1"
        else:
            os.environ.pop("INNER_DTYPE_FP32", None)
    else:
        os.environ["USE_CUBLASLT"] = "0"
        os.environ.pop("INNER_DTYPE_FP32", None)
    
    test_decode = run_throughput_test(
        str(model_path),
        num_prompts=4,
        prompt_len=16,
        output_len=128,
        warmup=2,
    )
    test_decode_tps = test_decode["throughput_tok"]
    
    decode_speedup = test_decode_tps / baseline_decode_tps if baseline_decode_tps > 0 else 0
    results["decode"] = {
        "baseline": baseline_decode_tps,
        "test": test_decode_tps,
        "speedup": decode_speedup,
    }
    
    if verbose:
        print(f"\n  结果:")
        print(f"    原生 CUTLASS: {baseline_decode_tps:>10.1f} tok/s")
        print(f"    {backend_name}: {test_decode_tps:>10.1f} tok/s")
        speedup_str = f"{decode_speedup:.3f}x"
        if decode_speedup > 1.02:
            speedup_str = Colors.green(speedup_str + " ↑")
        elif decode_speedup < 0.98:
            speedup_str = Colors.red(speedup_str + " ↓")
        else:
            speedup_str = Colors.yellow(speedup_str + " ≈")
        print(f"    加速比: {speedup_str}")
    
    # 恢复环境变量
    if old_cublaslt is not None:
        os.environ["USE_CUBLASLT"] = old_cublaslt
    else:
        os.environ.pop("USE_CUBLASLT", None)
    if old_inner_fp32 is not None:
        os.environ["INNER_DTYPE_FP32"] = old_inner_fp32
    else:
        os.environ.pop("INNER_DTYPE_FP32", None)
    
    # 总结
    if verbose:
        print(f"\n{'=' * 90}")
        print(Colors.bold("总结"))
        print(f"{'=' * 90}")
        print(f"  Prefill: {results['prefill']['speedup']:.3f}x")
        print(f"  Decode:  {results['decode']['speedup']:.3f}x")
        if not use_cublaslt:
            print(f"\n  {Colors.yellow('注意')}: --ext-cutlass 模式下两边都是 CUTLASS，加速比应≈1.0")
        print(f"{'=' * 90}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = parse_common_args("吞吐量对比测试")
    args = parser.parse_args()
    
    # 注意：不调用 apply_env_args(args)
    # 因为 run_comparison_throughput 会自己管理环境变量
    
    # 打印环境信息
    EnvironmentChecker.print_env_info()
    
    # 查找模型
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        print(Colors.red("错误: 未找到 FP8 模型"))
        sys.exit(1)
    
    # 根据参数决定测试的 slidesparse 后端
    use_cublaslt = not getattr(args, 'ext_cutlass', False)
    inner_fp32 = getattr(args, 'inner_fp32', False)
    
    # 运行吞吐量对比
    run_comparison_throughput(
        model_path=model_path,
        use_cublaslt=use_cublaslt,
        inner_fp32=inner_fp32,
        verbose=True,
    )
    
    sys.exit(0)
