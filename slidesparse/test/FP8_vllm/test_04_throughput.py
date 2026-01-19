#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_04_throughput.py - 吞吐量对比测试

分离测试 Prefill 和 Decode 两个阶段的吞吐量：
  - Prefill: 长输入 + 短输出（测试 prompt 处理速度）
  - Decode: 短输入 + 长输出（测试 token 生成速度）

对比路径:
=========
                        ┌─────────────────────────────────────┐
    [vLLM 原生路径]     │  DISABLE_SLIDESPARSE=1              │  ← 基准
                        └─────────────────────────────────────┘
                              vs
                        ┌─────────────────────────────────────┐
    [SlideSparse 后端]  │  根据参数选择不同 kernel            │  ← 测试
                        └─────────────────────────────────────┘

使用方法:
    python3 test_04_throughput.py                          # 默认: vs CUTLASS fallback
    python3 test_04_throughput.py --use-cublaslt           # vs cuBLASLt
    python3 test_04_throughput.py --use-cublaslt --inner-fp32  # cuBLASLt + FP32
    python3 test_04_throughput.py --use-cusparselt         # vs cuSPARSELt (TODO)
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    EnvironmentChecker,
    ModelFinder,
    Colors,
    cuda_memory_manager,
    parse_common_args,
    get_backend_name,
    set_env_for_baseline,
    set_env_for_test,
    restore_env,
)


# ============================================================================
# 测试配置
# ============================================================================

# Prefill 测试配置：长输入 + 短输出
PREFILL_CONFIG = {
    "num_prompts": 8,
    "prompt_len": 256,
    "output_len": 1,
}

# Decode 测试配置：短输入 + 长输出
DECODE_CONFIG = {
    "num_prompts": 4,
    "prompt_len": 16,
    "output_len": 128,
}


# ============================================================================
# 辅助函数
# ============================================================================

def generate_dummy_prompt(length: int, seed: int = 0) -> str:
    """
    生成指定 token 长度的 dummy prompt
    
    Args:
        length: 目标 token 长度（近似）
        seed: 随机种子，不同 seed 生成不同内容，避免 prefix caching
    """
    import random
    rng = random.Random(seed)
    
    # 使用多样化的词汇，每个词约 1-2 tokens
    words = [
        "Hello", "world", "this", "is", "a", "test", "prompt", "for", "benchmark",
        "The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and",
        "Python", "programming", "language", "machine", "learning", "artificial",
        "intelligence", "deep", "neural", "network", "transformer", "attention",
        "model", "training", "inference", "optimization", "performance", "speed",
    ]
    
    # 随机打乱词汇顺序，确保不同 seed 生成不同的 prompt
    shuffled_words = words.copy()
    rng.shuffle(shuffled_words)
    
    # 生成足够长度的文本
    result_words = []
    for i in range(length):
        result_words.append(shuffled_words[i % len(shuffled_words)])
    
    return " ".join(result_words)


@dataclass
class PhaseResult:
    """单阶段测试结果"""
    total_time_s: float
    input_tokens: int
    output_tokens: int
    
    @property
    def throughput_tps(self) -> float:
        """主要吞吐量指标 (tokens/s)"""
        return self.input_tokens / self.total_time_s if self.total_time_s > 0 else 0


# ============================================================================
# 核心测试函数
# ============================================================================

def run_phase_test(
    model_path: Path,
    num_prompts: int,
    prompt_len: int,
    output_len: int,
    warmup: int = 2,
) -> PhaseResult:
    """
    运行单阶段吞吐量测试
    
    Args:
        model_path: 模型路径
        num_prompts: 请求数量
        prompt_len: 输入 prompt 的 token 长度
        output_len: 输出的 token 长度
        warmup: 预热次数
    
    Returns:
        PhaseResult
    """
    from vllm import LLM, SamplingParams
    import torch
    
    with cuda_memory_manager():
        llm = LLM(
            model=str(model_path),
            max_model_len=prompt_len + output_len + 64,
            gpu_memory_utilization=0.8,
            disable_log_stats=True,
            enforce_eager=True,
        )
        
        # 生成 prompts（每个 prompt 使用不同 seed，避免 prefix caching）
        prompts = [generate_dummy_prompt(prompt_len, seed=i) for i in range(num_prompts)]
        
        # Warmup 用不同的 prompts（seed 从 1000 开始，避免和正式测试重叠）
        warmup_prompts = [generate_dummy_prompt(prompt_len, seed=1000+i) for i in range(2)]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=output_len,
            ignore_eos=True,  # 强制生成指定数量的 token
        )
        
        # Warmup
        for _ in range(warmup):
            _ = llm.generate(warmup_prompts, sampling_params)
        
        # 正式测试
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        outputs = llm.generate(prompts, sampling_params)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        # 统计
        total_input_tokens = sum(len(o.prompt_token_ids) for o in outputs)
        total_output_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        
        del llm
    
    return PhaseResult(
        total_time_s=elapsed,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def format_speedup(speedup: float) -> str:
    """格式化加速比显示"""
    speedup_str = f"{speedup:.3f}x"
    if speedup > 1.02:
        return Colors.green(speedup_str + " ↑")
    elif speedup < 0.98:
        return Colors.red(speedup_str + " ↓")
    else:
        return Colors.yellow(speedup_str + " ≈")


# ============================================================================
# 吞吐量对比测试
# ============================================================================

def run_throughput_comparison(
    model_path: Path,
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_fp32: bool = False,
    sparsity: str = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    运行完整的吞吐量对比测试（分离 Prefill 和 Decode）
    
    Args:
        model_path: 模型路径
        use_cublaslt: SlideSparse 后端是否使用 cuBLASLt
        use_cusparselt: SlideSparse 后端是否使用 cuSPARSELt
        inner_fp32: 是否使用 FP32 中间累加
        sparsity: 稀疏配置 (如 "2_8", "2_6")，仅 cuSPARSELt 有效
        verbose: 是否打印详细信息
    
    Returns:
        对比结果字典
    """
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_fp32, sparsity)
    results = {}
    
    # 对于 cuSPARSELt，跳过 baseline（使用不同 checkpoint）
    skip_baseline = use_cusparselt
    
    if verbose:
        print("\n" + "=" * 90)
        print(Colors.bold("vLLM 原生路径 vs SlideSparse 吞吐量对比"))
        print("=" * 90)
        print(f"模型: {model_path.name}")
        if skip_baseline:
            print(f"基准: [跳过] cuSPARSELt 使用 SlideSparse checkpoint")
        else:
            print(f"基准: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)")
        print(f"测试: {backend_name}")
        print("=" * 90)
    
    # ========== Prefill 测试 ==========
    if verbose:
        print(f"\n{Colors.cyan('━' * 50)}")
        print(Colors.bold("Prefill 吞吐量测试"))
        print(f"配置: {PREFILL_CONFIG['num_prompts']} prompts × "
              f"{PREFILL_CONFIG['prompt_len']} input tokens × "
              f"{PREFILL_CONFIG['output_len']} output token")
        print(f"{Colors.cyan('━' * 50)}")
    
    # 基准: vLLM 原生路径
    baseline_prefill_tps = None
    if skip_baseline:
        if verbose:
            print(f"\n  {Colors.blue('[基准] 跳过 (cuSPARSELt 使用 SlideSparse checkpoint)')}")
    else:
        if verbose:
            print(f"\n  {Colors.blue('[基准] vLLM 原生路径...')}")
        saved_env = set_env_for_baseline()
        try:
            baseline_prefill = run_phase_test(
                model_path,
                num_prompts=PREFILL_CONFIG["num_prompts"],
                prompt_len=PREFILL_CONFIG["prompt_len"],
                output_len=PREFILL_CONFIG["output_len"],
            )
            baseline_prefill_tps = baseline_prefill.input_tokens / baseline_prefill.total_time_s
        except Exception as e:
            if verbose:
                print(f"    {Colors.yellow(f'跳过 (CUTLASS 不支持当前 GPU): {e}')}")
        restore_env(saved_env)
    
    # 测试: SlideSparse 后端
    if verbose:
        print(f"  {Colors.green(f'[测试] {backend_name}...')}")
    saved_env = set_env_for_test(use_cublaslt, use_cusparselt, inner_fp32, sparsity)
    test_prefill = run_phase_test(
        model_path,
        num_prompts=PREFILL_CONFIG["num_prompts"],
        prompt_len=PREFILL_CONFIG["prompt_len"],
        output_len=PREFILL_CONFIG["output_len"],
    )
    test_prefill_tps = test_prefill.input_tokens / test_prefill.total_time_s
    restore_env(saved_env)
    
    if baseline_prefill_tps is not None:
        prefill_speedup = test_prefill_tps / baseline_prefill_tps if baseline_prefill_tps > 0 else 0
        results["prefill"] = {
            "baseline_tps": baseline_prefill_tps,
            "test_tps": test_prefill_tps,
            "speedup": prefill_speedup,
        }
        if verbose:
            print(f"\n  结果:")
            print(f"    vLLM 原生路径: {baseline_prefill_tps:>10.1f} tok/s")
            print(f"    {backend_name}: {test_prefill_tps:>10.1f} tok/s")
            print(f"    加速比: {format_speedup(prefill_speedup)}")
    else:
        results["prefill"] = {"test_tps": test_prefill_tps}
        if verbose:
            print(f"\n  结果:")
            print(f"    {backend_name}: {test_prefill_tps:>10.1f} tok/s")
            print(f"    (无 baseline 对比)")
    
    # ========== Decode 测试 ==========
    if verbose:
        print(f"\n{Colors.cyan('━' * 50)}")
        print(Colors.bold("Decode 吞吐量测试"))
        print(f"配置: {DECODE_CONFIG['num_prompts']} prompts × "
              f"{DECODE_CONFIG['prompt_len']} input tokens × "
              f"{DECODE_CONFIG['output_len']} output tokens")
        print(f"{Colors.cyan('━' * 50)}")
    
    # 基准: vLLM 原生路径
    baseline_decode_tps = None
    if skip_baseline:
        if verbose:
            print(f"\n  {Colors.blue('[基准] 跳过 (cuSPARSELt 使用 SlideSparse checkpoint)')}")
    else:
        if verbose:
            print(f"\n  {Colors.blue('[基准] vLLM 原生路径...')}")
        saved_env = set_env_for_baseline()
        try:
            baseline_decode = run_phase_test(
                model_path,
                num_prompts=DECODE_CONFIG["num_prompts"],
                prompt_len=DECODE_CONFIG["prompt_len"],
                output_len=DECODE_CONFIG["output_len"],
            )
            baseline_decode_tps = baseline_decode.output_tokens / baseline_decode.total_time_s
        except Exception as e:
            if verbose:
                print(f"    {Colors.yellow(f'跳过 (CUTLASS 不支持当前 GPU): {e}')}")
        restore_env(saved_env)
    
    # 测试: SlideSparse 后端
    if verbose:
        print(f"  {Colors.green(f'[测试] {backend_name}...')}")
    saved_env = set_env_for_test(use_cublaslt, use_cusparselt, inner_fp32, sparsity)
    test_decode = run_phase_test(
        model_path,
        num_prompts=DECODE_CONFIG["num_prompts"],
        prompt_len=DECODE_CONFIG["prompt_len"],
        output_len=DECODE_CONFIG["output_len"],
    )
    test_decode_tps = test_decode.output_tokens / test_decode.total_time_s
    restore_env(saved_env)
    
    if baseline_decode_tps is not None:
        decode_speedup = test_decode_tps / baseline_decode_tps if baseline_decode_tps > 0 else 0
        results["decode"] = {
            "baseline_tps": baseline_decode_tps,
            "test_tps": test_decode_tps,
            "speedup": decode_speedup,
        }
        if verbose:
            print(f"\n  结果:")
            print(f"    vLLM 原生路径: {baseline_decode_tps:>10.1f} tok/s")
            print(f"    {backend_name}: {test_decode_tps:>10.1f} tok/s")
            print(f"    加速比: {format_speedup(decode_speedup)}")
    else:
        results["decode"] = {"test_tps": test_decode_tps}
        if verbose:
            print(f"\n  结果:")
            print(f"    {backend_name}: {test_decode_tps:>10.1f} tok/s")
            print(f"    (无 baseline 对比)")
    
    # ========== 总结 ==========
    if verbose:
        print(f"\n{'=' * 90}")
        print(Colors.bold("总结"))
        print(f"{'=' * 90}")
        if "speedup" in results.get("prefill", {}):
            print(f"  Prefill (长输入): {format_speedup(results['prefill']['speedup'])}")
        else:
            print(f"  Prefill (长输入): {results['prefill']['test_tps']:.1f} tok/s (无 baseline)")
        if "speedup" in results.get("decode", {}):
            print(f"  Decode  (长输出): {format_speedup(results['decode']['speedup'])}")
        else:
            print(f"  Decode  (长输出): {results['decode']['test_tps']:.1f} tok/s (无 baseline)")
        print(f"{'=' * 90}")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = parse_common_args("吞吐量对比测试（分离 Prefill/Decode）")
    args = parser.parse_args()
    
    # 注意：不调用 apply_env_args(args)
    # 因为 run_throughput_comparison 会自己管理环境变量
    
    # 打印环境信息
    EnvironmentChecker.print_env_info()
    
    # 根据参数决定测试的 SlideSparse 后端
    use_cublaslt = getattr(args, 'use_cublaslt', False)
    use_cusparselt = getattr(args, 'use_cusparselt', False)
    inner_fp32 = getattr(args, 'inner_fp32', False)
    sparsity = getattr(args, 'sparsity', None)
    
    # 查找模型
    if use_cusparselt:
        # cuSPARSELt 需要 SlideSparse checkpoint（已预先稀疏化）
        model_path = ModelFinder.find_slidesparse_model("FP8", sparsity)
        if model_path is None:
            print(Colors.red(f"错误: 未找到 SlideSparse checkpoint (sparsity={sparsity or '2_8'})"))
            sys.exit(1)
    else:
        # 其他后端使用普通 FP8 checkpoint
        model_path = ModelFinder.find_small_model("FP8")
        if model_path is None:
            print(Colors.red("错误: 未找到 FP8 模型"))
            sys.exit(1)
    
    # 运行吞吐量对比
    run_throughput_comparison(
        model_path=model_path,
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_fp32=inner_fp32,
        sparsity=sparsity,
        verbose=True,
    )
    
    sys.exit(0)
