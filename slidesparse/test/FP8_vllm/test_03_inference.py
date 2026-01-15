#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_03_inference.py - 端到端推理输出对比

对于相同的 prompt，分别用 vLLM 原生路径 和 SlideSparse 后端运行推理，
并排打印输出让用户直观比较精度差异。

对比路径:
=========
    [vLLM 原生路径]     DISABLE_SLIDESPARSE=1     ← baseline
                              vs
    [SlideSparse 后端]  根据参数选择不同 kernel    ← test

使用方法:
    python3 test_03_inference.py                          # 默认: vs CUTLASS fallback
    python3 test_03_inference.py --use-cublaslt           # vs cuBLASLt
    python3 test_03_inference.py --use-cublaslt --inner-fp32  # cuBLASLt + FP32
    python3 test_03_inference.py --use-cusparselt         # vs cuSPARSELt (TODO)
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

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

# 测试提示词
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in one sentence.",
    "Write a haiku about programming.",
]


# ============================================================================
# 核心功能：vLLM 原生路径 vs SlideSparse 后端 输出对比
# ============================================================================

def run_comparison_inference(
    model_path: Path,
    prompts: List[str],
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_fp32: bool = False,
    max_tokens: int = 48,
    verbose: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    运行 vLLM 原生路径 和 SlideSparse 后端推理并对比输出
    
    Args:
        model_path: 模型路径
        prompts: 提示词列表
        use_cublaslt: SlideSparse 后端是否使用 cuBLASLt
        use_cusparselt: SlideSparse 后端是否使用 cuSPARSELt
        inner_fp32: 是否使用 FP32 中间累加
        max_tokens: 最大生成 token 数
        verbose: 是否打印详细信息
    
    Returns:
        List of (prompt, baseline_output, test_output)
        如果 baseline 不可用，baseline_output 为 None
    """
    from vllm import LLM, SamplingParams
    
    results = []
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_fp32)
    
    # 检测 CUTLASS 是否支持当前 GPU
    cutlass_supported = EnvironmentChecker.supports_cutlass_fp8()
    
    sampling_params = SamplingParams(
        temperature=0.0,  # 贪婪采样确保可复现
        max_tokens=max_tokens,
    )
    
    if verbose:
        print("\n" + "=" * 80)
        if cutlass_supported:
            print(Colors.bold("vLLM 原生路径 vs SlideSparse 推理输出对比"))
        else:
            print(Colors.bold(f"{backend_name} 推理测试"))
        print("=" * 80)
        print(f"模型: {model_path.name}")
        if cutlass_supported:
            print(f"基准: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)")
        else:
            cc = EnvironmentChecker.cuda_compute_capability()
            print(Colors.yellow(f"注意: CUTLASS 不支持当前 GPU (sm_{cc[0]}{cc[1]})，跳过 baseline"))
        print(f"测试: {backend_name}")
        print(f"采样: temperature=0.0, max_tokens={max_tokens}")
        print("=" * 80)
    
    baseline_texts = None
    
    # 1. 运行 vLLM 原生路径 (基准) - 仅当 CUTLASS 支持时
    if cutlass_supported:
        if verbose:
            print(f"\n{Colors.cyan('[1/2] 运行 vLLM 原生路径 (基准)...')}")
        
        saved_env = set_env_for_baseline()
        
        try:
            with cuda_memory_manager():
                llm_baseline = LLM(
                    model=str(model_path),
                    max_model_len=256,
                    gpu_memory_utilization=0.45,
                    disable_log_stats=True,
                    enforce_eager=True,
                )
                
                outputs_baseline = llm_baseline.generate(prompts, sampling_params)
                baseline_texts = [o.outputs[0].text.strip() for o in outputs_baseline]
                
                del llm_baseline
        except RuntimeError as e:
            if "Error Internal" in str(e) or "cutlass" in str(e).lower():
                if verbose:
                    print(Colors.yellow(f"  CUTLASS 运行失败: {e}"))
                    print(Colors.yellow("  跳过 baseline 对比"))
                baseline_texts = None
            else:
                raise
        
        restore_env(saved_env)
    
    # 2. 运行 SlideSparse 后端 (测试)
    if verbose:
        step = "[1/1]" if not cutlass_supported else "[2/2]"
        print(f"\n{Colors.cyan(f'{step} 运行 {backend_name}...')}")
    
    saved_env = set_env_for_test(use_cublaslt, use_cusparselt, inner_fp32)
    
    with cuda_memory_manager():
        llm_test = LLM(
            model=str(model_path),
            max_model_len=256,
            gpu_memory_utilization=0.45,
            disable_log_stats=True,
            enforce_eager=True,
        )
        
        outputs_test = llm_test.generate(prompts, sampling_params)
        test_texts = [o.outputs[0].text.strip() for o in outputs_test]
        
        del llm_test
    
    restore_env(saved_env)
    
    # 3. 打印对比结果
    if verbose:
        print("\n" + "=" * 80)
        if baseline_texts is not None:
            print(Colors.bold("输出对比"))
        else:
            print(Colors.bold(f"{backend_name} 输出"))
        print("=" * 80)
        
        for i, prompt in enumerate(prompts):
            test_out = test_texts[i]
            baseline_out = baseline_texts[i] if baseline_texts else None
            
            print(f"\n{Colors.bold(f'[Prompt {i+1}]')} {prompt}")
            print("-" * 80)
            
            if baseline_out is not None:
                print(f"{Colors.blue('vLLM 原生:')} {baseline_out}")
                print()
                print(f"{Colors.green(f'{backend_name}:')} {test_out}")
                
                # 比较是否完全相同
                if baseline_out == test_out:
                    print(f"\n  {Colors.green('✓ 输出完全一致')}")
                else:
                    print(f"\n  {Colors.yellow('⚠ 输出有差异（FP8 精度正常）')}")
                
                results.append((prompt, baseline_out, test_out))
            else:
                # 无 baseline，只显示 test 输出
                print(f"{Colors.green(f'{backend_name}:')} {test_out}")
                results.append((prompt, None, test_out))
        
        print("\n" + "=" * 80)
        
        # 统计
        if baseline_texts is not None:
            identical = sum(1 for _, b, t in results if b == t)
            print(f"统计: {identical}/{len(results)} 个输出完全一致")
        else:
            print(f"已完成 {len(results)} 个推理测试 (无 baseline 对比)")
        print("=" * 80)
    
    return results


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    parser = parse_common_args("端到端推理输出对比")
    args = parser.parse_args()
    
    # 注意：这里不调用 apply_env_args(args)
    # 因为 run_comparison_inference 会自己管理环境变量
    
    # 打印环境信息
    EnvironmentChecker.print_env_info()
    
    # 查找模型
    model_path = ModelFinder.find_small_model("FP8")
    if model_path is None:
        print(Colors.red("错误: 未找到 FP8 模型"))
        sys.exit(1)
    
    # 根据参数决定测试的 SlideSparse 后端
    use_cublaslt = getattr(args, 'use_cublaslt', False)
    use_cusparselt = getattr(args, 'use_cusparselt', False)
    inner_fp32 = getattr(args, 'inner_fp32', False)
    
    # 运行输出对比
    run_comparison_inference(
        model_path=model_path,
        prompts=TEST_PROMPTS,
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_fp32=inner_fp32,
        max_tokens=64,
        verbose=True,
    )
    
    sys.exit(0)
