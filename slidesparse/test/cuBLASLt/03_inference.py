#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
03_inference.py - 端到端推理输出对比

对于相同的 prompt，分别用 vLLM 原生路径 和 SlideSparse 后端运行推理，
并排打印输出让用户直观比较精度差异。

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
    python3 03_inference.py                          # 默认: vLLM 原生 vs SlideSparse+外挂CUTLASS
    python3 03_inference.py --use-cublaslt           # vLLM 原生 vs SlideSparse+cuBLASLt
    python3 03_inference.py --use-cublaslt --inner-fp32  # cuBLASLt + FP32 中间累加

三种测试路径:
    1. vLLM 原生路径 (baseline): DISABLE_SLIDESPARSE=1
    2. SlideSparse + 外挂 CUTLASS (对照组): 默认
    3. SlideSparse + cuBLASLt (实验组): USE_CUBLASLT=1
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    EnvironmentChecker,
    ModelFinder,
    Colors,
    cuda_memory_manager,
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


# ============================================================================
# 核心功能：vLLM 原生路径 vs SlideSparse 后端 输出对比
# ============================================================================

def get_backend_name(use_cublaslt: bool, inner_fp32: bool) -> str:
    """获取后端名称"""
    if use_cublaslt:
        if inner_fp32:
            return "SlideSparse + cuBLASLt (FP32累加)"
        return "SlideSparse + cuBLASLt"
    else:
        return "SlideSparse + 外挂 CUTLASS"


def run_comparison_inference(
    model_path: Path,
    prompts: List[str],
    use_cublaslt: bool,
    inner_fp32: bool,
    max_tokens: int = 48,
    verbose: bool = True,
) -> List[Tuple[str, str, str]]:
    """
    运行 vLLM 原生路径 和 SlideSparse 后端推理并对比输出
    
    Args:
        model_path: 模型路径
        prompts: 提示词列表
        use_cublaslt: SlideSparse 后端是否使用 cuBLASLt (False=外挂CUTLASS)
        inner_fp32: 是否使用 FP32 中间累加
        max_tokens: 最大生成 token 数
        verbose: 是否打印详细信息
    
    Returns:
        List of (prompt, baseline_output, test_output)
    """
    from vllm import LLM, SamplingParams
    
    results = []
    backend_name = get_backend_name(use_cublaslt, inner_fp32)
    
    sampling_params = SamplingParams(
        temperature=0.0,  # 贪婪采样确保可复现
        max_tokens=max_tokens,
    )
    
    if verbose:
        print("\n" + "=" * 80)
        print(Colors.bold("vLLM 原生路径 vs SlideSparse 推理输出对比"))
        print("=" * 80)
        print(f"模型: {model_path.name}")
        print(f"基准: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)")
        print(f"测试: {backend_name}")
        print(f"采样: temperature=0.0, max_tokens={max_tokens}")
        print("=" * 80)
    
    # 1. 运行 vLLM 原生路径 (基准)
    #    设置 DISABLE_SLIDESPARSE=1，SlideSparse hook 不生效，走 vLLM 原生路径
    if verbose:
        print(f"\n{Colors.cyan('[1/2] 运行 vLLM 原生路径 (基准)...')}")
    
    # 保存原环境变量
    old_disable = os.environ.get("DISABLE_SLIDESPARSE")
    old_cublaslt = os.environ.get("USE_CUBLASLT")
    old_inner_fp32 = os.environ.get("INNER_DTYPE_FP32")
    
    # 基准: 禁用 SlideSparse hook
    os.environ["DISABLE_SLIDESPARSE"] = "1"
    os.environ.pop("USE_CUBLASLT", None)
    os.environ.pop("INNER_DTYPE_FP32", None)
    
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
    
    # 2. 运行 SlideSparse 后端 (测试)
    if verbose:
        print(f"\n{Colors.cyan(f'[2/2] 运行 {backend_name}...')}")
    
    # 启用 SlideSparse
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    # 设置 SlideSparse 后端参数
    if use_cublaslt:
        os.environ["USE_CUBLASLT"] = "1"
        if inner_fp32:
            os.environ["INNER_DTYPE_FP32"] = "1"
        else:
            os.environ.pop("INNER_DTYPE_FP32", None)
    else:
        # 外挂 CUTLASS
        os.environ["USE_CUBLASLT"] = "0"
        os.environ.pop("INNER_DTYPE_FP32", None)
    
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
    
    # 恢复环境变量
    if old_disable is not None:
        os.environ["DISABLE_SLIDESPARSE"] = old_disable
    else:
        os.environ.pop("DISABLE_SLIDESPARSE", None)
    if old_cublaslt is not None:
        os.environ["USE_CUBLASLT"] = old_cublaslt
    else:
        os.environ.pop("USE_CUBLASLT", None)
    if old_inner_fp32 is not None:
        os.environ["INNER_DTYPE_FP32"] = old_inner_fp32
    else:
        os.environ.pop("INNER_DTYPE_FP32", None)
    
    # 3. 打印对比结果
    if verbose:
        print("\n" + "=" * 80)
        print(Colors.bold("输出对比"))
        print("=" * 80)
        
        for i, (prompt, baseline_out, test_out) in enumerate(
            zip(prompts, baseline_texts, test_texts)
        ):
            print(f"\n{Colors.bold(f'[Prompt {i+1}]')} {prompt}")
            print("-" * 80)
            print(f"{Colors.blue('vLLM 原生:')} {baseline_out}")
            print()
            print(f"{Colors.green(f'{backend_name}:')} {test_out}")
            
            results.append((prompt, baseline_out, test_out))
        
        print("\n" + "=" * 80)
    
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
    # --use-cublaslt: 测试 cuBLASLt kernel
    # 默认: 测试外挂 CUTLASS (USE_CUBLASLT=0)
    use_cublaslt = getattr(args, 'use_cublaslt', False)
    inner_fp32 = getattr(args, 'inner_fp32', False)
    
    # 运行输出对比
    run_comparison_inference(
        model_path=model_path,
        prompts=TEST_PROMPTS,
        use_cublaslt=use_cublaslt,
        inner_fp32=inner_fp32,
        max_tokens=64,
        verbose=True,
    )
    
    sys.exit(0)
