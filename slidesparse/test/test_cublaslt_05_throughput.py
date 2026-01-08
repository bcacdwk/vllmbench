#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 5: 吞吐量测试

对比 cuBLASLt 后端与原生 cutlass 后端的吞吐量。

测试覆盖:
- 不同 batch size
- 不同矩阵尺寸
- 典型 LLM 层尺寸
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from slidesparse.test.test_base import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
    Benchmarker,
    cuda_memory_manager,
    skip_if_no_cuda,
    skip_if_no_fp8,
)


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class GEMMTestCase:
    """GEMM 测试用例"""
    name: str
    M: int  # batch size
    K: int  # input dim
    N: int  # output dim
    
    @property
    def flops(self) -> int:
        """计算 FLOPS"""
        return 2 * self.M * self.K * self.N


# 标准测试用例
STANDARD_TEST_CASES = [
    GEMMTestCase("单 token (4K x 4K)", M=1, K=4096, N=4096),
    GEMMTestCase("小 batch (4K x 4K)", M=16, K=4096, N=4096),
    GEMMTestCase("中 batch (4K x 4K)", M=64, K=4096, N=4096),
]

# LLM 典型层尺寸 (只测试 Qwen2.5-0.5B 和 Llama3.2-1B)
LLM_TEST_CASES = [
    GEMMTestCase("Qwen2.5-0.5B QKV", M=32, K=896, N=896*3),
    GEMMTestCase("Qwen2.5-0.5B FFN Up", M=32, K=896, N=4864),
    GEMMTestCase("Llama3.2-1B QKV", M=32, K=2048, N=2048*3),
    GEMMTestCase("Llama3.2-1B FFN Up", M=32, K=2048, N=8192),
]


# ============================================================================
# 辅助函数
# ============================================================================

def create_fp8_weight(K: int, N: int, device):
    """创建符合 CUTLASS 要求的 FP8 权重"""
    import torch
    
    # 创建 [N, K] 的 float32 张量，转为 FP8，然后转置得到 [K, N] 列优先
    weight_float = torch.randn(N, K, dtype=torch.float32, device=device)
    weight = weight_float.to(torch.float8_e4m3fn).T  # shape=[K, N], 列优先
    
    return weight


def run_single_benchmark(case: GEMMTestCase, warmup: int = 10, repeat: int = 100) -> Dict[str, Any]:
    """运行单个 benchmark"""
    import torch
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    device = torch.device("cuda:0")
    
    # 创建输入和权重
    input_tensor = torch.randn(case.M, case.K, dtype=torch.bfloat16, device=device)
    weight = create_fp8_weight(case.K, case.N, device)
    weight_scale = torch.ones(1, dtype=torch.float32, device=device)
    
    # 创建 Op
    fp8_op = Fp8LinearOp(act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
    cublaslt_op = CuBLASLtFp8LinearOp(act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
    
    # Benchmark 函数
    def run_fp8():
        return fp8_op.apply(
            input=input_tensor,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )
    
    def run_cublaslt():
        return cublaslt_op.apply(
            input=input_tensor,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )
    
    # 运行 benchmark
    fp8_time, fp8_std = Benchmarker.benchmark(run_fp8, warmup=warmup, repeat=repeat)
    cublaslt_time, cublaslt_std = Benchmarker.benchmark(run_cublaslt, warmup=warmup, repeat=repeat)
    
    # 计算指标
    fp8_tflops = Benchmarker.compute_tflops(case.flops, fp8_time)
    cublaslt_tflops = Benchmarker.compute_tflops(case.flops, cublaslt_time)
    speedup = fp8_time / cublaslt_time
    
    return {
        "name": case.name,
        "M": case.M,
        "K": case.K,
        "N": case.N,
        "fp8_time": fp8_time,
        "fp8_std": fp8_std,
        "fp8_tflops": fp8_tflops,
        "cublaslt_time": cublaslt_time,
        "cublaslt_std": cublaslt_std,
        "cublaslt_tflops": cublaslt_tflops,
        "speedup": speedup,
    }


# ============================================================================
# 测试用例
# ============================================================================

@test_case("GEMM 基准测试", skip_if=skip_if_no_fp8)
def test_gemm_benchmark():
    """运行 GEMM 基准测试"""
    results = []
    
    with cuda_memory_manager():
        for case in STANDARD_TEST_CASES:
            try:
                result = run_single_benchmark(case)
                results.append(result)
            except Exception as e:
                results.append({
                    "name": case.name,
                    "error": str(e),
                })
    
    # 计算平均加速比
    valid_results = [r for r in results if "speedup" in r]
    if valid_results:
        avg_speedup = sum(r["speedup"] for r in valid_results) / len(valid_results)
    else:
        avg_speedup = 0
    
    return True, f"平均加速比 {avg_speedup:.3f}x", {"results": results}


@test_case("LLM 层尺寸测试", skip_if=skip_if_no_fp8)
def test_llm_layer_sizes():
    """测试 LLM 典型层尺寸"""
    results = []
    
    with cuda_memory_manager():
        for case in LLM_TEST_CASES:  # 测试所有用例
            try:
                result = run_single_benchmark(case, warmup=5, repeat=50)
                results.append(result)
            except Exception as e:
                results.append({
                    "name": case.name,
                    "error": str(e),
                })
    
    successful = sum(1 for r in results if "speedup" in r)
    
    return True, f"{successful}/{len(results)} 测试成功", {"results": results}


@test_case("性能一致性验证", skip_if=skip_if_no_fp8)
def test_performance_consistency():
    """验证性能一致性 (当前阶段两者应该接近)"""
    case = GEMMTestCase("一致性测试", M=64, K=4096, N=4096)
    
    with cuda_memory_manager():
        result = run_single_benchmark(case, warmup=20, repeat=200)
    
    speedup = result["speedup"]
    
    # 当前阶段使用相同后端，加速比应该接近 1.0
    # 允许 5% 的误差
    if 0.95 <= speedup <= 1.05:
        return True, f"加速比 {speedup:.3f}x (在预期范围内)"
    else:
        return TestResult(
            name="性能一致性验证",
            status=TestStatus.WARNING,
            message=f"加速比 {speedup:.3f}x (偏差较大)"
        )


# ============================================================================
# 结果格式化
# ============================================================================

def format_benchmark_table(results: List[Dict]) -> str:
    """格式化 benchmark 结果为表格"""
    lines = [
        "-" * 80,
        f"{'测试用例':<25} {'原生(ms)':<12} {'cuBLASLt(ms)':<12} {'加速比':<10} {'TFLOPS':<10}",
        "-" * 80,
    ]
    
    for r in results:
        if "error" in r:
            lines.append(f"{r['name']:<25} ERROR: {r['error']}")
        else:
            lines.append(
                f"{r['name']:<25} "
                f"{r['fp8_time']:<12.3f} "
                f"{r['cublaslt_time']:<12.3f} "
                f"{r['speedup']:<10.3f} "
                f"{r['cublaslt_tflops']:<10.1f}"
            )
    
    lines.append("-" * 80)
    
    return "\n".join(lines)


# ============================================================================
# 主函数
# ============================================================================

def run_tests(verbose: bool = True, full: bool = False) -> bool:
    """运行所有吞吐量测试"""
    tests = [
        test_gemm_benchmark,
        test_llm_layer_sizes,
        test_performance_consistency,
    ]
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("吞吐量测试", verbose=verbose)
    result = runner.run_all(tests)
    
    # 打印详细结果
    if verbose:
        for test_result in result.results:
            if test_result.details and "results" in test_result.details:
                print("\n" + format_benchmark_table(test_result.details["results"]))
    
    return result.success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="吞吐量测试")
    parser.add_argument("-q", "--quiet", action="store_true", help="静默模式")
    parser.add_argument("--full", action="store_true", help="运行完整测试")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()
    
    if args.json:
        import json
        
        # 收集结果
        all_results = []
        for case in STANDARD_TEST_CASES:
            try:
                with cuda_memory_manager():
                    result = run_single_benchmark(case)
                all_results.append(result)
            except Exception as e:
                all_results.append({"name": case.name, "error": str(e)})
        
        print(json.dumps({"benchmarks": all_results}, indent=2))
    else:
        success = run_tests(verbose=not args.quiet, full=args.full)
        sys.exit(0 if success else 1)
