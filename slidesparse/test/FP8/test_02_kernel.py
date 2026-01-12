#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
test_02_kernel.py - SlideSparse Kernel 正确性测试

验证 SlideSparse FP8 GEMM kernel 的计算正确性：
1. 使用随机数据对比 vLLM 原生路径 和 SlideSparse 路径的输出
2. 测试不同矩阵尺寸 (M, N, K)
3. 验证 INNER_DTYPE_FP32 选项

测试流程:
    input (BF16) -> quant (FP8) -> GEMM -> dequant+bias -> output (BF16)
                                    ↑
                    baseline (vLLM 原生) vs test (SlideSparse)

使用方法:
    python3 test_02_kernel.py                        # 默认: vs CUTLASS fallback
    python3 test_02_kernel.py --use-cublaslt         # vs cuBLASLt
    python3 test_02_kernel.py --use-cublaslt --inner-fp32  # cuBLASLt + FP32
    python3 test_02_kernel.py --use-cusparselt       # vs cuSPARSELt (TODO)

对比说明:
    - baseline: vLLM 原生路径 (DISABLE_SLIDESPARSE=1)
    - test: SlideSparse 路径 (根据参数选择 kernel)
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
from dataclasses import dataclass

# 抑制 vLLM 日志
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

sys.path.insert(0, str(Path(__file__).parent))
from test_utils import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
    Colors,
    Benchmarker,
    cuda_memory_manager,
    skip_if_no_cuda,
    skip_if_no_fp8,
    parse_common_args,
    apply_env_args,
    get_backend_name,
    set_env_for_baseline,
    set_env_for_test,
    restore_env,
)

import torch


# ============================================================================
# 测试配置
# ============================================================================

@dataclass
class GEMMTestCase:
    """GEMM 测试用例"""
    name: str
    M: int
    N: int
    K: int
    
    @property
    def shape_str(self) -> str:
        return f"M={self.M}, N={self.N}, K={self.K}"


# 测试矩阵尺寸 - 覆盖不同场景
TEST_CASES = [
    # 小矩阵
    GEMMTestCase("Small (M=16)", M=16, N=896, K=896),
    GEMMTestCase("Small (M=32)", M=32, N=896, K=896),
    # 中等矩阵
    GEMMTestCase("Medium (M=128)", M=128, N=4096, K=4096),
    GEMMTestCase("Medium (M=256)", M=256, N=4096, K=4096),
    # 大矩阵
    GEMMTestCase("Large (M=1024)", M=1024, N=4096, K=4096),
    GEMMTestCase("Large (M=4096)", M=4096, N=4096, K=4096),
    # Qwen2.5-0.5B 典型尺寸
    GEMMTestCase("Qwen-0.5B QKV", M=64, N=896*3, K=896),
    GEMMTestCase("Qwen-0.5B FFN", M=64, N=4864, K=896),
    # Llama3.2-1B 典型尺寸
    GEMMTestCase("Llama-1B QKV", M=64, N=2048*3, K=2048),
    GEMMTestCase("Llama-1B FFN", M=64, N=8192, K=2048),
]


# ============================================================================
# 辅助函数
# ============================================================================

def get_fp8_dtype():
    """获取 FP8 数据类型"""
    return torch.float8_e4m3fn


def generate_test_data(
    M: int,
    N: int,
    K: int,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成测试数据
    
    Returns:
        input_bf16: [M, K] BF16 输入
        weight_fp8_t: [K, N] FP8 权重 (column-major)
        weight_scale: [N, 1] FP32 权重 scale
        bias: [N] BF16 偏置
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 1. 生成 BF16 输入
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # 2. 生成权重并量化为 FP8
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1
    
    fp8_max = torch.finfo(get_fp8_dtype()).max
    weight_absmax = weight_bf16.abs().max(dim=1, keepdim=True).values
    weight_scale = (weight_absmax / fp8_max).to(torch.float32)
    weight_scale = torch.clamp(weight_scale, min=1e-12)
    
    weight_scaled = weight_bf16.float() / weight_scale
    weight_fp8 = weight_scaled.to(get_fp8_dtype())  # [N, K]
    
    # 3. 转置为 column-major [K, N]
    weight_fp8_t = weight_fp8.t()
    
    # 4. 生成 bias
    bias = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01
    
    return input_bf16, weight_fp8_t, weight_scale, bias


def run_baseline(
    input_bf16: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """运行 vLLM 原生路径 (baseline)"""
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    op = Fp8LinearOp(
        act_quant_static=False,
        act_quant_group_shape=GroupShape.PER_TOKEN,
        pad_output=False,
    )
    
    return op.apply(
        input=input_bf16,
        weight=weight_fp8_t,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        input_scale=None,
        input_scale_ub=None,
        bias=bias,
    )


def run_slidesparse(
    input_bf16: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """运行 SlideSparse 路径（根据环境变量选择 kernel）"""
    from slidesparse.core.SlideSparseLinearMethod_FP8 import SlideSparseFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    op = SlideSparseFp8LinearOp(
        act_quant_static=False,
        act_quant_group_shape=GroupShape.PER_TOKEN,
    )
    
    return op.apply(
        input=input_bf16,
        weight=weight_fp8_t,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
        input_scale=None,
        input_scale_ub=None,
        bias=bias,
    )


def check_correctness(
    baseline_output: torch.Tensor,
    test_output: torch.Tensor,
    rtol: float = 0.1,
    atol: float = 0.1,
) -> Tuple[bool, float, float]:
    """
    检查输出正确性
    
    FP8 精度较低，误差在 5-10% 是正常的
    
    Returns:
        (is_match, max_diff, mean_diff)
    """
    diff = (test_output - baseline_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_match = torch.allclose(test_output, baseline_output, rtol=rtol, atol=atol)
    
    return is_match, max_diff, mean_diff


# ============================================================================
# 测试用例
# ============================================================================

@test_case("CUDA 可用性", skip_if=skip_if_no_cuda)
def test_cuda_available():
    """验证 CUDA 可用"""
    device = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    return True, f"{device} (sm_{cc[0]}{cc[1]})"


@test_case("FP8 支持", skip_if=skip_if_no_fp8)
def test_fp8_support():
    """验证 FP8 支持"""
    cc = torch.cuda.get_device_capability(0)
    return True, f"sm_{cc[0]}{cc[1]} >= sm_89"


@test_case("SlideSparseFp8LinearOp 基本功能", skip_if=skip_if_no_fp8)
def test_op_basic():
    """测试 Op 基本运行"""
    from slidesparse.core.SlideSparseLinearMethod_FP8 import SlideSparseFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    with cuda_memory_manager():
        M, N, K = 64, 512, 256
        input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(M, N, K)
        
        op = SlideSparseFp8LinearOp(
            act_quant_static=False,
            act_quant_group_shape=GroupShape.PER_TOKEN,
        )
        
        output = op.apply(
            input=input_bf16,
            weight=weight_fp8_t,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
        )
        
        assert output.shape == (M, N), f"输出形状错误: {output.shape}"
        assert output.dtype == torch.bfloat16, f"输出类型错误: {output.dtype}"
    
    return True, f"输出形状 {output.shape}, kernel={op._kernel_name}"


@test_case("单次正确性验证", skip_if=skip_if_no_fp8)
def test_single_correctness():
    """单次正确性测试（baseline vs test）"""
    with cuda_memory_manager():
        M, N, K = 128, 1024, 512
        input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(M, N, K)
        
        # 运行 baseline
        baseline_output = run_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
        
        # 运行 test
        test_output = run_slidesparse(input_bf16, weight_fp8_t, weight_scale, bias)
        
        is_match, max_diff, mean_diff = check_correctness(baseline_output, test_output)
    
    if is_match:
        return True, f"max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    else:
        return False, f"误差过大: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"


# ============================================================================
# 批量正确性测试
# ============================================================================

def run_batch_correctness_test(
    test_cases: List[GEMMTestCase],
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_fp32: bool = False,
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    """
    批量运行正确性测试
    
    Args:
        test_cases: 测试用例列表
        use_cublaslt: 测试组使用 cuBLASLt
        use_cusparselt: 测试组使用 cuSPARSELt
        inner_fp32: 使用 FP32 累加
        verbose: 是否打印详细信息
    
    Returns:
        (passed, total, results)
    """
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_fp32)
    results = []
    passed = 0
    
    if verbose:
        print("\n" + "=" * 100)
        print(Colors.bold(f"vLLM 原生 vs {backend_name} 正确性对比"))
        print("=" * 100)
        print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
              f"{'Max Diff':>10} | {'Mean Diff':>12} | {'Status':>8}")
        print("-" * 100)
    
    for case in test_cases:
        try:
            with cuda_memory_manager():
                # 生成测试数据
                input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(
                    case.M, case.N, case.K
                )
                
                # 1. 运行 baseline (vLLM 原生)
                saved = set_env_for_baseline()
                baseline_output = run_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
                restore_env(saved)
                
                # 2. 运行 test (SlideSparse)
                saved = set_env_for_test(use_cublaslt, use_cusparselt, inner_fp32)
                test_output = run_slidesparse(input_bf16, weight_fp8_t, weight_scale, bias)
                restore_env(saved)
                
                # 检查正确性
                is_match, max_diff, mean_diff = check_correctness(
                    baseline_output, test_output
                )
                
                result = {
                    "name": case.name,
                    "M": case.M,
                    "N": case.N,
                    "K": case.K,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "match": is_match,
                }
                results.append(result)
                
                if is_match:
                    passed += 1
                    status = Colors.green("PASS")
                else:
                    status = Colors.red("FAIL")
                
                if verbose:
                    print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                          f"{max_diff:>10.6f} | {mean_diff:>12.8f} | {status}")
                
        except Exception as e:
            results.append({
                "name": case.name,
                "error": str(e),
                "match": False,
            })
            if verbose:
                print(f"{case.name:<20} | {Colors.red('ERROR')}: {e}")
    
    if verbose:
        print("-" * 100)
        print(f"总计: {passed}/{len(test_cases)} 通过")
        print("=" * 100)
    
    return passed, len(test_cases), results


@test_case("批量正确性测试", skip_if=skip_if_no_fp8)
def test_batch_correctness():
    """批量正确性测试"""
    # 从环境变量获取当前配置
    use_cublaslt = EnvironmentChecker.is_cublaslt_enabled()
    use_cusparselt = EnvironmentChecker.is_cusparselt_enabled()
    inner_fp32 = EnvironmentChecker.is_inner_dtype_fp32()
    
    passed, total, results = run_batch_correctness_test(
        TEST_CASES, 
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_fp32=inner_fp32,
        verbose=True
    )
    
    if passed == total:
        return True, f"全部 {total} 个测试通过"
    else:
        return False, f"{total - passed}/{total} 个测试失败"


# ============================================================================
# 性能对比测试
# ============================================================================

def run_performance_comparison(
    test_cases: List[GEMMTestCase],
    use_cublaslt: bool = False,
    use_cusparselt: bool = False,
    inner_fp32: bool = False,
    warmup: int = 25,
    repeat: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """运行性能对比"""
    backend_name = get_backend_name(use_cublaslt, use_cusparselt, inner_fp32)
    results = []
    
    if verbose:
        print("\n" + "=" * 130)
        print(Colors.bold(f"vLLM 原生 vs {backend_name} 性能对比"))
        print("=" * 130)
        print(f"Warmup: {warmup}, Repeat: {repeat}")
        print("-" * 130)
        print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
              f"{'Baseline(ms)':>12} | {'Test(ms)':>12} | {'Speedup':>8} | {'Match':>6}")
        print("-" * 130)
    
    for case in test_cases:
        try:
            with cuda_memory_manager():
                input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(
                    case.M, case.N, case.K
                )
                
                # Baseline 性能
                saved = set_env_for_baseline()
                baseline_time, _ = Benchmarker.benchmark(
                    lambda: run_baseline(input_bf16, weight_fp8_t, weight_scale, bias),
                    warmup=warmup,
                    repeat=repeat,
                )
                baseline_output = run_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
                restore_env(saved)
                
                # Test 性能
                saved = set_env_for_test(use_cublaslt, use_cusparselt, inner_fp32)
                test_time, _ = Benchmarker.benchmark(
                    lambda: run_slidesparse(input_bf16, weight_fp8_t, weight_scale, bias),
                    warmup=warmup,
                    repeat=repeat,
                )
                test_output = run_slidesparse(input_bf16, weight_fp8_t, weight_scale, bias)
                restore_env(saved)
                
                # 正确性检查
                is_match, _, _ = check_correctness(baseline_output, test_output)
                
                speedup = baseline_time / test_time if test_time > 0 else 0
                
                result = {
                    "name": case.name,
                    "M": case.M,
                    "N": case.N,
                    "K": case.K,
                    "baseline_ms": baseline_time,
                    "test_ms": test_time,
                    "speedup": speedup,
                    "match": is_match,
                }
                results.append(result)
                
                match_str = Colors.green("✓") if is_match else Colors.red("✗")
                speedup_str = f"{speedup:.3f}x"
                if speedup > 1.02:
                    speedup_str = Colors.green(speedup_str)
                elif speedup < 0.98:
                    speedup_str = Colors.red(speedup_str)
                
                if verbose:
                    print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                          f"{baseline_time:>12.4f} | {test_time:>12.4f} | {speedup_str:>8} | {match_str:>6}")
                
        except Exception as e:
            results.append({
                "name": case.name,
                "error": str(e),
            })
            if verbose:
                print(f"{case.name:<20} | {Colors.red('ERROR')}: {e}")
    
    if verbose:
        print("-" * 130)
        # 计算平均 speedup
        valid_results = [r for r in results if "speedup" in r]
        if valid_results:
            avg_speedup = sum(r["speedup"] for r in valid_results) / len(valid_results)
            print(f"平均加速比: {avg_speedup:.3f}x")
        print("=" * 130)
    
    return results


@test_case("性能对比测试", skip_if=skip_if_no_fp8)
def test_performance_comparison():
    """性能对比测试"""
    # 从环境变量获取当前配置
    use_cublaslt = EnvironmentChecker.is_cublaslt_enabled()
    use_cusparselt = EnvironmentChecker.is_cusparselt_enabled()
    inner_fp32 = EnvironmentChecker.is_inner_dtype_fp32()
    
    results = run_performance_comparison(
        TEST_CASES, 
        use_cublaslt=use_cublaslt,
        use_cusparselt=use_cusparselt,
        inner_fp32=inner_fp32,
        warmup=10, 
        repeat=50
    )
    
    valid_results = [r for r in results if "speedup" in r]
    if valid_results:
        avg_speedup = sum(r["speedup"] for r in valid_results) / len(valid_results)
        return True, f"平均加速比 {avg_speedup:.3f}x"
    return True, "测试完成"


# ============================================================================
# 主函数
# ============================================================================

def get_all_tests():
    """获取所有测试"""
    return [
        test_cuda_available,
        test_fp8_support,
        test_op_basic,
        test_single_correctness,
        test_batch_correctness,
        test_performance_comparison,
    ]


def run_tests(verbose: bool = True) -> bool:
    """运行所有测试"""
    tests = get_all_tests()
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("SlideSparse Kernel 正确性测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    parser = parse_common_args("SlideSparse Kernel 正确性测试")
    args = parser.parse_args()
    
    apply_env_args(args)
    
    success = run_tests(verbose=True)
    
    sys.exit(0 if success else 1)
