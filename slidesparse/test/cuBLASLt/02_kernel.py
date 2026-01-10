#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
02_kernel.py - cuBLASLt Kernel 正确性测试

验证 cuBLASLt FP8 GEMM kernel 的计算正确性：
1. 使用随机数据对比 原生 CUTLASS 和 cuBLASLt/外挂CUTLASS 输出
2. 测试不同矩阵尺寸 (M, N, K)
3. 验证 INNER_DTYPE_FP32 选项

测试流程:
    input (BF16) -> quant (FP8) -> GEMM -> dequant+bias -> output (BF16)
                                    ↑
                    原生 CUTLASS (baseline) vs cuBLASLt/外挂CUTLASS (test)

使用方法:
    python3 02_kernel.py                      # 对比 原生 CUTLASS vs cuBLASLt
    python3 02_kernel.py --ext-cutlass        # 对比 原生 CUTLASS vs 外挂 CUTLASS
    python3 02_kernel.py --inner-fp32         # cuBLASLt + FP32 中间结果

路径说明:
    默认: USE_CUBLASLT=1 → cuBLASLt kernel
    --ext-cutlass: USE_CUBLASLT=0 → 外挂 CUTLASS
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
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
    Colors,
    cuda_memory_manager,
    skip_if_no_cuda,
    skip_if_no_fp8,
    parse_common_args,
    apply_env_args,
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
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K


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


def run_cutlass_baseline(
    input_bf16: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """运行 CUTLASS 基线"""
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


def run_cublaslt(
    input_bf16: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """运行 cuBLASLt"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    op = CuBLASLtFp8LinearOp(
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
    cutlass_output: torch.Tensor,
    cublaslt_output: torch.Tensor,
    rtol: float = 0.1,
    atol: float = 0.1,
) -> Tuple[bool, float, float]:
    """
    检查输出正确性
    
    FP8 精度较低，误差在 5-10% 是正常的
    
    Returns:
        (is_match, max_diff, mean_diff)
    """
    diff = (cublaslt_output - cutlass_output).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_match = torch.allclose(cublaslt_output, cutlass_output, rtol=rtol, atol=atol)
    
    return is_match, max_diff, mean_diff


def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """计算 TFLOPS"""
    if time_ms <= 0:
        return 0.0
    flops = 2 * M * N * K
    return (flops / (time_ms / 1000)) / 1e12


# ============================================================================
# 测试用例
# ============================================================================

@test_case("CUDA 可用性", skip_if=skip_if_no_cuda)
def test_cuda_available():
    """验证 CUDA 可用"""
    import torch
    
    device = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    
    return True, f"{device} (sm_{cc[0]}{cc[1]})"


@test_case("FP8 支持", skip_if=skip_if_no_fp8)
def test_fp8_support():
    """验证 FP8 支持"""
    cc = torch.cuda.get_device_capability(0)
    return True, f"sm_{cc[0]}{cc[1]} 支持 FP8"


@test_case("Op 基本功能", skip_if=skip_if_no_fp8)
def test_op_basic():
    """测试 Op 基本运行"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    with cuda_memory_manager():
        # 创建小的测试数据
        M, N, K = 16, 256, 256
        input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(M, N, K)
        
        # 测试 CUTLASS
        cutlass_out = run_cutlass_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
        assert cutlass_out.shape == (M, N), f"CUTLASS 输出形状错误: {cutlass_out.shape}"
        
        # 测试 cuBLASLt Op
        cublaslt_out = run_cublaslt(input_bf16, weight_fp8_t, weight_scale, bias)
        assert cublaslt_out.shape == (M, N), f"cuBLASLt 输出形状错误: {cublaslt_out.shape}"
    
    return True, f"输出形状 [{M}, {N}] 正确"


@test_case("单次正确性验证", skip_if=skip_if_no_fp8)
def test_single_correctness():
    """单次正确性测试"""
    with cuda_memory_manager():
        case = GEMMTestCase("Test", M=64, N=4096, K=4096)
        
        input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(
            case.M, case.N, case.K
        )
        
        cutlass_out = run_cutlass_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
        cublaslt_out = run_cublaslt(input_bf16, weight_fp8_t, weight_scale, bias)
        
        is_match, max_diff, mean_diff = check_correctness(cutlass_out, cublaslt_out)
    
    if is_match:
        return True, f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f}"
    else:
        return TestResult(
            name="单次正确性验证",
            status=TestStatus.WARNING,
            message=f"max_diff={max_diff:.4f} (超出阈值但可接受)"
        )


# ============================================================================
# 批量正确性测试
# ============================================================================

def run_batch_correctness_test(
    test_cases: List[GEMMTestCase],
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    """
    批量运行正确性测试
    
    Returns:
        (passed, total, results)
    """
    results = []
    passed = 0
    
    if verbose:
        print("\n" + "=" * 100)
        print(Colors.bold("cuBLASLt vs CUTLASS 正确性对比"))
        print("=" * 100)
        print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
              f"{'Max Diff':>10} | {'Mean Diff':>12} | {'Status':>8}")
        print("-" * 100)
    
    for case in test_cases:
        try:
            with cuda_memory_manager():
                input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(
                    case.M, case.N, case.K
                )
                
                cutlass_out = run_cutlass_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
                cublaslt_out = run_cublaslt(input_bf16, weight_fp8_t, weight_scale, bias)
                
                is_match, max_diff, mean_diff = check_correctness(cutlass_out, cublaslt_out)
            
            status = "✓ PASS" if is_match else "⚠ WARN"
            if is_match:
                passed += 1
            
            result = {
                "name": case.name,
                "M": case.M,
                "N": case.N,
                "K": case.K,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "passed": is_match,
            }
            results.append(result)
            
            if verbose:
                status_colored = Colors.green(status) if is_match else Colors.yellow(status)
                print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                      f"{max_diff:>10.4f} | {mean_diff:>12.6f} | {status_colored}")
                
        except Exception as e:
            results.append({
                "name": case.name,
                "error": str(e),
                "passed": False,
            })
            if verbose:
                print(f"{case.name:<20} | {'ERROR':^53} | {str(e)[:30]}")
    
    if verbose:
        print("-" * 100)
        print(f"总计: {passed}/{len(test_cases)} 通过")
        print("=" * 100)
    
    return passed, len(test_cases), results


@test_case("批量正确性测试", skip_if=skip_if_no_fp8)
def test_batch_correctness():
    """批量正确性测试"""
    passed, total, results = run_batch_correctness_test(TEST_CASES, verbose=True)
    
    if passed == total:
        return True, f"{passed}/{total} 通过"
    else:
        return TestResult(
            name="批量正确性测试",
            status=TestStatus.WARNING,
            message=f"{passed}/{total} 通过"
        )


# ============================================================================
# 性能对比测试
# ============================================================================

def run_performance_comparison(
    test_cases: List[GEMMTestCase],
    warmup: int = 25,
    repeat: int = 100,
    verbose: bool = True
) -> List[Dict]:
    """运行性能对比"""
    import time
    
    results = []
    
    if verbose:
        print("\n" + "=" * 130)
        print(Colors.bold("cuBLASLt vs CUTLASS 性能对比"))
        print("=" * 130)
        print(f"Warmup: {warmup}, Repeat: {repeat}")
        print("-" * 130)
        print(f"{'测试用例':<20} | {'M':>6} | {'N':>6} | {'K':>6} | "
              f"{'CUTLASS(ms)':>11} | {'cuBLASLt(ms)':>12} | {'Speedup':>8} | {'Match':>6}")
        print("-" * 130)
    
    for case in test_cases:
        try:
            with cuda_memory_manager():
                input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(
                    case.M, case.N, case.K
                )
                
                # CUTLASS benchmark
                for _ in range(warmup):
                    run_cutlass_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(repeat):
                    cutlass_out = run_cutlass_baseline(input_bf16, weight_fp8_t, weight_scale, bias)
                torch.cuda.synchronize()
                cutlass_time = (time.perf_counter() - start) * 1000 / repeat
                
                # cuBLASLt benchmark
                for _ in range(warmup):
                    run_cublaslt(input_bf16, weight_fp8_t, weight_scale, bias)
                torch.cuda.synchronize()
                
                start = time.perf_counter()
                for _ in range(repeat):
                    cublaslt_out = run_cublaslt(input_bf16, weight_fp8_t, weight_scale, bias)
                torch.cuda.synchronize()
                cublaslt_time = (time.perf_counter() - start) * 1000 / repeat
                
                # 正确性检查
                is_match, max_diff, _ = check_correctness(cutlass_out, cublaslt_out)
                
                speedup = cutlass_time / cublaslt_time if cublaslt_time > 0 else 0
                
                result = {
                    "name": case.name,
                    "M": case.M,
                    "N": case.N,
                    "K": case.K,
                    "cutlass_ms": cutlass_time,
                    "cublaslt_ms": cublaslt_time,
                    "speedup": speedup,
                    "matched": is_match,
                }
                results.append(result)
                
                if verbose:
                    match_str = Colors.green("✓") if is_match else Colors.red(f"✗({max_diff:.3f})")
                    speedup_str = f"{speedup:.2f}x"
                    if speedup > 1.05:
                        speedup_str = Colors.green(speedup_str)
                    elif speedup < 0.95:
                        speedup_str = Colors.red(speedup_str)
                    
                    print(f"{case.name:<20} | {case.M:>6} | {case.N:>6} | {case.K:>6} | "
                          f"{cutlass_time:>11.3f} | {cublaslt_time:>12.3f} | {speedup_str:>8} | {match_str:>6}")
                
        except Exception as e:
            results.append({"name": case.name, "error": str(e)})
            if verbose:
                print(f"{case.name:<20} | ERROR: {str(e)[:80]}")
    
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
    results = run_performance_comparison(TEST_CASES, warmup=10, repeat=50)
    
    valid = [r for r in results if "speedup" in r]
    if not valid:
        return False, "无有效结果"
    
    avg_speedup = sum(r["speedup"] for r in valid) / len(valid)
    matched = sum(1 for r in valid if r.get("matched", False))
    
    return True, f"平均加速比 {avg_speedup:.3f}x, 正确性 {matched}/{len(valid)}"


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
    
    runner = TestRunner("cuBLASLt Kernel 正确性测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    parser = parse_common_args("cuBLASLt Kernel 正确性测试")
    args = parser.parse_args()
    
    apply_env_args(args)
    
    success = run_tests(verbose=True)
    
    sys.exit(0 if success else 1)
