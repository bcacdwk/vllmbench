#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dequant + Bias Kernel Benchmark

Usage:
    python3 run_benchmark.py --dtype bf16   # Test BF16 input
    python3 run_benchmark.py --dtype fp32   # Test FP32 input
"""

import sys
import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import torch
import triton
import triton.language as tl
import triton.testing as testing

# 设置路径以导入 slidesparse 模块
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent.parent  # slidesparse/
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent       # vllmbench/

# 将项目根目录添加到 sys.path以支持 "from slidesparse.utils import ..."
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    build_filename,
    get_gpu_cc,
    load_module,
)


# =============================================================================
# Test Configuration
# =============================================================================

M_VALUES = [16, 128, 1024, 4096, 16384, 65536]
N_VALUES = [2560, 3840, 13824]

WARMUP = 25
REP = 100


# =============================================================================
# Theoretical Baseline (pure memory copy with tuned config)
# =============================================================================

@triton.jit
def _memcpy_kernel(
    in_ptr, out_ptr, M, N,
    stride_im, stride_in, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    in_offs = offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    val = tl.load(in_ptr + in_offs, mask=mask, other=0.0)
    
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=mask)


def make_theoretical_baseline(get_config_func):
    """Create theoretical baseline function that uses tuned config"""
    
    def theoretical_baseline(input_tensor: torch.Tensor) -> torch.Tensor:
        """Pure memory copy baseline using tuned BLOCK_M, BLOCK_N, num_warps, num_stages"""
        M, N = input_tensor.shape
        output = torch.empty((M, N), dtype=torch.bfloat16, device=input_tensor.device)
        
        # Use the same config as tuned kernel
        BLOCK_M, BLOCK_N, num_warps, num_stages = get_config_func(M, N)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        _memcpy_kernel[grid](
            input_tensor, output, M, N,
            input_tensor.stride(0), input_tensor.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=num_warps, num_stages=num_stages,
        )
        return output
    
    return theoretical_baseline


# =============================================================================
# Load Kernels
# =============================================================================

def load_tuned_module() -> ModuleType | None:
    """Load the auto-tuned kernel module from build/"""
    build_dir = Path(__file__).parent / "build"
    
    try:
        module = load_module("dequant_bias_tuned", search_dir=build_dir, ext=".py")
        return module
    except FileNotFoundError:
        filename = build_filename("dequant_bias_tuned", ext=".py")
        print(f"ERROR: Tuned kernel not found: {build_dir / filename}")
        print(f"Please run: python3 autotune_autogen_dequant_bias.py")
        return None


def load_basic_module() -> ModuleType | None:
    """Load the basic kernel module (contains untuned kernel and pytorch reference)"""
    module_path = Path(__file__).parent / "basic_dequant_bias_triton.py"
    
    if not module_path.exists():
        print(f"ERROR: Basic kernel not found: {module_path}")
        return None
    
    # 使用 importlib 加载本地文件
    spec = importlib.util.spec_from_file_location("basic_kernel", module_path)
    if spec is None or spec.loader is None:
        print(f"ERROR: Failed to load module: {module_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Correctness Test
# =============================================================================

def test_correctness(tuned_func, untuned_func, pytorch_ref_func, input_dtype: torch.dtype):
    """Test correctness against PyTorch reference"""
    print("\n" + "=" * 70)
    print("Correctness Test")
    print("=" * 70)
    
    torch.manual_seed(42)
    all_passed = True
    
    # Sample shapes to avoid OOM (skip large M * N combinations)
    test_shapes = [(M, N) for M in M_VALUES for N in N_VALUES if M * N <= 64 * 1024 * 1024]
    
    for M, N in test_shapes:
        gemm = torch.randn(M, N, dtype=input_dtype, device='cuda')
        scale_a = torch.rand(M, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        scale_b = torch.rand(N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
        
        ref = pytorch_ref_func(gemm, scale_a, scale_b, bias, torch.bfloat16)
        out_tuned = tuned_func(gemm, scale_a, scale_b, bias)
        out_untuned = untuned_func(gemm, scale_a, scale_b, bias)
        
        diff_tuned = (ref.float() - out_tuned.float()).abs().max().item()
        diff_untuned = (ref.float() - out_untuned.float()).abs().max().item()
        
        passed = diff_tuned < 1e-2 and diff_untuned < 1e-2
        all_passed = all_passed and passed
        
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] M={M:<6} N={N:<6} tuned_diff={diff_tuned:.6f} untuned_diff={diff_untuned:.6f}")
        
        # Free memory
        del gemm, scale_a, scale_b, bias, ref, out_tuned, out_untuned
        torch.cuda.empty_cache()
    
    print("-" * 70)
    print(f"Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    return all_passed


# =============================================================================
# Throughput Benchmark
# =============================================================================

def run_benchmark(tuned_func, untuned_func, baseline_func, input_dtype: torch.dtype):
    """Run throughput benchmark: theoretical vs untuned vs tuned"""
    print("\n" + "=" * 70)
    print("Throughput Benchmark")
    print("=" * 70)
    dtype_name = "FP32" if input_dtype == torch.float32 else "BF16"
    print(f"Input: {dtype_name}, Output: BF16, Warmup: {WARMUP}, Rep: {REP}")
    print(f"Baseline: memcpy kernel using same tuned (BLOCK_M, BLOCK_N, warps, stages)")
    print()
    
    # Header
    print(f"{'M':<7} {'N':<7} | {'Baseline(us)':<12} {'Untuned(us)':<12} {'Tuned(us)':<11} | {'vs Base':<10} {'vs Untuned':<10}")
    print("-" * 90)
    
    results: list[dict] = []
    
    for M in M_VALUES:
        for N in N_VALUES:
            # Generate data
            gemm = torch.randn(M, N, dtype=input_dtype, device='cuda')
            scale_a = torch.rand(M, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            scale_b = torch.rand(N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
            
            # Benchmark baseline (pure memory copy with tuned config)
            t_baseline: float = testing.do_bench(
                lambda: baseline_func(gemm),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # type: ignore  # ms -> us
            
            # Benchmark untuned kernel
            t_untuned: float = testing.do_bench(
                lambda: untuned_func(gemm, scale_a, scale_b, bias),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # type: ignore  # ms -> us
            
            # Benchmark tuned kernel
            t_tuned: float = testing.do_bench(
                lambda: tuned_func(gemm, scale_a, scale_b, bias),
                warmup=WARMUP, rep=REP, return_mode="min"
            ) * 1000  # type: ignore  # ms -> us
            
            # Speedup ratios
            speedup_vs_baseline = t_baseline / t_tuned  # <1 expected (tuned slower than memcpy)
            speedup_vs_untuned = t_untuned / t_tuned  # >1 means tuned is faster
            
            results.append({
                'M': M, 'N': N,
                'baseline': t_baseline,
                'untuned': t_untuned,
                'tuned': t_tuned,
                'speedup_baseline': speedup_vs_baseline,
                'speedup_untuned': speedup_vs_untuned,
            })
            
            print(f"{M:<7} {N:<7} | {t_baseline:<12.2f} {t_untuned:<12.2f} {t_tuned:<11.2f} | {speedup_vs_baseline:<10.2f} {speedup_vs_untuned:<10.2f}")
            
            # Free memory
            del gemm, scale_a, scale_b, bias
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "-" * 90)
    print("Summary:")
    avg_baseline = sum(r['speedup_baseline'] for r in results) / len(results)
    avg_untuned = sum(r['speedup_untuned'] for r in results) / len(results)
    max_untuned = max(r['speedup_untuned'] for r in results)
    min_untuned = min(r['speedup_untuned'] for r in results)
    print(f"  vs Baseline: Avg {avg_baseline:.2f}x  (expected <1, tuned does compute, baseline only memcpy)")
    print(f"  vs Untuned:  Avg {avg_untuned:.2f}x  Min {min_untuned:.2f}x  Max {max_untuned:.2f}x")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dequant + Bias Kernel Benchmark")
    parser.add_argument('--dtype', type=str, required=True, choices=['bf16', 'fp32'],
                        help='Input dtype: bf16 or fp32')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    input_dtype = torch.float32 if args.dtype == 'fp32' else torch.bfloat16
    
    print("=" * 70)
    print("Dequant + Bias Kernel Benchmark")
    print("=" * 70)
    print(f"GPU:     {torch.cuda.get_device_name()} ({get_gpu_cc()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Input:   {args.dtype.upper()}")
    print(f"Kernel:  {build_filename('dequant_bias_tuned', ext='.py')}")
    
    # Load tuned module (supports both BF16 and FP32 input)
    tuned_module = load_tuned_module()
    if tuned_module is None:
        return 1
    
    tuned_func = tuned_module.dequant_bias_triton
    get_config_func = tuned_module._get_config
    
    # Create baseline using tuned config
    baseline_func = make_theoretical_baseline(get_config_func)
    
    # Load basic module
    basic_module = load_basic_module()
    if basic_module is None:
        return 1
    
    untuned_func = basic_module.dequant_bias_triton
    pytorch_ref_func = basic_module.dequant_bias_pytorch
    
    # Correctness test
    if not test_correctness(tuned_func, untuned_func, pytorch_ref_func, input_dtype):
        print("\nERROR: Correctness test failed!")
        return 1
    
    # Throughput benchmark
    run_benchmark(tuned_func, untuned_func, baseline_func, input_dtype)
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
