#!/usr/bin/env python3
"""
Dequant + Bias Triton Kernel 性能测试

用法:
  python3 run_benchmark.py                  # 使用 autotune 版本（预热时自动调优）
  python3 run_benchmark.py --tuned          # 使用已调优的固定配置版本
  python3 run_benchmark.py --correctness    # 只运行正确性测试
  python3 run_benchmark.py --dtype bf16     # 只测试 BF16 输入
  python3 run_benchmark.py --dtype fp32     # 只测试 FP32 输入
"""

import torch
import triton.testing as testing
import argparse
import sys
import os
from pathlib import Path
import importlib.util


# =============================================================================
# 理论带宽计算
# =============================================================================

# H100 PCIe 峰值内存带宽 (GB/s)
GPU_PEAK_BANDWIDTH_GBPS = 2000.0


def calculate_theoretical_time_us(M: int, N: int, input_dtype: torch.dtype, bandwidth_gbps: float) -> float:
    """
    计算理论最小执行时间 (微秒)
    
    内存访问:
    - 读: gemm_output [M, N] (input_dtype)
    - 读: scale_a [M, 1] (FP32 = 4 bytes)
    - 读: scale_b [1, N] (FP32 = 4 bytes)
    - 读: bias [N] (BF16 = 2 bytes)
    - 写: output [M, N] (BF16 = 2 bytes)
    """
    input_bytes = 2 if input_dtype == torch.bfloat16 else 4  # BF16=2, FP32=4
    
    # 计算总字节数
    bytes_read_gemm = M * N * input_bytes
    bytes_read_scale_a = M * 4  # FP32
    bytes_read_scale_b = N * 4  # FP32
    bytes_read_bias = N * 2    # BF16
    bytes_write_output = M * N * 2  # BF16
    
    total_bytes = (bytes_read_gemm + bytes_read_scale_a + bytes_read_scale_b + 
                   bytes_read_bias + bytes_write_output)
    
    # 带宽转换: GB/s -> bytes/µs (1 GB/s = 1e9 bytes/s = 1e3 bytes/µs)
    bandwidth_bytes_per_us = bandwidth_gbps * 1e3
    
    # 理论最小时间
    theoretical_time_us = total_bytes / bandwidth_bytes_per_us
    
    return theoretical_time_us


def calculate_bandwidth_efficiency(actual_time_us: float, M: int, N: int, 
                                   input_dtype: torch.dtype, bandwidth_gbps: float) -> float:
    """计算实际带宽利用率 (%)"""
    theoretical_time_us = calculate_theoretical_time_us(M, N, input_dtype, bandwidth_gbps)
    return (theoretical_time_us / actual_time_us) * 100.0


def calculate_achieved_bandwidth_gbps(actual_time_us: float, M: int, N: int, 
                                       input_dtype: torch.dtype) -> float:
    """计算实际达到的带宽 (GB/s)"""
    input_bytes = 2 if input_dtype == torch.bfloat16 else 4
    
    bytes_read_gemm = M * N * input_bytes
    bytes_read_scale_a = M * 4
    bytes_read_scale_b = N * 4
    bytes_read_bias = N * 2
    bytes_write_output = M * N * 2
    
    total_bytes = (bytes_read_gemm + bytes_read_scale_a + bytes_read_scale_b + 
                   bytes_read_bias + bytes_write_output)
    
    # bytes / µs -> GB/s (bytes / µs * 1e-3 = GB/s)
    achieved_bandwidth = total_bytes / actual_time_us * 1e-3
    
    return achieved_bandwidth


# =============================================================================
# 动态加载 Kernel
# =============================================================================

def load_triton_kernel(tuned: bool = False):
    """动态加载 triton kernel"""
    kernels_dir = Path(__file__).parent
    
    if tuned:
        module_path = kernels_dir / "dequant_bias_kernel_tuned.py"
        func_name = "dequant_bias_triton_tuned"
    else:
        module_path = kernels_dir / "autotune_dequant_bias.py"
        func_name = "dequant_bias_autotune"
    
    if not module_path.exists():
        print(f"❌ 文件不存在: {module_path}")
        return None, None
    
    spec = importlib.util.spec_from_file_location(func_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    triton_func = getattr(module, func_name)
    
    # PyTorch 参考实现
    if tuned:
        pytorch_func = module.dequant_bias_pytorch
    else:
        # autotune 版本需要单独加载 pytorch 实现
        from dequant_bias_kernel import dequant_bias_pytorch
        pytorch_func = dequant_bias_pytorch
    
    return triton_func, pytorch_func


# =============================================================================
# 测试配置 (参考 autotune_example)
# =============================================================================

# BitNet 模型常见的隐藏层大小
N_VALUES = [2560, 3840, 13824]

# batch size / sequence length 变化
M_VALUES = [
    1, 16, 32, 48, 64, 80, 96, 112, 128,
    192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
    6144, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 32768, 40960, 49152, 65536
]


# =============================================================================
# 正确性测试
# =============================================================================

def test_correctness(tuned: bool = False):
    """测试 Triton 实现的正确性"""
    print("=" * 80)
    print(f"正确性测试 {'[TUNED]' if tuned else '[AUTOTUNE]'}")
    print("=" * 80)
    
    triton_func, pytorch_func = load_triton_kernel(tuned)
    if triton_func is None:
        return False
    
    torch.manual_seed(42)
    
    basic_shapes = [
        (128, 256), (256, 512), (512, 1024), (1024, 2048),
        (2048, 4096), (4096, 4096), (1, 1024), (1024, 1), (127, 513),
    ]
    
    all_passed = True
    
    # BF16 输入测试
    print("\nBF16 输入测试:")
    print("-" * 60)
    for M, N in basic_shapes:
        gemm_output = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
        scale_a = torch.rand(M, 1, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        scale_b = torch.rand(1, N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
        
        ref = pytorch_func(gemm_output, scale_a, scale_b, bias)
        out = triton_func(gemm_output, scale_a, scale_b, bias)
        
        max_diff = (ref.float() - out.float()).abs().max().item()
        passed = max_diff < 1e-2
        all_passed = all_passed and passed
        print(f"[{'✓' if passed else '✗'}] BF16 ({M:5d}, {N:5d}): max_diff={max_diff:.6f}")
    
    # FP32 输入测试
    print("\nFP32 输入测试:")
    print("-" * 60)
    for M, N in basic_shapes:
        gemm_output = torch.randn(M, N, dtype=torch.float32, device='cuda')
        scale_a = torch.rand(M, 1, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        scale_b = torch.rand(1, N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
        bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
        
        ref = pytorch_func(gemm_output, scale_a, scale_b, bias)
        out = triton_func(gemm_output, scale_a, scale_b, bias)
        
        max_diff = (ref.float() - out.float()).abs().max().item()
        passed = max_diff < 1e-2
        all_passed = all_passed and passed
        print(f"[{'✓' if passed else '✗'}] FP32 ({M:5d}, {N:5d}): max_diff={max_diff:.6f}")
    
    # BitNet 常用形状
    print("\nBitNet 形状测试:")
    print("-" * 60)
    for N in N_VALUES:
        for M in [1, 64, 256, 1024, 4096]:
            gemm_output = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
            scale_a = torch.rand(M, 1, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            scale_b = torch.rand(1, N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
            
            ref = pytorch_func(gemm_output, scale_a, scale_b, bias)
            out = triton_func(gemm_output, scale_a, scale_b, bias)
            
            max_diff = (ref.float() - out.float()).abs().max().item()
            passed = max_diff < 1e-2
            all_passed = all_passed and passed
            print(f"[{'✓' if passed else '✗'}] ({M:5d}, {N:5d}): max_diff={max_diff:.6f}")
    
    print("\n" + "=" * 80)
    print(f"正确性测试 {'全部通过 ✓' if all_passed else '存在失败 ✗'}")
    print("=" * 80)
    
    return all_passed


# =============================================================================
# 性能测试
# =============================================================================

def run_benchmark_single_dtype(
    dtype: torch.dtype, 
    dtype_name: str,
    triton_func, 
    pytorch_func,
    warmup: int, 
    rep: int,
    tuned: bool,
    bandwidth_gbps: float,
):
    """单个 dtype 的性能测试"""
    tuned_str = "[TUNED]" if tuned else "[AUTOTUNE]"
    print(f"\n{'='*80}")
    print(f"{dtype_name} 输入性能测试 {tuned_str}")
    print(f"GPU 峰值带宽: {bandwidth_gbps:.0f} GB/s")
    print(f"{'='*80}")
    
    results = {}
    
    for N in N_VALUES:
        print(f"\n--- N={N} ---")
        print(f"{'M':<8} | {'Theory(µs)':<11} | {'PyTorch(µs)':<12} | {'Triton(µs)':<11} | {'BW(GB/s)':<10} | {'BW Eff%':<8} | {'Speedup'}")
        print("-" * 95)
        
        results[N] = {}
        
        for M in M_VALUES:
            # 生成数据
            gemm_output = torch.randn(M, N, dtype=dtype, device='cuda')
            scale_a = torch.rand(M, 1, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            scale_b = torch.rand(1, N, dtype=torch.float32, device='cuda') * 0.1 + 0.01
            bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')
            
            # 计算理论最小时间
            theoretical_time_us = calculate_theoretical_time_us(M, N, dtype, bandwidth_gbps)
            
            # Benchmark PyTorch
            t_pytorch = testing.do_bench(
                lambda: pytorch_func(gemm_output, scale_a, scale_b, bias),
                warmup=warmup, rep=rep, return_mode="min"
            )
            
            # Benchmark Triton
            t_triton = testing.do_bench(
                lambda: triton_func(gemm_output, scale_a, scale_b, bias),
                warmup=warmup, rep=rep, return_mode="min"
            )
            
            speedup = t_pytorch / t_triton
            
            # do_bench 返回 ms，转为 µs
            triton_time_us = t_triton * 1000
            pytorch_time_us = t_pytorch * 1000
            
            # 计算带宽效率和实际带宽
            bandwidth_efficiency = calculate_bandwidth_efficiency(triton_time_us, M, N, dtype, bandwidth_gbps)
            achieved_bandwidth = calculate_achieved_bandwidth_gbps(triton_time_us, M, N, dtype)
            
            results[N][M] = {
                'pytorch': pytorch_time_us,
                'triton': triton_time_us,
                'speedup': speedup,
                'theoretical': theoretical_time_us,
                'bandwidth_efficiency': bandwidth_efficiency,
                'achieved_bandwidth': achieved_bandwidth,
            }
            
            print(f"{M:<8} | {theoretical_time_us:>9.2f}   | {pytorch_time_us:>10.2f}   | {triton_time_us:>9.2f}   | {achieved_bandwidth:>8.1f}   | {bandwidth_efficiency:>6.1f}%  | {speedup:>6.2f}x")
    
    # 汇总
    print(f"\n{dtype_name} 汇总:")
    print("-" * 70)
    for N in N_VALUES:
        speedups = [results[N][M]['speedup'] for M in M_VALUES]
        efficiencies = [results[N][M]['bandwidth_efficiency'] for M in M_VALUES]
        bandwidths = [results[N][M]['achieved_bandwidth'] for M in M_VALUES]
        avg_speedup = sum(speedups) / len(speedups)
        avg_efficiency = sum(efficiencies) / len(efficiencies)
        max_efficiency = max(efficiencies)
        max_bandwidth = max(bandwidths)
        print(f"  N={N}: 平均加速 {avg_speedup:.2f}x | 平均带宽效率 {avg_efficiency:.1f}% | 最高效率 {max_efficiency:.1f}% | 峰值带宽 {max_bandwidth:.1f} GB/s")
    
    return results


def run_benchmark(dtype_filter: str, warmup: int, rep: int, tuned: bool):
    """运行性能测试"""
    triton_func, pytorch_func = load_triton_kernel(tuned)
    if triton_func is None:
        return None
    
    # 使用 H100 PCIe 峰值带宽
    bandwidth_gbps = GPU_PEAK_BANDWIDTH_GBPS
    
    tuned_str = "[TUNED]" if tuned else "[AUTOTUNE]"
    print(f"\n{'='*80}")
    print(f"Dequant + Bias Kernel 性能测试 {tuned_str}")
    print(f"{'='*80}")
    print(f"测试配置: {len(N_VALUES)} 个 N 值 x {len(M_VALUES)} 个 M 值")
    print(f"N 值: {N_VALUES}")
    print(f"Warmup: {warmup}, Rep: {rep}")
    print(f"GPU 峰值带宽: {bandwidth_gbps:.0f} GB/s")
    
    results = {}
    
    # BF16 测试
    if dtype_filter in [None, 'bf16']:
        results['bf16'] = run_benchmark_single_dtype(
            torch.bfloat16, "BF16", triton_func, pytorch_func, warmup, rep, tuned, bandwidth_gbps
        )
    
    # FP32 测试
    if dtype_filter in [None, 'fp32']:
        results['fp32'] = run_benchmark_single_dtype(
            torch.float32, "FP32", triton_func, pytorch_func, warmup, rep, tuned, bandwidth_gbps
        )
    
    # 总结
    print(f"\n{'='*80}")
    print("总结")
    print(f"{'='*80}")
    
    if 'bf16' in results and 'fp32' in results:
        print(f"\n{'N':<8} | {'BF16 Speedup':>14} | {'BF16 BW Eff':>12} | {'FP32 Speedup':>14} | {'FP32 BW Eff':>12}")
        print("-" * 75)
        for N in N_VALUES:
            bf16_speedups = [results['bf16'][N][M]['speedup'] for M in M_VALUES]
            bf16_effs = [results['bf16'][N][M]['bandwidth_efficiency'] for M in M_VALUES]
            fp32_speedups = [results['fp32'][N][M]['speedup'] for M in M_VALUES]
            fp32_effs = [results['fp32'][N][M]['bandwidth_efficiency'] for M in M_VALUES]
            bf16_avg_speedup = sum(bf16_speedups) / len(bf16_speedups)
            bf16_avg_eff = sum(bf16_effs) / len(bf16_effs)
            fp32_avg_speedup = sum(fp32_speedups) / len(fp32_speedups)
            fp32_avg_eff = sum(fp32_effs) / len(fp32_effs)
            print(f"{N:<8} | {bf16_avg_speedup:>13.2f}x | {bf16_avg_eff:>10.1f}%  | {fp32_avg_speedup:>13.2f}x | {fp32_avg_eff:>10.1f}%")
        
        print(f"\n说明:")
        print(f"  - Theory(µs): 基于 GPU 峰值带宽 ({bandwidth_gbps:.0f} GB/s) 计算的理论最小执行时间")
        print(f"  - BW(GB/s): Triton kernel 实际达到的内存带宽")
        print(f"  - BW Eff%: 带宽效率 = 理论时间 / 实际时间 × 100%")
        print(f"  - 带宽效率越接近 100%，说明 kernel 越接近理论峰值性能")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dequant + Bias Kernel Benchmark")
    parser.add_argument('--tuned', action='store_true',
                        help='使用已调优的固定配置版本 (dequant_bias_kernel_tuned.py)')
    parser.add_argument('--correctness', action='store_true',
                        help='只运行正确性测试')
    parser.add_argument('--dtype', type=str, default=None, choices=['bf16', 'fp32'],
                        help='只测试指定 dtype')
    parser.add_argument('--warmup', type=int, default=25, help='Warmup iterations')
    parser.add_argument('--rep', type=int, default=100, help='Benchmark repetitions')
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return 1
    
    print("=" * 80)
    print("Dequant + Bias Triton Kernel Benchmark")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    import triton
    print(f"Triton: {triton.__version__}")
    print(f"Mode: {'TUNED (固定配置)' if args.tuned else 'AUTOTUNE (预热时调优)'}")
    
    if args.correctness:
        success = test_correctness(args.tuned)
        return 0 if success else 1
    
    run_benchmark(args.dtype, args.warmup, args.rep, args.tuned)
    
    print(f"\n{'='*80}")
    print("Benchmark 完成!")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())
