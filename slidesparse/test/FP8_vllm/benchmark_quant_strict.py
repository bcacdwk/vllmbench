#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
benchmark_quant_strict.py - 严格的 Quant Kernel 性能测试

确保测量的是真正的 kernel 执行时间，而不是 JIT 编译时间。

策略:
1. 大量 warmup (100次) 确保 JIT 编译完成
2. 多轮测试检查结果一致性
3. 每轮内部批量测量，减少 overhead
"""

import os
import sys
from pathlib import Path

import torch
import torch.cuda

# 设置 Triton ptxas 路径
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# 测试参数
M_VALUES = [32, 2048]
K_VALUES = [896, 4864]
WARMUP_ITERS = 100  # 大量 warmup
MEASURE_ITERS = 100  # 测量次数
NUM_ROUNDS = 3       # 重复测试轮数

device = "cuda"
dtype = torch.bfloat16


def main():
    print("=" * 80)
    print("严格的 Quant Kernel 性能测试")
    print("=" * 80)
    print(f"设备: {torch.cuda.get_device_name()}")
    print(f"Warmup: {WARMUP_ITERS}, Measure: {MEASURE_ITERS}, Rounds: {NUM_ROUNDS}")
    print()
    
    # ========== 加载函数 ==========
    print("加载 kernel 函数...")
    
    # 1. CUTLASS (vLLM 原生)
    from vllm import _custom_ops as ops
    
    def cutlass_quant(x):
        return ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    
    # 2. Triton quant_only
    from slidesparse.core.SlideSparseLinearMethod_FP8 import quant_only_fp8_kernel
    
    def triton_quant_only(x):
        return quant_only_fp8_kernel(x)  # 无额外参数
    
    # 3. Triton quant_slide
    from slidesparse.core.SlideSparseLinearMethod_FP8 import quant_slide_fp8_kernel
    
    L = 8  # SlideSparse L 参数
    
    def triton_quant_slide(x):
        return quant_slide_fp8_kernel(x, L=L)
    
    print("Kernel 函数加载完成\n")
    
    # ========== 测试所有 M/K 组合 ==========
    results = {}
    
    for M in M_VALUES:
        for K in K_VALUES:
            print("=" * 80)
            print(f"测试 M={M}, K={K}")
            print("=" * 80)
            
            # 创建输入
            x = torch.randn(M, K, device=device, dtype=dtype)
            
            # 创建 CUDA Events
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # ===== Phase 1: 大量 Warmup =====
            print(f"\nPhase 1: Warmup ({WARMUP_ITERS} iterations per kernel)...")
            
            # Warmup CUTLASS
            for _ in range(WARMUP_ITERS):
                cutlass_quant(x)
            torch.cuda.synchronize()
            
            # Warmup Triton quant_only
            for _ in range(WARMUP_ITERS):
                triton_quant_only(x)
            torch.cuda.synchronize()
            
            # Warmup Triton quant_slide
            for _ in range(WARMUP_ITERS):
                triton_quant_slide(x)
            torch.cuda.synchronize()
            
            print("Warmup 完成")
            
            # ===== Phase 2: 多轮测量 =====
            print(f"\nPhase 2: 测量 ({NUM_ROUNDS} rounds x {MEASURE_ITERS} iterations)...")
            
            cutlass_times = []
            triton_only_times = []
            triton_slide_times = []
            
            for round_idx in range(NUM_ROUNDS):
                # CUTLASS
                torch.cuda.synchronize()
                start.record()
                for _ in range(MEASURE_ITERS):
                    cutlass_quant(x)
                end.record()
                torch.cuda.synchronize()
                cutlass_times.append(start.elapsed_time(end) / MEASURE_ITERS)
                
                # Triton quant_only
                torch.cuda.synchronize()
                start.record()
                for _ in range(MEASURE_ITERS):
                    triton_quant_only(x)
                end.record()
                torch.cuda.synchronize()
                triton_only_times.append(start.elapsed_time(end) / MEASURE_ITERS)
                
                # Triton quant_slide
                torch.cuda.synchronize()
                start.record()
                for _ in range(MEASURE_ITERS):
                    triton_quant_slide(x)
                end.record()
                torch.cuda.synchronize()
                triton_slide_times.append(start.elapsed_time(end) / MEASURE_ITERS)
            
            # ===== 结果 =====
            def stats(times):
                mean = sum(times) / len(times)
                std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
                return mean, std, min(times), max(times)
            
            cutlass_mean, cutlass_std, cutlass_min, cutlass_max = stats(cutlass_times)
            triton_only_mean, triton_only_std, triton_only_min, triton_only_max = stats(triton_only_times)
            triton_slide_mean, triton_slide_std, triton_slide_min, triton_slide_max = stats(triton_slide_times)
            
            print(f"\n结果 (M={M}, K={K}):")
            print("-" * 60)
            print(f"  CUTLASS:           {cutlass_mean*1000:.2f} us ± {cutlass_std*1000:.2f} us  (min={cutlass_min*1000:.2f}, max={cutlass_max*1000:.2f})")
            print(f"  Triton quant_only: {triton_only_mean*1000:.2f} us ± {triton_only_std*1000:.2f} us  (min={triton_only_min*1000:.2f}, max={triton_only_max*1000:.2f})")
            print(f"  Triton quant_slide:{triton_slide_mean*1000:.2f} us ± {triton_slide_std*1000:.2f} us  (min={triton_slide_min*1000:.2f}, max={triton_slide_max*1000:.2f})")
            print()
            print(f"  比值 (vs CUTLASS):")
            print(f"    quant_only:  {triton_only_mean/cutlass_mean:.2f}x")
            print(f"    quant_slide: {triton_slide_mean/cutlass_mean:.2f}x")
            print()
            print(f"  各轮次数据:")
            for i in range(NUM_ROUNDS):
                print(f"    Round {i+1}: CUTLASS={cutlass_times[i]*1000:.2f}us, quant_only={triton_only_times[i]*1000:.2f}us, quant_slide={triton_slide_times[i]*1000:.2f}us")
            
            results[(M, K)] = {
                'cutlass': cutlass_mean,
                'triton_only': triton_only_mean,
                'triton_slide': triton_slide_mean,
            }
    
    # ========== 汇总表格 ==========
    print("\n" + "=" * 80)
    print("汇总表格 (单位: us)")
    print("=" * 80)
    print(f"{'M':>6} │ {'K':>6} │ {'CUTLASS':>12} │ {'quant_only':>12} │ {'quant_slide':>12} │ {'only比值':>8} │ {'slide比值':>8}")
    print("-" * 80)
    for M in M_VALUES:
        for K in K_VALUES:
            r = results[(M, K)]
            cutlass_us = r['cutlass'] * 1000
            only_us = r['triton_only'] * 1000
            slide_us = r['triton_slide'] * 1000
            print(f"{M:>6} │ {K:>6} │ {cutlass_us:>10.2f}us │ {only_us:>10.2f}us │ {slide_us:>10.2f}us │ {r['triton_only']/r['cutlass']:>7.2f}x │ {r['triton_slide']/r['cutlass']:>7.2f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()
