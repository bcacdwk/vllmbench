#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_profile_accuracy.py - 验证 ProfileTimer 的测量准确性

对比两种计时方式：
1. 批量计时（精确）：先 warmup，然后 N 次循环，只记录总时间
2. 逐次计时（ProfileTimer）：每次调用都记录 start/end event
"""

import os
import sys
from pathlib import Path

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

_SCRIPT_DIR = Path(__file__).parent.absolute()
_SLIDESPARSE_TEST_DIR = _SCRIPT_DIR.parent
_SLIDESPARSE_DIR = _SLIDESPARSE_TEST_DIR.parent
_VLLMBENCH_DIR = _SLIDESPARSE_DIR.parent
sys.path.insert(0, str(_VLLMBENCH_DIR))
sys.path.insert(0, str(_SLIDESPARSE_TEST_DIR))

import torch
import time


def main():
    os.environ["USE_CUBLASLT"] = "1"
    os.environ["USE_CUSPARSELT"] = "0"
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        quant_only_fp8_kernel,
        dequant_bias_kernel,
        _get_gemm_extension,
        get_inner_dtype_str,
    )
    
    device = "cuda"
    ext = _get_gemm_extension("cublaslt")
    
    M, N, K = 32, 896, 896
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    weight_scale = torch.ones(N, 1, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.bfloat16, device=device)
    
    WARMUP = 100
    REPEAT = 1000
    
    print("=" * 80)
    print("验证 Profile 测量准确性")
    print("=" * 80)
    print(f"M={M}, N={N}, K={K}")
    print(f"Warmup={WARMUP}, Repeat={REPEAT}")
    
    # ========================================================================
    # 方法 1: 批量计时（精确基准）
    # ========================================================================
    print("\n" + "-" * 80)
    print("方法 1: 批量计时（精确基准）")
    print("-" * 80)
    
    def run_kernel():
        qinput, scale_a = quant_only_fp8_kernel(input_bf16)
        gemm_out = ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
        return dequant_bias_kernel(gemm_out[:M], scale_a[:M], weight_scale, bias, torch.bfloat16)
    
    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()
    
    # 批量计时
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEAT):
        run_kernel()
    end.record()
    torch.cuda.synchronize()
    
    batch_avg_ms = start.elapsed_time(end) / REPEAT
    print(f"批量平均: {batch_avg_ms:.4f} ms ({batch_avg_ms*1000:.2f} us)")
    
    # ========================================================================
    # 方法 2: 逐次计时（每次 record start/end）
    # ========================================================================
    print("\n" + "-" * 80)
    print("方法 2: 逐次计时（每次 record start/end，不同步）")
    print("-" * 80)
    
    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()
    
    # 逐次计时，但不立即同步
    events = []
    for _ in range(REPEAT):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        run_kernel()
        end_evt.record()
        events.append((start_evt, end_evt))
    
    # 最后同步并计算
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in events]
    per_iter_avg_ms = sum(times) / len(times)
    
    print(f"逐次平均: {per_iter_avg_ms:.4f} ms ({per_iter_avg_ms*1000:.2f} us)")
    print(f"差异: {per_iter_avg_ms / batch_avg_ms:.2f}x")
    
    # ========================================================================
    # 方法 3: 逐次计时 + 每次同步（最坏情况）
    # ========================================================================
    print("\n" + "-" * 80)
    print("方法 3: 逐次计时 + 每次同步（最坏情况）")
    print("-" * 80)
    
    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()
    
    times_sync = []
    for _ in range(REPEAT // 10):  # 减少次数，因为每次同步很慢
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        run_kernel()
        end_evt.record()
        torch.cuda.synchronize()  # 每次同步
        times_sync.append(start_evt.elapsed_time(end_evt))
    
    sync_avg_ms = sum(times_sync) / len(times_sync)
    print(f"同步平均: {sync_avg_ms:.4f} ms ({sync_avg_ms*1000:.2f} us)")
    print(f"差异: {sync_avg_ms / batch_avg_ms:.2f}x")
    
    # ========================================================================
    # 方法 4: 分别测量每个 kernel（模拟 ProfileTimer）
    # ========================================================================
    print("\n" + "-" * 80)
    print("方法 4: 分别测量每个 kernel（模拟 ProfileTimer）")
    print("-" * 80)
    
    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()
    
    quant_events = []
    gemm_events = []
    dequant_events = []
    
    for _ in range(REPEAT):
        # Quant
        qs = torch.cuda.Event(enable_timing=True)
        qe = torch.cuda.Event(enable_timing=True)
        qs.record()
        qinput, scale_a = quant_only_fp8_kernel(input_bf16)
        qe.record()
        quant_events.append((qs, qe))
        
        # GEMM
        gs = torch.cuda.Event(enable_timing=True)
        ge = torch.cuda.Event(enable_timing=True)
        gs.record()
        gemm_out = ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
        ge.record()
        gemm_events.append((gs, ge))
        
        # Dequant
        ds = torch.cuda.Event(enable_timing=True)
        de = torch.cuda.Event(enable_timing=True)
        ds.record()
        output = dequant_bias_kernel(gemm_out[:M], scale_a[:M], weight_scale, bias, torch.bfloat16)
        de.record()
        dequant_events.append((ds, de))
    
    torch.cuda.synchronize()
    
    quant_avg = sum(s.elapsed_time(e) for s, e in quant_events) / REPEAT
    gemm_avg = sum(s.elapsed_time(e) for s, e in gemm_events) / REPEAT
    dequant_avg = sum(s.elapsed_time(e) for s, e in dequant_events) / REPEAT
    total_avg = quant_avg + gemm_avg + dequant_avg
    
    print(f"Quant:   {quant_avg:.4f} ms ({quant_avg*1000:.2f} us)")
    print(f"GEMM:    {gemm_avg:.4f} ms ({gemm_avg*1000:.2f} us)")
    print(f"Dequant: {dequant_avg:.4f} ms ({dequant_avg*1000:.2f} us)")
    print(f"总计:    {total_avg:.4f} ms ({total_avg*1000:.2f} us)")
    print(f"差异:    {total_avg / batch_avg_ms:.2f}x")
    
    # ========================================================================
    # 方法 5: 使用 Python time.perf_counter（会包含 CPU 开销）
    # ========================================================================
    print("\n" + "-" * 80)
    print("方法 5: Python time.perf_counter（包含 CPU 开销）")
    print("-" * 80)
    
    # Warmup
    for _ in range(WARMUP):
        run_kernel()
    torch.cuda.synchronize()
    
    times_cpu = []
    for _ in range(REPEAT // 10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_kernel()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_cpu.append((t1 - t0) * 1000)  # ms
    
    cpu_avg_ms = sum(times_cpu) / len(times_cpu)
    print(f"Python 计时平均: {cpu_avg_ms:.4f} ms ({cpu_avg_ms*1000:.2f} us)")
    print(f"差异: {cpu_avg_ms / batch_avg_ms:.2f}x")
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print(f"""
批量计时（精确基准）:     {batch_avg_ms:.4f} ms ({batch_avg_ms*1000:.2f} us)
逐次计时（异步 events）:  {per_iter_avg_ms:.4f} ms ({per_iter_avg_ms / batch_avg_ms:.2f}x)
逐次计时（每次同步）:     {sync_avg_ms:.4f} ms ({sync_avg_ms / batch_avg_ms:.2f}x)
分 kernel 计时:          {total_avg:.4f} ms ({total_avg / batch_avg_ms:.2f}x)
Python time 计时:        {cpu_avg_ms:.4f} ms ({cpu_avg_ms / batch_avg_ms:.2f}x)

结论：
- 如果 ProfileTimer 使用的逐次计时方式准确，那么端到端 profile 
  应该与独立测试一致
- 如果端到端 profile 显示的时间明显更长，说明有其他因素：
  1. vLLM 内部的 Python 开销被算入 kernel 时间
  2. CUDA 内存分配/释放开销
  3. GPU 调度竞争（多 stream）
""")


if __name__ == "__main__":
    main()
