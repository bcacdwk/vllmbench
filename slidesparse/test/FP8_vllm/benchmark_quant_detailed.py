#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的 Quant Kernel 基准测试

模拟端到端推理中的真实调用模式：
1. 多种 MK 组合交替调用
2. 测量首次调用 vs 后续调用的差异
3. 模拟内存压力环境
"""

import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

# 添加项目路径
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================================
# Qwen2.5-0.5B 的真实 MK 组合
# ============================================================================

# Prefill M=2048, Decode M=32
# 每层有 7 个线性层，共 24 层
QWEN_MK_CONFIGS = [
    # (K, N, name)
    (896, 896, "q_proj"),
    (896, 128, "k_proj"),
    (896, 128, "v_proj"),
    (896, 896, "o_proj"),
    (896, 4864, "gate_proj"),
    (896, 4864, "up_proj"),
    (4864, 896, "down_proj"),
]

NUM_LAYERS = 24


def benchmark_quant_realistic():
    """模拟真实推理中的 quant kernel 调用模式"""
    print("=" * 100)
    print("Quant Kernel 真实场景基准测试")
    print("=" * 100)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模拟 Qwen2.5-0.5B: {NUM_LAYERS} layers × 7 linear layers = {NUM_LAYERS * 7} 次/token")
    print()
    
    # 加载 kernels
    print("加载 kernels...")
    
    # CUTLASS QuantFP8
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    print("  ✓ CUTLASS (QuantFP8)")
    
    # Triton quant_only
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        _load_quant_only_fp8_kernel,
        quant_only_fp8_kernel,
    )
    _load_quant_only_fp8_kernel()
    print("  ✓ Triton quant_only")
    
    print()
    
    device = torch.device("cuda:0")
    dtype = torch.float16
    
    for M in [32, 2048]:
        print(f"\n{'#' * 100}")
        print(f"# M = {M} ({'Decode' if M == 32 else 'Prefill'})")
        print(f"{'#' * 100}")
        
        # ======================================================================
        # 测试 1: 首次调用 vs 后续调用
        # ======================================================================
        print(f"\n--- 测试 1: 首次调用 vs 后续调用 (warm kernel cache) ---")
        
        for K, N, name in QWEN_MK_CONFIGS:
            input_tensor = torch.randn(M, K, dtype=dtype, device=device)
            
            # 首次调用
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = quant_only_fp8_kernel(input_tensor)
            torch.cuda.synchronize()
            first_call = (time.perf_counter() - start) * 1000
            
            # 后续调用 (10 次平均)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                _ = quant_only_fp8_kernel(input_tensor)
            torch.cuda.synchronize()
            avg_call = (time.perf_counter() - start) / 10 * 1000
            
            print(f"  {name:12s} (K={K:4d}): first={first_call:.4f} ms, avg={avg_call:.4f} ms, ratio={first_call/avg_call:.1f}x")
        
        # ======================================================================
        # 测试 2: 模拟一次 forward pass (24 layers × 7 linear)
        # ======================================================================
        print(f"\n--- 测试 2: 模拟 forward pass ({NUM_LAYERS} layers × 7 linear) ---")
        
        # 预先创建所有 input tensors
        all_inputs = []
        for layer_idx in range(NUM_LAYERS):
            layer_inputs = []
            for K, N, name in QWEN_MK_CONFIGS:
                inp = torch.randn(M, K, dtype=dtype, device=device)
                layer_inputs.append(inp)
            all_inputs.append(layer_inputs)
        
        # Warmup
        for _ in range(3):
            for layer_inputs in all_inputs[:2]:
                for inp in layer_inputs:
                    _ = quant_only_fp8_kernel(inp)
        torch.cuda.synchronize()
        
        # CUTLASS (QuantFP8)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for layer_inputs in all_inputs:
            for inp in layer_inputs:
                qinput, scale = quant_fp8(inp)
        torch.cuda.synchronize()
        cutlass_time = (time.perf_counter() - start) * 1000
        cutlass_avg = cutlass_time / (NUM_LAYERS * 7)
        
        # Triton quant_only
        torch.cuda.synchronize()
        start = time.perf_counter()
        for layer_inputs in all_inputs:
            for inp in layer_inputs:
                qinput, scale = quant_only_fp8_kernel(inp)
        torch.cuda.synchronize()
        triton_time = (time.perf_counter() - start) * 1000
        triton_avg = triton_time / (NUM_LAYERS * 7)
        
        print(f"  CUTLASS (QuantFP8):   total={cutlass_time:8.3f} ms, avg={cutlass_avg:.4f} ms/call")
        print(f"  Triton quant_only:    total={triton_time:8.3f} ms, avg={triton_avg:.4f} ms/call")
        print(f"  Ratio (Triton/CUTLASS): {triton_time/cutlass_time:.2f}x")
        
        # ======================================================================
        # 测试 3: CUDA Events 精确计时
        # ======================================================================
        print(f"\n--- 测试 3: CUDA Events 精确计时 (10 forward passes) ---")
        
        cutlass_events = []
        triton_events = []
        
        # CUTLASS
        for _ in range(10):
            for layer_inputs in all_inputs:
                for inp in layer_inputs:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    qinput, scale = quant_fp8(inp)
                    end_event.record()
                    cutlass_events.append((start_event, end_event))
        
        # Triton
        for _ in range(10):
            for layer_inputs in all_inputs:
                for inp in layer_inputs:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    qinput, scale = quant_only_fp8_kernel(inp)
                    end_event.record()
                    triton_events.append((start_event, end_event))
        
        torch.cuda.synchronize()
        
        cutlass_times = [s.elapsed_time(e) for s, e in cutlass_events]
        triton_times = [s.elapsed_time(e) for s, e in triton_events]
        
        cutlass_avg_event = sum(cutlass_times) / len(cutlass_times)
        triton_avg_event = sum(triton_times) / len(triton_times)
        
        print(f"  CUTLASS (QuantFP8):   avg={cutlass_avg_event:.4f} ms, min={min(cutlass_times):.4f} ms, max={max(cutlass_times):.4f} ms")
        print(f"  Triton quant_only:    avg={triton_avg_event:.4f} ms, min={min(triton_times):.4f} ms, max={max(triton_times):.4f} ms")
        print(f"  Ratio (Triton/CUTLASS): {triton_avg_event/cutlass_avg_event:.2f}x")
        
        # ======================================================================
        # 测试 4: 按 K 值统计
        # ======================================================================
        print(f"\n--- 测试 4: 按 K 值统计 (CUDA Events, 100 次) ---")
        
        for K, N, name in QWEN_MK_CONFIGS:
            inp = torch.randn(M, K, dtype=dtype, device=device)
            
            # Warmup
            for _ in range(10):
                _ = quant_only_fp8_kernel(inp)
            torch.cuda.synchronize()
            
            # CUTLASS
            cutlass_events = []
            for _ in range(100):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = quant_fp8(inp)
                end_event.record()
                cutlass_events.append((start_event, end_event))
            
            # Triton
            triton_events = []
            for _ in range(100):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = quant_only_fp8_kernel(inp)
                end_event.record()
                triton_events.append((start_event, end_event))
            
            torch.cuda.synchronize()
            
            cutlass_times = [s.elapsed_time(e) for s, e in cutlass_events]
            triton_times = [s.elapsed_time(e) for s, e in triton_events]
            
            cutlass_avg = sum(cutlass_times) / len(cutlass_times)
            triton_avg = sum(triton_times) / len(triton_times)
            
            print(f"  {name:12s} (K={K:4d}): CUTLASS={cutlass_avg:.4f} ms, Triton={triton_avg:.4f} ms, ratio={triton_avg/cutlass_avg:.2f}x")


def benchmark_with_memory_pressure():
    """模拟模型权重占用内存后的 quant kernel 性能"""
    print("\n" + "=" * 100)
    print("内存压力测试：模拟模型权重占用后的 quant kernel 性能")
    print("=" * 100)
    
    device = torch.device("cuda:0")
    
    # 获取可用内存
    free_mem = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"当前可用显存: {free_mem:.1f} GB")
    
    # 分配一些内存模拟模型权重 (约 1GB)
    print("分配 1GB 模拟权重...")
    dummy_weights = torch.randn(256, 1024, 1024, dtype=torch.float16, device=device)
    torch.cuda.synchronize()
    
    free_mem_after = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"分配后可用显存: {free_mem_after:.1f} GB")
    
    # 运行相同的测试
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from slidesparse.core.SlideSparseLinearMethod_FP8 import quant_only_fp8_kernel
    
    quant_fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)
    
    for M in [32, 2048]:
        print(f"\n--- M={M} ---")
        
        for K in [896, 4864]:
            inp = torch.randn(M, K, dtype=torch.float16, device=device)
            
            # Warmup
            for _ in range(10):
                _ = quant_only_fp8_kernel(inp)
            torch.cuda.synchronize()
            
            # CUTLASS
            cutlass_events = []
            for _ in range(100):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = quant_fp8(inp)
                end_event.record()
                cutlass_events.append((start_event, end_event))
            
            # Triton
            triton_events = []
            for _ in range(100):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = quant_only_fp8_kernel(inp)
                end_event.record()
                triton_events.append((start_event, end_event))
            
            torch.cuda.synchronize()
            
            cutlass_times = [s.elapsed_time(e) for s, e in cutlass_events]
            triton_times = [s.elapsed_time(e) for s, e in triton_events]
            
            cutlass_avg = sum(cutlass_times) / len(cutlass_times)
            triton_avg = sum(triton_times) / len(triton_times)
            
            print(f"  K={K:4d}: CUTLASS={cutlass_avg:.4f} ms, Triton={triton_avg:.4f} ms, ratio={triton_avg/cutlass_avg:.2f}x")
    
    # 清理
    del dummy_weights
    torch.cuda.empty_cache()


if __name__ == "__main__":
    benchmark_quant_realistic()
    benchmark_with_memory_pressure()
