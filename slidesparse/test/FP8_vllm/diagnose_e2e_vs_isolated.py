#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_e2e_vs_isolated.py - 精确对比端到端场景 vs 独立测试

关键问题：为什么 GEMM 在端到端测试中慢了 6x？
- 独立测试: ~9 us
- 端到端 profile: ~54 us

可能的原因：
1. 端到端有很多层并发计算，GPU 调度竞争
2. weight shape 不同（端到端是模型真实 shape）
3. input 来自 KV cache 操作后，内存布局不同
4. Profile 本身的开销在端到端场景下放大
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


def profile_kernel(fn, name, warmup=50, repeat=500):
    """精确的 kernel 计时"""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # 测量
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / repeat
    print(f"  {name}: {avg_ms:.4f} ms ({avg_ms*1000:.2f} us)")
    return avg_ms


def main():
    # 设置环境
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
    
    # ========================================================================
    # Qwen2.5-0.5B 模型真实参数
    # ========================================================================
    # hidden_size = 896
    # intermediate_size = 4864
    # num_attention_heads = 14
    # num_kv_heads = 2
    # head_dim = 64
    
    # 线性层配置（真实的 N, K 值）
    LAYERS = {
        # Attention 层
        "q_proj":    {"N": 896,  "K": 896},   # [hidden, hidden]
        "k_proj":    {"N": 128,  "K": 896},   # [kv_heads*head_dim, hidden]
        "v_proj":    {"N": 128,  "K": 896},
        "o_proj":    {"N": 896,  "K": 896},
        # MLP 层
        "gate_proj": {"N": 4864, "K": 896},   # [intermediate, hidden]
        "up_proj":   {"N": 4864, "K": 896},
        "down_proj": {"N": 896,  "K": 4864},  # [hidden, intermediate]
    }
    
    print("=" * 80)
    print("测试 Qwen2.5-0.5B 真实线性层维度")
    print("=" * 80)
    
    for M in [32, 2048]:
        phase = "Decode" if M == 32 else "Prefill"
        print(f"\n{'='*80}")
        print(f"M={M} ({phase} 场景)")
        print(f"{'='*80}")
        
        total_quant = 0
        total_gemm = 0
        total_dequant = 0
        layer_count = 0
        
        for layer_name, dims in LAYERS.items():
            N, K = dims["N"], dims["K"]
            print(f"\n{layer_name} (N={N}, K={K}):")
            
            # 创建 tensor
            input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            weight_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
            weight_scale = torch.ones(N, 1, dtype=torch.float32, device=device)
            bias = torch.zeros(N, dtype=torch.bfloat16, device=device)
            
            # Quant
            def run_quant():
                return quant_only_fp8_kernel(input_bf16)
            t_quant = profile_kernel(run_quant, "quant")
            total_quant += t_quant
            
            # GEMM
            qinput, scale_a = quant_only_fp8_kernel(input_bf16)
            def run_gemm():
                return ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
            t_gemm = profile_kernel(run_gemm, "gemm")
            total_gemm += t_gemm
            
            # Dequant
            gemm_out = ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
            gemm_out_slice = gemm_out[:M, :]
            scale_a_slice = scale_a[:M]
            def run_dequant():
                return dequant_bias_kernel(gemm_out_slice, scale_a_slice, weight_scale, bias, torch.bfloat16)
            t_dequant = profile_kernel(run_dequant, "dequant")
            total_dequant += t_dequant
            
            layer_count += 1
        
        # 汇总
        print(f"\n{'-'*80}")
        print(f"汇总 ({layer_count} 层):")
        print(f"  平均 Quant:   {total_quant/layer_count:.4f} ms ({total_quant/layer_count*1000:.2f} us)")
        print(f"  平均 GEMM:    {total_gemm/layer_count:.4f} ms ({total_gemm/layer_count*1000:.2f} us)")
        print(f"  平均 Dequant: {total_dequant/layer_count:.4f} ms ({total_dequant/layer_count*1000:.2f} us)")
        print(f"  平均总计:     {(total_quant+total_gemm+total_dequant)/layer_count:.4f} ms")
    
    # ========================================================================
    # 测试 Profile 开销对真实场景的影响
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 Profile 开销的影响")
    print("=" * 80)
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import ProfileTimer, _PROFILE_ENABLED
    
    M, N, K = 32, 896, 896
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    weight_scale = torch.ones(N, 1, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.bfloat16, device=device)
    
    print(f"\nSLIDESPARSE_PROFILE={_PROFILE_ENABLED}")
    
    # 模拟端到端场景：每个 kernel 都有 ProfileTimer 包装
    def run_with_profile():
        with ProfileTimer("test.quant", enabled=True):
            qinput, scale_a = quant_only_fp8_kernel(input_bf16)
        with ProfileTimer("test.gemm", enabled=True):
            gemm_out = ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
        with ProfileTimer("test.dequant", enabled=True):
            output = dequant_bias_kernel(gemm_out[:M], scale_a[:M], weight_scale, bias, torch.bfloat16)
        return output
    
    def run_without_profile():
        qinput, scale_a = quant_only_fp8_kernel(input_bf16)
        gemm_out = ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
        output = dequant_bias_kernel(gemm_out[:M], scale_a[:M], weight_scale, bias, torch.bfloat16)
        return output
    
    print("\n不带 profile:")
    t_no_profile = profile_kernel(run_without_profile, "total")
    
    print("\n带 profile:")
    t_with_profile = profile_kernel(run_with_profile, "total")
    
    print(f"\nProfile 开销: {(t_with_profile - t_no_profile)*1000:.2f} us ({(t_with_profile/t_no_profile - 1)*100:.1f}%)")
    
    # ========================================================================
    # 模拟 GPU 竞争场景
    # ========================================================================
    print("\n" + "=" * 80)
    print("模拟 GPU 竞争场景（多个操作并发）")
    print("=" * 80)
    
    # 创建多个层的数据，模拟真实推理场景
    num_layers = 24  # Qwen2.5-0.5B 有 24 层
    layer_data = []
    
    for _ in range(num_layers):
        layer_data.append({
            "input": torch.randn(M, 896, dtype=torch.bfloat16, device=device),
            "weight": torch.randn(896, 896, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn),
            "scale": torch.ones(896, 1, dtype=torch.float32, device=device),
            "bias": torch.zeros(896, dtype=torch.bfloat16, device=device),
        })
    
    def run_single_layer(idx):
        data = layer_data[idx]
        qinput, scale_a = quant_only_fp8_kernel(data["input"])
        gemm_out = ext.cublaslt_fp8_mm(data["weight"], qinput, get_inner_dtype_str())
        return dequant_bias_kernel(gemm_out[:M], scale_a[:M], data["scale"], data["bias"], torch.bfloat16)
    
    def run_all_layers():
        for i in range(num_layers):
            _ = run_single_layer(i)
    
    # 单层测试
    print("\n单层 (o_proj):")
    t_single = profile_kernel(lambda: run_single_layer(0), "total", warmup=20, repeat=100)
    
    # 所有层测试
    print(f"\n所有 {num_layers} 层顺序执行:")
    for _ in range(20):  # warmup
        run_all_layers()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        run_all_layers()
    end.record()
    torch.cuda.synchronize()
    t_all = start.elapsed_time(end) / 100 / num_layers
    print(f"  平均每层: {t_all:.4f} ms ({t_all*1000:.2f} us)")
    
    print(f"\n多层执行时的每层开销: {t_all:.4f} ms vs 单层: {t_single:.4f} ms")
    print(f"差异: {t_all/t_single:.2f}x")


if __name__ == "__main__":
    main()
