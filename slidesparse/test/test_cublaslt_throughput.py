#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 5: 吞吐量对比测试

对比 cuBLASLt 后端与原生 cutlass 后端的吞吐量。

当前阶段（Phase 3 初期）:
    由于 cuBLASLt Op 内部仍然使用 cutlass，
    所以吞吐量应该基本一致。这个测试主要验证框架开销。

后续阶段（Phase 3 完成后）:
    当替换为真正的 cuBLASLt kernel 后，
    可以观察到实际的性能差异。

运行方式:
    CUDA_VISIBLE_DEVICES=0 python3 slidesparse/test/test_cublaslt_throughput.py
"""

import sys
import os
import time

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def benchmark_kernel(op, input_tensor, weight, weight_scale, warmup=10, repeat=100):
    """
    Benchmark kernel 执行时间
    
    Returns:
        tuple: (平均时间 ms, 标准差 ms)
    """
    import torch
    
    # Warmup
    for _ in range(warmup):
        _ = op.apply(
            input=input_tensor,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        _ = op.apply(
            input=input_tensor,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    import statistics
    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    
    return mean_time, std_time


def test_throughput():
    """测试吞吐量"""
    print("=" * 60)
    print("测试 5: 吞吐量对比测试")
    print("=" * 60)
    
    import torch
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("\n✗ CUDA 不可用，跳过测试")
        return False
    
    device = torch.device("cuda:0")
    print(f"\n使用设备: {torch.cuda.get_device_name(0)}")
    
    # 导入测试模块
    print("\n[5.1] 导入测试模块...")
    try:
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
        from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        print("    ✓ 模块导入成功")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        return False
    
    # 测试多种尺寸
    print("\n[5.2] 吞吐量测试...")
    
    test_cases = [
        {"M": 1, "K": 4096, "N": 4096, "name": "单 token (4K x 4K)"},
        {"M": 16, "K": 4096, "N": 4096, "name": "小 batch (4K x 4K)"},
        {"M": 64, "K": 4096, "N": 4096, "name": "中 batch (4K x 4K)"},
        {"M": 128, "K": 4096, "N": 11008, "name": "FFN (4K x 11K)"},
        {"M": 256, "K": 4096, "N": 4096, "name": "大 batch (4K x 4K)"},
    ]
    
    results = []
    
    for case in test_cases:
        M, K, N = case["M"], case["K"], case["N"]
        name = case["name"]
        
        print(f"\n    测试用例: {name}")
        
        # 创建输入
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        weight = torch.randn(K, N, dtype=torch.float8_e4m3fn, device=device)
        weight_scale = torch.ones(1, dtype=torch.float32, device=device)
        
        # 创建两个 Op
        fp8_op = Fp8LinearOp(act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
        cublaslt_op = CuBLASLtFp8LinearOp(act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
        
        try:
            # Benchmark 原生 Op
            fp8_time, fp8_std = benchmark_kernel(fp8_op, input_tensor, weight, weight_scale)
            
            # Benchmark cuBLASLt Op
            cublaslt_time, cublaslt_std = benchmark_kernel(cublaslt_op, input_tensor, weight, weight_scale)
            
            # 计算吞吐量 (TFLOPS)
            flops = 2 * M * K * N  # GEMM FLOPs
            fp8_tflops = flops / (fp8_time * 1e-3) / 1e12
            cublaslt_tflops = flops / (cublaslt_time * 1e-3) / 1e12
            
            # 计算差异
            speedup = fp8_time / cublaslt_time
            
            print(f"        原生 Fp8LinearOp:    {fp8_time:.3f} ± {fp8_std:.3f} ms ({fp8_tflops:.2f} TFLOPS)")
            print(f"        CuBLASLtFp8LinearOp: {cublaslt_time:.3f} ± {cublaslt_std:.3f} ms ({cublaslt_tflops:.2f} TFLOPS)")
            print(f"        加速比: {speedup:.3f}x")
            
            results.append({
                "name": name,
                "M": M, "K": K, "N": N,
                "fp8_time": fp8_time,
                "cublaslt_time": cublaslt_time,
                "speedup": speedup,
            })
            
        except Exception as e:
            print(f"        ✗ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print("吞吐量测试总结:")
    print("-" * 60)
    print(f"{'测试用例':<25} {'原生(ms)':<12} {'cuBLASLt(ms)':<12} {'加速比':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<25} {r['fp8_time']:<12.3f} {r['cublaslt_time']:<12.3f} {r['speedup']:<8.3f}x")
    print("-" * 60)
    
    avg_speedup = sum(r['speedup'] for r in results) / len(results) if results else 0
    print(f"平均加速比: {avg_speedup:.3f}x")
    
    print("\n注意: 当前阶段 cuBLASLt Op 内部使用 cutlass，")
    print("      所以吞吐量应该基本一致。差异主要来自框架开销。")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_throughput()
    sys.exit(0 if success else 1)
