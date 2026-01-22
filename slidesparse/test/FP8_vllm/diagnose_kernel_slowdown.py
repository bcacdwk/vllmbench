#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_kernel_slowdown.py - 诊断 kernel 在不同调用环境下的性能差异

核心问题：
- test_02_kernel.py 单独测试时，cuBLASLt GEMM ~10us
- test_04_throughput.py 端到端测试时，cuBLASLt GEMM ~30us
- 明明是同一个 kernel，为什么差了 3x？

可能的原因：
1. Tensor stride/layout 不同（端到端可能有非连续 tensor）
2. Tensor 内存位置不同（碎片化）
3. GPU 状态不同（端到端有更多并发操作）
4. Profile 计时本身的干扰
5. 输入形状动态变化导致的 kernel 重编译
"""

import os
import sys
from pathlib import Path

os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

# 添加正确的路径
_SCRIPT_DIR = Path(__file__).parent.absolute()
_SLIDESPARSE_TEST_DIR = _SCRIPT_DIR.parent
_SLIDESPARSE_DIR = _SLIDESPARSE_TEST_DIR.parent
_VLLMBENCH_DIR = _SLIDESPARSE_DIR.parent
sys.path.insert(0, str(_VLLMBENCH_DIR))
sys.path.insert(0, str(_SLIDESPARSE_TEST_DIR))

import torch
import time


def profile_with_cuda_events(fn, warmup=10, repeat=100, sync_every_iter=False):
    """
    使用 CUDA events 精确计时
    
    Args:
        fn: 要测试的函数（无参数）
        warmup: 预热次数
        repeat: 重复次数
        sync_every_iter: 是否每次迭代后同步（True=更精确但有开销）
    """
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    if sync_every_iter:
        # 精确计时：每次迭代同步
        times = []
        for _ in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        return sum(times) / len(times)
    else:
        # 批量计时：减少同步开销
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat


def check_tensor_properties(tensor, name):
    """检查 tensor 的属性"""
    print(f"  {name}:")
    print(f"    shape: {tensor.shape}")
    print(f"    dtype: {tensor.dtype}")
    print(f"    device: {tensor.device}")
    print(f"    is_contiguous: {tensor.is_contiguous()}")
    print(f"    stride: {tensor.stride()}")
    print(f"    data_ptr: {tensor.data_ptr()}")
    print(f"    storage_offset: {tensor.storage_offset()}")


def main():
    print("=" * 80)
    print("诊断 Kernel 性能差异")
    print("=" * 80)
    
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
    M, N, K = 32, 896, 896  # decode 场景
    
    # ========================================================================
    # 测试 1: 标准连续 tensor
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 1: 标准连续 tensor (与 test_02_kernel.py 一致)")
    print("=" * 80)
    
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_fp8 = torch.randn(N, K, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    weight_scale = torch.ones(N, 1, dtype=torch.float32, device=device)
    bias = torch.zeros(N, dtype=torch.bfloat16, device=device)
    
    check_tensor_properties(input_bf16, "input_bf16")
    check_tensor_properties(weight_fp8, "weight_fp8")
    
    # 测试 quant kernel
    def run_quant():
        return quant_only_fp8_kernel(input_bf16)
    
    quant_time = profile_with_cuda_events(run_quant, warmup=50, repeat=500)
    print(f"\nQuant kernel: {quant_time:.4f} ms ({quant_time*1000:.2f} us)")
    
    # 测试 GEMM kernel
    ext = _get_gemm_extension("cublaslt")
    qinput, scale_a = quant_only_fp8_kernel(input_bf16)
    
    def run_gemm():
        return ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
    
    gemm_time = profile_with_cuda_events(run_gemm, warmup=50, repeat=500)
    print(f"GEMM kernel: {gemm_time:.4f} ms ({gemm_time*1000:.2f} us)")
    
    # 测试 dequant kernel
    gemm_out = ext.cublaslt_fp8_mm(weight_fp8, qinput, get_inner_dtype_str())
    
    def run_dequant():
        return dequant_bias_kernel(gemm_out[:M], scale_a[:M], weight_scale, bias, torch.bfloat16)
    
    dequant_time = profile_with_cuda_events(run_dequant, warmup=50, repeat=500)
    print(f"Dequant kernel: {dequant_time:.4f} ms ({dequant_time*1000:.2f} us)")
    
    total_time = quant_time + gemm_time + dequant_time
    print(f"\n总计: {total_time:.4f} ms ({total_time*1000:.2f} us)")
    
    # ========================================================================
    # 测试 2: view 操作后的 tensor
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 2: view 操作后的 tensor (模拟端到端场景)")
    print("=" * 80)
    
    # 端到端场景：input 可能来自 3D tensor 的 view
    input_3d = torch.randn(1, M, K, dtype=torch.bfloat16, device=device)
    input_bf16_view = input_3d.view(M, K)  # view 操作，不改变底层存储
    
    check_tensor_properties(input_bf16_view, "input_bf16_view")
    
    def run_quant_view():
        return quant_only_fp8_kernel(input_bf16_view)
    
    quant_time_view = profile_with_cuda_events(run_quant_view, warmup=50, repeat=500)
    print(f"\nQuant kernel (view): {quant_time_view:.4f} ms ({quant_time_view*1000:.2f} us)")
    print(f"差异: {quant_time_view/quant_time:.2f}x")
    
    # ========================================================================
    # 测试 3: 非连续 tensor (slice)
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 3: 非连续 tensor (slice)")
    print("=" * 80)
    
    # 创建一个更大的 tensor，然后 slice
    large_input = torch.randn(M * 2, K, dtype=torch.bfloat16, device=device)
    input_slice = large_input[:M]  # slice，stride 不变但可能有不同的内存对齐
    
    check_tensor_properties(input_slice, "input_slice")
    
    def run_quant_slice():
        return quant_only_fp8_kernel(input_slice)
    
    quant_time_slice = profile_with_cuda_events(run_quant_slice, warmup=50, repeat=500)
    print(f"\nQuant kernel (slice): {quant_time_slice:.4f} ms ({quant_time_slice*1000:.2f} us)")
    print(f"差异: {quant_time_slice/quant_time:.2f}x")
    
    # ========================================================================
    # 测试 4: contiguous() 调用
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 4: contiguous() 开销")
    print("=" * 80)
    
    def run_quant_contiguous():
        return quant_only_fp8_kernel(input_bf16_view.contiguous())
    
    quant_time_contig = profile_with_cuda_events(run_quant_contiguous, warmup=50, repeat=500)
    print(f"\nQuant kernel (contiguous): {quant_time_contig:.4f} ms ({quant_time_contig*1000:.2f} us)")
    print(f"差异: {quant_time_contig/quant_time:.2f}x")
    
    # ========================================================================
    # 测试 5: 模拟端到端的完整 Op.apply 调用
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 5: 完整 Op.apply 调用")
    print("=" * 80)
    
    from slidesparse.core.SlideSparseLinearMethod_FP8 import SlideSparseFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    op = SlideSparseFp8LinearOp(
        act_quant_static=False,
        act_quant_group_shape=GroupShape.PER_TOKEN,
    )
    
    def run_op_apply():
        return op.apply(
            input=input_bf16,
            weight=weight_fp8,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            bias=bias,
        )
    
    apply_time = profile_with_cuda_events(run_op_apply, warmup=50, repeat=500)
    print(f"\nOp.apply: {apply_time:.4f} ms ({apply_time*1000:.2f} us)")
    print(f"理论最小 (quant+gemm+dequant): {total_time:.4f} ms")
    print(f"Op 开销: {(apply_time - total_time)*1000:.2f} us ({(apply_time/total_time - 1)*100:.1f}%)")
    
    # ========================================================================
    # 测试 6: 不同 M 值的影响
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 6: 不同 M 值的影响")
    print("=" * 80)
    
    for test_M in [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
        test_input = torch.randn(test_M, K, dtype=torch.bfloat16, device=device)
        
        def run_quant_M():
            return quant_only_fp8_kernel(test_input)
        
        t = profile_with_cuda_events(run_quant_M, warmup=20, repeat=100)
        print(f"  M={test_M:>4}: quant {t:.4f} ms ({t*1000:.2f} us)")
    
    # ========================================================================
    # 测试 7: Profile 计时开销
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 7: Profile 计时开销")
    print("=" * 80)
    
    # 直接测量 CUDA event record 开销
    def empty_fn():
        pass
    
    # 方法 1: 外部计时
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(1000):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"CUDA event record (无同步): {elapsed/1000*1e6:.2f} us/iter")
    
    # 方法 2: 带同步
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(100):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        end.record()
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time
    print(f"CUDA event record (带同步): {elapsed/100*1e6:.2f} us/iter")
    
    # ========================================================================
    # 测试 8: 热/冷启动差异
    # ========================================================================
    print("\n" + "=" * 80)
    print("测试 8: 热/冷启动差异（清除 GPU 缓存）")
    print("=" * 80)
    
    # 分配大量内存来"冲刷"缓存
    def flush_gpu_cache():
        dummy = torch.randn(1024, 1024, 1024, dtype=torch.float32, device=device)
        del dummy
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # 冷启动
    flush_gpu_cache()
    cold_time = profile_with_cuda_events(run_quant, warmup=0, repeat=1, sync_every_iter=True)
    print(f"冷启动首次调用: {cold_time:.4f} ms")
    
    # 热启动
    hot_time = profile_with_cuda_events(run_quant, warmup=10, repeat=100)
    print(f"热启动平均: {hot_time:.4f} ms")
    print(f"冷/热比: {cold_time/hot_time:.2f}x")
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("诊断总结")
    print("=" * 80)
    print(f"""
标准连续 tensor:
  - Quant:   {quant_time:.4f} ms ({quant_time*1000:.2f} us)
  - GEMM:    {gemm_time:.4f} ms ({gemm_time*1000:.2f} us)
  - Dequant: {dequant_time:.4f} ms ({dequant_time*1000:.2f} us)
  - 总计:    {total_time:.4f} ms ({total_time*1000:.2f} us)

Op.apply 开销:
  - Op.apply:   {apply_time:.4f} ms
  - 理论最小:   {total_time:.4f} ms
  - Python 开销: {(apply_time - total_time)*1000:.2f} us ({(apply_time/total_time - 1)*100:.1f}%)

View 操作影响:
  - view tensor:   {quant_time_view/quant_time:.2f}x
  - slice tensor:  {quant_time_slice/quant_time:.2f}x
  - contiguous:    {quant_time_contig/quant_time:.2f}x
""")


if __name__ == "__main__":
    main()
