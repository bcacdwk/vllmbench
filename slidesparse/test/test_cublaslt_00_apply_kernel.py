#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt vs CUTLASS Apply Kernel 对比测试脚本

测试目标:
=========
1. 对比 cuBLASLt 路径和 CUTLASS 原生路径的性能
2. 验证 cuBLASLt 路径的计算正确性（与 CUTLASS 结果对比）

测试内容:
=========
完整的 Quant + GEMM + Dequant 三步流程：
- Quant: BF16 -> FP8E4M3 (使用 vLLM 官方 QuantFP8)
- GEMM: FP8 矩阵乘法 (cuBLASLt / CUTLASS)
- Dequant: FP8 -> BF16 + scale + bias

测试配置:
=========
- M: 16, 32, 64, ..., 16384 (外层循环，翻倍递增)
- (N, K): 4 种配置 (内层循环)
- Warmup: 25 次
- Test: 100 次

两条路径:
=========
1. CUTLASS (baseline): vLLM 原生 Fp8LinearOp.apply()
   - 完整路径: quant + GEMM (cutlass_scaled_mm) + dequant
   
2. cuBLASLt (test):    外挂 CuBLASLtFp8LinearOp.apply()
   - 完整路径: quant + GEMM (cublaslt_fp8_mm) + dequant
   - 启用方式: USE_CUBLASLT=1

两者输入完全一致 (A, W, scale, bias)，直接比较输出的 BF16 结果。

使用方法:
=========
# 默认使用 CUTLASS fallback
python3 slidesparse/test/test_cublaslt_00_apply_kernel.py

# 启用 cuBLASLt kernel
USE_CUBLASLT=1 python3 slidesparse/test/test_cublaslt_00_apply_kernel.py

"""

import os
import sys

# ============================================================================
# 抑制 vLLM 内部的警告和 INFO 日志（必须在 import vllm 之前设置）
# 这些日志是因为我们直接使用底层模块而不通过 vLLM 的正常启动流程
# "Current vLLM config is not set" - 会使用默认配置，不影响测试
# "Chunked prefill is enabled" - scheduler 配置，我们直接调用 GEMM 不经过 scheduler
# ============================================================================
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

import time
from typing import Tuple

import torch

# 确保可以导入 slidesparse 和 vllm
sys.path.insert(0, '/root/vllmbench')

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


def get_fp8_dtype():
    """获取当前平台的 FP8 数据类型"""
    return torch.float8_e4m3fn


def generate_test_data(
    M: int,
    N: int,
    K: int,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成固定的测试数据
    
    Returns:
        input_bf16: [M, K] BF16 输入
        weight_fp8_t: [K, N] FP8 权重（已转置, column-major）
        weight_scale: [N, 1] FP32 权重 scale (per-channel)
        bias: [N] BF16 偏置
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # 1. 生成 BF16 输入
    input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    
    # 2. 生成 FP8 权重 (先生成 BF16，然后量化)
    weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1
    
    # 3. 量化权重为 FP8 (per-channel)
    fp8_max = torch.finfo(get_fp8_dtype()).max
    weight_absmax = weight_bf16.abs().max(dim=1, keepdim=True).values  # [N, 1]
    weight_scale = (weight_absmax / fp8_max).to(torch.float32)  # [N, 1]
    weight_scale = torch.clamp(weight_scale, min=1e-12)
    
    # 量化: w_fp8 = w / scale
    weight_scaled = weight_bf16.float() / weight_scale
    weight_fp8 = weight_scaled.to(get_fp8_dtype())  # [N, K]
    
    # 4. 转置权重 [N, K] -> [K, N]，保持 column-major 布局
    # cutlass_scaled_mm 要求 B 是 column-major，即 stride(0) == 1
    weight_fp8_t = weight_fp8.t()  # [K, N], stride = (1, K) <- column-major
    
    # 5. 生成 bias
    bias = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.01
    
    return input_bf16, weight_fp8_t, weight_scale, bias


def benchmark_cutlass_baseline(
    input_bf16: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    warmup_iters: int = 25,
    test_iters: int = 100,
) -> Tuple[torch.Tensor, float]:
    """
    测试 CUTLASS 基线（vLLM 原生路径）
    
    路径: Fp8LinearOp.apply() -> quant + cutlass_scaled_mm + dequant
    
    Returns:
        output: 输出张量
        avg_time_ms: 平均耗时 (ms)
    """
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
    
    # 预先创建 Op 实例，避免重复创建开销
    op = Fp8LinearOp(
        act_quant_static=False,
        act_quant_group_shape=GroupShape.PER_TOKEN,
        pad_output=False,
    )
    
    # 预热
    for _ in range(warmup_iters):
        _ = op.apply(
            input=input_bf16,
            weight=weight_fp8_t,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
        )
    
    torch.cuda.synchronize()
    
    # 计时
    start_time = time.perf_counter()
    for _ in range(test_iters):
        output = op.apply(
            input=input_bf16,
            weight=weight_fp8_t,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / test_iters
    
    return output, avg_time_ms


def benchmark_cublaslt(
    input_bf16: torch.Tensor,
    weight_fp8_t: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    warmup_iters: int = 25,
    test_iters: int = 100,
) -> Tuple[torch.Tensor, float]:
    """
    测试 cuBLASLt 路径（外挂路径）
    
    路径: CuBLASLtFp8LinearOp.apply() -> quant + GEMM (将替换为cuBLASLt) + dequant
    
    Returns:
        output: 输出张量
        avg_time_ms: 平均耗时 (ms)
    """
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    
    # 预先创建 Op 实例，避免重复创建开销
    op = CuBLASLtFp8LinearOp(
        act_quant_static=False,
        act_quant_group_shape=GroupShape.PER_TOKEN,
    )
    
    # 预热
    for _ in range(warmup_iters):
        _ = op.apply(
            input=input_bf16,
            weight=weight_fp8_t,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
        )
    
    torch.cuda.synchronize()
    
    # 计时
    start_time = time.perf_counter()
    for _ in range(test_iters):
        output = op.apply(
            input=input_bf16,
            weight=weight_fp8_t,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
            input_scale=None,
            input_scale_ub=None,
            bias=bias,
        )
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time_ms = (end_time - start_time) * 1000
    avg_time_ms = total_time_ms / test_iters
    
    return output, avg_time_ms


def check_correctness(
    cutlass_output: torch.Tensor,
    cublaslt_output: torch.Tensor,
    rtol: float = 1e-1,  # FP8 精度较低，使用宽松阈值
    atol: float = 1e-1,
) -> tuple[bool, float]:
    """
    检查 cuBLASLt 输出与 CUTLASS 输出是否一致
    
    FP8 精度较低，误差在 5-10% 范围内是正常的。
    
    Returns:
        (is_match, max_diff): 是否匹配，最大差异
    """
    max_diff = (cublaslt_output - cutlass_output).abs().max().item()
    is_match = torch.allclose(cublaslt_output, cutlass_output, rtol=rtol, atol=atol)
    return is_match, max_diff


def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
    """计算 TFLOPS"""
    if time_ms <= 0:
        return 0.0
    flops = 2 * M * N * K  # GEMM 的 FLOPS
    return (flops / (time_ms / 1000)) / 1e12


def run_benchmark():
    """运行完整的对比测试"""
    
    # 测试配置
    # M: 16 到 16384，翻倍递增
    M_values = [16 * (2 ** i) for i in range(11)]  # 16, 32, 64, ..., 16384
    
    # (N, K) 配置 - 对应 Qwen 模型的典型维度
    NK_configs = [
        (896, 896),      # hidden -> hidden
        (4864, 896),     # hidden -> intermediate  
        (896, 4864),     # intermediate -> hidden
        (4096, 4096),    # 标准 GEMM
    ]
    
    warmup_iters = 25
    test_iters = 100
    
    # 打印表头
    print("=" * 130)
    print("cuBLASLt vs CUTLASS FP8 Quant+GEMM+Dequant Benchmark")
    print("=" * 130)
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Warmup: {warmup_iters}, Test iterations: {test_iters}")
    print(f"Quantization: input=per-token dynamic, weight=per-channel static")
    print("-" * 130)
    print("CUTLASS = vLLM原生路径 (Fp8LinearOp)")
    print("cuBLASLt = 外挂路径 (CuBLASLtFp8LinearOp)")
    print("=" * 130)
    print()
    
    # 表头
    header = (
        f"{'M':>6} | {'N':>5} | {'K':>5} | "
        f"{'CUTLASS(ms)':>11} | {'CUTLASS(TF)':>11} | "
        f"{'cuBLASLt(ms)':>12} | {'cuBLASLt(TF)':>12} | "
        f"{'Speedup':>7} | {'Match':>12}"
    )
    print(header)
    print("-" * 130)
    
    total_tests = 0
    passed_tests = 0    
    for M in M_values:
        for N, K in NK_configs:
            try:
                # 生成固定的测试数据
                input_bf16, weight_fp8_t, weight_scale, bias = generate_test_data(
                    M, N, K, device="cuda", seed=42
                )
                
                # 测试 CUTLASS 基线
                cutlass_output, cutlass_time = benchmark_cutlass_baseline(
                    input_bf16, weight_fp8_t, weight_scale, bias,
                    warmup_iters=warmup_iters, test_iters=test_iters
                )
                cutlass_tflops = compute_tflops(M, N, K, cutlass_time)
                
                # 测试 cuBLASLt 路径
                cublaslt_output, cublaslt_time = benchmark_cublaslt(
                    input_bf16, weight_fp8_t, weight_scale, bias,
                    warmup_iters=warmup_iters, test_iters=test_iters
                )
                cublaslt_tflops = compute_tflops(M, N, K, cublaslt_time)
                
                # 检查正确性
                is_match, max_diff = check_correctness(cutlass_output, cublaslt_output)
                
                # 计算加速比
                speedup = cutlass_time / cublaslt_time if cublaslt_time > 0 else 0.0
                
                # 打印结果
                match_str = "✓" if is_match else f"✗({max_diff:.3f})"
                print(
                    f"{M:>6} | {N:>5} | {K:>5} | "
                    f"{cutlass_time:>11.3f} | {cutlass_tflops:>11.2f} | "
                    f"{cublaslt_time:>12.3f} | {cublaslt_tflops:>12.2f} | "
                    f"{speedup:>7.2f} | {match_str:>12}"
                )
                
                total_tests += 1
                if is_match:
                    passed_tests += 1
                    
            except Exception as e:
                print(f"{M:>6} | {N:>5} | {K:>5} | ERROR: {e}")
                total_tests += 1
    
    # 汇总
    print("-" * 130)
    print(f"Total: {total_tests} tests, {passed_tests} passed, "
          f"{total_tests - passed_tests} failed")
    print("=" * 130)
    
    return passed_tests == total_tests


def main():
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("Error: CUDA is not available!")
        sys.exit(1)
    
    success = run_benchmark()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
