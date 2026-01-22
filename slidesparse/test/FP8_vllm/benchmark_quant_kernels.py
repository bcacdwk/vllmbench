#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
benchmark_quant_kernels.py - Quant Kernel 性能基准测试

对比三个 backend 的 quant kernel 性能：
  1. CUTLASS: vLLM 原生 QuantFP8 (ops.scaled_fp8_quant)
  2. cuBLASLt: Triton quant_only_fp8 kernel
  3. cuSPARSELt: Triton quant_slide_fp8 kernel

测试维度基于 Qwen2.5-0.5B 的线性层尺寸：
  - K=896 (hidden_size): q_proj, k_proj, v_proj, o_proj
  - K=4864 (intermediate_size): gate_proj, up_proj, down_proj

M 值对应：
  - Prefill: M=2048
  - Decode:  M=32

使用方法:
    python3 benchmark_quant_kernels.py
    python3 benchmark_quant_kernels.py --warmup 10 --repeat 100
    python3 benchmark_quant_kernels.py --M 32,2048 --K 896,4864
"""

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Callable

import torch
import torch.cuda

# 设置 Triton ptxas 路径
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ============================================================================
# 配置
# ============================================================================

# Qwen2.5-0.5B 的 K 值
QWEN_K_VALUES = [896, 4864]

# 测试 M 值
DEFAULT_M_VALUES = [32, 2048]

# 默认测试参数
DEFAULT_WARMUP = 10
DEFAULT_REPEAT = 100


# ============================================================================
# CUDA 计时工具
# ============================================================================

class CUDATimer:
    """精确的 CUDA 计时器（使用 CUDA Events）"""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
    def start(self):
        self.start_event.record()
        
    def stop(self) -> float:
        """返回毫秒"""
        self.end_event.record()
        torch.cuda.synchronize()
        return self.start_event.elapsed_time(self.end_event)


def benchmark_fn(
    fn: Callable,
    warmup: int = 10,
    repeat: int = 100,
) -> Tuple[float, float, float]:
    """
    基准测试一个函数
    
    Returns:
        (mean_ms, min_ms, max_ms)
    """
    timer = CUDATimer()
    
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(repeat):
        timer.start()
        fn()
        elapsed = timer.stop()
        times.append(elapsed)
    
    mean_ms = sum(times) / len(times)
    min_ms = min(times)
    max_ms = max(times)
    
    return mean_ms, min_ms, max_ms


# ============================================================================
# Quant Kernel 加载器
# ============================================================================

_cutlass_quant_fn = None
_triton_quant_only_fn = None
_triton_quant_slide_fn = None


def load_cutlass_quant():
    """加载 vLLM 原生 QuantFP8"""
    global _cutlass_quant_fn
    if _cutlass_quant_fn is not None:
        return _cutlass_quant_fn
    
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    
    def quant_fn(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """CUTLASS quant: 使用 vLLM 原生 scaled_fp8_quant"""
        return ops.scaled_fp8_quant(
            x,
            scale=None,  # 动态量化
            num_token_padding=None,
            scale_ub=None,
            use_per_token_if_dynamic=True,
        )
    
    _cutlass_quant_fn = quant_fn
    return quant_fn


def load_triton_quant_only():
    """加载 Triton quant_only_fp8 kernel (cuBLASLt 路径)"""
    global _triton_quant_only_fn
    if _triton_quant_only_fn is not None:
        return _triton_quant_only_fn
    
    from slidesparse.utils import load_module, find_file
    
    kernel_dir = Path(__file__).parent.parent.parent / "csrc" / "quant_only_triton"
    build_dir = kernel_dir / "build"
    
    module = load_module("quant_only_tuned", search_dir=build_dir)
    _triton_quant_only_fn = module.quant_only_fp8_triton
    return _triton_quant_only_fn


def load_triton_quant_slide():
    """加载 Triton quant_slide_fp8 kernel (cuSPARSELt 路径)"""
    global _triton_quant_slide_fn
    if _triton_quant_slide_fn is not None:
        return _triton_quant_slide_fn
    
    from slidesparse.utils import load_module, find_file
    
    kernel_dir = Path(__file__).parent.parent.parent / "csrc" / "fused_quant_slide_triton"
    build_dir = kernel_dir / "build"
    
    module = load_module("quant_slide_tuned", search_dir=build_dir)
    _triton_quant_slide_fn = module.quant_slide_fp8_triton
    return _triton_quant_slide_fn


# ============================================================================
# 基准测试函数
# ============================================================================

@dataclass
class BenchResult:
    """单个基准测试结果"""
    backend: str
    M: int
    K: int
    mean_ms: float
    min_ms: float
    max_ms: float
    output_shape: Tuple[int, ...]
    
    @property
    def throughput_gb_s(self) -> float:
        """计算吞吐量 (GB/s)，基于输入数据量"""
        # 输入: [M, K] BF16 (2 bytes)
        input_bytes = self.M * self.K * 2
        return (input_bytes / 1e9) / (self.mean_ms / 1e3)


def benchmark_cutlass_quant(
    M: int, K: int, warmup: int, repeat: int
) -> BenchResult:
    """基准测试 CUTLASS (vLLM QuantFP8)"""
    quant_fn = load_cutlass_quant()
    
    # 创建输入
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    # 获取输出形状
    qx, scale = quant_fn(x)
    output_shape = tuple(qx.shape)
    
    # 清理
    del qx, scale
    torch.cuda.empty_cache()
    
    # 基准测试
    def run():
        quant_fn(x)
    
    mean_ms, min_ms, max_ms = benchmark_fn(run, warmup, repeat)
    
    return BenchResult(
        backend="CUTLASS (QuantFP8)",
        M=M, K=K,
        mean_ms=mean_ms, min_ms=min_ms, max_ms=max_ms,
        output_shape=output_shape,
    )


def benchmark_triton_quant_only(
    M: int, K: int, warmup: int, repeat: int
) -> BenchResult:
    """基准测试 Triton quant_only (cuBLASLt 路径)"""
    quant_fn = load_triton_quant_only()
    
    # 创建输入
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    # 获取输出形状
    qx, scale = quant_fn(x)
    output_shape = tuple(qx.shape)
    
    # 清理
    del qx, scale
    torch.cuda.empty_cache()
    
    # 基准测试
    def run():
        quant_fn(x)
    
    mean_ms, min_ms, max_ms = benchmark_fn(run, warmup, repeat)
    
    return BenchResult(
        backend="Triton quant_only",
        M=M, K=K,
        mean_ms=mean_ms, min_ms=min_ms, max_ms=max_ms,
        output_shape=output_shape,
    )


def benchmark_triton_quant_slide(
    M: int, K: int, warmup: int, repeat: int, L: int = 8
) -> BenchResult:
    """基准测试 Triton quant_slide (cuSPARSELt 路径)"""
    quant_fn = load_triton_quant_slide()
    
    # 创建输入
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    
    # 获取输出形状
    qx, scale = quant_fn(x, L)
    output_shape = tuple(qx.shape)
    
    # 清理
    del qx, scale
    torch.cuda.empty_cache()
    
    # 基准测试
    def run():
        quant_fn(x, L)
    
    mean_ms, min_ms, max_ms = benchmark_fn(run, warmup, repeat)
    
    return BenchResult(
        backend=f"Triton quant_slide (L={L})",
        M=M, K=K,
        mean_ms=mean_ms, min_ms=min_ms, max_ms=max_ms,
        output_shape=output_shape,
    )


# ============================================================================
# 主测试函数
# ============================================================================

def run_all_benchmarks(
    M_values: List[int],
    K_values: List[int],
    warmup: int,
    repeat: int,
    L: int = 8,
) -> List[BenchResult]:
    """运行所有基准测试"""
    results = []
    
    print(f"\n{'=' * 100}")
    print("Quant Kernel 基准测试")
    print(f"{'=' * 100}")
    print(f"M 值: {M_values}")
    print(f"K 值: {K_values}")
    print(f"Warmup: {warmup}, Repeat: {repeat}")
    print(f"L (quant_slide): {L}")
    print(f"{'=' * 100}\n")
    
    # 预加载所有 kernel
    print("加载 kernels...")
    try:
        load_cutlass_quant()
        print("  ✓ CUTLASS (QuantFP8)")
    except Exception as e:
        print(f"  ✗ CUTLASS 加载失败: {e}")
    
    try:
        load_triton_quant_only()
        print("  ✓ Triton quant_only")
    except Exception as e:
        print(f"  ✗ Triton quant_only 加载失败: {e}")
    
    try:
        load_triton_quant_slide()
        print("  ✓ Triton quant_slide")
    except Exception as e:
        print(f"  ✗ Triton quant_slide 加载失败: {e}")
    
    print()
    
    for M in M_values:
        for K in K_values:
            print(f"\n{'─' * 80}")
            print(f"测试 M={M}, K={K}")
            print(f"{'─' * 80}")
            
            # CUTLASS
            try:
                result = benchmark_cutlass_quant(M, K, warmup, repeat)
                results.append(result)
                print(f"  {result.backend:<30} "
                      f"mean={result.mean_ms:>8.4f} ms, "
                      f"min={result.min_ms:>8.4f} ms, "
                      f"max={result.max_ms:>8.4f} ms, "
                      f"out={result.output_shape}")
            except Exception as e:
                print(f"  CUTLASS 失败: {e}")
            
            # Triton quant_only
            try:
                result = benchmark_triton_quant_only(M, K, warmup, repeat)
                results.append(result)
                print(f"  {result.backend:<30} "
                      f"mean={result.mean_ms:>8.4f} ms, "
                      f"min={result.min_ms:>8.4f} ms, "
                      f"max={result.max_ms:>8.4f} ms, "
                      f"out={result.output_shape}")
            except Exception as e:
                print(f"  Triton quant_only 失败: {e}")
            
            # Triton quant_slide
            try:
                result = benchmark_triton_quant_slide(M, K, warmup, repeat, L)
                results.append(result)
                print(f"  {result.backend:<30} "
                      f"mean={result.mean_ms:>8.4f} ms, "
                      f"min={result.min_ms:>8.4f} ms, "
                      f"max={result.max_ms:>8.4f} ms, "
                      f"out={result.output_shape}")
            except Exception as e:
                print(f"  Triton quant_slide 失败: {e}")
    
    return results


def print_summary_table(results: List[BenchResult]):
    """打印汇总表格"""
    print(f"\n{'=' * 120}")
    print("汇总表格 (mean time in ms)")
    print(f"{'=' * 120}")
    
    # 按 M, K 分组
    from collections import defaultdict
    grouped = defaultdict(dict)
    for r in results:
        key = (r.M, r.K)
        grouped[key][r.backend] = r.mean_ms
    
    # 表头
    backends = ["CUTLASS (QuantFP8)", "Triton quant_only", "Triton quant_slide (L=8)"]
    header = f"{'M':>8} │ {'K':>8} │"
    for b in backends:
        header += f" {b:<25} │"
    print(header)
    print("─" * len(header))
    
    # 数据行
    for (M, K), times in sorted(grouped.items()):
        row = f"{M:>8} │ {K:>8} │"
        cutlass_time = times.get("CUTLASS (QuantFP8)", float('inf'))
        
        for b in backends:
            t = times.get(b)
            if t is not None:
                # 计算相对 CUTLASS 的比值
                if cutlass_time > 0 and b != "CUTLASS (QuantFP8)":
                    ratio = t / cutlass_time
                    row += f" {t:>8.4f} ms ({ratio:>5.2f}x) │"
                else:
                    row += f" {t:>8.4f} ms         │"
            else:
                row += f" {'N/A':>18} │"
        print(row)
    
    print(f"{'=' * 120}")
    print("\n注: 比值 >1.0 表示比 CUTLASS 慢，<1.0 表示比 CUTLASS 快")


# ============================================================================
# 命令行接口
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quant Kernel 基准测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--M", type=str, default=",".join(map(str, DEFAULT_M_VALUES)),
        help=f"M 值列表 (逗号分隔)，默认: {DEFAULT_M_VALUES}"
    )
    parser.add_argument(
        "--K", type=str, default=",".join(map(str, QWEN_K_VALUES)),
        help=f"K 值列表 (逗号分隔)，默认: {QWEN_K_VALUES}"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"预热次数，默认: {DEFAULT_WARMUP}"
    )
    parser.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEAT,
        help=f"重复次数，默认: {DEFAULT_REPEAT}"
    )
    parser.add_argument(
        "--L", type=int, default=8,
        help="quant_slide 的 L 参数，默认: 8"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    M_values = [int(x.strip()) for x in args.M.split(",")]
    K_values = [int(x.strip()) for x in args.K.split(",")]
    
    # 打印 GPU 信息
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")
    
    # 运行基准测试
    results = run_all_benchmarks(
        M_values=M_values,
        K_values=K_values,
        warmup=args.warmup,
        repeat=args.repeat,
        L=args.L,
    )
    
    # 打印汇总
    print_summary_table(results)


if __name__ == "__main__":
    main()
