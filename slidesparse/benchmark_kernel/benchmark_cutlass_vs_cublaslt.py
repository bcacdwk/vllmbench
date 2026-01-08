#!/usr/bin/env python3
"""
Benchmark: CUTLASS vs cuBLASLt for FP8 GEMM

This script directly compares:
1. CUTLASS scaled_mm (vLLM's default)
2. cuBLASLt FP8 GEMM (our SlideSparse integration)

Usage:
    python benchmark_cutlass_vs_cublaslt.py
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import vLLM ops
try:
    from vllm._custom_ops import cutlass_scaled_mm
    HAS_CUTLASS = True
except ImportError:
    HAS_CUTLASS = False
    print("Warning: CUTLASS not available")

# Import cuBLASLt wrapper
try:
    from slidesparse.cublaslt_wrapper import CublasLtFp8Matmul
    HAS_CUBLASLT = True
except ImportError:
    HAS_CUBLASLT = False
    print("Warning: cuBLASLt wrapper not available")


@dataclass
class BenchmarkResult:
    """Benchmark result for a single configuration"""
    backend: str
    m: int
    n: int
    k: int
    dtype: str
    time_ms: float
    tflops: float
    memory_gb_s: float


class GEMMBenchmark:
    """GEMM Benchmark comparing CUTLASS and cuBLASLt"""
    
    def __init__(self, warmup_iters: int = 10, bench_iters: int = 100):
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters
        self.device = torch.device('cuda')
        
        # Check available backends
        self.backends = []
        if HAS_CUTLASS:
            self.backends.append('cutlass')
        if HAS_CUBLASLT:
            self.backends.append('cublaslt')
            self.cublaslt = CublasLtFp8Matmul()
            
    def _create_fp8_tensors(self, m: int, n: int, k: int) -> Tuple[torch.Tensor, ...]:
        """Create FP8 test tensors"""
        # Create FP16 tensors first, then convert
        a_fp16 = torch.randn(m, k, dtype=torch.float16, device=self.device)
        b_fp16 = torch.randn(k, n, dtype=torch.float16, device=self.device)
        
        # Convert to FP8
        a_fp8 = a_fp16.to(torch.float8_e4m3fn)
        b_fp8 = b_fp16.to(torch.float8_e4m3fn)
        
        # Scale factors (per-tensor for simplicity)
        scale_a = torch.ones(1, dtype=torch.float32, device=self.device)
        scale_b = torch.ones(1, dtype=torch.float32, device=self.device)
        
        return a_fp8, b_fp8, scale_a, scale_b
    
    def _benchmark_cutlass(self, a: torch.Tensor, b: torch.Tensor, 
                           scale_a: torch.Tensor, scale_b: torch.Tensor,
                           output_dtype: torch.dtype) -> float:
        """Benchmark CUTLASS scaled_mm"""
        # CUTLASS expects B in column-major (transposed)
        b_t = b.t().contiguous()
        
        # Warmup
        for _ in range(self.warmup_iters):
            _ = cutlass_scaled_mm(a, b_t, scale_a, scale_b, output_dtype)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(self.bench_iters):
            _ = cutlass_scaled_mm(a, b_t, scale_a, scale_b, output_dtype)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return (end - start) / self.bench_iters * 1000  # ms
    
    def _benchmark_cublaslt(self, a: torch.Tensor, b: torch.Tensor,
                            scale_a: torch.Tensor, scale_b: torch.Tensor,
                            output_dtype: torch.dtype) -> float:
        """Benchmark cuBLASLt FP8 GEMM"""
        # cuBLASLt expects specific layout
        b_t = b.t().contiguous()
        
        # Warmup
        for _ in range(self.warmup_iters):
            _ = self.cublaslt.matmul(a, b_t, scale_a, scale_b, output_dtype)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(self.bench_iters):
            _ = self.cublaslt.matmul(a, b_t, scale_a, scale_b, output_dtype)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        return (end - start) / self.bench_iters * 1000  # ms
    
    def run_benchmark(self, m: int, n: int, k: int, 
                      output_dtype: torch.dtype = torch.bfloat16) -> List[BenchmarkResult]:
        """Run benchmark for a single GEMM size"""
        results = []
        
        # Create test tensors
        a, b, scale_a, scale_b = self._create_fp8_tensors(m, n, k)
        
        # Calculate theoretical metrics
        flops = 2 * m * n * k  # multiply-add = 2 ops
        bytes_moved = (m * k + k * n + m * n) * 1  # FP8 = 1 byte
        
        for backend in self.backends:
            try:
                if backend == 'cutlass':
                    time_ms = self._benchmark_cutlass(a, b, scale_a, scale_b, output_dtype)
                elif backend == 'cublaslt':
                    time_ms = self._benchmark_cublaslt(a, b, scale_a, scale_b, output_dtype)
                else:
                    continue
                
                tflops = flops / (time_ms * 1e9)  # TFLOPS
                memory_gb_s = bytes_moved / (time_ms * 1e6)  # GB/s
                
                results.append(BenchmarkResult(
                    backend=backend,
                    m=m, n=n, k=k,
                    dtype='fp8_e4m3',
                    time_ms=time_ms,
                    tflops=tflops,
                    memory_gb_s=memory_gb_s
                ))
            except Exception as e:
                print(f"  {backend} failed: {e}")
        
        return results


def print_results(results: List[BenchmarkResult]):
    """Print benchmark results in a nice table"""
    print("\n" + "=" * 100)
    print(f"{'Backend':<12} {'M':>8} {'N':>8} {'K':>8} {'Time(ms)':>12} {'TFLOPS':>12} {'GB/s':>12}")
    print("=" * 100)
    
    for r in results:
        print(f"{r.backend:<12} {r.m:>8} {r.n:>8} {r.k:>8} {r.time_ms:>12.4f} {r.tflops:>12.2f} {r.memory_gb_s:>12.2f}")
    
    print("=" * 100)


def main():
    print("=" * 80)
    print("CUTLASS vs cuBLASLt FP8 GEMM Benchmark")
    print("=" * 80)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA Version: {torch.version.cuda}")
    
    # Initialize benchmark
    benchmark = GEMMBenchmark(warmup_iters=10, bench_iters=100)
    print(f"Available backends: {benchmark.backends}")
    
    if not benchmark.backends:
        print("Error: No backends available")
        return
    
    # Test configurations (typical LLM shapes)
    # Format: (M, N, K) - batch_size * seq_len, hidden_dim, intermediate_dim
    test_configs = [
        # Small models (Qwen2.5-0.5B like)
        (1, 1536, 896),      # Single token, down_proj
        (128, 1536, 896),    # Small batch
        (512, 1536, 896),    # Medium batch
        (1024, 1536, 896),   # Large batch
        
        # Llama3.2-1B like
        (1, 4096, 2048),     # Single token
        (128, 4096, 2048),   # Small batch
        (512, 4096, 2048),   # Medium batch
        
        # Square matrices (stress test)
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]
    
    all_results = []
    
    for m, n, k in test_configs:
        print(f"\nBenchmarking M={m}, N={n}, K={k}...")
        results = benchmark.run_benchmark(m, n, k)
        all_results.extend(results)
        
        # Print immediate results
        for r in results:
            print(f"  {r.backend}: {r.time_ms:.4f} ms, {r.tflops:.2f} TFLOPS")
    
    # Print summary
    print_results(all_results)
    
    # Calculate speedup if both backends available
    if 'cutlass' in benchmark.backends and 'cublaslt' in benchmark.backends:
        print("\n" + "=" * 60)
        print("Speedup Analysis (cuBLASLt vs CUTLASS)")
        print("=" * 60)
        
        cutlass_results = {(r.m, r.n, r.k): r for r in all_results if r.backend == 'cutlass'}
        cublaslt_results = {(r.m, r.n, r.k): r for r in all_results if r.backend == 'cublaslt'}
        
        for key in cutlass_results:
            if key in cublaslt_results:
                cutlass_time = cutlass_results[key].time_ms
                cublaslt_time = cublaslt_results[key].time_ms
                speedup = cutlass_time / cublaslt_time
                
                m, n, k = key
                winner = "cuBLASLt" if speedup > 1 else "CUTLASS"
                print(f"M={m:>5}, N={n:>5}, K={k:>5}: {speedup:.3f}x ({winner} wins)")


if __name__ == "__main__":
    main()
