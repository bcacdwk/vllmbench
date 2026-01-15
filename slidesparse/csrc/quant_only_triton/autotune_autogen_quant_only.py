#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quant Only Kernel Autotune & Code Generation Script

基于 per-row kernel 设计：
- 每行一个 program（grid = M）
- BLOCK_K 是主要调优参数
- M 虽然不在 kernel 块参数中，但影响最佳 num_warps/num_stages
- autotune key = ['M', 'K']

Usage:
    python3 autotune_autogen_quant_only.py --dtype fp8   # FP8E4M3 output (default)
    python3 autotune_autogen_quant_only.py --dtype int8  # INT8 output
    python3 autotune_autogen_quant_only.py --quick --dtype fp8

Output:
    build/quant_only_tuned_{GPU}_{CC}_{dtype}_{PyVer}_{CUDAVer}_{Arch}.py
"""

import os
import sys
import argparse

# 优先使用系统 CUDA ptxas（支持更新的 GPU 架构如 sm_121）
# Triton 内置的 ptxas 版本可能较旧，不支持最新架构
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

import torch
import triton
import triton.language as tl
from pathlib import Path

# 设置路径以导入 slidesparse 模块
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent.parent  # slidesparse/
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent       # vllmbench/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    build_filename,
    get_python_version_tag,
    get_arch_tag,
    get_gpu_cc,
    get_gpu_name,
)

# 将 csrc 目录添加到 sys.path 以导入 utils
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from utils import get_quant_autotune_configs


def get_output_filename(dtype: str) -> str:
    """Generate output filename"""
    dtype_tag = dtype.upper()
    return build_filename("quant_only_tuned", dtype=dtype_tag, ext=".py")


# Get autotune configs
AUTOTUNE_CONFIGS = get_quant_autotune_configs()


# =============================================================================
# Test Matrix Sizes
# =============================================================================

K_VALUES = [2560, 6912]  # Hidden sizes for quantization

# M search strategy:
# - Small M (1-512): Dense search - high variance in optimal config
# - Medium M (512-4096): Medium density
# - Large M (4096+): Sparse search - configs converge to similar values
M_VALUES = [
    # Small M: dense (high variance)
    1, 16, 32, 64, 128, 256, 512,
    # Medium M: medium density
    1024, 2048, 4096,
    # Large M: sparse (results converge)
    8192, 16384, 32768, 65536
]


# =============================================================================
# FP8 Autotune Kernel (per-row design)
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'K'],
    warmup=5,
    rep=30,
)
@triton.jit
def _quant_fp8_kernel_autotune(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """FP8 per-row quantization kernel for autotuning"""
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(x_val * inv_scale, -FP8_MAX, FP8_MAX)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.float8e4nv), mask=mask_k)


# =============================================================================
# INT8 Autotune Kernel (per-row design)
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'K'],
    warmup=5,
    rep=30,
)
@triton.jit
def _quant_int8_kernel_autotune(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """INT8 per-row quantization kernel for autotuning"""
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = tl.clamp(tl.extra.cuda.libdevice.rint(x_val * inv_scale), -128.0, 127.0)
        tl.store(out_row_ptr + offs_k, y_val.to(tl.int8), mask=mask_k)


# =============================================================================
# Autotune Wrapper Functions
# =============================================================================

def quant_fp8_autotune(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 quantization with autotune - grid = (M,)"""
    M, K = x.shape
    
    out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    
    # 每行一个 program
    _quant_fp8_kernel_autotune[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), out.stride(0),
    )
    return out, scale


def quant_int8_autotune(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 quantization with autotune - grid = (M,)"""
    M, K = x.shape
    
    out = torch.empty(M, K, dtype=torch.int8, device=x.device)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    
    # 每行一个 program
    _quant_int8_kernel_autotune[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), out.stride(0),
    )
    return out, scale


# =============================================================================
# Tuning Runner
# =============================================================================

def run_tuning(dtype: str):
    """Run autotune and collect best configs for each (M, K)"""
    dtype_name = dtype.upper()
    quant_func = quant_fp8_autotune if dtype == "fp8" else quant_int8_autotune
    kernel_cache = _quant_fp8_kernel_autotune if dtype == "fp8" else _quant_int8_kernel_autotune
    
    print(f"\nTuning for {dtype_name} output...")
    print(f"K values: {K_VALUES}")
    print(f"M values: {M_VALUES}")
    print(f"Configs: {len(AUTOTUNE_CONFIGS)}")
    print("=" * 70)
    
    results = {}
    max_M, max_K = max(M_VALUES), max(K_VALUES)
    
    # Pre-allocate buffers
    x_buf = torch.randn(max_M, max_K, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    
    for k in K_VALUES:
        results[k] = {}
        print(f"\n[K={k}]")
        
        for m in M_VALUES:
            x = x_buf[:m, :k].contiguous()
            
            # Warmup & trigger autotune
            for _ in range(3):
                _ = quant_func(x)
            torch.cuda.synchronize()
            
            # Extract best config from cache by matching (M, K) prefix
            # Cache key format varies by Triton version, so we iterate and match
            best_config = None
            for key, cfg in kernel_cache.cache.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    cached_m, cached_k = key[0], key[1]
                    if cached_m == m and cached_k == k:
                        best_config = cfg
                        break
            
            if best_config:
                block_k = best_config.kwargs.get('BLOCK_K', 0)
                num_warps = best_config.num_warps
                num_stages = best_config.num_stages
                
                results[k][m] = (block_k, num_warps, num_stages)
                print(f"  M={m:<6} -> BLOCK_K={block_k:<5} num_warps={num_warps:<2} num_stages={num_stages}")
            else:
                print(f"  M={m:<6} -> [WARN] No cache entry")
    
    return results


def build_branches(results):
    """Analyze results and build interval-based branch strategy"""
    branches = {}
    
    for k, m_configs in results.items():
        sorted_ms = sorted(m_configs.keys())
        if not sorted_ms:
            continue
        
        intervals = []
        prev_key = None
        interval_start = None
        
        for m in sorted_ms:
            cfg = m_configs[m]
            key = cfg  # (BLOCK_K, num_warps, num_stages)
            
            if key != prev_key:
                if interval_start is not None:
                    intervals.append((interval_start, prev_m, prev_key))
                interval_start = m
                prev_key = key
            prev_m = m
        
        if interval_start is not None:
            intervals.append((interval_start, prev_m, prev_key))
        
        branches[k] = intervals
    
    return branches


# =============================================================================
# Code Generator
# =============================================================================

def generate_kernel_code(branches, results, dtype: str) -> str:
    """Generate the tuned kernel Python file"""
    
    dtype_tag = dtype.upper()
    is_fp8 = dtype == "fp8"
    
    out_dtype_torch = "torch.float8_e4m3fn" if is_fp8 else "torch.int8"
    out_dtype_triton = "tl.float8e4nv" if is_fp8 else "tl.int8"
    qmax = "448.0" if is_fp8 else "127.0"
    qmin = "-448.0" if is_fp8 else "-128.0"
    min_scale_denom = "448.0" if is_fp8 else "127.0"
    need_round = "" if is_fp8 else "y_val = tl.math.round(y_val)"
    
    # Generate config selector function
    def gen_config_selector():
        lines = ["def _get_config(M: int, K: int) -> tuple:"]
        lines.append('    """Returns (BLOCK_K, num_warps, num_stages)"""')
        
        k_values = sorted(branches.keys())
        for i, k in enumerate(k_values):
            cond = "if" if i == 0 else "elif"
            lines.append(f"    {cond} K == {k}:")
            
            intervals = branches[k]
            for j, (m_start, m_end, cfg) in enumerate(intervals):
                block_k, num_warps, num_stages = cfg
                if j == 0:
                    lines.append(f"        if M <= {m_end}:")
                elif j == len(intervals) - 1:
                    lines.append(f"        else:")
                else:
                    lines.append(f"        elif M <= {m_end}:")
                lines.append(f"            return {block_k}, {num_warps}, {num_stages}")
        
        # Default fallback
        lines.append("    # Default fallback")
        lines.append("    if K <= 4096:")
        lines.append("        return 4096, 8, 2")
        lines.append("    return 8192, 8, 2")
        
        return "\n".join(lines)
    
    config_selector = gen_config_selector()
    
    code = f'''# Auto-generated by autotune_autogen_quant_only.py
# Target: {get_gpu_name()} ({get_gpu_cc()}), Output: {dtype_tag}
# Design: Per-row kernel (grid = M)
# DO NOT EDIT

import torch
import triton
import triton.language as tl

{config_selector}


@triton.jit
def _quant_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K: tl.constexpr,
    stride_xm, stride_om,
    BLOCK_K: tl.constexpr,
):
    """Per-token {dtype_tag} quantization kernel - one program per row"""
    row = tl.program_id(0)
    
    QMAX: tl.constexpr = {qmax}
    MIN_SCALE: tl.constexpr = 1.0 / ({min_scale_denom} * 512.0)
    
    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om
    
    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(x_val)))
    
    # 计算 scale
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / QMAX, MIN_SCALE)
    inv_scale = QMAX / absmax
    
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: 量化
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_val = tl.load(x_row_ptr + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        y_val = x_val * inv_scale
        {need_round}
        y_val = tl.clamp(y_val, {qmin}, {qmax})
        tl.store(out_row_ptr + offs_k, y_val.to({out_dtype_triton}), mask=mask_k)


def quant_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token {dtype_tag} quantization using tuned kernel
    
    Args:
        x: Input tensor [M, K], BF16/FP16/FP32, must be contiguous
        
    Returns:
        out: Quantized tensor [M, K], {dtype_tag}
        scale: Per-token scale [M], FP32
    """
    assert x.is_cuda and x.is_contiguous()
    M, K = x.shape
    
    out = torch.empty(M, K, dtype={out_dtype_torch}, device=x.device)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    
    BLOCK_K, num_warps, num_stages = _get_config(M, K)
    
    # Per-row: grid = (M,)
    _quant_kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), out.stride(0),
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scale


__all__ = ['quant_triton', '_get_config']
'''
    return code


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quant Only Kernel Autotune & Codegen")
    parser.add_argument('--dtype', type=str, default='fp8', choices=['fp8', 'int8'],
                        help='Output dtype: fp8 or int8 (default: fp8)')
    parser.add_argument('--info', action='store_true', help='Show naming info only')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: use fewer M values for faster testing')
    args = parser.parse_args()
    
    # Quick mode: reduce M_VALUES for faster testing
    global M_VALUES
    if args.quick:
        M_VALUES = [16, 128, 1024, 4096, 16384]
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    print("=" * 70)
    print("Quant Only Kernel Autotune (Per-Row Design)")
    print("=" * 70)
    print(f"GPU:     {get_gpu_name()} ({get_gpu_cc()})")
    print(f"Python:  {get_python_version_tag()}")
    print(f"Arch:    {get_arch_tag()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Output dtype: {args.dtype.upper()}")
    print(f"Output file:  {get_output_filename(args.dtype)}")
    
    if args.info:
        return 0
    
    # Step 1: Run autotune
    print("\n" + "=" * 70)
    print("Step 1: Running autotune...")
    print("=" * 70)
    results = run_tuning(args.dtype)
    
    # Step 2: Build branches
    print("\n" + "=" * 70)
    print("Step 2: Building branch strategy...")
    print("=" * 70)
    branches = build_branches(results)
    
    for k, intervals in branches.items():
        print(f"\nK={k}: {len(intervals)} intervals")
        for m_start, m_end, cfg in intervals:
            block_k, num_warps, num_stages = cfg
            print(f"  M=[{m_start}, {m_end}] -> BLOCK_K={block_k}, warps={num_warps}, stages={num_stages}")
    
    # Step 3: Generate code
    print("\n" + "=" * 70)
    print("Step 3: Generating kernel code...")
    print("=" * 70)
    
    kernel_code = generate_kernel_code(branches, results, args.dtype)
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "build"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / get_output_filename(args.dtype)
    
    with open(output_file, "w") as f:
        f.write(kernel_code)
    
    print(f"\nGenerated: {output_file}")
    print(f"Size: {len(kernel_code)} bytes")
    print("\nDone!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
