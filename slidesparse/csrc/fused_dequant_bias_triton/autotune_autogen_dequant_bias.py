#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Dequant + Bias Kernel Autotune & Code Generation Script

生成的 kernel 支持 BF16 和 FP32 两种输入（自动转为 FP32 计算，输出 BF16）

Usage:
    python3 autotune_autogen_dequant_bias.py
    python3 autotune_autogen_dequant_bias.py --quick

Output:
    build/dequant_bias_tuned_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.py
    例如: dequant_bias_tuned_H100_cc90_py312_cu124_x86_64.py
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

import torch
import triton
import triton.language as tl

# 设置路径以导入 slidesparse 模块
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent.parent  # slidesparse/
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent       # vllmbench/

# 将项目根目录添加到 sys.path以支持 "from slidesparse.utils import ..."
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

from utils import get_dequant_autotune_configs


def get_output_filename() -> str:
    """Generate output filename: dequant_bias_tuned_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.py"""
    return build_filename("dequant_bias_tuned", ext=".py")


# Get autotune configs from utils
AUTOTUNE_CONFIGS = get_dequant_autotune_configs()


# =============================================================================
# Test Matrix Sizes
# =============================================================================

N_VALUES = [2560, 3840, 13824]  # BitNet hidden sizes

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
# Autotune Kernel (with reduced warmup/rep for faster tuning)
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'N'],
    warmup=5,
    rep=30,
)
@triton.jit
def _dequant_bias_kernel_autotune(
    gemm_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, out_ptr,
    M, N,
    stride_gm, stride_gn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, INPUT_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_2d = mask_m[:, None] & mask_n[None, :]
    
    scale_a = tl.load(scale_a_ptr + offs_m, mask=mask_m, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    
    gemm_offs = offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    val = tl.load(gemm_ptr + gemm_offs, mask=mask_2d, other=0.0)
    
    if not INPUT_FP32:
        val = val.to(tl.float32)
    
    val = val * scale_a[:, None] * scale_b[None, :] + bias[None, :]
    
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=mask_2d)


def dequant_bias_autotune(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    M, N = gemm_output.shape
    input_fp32 = gemm_output.dtype == torch.float32
    
    scale_a = scale_a.view(-1).contiguous().float() if scale_a.numel() > 1 else scale_a.view(1).expand(M).contiguous().float()
    scale_b = scale_b.view(-1).contiguous().float() if scale_b.numel() > 1 else scale_b.view(1).expand(N).contiguous().float()
    bias = bias.view(-1).contiguous().to(torch.bfloat16)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    _dequant_bias_kernel_autotune[grid](
        gemm_output, scale_a, scale_b, bias, output,
        M, N,
        gemm_output.stride(0), gemm_output.stride(1),
        output.stride(0), output.stride(1),
        INPUT_FP32=input_fp32,
    )
    return output


# =============================================================================
# Tuning Runner
# =============================================================================

def run_tuning():
    """Run autotune and collect best configs for each (M, N)"""
    # 使用 BF16 进行 autotune（生成的 kernel 同样支持 FP32 输入）
    input_dtype = torch.bfloat16
    
    print(f"\nTuning (input: BF16, kernel supports both BF16/FP32)...")
    print(f"N values: {N_VALUES}")
    print(f"M values: {len(M_VALUES)} points")
    print("=" * 70)
    
    results = {}
    max_M, max_N = max(M_VALUES), max(N_VALUES)
    
    # Pre-allocate buffers
    gemm_buf = torch.randn(max_M, max_N, dtype=input_dtype, device="cuda")
    scale_a_buf = torch.rand(max_M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    scale_b_buf = torch.rand(max_N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    bias_buf = torch.randn(max_N, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    
    for n in N_VALUES:
        results[n] = {}
        print(f"\n[N={n}]")
        
        for m in M_VALUES:
            gemm = gemm_buf[:m, :n].contiguous()
            scale_a = scale_a_buf[:m]
            scale_b = scale_b_buf[:n]
            bias = bias_buf[:n]
            
            try:
                # Run autotune
                dequant_bias_autotune(gemm, scale_a, scale_b, bias)
                torch.cuda.synchronize()
                
                # Extract best config from cache
                best_cfg = None
                for key, cfg in _dequant_bias_kernel_autotune.cache.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        cached_m, cached_n = key[0], key[1]
                        if cached_m == m and cached_n == n:
                            best_cfg = cfg
                            break
                
                if best_cfg:
                    results[n][m] = {
                        'BLOCK_M': best_cfg.kwargs['BLOCK_M'],
                        'BLOCK_N': best_cfg.kwargs['BLOCK_N'],
                        'num_warps': best_cfg.num_warps,
                        'num_stages': best_cfg.num_stages,
                    }
                    cfg = results[n][m]
                    print(f"  M={m:<6} -> ({cfg['BLOCK_M']:>3}, {cfg['BLOCK_N']:>3}) w={cfg['num_warps']:<2} s={cfg['num_stages']}")
                else:
                    print(f"  M={m:<6} -> [cache miss]")
                    
            except Exception as e:
                print(f"  M={m:<6} -> ERROR: {e}")
    
    return results


def build_branches(results):
    """Analyze results and build interval-based branch strategy"""
    branches = {}
    
    for n, m_configs in results.items():
        sorted_ms = sorted(m_configs.keys())
        if not sorted_ms:
            continue
        
        intervals = []
        prev_key = None
        interval_start = None
        
        for m in sorted_ms:
            cfg = m_configs[m]
            cfg_key = (cfg['BLOCK_M'], cfg['BLOCK_N'], cfg['num_warps'], cfg['num_stages'])
            
            if cfg_key != prev_key:
                if prev_key is not None:
                    intervals.append((interval_start, m, m_configs[interval_start]))
                interval_start = m
                prev_key = cfg_key
        
        if interval_start is not None:
            intervals.append((interval_start, None, m_configs[interval_start]))
        
        branches[n] = intervals
    
    return branches


# =============================================================================
# Code Generator
# =============================================================================

def generate_kernel_code(branches) -> str:
    """Generate the tuned kernel Python file"""
    
    # Generate config selector function
    def gen_config_selector():
        lines = ["def _get_config(M: int, N: int) -> tuple:"]
        lines.append('    """Returns (BLOCK_M, BLOCK_N, num_warps, num_stages)"""')
        
        n_values = sorted(branches.keys())
        for i, n in enumerate(n_values):
            cond = "if" if i == 0 else "elif"
            lines.append(f"    {cond} N == {n}:")
            
            intervals = branches.get(n, [])
            if not intervals:
                lines.append("        return 64, 64, 8, 4")
                continue
            
            for j, (m_start, m_end, cfg) in enumerate(intervals):
                ret = f"{cfg['BLOCK_M']}, {cfg['BLOCK_N']}, {cfg['num_warps']}, {cfg['num_stages']}"
                if j == 0:
                    if m_end is None:
                        lines.append(f"        return {ret}")
                    else:
                        lines.append(f"        if M < {m_end}:")
                        lines.append(f"            return {ret}")
                elif m_end is None:
                    lines.append(f"        return {ret}")
                else:
                    lines.append(f"        elif M < {m_end}:")
                    lines.append(f"            return {ret}")
        
        # Default fallback for unknown N
        lines.append("    if M <= 128:")
        lines.append("        return 32, 64, 4, 4")
        lines.append("    elif M <= 4096:")
        lines.append("        return 64, 64, 8, 4")
        lines.append("    return 128, 64, 8, 4")
        
        return "\n".join(lines)
    
    config_selector = gen_config_selector()
    
    code = f'''# Auto-generated by autotune_autogen_dequant_bias.py
# Target: {get_gpu_name()} ({get_gpu_cc()})
# Supports both BF16 and FP32 input (auto-converts to FP32 for computation)
# DO NOT EDIT

import torch
import triton
import triton.language as tl

{config_selector}


@triton.jit
def _dequant_bias_kernel(
    gemm_ptr, scale_a_ptr, scale_b_ptr, bias_ptr, out_ptr,
    M, N, stride_gm, stride_gn, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_2d = mask_m[:, None] & mask_n[None, :]
    
    scale_a = tl.load(scale_a_ptr + offs_m, mask=mask_m, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=mask_n, other=1.0)
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    
    gemm_offs = offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    val = tl.load(gemm_ptr + gemm_offs, mask=mask_2d, other=0.0)
    val = val.to(tl.float32) * scale_a[:, None] * scale_b[None, :] + bias[None, :]
    
    out_offs = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptr + out_offs, val.to(tl.bfloat16), mask=mask_2d)


def dequant_bias_triton(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert gemm_output.is_cuda and gemm_output.is_contiguous()
    M, N = gemm_output.shape
    
    scale_a = scale_a.view(-1).contiguous().float() if scale_a.numel() > 1 else scale_a.expand(M).contiguous().float()
    scale_b = scale_b.view(-1).contiguous().float() if scale_b.numel() > 1 else scale_b.expand(N).contiguous().float()
    bias = bias.view(-1).contiguous().to(torch.bfloat16)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    BLOCK_M, BLOCK_N, num_warps, num_stages = _get_config(M, N)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    _dequant_bias_kernel[grid](
        gemm_output, scale_a, scale_b, bias, output,
        M, N,
        gemm_output.stride(0), gemm_output.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        num_warps=num_warps, num_stages=num_stages,
    )
    return output.to(out_dtype) if out_dtype != torch.bfloat16 else output


__all__ = ['dequant_bias_triton', '_get_config']
'''
    return code


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dequant+Bias Kernel Autotune & Codegen")
    parser.add_argument('--info', action='store_true', help='Show naming info only')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: ./build)')
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
    print("Dequant + Bias Kernel Autotune")
    print("=" * 70)
    print(f"GPU:     {get_gpu_name()} ({get_gpu_cc()})")
    print(f"Python:  {get_python_version_tag()}")
    print(f"Arch:    {get_arch_tag()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Input:   BF16/FP32 (auto-handled)")
    print(f"Output:  {get_output_filename()}")
    
    if args.info:
        return 0
    
    # Step 1: Run autotune
    print("\n" + "=" * 70)
    print("Step 1: Running autotune...")
    print("=" * 70)
    results = run_tuning()
    
    # Step 2: Build branches
    print("\n" + "=" * 70)
    print("Step 2: Building branch strategy...")
    print("=" * 70)
    branches = build_branches(results)
    
    for n, intervals in branches.items():
        print(f"\nN={n}: {len(intervals)} intervals")
        for m_start, m_end, cfg in intervals:
            end_str = f"< {m_end}" if m_end else "to max"
            print(f"  M >= {m_start:<5} {end_str:<12} -> ({cfg['BLOCK_M']}, {cfg['BLOCK_N']}) w={cfg['num_warps']} s={cfg['num_stages']}")
    
    # Step 3: Generate code
    print("\n" + "=" * 70)
    print("Step 3: Generating kernel code...")
    print("=" * 70)
    
    kernel_code = generate_kernel_code(branches)
    
    # Determine output path
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "build"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / get_output_filename()
    
    with open(output_file, "w") as f:
        f.write(kernel_code)
    
    print(f"\nGenerated: {output_file}")
    print(f"Size: {len(kernel_code)} bytes")
    print("\nDone!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
