#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Quant + Slide Kernel Autotune & Code Generation Script

基于 per-row kernel 设计：
- 每行一个 program（grid = M）
- BLOCK_GROUPS 是主要调优参数
- L 和 NUM_WINDOWS 作为 constexpr，kernel 会为每个 L 值编译一次

统一 FP8/INT8 设计：
- 生成单一文件，包含 FP8 和 INT8 两个函数
- 只用 FP8 + L=8 进行 autotune（IO-bound kernel，config 通用）
- 如果 FP8 不支持则 fallback 到 INT8 autotune

Usage:
    python3 autotune_autogen_quant_slide.py           # 默认 FP8 autotune
    python3 autotune_autogen_quant_slide.py --quick   # 快速模式

Output:
    build/quant_slide_tuned_{GPU}_{CC}_{PyVer}_{CUDAVer}_{Arch}.py
"""

import os
import sys
import argparse

# 优先使用系统 CUDA ptxas
_CUDA_PTXAS = "/usr/local/cuda/bin/ptxas"
if os.path.exists(_CUDA_PTXAS) and "TRITON_PTXAS_PATH" not in os.environ:
    os.environ["TRITON_PTXAS_PATH"] = _CUDA_PTXAS

import torch
import triton
import triton.language as tl
from pathlib import Path
from typing import Tuple

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


def get_output_filename() -> str:
    """Generate output filename (unified, no dtype suffix)"""
    return build_filename("quant_slide_tuned", ext=".py")


# =============================================================================
# Autotune Configs
# =============================================================================

def get_quant_slide_autotune_configs():
    """
    获取 quant_slide kernel 的 Triton autotune 配置
    
    主要参数:
    - BLOCK_GROUPS: 每次处理的组数
    - num_warps: warp 数量
    - num_stages: pipeline stages
    
    Note: BLOCK_K 固定为 2048/4096，不在 autotune 范围内
    """
    return [
        # Small M (1-64): lower parallelism
        triton.Config({'BLOCK_GROUPS': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_GROUPS': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_GROUPS': 64}, num_warps=4, num_stages=2),
        
        triton.Config({'BLOCK_GROUPS': 128}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_GROUPS': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_GROUPS': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 128}, num_warps=4, num_stages=3),
        
        # Medium M (64-4096): balanced
        triton.Config({'BLOCK_GROUPS': 256}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_GROUPS': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_GROUPS': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_GROUPS': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 256}, num_warps=8, num_stages=3),
        
        # Large M (4096+): higher parallelism
        triton.Config({'BLOCK_GROUPS': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_GROUPS': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 512}, num_warps=16, num_stages=3),
        
        # Even larger
        triton.Config({'BLOCK_GROUPS': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_GROUPS': 1024}, num_warps=16, num_stages=3),
    ]


AUTOTUNE_CONFIGS = get_quant_slide_autotune_configs()


# =============================================================================
# Test Matrix Sizes
# =============================================================================

K_VALUES = [2560, 6912]

M_VALUES = [
    1, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096,
    8192, 16384, 32768, 65536
]

# Fixed L for autotune (config is transferable to other L values)
AUTOTUNE_L = 8


# =============================================================================
# Helper Functions
# =============================================================================

def _get_num_windows(L: int) -> int:
    """Calculate number of windows: L/2 - 1"""
    return L // 2 - 1


def _compute_output_k(K_in: int, L: int) -> Tuple[int, int, int]:
    """Compute output dimensions"""
    K_in_padded = ((K_in + L - 1) // L) * L
    num_groups = K_in_padded // L
    num_windows = _get_num_windows(L)
    K_out = num_groups * num_windows * 4
    return K_in_padded, K_out, num_groups


def _get_block_k(K: int) -> int:
    """Get BLOCK_K for Pass 1"""
    if K <= 2048:
        return 2048
    elif K <= 4096:
        return 4096
    else:
        return 4096


# =============================================================================
# FP8 Autotune Kernel
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'K_in_orig'],
    warmup=5,
    rep=30,
)
@triton.jit
def _quant_slide_fp8_kernel_autotune(
    x_ptr, out_ptr, scale_ptr,
    M, K_in_orig, K_in_padded, K_out, num_groups,
    stride_x, stride_out,
    L: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP8 Quant + Slide kernel for autotuning"""
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row = x_ptr + row * stride_x
    out_row32 = out_ptr.to(tl.pointer_type(tl.int32)) + row * (stride_out // 4)
    
    # Pass 1: Compute absmax
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K_in_padded, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K_in_orig
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb)))
    
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: Quant + Slide
    for g_start in range(0, num_groups, BLOCK_GROUPS):
        offs_g = tl.arange(0, BLOCK_GROUPS)
        gid = g_start + offs_g
        mask_g = gid < num_groups
        base_in = gid * L
        base_out = gid * NUM_WINDOWS
        
        for w in tl.static_range(NUM_WINDOWS):
            win_start = 2 * w
            
            x0 = tl.load(x_row + base_in + win_start + 0, 
                        mask=mask_g & ((base_in + win_start + 0) < K_in_orig), other=0.0).to(tl.float32)
            x1 = tl.load(x_row + base_in + win_start + 1,
                        mask=mask_g & ((base_in + win_start + 1) < K_in_orig), other=0.0).to(tl.float32)
            x2 = tl.load(x_row + base_in + win_start + 2,
                        mask=mask_g & ((base_in + win_start + 2) < K_in_orig), other=0.0).to(tl.float32)
            x3 = tl.load(x_row + base_in + win_start + 3,
                        mask=mask_g & ((base_in + win_start + 3) < K_in_orig), other=0.0).to(tl.float32)
            
            q0 = tl.clamp(x0 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q1 = tl.clamp(x1 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q2 = tl.clamp(x2 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q3 = tl.clamp(x3 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            
            b0 = q0.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b1 = q1.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b2 = q2.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b3 = q3.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            
            packed = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.int32)
            tl.store(out_row32 + base_out + w, packed, mask=mask_g)


# =============================================================================
# INT8 Autotune Kernel
# =============================================================================

@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=['M', 'K_in_orig'],
    warmup=5,
    rep=30,
)
@triton.jit
def _quant_slide_int8_kernel_autotune(
    x_ptr, out_ptr, scale_ptr,
    M, K_in_orig, K_in_padded, K_out, num_groups,
    stride_x, stride_out,
    L: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """INT8 Quant + Slide kernel for autotuning"""
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row = x_ptr + row * stride_x
    out_row32 = out_ptr.to(tl.pointer_type(tl.int32)) + row * (stride_out // 4)
    
    # Pass 1: Compute absmax
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K_in_padded, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K_in_orig
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb)))
    
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: Quant + Slide
    for g_start in range(0, num_groups, BLOCK_GROUPS):
        offs_g = tl.arange(0, BLOCK_GROUPS)
        gid = g_start + offs_g
        mask_g = gid < num_groups
        base_in = gid * L
        base_out = gid * NUM_WINDOWS
        
        for w in tl.static_range(NUM_WINDOWS):
            win_start = 2 * w
            
            x0 = tl.load(x_row + base_in + win_start + 0, 
                        mask=mask_g & ((base_in + win_start + 0) < K_in_orig), other=0.0).to(tl.float32)
            x1 = tl.load(x_row + base_in + win_start + 1,
                        mask=mask_g & ((base_in + win_start + 1) < K_in_orig), other=0.0).to(tl.float32)
            x2 = tl.load(x_row + base_in + win_start + 2,
                        mask=mask_g & ((base_in + win_start + 2) < K_in_orig), other=0.0).to(tl.float32)
            x3 = tl.load(x_row + base_in + win_start + 3,
                        mask=mask_g & ((base_in + win_start + 3) < K_in_orig), other=0.0).to(tl.float32)
            
            q0 = tl.clamp(tl.extra.cuda.libdevice.rint(x0 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q1 = tl.clamp(tl.extra.cuda.libdevice.rint(x1 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q2 = tl.clamp(tl.extra.cuda.libdevice.rint(x2 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q3 = tl.clamp(tl.extra.cuda.libdevice.rint(x3 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            
            packed = (q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)).to(tl.int32)
            tl.store(out_row32 + base_out + w, packed, mask=mask_g)


# =============================================================================
# Autotune Wrapper Functions
# =============================================================================

def quant_slide_fp8_autotune(x: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP8 quant+slide with autotune"""
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    num_windows = _get_num_windows(L)
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_out_padded = ((K_out + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    out = torch.zeros(M_padded, K_out_padded, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    block_k = _get_block_k(K_in_orig)
    
    _quant_slide_fp8_kernel_autotune[(M,)](
        x, out, scale,
        M, K_in_orig, K_in_padded, K_out, num_groups,
        x.stride(0), K_out_padded,  # output stride 使用 K_out_padded
        L=L,
        NUM_WINDOWS=num_windows,
        BLOCK_K=block_k,
    )
    return out, scale


def quant_slide_int8_autotune(x: torch.Tensor, L: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """INT8 quant+slide with autotune"""
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    num_windows = _get_num_windows(L)
    
    # Padding: K -> 32 aligned, M -> 16 aligned
    K_out_padded = ((K_out + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    out = torch.zeros(M_padded, K_out_padded, dtype=torch.int8, device=x.device)
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    block_k = _get_block_k(K_in_orig)
    
    _quant_slide_int8_kernel_autotune[(M,)](
        x, out, scale,
        M, K_in_orig, K_in_padded, K_out, num_groups,
        x.stride(0), K_out_padded,  # output stride 使用 K_out_padded
        L=L,
        NUM_WINDOWS=num_windows,
        BLOCK_K=block_k,
    )
    return out, scale


# =============================================================================
# Tuning Runner
# =============================================================================

def check_fp8_support():
    """Check if the GPU supports FP8"""
    cc = get_gpu_cc()
    # cc format is "cc90", "cc89", "cc100", etc.
    cc_num = cc.replace("cc", "")
    cc_major = int(cc_num)
    return cc_major >= 89


def run_tuning():
    """Run autotune with FP8 and L=8"""
    use_fp8 = check_fp8_support()
    dtype_name = "FP8" if use_fp8 else "INT8 (fallback)"
    quant_func = quant_slide_fp8_autotune if use_fp8 else quant_slide_int8_autotune
    kernel_cache = _quant_slide_fp8_kernel_autotune if use_fp8 else _quant_slide_int8_kernel_autotune
    
    print(f"\nTuning with {dtype_name}, L={AUTOTUNE_L}...")
    if not use_fp8:
        print("  Note: FP8 not supported, using INT8 for tuning.")
    print(f"K values: {K_VALUES}")
    print(f"M values: {M_VALUES}")
    print(f"Configs: {len(AUTOTUNE_CONFIGS)}")
    print("=" * 70)
    
    results = {}
    max_M, max_K = max(M_VALUES), max(K_VALUES)
    
    x_buf = torch.randn(max_M, max_K, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    
    for k in K_VALUES:
        results[k] = {}
        print(f"\n[K={k}]")
        
        for m in M_VALUES:
            x = x_buf[:m, :k].contiguous()
            
            for _ in range(3):
                _ = quant_func(x, AUTOTUNE_L)
            torch.cuda.synchronize()
            
            best_config = None
            for key, cfg in kernel_cache.cache.items():
                if isinstance(key, tuple) and len(key) >= 2:
                    cached_m, cached_k = key[0], key[1]
                    if cached_m == m and cached_k == k:
                        best_config = cfg
                        break
            
            if best_config:
                block_groups = best_config.kwargs.get('BLOCK_GROUPS', 0)
                num_warps = best_config.num_warps
                num_stages = best_config.num_stages
                
                results[k][m] = (block_groups, num_warps, num_stages)
                print(f"  M={m:<6} -> BLOCK_GROUPS={block_groups:<4} num_warps={num_warps:<2} num_stages={num_stages}")
            else:
                print(f"  M={m:<6} -> [WARN] No cache entry")
    
    return results


def build_branches(results):
    """Build interval-based branch strategy"""
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
            key = cfg
            
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

def generate_kernel_code(branches, results) -> str:
    """Generate the tuned kernel Python file with both FP8 and INT8"""
    
    def gen_config_selector():
        lines = ["def _get_config(M: int, K: int) -> tuple:"]
        lines.append('    """Returns (BLOCK_GROUPS, num_warps, num_stages)"""')
        
        k_values = sorted(branches.keys())
        for i, k in enumerate(k_values):
            cond = "if" if i == 0 else "elif"
            lines.append(f"    {cond} K == {k}:")
            
            intervals = branches[k]
            for j, (m_start, m_end, cfg) in enumerate(intervals):
                block_groups, num_warps, num_stages = cfg
                if j == 0:
                    lines.append(f"        if M <= {m_end}:")
                elif j == len(intervals) - 1:
                    lines.append(f"        else:")
                else:
                    lines.append(f"        elif M <= {m_end}:")
                lines.append(f"            return {block_groups}, {num_warps}, {num_stages}")
        
        lines.append("    # Default fallback")
        lines.append("    if K <= 4096:")
        lines.append("        return 256, 8, 2")
        lines.append("    return 256, 8, 2")
        
        return "\n".join(lines)
    
    config_selector = gen_config_selector()
    
    code = f'''# Auto-generated by autotune_autogen_quant_slide.py
# Target: {get_gpu_name()} ({get_gpu_cc()})
# Design: Per-row kernel (grid = M), Unified FP8/INT8, L as constexpr
# DO NOT EDIT

import torch
import triton
import triton.language as tl
from typing import Tuple

{config_selector}


def _get_num_windows(L: int) -> int:
    """Calculate number of windows: L/2 - 1"""
    return L // 2 - 1


def _compute_output_k(K_in: int, L: int) -> Tuple[int, int, int]:
    """Compute output dimensions"""
    K_in_padded = ((K_in + L - 1) // L) * L
    num_groups = K_in_padded // L
    num_windows = _get_num_windows(L)
    K_out = num_groups * num_windows * 4
    return K_in_padded, K_out, num_groups


def _get_block_k(K: int) -> int:
    """Get BLOCK_K for Pass 1"""
    if K <= 2048:
        return 2048
    elif K <= 4096:
        return 4096
    else:
        return 4096


# =============================================================================
# FP8 Kernel
# =============================================================================

@triton.jit
def _quant_slide_fp8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K_in_orig, K_in_padded, K_out, num_groups,
    stride_x, stride_out,
    L: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-token FP8 Quant + Slide kernel"""
    row = tl.program_id(0)
    
    FP8_MAX: tl.constexpr = 448.0
    MIN_SCALE: tl.constexpr = 1.0 / (448.0 * 512.0)
    
    x_row = x_ptr + row * stride_x
    out_row32 = out_ptr.to(tl.pointer_type(tl.int32)) + row * (stride_out // 4)
    
    # Pass 1: Compute absmax
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K_in_padded, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K_in_orig
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb)))
    
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / FP8_MAX, MIN_SCALE)
    inv_scale = FP8_MAX / absmax
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: Quant + Slide
    for g_start in range(0, num_groups, BLOCK_GROUPS):
        offs_g = tl.arange(0, BLOCK_GROUPS)
        gid = g_start + offs_g
        mask_g = gid < num_groups
        base_in = gid * L
        base_out = gid * NUM_WINDOWS
        
        for w in tl.static_range(NUM_WINDOWS):
            win_start = 2 * w
            
            x0 = tl.load(x_row + base_in + win_start + 0, 
                        mask=mask_g & ((base_in + win_start + 0) < K_in_orig), other=0.0).to(tl.float32)
            x1 = tl.load(x_row + base_in + win_start + 1,
                        mask=mask_g & ((base_in + win_start + 1) < K_in_orig), other=0.0).to(tl.float32)
            x2 = tl.load(x_row + base_in + win_start + 2,
                        mask=mask_g & ((base_in + win_start + 2) < K_in_orig), other=0.0).to(tl.float32)
            x3 = tl.load(x_row + base_in + win_start + 3,
                        mask=mask_g & ((base_in + win_start + 3) < K_in_orig), other=0.0).to(tl.float32)
            
            q0 = tl.clamp(x0 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q1 = tl.clamp(x1 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q2 = tl.clamp(x2 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            q3 = tl.clamp(x3 * inv_scale, -FP8_MAX, FP8_MAX).to(tl.float8e4nv)
            
            b0 = q0.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b1 = q1.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b2 = q2.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            b3 = q3.to(tl.int8, bitcast=True).to(tl.int32) & 0xFF
            
            packed = (b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)).to(tl.int32)
            tl.store(out_row32 + base_out + w, packed, mask=mask_g)


def quant_slide_fp8_triton(
    x: torch.Tensor,
    L: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert x.is_cuda and x.is_contiguous()
    assert L >= 4 and L % 2 == 0
    
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    num_windows = _get_num_windows(L)
    
    K_out_padded = ((K_out + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    out = torch.zeros(M_padded, K_out_padded, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    BLOCK_GROUPS, num_warps, num_stages = _get_config(M, K_in_orig)
    block_k = _get_block_k(K_in_orig)
    
    _quant_slide_fp8_kernel[(M,)](
        x, out, scale,
        M, K_in_orig, K_in_padded, K_out, num_groups,
        x.stride(0), K_out_padded,  # output stride 使用 K_out_padded
        L=L,
        NUM_WINDOWS=num_windows,
        BLOCK_GROUPS=BLOCK_GROUPS,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scale


# =============================================================================
# INT8 Kernel
# =============================================================================

@triton.jit
def _quant_slide_int8_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K_in_orig, K_in_padded, K_out, num_groups,
    stride_x, stride_out,
    L: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Per-token INT8 Quant + Slide kernel"""
    row = tl.program_id(0)
    
    INT8_MAX: tl.constexpr = 127.0
    MIN_SCALE: tl.constexpr = 1.0 / (127.0 * 512.0)
    
    x_row = x_ptr + row * stride_x
    out_row32 = out_ptr.to(tl.pointer_type(tl.int32)) + row * (stride_out // 4)
    
    # Pass 1: Compute absmax
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K_in_padded, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K_in_orig
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb)))
    
    absmax = tl.maximum(absmax, 1e-12)
    scale = tl.maximum(absmax / INT8_MAX, MIN_SCALE)
    inv_scale = INT8_MAX / absmax
    tl.store(scale_ptr + row, scale)
    
    # Pass 2: Quant + Slide
    for g_start in range(0, num_groups, BLOCK_GROUPS):
        offs_g = tl.arange(0, BLOCK_GROUPS)
        gid = g_start + offs_g
        mask_g = gid < num_groups
        base_in = gid * L
        base_out = gid * NUM_WINDOWS
        
        for w in tl.static_range(NUM_WINDOWS):
            win_start = 2 * w
            
            x0 = tl.load(x_row + base_in + win_start + 0, 
                        mask=mask_g & ((base_in + win_start + 0) < K_in_orig), other=0.0).to(tl.float32)
            x1 = tl.load(x_row + base_in + win_start + 1,
                        mask=mask_g & ((base_in + win_start + 1) < K_in_orig), other=0.0).to(tl.float32)
            x2 = tl.load(x_row + base_in + win_start + 2,
                        mask=mask_g & ((base_in + win_start + 2) < K_in_orig), other=0.0).to(tl.float32)
            x3 = tl.load(x_row + base_in + win_start + 3,
                        mask=mask_g & ((base_in + win_start + 3) < K_in_orig), other=0.0).to(tl.float32)
            
            q0 = tl.clamp(tl.extra.cuda.libdevice.rint(x0 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q1 = tl.clamp(tl.extra.cuda.libdevice.rint(x1 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q2 = tl.clamp(tl.extra.cuda.libdevice.rint(x2 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            q3 = tl.clamp(tl.extra.cuda.libdevice.rint(x3 * inv_scale), -128.0, 127.0).to(tl.int32) & 0xFF
            
            packed = (q0 | (q1 << 8) | (q2 << 16) | (q3 << 24)).to(tl.int32)
            tl.store(out_row32 + base_out + w, packed, mask=mask_g)


def quant_slide_int8_triton(
    x: torch.Tensor,
    L: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:

    assert x.is_cuda and x.is_contiguous()
    assert L >= 4 and L % 2 == 0
    
    M, K_in_orig = x.shape
    K_in_padded, K_out, num_groups = _compute_output_k(K_in_orig, L)
    num_windows = _get_num_windows(L)
    
    K_out_padded = ((K_out + 31) // 32) * 32
    M_padded = ((M + 15) // 16) * 16
    
    out = torch.zeros(M_padded, K_out_padded, dtype=torch.int8, device=x.device)
    scale = torch.ones(M_padded, dtype=torch.float32, device=x.device)
    
    BLOCK_GROUPS, num_warps, num_stages = _get_config(M, K_in_orig)
    block_k = _get_block_k(K_in_orig)
    
    _quant_slide_int8_kernel[(M,)](
        x, out, scale,
        M, K_in_orig, K_in_padded, K_out, num_groups,
        x.stride(0), K_out_padded,  # output stride 使用 K_out_padded
        L=L,
        NUM_WINDOWS=num_windows,
        BLOCK_GROUPS=BLOCK_GROUPS,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, scale


__all__ = ['quant_slide_fp8_triton', 'quant_slide_int8_triton', '_get_config', '_compute_output_k', '_get_num_windows']
'''
    return code


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Quant + Slide Kernel Autotune & Codegen (Unified FP8/INT8)")
    parser.add_argument('--info', action='store_true', help='Show naming info only')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--quick', action='store_true', 
                        help='Quick mode: use fewer M values for faster testing')
    args = parser.parse_args()
    
    global M_VALUES
    if args.quick:
        M_VALUES = [16, 128, 1024, 4096, 16384]
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return 1
    
    use_fp8 = check_fp8_support()
    tune_dtype = "FP8" if use_fp8 else "INT8 (fallback)"
    
    print("=" * 70)
    print("Quant + Slide Kernel Autotune (Per-Row Design, Unified FP8/INT8)")
    print("=" * 70)
    print(f"GPU:     {get_gpu_name()} ({get_gpu_cc()})")
    print(f"Python:  {get_python_version_tag()}")
    print(f"Arch:    {get_arch_tag()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton:  {triton.__version__}")
    print(f"Tune dtype: {tune_dtype}")
    print(f"Tune L: {AUTOTUNE_L}")
    print(f"Output file: {get_output_filename()}")
    
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
    
    for k, intervals in branches.items():
        print(f"\nK={k}: {len(intervals)} intervals")
        for m_start, m_end, cfg in intervals:
            block_groups, num_warps, num_stages = cfg
            print(f"  M=[{m_start}, {m_end}] -> BLOCK_GROUPS={block_groups}, warps={num_warps}, stages={num_stages}")
    
    # Step 3: Generate code
    print("\n" + "=" * 70)
    print("Step 3: Generating kernel code...")
    print("=" * 70)
    
    kernel_code = generate_kernel_code(branches, results)
    
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
