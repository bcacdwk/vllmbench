"""
Dequant + Bias Kernel 自动调优脚本

python3 autotune_autogen_dequant_bias.py

功能:
1. 对所有 (M, N) 组合使用 autotune 找最优配置
2. 分析结果，生成最优的 if-else 分支策略
3. 自动生成一个不需要 autotune 的固定配置 Kernel 文件
"""

import torch
import triton
import triton.language as tl
from typing import Optional
import os


# =============================================================================
# Autotune 版本的 Kernel（用于找最优配置）
# =============================================================================

@triton.autotune(
    configs=[
        # Tier 1: 基础配置
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        
        # Tier 2: 中等配置
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        
        # Tier 3: 大块配置
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        
        # Tier 4: H100 优化配置
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def _dequant_bias_kernel_autotune(
    gemm_output_ptr,
    scale_a_ptr,
    scale_b_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = col_start + tl.arange(0, BLOCK_N)
    
    row_mask = row_offs < M
    col_mask = col_offs < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    
    scale_a = tl.load(scale_a_ptr + row_offs, mask=row_mask, other=1.0)
    scale_b = tl.load(scale_b_ptr + col_offs, mask=col_mask, other=1.0)
    bias = tl.load(bias_ptr + col_offs, mask=col_mask, other=0.0)
    bias = bias.to(tl.float32)
    
    gemm_offs = row_offs[:, None] * stride_gm + col_offs[None, :] * stride_gn
    gemm_val = tl.load(gemm_output_ptr + gemm_offs, mask=mask_2d, other=0.0)
    
    if not INPUT_FP32:
        gemm_val = gemm_val.to(tl.float32)
    
    output_val = gemm_val * scale_a[:, None] * scale_b[None, :]
    output_val = output_val + bias[None, :]
    output_val = output_val.to(tl.bfloat16)
    
    output_offs = row_offs[:, None] * stride_om + col_offs[None, :] * stride_on
    tl.store(output_ptr + output_offs, output_val, mask=mask_2d)


def dequant_bias_autotune(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """用于 autotune 的包装函数"""
    M, N = gemm_output.shape
    input_fp32 = gemm_output.dtype == torch.float32
    
    if scale_a.numel() == 1:
        scale_a = scale_a.view(1).expand(M).contiguous().float()
    else:
        scale_a = scale_a.view(-1).contiguous().float()
    
    if scale_b.numel() == 1:
        scale_b = scale_b.view(1).expand(N).contiguous().float()
    else:
        scale_b = scale_b.view(-1).contiguous().float()
    
    bias = bias.view(-1).contiguous().to(torch.bfloat16)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    stride_gm, stride_gn = gemm_output.stride()
    stride_om, stride_on = output.stride()
    
    _dequant_bias_kernel_autotune[grid](
        gemm_output,
        scale_a,
        scale_b,
        bias,
        output,
        M, N,
        stride_gm, stride_gn,
        stride_om, stride_on,
        INPUT_FP32=input_fp32,
    )
    
    return output


# =============================================================================
# Autotune 运行
# =============================================================================

def run_tuning(input_dtype=torch.bfloat16):
    """运行调优，返回结果字典 {N: {M: config_dict}}"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return None

    N_values = [2560, 3840, 13824]
    M_values = [
        1, 16, 32, 48, 64, 80, 96, 112, 128,
        192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
        6144, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 32768, 40960, 49152, 65536
    ]

    dtype_name = "FP32" if input_dtype == torch.float32 else "BF16"
    print(f"\nStarting offline tuning for {dtype_name} input...")
    print(f"{len(N_values)} N values x {len(M_values)} M values")
    print("=" * 80)

    results = {}
    max_N, max_M = max(N_values), max(M_values)
    
    # 预分配缓冲区
    gemm_buffer = torch.randn(max_M, max_N, dtype=input_dtype, device="cuda")
    scale_a_buffer = torch.rand(max_M, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    scale_b_buffer = torch.rand(max_N, dtype=torch.float32, device="cuda") * 0.1 + 0.01
    bias_buffer = torch.randn(max_N, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    for n in N_values:
        results[n] = {}
        print(f"\n[Tuning N={n}]")
        for m in M_values:
            print(f"  M={m:<5} ... ", end="", flush=True)
            
            gemm = gemm_buffer[:m, :n].contiguous()
            scale_a = scale_a_buffer[:m]
            scale_b = scale_b_buffer[:n]
            bias = bias_buffer[:n]
            
            try:
                # 运行 autotune
                dequant_bias_autotune(gemm, scale_a, scale_b, bias)
                torch.cuda.synchronize()
                
                # 获取最优配置
                best_config = None
                for key, config in _dequant_bias_kernel_autotune.cache.items():
                    if isinstance(key, tuple) and len(key) >= 2:
                        if key[0] == m and key[1] == n:
                            best_config = config
                            break
                
                if best_config:
                    cfg = {
                        'BLOCK_M': best_config.kwargs['BLOCK_M'],
                        'BLOCK_N': best_config.kwargs['BLOCK_N'],
                        'num_warps': best_config.num_warps,
                        'num_stages': best_config.num_stages,
                    }
                    print(f"Done. BLOCK_M={cfg['BLOCK_M']}, BLOCK_N={cfg['BLOCK_N']}, warps={cfg['num_warps']}, stages={cfg['num_stages']}")
                    results[n][m] = cfg
                else:
                    print("Config not found in cache")
            except Exception as e:
                print(f"Error: {e}")
    
    return results, M_values


def analyze_and_build_branches(results, M_values):
    """分析调优结果，构建 if-else 分支策略"""
    branches = {}
    for n, m_configs in results.items():
        sorted_ms = sorted(m_configs.keys())
        if not sorted_ms:
            continue
        
        intervals = []
        prev_cfg = None
        interval_start = None
        
        for m in sorted_ms:
            cfg = m_configs[m]
            cfg_key = (cfg['BLOCK_M'], cfg['BLOCK_N'], cfg['num_warps'], cfg['num_stages'])
            if cfg_key != prev_cfg:
                if prev_cfg is not None:
                    intervals.append((interval_start, m, m_configs[interval_start]))
                interval_start = m
                prev_cfg = cfg_key
        
        if interval_start is not None:
            intervals.append((interval_start, None, m_configs[interval_start]))
        
        branches[n] = intervals
    return branches


def generate_kernel_code(results, M_values, branches):
    """生成完整的 Kernel 代码"""
    
    def gen_m_branches(intervals, indent):
        lines = []
        if not intervals:
            lines.append(f"{indent}return 64, 64, 8, 4  # default")
            return lines
        
        for i, (m_start, m_end, cfg) in enumerate(intervals):
            ret = f"{cfg['BLOCK_M']}, {cfg['BLOCK_N']}, {cfg['num_warps']}, {cfg['num_stages']}"
            if i == 0:
                if m_end is None:
                    lines.append(f"{indent}return {ret}")
                else:
                    lines.append(f"{indent}if M < {m_end}:")
                    lines.append(f"{indent}    return {ret}")
            elif m_end is None:
                lines.append(f"{indent}else:")
                lines.append(f"{indent}    return {ret}")
            else:
                lines.append(f"{indent}elif M < {m_end}:")
                lines.append(f"{indent}    return {ret}")
        return lines
    
    def gen_config_selector():
        lines = []
        lines.append("def _get_tuned_config(M: int, N: int) -> tuple:")
        lines.append('    """根据 M, N 返回最优配置: (BLOCK_M, BLOCK_N, num_warps, num_stages)"""')
        
        n_values = sorted(branches.keys())
        for i, n in enumerate(n_values):
            if i == 0:
                lines.append(f"    if N == {n}:")
            else:
                lines.append(f"    elif N == {n}:")
            lines.extend(gen_m_branches(branches.get(n, []), "        "))
        
        lines.append("    else:")
        lines.append("        # 默认配置")
        lines.append("        if M <= 128:")
        lines.append("            return 32, 64, 4, 4")
        lines.append("        elif M <= 2048:")
        lines.append("            return 64, 64, 4, 4")
        lines.append("        else:")
        lines.append("            return 128, 64, 8, 4")
        return "\n".join(lines)

    config_selector = gen_config_selector()
    
    kernel_code = f'''"""
Auto-generated Dequant + Bias Triton Kernel (Tuned, No Autotune)
Generated by autotune_autogen_dequant_bias.py
DO NOT EDIT MANUALLY

"""

import torch
import triton
import triton.language as tl



{config_selector}



@triton.jit
def _dequant_bias_kernel_tuned(
    gemm_output_ptr,
    scale_a_ptr,
    scale_b_ptr,
    bias_ptr,
    output_ptr,
    M,
    N,
    stride_gm,
    stride_gn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INPUT_FP32: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N
    
    row_offs = row_start + tl.arange(0, BLOCK_M)
    col_offs = col_start + tl.arange(0, BLOCK_N)
    
    row_mask = row_offs < M
    col_mask = col_offs < N
    mask_2d = row_mask[:, None] & col_mask[None, :]
    
    scale_a = tl.load(scale_a_ptr + row_offs, mask=row_mask, other=1.0)
    scale_b = tl.load(scale_b_ptr + col_offs, mask=col_mask, other=1.0)
    bias = tl.load(bias_ptr + col_offs, mask=col_mask, other=0.0)
    bias = bias.to(tl.float32)
    
    gemm_offs = row_offs[:, None] * stride_gm + col_offs[None, :] * stride_gn
    gemm_val = tl.load(gemm_output_ptr + gemm_offs, mask=mask_2d, other=0.0)
    
    if not INPUT_FP32:
        gemm_val = gemm_val.to(tl.float32)
    
    output_val = gemm_val * scale_a[:, None] * scale_b[None, :]
    output_val = output_val + bias[None, :]
    
    output_val = output_val.to(tl.bfloat16)
    
    output_offs = row_offs[:, None] * stride_om + col_offs[None, :] * stride_on
    tl.store(output_ptr + output_offs, output_val, mask=mask_2d)


def dequant_bias_triton_tuned(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:

    assert gemm_output.is_cuda, "gemm_output must be on CUDA"
    assert gemm_output.is_contiguous(), "gemm_output must be contiguous"
    assert gemm_output.dtype in [torch.bfloat16, torch.float32]
    
    M, N = gemm_output.shape
    input_fp32 = gemm_output.dtype == torch.float32
    
    if scale_a.numel() == 1:
        scale_a = scale_a.view(1).expand(M).contiguous().float()
    else:
        scale_a = scale_a.view(-1).contiguous().float()
    
    if scale_b.numel() == 1:
        scale_b = scale_b.view(1).expand(N).contiguous().float()
    else:
        scale_b = scale_b.view(-1).contiguous().float()
    
    bias = bias.view(-1).contiguous().to(torch.bfloat16)
    
    output = torch.empty((M, N), dtype=torch.bfloat16, device=gemm_output.device)
    
    BLOCK_M, BLOCK_N, num_warps, num_stages = _get_tuned_config(M, N)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    stride_gm, stride_gn = gemm_output.stride()
    stride_om, stride_on = output.stride()
    
    _dequant_bias_kernel_tuned[grid](
        gemm_output,
        scale_a,
        scale_b,
        bias,
        output,
        M, N,
        stride_gm, stride_gn,
        stride_om, stride_on,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        INPUT_FP32=input_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    
    if out_dtype != torch.bfloat16:
        output = output.to(out_dtype)
    
    return output

__all__ = ['dequant_bias_triton_tuned', '_get_tuned_config']
'''
    return kernel_code


def main():
    print("=" * 80)
    print("Dequant + Bias Kernel Autotune")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    
    # 运行 BF16 输入的调优
    print("\n" + "=" * 80)
    print("Step 1: Running autotune for BF16 input...")
    print("=" * 80)
    
    result = run_tuning(torch.bfloat16)
    if result is None:
        return
    results, M_values = result
    
    # 分析结果
    print("\n" + "=" * 80)
    print("Step 2: Analyzing results and building branch strategy...")
    print("=" * 80)
    
    branches = analyze_and_build_branches(results, M_values)
    for n, intervals in branches.items():
        print(f"\nN={n}: {len(intervals)} branches")
        for m_start, m_end, cfg in intervals:
            end_str = f"< {m_end}" if m_end else "to max"
            print(f"  M >= {m_start} {end_str}: BLOCK_M={cfg['BLOCK_M']}, BLOCK_N={cfg['BLOCK_N']}, warps={cfg['num_warps']}, stages={cfg['num_stages']}")
    
    # 生成代码
    print("\n" + "=" * 80)
    print("Step 3: Generating tuned kernel code...")
    print("=" * 80)
    
    kernel_code = generate_kernel_code(results, M_values, branches)
    
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               "Tuned_Fused_Dequant_Bias_donotmodify.py")
    with open(output_file, "w") as f:
        f.write(kernel_code)
    
    print(f"\nGenerated kernel saved to: {output_file}")
    print(f"File size: {len(kernel_code)} bytes")
    print("\nDone! You can now use dequant_bias_triton_tuned() from Tuned_Fused_Dequant_Bias_donotmodify.py")


if __name__ == "__main__":
    main()
