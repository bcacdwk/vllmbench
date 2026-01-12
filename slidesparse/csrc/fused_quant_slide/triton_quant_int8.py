"""
Triton INT8 Quantization Kernel

统一配置:
- BLOCK_K: 每次处理的元素数 (2的幂次)
- autotune key: K
- 最全面的配置覆盖
"""

import torch
import triton
import triton.language as tl


def get_autotune_configs():
    """最全面的 autotune 配置"""
    return [
        # Tier 1: 小 BLOCK_K (适合小 K 或高寄存器压力场景)
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        # Tier 2: 中等 BLOCK_K (通用场景)
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=4),
        # Tier 3: 大 BLOCK_K (H100 高带宽利用)
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=5),
        # Tier 4: 超大 BLOCK_K (大 K 高吞吐场景)
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 16384}, num_warps=8, num_stages=3),
    ]


@triton.autotune(configs=get_autotune_configs(), key=['K'])
@triton.jit
def _triton_quant_int8_kernel(
    x_ptr, y_ptr, scale_ptr,
    M, K,
    stride_x, stride_y,
    BLOCK_K: tl.constexpr,
):
    """Triton INT8 per-row symmetric quantization"""
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y

    # Pass 1: 计算 absmax
    absmax = tl.zeros((), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(xb), axis=0))

    absmax = tl.maximum(absmax, 1e-5)
    scale = absmax / 127.0
    inv_scale = 127.0 / absmax
    tl.store(scale_ptr + row, scale)

    # Pass 2: 量化
    for k in range(0, K, BLOCK_K):
        offs = k + tl.arange(0, BLOCK_K)
        mask = offs < K
        xb = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        yb = tl.extra.cuda.libdevice.rint(xb * inv_scale).to(tl.int8)
        tl.store(y_row + offs, yb, mask=mask)


def triton_quant_int8(x: torch.Tensor):
    """
    Triton INT8 per-row symmetric quantization
    
    Args:
        x: Input tensor [M, K], bf16/fp16/fp32
    
    Returns:
        y: Quantized tensor [M, K], int8
        scale: Scale tensor [M], fp32
    """
    assert x.dim() == 2, f"Expected 2D tensor, got {x.dim()}D"
    M, K = x.shape
    
    y = torch.empty(M, K, dtype=torch.int8, device=x.device)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    
    _triton_quant_int8_kernel[(M,)](
        x, y, scale, M, K, x.stride(0), y.stride(0),
    )
    return y, scale


# 兼容旧接口
triton_quant = triton_quant_int8
_triton_quant_kernel = _triton_quant_int8_kernel
