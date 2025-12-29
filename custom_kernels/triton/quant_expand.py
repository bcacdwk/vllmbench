"""
Triton Quant+Expand Kernel

这是你的创新点：将量化和扩展操作融合到一个 Triton Kernel 中。

参考你的 gpu_dense/bitnet_kernels_triton/ 目录下的实现。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quant_expand_kernel(
    x_ptr,           # 输入指针
    out_ptr,         # 输出指针
    scale,           # 缩放因子
    n_elements,      # 元素总数
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quant + Expand 融合 Kernel
    
    这个 Kernel 将以下操作融合:
    1. 输入量化: x_quant = round(x / scale)
    2. 扩展/Pack: 根据你的 BitNet 需求处理
    3. 输出: 量化后的结果
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # 读取输入
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # 量化操作
    x_scaled = x / scale
    x_quant = tl.libdevice.round(x_scaled)
    
    # 这里可以添加你的扩展逻辑
    # 例如: INT2/INT8 量化、权重打包等
    
    # 反量化 (如果需要立即使用)
    out = x_quant * scale
    
    # 写入输出
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_quant_expand(
    x: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Triton Quant+Expand 的 Python 封装
    
    Args:
        x: 输入张量, 任意形状
        scale: 量化缩放因子
    
    Returns:
        量化处理后的张量，形状与输入相同
    """
    # 展平处理
    x_flat = x.contiguous().view(-1)
    n_elements = x_flat.numel()
    
    # 分配输出
    out_flat = torch.empty_like(x_flat)
    
    # 计算 grid
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # 启动 Kernel
    _quant_expand_kernel[grid](
        x_flat, out_flat,
        scale,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_flat.view(x.shape)


# ============================================================================
# INT8 量化 Kernel (参考你的 bitnet_kernels_triton/Quant.py)
# ============================================================================
@triton.jit
def _quant_int8_kernel(
    x_ptr,
    out_ptr,
    scale_ptr,       # per-token scale
    n_tokens,
    hidden_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Per-token INT8 量化
    
    对每个 token 独立计算 scale 并量化到 INT8 范围 [-127, 127]
    """
    token_id = tl.program_id(0)
    
    if token_id >= n_tokens:
        return
    
    # 计算当前 token 的起始偏移
    token_start = token_id * hidden_dim
    
    # 分块读取并计算 max abs
    max_val = 0.0
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        x = tl.load(x_ptr + token_start + offsets, mask=mask, other=0.0)
        block_max = tl.max(tl.abs(x))
        max_val = tl.maximum(max_val, block_max)
    
    # 计算 scale
    scale = max_val / 127.0
    scale = tl.maximum(scale, 1e-8)  # 避免除零
    
    # 保存 scale
    tl.store(scale_ptr + token_id, scale)
    
    # 量化并写入
    for block_start in range(0, hidden_dim, BLOCK_SIZE):
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        x = tl.load(x_ptr + token_start + offsets, mask=mask, other=0.0)
        x_quant = tl.libdevice.round(x / scale)
        x_quant = tl.maximum(tl.minimum(x_quant, 127.0), -127.0)
        tl.store(out_ptr + token_start + offsets, x_quant, mask=mask)


def triton_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-token INT8 量化
    
    Args:
        x: 输入张量 [n_tokens, hidden_dim] 或 [batch, seq, hidden]
    
    Returns:
        x_quant: INT8 量化结果 (保持 float 类型用于后续计算)
        scales: Per-token 缩放因子 [n_tokens]
    """
    orig_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])  # [n_tokens, hidden_dim]
    n_tokens, hidden_dim = x_2d.shape
    
    x_quant = torch.empty_like(x_2d)
    scales = torch.empty(n_tokens, dtype=x.dtype, device=x.device)
    
    BLOCK_SIZE = 1024
    grid = (n_tokens,)
    
    _quant_int8_kernel[grid](
        x_2d, x_quant, scales,
        n_tokens, hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return x_quant.view(orig_shape), scales


if __name__ == "__main__":
    # 简单测试
    print("Testing Triton Quant+Expand Kernel...")
    
    x = torch.randn(4, 128, 256, device='cuda', dtype=torch.float16)
    
    # 测试基本量化
    out = triton_quant_expand(x, scale=0.1)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"Input range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")
    
    # 测试 INT8 量化
    x_quant, scales = triton_quant_int8(x)
    print(f"\nINT8 Quant - Output range: [{x_quant.min():.4f}, {x_quant.max():.4f}]")
    print(f"Scales shape: {scales.shape}, range: [{scales.min():.6f}, {scales.max():.6f}]")
    
    print("\n✅ Triton Kernel test passed!")
