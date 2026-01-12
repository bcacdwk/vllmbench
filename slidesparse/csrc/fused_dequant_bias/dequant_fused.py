def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequant + Bias 操作
    
    计算: output = gemm_output * scale_a[M,1] * scale_b[1,N] + bias[1,N]
    
    TODO: 实现 Triton kernel 以获得更好的性能
          当前使用 PyTorch 原生操作作为 placeholder
    
    Args:
        gemm_output: GEMM 输出 [M, N]，inner_dtype（BF16 或 FP32）
        scale_a: 输入 scale [M, 1] 或 [1] FP32
        scale_b: 权重 scale [N, 1] 或 [1] FP32
        bias: 偏置 [N] BF16 或 None
        out_dtype: 输出数据类型
        
    Returns:
        dequant 后的输出 [M, N]，out_dtype
    """
    M, N = gemm_output.shape
    
    # 准备 scale 的广播形状
    # scale_a: [M, 1] 或 [1] -> [M, 1]
    if scale_a.numel() == 1:
        scale_a_broadcast = scale_a.view(1, 1)
    else:
        scale_a_broadcast = scale_a.view(-1, 1)
    
    # scale_b: [N, 1] 或 [1] -> [1, N]
    if scale_b.numel() == 1:
        scale_b_broadcast = scale_b.view(1, 1)
    else:
        scale_b_broadcast = scale_b.view(1, -1)
    
    # 计算 dequant: output = gemm_output * scale_a * scale_b
    # 先转为 FP32 计算以保证精度
    output = gemm_output.float() * scale_a_broadcast * scale_b_broadcast
    
    # 加 bias
    if bias is not None and bias.numel() > 0:
        output = output + bias.float().view(1, -1)
    
    # 转换为目标精度
    return output.to(out_dtype)