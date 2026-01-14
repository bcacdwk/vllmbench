"""
BitNet权重转换脚本 - 支持按需输出指定精度（16F/16BF/8I/E4M3/E5M2/E2M1/pack2I）
本脚本将标准或已剪枝的PyTorch检查点转换为BitNet专用的三元量化格式。

输入：
- 标准的PyTorch bf16检查点文件
- 经过剪枝的bf16精度检查点文件 (see convert_checkpoint_prune_to_Z_L.py)

输出（按选择的dtype追加后缀，不覆盖原文件）：
- _16F.pt   : FP16存储的三元权重 (-1,0,1)
- _16BF.pt  : BF16存储的三元权重 (-1,0,1)
- _8I.pt    : INT8存储的三元权重 (-1,0,1)
- _E4M3.pt  : FP8 E4M3存储的三元权重 (-1,0,1)
- _E5M2.pt  : FP8 E5M2存储的三元权重 (-1,0,1)
- _E2M1.pt  : FP4 E2M1三元权重（4bit打包）
- _pack2I.pt: INT2打包存储（与原pack_weight兼容）


分别执行：
python convert_checkpoint_quant.py --input ./checkpoints/model_state.pt --dtype 8I
python convert_checkpoint_quant.py --input ./checkpoints/model_state.pt --dtype pack2I


注意：仅输出所选精度，避免生成多个文件覆盖；所有精度都会写入scale，因为 BitNet 反量化需要。
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch
from einops import rearrange
from safetensors.torch import save_file

import model_allint8 as fast

from pack_weight import convert_weight_int8_to_int2  # int8到int2的打包转换函数

@torch.inference_mode()
def convert_ts_checkpoint(
    *,
    input_path: str = "",
    dtype: str = "8I",
) -> None:

    # input_path: 输入的模型检查点文件路径
    # dtype: 目标存储精度后缀（16F/16BF/8I/E4M3/E5M2/E2M1/pack2I）
    config = fast.ModelArgs()
    print(f"Model config {config.__dict__}")
    print(f"Target dtype: {dtype}")

    supported_dtypes = {"16F", "16BF", "8I", "E4M3", "E5M2", "E2M1", "pack2I"}
    if dtype not in supported_dtypes:
        raise ValueError(f"dtype必须是{supported_dtypes}")

    def ternarize(weight):
        """
        将权重量化为三元(-1,0,1)并返回scale，用于需要scale的整数格式。
        """
        s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
        ternary = (weight * s).round().clamp(-1, 1)
        scale = (1.0 / s).to(torch.bfloat16)
        return ternary, scale.reshape(1)

    def to_int8(ternary):
        return ternary.to(torch.int8)

    def to_fp16(ternary):
        return ternary.to(torch.float16)

    def to_bf16(ternary):
        return ternary.to(torch.bfloat16)

    def to_fp8_e4m3(ternary):
        if not hasattr(torch, "float8_e4m3fn"):
            raise RuntimeError("当前PyTorch不支持float8_e4m3fn")
        return ternary.to(torch.float8_e4m3fn)

    def to_fp8_e5m2(ternary):
        if not hasattr(torch, "float8_e5m2"):
            raise RuntimeError("当前PyTorch不支持float8_e5m2")
        return ternary.to(torch.float8_e5m2)

    def to_fp4_e2m1_pack(ternary_int8: torch.Tensor):
        """
        将三元int8 (-1,0,1) 编码为FP4 E2M1 (4bit)，并打包。
        输入形状: [Rows, Cols]
        输出形状: [Rows, Cols // 2] (如果Cols是奇数则会有padding)
        """
        device = ternary_int8.device
        # 1. 编码映射 (已修正，完全正确)
        lut = torch.tensor([0xE, 0x0, 0x6], device=device, dtype=torch.uint8)
        idx = ternary_int8.add(1)
        encoded = lut[idx]
        
        # 2. 记录原始形状，用于最后恢复 2D 结构
        rows, cols = encoded.shape
        
        # 3. 处理 Padding (如果列数是奇数)
        # 展平处理最简单
        encoded_flat = encoded.flatten()
        if encoded_flat.numel() % 2 != 0:
            encoded_flat = torch.cat([encoded_flat, torch.zeros(1, device=device, dtype=torch.uint8)], dim=0)
        
        # 4. 打包
        # view 成 (-1, 2) -> 第一列高4位，第二列低4位
        pairs = encoded_flat.view(-1, 2)
        packed_flat = (pairs[:, 0] << 4) | (pairs[:, 1])
        
        # 5. 恢复 2D 形状 [Rows, Packed_Cols]
        # 计算新的列数 (向上取整)
        packed_cols = (cols + 1) // 2
        
        return packed_flat.view(rows, packed_cols)

    def to_int2_pack(ternary_int8: torch.Tensor):
        return convert_weight_int8_to_int2(ternary_int8)

    def convert_ternary(ternary, scale, mode: str):
        """
        按目标dtype将三元值转存。所有模式都返回scale，方便反量化 W ≈ W_ternary * beta。
        """
        if mode == "8I":
            return to_int8(ternary), scale
        if mode == "pack2I":
            return to_int2_pack(to_int8(ternary)), scale
        if mode == "16F":
            return to_fp16(ternary), scale
        if mode == "16BF":
            return to_bf16(ternary), scale
        if mode == "E4M3":
            return to_fp8_e4m3(ternary), scale
        if mode == "E5M2":
            return to_fp8_e5m2(ternary), scale
        if mode == "E2M1":
            return to_fp4_e2m1_pack(to_int8(ternary)), scale
        raise ValueError(f"未知dtype: {mode}")

    # 加载原始模型检查点（通常为bf16精度）
    merged_result = torch.load(input_path, map_location="cpu", mmap=True)

    if not isinstance(merged_result, dict):
        raise TypeError(
            f"Unsupported checkpoint format: expected dict but got {type(merged_result).__name__}"
        )

    print(f"Loaded checkpoint from {input_path}")
    print(f"包含参数数量: {len(merged_result)}")
    print("检查点参数顺序预览：")
    for idx, (key, value) in enumerate(merged_result.items()):
        if torch.is_tensor(value):
            shape = tuple(value.shape)
            numel = value.numel()
            val_dtype = value.dtype
            device = value.device
            requires_grad = getattr(value, "requires_grad", False)
            info = (
                f"shape={shape}, size={numel}, dtype={val_dtype}, device={device}, "
                f"requires_grad={requires_grad}"
            )
        else:
            info = f"type={type(value).__name__}"
        print(f"  [{idx:04d}] {key}: {info}")
    
    result = {}
    zero = torch.zeros(1).to(torch.bfloat16)

    # 仅处理与int8逻辑一致的矩阵权重：wqkv、w13、w2、wo，其余原样复制
    for key, value in merged_result.items():
        if torch.is_tensor(value) and value.dim() == 2 and ("wqkv" in key):
            wq = value[:config.dim]
            wk = value[config.dim:config.dim // config.n_heads * config.n_kv_heads + config.dim]
            wv = value[config.dim // config.n_heads * config.n_kv_heads + config.dim:]

            wq_t, wa_scale = ternarize(wq)
            wk_t, wb_scale = ternarize(wk)
            wv_t, wc_scale = ternarize(wv)

            wqkv_t = torch.cat([wq_t, wk_t, wv_t], dim=0)
            wqkv_weight, _ = convert_ternary(wqkv_t, wa_scale, dtype)

            result[key] = wqkv_weight
            wqkv_scale = torch.cat([wa_scale, wa_scale, wa_scale, wa_scale, wb_scale, wc_scale], dim=0)
            result[key.replace('weight', 'weight_scale')] = wqkv_scale

        elif torch.is_tensor(value) and value.dim() == 2 and ("w13" in key):
            w1 = value[:config.ffn_dim]
            w3 = value[config.ffn_dim:]

            w1_t, w1_scale = ternarize(w1)
            w3_t, w3_scale = ternarize(w3)

            w13_t = torch.cat([w1_t, w3_t], dim=0)
            w13_weight, _ = convert_ternary(w13_t, w1_scale, dtype)

            result[key] = w13_weight
            w13_scale = torch.cat([w1_scale, w3_scale, zero, zero, zero, zero], dim=0)
            result[key.replace('weight', 'weight_scale')] = w13_scale

        elif torch.is_tensor(value) and value.dim() == 2 and ("w2" in key or "wo" in key):
            t, scale = ternarize(value)
            weight_out, _ = convert_ternary(t, scale, dtype)
            result[key] = weight_out
            scale = torch.cat([scale, zero, zero, zero, zero, zero], dim=0)
            result[key.replace('weight', 'weight_scale')] = scale

        else:
            # 非目标权重（或非2D）保持原样 去掉 .clone()，直接引用，节省内存
            result[key] = value

    # 输出文件名：在原输入文件名stem后追加后缀，避免覆盖
    input_path_obj = Path(input_path)
    output_dir = input_path_obj.parent
    stem = input_path_obj.stem
    output_path = output_dir / f"{stem}_{dtype}.pt"

    print(f"Saving checkpoint to {output_path}")
    torch.save(result, output_path)
    print("权重转换完成！输出文件：")
    print(f"  - {output_path}")


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='将TorchScale检查点转换为BitNet量化格式（按需指定输出精度）')
    default_input = Path(__file__).resolve().parent / 'checkpoints' / 'model_state.pt'
    parser.add_argument(
        '--input',
        type=str,
        default=str(default_input),
        help=f'输入的模型检查点文件路径 (默认: {default_input})'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='8I',
        choices=['16F','16BF','8I','E4M3','E5M2','E2M1','pack2I'],
        help='目标存储精度后缀，仅输出一种精度，避免文件覆盖'
    )

    args = parser.parse_args()
    convert_ts_checkpoint(
        input_path=args.input,
        dtype=args.dtype,
    )
