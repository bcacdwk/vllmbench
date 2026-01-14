"""
BitNet权重转换脚本 - 拆分为2:8稀疏的weight，变成结构化稀疏矩阵Wbase、补充稀疏矩阵Wresi、扩展矩阵Wexpand
（注意，前两个的大小都是N,K， 而Wexpand是N,3/2K）

输入：
- model_state_8I_pruned_2_8_magnitude.pt: 2:8稀疏的int8权重文件

输出：
1. _base: 结构化稀疏 (2:4)，保留每组前2个非零值
2. _resi: 剩余的非零值
3. _expand: 拼接 _base 和 _resi 的后半部分 (形状为原始的 1.5 倍)

分别执行：
python convert_checkpoint_expand_from_2_8.py --input ./checkpoints/model_state_8I_pruned_2_8_magnitude.pt

"""

import os
import sys
import time
import argparse
import gc  # 引入垃圾回收模块
from pathlib import Path
import torch

@torch.inference_mode()
def split_to_structured_sparse_and_expand(
    *,
    input_path: str = "",
    validate: bool = True,
    use_gpu: bool = True
) -> tuple:
    """
    将int8权重拆分为 Wbase, Wresi, Wexpand
    """
    print(f"正在从 {input_path} 加载权重...")
    start_time = time.time()
    
    # 加载权重
    # map_location='cpu' 避免直接占用GPU显存
    checkpoint = torch.load(input_path, map_location="cpu")
    
    # 结果容器
    Wbase_result = {}
    Wresi_result = {}
    Wexpand_result = {}
    
    # 设备选择
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"处理设备: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 目标层关键字（只处理包含这些字符串的层）
    # w13 通常指 Gate+Up 的融合层，wqkv 是 QKV 融合层
    TARGET_LAYERS = ['wqkv', 'wo', 'w13', 'w2']

    def is_target_layer(key_name):
        return any(t in key_name for t in TARGET_LAYERS) and ('weight' in key_name) and ('scale' not in key_name)

    # --------------------------------------------------------------------------
    # 核心拆分逻辑：使用 Cumsum 实现极速拆分
    # --------------------------------------------------------------------------
    def split_and_expand_matrix(weight_tensor):
        # 1. 准备数据
        # 移动到设备并确保是 float 或 int 类型进行计算，最后转回原类型
        # 使用 non_blocking=True 加速数据传输
        w = weight_tensor.to(device, non_blocking=True)
        N, K = w.shape
        
        if K % 4 != 0:
            raise ValueError(f"矩阵列数 {K} 不是4的倍数，无法进行2:4拆分")

        # 2. 展平为 (Total_Blocks, 4)
        flat_w = w.reshape(-1, 4)
        
        # 3. 生成非零掩码 [True, False, True, True]
        nonzero_mask = (flat_w != 0)
        
        # 4. 计算累积和 (Cumsum) - 极速核心
        # 这告诉我们这是当前行第几个非零数。例如 [1, 0, 1, 1] -> [1, 1, 2, 3]
        cumsum_mask = nonzero_mask.cumsum(dim=1)
        
        # 5. 生成保留掩码：保留非零且累积计数 <= 2 的元素
        # 这一步保证了优先保留靠前的两个非零值
        keep_mask = nonzero_mask & (cumsum_mask <= 2)
        
        # 6. 构建 Wbase (结构化稀疏)
        w_base_flat = torch.zeros_like(flat_w)
        w_base_flat[keep_mask] = flat_w[keep_mask]
        
        # 7. 构建 Wresi (剩余部分)
        # 直接相减即可得到剩余部分
        w_resi_flat = flat_w - w_base_flat
        
        # 8. 构建 Wexpand (扩展矩阵)
        # 逻辑：Wbase 完整保留 + Wresi 的后半部分 (列索引 2,3)
        # 解释：因为我们优先保留了前两个非零到 Wbase，所以 Wresi 在每4个元素的前2个位置(索引0,1)必然是0。
        # 因此，Wresi 有效信息只存在于索引 2,3。我们只拼接这一半，可以节省带宽。
        
        # 取 Wresi 的后两列 (每4列取后2列)
        # current shape: (Total_Blocks, 4) -> slice -> (Total_Blocks, 2)
        w_resi_half_flat = w_resi_flat[:, 2:]
        
        # 还原形状
        # Wbase: (N, K)
        # Wresi: (N, K)
        w_base = w_base_flat.reshape(N, K)
        w_resi = w_resi_flat.reshape(N, K)
        
        # Wexpand: 拼接 [Wbase(N, K), Wresi_half(N, K/2)] -> (N, 1.5K)
        # 先将 flat 的两部分 reshape 回 2D 空间
        # w_base 已经是 (N, K)
        # w_resi_half 需要变为 (N, K/2)
        w_resi_half = w_resi_half_flat.reshape(N, K // 2)
        
        w_expand = torch.cat([w_base, w_resi_half], dim=1)
        
        # 立即返回 CPU 释放 GPU 显存
        return w_base.to('cpu'), w_resi.to('cpu'), w_expand.to('cpu')

    # --------------------------------------------------------------------------
    # 验证逻辑
    # --------------------------------------------------------------------------
    def validate_sparsity(tensor):
        flat = tensor.reshape(-1, 4)
        counts = (flat != 0).sum(dim=1)
        return (counts <= 2).all().item()

    # 主循环
    # 优化：使用 list(keys) 静态列表，以便在循环中安全删除字典中的元素
    all_keys = list(checkpoint.keys())
    total_params = len(all_keys)
    processed_count = 0
    skipped_count = 0
    
    print(f"\n开始处理 {total_params} 个张量 (内存优化模式)...")
    print("-" * 60)

    # 显式进行第一次垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for key in all_keys:
        # 【内存优化关键点】：
        # 使用 pop 将 tensor 从原始 checkpoint 中移出。
        # 这样处理完一个 layer，原始 layer 的内存就会被释放，避免内存爆炸。
        value = checkpoint.pop(key)
        
        processed_count += 1
        
        # 1. 检查是否为需要拆分的目标层
        if is_target_layer(key) and value.dim() == 2:
            try:
                # 执行拆分
                Wbase, Wresi, Wexpand = split_and_expand_matrix(value)
                
                # 记录结果
                Wbase_result[key] = Wbase
                Wresi_result[key] = Wresi
                Wexpand_result[key] = Wexpand
                
                # 验证重构 (Wbase + Wresi == Original)
                if validate:
                    # 注意：验证时需要把 tensor 转为 float 避免 overflow，但此处为了速度保持原样
                    # 因为都是 int8 且只是加减，一般不会溢出
                    recon_err = (Wbase + Wresi - value).abs().max().item()
                    is_Wbase_sparse = validate_sparsity(Wbase)
                    
                    # 注意：Wexpand 包含 Resi 部分，Resi 部分通常是不稀疏的，
                    # 所以 Wexpand 整体不做 2:4 check 可能是合理的，或者仅做 informational check
                    is_Wexpand_sparse = validate_sparsity(Wexpand) 
                    
                    status = "✅" if (recon_err == 0 and is_Wbase_sparse) else "❌"
                    
                    # 紧凑输出
                    print(f"{status} [{processed_count}/{total_params}] 拆分: {key:<40}")
                    print(f"   Shape: 原{tuple(value.shape)} -> 扩{tuple(Wexpand.shape)}")
                    if recon_err > 0: print(f"   警告: 重构误差 {recon_err}")
                    if not is_Wbase_sparse: print(f"   警告: Wbase不符合2:4稀疏")
                    # 如果W不满足2:8稀疏，则Wexpand不满足2:4不稀疏
                    if not is_Wexpand_sparse: print(f"   提示: Wexpand不符合2:4稀疏")
                else:
                    print(f"✅ [{processed_count}/{total_params}] 拆分: {key}")

            except Exception as e:
                print(f"❌ [{processed_count}/{total_params}] 错误 {key}: {e}")
                print("   -> 回退: 原样保留，Wresi全0")
                Wbase_result[key] = value
                Wresi_result[key] = torch.zeros_like(value)
                Wexpand_result[key] = value # 保持原样防止报错
                
        else:
            # 2. 不需要拆分的层 (Embedding, Norm, 或非Target的Linear)
            skipped_count += 1
            print(f"⏩ [{processed_count}/{total_params}] 跳过: {key:<40} Shape: {tuple(value.shape)}")
            
            # 原封不动保留
            Wbase_result[key] = value
            # 对于不需要拆分的层，Wresi 通常置0或不存，为了保持一致性这里置0
            Wresi_result[key] = torch.zeros_like(value) if value.dim() > 0 else value
            # Wexpand 保持原状
            Wexpand_result[key] = value

        # 【内存优化关键点】：
        # 手动删除局部变量引用
        del value
        # 定期清理显存和内存 (每处理10个Layer清理一次，避免频繁GC降低速度)
        if processed_count % 10 == 0:
            gc.collect()
            if use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # 循环结束后，原始 checkpoint 应该已经是空的了，确保释放
    del checkpoint
    gc.collect()

    # --------------------------------------------------------------------------
    # 保存与统计
    # --------------------------------------------------------------------------
    output_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path).split('.')[0]
    
    paths = {
        "Base": os.path.join(output_dir, f"{base_name}_base.pt"),
        "Resi": os.path.join(output_dir, f"{base_name}_residual.pt"),
        "Expand": os.path.join(output_dir, f"{base_name}_expand.pt")
    }
    
    print("-" * 60)
    print("正在保存pt文件...")
    
    # 分步保存并清理，进一步降低峰值内存
    torch.save(Wbase_result, paths["Base"])
    print(f"已保存 W_base:   {paths['Base']}")
    
    # 保存完 Base 后可以考虑释放 Wbase_result 占用的内存，
    # 但由于后面还要统计非零值，这里暂时保留。
    # 如果内存极其紧张，可以先统计完再保存。
    
    torch.save(Wresi_result, paths["Resi"])
    print(f"已保存 W_resi:   {paths['Resi']}")
    
    torch.save(Wexpand_result, paths["Expand"])
    print(f"已保存 W_expand: {paths['Expand']}")
    
    # 统计非零比率
    print("正在统计非零元素...")
    with torch.no_grad():
        def count_nz(state_dict):
            # 注意：这里需要处理 scalar 或者 0维tensor
            count = 0
            for t in state_dict.values():
                if isinstance(t, torch.Tensor):
                    count += t.count_nonzero().item()
            return count
        
        # 由于 checkpoint 已经被 pop 空了，我们需要重新计算原始非零值
        # 实际上 Wbase + Wresi 的非零值之和约等于原始值（除了重叠部分，但在BitNet里通常不重叠或者我们只关心总数）
        # 或者更准确地说，我们用 Base + Resi 来代表原始信息量
        total_nz_base = count_nz(Wbase_result)
        total_nz_resi = count_nz(Wresi_result)
        total_nz_orig = total_nz_base + total_nz_resi # 近似统计，因为原始数据已被释放
        
    print("-" * 60)
    print(f"处理完成! 耗时: {time.time() - start_time:.2f}s")
    print(f"统计信息:")
    print(f"  原始非零元素(估算): {total_nz_orig}")
    print(f"  Wbase非零元素: {total_nz_base} (占比 {total_nz_base/total_nz_orig*100:.1f}%)")
    print(f"  Wresi非零元素: {total_nz_resi} (占比 {total_nz_resi/total_nz_orig*100:.1f}%)")
    print(f"  Target层数: {total_params - skipped_count}, 跳过层数: {skipped_count}")
    
    return paths["Base"], paths["Resi"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BitNet权重转换: 结构化稀疏 + Expand矩阵生成')
    default_input = Path(__file__).resolve().parent / 'checkpoints' / 'model_state_int8.pt'
    parser.add_argument('--input', type=str, default=str(default_input), help='输入权重路径')
    parser.add_argument('--skip-validate', action='store_false', dest='validate', help='跳过验证')
    parser.add_argument('--use-cpu', action='store_false', dest='use_gpu', help='强制使用CPU')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 {args.input}")
        sys.exit(1)
        
    split_to_structured_sparse_and_expand(
        input_path=args.input,
        validate=args.validate,
        use_gpu=args.use_gpu
    )