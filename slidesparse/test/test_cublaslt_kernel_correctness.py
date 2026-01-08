#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 4: cuBLASLt Kernel 正确性测试

验证 CuBLASLtFp8LinearOp 的输出与原生 Fp8LinearOp 一致。

当前阶段（Phase 3 初期）:
    由于 CuBLASLtFp8LinearOp 内部仍然使用 Fp8LinearOp，
    所以输出应该完全一致。这个测试主要验证包装逻辑的正确性。

后续阶段（Phase 3 完成后）:
    当替换为真正的 cuBLASLt kernel 后，
    输出应该在数值精度范围内接近。

运行方式:
    CUDA_VISIBLE_DEVICES=0 python3 slidesparse/test/test_cublaslt_kernel_correctness.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def test_kernel_correctness():
    """测试 kernel 正确性"""
    print("=" * 60)
    print("测试 4: cuBLASLt Kernel 正确性测试")
    print("=" * 60)
    
    import torch
    
    # 检查 CUDA 可用性
    if not torch.cuda.is_available():
        print("\n✗ CUDA 不可用，跳过测试")
        return False
    
    device = torch.device("cuda:0")
    print(f"\n使用设备: {torch.cuda.get_device_name(0)}")
    
    # 导入测试模块
    print("\n[4.1] 导入测试模块...")
    try:
        from vllm.model_executor.layers.quantization.utils.w8a8_utils import Fp8LinearOp
        from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
        from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
        print("    ✓ 模块导入成功")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        return False
    
    # 创建测试数据
    print("\n[4.2] 创建测试数据...")
    
    # 测试多种尺寸
    test_cases = [
        {"M": 1, "K": 256, "N": 512, "name": "单 token"},
        {"M": 16, "K": 256, "N": 512, "name": "小 batch"},
        {"M": 64, "K": 512, "N": 1024, "name": "中 batch"},
        {"M": 128, "K": 1024, "N": 2048, "name": "大 batch"},
    ]
    
    all_passed = True
    
    for case in test_cases:
        M, K, N = case["M"], case["K"], case["N"]
        name = case["name"]
        
        print(f"\n    测试用例: {name} (M={M}, K={K}, N={N})")
        
        # 创建输入
        input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        weight = torch.randn(K, N, dtype=torch.float8_e4m3fn, device=device)
        weight_scale = torch.ones(1, dtype=torch.float32, device=device)
        
        # 创建两个 Op
        fp8_op = Fp8LinearOp(act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
        cublaslt_op = CuBLASLtFp8LinearOp(act_quant_static=False, act_quant_group_shape=GroupShape.PER_TOKEN)
        
        try:
            # 执行原生 Op
            output_fp8 = fp8_op.apply(
                input=input_tensor,
                weight=weight,
                weight_scale=weight_scale,
                out_dtype=torch.bfloat16,
            )
            
            # 执行 cuBLASLt Op
            output_cublaslt = cublaslt_op.apply(
                input=input_tensor,
                weight=weight,
                weight_scale=weight_scale,
                out_dtype=torch.bfloat16,
            )
            
            # 比较结果
            max_diff = (output_fp8 - output_cublaslt).abs().max().item()
            mean_diff = (output_fp8 - output_cublaslt).abs().mean().item()
            
            # 由于当前 cuBLASLt Op 内部使用 Fp8LinearOp，输出应该完全一致
            if max_diff == 0:
                print(f"        ✓ 输出完全一致 (max_diff=0)")
            elif max_diff < 1e-3:
                print(f"        ✓ 输出接近 (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
            else:
                print(f"        ⚠ 输出差异较大 (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
                all_passed = False
                
        except Exception as e:
            print(f"        ✗ 执行失败: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # 总结
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有 kernel 正确性测试通过!")
    else:
        print("✗ 部分 kernel 正确性测试失败!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_kernel_correctness()
    sys.exit(0 if success else 1)
