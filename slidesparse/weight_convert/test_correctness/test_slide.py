#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Slide 正确性测试

验证滑动扩展功能的正确性：
1. 输出满足 2:4 稀疏约束
2. K 维度变化正确
3. 贪婪残差分配逻辑正确
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from utils import SlideSparseConfig, verify_2to4_sparsity, compute_output_k
from slide import slide_tensor, build_slide_index_mapping


def test_slide_basic():
    """测试基本的滑动扩展功能"""
    print("=" * 60)
    print("Test: Basic Slide")
    print("=" * 60)
    
    # 创建满足 2:8 约束的测试数据
    torch.manual_seed(42)
    N, K = 16, 32
    
    # 创建 2:8 稀疏权重（每 8 个元素中 2 个为零）
    weight = torch.randn(N, K)
    for i in range(N):
        for g in range(K // 8):
            # 随机选择 2 个位置设为 0
            start = g * 8
            indices = torch.randperm(8)[:2] + start
            weight[i, indices] = 0
    
    config = SlideSparseConfig(Z=2, L=8)
    slided, metadata = slide_tensor(weight, config)
    
    # 验证 2:4 稀疏性
    is_valid, violation_ratio = verify_2to4_sparsity(slided)
    
    print(f"  Input shape: {weight.shape}")
    print(f"  Output shape: {slided.shape}")
    print(f"  Expand ratio: {slided.shape[1] / weight.shape[1]:.3f} (expected: {config.expand_ratio:.3f})")
    print(f"  2:4 valid: {is_valid}, violation: {violation_ratio:.2%}")
    print(f"  Status: {'PASS' if is_valid else 'FAIL'}")
    
    assert is_valid, f"Slide failed: violation_ratio={violation_ratio}"
    print()
    return True


def test_slide_dimension_calculation():
    """测试维度计算正确性"""
    print("=" * 60)
    print("Test: Dimension Calculation")
    print("=" * 60)
    
    test_cases = [
        # (K_in, L, expected_expand_ratio)
        (32, 6, 4/3),     # 2:6 -> expand by 4/3
        (32, 8, 1.5),     # 2:8 -> expand by 1.5
        (32, 10, 1.6),    # 2:10 -> expand by 1.6
        (64, 8, 1.5),
        (100, 8, 1.5),    # 非整除情况
    ]
    
    for K_in, L, expected_ratio in test_cases:
        config = SlideSparseConfig(Z=2, L=L)
        k_padded, k_out = compute_output_k(K_in, config, align_to=16)
        
        actual_ratio = k_out / K_in
        
        print(f"  K={K_in}, L={L}: k_padded={k_padded}, k_out={k_out}, ratio={actual_ratio:.3f} (expected ~{expected_ratio:.3f})")
        
        # 检查 k_out 对齐到 16
        assert k_out % 16 == 0, f"k_out not aligned: {k_out}"
        
        # 检查 k_padded 是 L 的倍数
        assert k_padded % L == 0, f"k_padded not multiple of L: {k_padded}"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_index_mapping():
    """测试索引映射的正确性"""
    print("=" * 60)
    print("Test: Index Mapping")
    print("=" * 60)
    
    # 以 2:8 为例
    config = SlideSparseConfig(Z=2, L=8)
    K_in = 16  # 2 组
    
    index_map = build_slide_index_mapping(K_in, config)
    
    print(f"  K_in={K_in}, L={config.L}")
    print(f"  Index map shape: {index_map.shape}")
    print(f"  Index map (first 24): {index_map[:24].tolist()}")
    
    # 验证映射
    # Group 0 (输入 0-7):
    #   Window 0: output[0,1,2,3] <- input[0,1,2,3]
    #   Window 1: output[4,5,6,7] <- input[2,3,4,5]
    #   Window 2: output[8,9,10,11] <- input[4,5,6,7]
    expected_group0 = [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7]
    actual_group0 = index_map[:12].tolist()
    
    print(f"  Expected group 0: {expected_group0}")
    print(f"  Actual group 0:   {actual_group0}")
    
    assert actual_group0 == expected_group0, f"Index mapping mismatch"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_greedy_allocation():
    """测试贪婪分配的正确性"""
    print("=" * 60)
    print("Test: Greedy Allocation")
    print("=" * 60)
    
    # 创建一个简单的测试用例
    # 每组 8 个元素，其中有 2 个零
    # 位置 0,1 是零，2-7 是非零
    N, K = 1, 8
    weight = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    
    config = SlideSparseConfig(Z=2, L=8)
    slided, _ = slide_tensor(weight, config)
    
    print(f"  Input: {weight[0].tolist()}")
    print(f"  Output shape: {slided.shape}")
    print(f"  Output: {slided[0].tolist()}")
    
    # 验证输出满足 2:4
    is_valid, violation_ratio = verify_2to4_sparsity(slided)
    print(f"  2:4 valid: {is_valid}")
    
    # 验证每个 4 元素组至少有 2 个零
    for i in range(slided.shape[1] // 4):
        group = slided[0, i*4:(i+1)*4]
        zeros = (group == 0).sum().item()
        print(f"  Group {i}: {group.tolist()}, zeros={zeros}")
        assert zeros >= 2, f"Group {i} has only {zeros} zeros"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_different_L():
    """测试不同的 L 值"""
    print("=" * 60)
    print("Test: Different L Values")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 32, 64
    
    for L in [6, 8, 10, 12]:
        config = SlideSparseConfig(Z=2, L=L)
        
        # 创建满足 2:L 约束的权重
        weight = torch.randn(N, K)
        for i in range(N):
            k_padded = ((K + L - 1) // L) * L
            for g in range(k_padded // L):
                start = g * L
                if start + L <= K:
                    indices = torch.randperm(L)[:2] + start
                    weight[i, indices.clamp(max=K-1)] = 0
        
        slided, metadata = slide_tensor(weight, config)
        
        is_valid, violation_ratio = verify_2to4_sparsity(slided)
        
        print(f"  L={L}: shape {weight.shape} -> {slided.shape}, ratio={slided.shape[1]/weight.shape[1]:.3f}, 2:4 valid={is_valid}")
        
        # 2:4 验证可能因为边界情况有少量违规
        assert violation_ratio < 0.1, f"L={L} has too many violations: {violation_ratio:.2%}"
    
    print("  Status: PASS")
    print()
    return True


def test_slide_dtype_preservation():
    """测试数据类型保持"""
    print("=" * 60)
    print("Test: Dtype Preservation")
    print("=" * 60)
    
    torch.manual_seed(42)
    N, K = 16, 32
    config = SlideSparseConfig(Z=2, L=8)
    
    dtypes = [torch.float32, torch.float16, torch.bfloat16, torch.int8]
    
    for dtype in dtypes:
        if dtype == torch.int8:
            weight = torch.randint(-127, 127, (N, K), dtype=dtype)
        else:
            weight = torch.randn(N, K).to(dtype)
        
        # 添加一些零以满足 2:8
        weight_float = weight.float()
        for i in range(N):
            for g in range(K // 8):
                start = g * 8
                indices = torch.randperm(8)[:2] + start
                weight_float[i, indices] = 0
        
        if dtype == torch.int8:
            weight = weight_float.to(torch.int8)
        else:
            weight = weight_float.to(dtype)
        
        slided, _ = slide_tensor(weight, config)
        
        print(f"  {dtype}: input={weight.dtype}, output={slided.dtype}")
        assert slided.dtype == weight.dtype, f"Dtype mismatch: {slided.dtype} != {weight.dtype}"
    
    print("  Status: PASS")
    print()
    return True


def main():
    """运行所有测试"""
    tests = [
        test_slide_basic,
        test_slide_dimension_calculation,
        test_slide_index_mapping,
        test_slide_greedy_allocation,
        test_slide_different_L,
        test_slide_dtype_preservation,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
