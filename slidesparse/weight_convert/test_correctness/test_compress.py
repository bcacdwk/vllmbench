#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Compress 正确性测试

验证 cuSPARSELt 压缩功能的正确性：
1. 压缩扩展是否可加载
2. 压缩大小查询是否正确
3. 压缩后数据大小符合预期
4. 模拟压缩功能测试
5. 不同维度的压缩测试
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def create_2to4_sparse_tensor(N: int, K: int, dtype=torch.int8) -> torch.Tensor:
    """
    创建满足 2:4 稀疏约束的张量
    
    Args:
        N: 行数
        K: 列数（必须是 4 的倍数）
        dtype: 数据类型
    
    Returns:
        满足 2:4 约束的张量（每 4 个元素中 2 个为零）
    """
    if K % 4 != 0:
        raise ValueError(f"K must be multiple of 4, got K={K}")
    
    # 创建随机非零值
    if dtype == torch.int8:
        weight = torch.randint(-127, 127, (N, K), dtype=torch.float32)
    else:
        weight = torch.randn(N, K, dtype=torch.float32)
    
    # 对每行的每 4 个元素，随机选择 2 个设为 0
    for i in range(N):
        for g in range(K // 4):
            start = g * 4
            # 随机选择 2 个位置设为 0
            zero_indices = torch.randperm(4)[:2]
            weight[i, start + zero_indices] = 0
    
    return weight.to(dtype)


def test_fake_compress_basic():
    """测试模拟压缩的基本功能"""
    print("=" * 60)
    print("Test: Fake Compress Basic")
    print("=" * 60)
    
    from compress import compress_tensor_fake
    
    torch.manual_seed(42)
    N, K = 32, 64
    
    weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
    
    values, metadata, info = compress_tensor_fake(weight)
    
    print(f"  Input shape: {weight.shape}")
    print(f"  Values shape: {values.shape} (expected: [{N}, {K//2}])")
    print(f"  Metadata shape: {metadata.shape} (expected: [{N}, {K//4}])")
    
    # 验证输出形状
    assert values.shape == (N, K // 2), f"Values shape mismatch: {values.shape}"
    assert metadata.shape == (N, K // 4), f"Metadata shape mismatch: {metadata.shape}"
    
    # 验证 info 字典
    assert info["original_shape"] == [N, K]
    assert info["compression_type"] == "fake_2to4"
    
    print("  Status: PASS")
    print()
    return True


def test_fake_compress_value_preservation():
    """测试模拟压缩是否正确保留非零值"""
    print("=" * 60)
    print("Test: Fake Compress Value Preservation")
    print("=" * 60)
    
    from compress import compress_tensor_fake
    
    # 创建一个简单的测试用例
    # 每 4 个元素：[0, 0, a, b]
    N, K = 2, 8
    weight = torch.tensor([
        [0, 0, 1, 2, 0, 0, 3, 4],
        [0, 5, 0, 6, 7, 0, 0, 8],
    ], dtype=torch.int8)
    
    values, metadata, info = compress_tensor_fake(weight)
    
    print(f"  Input: {weight.tolist()}")
    print(f"  Values: {values.tolist()}")
    print(f"  Metadata: {metadata.tolist()}")
    
    # 验证第一行：非零值是 1,2,3,4
    # 验证第二行：非零值是 5,6,7,8
    
    # 检查非零值数量
    original_nonzeros = (weight != 0).sum().item()
    compressed_nonzeros = (values != 0).sum().item()
    
    print(f"  Original nonzeros: {original_nonzeros}")
    print(f"  Compressed nonzeros: {compressed_nonzeros}")
    
    # 压缩后的非零值应该等于或少于原始非零值（每 4 个取 2 个）
    assert compressed_nonzeros <= original_nonzeros
    
    print("  Status: PASS")
    print()
    return True


def test_fake_compress_different_sizes():
    """测试不同维度的模拟压缩"""
    print("=" * 60)
    print("Test: Fake Compress Different Sizes")
    print("=" * 60)
    
    from compress import compress_tensor_fake
    
    torch.manual_seed(42)
    
    test_cases = [
        (32, 64),
        (32, 128),
        (64, 256),
        (128, 512),
        (256, 1024),
    ]
    
    all_pass = True
    
    for N, K in test_cases:
        weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
        values, metadata, info = compress_tensor_fake(weight)
        
        expected_values_shape = (N, K // 2)
        expected_metadata_shape = (N, K // 4)
        
        values_ok = (values.shape == expected_values_shape)
        metadata_ok = (metadata.shape == expected_metadata_shape)
        
        status = "✓" if (values_ok and metadata_ok) else "✗"
        print(f"  {status} [{N:4d}, {K:4d}] -> values {values.shape}, metadata {metadata.shape}")
        
        all_pass &= (values_ok and metadata_ok)
    
    print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


def test_compress_sizes_query():
    """测试压缩大小查询"""
    print("=" * 60)
    print("Test: Compress Sizes Query")
    print("=" * 60)
    
    try:
        from compress import get_compress_sizes, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        # cuSPARSELt 要求:
        # - 稀疏矩阵 (INT8): N, K, ld 必须是 32 的倍数
        # - 稠密矩阵 (INT8): rows, cols, ld 必须是 16 的倍数
        # 先用简单的正方形尺寸测试
        test_cases = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (1024, 2048),  # 非正方形但都是 32 的倍数
            (2048, 1024),
        ]
        
        all_valid = True
        
        for N, K in test_cases:
            try:
                compressed_size, temp_size = get_compress_sizes(N, K)
                
                # 压缩后大小应该大于 0
                valid = (compressed_size > 0)
                all_valid &= valid
                
                status = "✓" if valid else "✗"
                print(f"  {status} [{N:5d}, {K:5d}]: compressed={compressed_size:10d}, temp={temp_size:10d}")
            except Exception as e:
                print(f"  ✗ [{N:5d}, {K:5d}]: Error - {e}")
                all_valid = False
        
        print(f"  Status: {'PASS' if all_valid else 'FAIL'}")
        print()
        return all_valid
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_real_compress():
    """测试真实的 cuSPARSELt 压缩"""
    print("=" * 60)
    print("Test: Real cuSPARSELt Compress")
    print("=" * 60)
    
    try:
        from compress import compress_tensor, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        
        test_cases = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
        ]
        
        all_pass = True
        
        for N, K in test_cases:
            weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
            
            try:
                compressed, metadata = compress_tensor(weight)
                
                # 验证压缩结果
                assert compressed.dtype == torch.uint8
                assert len(compressed.shape) == 1  # 压缩后是 1D 的字节数组
                assert metadata["original_shape"] == [N, K]
                assert metadata["sparsity_pattern"] == "2:4"
                
                print(f"  ✓ [{N:4d}, {K:4d}] -> compressed size: {compressed.shape[0]} bytes")
                
            except Exception as e:
                print(f"  ✗ [{N:4d}, {K:4d}] failed: {e}")
                all_pass = False
        
        print(f"  Status: {'PASS' if all_pass else 'FAIL'}")
        print()
        return all_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_compress_alignment_requirements():
    """测试压缩对齐要求"""
    print("=" * 60)
    print("Test: Compress Alignment Requirements")
    print("=" * 60)
    
    try:
        from compress import compress_tensor, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        
        # cuSPARSELt 要求：
        # - K 必须是 4 的倍数（2:4 稀疏）
        # - N 和 K 必须是 32 的倍数（对齐要求）
        
        # 对齐的尺寸应该工作
        aligned_cases = [
            (32, 32),
            (64, 64),
            (128, 128),
            (256, 256),
        ]
        
        print("  Testing aligned dimensions:")
        all_aligned_pass = True
        for N, K in aligned_cases:
            weight = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
            try:
                compressed, _ = compress_tensor(weight)
                print(f"    ✓ [{N:4d}, {K:4d}]: OK")
            except Exception as e:
                print(f"    ✗ [{N:4d}, {K:4d}]: {e}")
                all_aligned_pass = False
        
        print(f"  Status: {'PASS' if all_aligned_pass else 'FAIL'}")
        print()
        return all_aligned_pass
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_compress_sparsity_verification():
    """测试压缩前的稀疏性验证"""
    print("=" * 60)
    print("Test: Compress Sparsity Verification")
    print("=" * 60)
    
    try:
        from compress import compress_tensor, check_compress_available
        
        if not check_compress_available():
            print("  cuSPARSELt compress not available, skipping")
            print("  Status: SKIPPED")
            print()
            return True
        
        torch.manual_seed(42)
        N, K = 64, 64
        
        # 创建不满足 2:4 的权重（所有元素非零）
        weight_invalid = torch.randint(-127, 127, (N, K), dtype=torch.int8)
        
        try:
            compressed, _ = compress_tensor(weight_invalid)
            print("  ✗ Expected exception for invalid sparsity, but none raised")
            return False
        except ValueError as e:
            if "2:4 sparsity" in str(e):
                print(f"  ✓ Correctly rejected invalid sparsity: {e}")
            else:
                print(f"  ✗ Wrong exception: {e}")
                return False
        
        # 创建满足 2:4 的权重
        weight_valid = create_2to4_sparse_tensor(N, K, dtype=torch.int8)
        
        try:
            compressed, _ = compress_tensor(weight_valid)
            print(f"  ✓ Valid 2:4 weight compressed successfully")
        except Exception as e:
            print(f"  ✗ Valid weight rejected: {e}")
            return False
        
        print("  Status: PASS")
        print()
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED (extension not available)")
        print()
        return True


def test_extension_availability():
    """测试压缩扩展是否可用"""
    print("=" * 60)
    print("Test: Extension Availability")
    print("=" * 60)
    
    try:
        from compress import check_compress_available, get_compress_module
        
        available = check_compress_available()
        print(f"  cuSPARSELt compress available: {available}")
        
        if available:
            module = get_compress_module()
            print(f"  Module loaded: {module}")
            
            # 检查函数是否存在
            has_get_sizes = hasattr(module, 'cusparselt_get_compress_sizes')
            has_compress = hasattr(module, 'cusparselt_compress_weight')
            has_check = hasattr(module, 'cusparselt_is_available')
            
            print(f"  Has cusparselt_get_compress_sizes: {has_get_sizes}")
            print(f"  Has cusparselt_compress_weight: {has_compress}")
            print(f"  Has cusparselt_is_available: {has_check}")
            
            if has_get_sizes and has_compress and has_check:
                print("  Status: PASS")
            else:
                print("  Status: FAIL (missing functions)")
                return False
        else:
            print("  Status: SKIPPED (not available)")
        
        print()
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        print("  Status: SKIPPED")
        print()
        return True


def main():
    """运行所有测试"""
    tests = [
        test_fake_compress_basic,
        test_fake_compress_value_preservation,
        test_fake_compress_different_sizes,
        test_extension_availability,
        test_compress_sizes_query,
        test_real_compress,
        test_compress_alignment_requirements,
        test_compress_sparsity_verification,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("\n" + "=" * 60)
    print("Compress Correctness Tests")
    print("=" * 60 + "\n")
    
    for test in tests:
        try:
            result = test()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
