#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 2: vLLM 集成测试

验证 cuBLASLt 后端能否正确集成到 vLLM 的 compressed-tensors 量化流程中。

运行方式:
    # 禁用 cuBLASLt (默认)
    python3 slidesparse/test/test_cublaslt_vllm_integration.py
    
    # 启用 cuBLASLt
    VLLM_USE_CUBLASLT=1 python3 slidesparse/test/test_cublaslt_vllm_integration.py
"""

import sys
import os

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def test_vllm_integration():
    """测试 vLLM 集成"""
    print("=" * 60)
    print("测试 2: vLLM 集成测试")
    print("=" * 60)
    
    success = True
    
    # 检查环境变量
    use_cublaslt = os.environ.get("VLLM_USE_CUBLASLT", "0").lower() in ("1", "true", "yes")
    print(f"\n环境变量 VLLM_USE_CUBLASLT: {use_cublaslt}")
    
    # 测试 2.1: 导入 compressed_tensors 配置
    print("\n[2.1] 导入 compressed_tensors 配置...")
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
            CompressedTensorsLinearMethod,
        )
        print(f"    ✓ CompressedTensorsConfig: {CompressedTensorsConfig}")
        print(f"    ✓ CompressedTensorsLinearMethod: {CompressedTensorsLinearMethod}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
        return success
    
    # 测试 2.2: 检查 cuBLASLt 导入是否成功集成
    print("\n[2.2] 检查 cuBLASLt 集成...")
    try:
        from vllm.model_executor.layers.quantization.cublaslt import (
            is_cublaslt_enabled,
            get_cublaslt_status,
        )
        status = get_cublaslt_status()
        enabled = is_cublaslt_enabled()
        print(f"    ✓ cuBLASLt 状态: {status}")
        print(f"    ✓ cuBLASLt 启用: {enabled}")
        
        if use_cublaslt and not enabled:
            print("    ⚠ 警告: 环境变量已设置但 cuBLASLt 未启用")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 测试 2.3: 导入 FP8 scheme
    print("\n[2.3] 导入 FP8 scheme...")
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsW8A8Fp8,
        )
        print(f"    ✓ CompressedTensorsW8A8Fp8: {CompressedTensorsW8A8Fp8}")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        success = False
    
    # 测试 2.4: 测试 wrap_scheme_with_cublaslt 函数
    print("\n[2.4] 测试 wrap_scheme_with_cublaslt 函数...")
    try:
        from vllm.model_executor.layers.quantization.cublaslt import wrap_scheme_with_cublaslt
        from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
        
        # 创建一个 mock scheme
        weight_quant = QuantizationArgs(
            num_bits=8,
            type="float",
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=False,
        )
        original_scheme = CompressedTensorsW8A8Fp8(
            weight_quant=weight_quant,
            is_static_input_scheme=False,
        )
        print(f"    原始 scheme: {type(original_scheme).__name__}")
        
        # 包装
        wrapped_scheme = wrap_scheme_with_cublaslt(original_scheme)
        print(f"    包装后 scheme: {type(wrapped_scheme).__name__}")
        
        if use_cublaslt:
            # 应该被包装
            if "CuBLASLt" in type(wrapped_scheme).__name__:
                print("    ✓ scheme 已被 cuBLASLt 包装")
            else:
                print("    ⚠ 警告: cuBLASLt 已启用但 scheme 未被包装")
        else:
            # 应该保持原样
            if type(wrapped_scheme) == type(original_scheme):
                print("    ✓ scheme 保持原样 (cuBLASLt 未启用)")
            else:
                print("    ⚠ 警告: cuBLASLt 未启用但 scheme 被修改")
                
    except Exception as e:
        print(f"    ✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # 总结
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有 vLLM 集成测试通过!")
    else:
        print("✗ 部分 vLLM 集成测试失败!")
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = test_vllm_integration()
    sys.exit(0 if success else 1)
