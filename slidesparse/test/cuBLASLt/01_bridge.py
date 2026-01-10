#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
01_bridge.py - cuBLASLt 桥接与集成测试

验证 slidesparse cuBLASLt 外挂模块的：
1. 模块导入和包结构
2. 配置系统（环境变量）
3. vLLM 集成点
4. 关键类和函数的存在性

使用方法:
    python3 01_bridge.py               # 测试 cuBLASLt 路径（默认）
    python3 01_bridge.py --ext-cutlass # 测试外挂 CUTLASS 路径
    python3 01_bridge.py -q            # 静默模式

路径说明:
    默认: USE_CUBLASLT=1 → cuBLASLt kernel
    --ext-cutlass: USE_CUBLASLT=0 → 外挂 CUTLASS
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    TestRunner,
    TestResult,
    TestStatus,
    test_case,
    EnvironmentChecker,
    Colors,
    parse_common_args,
    apply_env_args,
)


# ============================================================================
# 1. 模块导入测试
# ============================================================================

@test_case("导入 slidesparse 主模块")
def test_import_main():
    """测试 slidesparse 主模块导入"""
    import slidesparse
    
    assert hasattr(slidesparse, "__version__"), "缺少 __version__"
    return True, f"v{slidesparse.__version__}"


@test_case("导入 slidesparse.core")
def test_import_core():
    """测试 core 子模块导入"""
    from slidesparse.core import (
        is_cublaslt_enabled,
        is_inner_dtype_fp32,
        get_cublaslt_status,
        CuBLASLtFp8LinearMethod,
        CuBLASLtFp8LinearOp,
        wrap_scheme_with_cublaslt,
    )
    
    # 验证导出的符号类型
    assert callable(is_cublaslt_enabled)
    assert callable(is_inner_dtype_fp32)
    assert callable(get_cublaslt_status)
    assert callable(wrap_scheme_with_cublaslt)
    
    return True, "核心接口导入成功"


@test_case("导入配置模块")
def test_import_config():
    """测试配置模块单独导入"""
    from slidesparse.core.cublaslt_config import (
        is_cublaslt_enabled,
        is_inner_dtype_fp32,
        get_cublaslt_status,
    )
    
    # 验证返回类型
    assert isinstance(is_cublaslt_enabled(), bool)
    assert isinstance(is_inner_dtype_fp32(), bool)
    assert isinstance(get_cublaslt_status(), str)
    
    return True, f"配置解析正常"


@test_case("导入 LinearMethod 模块")
def test_import_linear_method():
    """测试 LinearMethod 模块"""
    from slidesparse.core.cublaslt_linear_method import (
        CuBLASLtFp8LinearOp,
        CuBLASLtFp8LinearMethod,
        wrap_scheme_with_cublaslt,
        cublaslt_fp8_linear,
    )
    
    # 验证类结构
    assert hasattr(CuBLASLtFp8LinearOp, "apply")
    assert hasattr(CuBLASLtFp8LinearMethod, "apply_weights")
    assert hasattr(CuBLASLtFp8LinearMethod, "create_weights")
    
    return True, "类结构正确"


# ============================================================================
# 2. 环境变量配置测试
# ============================================================================

@test_case("环境变量解析")
def test_env_vars():
    """测试环境变量读取"""
    from slidesparse.core.cublaslt_config import (
        is_cublaslt_enabled,
        is_inner_dtype_fp32,
    )
    
    enabled = is_cublaslt_enabled()
    fp32 = is_inner_dtype_fp32()
    
    return True, f"USE_CUBLASLT={enabled}, INNER_DTYPE_FP32={fp32}"


@test_case("get_cublaslt_status 格式")
def test_status_format():
    """测试状态字符串格式"""
    from slidesparse.core.cublaslt_config import get_cublaslt_status
    
    status = get_cublaslt_status()
    
    # 状态字符串应该包含关键信息
    assert "cublaslt" in status.lower() or "cutlass" in status.lower()
    
    return True, status


# ============================================================================
# 3. vLLM 桥接测试
# ============================================================================

@test_case("vLLM 桥接文件导入")
def test_vllm_bridge_import():
    """测试 vLLM 内的桥接文件"""
    from vllm.model_executor.layers.quantization.cublaslt import (
        is_cublaslt_enabled as vllm_is_enabled,
        is_inner_dtype_fp32 as vllm_is_fp32,
        wrap_scheme_with_cublaslt as vllm_wrap,
    )
    
    # 验证导入成功
    assert callable(vllm_is_enabled)
    assert callable(vllm_is_fp32)
    assert callable(vllm_wrap)
    
    return True, "vLLM 桥接模块正常"


@test_case("桥接函数引用一致性")
def test_bridge_reference_consistency():
    """验证 vLLM 桥接文件的函数是原始函数的引用"""
    from vllm.model_executor.layers.quantization.cublaslt import (
        is_cublaslt_enabled as vllm_is_enabled,
        is_inner_dtype_fp32 as vllm_is_fp32,
        wrap_scheme_with_cublaslt as vllm_wrap,
    )
    from slidesparse.core.cublaslt_config import is_cublaslt_enabled, is_inner_dtype_fp32
    from slidesparse.core.cublaslt_linear_method import wrap_scheme_with_cublaslt
    
    # 应该是同一个函数对象
    assert vllm_is_enabled is is_cublaslt_enabled, "is_cublaslt_enabled 引用不一致"
    assert vllm_is_fp32 is is_inner_dtype_fp32, "is_inner_dtype_fp32 引用不一致"
    assert vllm_wrap is wrap_scheme_with_cublaslt, "wrap_scheme_with_cublaslt 引用不一致"
    
    return True, "函数引用一致"


@test_case("CompressedTensors 模块导入")
def test_compressed_tensors_import():
    """测试 CompressedTensors 量化模块"""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
        CompressedTensorsLinearMethod,
    )
    
    assert CompressedTensorsConfig is not None
    assert hasattr(CompressedTensorsConfig, "get_scheme")
    
    return True, "CompressedTensors 可用"


@test_case("FP8 Scheme 类可用")
def test_fp8_scheme():
    """测试 FP8 scheme 类"""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Fp8,
    )
    
    # 验证必要方法
    assert hasattr(CompressedTensorsW8A8Fp8, "create_weights")
    assert hasattr(CompressedTensorsW8A8Fp8, "process_weights_after_loading")
    assert hasattr(CompressedTensorsW8A8Fp8, "apply_weights")
    
    return True, "FP8 scheme 可用"


# ============================================================================
# 4. Op 实例化测试
# ============================================================================

@test_case("CuBLASLtFp8LinearOp 创建")
def test_create_op():
    """测试 Op 实例创建"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    
    # 默认配置
    op1 = CuBLASLtFp8LinearOp()
    assert op1 is not None
    assert hasattr(op1, "apply")
    assert hasattr(op1, "quant_fp8")
    
    # 自定义配置
    op2 = CuBLASLtFp8LinearOp(
        act_quant_static=True,
        act_quant_group_shape=GroupShape.PER_TENSOR
    )
    assert op2 is not None
    
    return True, "Op 创建成功"


@test_case("QuantFP8 配置一致性")
def test_quant_fp8_config():
    """测试内部 QuantFP8 配置"""
    from slidesparse.core.cublaslt_linear_method import CuBLASLtFp8LinearOp
    
    # 使用默认配置 (动态量化)
    op = CuBLASLtFp8LinearOp(act_quant_static=False)
    
    assert op.quant_fp8 is not None
    assert op.quant_fp8.static == op.act_quant_static
    
    return True, "配置一致"


@test_case("wrap_scheme_with_cublaslt 函数")
def test_wrap_scheme_function():
    """测试 scheme 包装函数"""
    from slidesparse.core.cublaslt_linear_method import wrap_scheme_with_cublaslt
    
    # 不支持的类型应返回原对象
    class MockScheme:
        pass
    
    mock = MockScheme()
    wrapped = wrap_scheme_with_cublaslt(mock)
    
    assert wrapped is mock, "不支持的类型应返回原对象"
    
    return True, "包装函数正常"


# ============================================================================
# 5. cuBLASLt Extension 测试
# ============================================================================

@test_case("cuBLASLt Extension 加载状态")
def test_extension_load_status():
    """测试 cuBLASLt extension 加载状态"""
    from slidesparse.core.cublaslt_linear_method import _load_cublaslt_extension
    
    ext = _load_cublaslt_extension()
    
    if ext is not None:
        # 验证导出的函数
        has_fp8_mm = hasattr(ext, "cublaslt_fp8_mm")
        has_int8_mm = hasattr(ext, "cublaslt_int8_mm")
        return True, f"已加载 (fp8_mm={has_fp8_mm}, int8_mm={has_int8_mm})"
    else:
        return TestResult(
            name="cuBLASLt Extension 加载状态",
            status=TestStatus.WARNING,
            message="Extension 未加载，将使用 CUTLASS fallback"
        )


@test_case("cuBLASLt Extension 函数签名")
def test_extension_signatures():
    """测试 extension 导出函数的签名"""
    from slidesparse.core.cublaslt_linear_method import _load_cublaslt_extension
    
    ext = _load_cublaslt_extension()
    
    if ext is None:
        return TestResult(
            name="cuBLASLt Extension 函数签名",
            status=TestStatus.SKIPPED,
            message="Extension 未加载"
        )
    
    # 检查 cublaslt_fp8_mm
    assert hasattr(ext, "cublaslt_fp8_mm"), "缺少 cublaslt_fp8_mm"
    
    return True, "函数签名正确"


# ============================================================================
# 主函数
# ============================================================================

def get_all_tests():
    """获取所有测试"""
    return [
        # 1. 模块导入
        test_import_main,
        test_import_core,
        test_import_config,
        test_import_linear_method,
        # 2. 环境变量
        test_env_vars,
        test_status_format,
        # 3. vLLM 桥接
        test_vllm_bridge_import,
        test_bridge_reference_consistency,
        test_compressed_tensors_import,
        test_fp8_scheme,
        # 4. Op 实例化
        test_create_op,
        test_quant_fp8_config,
        test_wrap_scheme_function,
        # 5. Extension
        test_extension_load_status,
        test_extension_signatures,
    ]


def run_tests(verbose: bool = True) -> bool:
    """运行所有测试"""
    tests = get_all_tests()
    
    if verbose:
        EnvironmentChecker.print_env_info()
    
    runner = TestRunner("cuBLASLt 桥接与集成测试", verbose=verbose)
    result = runner.run_all(tests)
    
    return result.success


if __name__ == "__main__":
    parser = parse_common_args("cuBLASLt 桥接与集成测试")
    args = parser.parse_args()
    
    apply_env_args(args)
    
    success = run_tests(verbose=True)
    
    sys.exit(0 if success else 1)
