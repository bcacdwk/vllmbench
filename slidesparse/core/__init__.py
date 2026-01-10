# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 核心逻辑模块

包含:
- cublaslt_config: cuBLASLt 配置函数
- CuBLASLtSchemeWrapper: cuBLASLt Scheme 包装器
- CuBLASLtFp8LinearMethod: cuBLASLt FP8 线性层方法

环境变量（只有两个）:
==================
1. USE_CUBLASLT=1        → 从 CUTLASS 切换到 cuBLASLt
2. INNER_DTYPE_FP32=1    → GEMM 输出用 FP32（仅 USE_CUBLASLT=1 时生效）
"""

from slidesparse.core.cublaslt_config import (
    is_cublaslt_enabled,
    is_inner_dtype_fp32,
    get_cublaslt_status,
)
from slidesparse.core.cublaslt_scheme_wrapper import (
    CuBLASLtSchemeWrapper,
    wrap_scheme_if_enabled,
    is_cublaslt_scheme,
)
from slidesparse.core.cublaslt_linear_method import (
    CuBLASLtFp8LinearMethod,
    CuBLASLtFp8LinearOp,
    wrap_scheme_with_cublaslt,
)

__all__ = [
    # 配置相关
    "is_cublaslt_enabled",
    "is_inner_dtype_fp32",
    "get_cublaslt_status",
    # Scheme 包装器
    "CuBLASLtSchemeWrapper",
    "wrap_scheme_if_enabled",
    "is_cublaslt_scheme",
    # 线性层方法
    "CuBLASLtFp8LinearMethod",
    "CuBLASLtFp8LinearOp",
    "wrap_scheme_with_cublaslt",
]
