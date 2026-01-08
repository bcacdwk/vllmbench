# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 核心逻辑模块

包含:
- CuBLASLtFp8Config: cuBLASLt FP8 配置函数
- CuBLASLtSchemeWrapper: cuBLASLt Scheme 包装器
- CuBLASLtFp8LinearMethod: cuBLASLt FP8 线性层方法
"""

from slidesparse.core.cublaslt_config import (
    is_cublaslt_enabled,
    get_cublaslt_status,
    VLLM_USE_CUBLASLT,
    SLIDESPARSE_USE_CUBLASLT,
    SLIDESPARSE_CUBLASLT_DEBUG,
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
    "get_cublaslt_status",
    "VLLM_USE_CUBLASLT",
    "SLIDESPARSE_USE_CUBLASLT",
    "SLIDESPARSE_CUBLASLT_DEBUG",
    # Scheme 包装器
    "CuBLASLtSchemeWrapper",
    "wrap_scheme_if_enabled",
    "is_cublaslt_scheme",
    # 线性层方法
    "CuBLASLtFp8LinearMethod",
    "CuBLASLtFp8LinearOp",
    "wrap_scheme_with_cublaslt",
]
