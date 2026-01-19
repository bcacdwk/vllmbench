# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 核心逻辑模块

包含:
- config: SlideSparse 配置函数
- SlideSparseFp8LinearMethod: SlideSparse FP8 线性层方法
- SlideSparseFp8LinearOp: 根据环境变量选择 kernel

环境变量:
=========
1. DISABLE_SLIDESPARSE=1  → 完全禁用 SlideSparse，使用 vLLM 原生路径
2. USE_CUBLASLT=1         → 使用 cuBLASLt kernel
3. USE_CUSPARSELT=1       → 使用 cuSPARSELt kernel
4. INNER_DTYPE_FP32=1     → GEMM 输出用 FP32（仅 cuBLASLt/cuSPARSELt 时生效）
5. SPARSITY=2_8           → 稀疏格式（仅 cuSPARSELt 时生效，默认 2_8）

架构说明:
=========
- cuBLASLt_FP8_linear: 纯 cuBLASLt kernel 路径
- cuSPARSELt_FP8_linear: 纯 cuSPARSELt kernel 路径
- cutlass_FP8_linear: CUTLASS kernel fallback 路径
- SlideSparseFp8LinearOp: 根据环境变量选择上述三个 kernel 之一
"""

from slidesparse.core.config import (
    is_slidesparse_enabled,
    is_cublaslt_enabled,
    is_cusparselt_enabled,
    is_inner_dtype_fp32,
    get_slidesparse_status,
    get_sparsity_config,
    get_sparsity_str,
    clear_sparsity_cache,
)
from slidesparse.core.SlideSparseLinearMethod_FP8 import (
    SlideSparseFp8LinearMethod,
    SlideSparseFp8LinearOp,
    # 三个 kernel 函数
    cuBLASLt_FP8_linear,
    cuSPARSELt_FP8_linear,
    cutlass_FP8_linear,
    # 统一工厂函数
    wrap_scheme_fp8,
    # GEMM Extension
    _get_gemm_extension,
)

__all__ = [
    # 配置相关
    "is_slidesparse_enabled",
    "is_cublaslt_enabled",
    "is_cusparselt_enabled",
    "is_inner_dtype_fp32",
    "get_slidesparse_status",
    "get_sparsity_config",
    "get_sparsity_str",
    "clear_sparsity_cache",
    # 线性层方法
    "SlideSparseFp8LinearMethod",
    "SlideSparseFp8LinearOp",
    # 三个 kernel 函数
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    # 统一工厂函数
    "wrap_scheme_fp8",
    # GEMM Extension
    "_get_gemm_extension",
]