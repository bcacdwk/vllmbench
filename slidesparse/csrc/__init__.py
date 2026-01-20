# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse CSRC Package

提供 CUDA/Triton 扩展的编译工具和内核模块。

子模块
======
- utils: 编译工具（NVCC flags, build_cuda_extension 等）
- cublaslt_gemm: cuBLASLt FP8 GEMM 扩展
- fused_dequant_bias_triton: Triton 融合反量化偏置内核
- fused_quant_slide_triton: Triton 融合量化滑动内核
- quant_only_triton: Triton 纯量化内核（无 slide）
"""

from .utils import (
    get_nvcc_arch_flags,
    build_cuda_extension,
    clean_build_artifacts,
    get_gemm_ldflags,
    get_dequant_autotune_configs,
    CUBLASLT_LDFLAGS,
    CUSPARSELT_LDFLAGS,
)

__all__ = [
    "get_nvcc_arch_flags",
    "build_cuda_extension",
    "clean_build_artifacts",
    "get_gemm_ldflags",
    "get_dequant_autotune_configs",
    "CUBLASLT_LDFLAGS",
    "CUSPARSELT_LDFLAGS",
]
