# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SlideSparse 集成入口 - vLLM 空壳转发文件

这个文件是 SlideSparse 与 vLLM 集成的桥梁。
它负责将外挂模块 (slidesparse/) 的功能导入到 vLLM 的量化框架中。

设计原则:
=========
1. 最小侵入式修改：这个文件是 vLLM 源码中唯一需要新增的文件
2. 所有逻辑都在外挂模块 (slidesparse/) 中实现
3. 这里只做导入和转发

使用方式:
=========
SlideSparse 默认启用，通过环境变量控制具体行为:
    vllm serve model_path --quantization compressed-tensors

不需要修改 --quantization 参数，保持使用 compressed-tensors。
SlideSparse 会在 CompressedTensorsW8A8Fp8/Int8 的基础上进行透明 hook。

环境变量:
=========
- DISABLE_SLIDESPARSE=1: 完全禁用 SlideSparse，使用 vLLM 原生路径
- USE_CUBLASLT=1: 启用 cuBLASLt kernel
- USE_CUSPARSELT=1: 启用 cuSPARSELt kernel
- INNER_DTYPE_32=1: GEMM 使用高精度累加（FP8→FP32, INT8→INT32）

架构说明:
=========
FP8 路径:
- cuBLASLt_FP8_linear: cuBLASLt FP8 GEMM + Triton dequant
- cuSPARSELt_FP8_linear: cuSPARSELt 2:4 稀疏 FP8 GEMM + Triton dequant
- cutlass_FP8_linear: vLLM CUTLASS kernel fallback

INT8 路径:
- cuBLASLt_INT8_linear: cuBLASLt INT8 GEMM（输出 INT32）+ Triton dequant
- cuSPARSELt_INT8_linear: cuSPARSELt 2:4 稀疏 INT8 GEMM + Triton dequant
- cutlass_INT8_linear: vLLM CUTLASS kernel fallback（支持非对称量化）
"""

import os
import sys

# 将 slidesparse 模块添加到 Python 路径
# 仓库根目录计算: vllm/model_executor/layers/quantization/slidesparse.py
#                -> 回退 4 层到仓库根目录
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_CURRENT_DIR))))
)
_SLIDESPARSE_PATH = os.path.join(_REPO_ROOT, "slidesparse")

if _SLIDESPARSE_PATH not in sys.path:
    sys.path.insert(0, _SLIDESPARSE_PATH)

# 从外挂模块导入
try:
    # 配置
    from slidesparse.core.config import (
        is_slidesparse_enabled,
        is_cublaslt_enabled,
        is_cusparselt_enabled,
        is_inner_dtype_32,
        get_slidesparse_status,
        get_sparsity_config,
        get_sparsity_str,
        clear_sparsity_cache,
    )
    
    # FP8
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        SlideSparseFp8LinearMethod,
        SlideSparseFp8LinearOp,
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        wrap_scheme_fp8,
    )
    
    # INT8
    from slidesparse.core.SlideSparseLinearMethod_INT8 import (
        SlideSparseInt8LinearMethod,
        SlideSparseInt8LinearOp,
        cuBLASLt_INT8_linear,
        cuSPARSELt_INT8_linear,
        cutlass_INT8_linear,
        wrap_scheme_int8,
    )
    
    # 共享组件
    from slidesparse.core.gemm_wrapper import _get_gemm_extension
    from slidesparse.core.profiler import print_profile_stats, reset_profile_stats
    
    _IMPORT_SUCCESS = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import slidesparse modules: {e}. "
        "SlideSparse features will be disabled."
    )
    _IMPORT_SUCCESS = False
    
    # Fallback stub functions - 配置
    def is_slidesparse_enabled():
        return False
    
    def is_cublaslt_enabled():
        return False
    
    def is_cusparselt_enabled():
        return False
    
    def is_inner_dtype_32():
        return False
    
    def get_slidesparse_status():
        return "SlideSparse backend UNAVAILABLE (import failed)"
    
    def get_sparsity_config():
        return (2, 8, 8/6)
    
    def get_sparsity_str():
        return "2_8"
    
    def clear_sparsity_cache():
        pass
    
    # Fallback stub functions - FP8
    def wrap_scheme_fp8(scheme):
        return scheme
    
    class SlideSparseFp8LinearMethod:
        pass
    
    class SlideSparseFp8LinearOp:
        pass
    
    def cuBLASLt_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cuSPARSELt_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cutlass_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    # Fallback stub functions - INT8
    def wrap_scheme_int8(scheme):
        return scheme
    
    class SlideSparseInt8LinearMethod:
        pass
    
    class SlideSparseInt8LinearOp:
        pass
    
    def cuBLASLt_INT8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cuSPARSELt_INT8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cutlass_INT8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    # Fallback stub functions - 共享
    def _get_gemm_extension(backend):
        raise NotImplementedError("SlideSparse import failed")
    
    def print_profile_stats():
        pass
    
    def reset_profile_stats():
        pass

__all__ = [
    # 配置相关
    "is_slidesparse_enabled",
    "is_cublaslt_enabled",
    "is_cusparselt_enabled",
    "is_inner_dtype_32",
    "get_slidesparse_status",
    "get_sparsity_config",
    "get_sparsity_str",
    "clear_sparsity_cache",
    
    # FP8 线性层
    "SlideSparseFp8LinearMethod",
    "SlideSparseFp8LinearOp",
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    "wrap_scheme_fp8",
    
    # INT8 线性层
    "SlideSparseInt8LinearMethod",
    "SlideSparseInt8LinearOp",
    "cuBLASLt_INT8_linear",
    "cuSPARSELt_INT8_linear",
    "cutlass_INT8_linear",
    "wrap_scheme_int8",
    
    # 共享组件
    "_get_gemm_extension",
    "print_profile_stats",
    "reset_profile_stats",
]
