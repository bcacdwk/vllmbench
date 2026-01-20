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
SlideSparse 会在 CompressedTensorsW8A8Fp8 的基础上进行透明 hook。

环境变量:
=========
- DISABLE_SLIDESPARSE=1: 完全禁用 SlideSparse，使用 vLLM 原生路径
- USE_CUBLASLT=1: 启用 cuBLASLt kernel
- USE_CUSPARSELT=1: 启用 cuSPARSELt kernel
- INNER_DTYPE_32=1: GEMM 使用高精度累加（FP8→FP32, INT8→INT32）

架构说明:
=========
- cuBLASLt_FP8_linear: 纯 cuBLASLt kernel 路径
- cuSPARSELt_FP8_linear: 纯 cuSPARSELt kernel 路径
- cutlass_FP8_linear: CUTLASS kernel fallback 路径
- SlideSparseFp8LinearOp: 根据环境变量选择上述三个 kernel 之一
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
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        SlideSparseFp8LinearMethod,
        SlideSparseFp8LinearOp,
        # 三个 kernel 函数
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        # 统一工厂函数
        wrap_scheme_fp8,
        # Extension 加载（测试用）
        _get_gemm_extension,
    )
    
    _IMPORT_SUCCESS = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import slidesparse modules: {e}. "
        "SlideSparse features will be disabled."
    )
    _IMPORT_SUCCESS = False
    
    # Fallback stub functions
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
    
    def wrap_scheme_fp8(scheme):
        return scheme
    
    def _get_gemm_extension(backend):
        raise NotImplementedError("SlideSparse import failed")
    
    # Stub classes
    class SlideSparseFp8LinearMethod:
        pass
    
    class SlideSparseFp8LinearOp:
        pass
    
    # Stub kernel functions
    def cuBLASLt_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cuSPARSELt_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")
    
    def cutlass_FP8_linear(*args, **kwargs):
        raise NotImplementedError("SlideSparse import failed")

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
    # 线性层方法
    "SlideSparseFp8LinearMethod",
    "SlideSparseFp8LinearOp",
    # 三个 kernel 函数
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    # 统一工厂函数
    "wrap_scheme_fp8",
    # Extension 加载（测试用）
    "_get_gemm_extension",
]
