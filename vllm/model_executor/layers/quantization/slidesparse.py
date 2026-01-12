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
- USE_CUSPARSELT=1: 启用 cuSPARSELt kernel (TODO)
- INNER_DTYPE_FP32=1: GEMM 输出使用 FP32（仅 cuBLASLt/cuSPARSELt 时生效）

架构说明:
=========
- cuBLASLt_FP8_linear: 纯 cuBLASLt kernel 路径
- cuSPARSELt_FP8_linear: 纯 cuSPARSELt kernel 路径 (TODO)
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
        is_inner_dtype_fp32,
        get_slidesparse_status,
    )
    from slidesparse.core.SchemeWrapperW8A8_FP8 import (
        SlideSparseSchemeWrapperFP8,
        wrap_scheme_if_enabled,
        is_slidesparse_scheme,
        # 兼容别名
        cuBLASLtSchemeWrapper,
        cuSPARSELtSchemeWrapper,
        is_cublaslt_scheme,
    )
    from slidesparse.core.SlideSparseLinearMethod_FP8 import (
        # 主要类（新名称）
        SlideSparseFp8LinearMethod,
        SlideSparseFp8LinearOp,
        # 三个 kernel 函数
        cuBLASLt_FP8_linear,
        cuSPARSELt_FP8_linear,
        cutlass_FP8_linear,
        # 工厂函数
        wrap_scheme_with_cublaslt,
        wrap_scheme_with_cusparselt,
        wrap_scheme_fp8,
        # 兼容别名
        cuBLASLtFp8LinearMethod,
        cuBLASLtFp8LinearOp,
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
    
    def is_inner_dtype_fp32():
        return False
    
    def get_slidesparse_status():
        return "SlideSparse backend UNAVAILABLE (import failed)"
    
    # Fallback stub class
    class SlideSparseSchemeWrapperFP8:
        def __init__(self, scheme):
            self._original_scheme = scheme
    
    cuBLASLtSchemeWrapper = SlideSparseSchemeWrapperFP8
    cuSPARSELtSchemeWrapper = SlideSparseSchemeWrapperFP8
    
    def wrap_scheme_if_enabled(scheme):
        return scheme
    
    def is_slidesparse_scheme(scheme):
        return False
    
    is_cublaslt_scheme = is_slidesparse_scheme
    
    def wrap_scheme_with_cublaslt(scheme):
        return scheme
    
    def wrap_scheme_with_cusparselt(scheme):
        return scheme
    
    def wrap_scheme_fp8(scheme):
        return scheme
    
    # Stub classes
    class SlideSparseFp8LinearMethod:
        pass
    
    class SlideSparseFp8LinearOp:
        pass
    
    cuBLASLtFp8LinearMethod = SlideSparseFp8LinearMethod
    cuBLASLtFp8LinearOp = SlideSparseFp8LinearOp
    
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
    "is_inner_dtype_fp32",
    "get_slidesparse_status",
    # Scheme 包装器
    "SlideSparseSchemeWrapperFP8",
    "wrap_scheme_if_enabled",
    "is_slidesparse_scheme",
    # 线性层方法（主要类）
    "SlideSparseFp8LinearMethod",
    "SlideSparseFp8LinearOp",
    # 三个 kernel 函数
    "cuBLASLt_FP8_linear",
    "cuSPARSELt_FP8_linear",
    "cutlass_FP8_linear",
    # 工厂函数
    "wrap_scheme_with_cublaslt",
    "wrap_scheme_with_cusparselt",
    "wrap_scheme_fp8",
    # 兼容别名（向后兼容）
    "cuBLASLtSchemeWrapper",
    "cuSPARSELtSchemeWrapper",
    "is_cublaslt_scheme",
    "cuBLASLtFp8LinearMethod",
    "cuBLASLtFp8LinearOp",
]
