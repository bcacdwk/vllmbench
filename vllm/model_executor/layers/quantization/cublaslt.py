# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SlideSparse cuBLASLt 集成入口 - vLLM 空壳转发文件

这个文件是 SlideSparse 与 vLLM 集成的桥梁。
它负责将外挂模块 (slidesparse/) 的功能导入到 vLLM 的量化框架中。

设计原则:
=========
1. 最小侵入式修改：这个文件是 vLLM 源码中唯一需要新增的文件
2. 所有逻辑都在外挂模块 (slidesparse/) 中实现
3. 这里只做导入和转发

使用方式:
=========
通过环境变量 VLLM_USE_CUBLASLT=1 启用:
    VLLM_USE_CUBLASLT=1 vllm serve model_path --quantization compressed-tensors

不需要修改 --quantization 参数，保持使用 compressed-tensors。
cuBLASLt 后端会在 CompressedTensorsW8A8Fp8 的基础上进行透明替换。
"""

import os
import sys

# 将 slidesparse 模块添加到 Python 路径
# 仓库根目录计算: vllm/model_executor/layers/quantization/cublaslt.py
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
    from slidesparse.core.cublaslt_config import (
        VLLM_USE_CUBLASLT,
        SLIDESPARSE_USE_CUBLASLT,
        SLIDESPARSE_CUBLASLT_DEBUG,
        get_cublaslt_status,
        is_cublaslt_enabled,
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
    
    _IMPORT_SUCCESS = True
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import slidesparse modules: {e}. "
        "cuBLASLt features will be disabled."
    )
    _IMPORT_SUCCESS = False
    VLLM_USE_CUBLASLT = False
    SLIDESPARSE_USE_CUBLASLT = False
    SLIDESPARSE_CUBLASLT_DEBUG = False
    
    # Fallback stub class
    class CuBLASLtSchemeWrapper:
        def __init__(self, scheme):
            self._original_scheme = scheme
    
    def is_cublaslt_enabled():
        return False
    
    def get_cublaslt_status():
        return "cuBLASLt backend UNAVAILABLE (import failed)"
    
    def wrap_scheme_if_enabled(scheme):
        return scheme
    
    def wrap_scheme_with_cublaslt(scheme):
        return scheme
    
    def is_cublaslt_scheme(scheme):
        return False

__all__ = [
    # 配置相关
    "VLLM_USE_CUBLASLT",
    "SLIDESPARSE_USE_CUBLASLT",
    "SLIDESPARSE_CUBLASLT_DEBUG",
    "is_cublaslt_enabled",
    "get_cublaslt_status",
    # Scheme 包装器
    "CuBLASLtSchemeWrapper",
    "wrap_scheme_if_enabled",
    "is_cublaslt_scheme",
    # 线性层方法 (保留兼容)
    "wrap_scheme_with_cublaslt",
    "CuBLASLtFp8LinearMethod",
    "CuBLASLtFp8LinearOp",
]
