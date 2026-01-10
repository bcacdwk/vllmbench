# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt FP8 配置模块

精简的环境变量控制，只有两个开关：

环境变量:
=========
1. USE_CUBLASLT=1
   - 作用：从 CUTLASS 路径切换到 cuBLASLt 路径
   - 默认：0（使用 CUTLASS）

2. INNER_DTYPE_FP32=1
   - 作用：cuBLASLt GEMM 输出使用 FP32 而非 BF16
   - 默认：0（使用 BF16）
   - 前提：只有 USE_CUBLASLT=1 时才生效
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)


# ============================================================================
# 环境变量配置（只有两个开关）
# ============================================================================

def is_cublaslt_enabled() -> bool:
    """检查是否启用 cuBLASLt 后端
    
    设置 USE_CUBLASLT=1 启用
    """
    return os.environ.get("USE_CUBLASLT", "0") == "1"


def is_inner_dtype_fp32() -> bool:
    """检查 GEMM 输出是否使用 FP32
    
    设置 INNER_DTYPE_FP32=1 启用（仅在 USE_CUBLASLT=1 时生效）
    """
    return os.environ.get("INNER_DTYPE_FP32", "0") == "1"


def get_cublaslt_status() -> str:
    """获取 cuBLASLt 状态信息"""
    if is_cublaslt_enabled():
        inner = "FP32" if is_inner_dtype_fp32() else "BF16"
        return f"cuBLASLt ENABLED (inner_dtype={inner})"
    else:
        return "cuBLASLt DISABLED (set USE_CUBLASLT=1 to enable)"
