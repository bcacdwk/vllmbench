# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 配置模块

环境变量控制:
============

1. DISABLE_SLIDESPARSE=1
   - 作用：完全禁用 SlideSparse hook，使用 vLLM 原生路径
   - 默认：0（启用 SlideSparse）

2. USE_CUBLASLT=1
   - 作用：从外挂 CUTLASS 路径切换到 cuBLASLt 路径
   - 默认：0（使用外挂 CUTLASS）
   - 前提：SlideSparse 启用时生效
   - 注意：与 USE_CUSPARSELT 互斥

3. USE_CUSPARSELT=1
   - 作用：从外挂 CUTLASS 路径切换到 cuSPARSELt 路径
   - 默认：0（使用外挂 CUTLASS）
   - 前提：SlideSparse 启用时生效
   - 注意：与 USE_CUBLASLT 互斥

4. INNER_DTYPE_FP32=1
   - 作用：GEMM 输出使用 FP32 而非 BF16
   - 默认：0（使用 BF16）
   - 前提：USE_CUBLASLT=1 或 USE_CUSPARSELT=1 时生效

调度逻辑:
=========
1. DISABLE_SLIDESPARSE=1 → vLLM 原生路径（与 SlideSparse 无关）
2. DISABLE_SLIDESPARSE=0（默认）→ SlideSparse hook 路径
   2-1. USE_CUBLASLT=0 且 USE_CUSPARSELT=0（默认）→ 外挂 CUTLASS kernel（fallback）
   2-2. USE_CUBLASLT=1 → cuBLASLt kernel
   2-3. USE_CUSPARSELT=1 → cuSPARSELt kernel
   2-4. 两者同时为 1 → 报错（互斥）
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# 模块级别的配置校验标志（避免重复警告）
_config_validated = False


# ============================================================================
# 配置校验
# ============================================================================

def _validate_config():
    """校验环境变量配置的合法性（仅执行一次）"""
    global _config_validated
    if _config_validated:
        return
    _config_validated = True
    
    use_cublaslt = os.environ.get("USE_CUBLASLT", "0") == "1"
    use_cusparselt = os.environ.get("USE_CUSPARSELT", "0") == "1"
    
    if use_cublaslt and use_cusparselt:
        raise ValueError(
            "USE_CUBLASLT=1 and USE_CUSPARSELT=1 are mutually exclusive. "
            "Please set only one of them."
        )


# ============================================================================
# 主开关：SlideSparse 启用/禁用
# ============================================================================

def is_slidesparse_enabled() -> bool:
    """检查 SlideSparse 是否启用
    
    SlideSparse 默认启用，设置 DISABLE_SLIDESPARSE=1 禁用
    禁用后将使用 vLLM 原生路径
    """
    return os.environ.get("DISABLE_SLIDESPARSE", "0") != "1"


# ============================================================================
# Kernel 后端选择
# ============================================================================

def is_cublaslt_enabled() -> bool:
    """检查是否启用 cuBLASLt kernel
    
    设置 USE_CUBLASLT=1 启用
    默认为 0，使用外挂 CUTLASS 作为 fallback
    仅在 SlideSparse 启用时有意义
    
    注意：与 USE_CUSPARSELT 互斥
    """
    _validate_config()
    return os.environ.get("USE_CUBLASLT", "0") == "1"


def is_cusparselt_enabled() -> bool:
    """检查是否启用 cuSPARSELt kernel (TODO)
    
    设置 USE_CUSPARSELT=1 启用
    默认为 0，使用外挂 CUTLASS 作为 fallback
    仅在 SlideSparse 启用时有意义
    
    注意：与 USE_CUBLASLT 互斥
    """
    _validate_config()
    return os.environ.get("USE_CUSPARSELT", "0") == "1"


# ============================================================================
# GEMM 输出精度控制
# ============================================================================

def is_inner_dtype_fp32() -> bool:
    """检查 GEMM 输出是否使用 FP32
    
    设置 INNER_DTYPE_FP32=1 启用
    仅在 USE_CUBLASLT=1 或 USE_CUSPARSELT=1 时生效
    """
    return os.environ.get("INNER_DTYPE_FP32", "0") == "1"


# ============================================================================
# 状态信息
# ============================================================================

def get_slidesparse_status() -> str:
    """获取 SlideSparse 整体状态信息"""
    if not is_slidesparse_enabled():
        return "SlideSparse DISABLED (set DISABLE_SLIDESPARSE=0 to enable)"
    
    # SlideSparse 启用，检查具体 kernel 后端
    if is_cublaslt_enabled():
        inner = "FP32" if is_inner_dtype_fp32() else "BF16"
        return f"SlideSparse ENABLED, cuBLASLt kernel (inner_dtype={inner})"
    elif is_cusparselt_enabled():
        inner = "FP32" if is_inner_dtype_fp32() else "BF16"
        return f"SlideSparse ENABLED, cuSPARSELt kernel (inner_dtype={inner})"
    else:
        return "SlideSparse ENABLED, CUTLASS kernel (fallback)"
