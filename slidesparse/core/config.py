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

4. INNER_DTYPE_32=1
   - 作用：GEMM 使用高精度累加（FP8→FP32, INT8→INT32）
   - 默认：0（使用 BF16）
   - 前提：USE_CUBLASLT=1 或 USE_CUSPARSELT=1 时生效

5. SPARSITY=2_8
   - 作用：指定稀疏格式 Z:L（如 2:8, 2:6, 2:10）
   - 默认：2_8
   - 前提：USE_CUSPARSELT=1 时生效（在线压缩需要知道稀疏格式）
   - 格式：Z_L（如 "2_8" 表示 2:8 稀疏）

调度逻辑:
=========
1. DISABLE_SLIDESPARSE=1 → vLLM 原生路径（与 SlideSparse 无关）
2. DISABLE_SLIDESPARSE=0（默认）→ SlideSparse hook 路径
   2-1. USE_CUBLASLT=0 且 USE_CUSPARSELT=0（默认）→ 外挂 CUTLASS kernel（fallback）
   2-2. USE_CUBLASLT=1 → cuBLASLt kernel
   2-3. USE_CUSPARSELT=1 → cuSPARSELt kernel（需要 SPARSITY 配置）
   2-4. 两者同时为 1 → 报错（互斥）
"""

import os

from vllm.logger import init_logger

# 从顶层 utils 导入稀疏配置解析函数
from slidesparse.utils import (
    get_sparsity_config_cached,
    get_sparsity_str,
    clear_sparsity_cache,
)

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
    """Check if cuSPARSELt kernel is enabled
    
    Set USE_CUSPARSELT=1 to enable.
    Mutually exclusive with USE_CUBLASLT.
    """
    _validate_config()
    return os.environ.get("USE_CUSPARSELT", "0") == "1"


# ============================================================================
# GEMM 输出精度控制
# ============================================================================

def is_inner_dtype_32() -> bool:
    """检查 GEMM 是否使用高精度累加
    
    设置 INNER_DTYPE_32=1 启用
    FP8 输入时使用 FP32，INT8 输入时使用 INT32
    仅在 USE_CUBLASLT=1 或 USE_CUSPARSELT=1 时生效, 且 cuBLASLt + INT8输入的情况是开启的, 因为不支持BF16输出
    """
    return os.environ.get("INNER_DTYPE_32", "0") == "1"


# ============================================================================
# 稀疏格式配置（委托给顶层 utils）
# ============================================================================

def get_sparsity_config() -> tuple:
    """
    获取稀疏格式配置
    
    委托给 slidesparse.utils.get_sparsity_config_cached()
    
    Returns:
        (Z, L, expand_ratio) 元组
    """
    return get_sparsity_config_cached()


# ============================================================================
# 状态信息
# ============================================================================

def get_slidesparse_status() -> str:
    """获取 SlideSparse 整体状态信息"""
    if not is_slidesparse_enabled():
        return "SlideSparse DISABLED (set DISABLE_SLIDESPARSE=0 to enable)"
    
    # SlideSparse 启用，检查具体 kernel 后端
    if is_cublaslt_enabled():
        inner = "FP32/INT32" if is_inner_dtype_32() else "BF16"
        return f"SlideSparse ENABLED, cuBLASLt kernel (inner_dtype={inner})"
    elif is_cusparselt_enabled():
        inner = "FP32/INT32" if is_inner_dtype_32() else "BF16"
        Z, L, expand_ratio = get_sparsity_config()
        return (f"SlideSparse ENABLED, cuSPARSELt kernel "
                f"(inner_dtype={inner}, sparsity={Z}:{L}, expand_ratio={expand_ratio:.3f})")
    else:
        return "SlideSparse ENABLED, CUTLASS kernel (fallback)"
