# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 核心逻辑模块

包含:
- config: SlideSparse 配置函数
- profiler: 计时诊断模块
- kernels: Triton kernel 加载
- gemm_wrapper: GEMM wrapper 和 Custom Op
- SlideSparseLinearMethod_FP8: FP8 线性层方法
- SlideSparseLinearMethod_INT8: INT8 线性层方法

环境变量:
=========
1. DISABLE_SLIDESPARSE=1  → 完全禁用 SlideSparse，使用 vLLM 原生路径
2. USE_CUBLASLT=1         → 使用 cuBLASLt kernel
3. USE_CUSPARSELT=1       → 使用 cuSPARSELt kernel
4. INNER_DTYPE_32=1       → GEMM 使用高精度累加（FP8→FP32, INT8→INT32）
5. SPARSITY=2_8           → 稀疏格式（仅 cuSPARSELt 时生效，默认 2_8）
6. SLIDESPARSE_PROFILE=1  → 启用计时诊断

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
from slidesparse.core.gemm_wrapper import (
    _get_gemm_extension,
    get_algo_config_manager,
    AlgorithmConfigManager,
)
from slidesparse.core.profiler import print_profile_stats, reset_profile_stats


# ============================================================================
# 初始化函数
# ============================================================================

def init_slidesparse(model_name: str) -> None:
    """
    初始化 SlideSparse 系统
    
    必须在使用任何 SlideSparse GEMM kernel 之前调用（CUTLASS fallback 除外）。
    通常在模型加载时调用，或在测试开始前调用。
    
    此函数会：
    1. 设置当前模型名称（用于加载 model-specific 的 tuned kernels）
    2. 预加载 GEMM 算法配置（如果有离线搜索结果）
    
    Args:
        model_name: 模型名称，例如 "Qwen2.5-0.5B-FP8", "Llama3.2-1B-INT8"
                    应与 checkpoints 目录名一致
    
    Example:
        >>> from slidesparse.core import init_slidesparse
        >>> init_slidesparse("Llama3.2-1B-FP8")
        >>> # 现在可以使用 cuBLASLt/cuSPARSELt kernel 了
    """
    manager = get_algo_config_manager()
    manager.set_model(model_name)
    # load_all_configs 在 get_algo_config_manager() 中已自动调用


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
    
    # 初始化
    "init_slidesparse",
    "get_algo_config_manager",
    "AlgorithmConfigManager",
]