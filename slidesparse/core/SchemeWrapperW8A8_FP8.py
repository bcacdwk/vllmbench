# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Scheme 包装器

通过包装器模式，在不修改原有 Scheme 类的情况下，
替换底层的 GEMM kernel 为 SlideSparse 实现。

设计思路:
=========
1. 包装原有的 CompressedTensorsW8A8Fp8 scheme
2. 委托大部分方法给原 scheme
3. 仅在 apply_weights 中替换 GEMM 调用
4. 内部根据环境变量选择 cuBLASLt 或 cuSPARSELt kernel

环境变量:
=========
1. DISABLE_SLIDESPARSE=1  → 完全禁用 SlideSparse，使用 vLLM 原生路径
2. USE_CUBLASLT=1         → 使用 cuBLASLt kernel
3. USE_CUSPARSELT=1       → 使用 cuSPARSELt kernel (TODO)
4. INNER_DTYPE_FP32=1     → GEMM 输出用 FP32（仅 cuBLASLt/cuSPARSELt 时生效）

默认行为:
=========
- SlideSparse 默认启用，hook 到 CompressedTensors
- 默认使用外挂 CUTLASS kernel (fallback)
- USE_CUBLASLT=1 或 USE_CUSPARSELT=1 切换到对应 kernel
"""

import os
from typing import Any, Dict, List, Optional, Callable, Set

import torch
from torch.nn import Parameter

from vllm.logger import init_logger

from .config import is_slidesparse_enabled, is_cublaslt_enabled, is_cusparselt_enabled

logger = init_logger(__name__)


class SlideSparseSchemeWrapperFP8:
    """
    SlideSparse FP8 Scheme 包装器
    
    包装原有的量化 Scheme，替换底层 GEMM 实现。
    内部根据环境变量选择 cuBLASLt 或 cuSPARSELt kernel。
    
    Attributes:
        _original_scheme: 被包装的原始 scheme
        _use_cublaslt: 是否使用 cuBLASLt kernel
        _use_cusparselt: 是否使用 cuSPARSELt kernel
    """
    
    def __init__(self, original_scheme: Any):
        """
        初始化包装器
        
        Args:
            original_scheme: 原始的 CompressedTensors scheme 实例
        """
        self._original_scheme = original_scheme
        self._use_cublaslt = is_cublaslt_enabled()
        self._use_cusparselt = is_cusparselt_enabled()
        self._call_count = 0  # 调用计数（debug 用）
        
        # 确定 kernel 名称
        if self._use_cublaslt:
            kernel_name = "cuBLASLt"
        elif self._use_cusparselt:
            kernel_name = "cuSPARSELt"
        else:
            kernel_name = "CUTLASS (fallback)"
        
        logger.debug(f"[SlideSparse] Created FP8 wrapper for {type(original_scheme).__name__}, kernel: {kernel_name}")
    
    @property
    def original_scheme(self) -> Any:
        """获取原始 scheme"""
        return self._original_scheme
    
    def get_min_capability(self) -> int:
        """
        获取最小 GPU 计算能力要求
        
        委托给原 scheme，cuBLASLt/cuSPARSELt FP8 需要 SM89+
        """
        original_cap = self._original_scheme.get_min_capability()
        # cuBLASLt/cuSPARSELt FP8 需要 Ada/Hopper 架构 (SM89+)
        slidesparse_cap = 89
        return max(original_cap, slidesparse_cap)
    
    def create_weights(
        self,
        layer: Any,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any
    ) -> Dict[str, Any]:
        """
        创建权重
        
        完全委托给原 scheme，不做任何修改。
        权重格式保持与 compressed-tensors 完全一致。
        """
        return self._original_scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            **extra_weight_attrs
        )
    
    def process_weights_after_loading(self, layer: Any) -> None:
        """
        加载权重后的处理
        
        委托给原 scheme。这包括权重转置、scale 处理等。
        """
        self._original_scheme.process_weights_after_loading(layer)
    
    def apply_weights(
        self,
        layer: Any,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        应用权重进行前向计算
        
        根据环境变量选择 kernel:
        - USE_CUBLASLT=1: 使用 cuBLASLt kernel
        - USE_CUSPARSELT=1: 使用 cuSPARSELt kernel (TODO)
        - 默认: 使用外挂 CUTLASS kernel
        
        计算流程:
        1. 输入 x 量化为 FP8（原 scheme 处理）
        2. FP8 GEMM（根据环境变量选择 kernel）
        3. 反量化 + bias（原 scheme 处理）
        
        Args:
            layer: 包含权重的层
            x: 输入张量 [batch, seq_len, hidden_size] 或 [tokens, hidden_size]
            bias: 可选的 bias 张量
            
        Returns:
            输出张量，dtype 与输入一致
        """
        self._call_count += 1
        
        # 根据配置选择 kernel
        if self._use_cusparselt:
            # TODO: 实现 cuSPARSELt kernel
            raise NotImplementedError(
                "cuSPARSELt kernel is not implemented yet. "
                "Please use USE_CUBLASLT=1 or default CUTLASS fallback."
            )
        
        # cuBLASLt 或 CUTLASS fallback: 当前都委托给原 scheme
        # 后续阶段将根据 _use_cublaslt 选择不同的 GEMM kernel
        output = self._original_scheme.apply_weights(layer, x, bias)
        
        return output
    
    def __getattr__(self, name: str) -> Any:
        """
        代理未定义的属性到原 scheme
        
        这确保了任何未显式覆盖的方法/属性都能正常工作
        """
        return getattr(self._original_scheme, name)


# ============================================================================
# 工厂函数
# ============================================================================

def wrap_scheme_if_enabled(scheme: Any) -> Any:
    """
    如果启用了 SlideSparse，则包装 scheme
    
    注意：此函数检查的是 is_slidesparse_enabled()
    - is_slidesparse_enabled(): 控制是否启用 SlideSparse hook（默认启用）
    - is_cublaslt_enabled() / is_cusparselt_enabled(): 控制使用哪种 kernel
    
    Args:
        scheme: 原始的 CompressedTensors scheme
        
    Returns:
        如果启用 SlideSparse，返回 SlideSparseSchemeWrapperFP8
        否则返回原 scheme（vLLM 原生路径）
    """
    if is_slidesparse_enabled():
        # 确定 kernel 名称用于日志
        if is_cublaslt_enabled():
            kernel_name = "cuBLASLt"
        elif is_cusparselt_enabled():
            kernel_name = "cuSPARSELt"
        else:
            kernel_name = "CUTLASS (fallback)"
        logger.debug(f"[SlideSparse] Wrapping scheme {type(scheme).__name__}, kernel: {kernel_name}")
        return SlideSparseSchemeWrapperFP8(scheme)
    return scheme


def is_slidesparse_scheme(scheme: Any) -> bool:
    """检查 scheme 是否是 SlideSparse 包装器"""
    return isinstance(scheme, SlideSparseSchemeWrapperFP8)


# ============================================================================
# 兼容别名（对称命名）
# ============================================================================

# cuBLASLt 别名
cuBLASLtSchemeWrapper = SlideSparseSchemeWrapperFP8
is_cublaslt_scheme = is_slidesparse_scheme

# cuSPARSELt 别名（指向同一个类，内部根据环境变量选择 kernel）
cuSPARSELtSchemeWrapper = SlideSparseSchemeWrapperFP8
is_cusparselt_scheme = is_slidesparse_scheme
