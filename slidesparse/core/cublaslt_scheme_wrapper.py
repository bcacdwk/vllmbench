# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt Scheme 包装器

通过包装器模式，在不修改原有 Scheme 类的情况下，
替换底层的 GEMM kernel 为 cuBLASLt 实现。

设计思路:
=========
1. 包装原有的 CompressedTensorsW8A8Fp8 scheme
2. 委托大部分方法给原 scheme
3. 仅在 apply_weights 中替换 GEMM 调用

当前阶段（Phase 3 初期）:
========================
- 使用相同的 apply_weights 逻辑（调用 vllm 官方 cutlass_scaled_mm）
- 为后续替换为 cuBLASLt kernel 做好准备
- 添加 debug 日志以追踪调用
"""

import os
from typing import Any, Dict, List, Optional, Callable, Set

import torch
from torch.nn import Parameter

from vllm.logger import init_logger

from .cublaslt_config import is_cublaslt_enabled

logger = init_logger(__name__)


class CuBLASLtSchemeWrapper:
    """
    cuBLASLt Scheme 包装器
    
    包装原有的量化 Scheme，替换底层 GEMM 实现。
    当前阶段仅做包装和日志记录，GEMM 仍使用原实现。
    
    Attributes:
        _original_scheme: 被包装的原始 scheme
        _use_cublaslt: 是否使用 cuBLASLt（当前阶段始终为 True，但使用原 GEMM）
    """
    
    def __init__(self, original_scheme: Any):
        """
        初始化包装器
        
        Args:
            original_scheme: 原始的 CompressedTensors scheme 实例
        """
        self._original_scheme = original_scheme
        self._use_cublaslt = True  # 标记使用 cuBLASLt 路径
        self._call_count = 0  # 调用计数（debug 用）
        
        logger.debug(f"[SlideSparse] Created CuBLASLtSchemeWrapper for {type(original_scheme).__name__}")
    
    @property
    def original_scheme(self) -> Any:
        """获取原始 scheme"""
        return self._original_scheme
    
    def get_min_capability(self) -> int:
        """
        获取最小 GPU 计算能力要求
        
        委托给原 scheme，cuBLASLt FP8 需要 SM89+
        """
        original_cap = self._original_scheme.get_min_capability()
        # cuBLASLt FP8 需要 Ada/Hopper 架构 (SM89+)
        cublaslt_cap = 89
        return max(original_cap, cublaslt_cap)
    
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
        
        当前阶段: 完全委托给原 scheme，使用 vllm 官方的 cutlass_scaled_mm
        后续阶段: 将替换为 cuBLASLt kernel
        
        计算流程:
        1. 输入 x 量化为 FP8（原 scheme 处理）
        2. FP8 GEMM（当前: cutlass_scaled_mm，后续: cuBLASLt）
        3. 反量化 + bias（原 scheme 处理）
        
        Args:
            layer: 包含权重的层
            x: 输入张量 [batch, seq_len, hidden_size] 或 [tokens, hidden_size]
            bias: 可选的 bias 张量
            
        Returns:
            输出张量，dtype 与输入一致
        """
        self._call_count += 1
        
        # 当前阶段: 直接调用原 scheme 的 apply_weights
        # 这里包含了完整的 quant -> GEMM -> dequant 流程
        # 后续阶段将在这里替换 GEMM 部分
        output = self._original_scheme.apply_weights(layer, x, bias)
        
        return output
    
    def __getattr__(self, name: str) -> Any:
        """
        代理未定义的属性到原 scheme
        
        这确保了任何未显式覆盖的方法/属性都能正常工作
        """
        return getattr(self._original_scheme, name)


def wrap_scheme_if_enabled(scheme: Any) -> Any:
    """
    如果启用了 cuBLASLt，则包装 scheme
    
    Args:
        scheme: 原始的 CompressedTensors scheme
        
    Returns:
        如果启用 cuBLASLt，返回 CuBLASLtSchemeWrapper
        否则返回原 scheme
    """
    if is_cublaslt_enabled():
        logger.debug(f"[SlideSparse] Wrapping scheme {type(scheme).__name__} with CuBLASLtSchemeWrapper")
        return CuBLASLtSchemeWrapper(scheme)
    return scheme


def is_cublaslt_scheme(scheme: Any) -> bool:
    """检查 scheme 是否是 cuBLASLt 包装器"""
    return isinstance(scheme, CuBLASLtSchemeWrapper)
