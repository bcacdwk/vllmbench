# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt FP8 Linear Method

这是 Phase 3 的核心实现：cuBLASLt Dense 基线搭建。

设计原则:
=========
1. 完全复用 CompressedTensorsW8A8Fp8 的 create_weights() 和 process_weights_after_loading()
2. 仅在 apply_weights() 中替换 GEMM 后端
3. 保持 quant 逻辑不变，沿用 Fp8LinearOp.quant_fp8()
4. 当前阶段：使用 vllm 官方的 cutlass_scaled_mm，为 cuBLASLt 替换做准备

架构说明:
=========
原始 CompressedTensorsW8A8Fp8.apply_weights():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cutlass_scaled_mm) -> output (BF16)

我们的 CuBLASLtFp8LinearMethod.apply():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cuBLASLt/cutlass) -> output (BF16)
                      ↑                    ↑
               完全复用原代码         这里是替换点（当前阶段仍用 cutlass）

使用方式:
=========
通过环境变量 VLLM_USE_CUBLASLT=1 启用:
    VLLM_USE_CUBLASLT=1 vllm serve model_path --quantization compressed-tensors

后续 Phase 3 完成后，会替换为真正的 cuBLASLt kernel:
    slidesparse/csrc/cublaslt_fp8_gemm.cu
"""

from .cublaslt_config import is_cublaslt_enabled, get_cublaslt_status

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    cutlass_fp8_supported,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

logger = init_logger(__name__)


class CuBLASLtFp8LinearOp:
    """
    cuBLASLt FP8 Linear Operation
    
    当前阶段（Phase 3 初期）：复用 vllm 官方的 cutlass_scaled_mm
    后续阶段：替换为真正的 cuBLASLt kernel
    
    这个类的设计目的是：
    1. 为 cuBLASLt 替换提供清晰的接口
    2. 隔离 GEMM 后端的实现细节
    3. 便于后续的 kernel 替换和性能测试
    """
    
    # 标记是否使用真正的 cuBLASLt（当前为 False，使用 cutlass 作为过渡）
    USE_REAL_CUBLASLT = False
    
    def __init__(
        self,
        act_quant_static: bool = False,
        act_quant_group_shape: GroupShape = GroupShape.PER_TOKEN,
    ):
        """
        初始化 cuBLASLt FP8 Linear Op
        
        Args:
            act_quant_static: 是否使用静态激活量化
            act_quant_group_shape: 激活量化的分组形状
        """
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        
        # 复用 vllm 的 Fp8LinearOp 作为过渡实现
        # 这样可以确保 quant 逻辑完全一致
        self._fp8_linear_op = Fp8LinearOp(
            act_quant_static=act_quant_static,
            act_quant_group_shape=act_quant_group_shape,
        )
        
        logger.info_once(
            "CuBLASLtFp8LinearOp initialized "
            f"(USE_REAL_CUBLASLT={self.USE_REAL_CUBLASLT}, "
            f"static={act_quant_static})"
        )
    
    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        input_scale_ub: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        执行 FP8 Linear 操作
        
        当前实现：委托给 vllm 的 Fp8LinearOp（使用 cutlass）
        后续实现：替换为 cuBLASLt kernel
        
        流程:
        1. input (BF16) -> quant -> qinput (FP8), scale_a
        2. qinput @ weight.T -> output_raw (FP32/INT32)  [这里是 GEMM]
        3. output_raw * scale_a * scale_b -> output (BF16)  [这里是 Dequant]
        
        Args:
            input: 输入张量 [M, K]，BF16/FP16
            weight: 权重张量 [K, N]（转置后），FP8
            weight_scale: 权重 scale
            out_dtype: 输出数据类型
            input_scale: 输入 scale（静态量化时使用）
            input_scale_ub: 输入 scale 上界
            bias: 偏置
            
        Returns:
            输出张量 [M, N]，BF16
        """
        # 注意：不要在 apply() 中使用 logger，会导致 torch.compile 图中断
        # logger 调用已移至 __init__
        
        if self.USE_REAL_CUBLASLT:
            # TODO: Phase 3 完成后替换为真正的 cuBLASLt kernel
            # return self._apply_cublaslt(input, weight, weight_scale, ...)
            raise NotImplementedError(
                "Real cuBLASLt kernel not implemented yet. "
                "This will be added in Phase 3 completion."
            )
        else:
            # 当前阶段：使用 vllm 官方的 Fp8LinearOp
            # 这确保了完全的功能兼容性
            return self._fp8_linear_op.apply(
                input=input,
                weight=weight,
                weight_scale=weight_scale,
                out_dtype=out_dtype,
                input_scale=input_scale,
                input_scale_ub=input_scale_ub,
                bias=bias,
            )
    
    def _apply_cublaslt(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        使用真正的 cuBLASLt kernel 执行 FP8 GEMM
        
        TODO: Phase 3 完成时实现
        
        实现要点:
        1. 调用 slidesparse.csrc.cublaslt_fp8_gemm 模块
        2. 使用 T/N + Col/Col + Col 的 layout
        3. 支持 BF16 -> FP8 -> BF16 的数据流
        4. 融合 Dequant 到 epilogue 中
        """
        raise NotImplementedError("cuBLASLt kernel will be implemented in Phase 3")


class CuBLASLtFp8LinearMethod:
    """
    cuBLASLt FP8 Linear Method
    
    这是一个独立的 LinearMethod 实现，用于验证 cuBLASLt 替换的可行性。
    
    关键设计:
    1. 不继承任何 vllm 的类，避免依赖问题
    2. 所有方法签名与 LinearMethodBase 兼容
    3. 内部复用 CompressedTensorsW8A8Fp8 的 create_weights 和 process_weights_after_loading
    """
    
    def __init__(self, original_scheme):
        """
        初始化 cuBLASLt FP8 Linear Method
        
        Args:
            original_scheme: 原始的 CompressedTensorsW8A8Fp8 scheme
                            我们复用它的 create_weights 和 process_weights_after_loading
        """
        self.original_scheme = original_scheme
        self.out_dtype = original_scheme.out_dtype
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.act_q_group_shape = original_scheme.act_q_group_shape
        
        # 创建我们的 cuBLASLt Op
        self.cublaslt_fp8_linear = CuBLASLtFp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape,
        )
        
        logger.info_once(
            "CuBLASLtFp8LinearMethod initialized, "
            f"wrapping original scheme: {type(original_scheme).__name__}"
        )
    
    def create_weights(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ):
        """
        创建权重参数
        
        完全委托给原始 scheme，保证权重格式完全兼容
        """
        return self.original_scheme.create_weights(
            layer=layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=output_size,
            params_dtype=params_dtype,
            weight_loader=weight_loader,
            **kwargs,
        )
    
    def process_weights_after_loading(self, layer: Module) -> None:
        """
        权重加载后处理
        
        完全委托给原始 scheme，保证权重处理完全兼容
        """
        return self.original_scheme.process_weights_after_loading(layer)
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        应用权重（执行线性变换）
        
        这是我们替换 GEMM 后端的关键点！
        
        当前实现：使用 CuBLASLtFp8LinearOp（内部委托给 cutlass）
        后续实现：CuBLASLtFp8LinearOp 内部替换为真正的 cuBLASLt kernel
        """
        return self.cublaslt_fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=getattr(layer, "input_scale", None),
            bias=bias,
        )


def wrap_scheme_with_cublaslt(original_scheme):
    """
    工厂函数：将原始 scheme 包装为 cuBLASLt 版本
    
    这个函数用于在 compressed_tensors.py 中进行最小侵入式修改。
    
    使用示例（在 compressed_tensors.py 的 get_scheme() 中）:
        if is_cublaslt_enabled():
            scheme = wrap_scheme_with_cublaslt(scheme)
        return scheme
    
    Args:
        original_scheme: 原始的 CompressedTensorsScheme（如 CompressedTensorsW8A8Fp8）
        
    Returns:
        如果 cuBLASLt 启用且 scheme 是 W8A8Fp8，返回 CuBLASLtFp8LinearMethod 包装后的 scheme
        否则返回原始 scheme
    """
    # 首先检查 cuBLASLt 是否启用
    if not is_cublaslt_enabled():
        return original_scheme
    
    # 检查是否是 W8A8Fp8 scheme
    scheme_name = type(original_scheme).__name__
    if "W8A8Fp8" in scheme_name:
        logger.info_once(
            f"Wrapping {scheme_name} with CuBLASLtFp8LinearMethod"
        )
        return CuBLASLtFp8LinearMethod(original_scheme)
    else:
        logger.warning_once(
            f"cuBLASLt wrapper not supported for {scheme_name}, "
            "using original scheme"
        )
        return original_scheme
