# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt FP8 Linear Method

这是 Phase 3 的核心实现：cuBLASLt Dense 基线搭建。

设计原则:
=========
1. 完全复用 CompressedTensorsW8A8Fp8 的 create_weights() 和 process_weights_after_loading()
2. 仅在 apply_weights() 中替换 GEMM 后端
3. 保持 quant 逻辑不变，沿用 QuantFP8 进行输入量化
4. 当前阶段：使用独立的 cublaslt_w8a8_scaled_mm 函数（内部仍调用 cutlass）

架构说明:
=========
原始 CompressedTensorsW8A8Fp8.apply_weights():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cutlass_scaled_mm) -> output (BF16)

我们的 CuBLASLtFp8LinearMethod.apply():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cublaslt_scaled_mm) -> output (BF16)
                      ↑                    ↑
               完全复制原代码         这里是替换点（当前阶段仍用 cutlass）

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
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

logger = init_logger(__name__)


# ============================================================================
# cuBLASLt GEMM 函数
# ============================================================================

def cublaslt_w8a8_scaled_mm(
    *,
    qinput: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    output_shape: list,
    **kwargs,
) -> torch.Tensor:
    """
    cuBLASLt FP8 Scaled Matrix Multiplication
    
    当前实现：调用 cutlass_scaled_mm（验证架构正确性）
    后续实现：替换为真正的 cuBLASLt kernel
    
    Args:
        qinput: 量化后的输入 [M, K] FP8
        weight: 量化后的权重 [K, N] FP8 (column-major)
        out_dtype: 输出数据类型
        scale_a: 输入 scale [M, 1] 或 [1] FP32
        scale_b: 权重 scale [N, 1] 或 [1] FP32
        bias: 偏置 [N] 或 None
        output_shape: 输出形状
        
    Returns:
        输出张量 [M, N]
    """
    # 当前阶段：调用 cutlass_scaled_mm 验证架构正确性
    # TODO: Phase 3 完成后替换为真正的 cuBLASLt kernel
    # output = ops.cublaslt_scaled_mm(...)
    output = ops.cutlass_scaled_mm(
        qinput, weight, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias
    )
    return output.view(*output_shape)


class CuBLASLtFp8LinearOp:
    """
    cuBLASLt FP8 Linear Operation
    
    这个类独立实现 FP8 Linear 操作，不依赖 vLLM 的 Fp8LinearOp：
    1. 自己创建 QuantFP8 实例进行输入量化
    2. 调用 cublaslt_w8a8_scaled_mm 执行 GEMM
    3. 便于后续的 kernel 替换和性能测试
    
    当前阶段（Phase 3）：cublaslt_w8a8_scaled_mm 内部仍调用 cutlass
    后续阶段：替换为真正的 cuBLASLt kernel
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
        
        # 创建自己的 QuantFP8 实例（不依赖 Fp8LinearOp）
        # 不使用 output padding（测试和推理时保持一致）
        self.quant_fp8 = QuantFP8(
            static=act_quant_static,
            group_shape=act_quant_group_shape,
            num_token_padding=None,  # 不使用 padding
        )
        
        logger.info_once(
            "CuBLASLtFp8LinearOp initialized "
            f"(USE_REAL_CUBLASLT={self.USE_REAL_CUBLASLT}, "
            f"static={act_quant_static}, "
            f"group_shape={act_quant_group_shape})"
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
        
        完整流程（从 Fp8LinearOp.apply 复制并修改）:
        1. input (BF16) -> quant -> qinput (FP8), x_scale
        2. qinput @ weight.T -> output (使用 cublaslt_w8a8_scaled_mm)
        3. output * scale_a * scale_b -> output (BF16)  [融合在 GEMM epilogue 中]
        
        Args:
            input: 输入张量 [..., K]，BF16/FP16
            weight: 权重张量 [K, N]（已转置，column-major），FP8
            weight_scale: 权重 scale [N, 1] 或 [1]
            out_dtype: 输出数据类型
            input_scale: 输入 scale（静态量化时使用）
            input_scale_ub: 输入 scale 上界
            bias: 偏置 [N]
            
        Returns:
            输出张量 [..., N]，BF16
        """
        # View input as 2D matrix for fp8 methods
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        
        if out_dtype is None:
            out_dtype = input.dtype
        
        # Quantize input if not already FP8
        # ops.scaled_fp8_quant supports both dynamic and static quant.
        #   If dynamic, input_scale is None and x_scale computed from input.
        #   If static, input_scale is scalar and x_scale is input_scale.
        if input.dtype != current_platform.fp8_dtype():
            qinput, x_scale = self.quant_fp8(
                input_2d,
                input_scale,
                input_scale_ub,
            )
        else:
            qinput, x_scale = input_2d, input_scale
        
        # GEMM + Dequant (使用我们的 cublaslt_w8a8_scaled_mm)
        # 当前阶段：内部调用 cutlass_scaled_mm
        # 后续阶段：替换为真正的 cuBLASLt kernel
        return cublaslt_w8a8_scaled_mm(
            qinput=qinput,
            weight=weight,
            out_dtype=out_dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
            output_shape=output_shape,
        )
    
    @staticmethod
    def apply_for_test(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype | None = None,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        act_quant_static: bool = False,
        act_quant_group_shape: GroupShape = GroupShape.PER_TOKEN,
    ) -> torch.Tensor:
        """
        Static method for standalone testing of cuBLASLt FP8 Linear operation.
        
        This method creates a temporary CuBLASLtFp8LinearOp instance and executes
        the full quant + GEMM + dequant pipeline through the cuBLASLt path.
        
        Args:
            input: Input tensor [M, K], BF16/FP16 (will be quantized to FP8)
            weight: Weight tensor [K, N], FP8 (already transposed, column-major)
            weight_scale: Weight scale [N, 1] or [1]
            out_dtype: Output dtype, defaults to input.dtype
            input_scale: Input scale (for static quantization)
            bias: Optional bias [N]
            act_quant_static: Whether to use static activation quantization
            act_quant_group_shape: Activation quantization group shape
            
        Returns:
            Output tensor [M, N], BF16
        """
        op = CuBLASLtFp8LinearOp(
            act_quant_static=act_quant_static,
            act_quant_group_shape=act_quant_group_shape,
        )
        return op.apply(
            input=input,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=out_dtype,
            input_scale=input_scale,
            input_scale_ub=None,
            bias=bias,
        )


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
