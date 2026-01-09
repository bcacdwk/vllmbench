# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt FP8 Linear Method

这是 Phase 3 的核心实现：cuBLASLt Dense 基线搭建。

设计原则:
=========
1. 完全复用 CompressedTensorsW8A8Fp8 的 create_weights() 和 process_weights_after_loading()
2. 仅在 apply_weights() 中替换 GEMM 后端
3. 保持 quant 逻辑不变，沿用 QuantFP8 进行输入量化
4. cuBLASLt kernel 已实现，支持 Outer Vector Scaling

架构说明:
=========
原始 CompressedTensorsW8A8Fp8.apply_weights():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cutlass_scaled_mm) -> output (BF16)

我们的 CuBLASLtFp8LinearMethod.apply():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cublaslt_scaled_mm) -> output (BF16)
                      ↑                    ↑
               完全复制原代码         cuBLASLt kernel (Outer Vector Scaling)

使用方式:
=========
通过环境变量 VLLM_USE_CUBLASLT=1 启用:
    VLLM_USE_CUBLASLT=1 vllm serve model_path --quantization compressed-tensors

cuBLASLt kernel 位置:
    slidesparse/csrc/cublaslt_fp8_gemm.cu
"""

from .cublaslt_config import is_cublaslt_enabled, get_cublaslt_status

import torch
from torch.nn import Module
from typing import Optional

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

logger = init_logger(__name__)


# ============================================================================
# cuBLASLt Extension 加载
# ============================================================================

_cublaslt_ext = None
_cublaslt_ext_loaded = False


def _load_cublaslt_extension():
    """
    懒加载 cuBLASLt extension
    
    返回:
        extension module 或 None（如果加载失败）
    """
    global _cublaslt_ext, _cublaslt_ext_loaded
    
    if _cublaslt_ext_loaded:
        return _cublaslt_ext
    
    _cublaslt_ext_loaded = True
    
    try:
        import slidesparse_cublaslt
        _cublaslt_ext = slidesparse_cublaslt
        logger.info_once("cuBLASLt extension loaded successfully")
    except ImportError:
        # 尝试从 slidesparse/csrc 目录加载
        import sys
        import os
        csrc_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "csrc"
        )
        if csrc_path not in sys.path:
            sys.path.insert(0, csrc_path)
        
        try:
            import slidesparse_cublaslt
            _cublaslt_ext = slidesparse_cublaslt
            logger.info_once("cuBLASLt extension loaded successfully from csrc/")
        except ImportError as e:
            logger.warning_once(
                f"cuBLASLt extension not available: {e}. "
                "Falling back to CUTLASS. "
                "To build: cd slidesparse/csrc && python3 setup_cublaslt.py build_ext --inplace"
            )
            _cublaslt_ext = None
    
    return _cublaslt_ext


# ============================================================================
# 输入验证和预处理
# ============================================================================

def _validate_gemm_inputs(
    qinput: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    """
    验证 GEMM 输入的合法性
    
    参考 csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu 的检查逻辑
    
    Raises:
        ValueError: 输入不合法时抛出
    """
    # 1. 维度检查
    if qinput.dim() != 2:
        raise ValueError(f"qinput must be 2D, got {qinput.dim()}D")
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got {weight.dim()}D")
    
    M, K_in = qinput.shape
    K_w, N = weight.shape
    
    # 2. K 维度匹配
    if K_in != K_w:
        raise ValueError(
            f"K dimension mismatch: qinput.K={K_in} vs weight.K={K_w}"
        )
    
    # 3. 数据类型检查
    fp8_dtype = current_platform.fp8_dtype()
    if qinput.dtype != fp8_dtype:
        raise ValueError(
            f"qinput must be {fp8_dtype}, got {qinput.dtype}"
        )
    if weight.dtype != fp8_dtype:
        raise ValueError(
            f"weight must be {fp8_dtype}, got {weight.dtype}"
        )
    
    # 4. Scale 检查
    if scale_a.dtype != torch.float32:
        raise ValueError(f"scale_a must be FP32, got {scale_a.dtype}")
    if scale_b.dtype != torch.float32:
        raise ValueError(f"scale_b must be FP32, got {scale_b.dtype}")
    
    # 5. Scale 维度检查（per-token 或 per-tensor）
    # scale_a: [M, 1] 或 [1] 或 [M]
    # scale_b: [N, 1] 或 [1] 或 [N]
    scale_a_numel = scale_a.numel()
    scale_b_numel = scale_b.numel()
    
    if scale_a_numel != 1 and scale_a_numel != M:
        raise ValueError(
            f"scale_a size mismatch: expected 1 or {M}, got {scale_a_numel}"
        )
    if scale_b_numel != 1 and scale_b_numel != N:
        raise ValueError(
            f"scale_b size mismatch: expected 1 or {N}, got {scale_b_numel}"
        )
    
    # 6. Bias 检查
    if bias is not None and bias.numel() > 0:
        if bias.dim() != 1:
            raise ValueError(f"bias must be 1D, got {bias.dim()}D")
        if bias.numel() != N:
            raise ValueError(
                f"bias size mismatch: expected {N}, got {bias.numel()}"
            )
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")
    
    # 7. 设备检查
    device = qinput.device
    if weight.device != device:
        raise ValueError(
            f"weight device mismatch: expected {device}, got {weight.device}"
        )
    if scale_a.device != device:
        raise ValueError(
            f"scale_a device mismatch: expected {device}, got {scale_a.device}"
        )
    if scale_b.device != device:
        raise ValueError(
            f"scale_b device mismatch: expected {device}, got {scale_b.device}"
        )
    if bias is not None and bias.numel() > 0 and bias.device != device:
        raise ValueError(
            f"bias device mismatch: expected {device}, got {bias.device}"
        )


def _prepare_scales_for_cublaslt(
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    为 cuBLASLt Outer Vector Scaling 准备 scale
    
    cuBLASLt Outer Vector Scaling 要求:
    - scale_A (对应我们的 scale_b/weight scale): [N] 向量
    - scale_B (对应我们的 scale_a/input scale): [M] 向量
    
    注意:
    - 由于我们使用 W 左 A 右的布局，scale 顺序会交换
    - 这个函数将 scale 从 [X, 1] 或 [1] 转换为 [X] 向量
    
    Args:
        scale_a: 输入 scale [M, 1] 或 [1]
        scale_b: 权重 scale [N, 1] 或 [1]
        M: batch size
        N: output dim
    
    Returns:
        (scale_input, scale_weight): 准备好的 scale 向量
    """
    # 处理 scale_a (input scale)
    if scale_a.numel() == 1:
        # per-tensor -> 广播为 per-token
        scale_input = scale_a.expand(M).contiguous()
    else:
        # per-token: squeeze 掉多余维度
        scale_input = scale_a.view(-1).contiguous()
    
    # 处理 scale_b (weight scale)
    if scale_b.numel() == 1:
        # per-tensor -> 广播为 per-channel
        scale_weight = scale_b.expand(N).contiguous()
    else:
        # per-channel: squeeze 掉多余维度
        scale_weight = scale_b.view(-1).contiguous()
    
    return scale_input, scale_weight


# ============================================================================
# cuBLASLt GEMM 函数
# ============================================================================

# 是否使用真正的 cuBLASLt kernel（可通过环境变量控制）
import os
USE_REAL_CUBLASLT = os.environ.get("SLIDESPARSE_USE_REAL_CUBLASLT", "1") == "1"


def cublaslt_w8a8_scaled_mm(
    *,
    qinput: torch.Tensor,
    weight: torch.Tensor,
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    output_shape: list,
    **kwargs,
) -> torch.Tensor:
    """
    cuBLASLt FP8 Scaled Matrix Multiplication
    
    计算: D[M,N] = scale_a[M] * scale_b[N] * (qinput[M,K] @ weight[K,N]) + bias[N]
    
    注意: 这里的 weight 是 vLLM 经过 .t() 后的 view，stride=(1, K)
          物理内存实际是 [N, K] 行主序
    
    Args:
        qinput: 量化后的输入 [M, K] FP8，行主序
        weight: 量化后的权重 [K, N] FP8 (.t() 后的 view，实际是 [N, K] 行主序)
        out_dtype: 输出数据类型
        scale_a: 输入 scale [M, 1] 或 [1] FP32 (per-token)
        scale_b: 权重 scale [N, 1] 或 [1] FP32 (per-channel)
        bias: 偏置 [N] 或 None
        output_shape: 输出形状
        
    Returns:
        输出张量 [M, N]
    """
    M, K = qinput.shape
    N = weight.shape[1]
    
    # 输入验证
    _validate_gemm_inputs(qinput, weight, scale_a, scale_b, bias)
    
    # 尝试使用真正的 cuBLASLt kernel
    ext = _load_cublaslt_extension() if USE_REAL_CUBLASLT else None
    
    if ext is not None:
        # ========== cuBLASLt 路径 ==========
        # 
        # 关键转换:
        # 1. weight 是 [K, N] stride=(1, K)，需要 .t() 回 [N, K] stride=(K, 1)
        #    这个 .t() 只改变 view，不移动内存
        # 2. cuBLASLt 需要 [N, K] 行主序的 weight
        # 3. scale 需要从 [X, 1] 转为 [X] 向量
        
        # 恢复 weight 为 [N, K] 行主序
        # weight 当前是 [K, N] stride=(1, K)
        # .t() 后变成 [N, K] stride=(K, 1)，这是行主序！
        weight_row_major = weight.t()
        
        # 如果 weight_row_major 不是 contiguous，说明有问题
        # 正常情况下 .t() 后应该是 contiguous（因为原始就是行主序存储）
        if not weight_row_major.is_contiguous():
            logger.warning_once(
                "weight.t() is not contiguous, making contiguous copy. "
                "This may indicate a memory layout issue."
            )
            weight_row_major = weight_row_major.contiguous()
        
        # 准备 scale
        scale_input, scale_weight = _prepare_scales_for_cublaslt(
            scale_a, scale_b, M, N
        )
        
        # 准备 bias（如果没有，传空 tensor）
        if bias is None:
            bias_tensor = torch.empty(0, dtype=out_dtype, device=qinput.device)
        else:
            bias_tensor = bias
        
        # 调用 cuBLASLt kernel
        # D[M,N] = scale_input[M] * scale_weight[N] * (qinput[M,K] @ weight[N,K]^T) + bias[N]
        try:
            output = ext.cublaslt_scaled_mm(
                weight_row_major,  # W [N, K] FP8 行主序
                qinput,            # A [M, K] FP8 行主序
                scale_weight,      # scale_W [N] FP32
                scale_input,       # scale_A [M] FP32
                bias_tensor,       # bias [N] 或空
                out_dtype,         # 输出类型
            )
            return output.view(*output_shape)
        except Exception as e:
            logger.warning_once(
                f"cuBLASLt kernel failed: {e}. Falling back to CUTLASS."
            )
            # Fall through to CUTLASS
    
    # ========== CUTLASS fallback 路径 ==========
    # 使用 vLLM 原生的 cutlass_scaled_mm
    output = ops.cutlass_scaled_mm(
        qinput, weight, out_dtype=out_dtype,
        scale_a=scale_a, scale_b=scale_b, bias=bias
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
