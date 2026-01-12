# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse FP8 Linear Method

设计原则:
=========
1. 完全复用 CompressedTensorsW8A8Fp8 的 create_weights() 和 process_weights_after_loading()
2. 仅在 apply_weights() 中替换 GEMM 后端
3. cuBLASLt 只做纯矩阵乘法，不融合 scale/bias
4. Dequant + bias 由后续 Triton kernel 处理（TODO）

架构说明:
=========
原始 CompressedTensorsW8A8Fp8.apply_weights():
    input (BF16) -> quant (FP8) -> GEMM+Dequant (cutlass_scaled_mm) -> output (BF16)

我们的 cuBLASLtFp8LinearMethod.apply():
    input (BF16) -> quant (FP8) -> GEMM (cublaslt_mm) -> Dequant+Bias (Triton) -> output (BF16)
                      ↑                    ↑                    ↑
               完全复制原代码      cuBLASLt kernel        TODO: Triton kernel
                              (无 scale/bias 融合)

数据类型说明:
============
- input_dtype: 输入量化精度，目前支持 FP8E4M3（Python 端固定）
- inner_dtype: GEMM 输出精度，BF16（默认）或 FP32（通过环境变量控制）
- out_dtype: 最终输出精度，由 vLLM 上层指定，经过 dequant 后得到

环境变量:
=========
1. DISABLE_SLIDESPARSE=1  → 完全禁用 SlideSparse，使用 vLLM 原生路径
2. USE_CUBLASLT=1         → 从外挂 CUTLASS 切换到 cuBLASLt kernel
3. INNER_DTYPE_FP32=1     → GEMM 输出用 FP32（仅 USE_CUBLASLT=1 时生效）

cuBLASLt kernel 位置:
    slidesparse/csrc/cublaslt_gemm.cu
"""

from .config import is_slidesparse_enabled, is_cublaslt_enabled, is_cusparselt_enabled, is_inner_dtype_fp32, get_slidesparse_status

import os
import sys
import platform
from pathlib import Path
from typing import Optional

import torch
from torch.nn import Module

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

logger = init_logger(__name__)


# ============================================================================
# 环境变量配置（从 cublaslt_config 统一管理）
# ============================================================================

def get_inner_dtype_str() -> str:
    """获取 GEMM 输出精度字符串"""
    return "fp32" if is_inner_dtype_fp32() else "bf16"


def get_inner_dtype_torch() -> torch.dtype:
    """获取 GEMM 输出精度的 PyTorch dtype"""
    return torch.float32 if is_inner_dtype_fp32() else torch.bfloat16


# ============================================================================
# cuBLASLt Extension 加载
# ============================================================================

_cublaslt_ext = None
_cublaslt_ext_loaded = False


def _get_extension_name() -> str:
    """
    生成与编译脚本一致的扩展名
    
    格式: slidesparse_cublaslt_py312_x86_64_cc90
    """
    # Python 版本
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    
    # 系统架构
    machine = platform.machine()
    if machine in ("x86_64", "AMD64"):
        arch_tag = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch_tag = "aarch64"
    else:
        arch_tag = machine.lower()
    
    # GPU CC
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        cc_tag = f"cc{prop.major}{prop.minor}"
    else:
        cc_tag = "cc00"  # fallback
    
    return f"slidesparse_cublaslt_{py_tag}_{arch_tag}_{cc_tag}"


def _load_cublaslt_extension():
    """
    懒加载 cuBLASLt extension
    
    加载顺序:
    1. 尝试直接 import（如果已安装到 Python 环境）
    2. 尝试从 slidesparse/csrc/build 目录加载对应架构的 .so
    
    返回:
        extension module 或 None（如果加载失败）
    """
    global _cublaslt_ext, _cublaslt_ext_loaded
    
    if _cublaslt_ext_loaded:
        return _cublaslt_ext
    
    _cublaslt_ext_loaded = True
    ext_name = _get_extension_name()
    
    # 方法 1: 尝试直接 import
    try:
        import importlib
        _cublaslt_ext = importlib.import_module(ext_name)
        logger.info_once(f"cuBLASLt extension loaded: {ext_name}")
        return _cublaslt_ext
    except ImportError:
        pass
    
    # 方法 2: 从 build 目录加载
    csrc_dir = Path(__file__).parent.parent / "csrc"
    build_dir = csrc_dir / "build"
    
    if build_dir.exists():
        # 查找匹配的 .so 文件
        so_pattern = f"{ext_name}*.so"
        so_files = list(build_dir.glob(so_pattern))
        
        if so_files:
            so_path = so_files[0]
            
            # 将 build 目录加入 sys.path
            build_dir_str = str(build_dir)
            if build_dir_str not in sys.path:
                sys.path.insert(0, build_dir_str)
            
            try:
                import importlib
                _cublaslt_ext = importlib.import_module(ext_name)
                logger.info_once(
                    f"cuBLASLt extension loaded from build/: {so_path.name}"
                )
                return _cublaslt_ext
            except ImportError as e:
                logger.warning_once(
                    f"Failed to load {so_path.name}: {e}"
                )
    
    # 加载失败
    logger.warning_once(
        f"cuBLASLt extension not available ({ext_name}). "
        "Falling back to CUTLASS. "
        "To build: cd slidesparse/csrc && python setup_cublaslt.py build"
    )
    _cublaslt_ext = None
    return _cublaslt_ext


# ============================================================================
# Dequant + Bias Kernel (TODO: Triton 实现)
# ============================================================================

def dequant_bias_kernel(
    gemm_output: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Dequant + Bias 操作
    
    计算: output = gemm_output * scale_a[M,1] * scale_b[1,N] + bias[1,N]
    
    TODO: 实现 Triton kernel 以获得更好的性能
          当前使用 PyTorch 原生操作作为 placeholder
    
    Args:
        gemm_output: GEMM 输出 [M, N]，inner_dtype（BF16 或 FP32）
        scale_a: 输入 scale [M, 1] 或 [1] FP32
        scale_b: 权重 scale [N, 1] 或 [1] FP32
        bias: 偏置 [N] BF16 或 None
        out_dtype: 输出数据类型
        
    Returns:
        dequant 后的输出 [M, N]，out_dtype
    """
    M, N = gemm_output.shape
    
    # 准备 scale 的广播形状
    # scale_a: [M, 1] 或 [1] -> [M, 1]
    if scale_a.numel() == 1:
        scale_a_broadcast = scale_a.view(1, 1)
    else:
        scale_a_broadcast = scale_a.view(-1, 1)
    
    # scale_b: [N, 1] 或 [1] -> [1, N]
    if scale_b.numel() == 1:
        scale_b_broadcast = scale_b.view(1, 1)
    else:
        scale_b_broadcast = scale_b.view(1, -1)
    
    # 计算 dequant: output = gemm_output * scale_a * scale_b
    # 先转为 FP32 计算以保证精度
    output = gemm_output.float() * scale_a_broadcast * scale_b_broadcast
    
    # 加 bias
    if bias is not None and bias.numel() > 0:
        output = output + bias.float().view(1, -1)
    
    # 转换为目标精度
    return output.to(out_dtype)


# ============================================================================
# FP8 Linear 函数（三个平级的 kernel 路径）
# ============================================================================

def cuBLASLt_FP8_linear(
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
    cuBLASLt FP8 Linear（完整流程：GEMM + Dequant + Bias）
    
    调用链:
        cuBLASLt_FP8_linear()           # Python: 完整 Linear
          ├── ext.cublaslt_fp8_mm()      # CUDA: 纯 GEMM
          └── dequant_bias_kernel()      # Python/Triton: dequant
    
    计算流程:
    1. GEMM: inner_output[M,N] = qinput[M,K] @ weight[N,K]^T  (cuBLASLt)
    2. Dequant: output[M,N] = inner_output * scale_a * scale_b + bias
    
    注意: weight 是 vLLM 经过 .t() 后的 view，stride=(1, K)
          物理内存实际是 [N, K] 行主序
    
    Args:
        qinput: 量化后的输入 [M, K] FP8E4M3，行主序
        weight: 量化后的权重 [K, N] FP8 (.t() 后的 view，实际是 [N, K] 行主序)
        out_dtype: 最终输出数据类型（经过 dequant 后）
        scale_a: 输入 scale [M, 1] 或 [1] FP32 (per-token)
        scale_b: 权重 scale [N, 1] 或 [1] FP32 (per-channel)
        bias: 偏置 [N] 或 None
        output_shape: 输出形状
        
    Returns:
        输出张量 [*output_shape]，out_dtype
    """
    # 必须成功加载 extension，否则报错
    ext = _load_cublaslt_extension()
    if ext is None:
        raise RuntimeError(
            "cuBLASLt extension failed to load. "
            "Please build the extension: cd slidesparse/csrc && python setup_cublaslt.py build"
        )
    
    # 关键转换:
    # weight 是 [K, N] stride=(1, K)，需要 .t() 回 [N, K] stride=(K, 1)
    # 这个 .t() 只改变 view，不移动内存
    weight_row_major = weight.t()
    
    if not weight_row_major.is_contiguous():
        logger.warning_once(
            "weight.t() is not contiguous, making contiguous copy. "
            "This may indicate a memory layout issue."
        )
        weight_row_major = weight_row_major.contiguous()
    
    # 获取 inner_dtype
    inner_dtype_str = get_inner_dtype_str()
    
    try:
        # 调用 cuBLASLt kernel（纯 GEMM，无 scale/bias）
        # D[M,N] = qinput[M,K] @ weight[N,K]^T
        gemm_output = ext.cublaslt_fp8_mm(
            weight_row_major,  # W [N, K] FP8 行主序
            qinput,            # A [M, K] FP8 行主序
            inner_dtype_str,   # "bf16" 或 "fp32"
        )
        
        # Dequant + Bias (TODO: 替换为 Triton kernel)
        output = dequant_bias_kernel(
            gemm_output=gemm_output,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=bias,
            out_dtype=out_dtype,
        )
        
        return output.view(*output_shape)
        
    except Exception as e:
        raise RuntimeError(
            f"cuBLASLt kernel execution failed: {e}. "
            "If this is unexpected, check GPU compatibility or rebuild the extension."
        ) from e


def cuSPARSELt_FP8_linear(
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
    cuSPARSELt FP8 Linear（完整流程：稀疏 GEMM + Dequant + Bias）
    
    TODO: 实现 cuSPARSELt 稀疏加速
    
    调用链:
        cuSPARSELt_FP8_linear()          # Python: 完整 Linear
          ├── ext.cusparselt_fp8_mm()     # CUDA: 稀疏 GEMM (TODO)
          └── dequant_bias_kernel()       # Python/Triton: dequant
    
    计算流程:
    1. GEMM: inner_output[M,N] = qinput[M,K] @ sparse_weight[N,K]^T  (cuSPARSELt)
    2. Dequant: output[M,N] = inner_output * scale_a * scale_b + bias
    
    Args:
        qinput: 量化后的输入 [M, K] FP8E4M3，行主序
        weight: 量化后的权重 [K, N] FP8（稀疏格式）
        out_dtype: 最终输出数据类型（经过 dequant 后）
        scale_a: 输入 scale [M, 1] 或 [1] FP32 (per-token)
        scale_b: 权重 scale [N, 1] 或 [1] FP32 (per-channel)
        bias: 偏置 [N] 或 None
        output_shape: 输出形状
        
    Returns:
        输出张量 [*output_shape]，out_dtype
    """
    # TODO: 实现 cuSPARSELt extension 加载和调用
    raise NotImplementedError(
        "cuSPARSELt FP8 Linear is not implemented yet. "
        "Please use USE_CUBLASLT=1 or default CUTLASS fallback."
    )


def cutlass_FP8_linear(
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
    CUTLASS FP8 Linear（vLLM 原生 cutlass_scaled_mm 的薄包装）
    
    这是 SlideSparse 的默认 fallback 路径。
    直接调用 vLLM 的 ops.cutlass_scaled_mm。
    
    Args:
        qinput: 量化后的输入 [M, K] FP8E4M3
        weight: 量化后的权重 [K, N] FP8
        out_dtype: 最终输出数据类型
        scale_a: 输入 scale [M, 1] 或 [1] FP32
        scale_b: 权重 scale [N, 1] 或 [1] FP32
        bias: 偏置 [N] 或 None
        output_shape: 输出形状
        
    Returns:
        输出张量 [*output_shape]，out_dtype
    """
    output = ops.cutlass_scaled_mm(
        qinput, weight, out_dtype=out_dtype,
        scale_a=scale_a, scale_b=scale_b, bias=bias
    )
    return output.view(*output_shape)


# ============================================================================
# SlideSparse FP8 Linear Op（统一入口，根据环境变量选择 kernel）
# ============================================================================

class SlideSparseFp8LinearOp:
    """
    SlideSparse FP8 Linear Operation
    
    这个类独立实现 FP8 Linear 操作，不依赖 vLLM 的 Fp8LinearOp：
    1. 自己创建 QuantFP8 实例进行输入量化
    2. 根据环境变量选择 kernel 路径：
       - USE_CUBLASLT=1: cuBLASLt_FP8_linear
       - USE_CUSPARSELT=1: cuSPARSELt_FP8_linear
       - 默认: cutlass_FP8_linear (fallback)
    
    调用链:
        SlideSparseFp8LinearOp.apply()
          ├── cuBLASLt_FP8_linear()    # USE_CUBLASLT=1
          ├── cuSPARSELt_FP8_linear()  # USE_CUSPARSELT=1
          └── cutlass_FP8_linear()     # 默认 fallback
    """
    
    def __init__(
        self,
        act_quant_static: bool = False,
        act_quant_group_shape: GroupShape = GroupShape.PER_TOKEN,
    ):
        """
        初始化 SlideSparse FP8 Linear Op
        
        Args:
            act_quant_static: 是否使用静态激活量化
            act_quant_group_shape: 激活量化的分组形状
        """
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        
        # 创建自己的 QuantFP8 实例（不依赖 Fp8LinearOp）
        self.quant_fp8 = QuantFP8(
            static=act_quant_static,
            group_shape=act_quant_group_shape,
            num_token_padding=None,
        )
        
        # 确定 kernel 路径
        if is_cublaslt_enabled():
            self._kernel_name = "cuBLASLt"
            self._linear_fn = cuBLASLt_FP8_linear
        elif is_cusparselt_enabled():
            self._kernel_name = "cuSPARSELt"
            self._linear_fn = cuSPARSELt_FP8_linear
        else:
            self._kernel_name = "CUTLASS"
            self._linear_fn = cutlass_FP8_linear
        
        logger.info_once(
            f"SlideSparseFp8LinearOp initialized "
            f"(kernel={self._kernel_name}, "
            f"inner_dtype={get_inner_dtype_str() if is_cublaslt_enabled() or is_cusparselt_enabled() else 'N/A'}, "
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
        
        完整流程:
        1. input (BF16) -> quant -> qinput (FP8), x_scale
        2. qinput @ weight.T -> inner_output (选定的 GEMM kernel)
        3. inner_output * scale_a * scale_b + bias -> output (Dequant)
        
        Args:
            input: 输入张量 [..., K]，BF16/FP16
            weight: 权重张量 [K, N]（已转置，column-major），FP8
            weight_scale: 权重 scale [N, 1] 或 [1]
            out_dtype: 输出数据类型（由 vLLM 上层指定）
            input_scale: 输入 scale（静态量化时使用）
            input_scale_ub: 输入 scale 上界
            bias: 偏置 [N]
            
        Returns:
            输出张量 [..., N]，out_dtype
        """
        # View input as 2D matrix
        input_2d = input.view(-1, input.shape[-1])
        output_shape = [*input.shape[:-1], weight.shape[1]]
        
        if out_dtype is None:
            out_dtype = input.dtype
        
        # Quantize input if not already FP8
        if input.dtype != current_platform.fp8_dtype():
            qinput, x_scale = self.quant_fp8(
                input_2d,
                input_scale,
                input_scale_ub,
            )
        else:
            qinput, x_scale = input_2d, input_scale
        
        # 调用选定的 kernel 路径
        return self._linear_fn(
            qinput=qinput,
            weight=weight,
            out_dtype=out_dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
            output_shape=output_shape,
        )


# 兼容别名
cuBLASLtFp8LinearOp = SlideSparseFp8LinearOp


# ============================================================================
# SlideSparse FP8 Linear Method
# ============================================================================

class SlideSparseFp8LinearMethod:
    """
    SlideSparse FP8 Linear Method
    
    这是一个独立的 LinearMethod 实现，根据环境变量选择 kernel 路径。
    
    关键设计:
    1. 不继承任何 vllm 的类，避免依赖问题
    2. 所有方法签名与 LinearMethodBase 兼容
    3. 内部复用 CompressedTensorsW8A8Fp8 的 create_weights 和 process_weights_after_loading
    4. 通过 SlideSparseFp8LinearOp 选择 cuBLASLt/cuSPARSELt/CUTLASS 路径
    """
    
    def __init__(self, original_scheme):
        """
        初始化 SlideSparse FP8 Linear Method
        
        Args:
            original_scheme: 原始的 CompressedTensorsW8A8Fp8 scheme
        """
        self.original_scheme = original_scheme
        self.out_dtype = original_scheme.out_dtype
        self.is_static_input_scheme = original_scheme.is_static_input_scheme
        self.act_q_group_shape = original_scheme.act_q_group_shape
        
        # 创建 SlideSparse Op（内部根据环境变量选择 kernel）
        self.slidesparse_fp8_linear = SlideSparseFp8LinearOp(
            act_quant_static=self.is_static_input_scheme,
            act_quant_group_shape=self.act_q_group_shape,
        )
        
        logger.info_once(
            f"SlideSparseFp8LinearMethod initialized, "
            f"wrapping: {type(original_scheme).__name__}, "
            f"kernel: {self.slidesparse_fp8_linear._kernel_name}"
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
        """创建权重参数，完全委托给原始 scheme"""
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
        """权重加载后处理，完全委托给原始 scheme"""
        return self.original_scheme.process_weights_after_loading(layer)
    
    def apply_weights(
        self,
        layer: Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """应用权重（执行线性变换），使用选定的 kernel 路径"""
        return self.slidesparse_fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=getattr(layer, "input_scale", None),
            bias=bias,
        )


# 兼容别名
cuBLASLtFp8LinearMethod = SlideSparseFp8LinearMethod


# ============================================================================
# 工厂函数
# ============================================================================


def wrap_scheme_with_cublaslt(original_scheme):
    """
    工厂函数：将原始 scheme 包装为 cuBLASLt 版本
    
    使用示例（在 compressed_tensors.py 的 get_scheme() 中）:
        if is_cublaslt_enabled():
            scheme = wrap_scheme_with_cublaslt(scheme)
        return scheme
    
    Args:
        original_scheme: 原始的 CompressedTensorsScheme
        
    Returns:
        如果 scheme 是 W8A8Fp8，返回包装后的 scheme
        否则返回原始 scheme
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Fp8" in scheme_name:
        logger.info_once(
            f"Wrapping {scheme_name} with SlideSparseFp8LinearMethod (cuBLASLt)"
        )
        return SlideSparseFp8LinearMethod(original_scheme)
    else:
        logger.warning_once(
            f"cuBLASLt wrapper not supported for {scheme_name}, "
            "using original scheme"
        )
        return original_scheme


def wrap_scheme_with_cusparselt(original_scheme):
    """
    工厂函数：将原始 scheme 包装为 cuSPARSELt 版本
    
    使用示例（在 compressed_tensors.py 的 get_scheme() 中）:
        if is_cusparselt_enabled():
            scheme = wrap_scheme_with_cusparselt(scheme)
        return scheme
    
    Args:
        original_scheme: 原始的 CompressedTensorsScheme
        
    Returns:
        如果 scheme 是 W8A8Fp8，返回包装后的 scheme（内部会检测 kernel）
        否则返回原始 scheme
    """
    scheme_name = type(original_scheme).__name__
    if "W8A8Fp8" in scheme_name:
        logger.info_once(
            f"Wrapping {scheme_name} with SlideSparseFp8LinearMethod (cuSPARSELt)"
        )
        # 使用统一的 SlideSparseFp8LinearMethod，内部会根据环境变量选择 kernel
        return SlideSparseFp8LinearMethod(original_scheme)
    else:
        logger.warning_once(
            f"cuSPARSELt wrapper not supported for {scheme_name}, "
            "using original scheme"
        )
        return original_scheme


def wrap_scheme_fp8(original_scheme):
    """
    统一的 FP8 scheme 包装入口
    
    根据环境变量自动选择 cuBLASLt 或 cuSPARSELt 包装器。
    
    Args:
        original_scheme: 原始的 CompressedTensorsScheme
        
    Returns:
        包装后的 scheme 或原始 scheme
    """
    if is_cublaslt_enabled():
        return wrap_scheme_with_cublaslt(original_scheme)
    elif is_cusparselt_enabled():
        return wrap_scheme_with_cusparselt(original_scheme)
    else:
        # 默认 CUTLASS fallback，不做包装
        return original_scheme
