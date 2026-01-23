# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse GEMM Wrapper 模块

包含:
- cuBLASLt GEMM ctypes 包装器（FP8 + INT8）
- cuSPARSELt GEMM ctypes 包装器（FP8 + INT8）
- torch.library Custom Op 注册（让 torch.compile 能追踪）
- Custom Op 调用包装函数

架构说明:
=========
ctypes 包装器直接调用 CUDA 扩展库（.so 文件），
但 torch.compile 无法追踪 ctypes 调用。

通过 torch.library 注册 custom op，提供：
- 实际实现：调用 ctypes 包装器
- fake 实现：返回正确形状的空 tensor，用于追踪

torch.compile 追踪时使用 fake 实现，执行时使用实际实现。
"""

import ctypes
from pathlib import Path

import torch
from torch.library import Library

from vllm.logger import init_logger

from slidesparse.utils import (
    find_file,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
)
from .kernels import _build_search_kernel, _CSRC_DIR

logger = init_logger(__name__)


# ============================================================================
# Extension 缓存
# ============================================================================

_gemm_extensions = {}


# ============================================================================
# cuBLASLt GEMM Wrapper
# ============================================================================

class cuBLASLtGemmWrapper:
    """cuBLASLt GEMM ctypes 包装器（支持 FP8 和 INT8）"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cublaslt_gemm_get_last_error.argtypes = []
        self._lib.cublaslt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # FP8 GEMM 签名: int fn(W, A, D, M, N, K, inner_dtype, stream)
        gemm_sig = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_char_p, ctypes.c_void_p]
        self._lib.cublaslt_fp8_mm.argtypes = gemm_sig
        self._lib.cublaslt_fp8_mm.restype = ctypes.c_int
        
        # INT8 GEMM 签名: int fn(W, A, D, M, N, K, inner_dtype, stream)
        # 注意: INT8 GEMM 忽略 inner_dtype 参数，输出固定为 INT32
        self._lib.cublaslt_int8_mm.argtypes = gemm_sig
        self._lib.cublaslt_int8_mm.restype = ctypes.c_int
        
    def cublaslt_fp8_mm(
        self,
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """
        cuBLASLt FP8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] FP8，权重（行主序，未转置）
            qinput: [M_pad, K_pad] FP8，量化后的激活
            inner_dtype: GEMM 输出精度 ("bf16" 或 "fp32")
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        ret = self._lib.cublaslt_fp8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_pad,
            inner_dtype.encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cublaslt_gemm_get_last_error()
            raise RuntimeError(f"cublaslt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
        return output
    
    def cublaslt_int8_mm(
        self,
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """
        cuBLASLt INT8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] INT8，权重（行主序，未转置）
            qinput: [M_pad, K_pad] INT8，量化后的激活
            inner_dtype: 被忽略，输出固定为 INT32
            
        Returns:
            output: [M_pad, N] INT32
            
        Note:
            cuBLASLt INT8 GEMM 只支持 INT32 输出，inner_dtype 参数被忽略。
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        # INT8 GEMM 输出固定为 INT32
        output = torch.empty((M_pad, N), dtype=torch.int32, device=qinput.device)
        
        ret = self._lib.cublaslt_int8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_pad,
            "int32".encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cublaslt_gemm_get_last_error()
            raise RuntimeError(f"cublaslt_int8_mm failed: {err.decode() if err else 'Unknown'}")
        return output


# ============================================================================
# cuSPARSELt GEMM Wrapper
# ============================================================================

class cuSPARSELtGemmWrapper:
    """cuSPARSELt 2:4 Sparse GEMM ctypes 包装器（支持 FP8 和 INT8）"""
    
    def __init__(self, lib_path: Path):
        self._lib = ctypes.CDLL(str(lib_path))
        
        # 错误处理函数
        self._lib.cusparselt_gemm_get_last_error.argtypes = []
        self._lib.cusparselt_gemm_get_last_error.restype = ctypes.c_char_p
        
        # FP8 GEMM 签名: int fn(W_compressed, A, D, M, N, K, inner_dtype, stream)
        gemm_sig = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 + [ctypes.c_char_p, ctypes.c_void_p]
        self._lib.cusparselt_fp8_mm.argtypes = gemm_sig
        self._lib.cusparselt_fp8_mm.restype = ctypes.c_int
        
        # INT8 GEMM 签名（同上）
        self._lib.cusparselt_int8_mm.argtypes = gemm_sig
        self._lib.cusparselt_int8_mm.restype = ctypes.c_int
        
    def cusparselt_fp8_mm(
        self,
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """
        cuSPARSELt 2:4 Sparse FP8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_slide_pad] @ weight_decompressed.T
        
        Args:
            weight_compressed: [compressed_size] uint8 1D，压缩后的权重
            qinput: [M_pad, K_slide_pad] FP8，量化+slide 后的激活
            N: 权重的 N 维度
            K_slide: 权重的 K_slide 维度（slide 扩展后）
            inner_dtype: GEMM 输出精度 ("bf16" 或 "fp32")
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad = qinput.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        ret = self._lib.cusparselt_fp8_mm(
            weight_compressed.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_slide,
            inner_dtype.encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cusparselt_gemm_get_last_error()
            raise RuntimeError(f"cusparselt_fp8_mm failed: {err.decode() if err else 'Unknown'}")
        return output
    
    def cusparselt_int8_mm(
        self,
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """
        cuSPARSELt 2:4 Sparse INT8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_slide_pad] @ weight_decompressed.T
        
        Args:
            weight_compressed: [compressed_size] uint8 1D，压缩后的权重
            qinput: [M_pad, K_slide_pad] INT8，量化+slide 后的激活
            N: 权重的 N 维度
            K_slide: 权重的 K_slide 维度（slide 扩展后）
            inner_dtype: GEMM 输出精度 ("bf16" 或 "int32"，不支持 "fp32"）
            
        Returns:
            output: [M_pad, N] BF16/INT32
            
        Note:
            cuSPARSELt INT8 GEMM 不支持 FP32 输出，只支持 BF16 或 INT32。
        """
        M_pad = qinput.shape[0]
        
        # INT8 不支持 FP32 输出
        if inner_dtype == "fp32":
            raise ValueError(
                "cuSPARSELt INT8 GEMM does not support FP32 output. "
                "Use 'bf16' (default) or 'int32'."
            )
        
        out_dtype = torch.int32 if inner_dtype == "int32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        ret = self._lib.cusparselt_int8_mm(
            weight_compressed.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_slide,
            inner_dtype.encode(), torch.cuda.current_stream().cuda_stream
        )
        if ret != 0:
            err = self._lib.cusparselt_gemm_get_last_error()
            raise RuntimeError(f"cusparselt_int8_mm failed: {err.decode() if err else 'Unknown'}")
        return output


# ============================================================================
# Extension 加载
# ============================================================================

def _get_gemm_extension(backend: str):
    """
    获取 GEMM extension（懒加载）
    
    加载 ctypes 包装的 CUDA 扩展（纯 C 库，通过 ctypes.CDLL 加载）
    
    注意：SlideSparseFp8LinearOp.__init__ 会预加载 extension，
    所以在 torch.compile 追踪时会直接命中缓存。
    """
    if backend in _gemm_extensions:
        return _gemm_extensions[backend]
    
    if backend == "cublaslt":
        # 预加载系统 cuBLASLt 库（RTLD_GLOBAL 模式）
        ensure_cublaslt_loaded()
        # 目录名是 cublaslt_gemm
        kernel_dir = _CSRC_DIR / "cublaslt_gemm"
        build_dir = kernel_dir / "build"
        so_prefix = "cublaslt_gemm"
        wrapper_class = cuBLASLtGemmWrapper
    elif backend == "cusparselt":
        # 预加载系统 cuSPARSELt 库（0.8.1+，RTLD_GLOBAL 模式）
        ensure_cusparselt_loaded()
        # 目录名是 cusparselt_gemm
        kernel_dir = _CSRC_DIR / "cusparselt_gemm"
        build_dir = kernel_dir / "build"
        so_prefix = "cusparselt_gemm"
        wrapper_class = cuSPARSELtGemmWrapper
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # 查找 .so 文件
    so_path = find_file(so_prefix, search_dir=build_dir, ext=".so")
    
    if so_path is None:
        # 找不到，尝试自动编译
        _build_search_kernel(kernel_dir, kernel_type=backend)
        so_path = find_file(so_prefix, search_dir=build_dir, ext=".so")
        if so_path is None:
            raise FileNotFoundError(
                f"{backend} GEMM extension not found after build.\n"
                f"Build may have failed. Please check the logs."
            )
    
    # 创建 ctypes 包装器（传递 .so 路径）
    wrapper = wrapper_class(so_path)
    _gemm_extensions[backend] = wrapper
    logger.info_once(f"{backend} GEMM extension loaded: {so_path.name}")
    return wrapper


# ============================================================================
# torch.library Custom Ops
# ============================================================================

# 创建 SlideSparse library
_slidesparse_lib = Library("slidesparse", "FRAGMENT")


# ----------------------------------------------------------------------------
# FP8 Custom Ops
# ----------------------------------------------------------------------------

def _register_cublaslt_fp8_custom_op():
    """注册 cuBLASLt FP8 GEMM custom op"""
    
    # 定义 op schema
    _slidesparse_lib.define(
        "cublaslt_fp8_mm(Tensor weight, Tensor qinput, str inner_dtype) -> Tensor"
    )
    
    # 实际实现
    def _cublaslt_fp8_mm_impl(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt FP8 GEMM 实际执行"""
        ext = _gemm_extensions.get("cublaslt")
        if ext is None:
            raise RuntimeError(
                "cuBLASLt extension not loaded. "
                "Ensure _get_gemm_extension('cublaslt') is called first."
            )
        return ext.cublaslt_fp8_mm(weight, qinput, inner_dtype)
    
    # fake 实现（用于 torch.compile 追踪）
    def _cublaslt_fp8_mm_fake(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt FP8 GEMM fake 实现 - 返回正确形状的空 tensor"""
        M_pad = qinput.shape[0]
        N = weight.shape[0]
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        return torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
    
    # 注册实现
    _slidesparse_lib.impl("cublaslt_fp8_mm", _cublaslt_fp8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cublaslt_fp8_mm", _cublaslt_fp8_mm_fake)
    
    logger.info_once("cuBLASLt FP8 custom op registered: slidesparse::cublaslt_fp8_mm")


def _register_cusparselt_fp8_custom_op():
    """注册 cuSPARSELt FP8 GEMM custom op"""
    
    # 定义 op schema
    _slidesparse_lib.define(
        "cusparselt_fp8_mm(Tensor weight_compressed, Tensor qinput, "
        "int N, int K_slide, str inner_dtype) -> Tensor"
    )
    
    # 实际实现
    def _cusparselt_fp8_mm_impl(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt FP8 GEMM 实际执行"""
        ext = _gemm_extensions.get("cusparselt")
        if ext is None:
            raise RuntimeError(
                "cuSPARSELt extension not loaded. "
                "Ensure _get_gemm_extension('cusparselt') is called first."
            )
        return ext.cusparselt_fp8_mm(weight_compressed, qinput, N, K_slide, inner_dtype)
    
    # fake 实现（用于 torch.compile 追踪）
    def _cusparselt_fp8_mm_fake(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt FP8 GEMM fake 实现 - 返回正确形状的空 tensor"""
        M_pad = qinput.shape[0]
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        return torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
    
    # 注册实现
    _slidesparse_lib.impl("cusparselt_fp8_mm", _cusparselt_fp8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cusparselt_fp8_mm", _cusparselt_fp8_mm_fake)
    
    logger.info_once("cuSPARSELt FP8 custom op registered: slidesparse::cusparselt_fp8_mm")


# ----------------------------------------------------------------------------
# INT8 Custom Ops
# ----------------------------------------------------------------------------

def _register_cublaslt_int8_custom_op():
    """注册 cuBLASLt INT8 GEMM custom op"""
    
    # 定义 op schema
    # 注意：保留 inner_dtype 参数以保持接口一致性，但实际被忽略
    _slidesparse_lib.define(
        "cublaslt_int8_mm(Tensor weight, Tensor qinput, str inner_dtype) -> Tensor"
    )
    
    # 实际实现
    def _cublaslt_int8_mm_impl(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt INT8 GEMM 实际执行（输出固定为 INT32）"""
        ext = _gemm_extensions.get("cublaslt")
        if ext is None:
            raise RuntimeError(
                "cuBLASLt extension not loaded. "
                "Ensure _get_gemm_extension('cublaslt') is called first."
            )
        return ext.cublaslt_int8_mm(weight, qinput, inner_dtype)
    
    # fake 实现（用于 torch.compile 追踪）
    def _cublaslt_int8_mm_fake(
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuBLASLt INT8 GEMM fake 实现 - 返回正确形状的 INT32 tensor"""
        M_pad = qinput.shape[0]
        N = weight.shape[0]
        # INT8 GEMM 输出固定为 INT32
        return torch.empty((M_pad, N), dtype=torch.int32, device=qinput.device)
    
    # 注册实现
    _slidesparse_lib.impl("cublaslt_int8_mm", _cublaslt_int8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cublaslt_int8_mm", _cublaslt_int8_mm_fake)
    
    logger.info_once("cuBLASLt INT8 custom op registered: slidesparse::cublaslt_int8_mm")


def _register_cusparselt_int8_custom_op():
    """注册 cuSPARSELt INT8 GEMM custom op"""
    
    # 定义 op schema
    _slidesparse_lib.define(
        "cusparselt_int8_mm(Tensor weight_compressed, Tensor qinput, "
        "int N, int K_slide, str inner_dtype) -> Tensor"
    )
    
    # 实际实现
    def _cusparselt_int8_mm_impl(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt INT8 GEMM 实际执行"""
        ext = _gemm_extensions.get("cusparselt")
        if ext is None:
            raise RuntimeError(
                "cuSPARSELt extension not loaded. "
                "Ensure _get_gemm_extension('cusparselt') is called first."
            )
        return ext.cusparselt_int8_mm(weight_compressed, qinput, N, K_slide, inner_dtype)
    
    # fake 实现（用于 torch.compile 追踪）
    def _cusparselt_int8_mm_fake(
        weight_compressed: torch.Tensor,
        qinput: torch.Tensor,
        N: int,
        K_slide: int,
        inner_dtype: str,
    ) -> torch.Tensor:
        """cuSPARSELt INT8 GEMM fake 实现 - 返回正确形状的空 tensor"""
        M_pad = qinput.shape[0]
        # INT8 不支持 FP32，只能是 BF16 或 INT32
        out_dtype = torch.int32 if inner_dtype == "int32" else torch.bfloat16
        return torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
    
    # 注册实现
    _slidesparse_lib.impl("cusparselt_int8_mm", _cusparselt_int8_mm_impl, "CUDA")
    _slidesparse_lib._register_fake("cusparselt_int8_mm", _cusparselt_int8_mm_fake)
    
    logger.info_once("cuSPARSELt INT8 custom op registered: slidesparse::cusparselt_int8_mm")


# 在模块加载时注册 custom ops
# 注意：这里只是注册 op schema 和实现，不加载 ctypes 库
# ctypes 库在 LinearOp.__init__ 中预加载
_register_cublaslt_fp8_custom_op()
_register_cusparselt_fp8_custom_op()
_register_cublaslt_int8_custom_op()
_register_cusparselt_int8_custom_op()


# ============================================================================
# Custom Op 调用包装函数
# ============================================================================

def cublaslt_fp8_mm_op(
    weight: torch.Tensor,
    qinput: torch.Tensor,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuBLASLt FP8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op，而非直接调用 ctypes。
    """
    return torch.ops.slidesparse.cublaslt_fp8_mm(weight, qinput, inner_dtype)


def cusparselt_fp8_mm_op(
    weight_compressed: torch.Tensor,
    qinput: torch.Tensor,
    N: int,
    K_slide: int,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuSPARSELt FP8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op，而非直接调用 ctypes。
    """
    return torch.ops.slidesparse.cusparselt_fp8_mm(
        weight_compressed, qinput, N, K_slide, inner_dtype
    )


def cublaslt_int8_mm_op(
    weight: torch.Tensor,
    qinput: torch.Tensor,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuBLASLt INT8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op。
    注意：输出固定为 INT32，inner_dtype 参数被忽略。
    """
    return torch.ops.slidesparse.cublaslt_int8_mm(weight, qinput, inner_dtype)


def cusparselt_int8_mm_op(
    weight_compressed: torch.Tensor,
    qinput: torch.Tensor,
    N: int,
    K_slide: int,
    inner_dtype: str,
) -> torch.Tensor:
    """
    cuSPARSELt INT8 GEMM - torch.compile 兼容版本
    
    通过 torch.ops 调用注册的 custom op。
    注意：不支持 FP32 输出，只能使用 "bf16" 或 "int32"。
    """
    return torch.ops.slidesparse.cusparselt_int8_mm(
        weight_compressed, qinput, N, K_slide, inner_dtype
    )


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # Wrapper 类
    "cuBLASLtGemmWrapper",
    "cuSPARSELtGemmWrapper",
    
    # Extension 加载
    "_get_gemm_extension",
    
    # FP8 Custom Op 包装函数
    "cublaslt_fp8_mm_op",
    "cusparselt_fp8_mm_op",
    
    # INT8 Custom Op 包装函数
    "cublaslt_int8_mm_op",
    "cusparselt_int8_mm_op",
]
