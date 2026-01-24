# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse GEMM Wrapper 模块

包含:
- cuBLASLt GEMM ctypes 包装器（FP8 + INT8）
- cuSPARSELt GEMM ctypes 包装器（FP8 + INT8）
- AlgorithmConfigManager: GEMM 算法配置管理器（离线调优结果查表）
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

算法配置查表:
============
AlgorithmConfigManager 在启动时加载离线调优的 JSON 配置，
运行时根据 (model, N, K, M) 快速查找最优算法配置，
透明传递给 CUDA Kernel 执行。找不到配置时优雅 Fallback 到默认算法。
"""

import base64
import bisect
import ctypes
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.library import Library

from vllm.logger import init_logger

from slidesparse.utils import (
    find_file,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
    build_hw_dir_name,
)
from .kernels import _CSRC_DIR

logger = init_logger(__name__)


# ============================================================================
# 搜索结果目录
# ============================================================================

_SEARCH_DIR = Path(__file__).parent.parent / "search"


# ============================================================================
# GEMM 库编译配置
# ============================================================================

_GEMM_BUILD_CONFIG = {
    "cublaslt":   ("build_cublaslt.py",   ["build"]),
    "cusparselt": ("build_cusparselt.py", ["build"]),
}


def _build_gemm_library(kernel_dir: Path, backend: str) -> None:
    """在找不到 GEMM .so 库时，自动编译"""
    import subprocess
    
    if backend not in _GEMM_BUILD_CONFIG:
        raise ValueError(f"Unknown GEMM backend: {backend}")
    
    script_name, extra_args = _GEMM_BUILD_CONFIG[backend]
    script_path = kernel_dir / script_name
    
    if not script_path.exists():
        raise FileNotFoundError(f"Build script not found: {script_path}")
    
    logger.info(f"Auto-building {backend} GEMM library from {script_path}...")
    
    try:
        subprocess.run(
            ["python3", str(script_path)] + extra_args,
            cwd=kernel_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        logger.info(f"{backend} GEMM library build completed")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to build {backend} GEMM library:\n"
            f"stdout: {e.stdout}\n"
            f"stderr: {e.stderr}"
        ) from e


# ============================================================================
# Extension 缓存
# ============================================================================

_gemm_extensions = {}


# ============================================================================
# AlgorithmConfigManager: GEMM 算法配置管理器
# ============================================================================

class AlgorithmConfigManager:
    """
    GEMM 算法配置管理器（单例）
    
    设计要点:
    1. 启动时一次性加载当前硬件文件夹下所有模型 JSON（KB 级别）
    2. 使用 bisect 二分查找 M 阈值，O(log n)
    3. torch.compile 兼容：查询发生在 ctypes 调用前
    """
    
    _instance: Optional["AlgorithmConfigManager"] = None
    
    @classmethod
    def get_instance(cls) -> "AlgorithmConfigManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # {model_name: {(N,K): {"m_thresholds": [...], "alg_by_m": {...}}}}
        self._cublaslt_configs: Dict[str, Dict] = {}
        self._cusparselt_configs: Dict[str, Dict] = {}
        self._current_model: Optional[str] = None      # 原始模型名（checkpoint 名）
        self._base_model: Optional[str] = None         # 基础模型名（用于查表和 kernel 加载）
        self._loaded = False
        self._cublaslt_warned_models: set = set()  # 避免 cuBLASLt 重复警告
        self._cusparselt_warned_models: set = set()  # 避免 cuSPARSELt 重复警告
    
    @staticmethod
    def _extract_base_model_name(model_name: str) -> str:
        """
        从完整模型名中提取基础模型名
        
        例如:
            Llama3.2-1B-FP8-SlideSparse-2_8 -> Llama3.2-1B-FP8
            Qwen2.5-0.5B-INT8-SlideSparse-2_10 -> Qwen2.5-0.5B-INT8
            Llama3.2-1B-FP8 -> Llama3.2-1B-FP8 (不变)
        """
        marker = "-SlideSparse-"
        if marker in model_name:
            return model_name.split(marker)[0]
        return model_name
    
    def load_all_configs(self) -> None:
        """加载当前硬件目录下所有 JSON 配置"""
        if self._loaded:
            return
        
        hw_folder = build_hw_dir_name()  # e.g., RTX5080_cc120_py312_cu129_x86_64
        
        # cuBLASLt 配置
        cublaslt_dir = _SEARCH_DIR / "cuBLASLt_AlgSearch" / "alg_search_results" / hw_folder
        if cublaslt_dir.exists():
            for json_file in cublaslt_dir.glob("alg_search_*.json"):
                self._load_single_config("cublaslt", json_file)
        
        # cuSPARSELt 配置
        cusparselt_dir = _SEARCH_DIR / "cuSPARSELt_AlgSearch" / "alg_search_results" / hw_folder
        if cusparselt_dir.exists():
            for json_file in cusparselt_dir.glob("alg_search_*.json"):
                self._load_single_config("cusparselt", json_file)
        
        self._loaded = True
        logger.info(f"Loaded algorithm configs: cuBLASLt={len(self._cublaslt_configs)}, "
                   f"cuSPARSELt={len(self._cusparselt_configs)} models")
    
    def _load_single_config(self, backend: str, json_path: Path) -> None:
        """加载单个 JSON 配置文件"""
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            model_name = data["meta"]["model_name"]
            nk_entries = data.get("nk_entries", {})
            
            # 转换为内部格式 {(N,K): entry}
            parsed = {}
            for nk_str, entry in nk_entries.items():
                # "(3072,2048)" -> (3072, 2048)
                n, k = eval(nk_str)
                parsed[(n, k)] = entry
            
            if backend == "cublaslt":
                self._cublaslt_configs[model_name] = parsed
            else:
                self._cusparselt_configs[model_name] = parsed
                
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
    
    def set_model(self, model_name: str) -> None:
        """设置当前模型（由 LinearMethod 调用）
        
        自动提取基础模型名用于查表和 kernel 加载。
        同时设置环境变量，以便子进程（如 vLLM EngineCore）也能获取到。
        
        Args:
            model_name: 完整模型名（可能包含 -SlideSparse-2_L 后缀）
        """
        self._current_model = model_name
        self._base_model = self._extract_base_model_name(model_name)
        os.environ["SLIDESPARSE_MODEL_NAME"] = model_name
        os.environ["SLIDESPARSE_BASE_MODEL_NAME"] = self._base_model
        
        if self._base_model != model_name:
            logger.info(f"Model name mapping: {model_name} -> {self._base_model}")
    
    def get_model(self) -> Optional[str]:
        """获取当前完整模型名（checkpoint 名）
        
        优先返回实例变量，其次从环境变量读取（支持子进程场景）。
        """
        if self._current_model is not None:
            return self._current_model
        # 子进程场景：从环境变量读取
        env_model = os.environ.get("SLIDESPARSE_MODEL_NAME")
        if env_model:
            self._current_model = env_model
            self._base_model = self._extract_base_model_name(env_model)
            return env_model
        return None
    
    def get_base_model(self) -> Optional[str]:
        """获取基础模型名（用于查表和 kernel 加载）
        
        基础模型名去除了 -SlideSparse-2_L 后缀。
        优先返回实例变量，其次从环境变量读取。
        """
        if self._base_model is not None:
            return self._base_model
        # 子进程场景：从环境变量读取
        env_base = os.environ.get("SLIDESPARSE_BASE_MODEL_NAME")
        if env_base:
            self._base_model = env_base
            return env_base
        # 如果没有 base，尝试从完整名称提取
        full_model = self.get_model()
        if full_model:
            self._base_model = self._extract_base_model_name(full_model)
            return self._base_model
        return None
    
    def lookup_cublaslt(self, N: int, K: int, M: int) -> Tuple[Optional[bytes], int]:
        """
        查找 cuBLASLt 最优配置
        
        策略：使用 M_upper（大于等于 M 的最小阈值）的配置
        
        原因：cublasLtMatmulAlgo_t 中的 split_k 和 workspace 与 M 相关。
        ws>0 的配置只能执行 M_exec <= M_from，因此必须用 M_upper 配置。
        
        区间划分（左开右闭）：
          (0, M_list[0]]        -> M_list[0] 配置
          (M_list[0], M_list[1]]-> M_list[1] 配置
          ...
          > M_list[-1]          -> fallback 到默认启发式
        
        Args:
            N: 权重的 N 维度
            K: 内维度
            M: 激活的 M 维度
        
        Returns:
            (algo_data_bytes, workspace)
            找不到返回 (None, 0)
        """
        base_model = self.get_base_model()
        if base_model is None:
            return (None, 0)
        
        model_config = self._cublaslt_configs.get(base_model)
        if model_config is None:
            if base_model not in self._cublaslt_warned_models:
                logger.warning(f"No cuBLASLt config for model '{base_model}', "
                             f"using default algorithm")
                self._cublaslt_warned_models.add(base_model)
            return (None, 0)
        
        nk_entry = model_config.get((N, K))
        if nk_entry is None:
            return (None, 0)
        
        # 使用 M_upper 策略：找 M_upper = min{M_i ∈ thresholds | M_i >= M}
        thresholds = nk_entry["m_thresholds"]
        idx = bisect.bisect_left(thresholds, M)
        
        # 如果 M 超过最大阈值，fallback 到默认启发式
        if idx >= len(thresholds):
            logger.warning(
                f"cuBLASLt: M={M} exceeds max searched M={thresholds[-1]} for "
                f"(N={N}, K={K}), falling back to default heuristic"
            )
            return (None, 0)
        
        m_key = str(thresholds[idx])
        algo_config = nk_entry["alg_by_m"].get(m_key)
        if algo_config is None:
            return (None, 0)
        
        # 解码 base64
        algo_data = base64.b64decode(algo_config["algo_data"])
        workspace = algo_config.get("workspace", 0)
        
        return (algo_data, workspace)
    
    def lookup_cusparselt(self, N: int, K: int, M: int) -> Tuple[int, int, int]:
        """
        查找 cuSPARSELt 最优配置
        
        使用 M_upper 策略：找到 M_upper = min{M_i | M_i >= M}，使用该配置。
        这确保 split_k > 1 的配置能够正确执行（workspace 足够）。
        
        Args:
            N: 权重的 N 维度
            K: 内维度（slide 后的 K'）
            M: 激活的 M 维度
        
        Returns:
            (alg_id, split_k, workspace)
            找不到返回 (-1, -1, 0)
        """
        base_model = self.get_base_model()
        if base_model is None:
            return (-1, -1, 0)
        
        model_config = self._cusparselt_configs.get(base_model)
        if model_config is None:
            if base_model not in self._cusparselt_warned_models:
                logger.warning(f"No cuSPARSELt config for model '{base_model}', "
                             f"using default algorithm")
                self._cusparselt_warned_models.add(base_model)
            return (-1, -1, 0)
        
        nk_entry = model_config.get((N, K))
        if nk_entry is None:
            return (-1, -1, 0)
        
        thresholds = nk_entry["m_thresholds"]
        
        # M_upper 策略：使用 bisect_left 找到第一个 >= M 的位置
        # 这样保证 workspace 足够（split_k > 1 时 ws ≈ M * N * split_k * dtype_size）
        idx = bisect.bisect_left(thresholds, M)
        if idx >= len(thresholds):
            # M 超过了所有离线搜索的 M 值，使用最大的配置
            # 注意：如果最大配置使用 split_k > 1，可能无法执行
            logger.warning(f"cuSPARSELt: M={M} exceeds max offline M={thresholds[-1]}, "
                         f"using largest config (may fail if split_k > 1)")
            idx = len(thresholds) - 1
        
        m_key = str(thresholds[idx])
        algo_config = nk_entry["alg_by_m"].get(m_key)
        if algo_config is None:
            return (-1, -1, 0)
        
        return (
            algo_config.get("alg_id", -1),
            algo_config.get("split_k", -1),
            algo_config.get("workspace", 0),
        )


# 全局单例
_algo_config_manager: Optional[AlgorithmConfigManager] = None


def get_algo_config_manager() -> AlgorithmConfigManager:
    """获取算法配置管理器（懒加载）"""
    global _algo_config_manager
    if _algo_config_manager is None:
        _algo_config_manager = AlgorithmConfigManager.get_instance()
        _algo_config_manager.load_all_configs()
    return _algo_config_manager


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
        
        # FP8 GEMM 签名: int fn(W, A, D, M, N, K, inner_dtype, algo_data, algo_workspace, stream)
        gemm_sig = ([ctypes.c_void_p] * 3 + 
                   [ctypes.c_int64] * 3 + 
                   [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p])
        self._lib.cublaslt_fp8_mm.argtypes = gemm_sig
        self._lib.cublaslt_fp8_mm.restype = ctypes.c_int
        
        # INT8 GEMM 签名（同上）
        self._lib.cublaslt_int8_mm.argtypes = gemm_sig
        self._lib.cublaslt_int8_mm.restype = ctypes.c_int
        
    def cublaslt_fp8_mm(
        self,
        weight: torch.Tensor,
        qinput: torch.Tensor,
        inner_dtype: str,
        algo_data: Optional[bytes] = None,
        algo_workspace: int = 0,
    ) -> torch.Tensor:
        """
        cuBLASLt FP8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] FP8，权重（行主序，未转置）
            qinput: [M_pad, K_pad] FP8，量化后的激活
            inner_dtype: GEMM 输出精度 ("bf16" 或 "fp32")
            algo_data: 算法配置数据（64字节），None表示使用默认启发式
            algo_workspace: 算法配置中指定的 workspace 大小
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        # 处理 algo_data
        algo_ptr = None
        if algo_data is not None and len(algo_data) == 64:
            algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data)
        
        ret = self._lib.cublaslt_fp8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_pad,
            inner_dtype.encode(),
            ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
            algo_workspace,
            torch.cuda.current_stream().cuda_stream
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
        algo_data: Optional[bytes] = None,
        algo_workspace: int = 0,
    ) -> torch.Tensor:
        """
        cuBLASLt INT8 GEMM
        
        计算: output[M_pad, N] = qinput[M_pad, K_pad] @ weight[N, K_pad].T
        
        Args:
            weight: [N, K_pad] INT8，权重（行主序，未转置）
            qinput: [M_pad, K_pad] INT8，量化后的激活
            inner_dtype: 被忽略，输出固定为 INT32
            algo_data: 算法配置数据（64字节），None表示使用默认启发式
            algo_workspace: 算法配置中指定的 workspace 大小
            
        Returns:
            output: [M_pad, N] INT32
            
        Note:
            cuBLASLt INT8 GEMM 只支持 INT32 输出，inner_dtype 参数被忽略。
        """
        M_pad, K_pad = qinput.shape
        N = weight.shape[0]
        
        # INT8 GEMM 输出固定为 INT32
        output = torch.empty((M_pad, N), dtype=torch.int32, device=qinput.device)
        
        # 处理 algo_data
        algo_ptr = None
        if algo_data is not None and len(algo_data) == 64:
            algo_ptr = (ctypes.c_uint8 * 64).from_buffer_copy(algo_data)
        
        ret = self._lib.cublaslt_int8_mm(
            weight.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_pad,
            "int32".encode(),
            ctypes.cast(algo_ptr, ctypes.c_void_p) if algo_ptr else None,
            algo_workspace,
            torch.cuda.current_stream().cuda_stream
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
        
        # FP8 GEMM 签名: int fn(W_compressed, A, D, M, N, K, inner_dtype, alg_id, split_k, algo_workspace, stream)
        gemm_sig = ([ctypes.c_void_p] * 3 + 
                   [ctypes.c_int64] * 3 + 
                   [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p])
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
        alg_id: int = -1,
        split_k: int = -1,
        algo_workspace: int = 0,
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
            alg_id: 算法 ID，-1 表示使用默认算法
            split_k: split_k 设置，-1 表示不设置
            algo_workspace: 算法配置中指定的 workspace 大小
            
        Returns:
            output: [M_pad, N] BF16/FP32
        """
        M_pad = qinput.shape[0]
        
        out_dtype = torch.float32 if inner_dtype == "fp32" else torch.bfloat16
        output = torch.empty((M_pad, N), dtype=out_dtype, device=qinput.device)
        
        ret = self._lib.cusparselt_fp8_mm(
            weight_compressed.data_ptr(), qinput.data_ptr(), output.data_ptr(),
            M_pad, N, K_slide,
            inner_dtype.encode(),
            alg_id, split_k, algo_workspace,
            torch.cuda.current_stream().cuda_stream
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
        alg_id: int = -1,
        split_k: int = -1,
        algo_workspace: int = 0,
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
            alg_id: 算法 ID，-1 表示使用默认算法
            split_k: split_k 设置，-1 表示不设置
            algo_workspace: 算法配置中指定的 workspace 大小
            
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
            inner_dtype.encode(),
            alg_id, split_k, algo_workspace,
            torch.cuda.current_stream().cuda_stream
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
        _build_gemm_library(kernel_dir, backend=backend)
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
        """cuBLASLt FP8 GEMM 实际执行（带算法配置查表）"""
        ext = _gemm_extensions.get("cublaslt")
        if ext is None:
            raise RuntimeError(
                "cuBLASLt extension not loaded. "
                "Ensure _get_gemm_extension('cublaslt') is called first."
            )
        
        # 查表获取最优算法配置
        M, K = qinput.shape
        N = weight.shape[0]
        mgr = get_algo_config_manager()
        algo_data, algo_workspace = mgr.lookup_cublaslt(N, K, M)
        
        return ext.cublaslt_fp8_mm(weight, qinput, inner_dtype, algo_data, algo_workspace)
    
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
        """cuSPARSELt FP8 GEMM 实际执行（带算法配置查表）"""
        ext = _gemm_extensions.get("cusparselt")
        if ext is None:
            raise RuntimeError(
                "cuSPARSELt extension not loaded. "
                "Ensure _get_gemm_extension('cusparselt') is called first."
            )
        
        # 查表获取最优算法配置
        M = qinput.shape[0]
        mgr = get_algo_config_manager()
        alg_id, split_k, algo_workspace = mgr.lookup_cusparselt(N, K_slide, M)
        
        return ext.cusparselt_fp8_mm(weight_compressed, qinput, N, K_slide, inner_dtype,
                                     alg_id, split_k, algo_workspace)
    
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
        """cuBLASLt INT8 GEMM 实际执行（输出固定为 INT32，带算法配置查表）"""
        ext = _gemm_extensions.get("cublaslt")
        if ext is None:
            raise RuntimeError(
                "cuBLASLt extension not loaded. "
                "Ensure _get_gemm_extension('cublaslt') is called first."
            )
        
        # 查表获取最优算法配置
        M, K = qinput.shape
        N = weight.shape[0]
        mgr = get_algo_config_manager()
        algo_data, algo_workspace = mgr.lookup_cublaslt(N, K, M)
        
        return ext.cublaslt_int8_mm(weight, qinput, inner_dtype, algo_data, algo_workspace)
    
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
        """cuSPARSELt INT8 GEMM 实际执行（带算法配置查表）"""
        ext = _gemm_extensions.get("cusparselt")
        if ext is None:
            raise RuntimeError(
                "cuSPARSELt extension not loaded. "
                "Ensure _get_gemm_extension('cusparselt') is called first."
            )
        
        # 查表获取最优算法配置
        M = qinput.shape[0]
        mgr = get_algo_config_manager()
        alg_id, split_k, algo_workspace = mgr.lookup_cusparselt(N, K_slide, M)
        
        return ext.cusparselt_int8_mm(weight_compressed, qinput, N, K_slide, inner_dtype,
                                      alg_id, split_k, algo_workspace)
    
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
    # 算法配置管理器
    "AlgorithmConfigManager",
    "get_algo_config_manager",
    
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
