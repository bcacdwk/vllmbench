#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 算法/布局搜索共享工具库

提供搜索脚本所需的通用功能：
- 模型 NK 尺寸获取
- 默认参数配置
- 数据类型验证
- 搜索元数据构建
- 结果文件命名与保存

该工具库依赖顶层 slidesparse.utils，提供搜索场景的封装。
"""

import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

# 导入顶层 utils
SEARCH_DIR = Path(__file__).parent.absolute()
SLIDESPARSE_DIR = SEARCH_DIR.parent
PROJECT_ROOT = SLIDESPARSE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slidesparse.utils import (
    # 硬件信息
    hw_info,
    HardwareInfo,
    normalize_dtype,
    # 编译与加载
    load_cuda_extension,
    ensure_cublaslt_loaded,
    ensure_cusparselt_loaded,
    BACKEND_LDFLAGS,
    SUPPORTED_BACKENDS,
    # 文件命名
    build_filename,
    build_stem,
    # 模型信息
    get_model_nk_sizes,
    get_model_nk_sizes_slided,
    get_model_nk_sizes_compressed,
)


# =============================================================================
# 支持的数据类型
# =============================================================================

SUPPORTED_DTYPES = ["int8", "fp8e4m3"]
SUPPORTED_OUTDTYPES = ["bf16", "fp32"]


# =============================================================================
# 默认配置
# =============================================================================

def default_m_list() -> List[int]:
    """
    返回默认的 M 值列表。
    
    覆盖从 decode (小 batch) 到 prefill (大 batch) 的典型场景。
    """
    return [16, 128, 512, 2048, 16384]


def default_nk_list_bitnet_2b() -> List[Tuple[int, int]]:
    """
    BitNet-2B4T 模型的默认 NK 尺寸。
    
    基于标准 vLLM 格式的线性层尺寸。
    """
    return [
        (3840, 2560),   # qkv_proj (Wqkv)
        (2560, 2560),   # o_proj (Wo)
        (13824, 2560),  # gate_up_proj (W13)
        (2560, 6912),   # down_proj (W2)
    ]


# =============================================================================
# 模型 NK 尺寸工具
# =============================================================================

def get_nk_list_from_model(
    model_path: Union[str, Path],
    *,
    layer_index: int = 0,
    with_names: bool = False,
) -> Union[List[Tuple[int, int]], List[Tuple[int, int, str]]]:
    """
    从模型路径提取 NK 尺寸列表。
    
    Args:
        model_path: 模型目录或 safetensor 文件路径
        layer_index: 使用哪一层的尺寸（默认 0）
        with_names: 是否返回层名称
    
    Returns:
        List of (N, K) 或 (N, K, name) tuples
    """
    sizes = get_model_nk_sizes(model_path, layer_index=layer_index)
    
    # 转换为列表格式，按标准顺序
    order = ["qkv", "wo", "w13", "w2"]
    name_map = {
        "qkv": "Wqkv",
        "wo": "Wo",
        "w13": "W13",
        "w2": "W2",
    }
    
    result = []
    for key in order:
        if key in sizes:
            n, k = sizes[key]
            if with_names:
                result.append((n, k, name_map.get(key, key)))
            else:
                result.append((n, k))
    
    return result


def get_nk_list_auto(
    model: Optional[str] = None,
    *,
    with_names: bool = False,
) -> Union[List[Tuple[int, int]], List[Tuple[int, int, str]]]:
    """
    自动获取 NK 尺寸列表。
    
    如果提供了 model 参数且找到对应模型目录，则从模型提取；
    否则使用 BitNet-2B4T 默认值。
    
    Args:
        model: 模型名称（如 "BitNet-2B4T"）或路径
        with_names: 是否返回层名称
    
    Returns:
        List of (N, K) 或 (N, K, name) tuples
    """
    # 尝试查找模型目录
    if model:
        # 检查常见位置
        search_paths = [
            PROJECT_ROOT / "checkpoints" / model,
            PROJECT_ROOT / "checkpoints_slidesparse" / model,
            Path(model),
        ]
        
        for path in search_paths:
            if path.exists() and path.is_dir():
                try:
                    return get_nk_list_from_model(path, with_names=with_names)
                except Exception:
                    pass
    
    # 使用默认值
    default = default_nk_list_bitnet_2b()
    if with_names:
        names = ["Wqkv", "Wo", "W13", "W2"]
        return [(n, k, name) for (n, k), name in zip(default, names)]
    return default


# =============================================================================
# 模型名称处理
# =============================================================================

def build_model_name_with_dtype(base_name: str, dtype: str) -> str:
    """
    构建带 dtype 后缀的模型名称。
    
    Args:
        base_name: 基础模型名称（如 "BitNet-2B4T"）
        dtype: 数据类型（如 "int8", "fp8e4m3"）
    
    Returns:
        带后缀的模型名称（如 "BitNet-2B4T-INT8"）
    """
    dtype_suffix = normalize_dtype(dtype)
    return f"{base_name}-{dtype_suffix}"


# =============================================================================
# 输出目录与文件命名
# =============================================================================

def build_output_dir_name(
    model_name: str,
    dtype: str,
    outdtype: str,
) -> str:
    """
    构建输出目录名称。
    
    格式: {GPU}_{CC}_out-{outdtype}_{model_name}_{PyVer}_{CUDAVer}_{arch}
    示例: H100_cc90_out-BF16_BitNet-2B4T-FP8E4M3_py312_cu129_x86_64
    
    Args:
        model_name: 模型名称（已包含 dtype 后缀）
        dtype: 输入数据类型（保留用于兼容性）
        outdtype: 输出数据类型
    
    Returns:
        目录名称
    """
    outdtype_norm = normalize_dtype(outdtype)
    return (
        f"{hw_info.gpu_name}_{hw_info.cc_tag}_out-{outdtype_norm}_{model_name}"
        f"_{hw_info.python_tag}_{hw_info.cuda_tag}_{hw_info.arch_tag}"
    )


def build_result_filename(
    prefix: str,
    model_name: str,
    ext: str = "",
) -> str:
    """
    构建结果文件名称。
    
    格式: {prefix}_{model_name}.{ext}
    示例: alg_search_LUT_BitNet-2B4T.json
    
    Args:
        prefix: 文件前缀（如 "alg_search_LUT", "layout_search_bench"）
        model_name: 模型名称
        ext: 文件扩展名
    
    Returns:
        文件名称
    """
    name = f"{prefix}_{model_name}"
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        name += ext
    return name


# =============================================================================
# 搜索元数据构建
# =============================================================================

def build_search_meta(
    *,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List[Tuple[int, int]],
    model_name: str,
    layout: str = "TNCCcol",
    alg_count: int = 0,
    config_count: int = 0,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建搜索结果的元数据。
    
    Args:
        dtype: 输入数据类型
        outdtype: 输出数据类型
        warmup: 预热次数
        repeat: 重复次数
        verify: 是否验证
        m_list: M 列表
        nk_list: NK 列表
        model_name: 模型名称
        layout: 布局类型
        alg_count: 算法数量
        config_count: 配置数量
        extra: 额外元数据
    
    Returns:
        元数据字典
    """
    import torch
    
    meta = {
        "gpu_name": hw_info.gpu_full_name,
        "gpu_short_name": hw_info.gpu_name,
        "compute_capability": hw_info.cc_tag,
        "arch_name": hw_info.arch_name,
        "model_name": model_name,
        "layout": layout,
        "dtype": dtype,
        "outdtype": outdtype,
        "alg_count": alg_count,
        "config_count": config_count,
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "torch_version": torch.__version__,
        "cuda_version_driver": hw_info.cuda_driver_version,
        "cuda_version_runtime": hw_info.cuda_runtime_version,
        "time": datetime.datetime.now().isoformat(),
        "M_list": m_list,
        "NK_list": [(n, k) if len(t) == 2 else t for t in (nk_list if nk_list else [])],
    }
    
    if extra:
        meta.update(extra)
    
    return meta


# =============================================================================
# Segment-K 支持检测
# =============================================================================

def supports_segment_k() -> Tuple[bool, str]:
    """
    检测当前 GPU 是否支持 Segment-K (split_k=-1)。
    
    Segment-K 仅在 SM 9.0 (Hopper) 和 SM 10.x (Blackwell) 上支持。
    在 cuSPARSELt 0.8.1 及更早版本中，对不支持的架构调用 Segment-K
    会导致 planInit 阻塞挂起。
    
    Returns:
        (supported, reason_if_not)
    """
    major = hw_info.cc_major
    
    if major in (9, 10):
        return True, ""
    else:
        return False, (
            f"Segment-K (split_k=-1) 仅在 SM 9.0/10.x (Hopper/Blackwell) 上支持，"
            f"当前架构 SM {hw_info.cc_major}.{hw_info.cc_minor} 不支持。"
        )


# =============================================================================
# dtype 兼容性探测与验证
# =============================================================================

def probe_cublaslt_search(
    ext,
    dtype: str,
    outdtype: str = "bf16",
    layout: str = "TNCCcol",
) -> Tuple[bool, str]:
    """
    探测 cuBLASLt 算法搜索扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        out = ext.search_topk(
            W, A,
            [M],
            layout,
            dtype,
            outdtype,
            1, 1,  # warmup, repeat
            False, # verify
            [],    # blacklist
            1,     # topk
        )
        
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


def probe_cublaslt_layout(
    ext,
    dtype: str,
    outdtype: str = "bf16",
) -> Tuple[bool, str]:
    """
    探测 cuBLASLt layout 搜索扩展对 dtype 的支持情况。
    """
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",
            dtype,
            outdtype,
            1, 1,
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"


def probe_cusparselt_search(
    ext,
    dtype: str,
    outdtype: str = "bf16",
    layout: str = "TNCCcol",
) -> Tuple[bool, str]:
    """
    探测 cuSPARSELt 算法搜索扩展对 dtype 的支持情况。
    """
    import torch
    N, K, M = 32, 32, 16
    
    try:
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        W_pruned = ext.prune_24(W, layout)
        
        out = ext.search_topk(
            W_pruned, A,
            [M],
            layout,
            dtype,
            outdtype,
            1, 1,
            False,
            [],
            1,
            False,  # test_segment_k
        )
        
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


def probe_cusparselt_layout(
    ext,
    dtype: str,
    outdtype: str = "bf16",
) -> Tuple[bool, str]:
    """
    探测 cuSPARSELt layout 搜索扩展对 dtype 的支持情况。
    """
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",
            dtype,
            outdtype,
            1, 1,
            False,  # test_segment_k
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"


def check_dtype_support(
    ext,
    dtype: str,
    outdtype: str,
    *,
    backend: str,
    script_type: str,
    verbose: bool = True,
) -> None:
    """
    检查 dtype 是否被当前 GPU 支持（通过实际测试）。
    
    Args:
        ext: 编译好的 CUDA 扩展模块
        dtype: 要测试的数据类型
        outdtype: 输出数据类型
        backend: 后端类型 ("cublaslt" 或 "cusparselt")
        script_type: 脚本类型 ("alg_search" 或 "layout_search")
        verbose: 是否显示详细信息
    
    Raises:
        ValueError: 如果 dtype 不被支持
    """
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"不支持的数据类型: {dtype}\n"
            f"支持的类型: {', '.join(SUPPORTED_DTYPES)}"
        )
    
    if outdtype not in SUPPORTED_OUTDTYPES:
        raise ValueError(
            f"不支持的输出数据类型: {outdtype}\n"
            f"支持的类型: {', '.join(SUPPORTED_OUTDTYPES)}"
        )
    
    # 获取对应的探测函数
    prober_map = {
        ("cublaslt", "alg_search"): probe_cublaslt_search,
        ("cublaslt", "layout_search"): probe_cublaslt_layout,
        ("cusparselt", "alg_search"): probe_cusparselt_search,
        ("cusparselt", "layout_search"): probe_cusparselt_layout,
    }
    
    prober_key = (backend.lower(), script_type.lower())
    if prober_key not in prober_map:
        raise ValueError(f"未知的 backend/script_type 组合: {prober_key}")
    
    prober = prober_map[prober_key]
    
    if verbose:
        print(f"[预测试] 检测 dtype={dtype}, outdtype={outdtype} ...", end=" ", flush=True)
    
    supported, message = prober(ext, dtype, outdtype)
    
    if supported:
        if verbose:
            print("✓", flush=True)
    else:
        if verbose:
            print("✗", flush=True)
        raise ValueError(
            f"数据类型 {dtype.upper()} 在当前 GPU ({hw_info.arch_name}) 上不可用。\n"
            f"原因: {message}"
        )


# =============================================================================
# CSV 表头构建
# =============================================================================

def build_csv_header_lines(
    *,
    model_name: str,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    verify: bool,
    m_list: List[int],
    nk_list: List[Tuple[int, int]],
    layout: str = "TNCCcol",
    alg_count: int = 0,
    config_count: int = 0,
) -> List[str]:
    """
    构建 CSV 文件的元数据头部行。
    """
    import torch
    
    return [
        f"# GPU: {hw_info.gpu_full_name}",
        f"# CC: {hw_info.cc_tag}",
        f"# Model: {model_name}",
        f"# alg_count: {alg_count}, config_count: {config_count}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {hw_info.cuda_driver_version}, runtime: {hw_info.cuda_runtime_version}",
        f"# layout: {layout}, dtype: {dtype}, outdtype: {outdtype}, warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {m_list}",
        f"# NK_list: {nk_list}",
    ]


# =============================================================================
# Layout 配置（用于 Layout Search）
# =============================================================================

# 4 种主要 layout 配置
LAYOUT_CONFIGS = [
    {"name": "TN+CC", "opW": "T", "opA": "N", "orderW": "Col", "orderA": "Col"},
    {"name": "NT+RR", "opW": "N", "opA": "T", "orderW": "Row", "orderA": "Row"},
    {"name": "NN+RC", "opW": "N", "opA": "N", "orderW": "Row", "orderA": "Col"},
    {"name": "TT+CR", "opW": "T", "opA": "T", "orderW": "Col", "orderA": "Row"},
]

# 输出矩阵顺序
OUTPUT_ORDERS = ["Col", "Row"]


# =============================================================================
# 导出接口
# =============================================================================

__all__ = [
    # 常量
    "SUPPORTED_DTYPES",
    "SUPPORTED_OUTDTYPES",
    "LAYOUT_CONFIGS",
    "OUTPUT_ORDERS",
    # 默认配置
    "default_m_list",
    "default_nk_list_bitnet_2b",
    # 模型 NK 工具
    "get_nk_list_from_model",
    "get_nk_list_auto",
    "build_model_name_with_dtype",
    # 输出命名
    "build_output_dir_name",
    "build_result_filename",
    # 元数据
    "build_search_meta",
    "build_csv_header_lines",
    # Segment-K
    "supports_segment_k",
    # dtype 检测
    "check_dtype_support",
    "probe_cublaslt_search",
    "probe_cublaslt_layout",
    "probe_cusparselt_search",
    "probe_cusparselt_layout",
    # 重导出顶层 utils
    "hw_info",
    "normalize_dtype",
    "load_cuda_extension",
    "ensure_cublaslt_loaded",
    "ensure_cusparselt_loaded",
    "get_model_nk_sizes",
]
