#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt Sparse GEMM 算法搜索 (2:4 稀疏)

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

支持的数据类型（输入/输出相同）:
==============================
- FP16:    FP16 输入, FP32 计算, FP16 输出
- BF16:    BF16 输入, FP32 计算, BF16 输出
- INT8:    INT8 输入, INT32 计算, INT8 输出
- FP8E4M3: FP8 输入, FP32 计算, FP8 输出
- FP4E2M1: FP4 输入, FP32 计算, FP4 输出

固定 Layout:
- T/N + Col/Col + Col (权重 W 在左, 2:4 稀疏)
- R[N,M]_col = W_compressed[K,N]^T_col @ A[K,M]_col

搜索策略:
- 自适应 Split-K 倍增策略 (1, 2, 4, 8, ...)
- Segment-K 测试 (SM90+ 支持 split_k=-1)
- 官方 API 搜索对比 (cusparseLtMatmulSearch)

Sparsity 说明:
=============
对于 K_slide 测试，用户需要传入已经计算好的 K_slide 作为 K 参数。
例如：sparsity=2_8 时，K_slide = 1.5 * K_original

运行示例:
    # Model-based 模式
    python3 alg_search.py --dtype int8 --model Qwen2.5-0.5B
    python3 alg_search.py --dtype fp8e4m3 --model Llama3.2-1B
    
    # Square Matrix 模式 (不指定 --model)
    python3 alg_search.py --dtype bf16 --N 4096 --K 4096 --M-quick
    
    # 指定 sparsity (会自动计算 K_slide)
    python3 alg_search.py --dtype fp8e4m3 --N 4096 --K 4096 --sparsity 2_8
"""

import argparse
import datetime
import json
import ctypes
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# 添加路径
SCRIPT_DIR = Path(__file__).parent.absolute()
BENCHMARK_KERNEL_DIR = SCRIPT_DIR.parent
SLIDESPARSE_DIR = BENCHMARK_KERNEL_DIR.parent
PROJECT_ROOT = SLIDESPARSE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from slidesparse.benchmark_kernel.utils import (
    # 常量
    DTYPE_CONFIG,
    SUPPORTED_DTYPES,
    DEFAULT_SPARSITY_LIST,
    ALIGNMENT,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    SQUARE_M_LIST,
    # 硬件检测
    hw_info,
    check_dtype_support,
    get_supported_dtypes_for_gpu,
    check_cusparselt_support,
    check_segment_k_support,
    # Sparsity 计算
    calculate_k_slide,
    get_k_expansion_factor,
    pad_to_alignment,
    # 数据准备
    quantize_int8,
    to_fp8_e4m3,
    to_fp4_e2m1_packed,
    get_output_torch_dtype,
    # NK 列表
    get_nk_list_for_benchmark,
    # 编译与加载
    build_benchmark_extension,
    load_benchmark_extension,
    # 文件命名
    build_hw_folder_name,
    build_result_filename,
    build_csv_header_lines,
)


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名"""
    lib.cusparselt_search_single_m.argtypes = [
        ctypes.c_void_p,   # W_pruned_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # R_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.c_int,      # topk
        ctypes.c_int,      # test_segment_k
        ctypes.c_int,      # do_api_search
        # 输出
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_int),        # out_split_k
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
        ctypes.POINTER(ctypes.c_int),        # out_config_count
        ctypes.POINTER(ctypes.c_int),        # out_api_alg_id
        ctypes.POINTER(ctypes.c_int),        # out_api_split_k
        ctypes.POINTER(ctypes.c_float),      # out_api_lat_us
        ctypes.POINTER(ctypes.c_int),        # out_api_rank
        ctypes.c_void_p,   # stream
    ]
    lib.cusparselt_search_single_m.restype = ctypes.c_int
    
    lib.cusparselt_prune_24.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.cusparselt_prune_24.restype = ctypes.c_int
    
    lib.cusparselt_compress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_void_p,
    ]
    lib.cusparselt_compress.restype = ctypes.c_int64
    
    lib.cusparselt_get_compressed_size.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
    ]
    lib.cusparselt_get_compressed_size.restype = ctypes.c_int64
    
    lib.cusparselt_supports_segment_k.argtypes = []
    lib.cusparselt_supports_segment_k.restype = ctypes.c_int
    
    lib.cusparselt_alg_search_is_available.argtypes = []
    lib.cusparselt_alg_search_is_available.restype = ctypes.c_int
    
    lib.cusparselt_alg_search_get_last_error.argtypes = []
    lib.cusparselt_alg_search_get_last_error.restype = ctypes.c_char_p


# =============================================================================
# 数据准备
# =============================================================================

def prepare_and_prune_weight(
    lib: ctypes.CDLL,
    W_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备并剪枝权重矩阵
    
    Returns:
        (W_pruned, W_q): 
            W_pruned: 剪枝后的矩阵 [K, N] (列主序)
            W_q: 原始量化权重 [N, K]
    """
    N, K = W_bf16.shape
    dtype_lower = dtype.lower()
    is_fp4 = dtype_lower in ("fp4e2m1", "fp4")
    
    # 量化
    if dtype_lower == "int8":
        W_q, _ = quantize_int8(W_bf16)
    elif dtype_lower in ("fp8e4m3", "fp8"):
        W_q = to_fp8_e4m3(W_bf16)
    elif dtype_lower == "fp16":
        W_q = W_bf16.to(torch.float16)
    elif dtype_lower == "bf16":
        W_q = W_bf16
    elif is_fp4:
        # FP4 打包：K 维度减半
        W_q = to_fp4_e2m1_packed(W_bf16)  # [N, K//2]
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # 转置为 K x N (列主序存储)
    # 对于 FP4，维度已经是打包后的
    W_t = W_q.t()
    
    # Prune 2:4
    W_pruned = torch.empty_like(W_t)
    
    # 对于 FP4，传递原始 K（CUDA 端知道实际存储是 K/2）
    ret = lib.cusparselt_prune_24(
        W_t.data_ptr(),
        W_pruned.data_ptr(),
        K, N,
        dtype.encode(),
        None,
    )
    if ret != 0:
        error = lib.cusparselt_alg_search_get_last_error()
        raise RuntimeError(f"Prune failed: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    return W_pruned, W_q


def prepare_activation(
    A_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备激活矩阵
    
    Returns:
        (A_transposed, A_q):
            A_transposed: 转置后的激活 [K, M] (用于 CUDA)
            A_q: 原始量化激活 [M, K]
    """
    dtype_lower = dtype.lower()
    is_fp4 = dtype_lower in ("fp4e2m1", "fp4")
    
    if dtype_lower == "int8":
        A_q, _ = quantize_int8(A_bf16)
    elif dtype_lower in ("fp8e4m3", "fp8"):
        A_q = to_fp8_e4m3(A_bf16)
    elif dtype_lower == "fp16":
        A_q = A_bf16.to(torch.float16)
    elif dtype_lower == "bf16":
        A_q = A_bf16
    elif is_fp4:
        # FP4 打包：K 维度减半
        A_q = to_fp4_e2m1_packed(A_bf16)  # [M, K//2]
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # 转置为 K x M (列主序)
    A_transposed = A_q.t()
    
    return A_transposed, A_q


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    W_pruned: torch.Tensor,
    A_transposed: torch.Tensor,
    dtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
    test_segment_k: bool = True,
    do_api_search: bool = True,
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的最佳算法"""
    # 分配输出缓冲
    # 注意：INT8 输出 BF16，FP8/FP4 输出 BF16
    R_torch_dtype = get_output_torch_dtype(dtype, backend="cusparselt")
    R_out = torch.empty_strided((N, M), (1, N), dtype=R_torch_dtype, device=A_transposed.device)
    R_out.zero_()
    
    # 分配输出数组
    out_alg_ids = (ctypes.c_int * topk)()
    out_split_k = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    out_config_count = ctypes.c_int(0)
    out_api_alg_id = ctypes.c_int(-1)
    out_api_split_k = ctypes.c_int(1)
    out_api_lat_us = ctypes.c_float(0.0)
    out_api_rank = ctypes.c_int(-1)
    
    # 调用 C 函数
    ret = lib.cusparselt_search_single_m(
        W_pruned.data_ptr(),
        A_transposed.data_ptr(),
        R_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        warmup,
        repeat,
        topk,
        1 if test_segment_k else 0,
        1 if do_api_search else 0,
        out_alg_ids,
        out_split_k,
        out_lat_us,
        out_tops,
        out_workspace,
        out_valid,
        ctypes.byref(out_num_valid),
        ctypes.byref(out_alg_count),
        ctypes.byref(out_config_count),
        ctypes.byref(out_api_alg_id),
        ctypes.byref(out_api_split_k),
        ctypes.byref(out_api_lat_us),
        ctypes.byref(out_api_rank),
        None,
    )
    
    if ret != 0:
        error = lib.cusparselt_alg_search_get_last_error()
        return {
            "results": [],
            "num_valid": 0,
            "alg_count": 0,
            "config_count": 0,
            "api_result": None,
            "error": error.decode() if error else "unknown error",
        }
    
    # 转换结果
    results = []
    for i in range(topk):
        if out_valid[i]:
            results.append({
                "alg_id": out_alg_ids[i],
                "split_k": out_split_k[i],
                "lat_us": out_lat_us[i],
                "tops": out_tops[i],
                "workspace": out_workspace[i],
            })
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
        "alg_count": out_alg_count.value,
        "config_count": out_config_count.value,
        "api_result": {
            "alg_id": out_api_alg_id.value,
            "split_k": out_api_split_k.value,
            "lat_us": out_api_lat_us.value,
            "rank": out_api_rank.value,
        } if out_api_alg_id.value >= 0 else None,
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    nk_list: List[Tuple[int, int]],
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    test_segment_k: bool = True,
    do_api_search: bool = True,
    verbose: bool = True,
    is_square_mode: bool = False,
) -> Dict:
    """
    运行完整的算法搜索
    
    Args:
        is_square_mode: 如果为 True，只测试 M=N=K 的组合
                       （此时 nk_list 应该是 [(m,m) for m in m_list]）
    """
    results = []
    total_nk = len(nk_list)
    
    max_alg_count = 0
    max_config_count = 0
    supports_segment_k_hw = bool(lib.cusparselt_supports_segment_k())
    
    # 搜索统计
    search_stats = {"total": 0, "success": 0, "failed": 0, "errors": 0}
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # Square 模式下只测试 M=N=K，否则测试所有 M
        effective_m_list = [N] if is_square_mode else m_list
        max_M = max(effective_m_list)
        
        # 生成随机数据
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        
        # 剪枝权重
        W_pruned, W_q = prepare_and_prune_weight(lib, W, dtype)
        
        # 准备激活
        A_transposed, A_q = prepare_activation(A, dtype)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": {},
        }
        
        for M in effective_m_list:
            # 切片
            A_slice = A_transposed[:, :M]
            
            out = search_single_nk(
                lib, N, K, M,
                W_pruned, A_slice,
                dtype,
                warmup, repeat, topk,
                test_segment_k=test_segment_k and supports_segment_k_hw,
                do_api_search=do_api_search,
            )
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
            if out.get("config_count", 0) > max_config_count:
                max_config_count = out["config_count"]
            
            # 更新搜索统计
            search_stats["total"] += 1
            if out.get("error"):
                search_stats["errors"] += 1
            elif out["num_valid"] > 0:
                search_stats["success"] += 1
            else:
                search_stats["failed"] += 1
        
        if verbose:
            first_m = effective_m_list[0]
            first_result = nk_results["m_results"][first_m]
            print(f"      → 算法数: {first_result['alg_count']}, 有效: {first_result['num_valid']}")
        
        results.append(nk_results)
        
        # 释放
        del W, A, W_pruned, A_transposed, W_q, A_q
        torch.cuda.empty_cache()
    
    # 打印搜索统计汇总
    if verbose:
        print()
        print(f"    搜索统计: 总计={search_stats['total']}, "
              f"成功={search_stats['success']}, "
              f"失败={search_stats['failed']}, "
              f"错误={search_stats['errors']}")
        if search_stats["total"] > 0:
            success_rate = 100.0 * search_stats["success"] / search_stats["total"]
            print(f"    成功率: {success_rate:.1f}%")
    
    return {
        "dtype": dtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
        "max_config_count": max_config_count,
        "supports_segment_k": supports_segment_k_hw,
        "search_stats": search_stats,
    }


# =============================================================================
# 结果保存
# =============================================================================

def save_results(
    out_dir: Path,
    model_name: str,
    dtype: str,
    mode: str,
    search_ret: Dict,
    warmup: int,
    repeat: int,
    sparsity: Optional[str] = None,
) -> Path:
    """
    保存搜索结果到 CSV 和 JSON 文件
    """
    hw_folder = build_hw_folder_name()
    subdir = out_dir / hw_folder
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = build_result_filename("alg_search", model_name, dtype, "csv", sparsity)
    json_filename = build_result_filename("alg_search", model_name, dtype, "json", sparsity)
    csv_path = subdir / csv_filename
    json_path = subdir / json_filename
    
    alg_count = search_ret.get("max_alg_count", 0)
    config_count = search_ret.get("max_config_count", 0)
    
    # === CSV 生成 ===
    header_lines = build_csv_header_lines(
        model_name=model_name,
        dtype=dtype,
        mode=mode,
        warmup=warmup,
        repeat=repeat,
        verify=False,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        backend="cuSPARSELt",
        alg_count=alg_count,
        config_count=config_count,
        sparsity=sparsity,
    )
    
    csv_lines = list(header_lines)
    csv_lines.append("M,N,K,alg_count,tops_1,lat_us_1,alg_id_1,split_k_1,workspace_1,tops_2,lat_us_2,alg_id_2,split_k_2,workspace_2,tops_3,lat_us_3,alg_id_3,split_k_3,workspace_3")
    
    csv_rows = []
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        
        # 遍历实际存在的 M 结果（Square 模式下只有 M=N）
        for M, m_res in nk_res["m_results"].items():
            results = m_res.get("results", [])
            
            values = [str(M), str(N), str(K), str(m_res.get("alg_count", 0))]
            
            for k in range(3):
                if k < len(results):
                    r = results[k]
                    values.extend([
                        f"{r['tops']:.6f}",
                        f"{r['lat_us']:.3f}",
                        str(r['alg_id']),
                        str(r['split_k']),
                        str(r['workspace']),
                    ])
                else:
                    values.extend(["", "", "", "", ""])
            
            csv_rows.append((M, N, K, ",".join(values)))
    
    csv_rows.sort(key=lambda x: (x[0], x[1], x[2]))
    for _, _, _, line in csv_rows:
        csv_lines.append(line)
    
    csv_path.write_text("\n".join(csv_lines))
    
    # === JSON 生成 ===
    meta = {
        "gpu_name": hw_info.gpu_full_name,
        "gpu_short_name": hw_info.gpu_name,
        "compute_capability": hw_info.cc_tag,
        "model_name": model_name,
        "mode": mode,
        "backend": "cuSPARSELt",
        "dtype": dtype,
        "sparsity": sparsity,
        "alg_count": alg_count,
        "config_count": config_count,
        "supports_segment_k": search_ret.get("supports_segment_k", False),
        "warmup": warmup,
        "repeat": repeat,
        "torch_version": torch.__version__,
        "time": datetime.datetime.now().isoformat(),
        "M_list": search_ret["M_list"],
        "NK_list": [[n, k] for n, k in search_ret["NK_list"]],
    }
    
    nk_entries = {}
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        nk_key = f"({N},{K})"
        
        m_thresholds = []
        alg_by_m = {}
        
        # 遍历实际存在的 M 结果（Square 模式下只有 M=N）
        for M, m_res in nk_res["m_results"].items():
            results = m_res.get("results", [])
            
            if results:
                m_thresholds.append(M)
                r = results[0]  # top1
                alg_by_m[str(M)] = {
                    "alg_id": r["alg_id"],
                    "split_k": r.get("split_k", 1),
                    "workspace": r.get("workspace", 0),
                    "tops": r["tops"],
                    "lat_us": r["lat_us"],
                }
        
        nk_entries[nk_key] = {
            "m_thresholds": m_thresholds,
            "alg_by_m": alg_by_m,
        }
    
    json_payload = {
        "meta": meta,
        "nk_entries": nk_entries,
    }
    
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))
    
    print(f"    CSV: {csv_path}")
    print(f"    JSON: {json_path}")
    
    return subdir


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="cuSPARSELt Sparse GEMM 算法搜索 (2:4 稀疏)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Model-based 模式
  python3 alg_search.py --dtype int8 --model Qwen2.5-0.5B
  python3 alg_search.py --dtype fp8e4m3 --model Llama3.2-1B --sparsity 2_8
  
  # Square 模式 (不指定 --model 或 --model square)
  python3 alg_search.py --dtype bf16
  python3 alg_search.py --dtype bf16 --model square --sparsity 2_6
        """
    )
    p.add_argument("--dtype", required=True, choices=SUPPORTED_DTYPES,
                   help="数据类型 (fp16, bf16, int8, fp8e4m3, fp4e2m1)")
    p.add_argument("--model", default=None,
                   help="模型名称（不指定或指定 'square' 进入 Square 模式）")
    p.add_argument("--sparsity", type=str, default=None,
                   help="稀疏度配置 (如 2_4, 2_6, 2_8)，会自动计算 K_slide")
    p.add_argument("--M-quick", action="store_true", dest="m_quick",
                   help="使用快速 M 列表 (仅 Model-based 模式有效)")
    p.add_argument("--m_list", type=str, default=None,
                   help="M 列表，逗号分隔 (Square 模式: M=N=K 使用此列表)")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--compile", action="store_true",
                   help="强制重新编译 CUDA 扩展")
    p.add_argument("--no_segment_k", action="store_true",
                   help="禁用 Segment-K 测试")
    p.add_argument("--no_api_search", action="store_true",
                   help="禁用官方 API 搜索对比")
    p.add_argument("--out_dir", default=None,
                   help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    
    # 检查 dtype 支持
    supported, reason = check_dtype_support(args.dtype)
    if not supported:
        print(f"[ERROR] {reason}")
        print(f"[INFO] 当前 GPU 支持的类型: {get_supported_dtypes_for_gpu()}")
        sys.exit(1)
    
    # 检查 cuSPARSELt 支持
    cusparselt_ok, cusparselt_reason = check_cusparselt_support()
    if not cusparselt_ok:
        print(f"[ERROR] {cusparselt_reason}")
        sys.exit(1)
    
    # 确定运行模式
    is_square_mode = (args.model is None or args.model.lower() == "square")
    
    # 确定 M 列表
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    elif args.m_quick:
        m_list = list(M_QUICK_LIST)
    else:
        m_list = list(M_QUICK_LIST)  # 默认使用 M_QUICK_LIST
    
    # 确保 M 是 32 的倍数
    m_list = [pad_to_alignment(m) for m in m_list]
    
    # 获取 NK 列表（Square 模式会使用 m_list 构建 M=N=K）
    nk_list, model_name, mode = get_nk_list_for_benchmark(
        model=args.model,
        m_list=m_list if is_square_mode else None
    )
    
    # 如果指定了 sparsity，计算 K_slide
    sparsity = args.sparsity
    if sparsity:
        k_factor = get_k_expansion_factor(sparsity)
        nk_list = [(n, calculate_k_slide(k, sparsity)) for n, k in nk_list]
        print(f"[INFO] Sparsity={sparsity}, K_factor={k_factor:.3f}")
        print(f"[INFO] NK list adjusted: {nk_list[:5]}..." if len(nk_list) > 5 else f"[INFO] NK list adjusted: {nk_list}")
    
    test_segment_k = not args.no_segment_k
    do_api_search = not args.no_api_search
    
    segment_k_ok, segment_k_reason = check_segment_k_support()
    
    # 显示配置
    print("=" * 60)
    print("cuSPARSELt Sparse GEMM 算法搜索 (2:4 稀疏)")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"Mode: {mode.upper()}")
    print(f"Model: {model_name}")
    print(f"dtype: {args.dtype} -> {args.dtype} (same input/output)")
    if sparsity:
        print(f"Sparsity: {sparsity}")
    print(f"Segment-K 测试: {'开启' if test_segment_k and segment_k_ok else '关闭'}")
    print(f"API 搜索对比: {'开启' if do_api_search else '关闭'}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print()
    
    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else SCRIPT_DIR / "alg_search_results"
    
    # 编译 CUDA 扩展
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "cusparselt_gemm.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_benchmark_extension(
        name="cusparselt_gemm",
        source_file=src_path,
        build_dir=build_dir,
        backend="cusparselt",
        force=args.compile,
    )
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_benchmark_extension(so_path, backend="cusparselt", setup_func=setup_lib_signatures)
    
    if not lib.cusparselt_alg_search_is_available():
        raise RuntimeError("cuSPARSELt 不可用")
    print("✓ cuSPARSELt 可用")
    
    supports_segment_k_hw = bool(lib.cusparselt_supports_segment_k())
    print(f"✓ Segment-K 支持: {'是' if supports_segment_k_hw else '否'}")
    
    print()
    print(f"[3/4] 开始算法搜索...")
    if is_square_mode:
        print(f"      M=N=K 列表: {m_list}")
    else:
        print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()
    
    ret = run_search(
        lib,
        args.dtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        topk=3,
        test_segment_k=test_segment_k,
        do_api_search=do_api_search,
        verbose=True,
        is_square_mode=is_square_mode,
    )
    
    print()
    print("[4/4] 保存结果...")
    saved_dir = save_results(
        out_dir,
        model_name,
        args.dtype,
        mode,
        ret,
        args.warmup,
        args.repeat,
        sparsity=sparsity,
    )
    
    print()
    print(f"✓ 完成! 结果已保存到: {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
