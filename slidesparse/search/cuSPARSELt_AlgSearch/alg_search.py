#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

2:4 稀疏矩阵乘法 (SpMM):
- 权重 W 进行 2:4 剪枝后压缩
- 固定 Layout: T/N + Col/Col + Col

运行示例:
    python3 alg_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T
    python3 alg_search.py --dtype fp8e4m3 --outdtype bf16 --model BitNet-2B4T
    python3 alg_search.py --dtype int8 --outdtype bf16 --search_split_k
"""

import argparse
import base64
import ctypes
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
import numpy as np

# 添加 search 目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SEARCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SEARCH_DIR))

from utils import (
    hw_info,
    normalize_dtype,
    ensure_cusparselt_loaded,
    get_nk_list_auto,
    build_model_name_with_dtype,
    build_output_dir_name,
    build_result_filename,
    build_search_meta,
    build_csv_header_lines,
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
    default_m_list,
)


# =============================================================================
# CUDA 扩展编译与加载
# =============================================================================

def build_cuda_extension(
    source_file: Path,
    build_dir: Path,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """使用 nvcc 直接编译 CUDA 扩展"""
    build_dir.mkdir(parents=True, exist_ok=True)
    
    so_name = f"alg_search_cusparselt_{hw_info.gpu_name}_{hw_info.cc_tag}.so"
    so_path = build_dir / so_name
    
    if so_path.exists() and not force:
        if source_file.stat().st_mtime <= so_path.stat().st_mtime:
            if verbose:
                print(f"✓ Using existing: {so_path.name}")
            return so_path
    
    if verbose:
        print(f"🔨 Building {so_name}...")
    
    import os
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    nvcc = Path(cuda_home) / 'bin' / 'nvcc'
    
    cmd = [
        str(nvcc),
        '-std=c++17', '-O3', '-Xcompiler', '-fPIC', '--shared',
        f'-gencode=arch=compute_{hw_info.cc_major}{hw_info.cc_minor},'
        f'code=sm_{hw_info.cc_major}{hw_info.cc_minor}',
        f'-I{cuda_home}/include',
        str(source_file),
        '-L/usr/lib/x86_64-linux-gnu',
        '-lcusparseLt', '-lcusparse', '-lcublas', '-lcuda',
        '-o', str(so_path),
    ]
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        error_msg = result.stderr or result.stdout
        raise RuntimeError(f"编译失败:\n{error_msg}")
    
    if verbose:
        print(f"✓ Built: {so_path.name}")
    
    return so_path


def load_extension(so_path: Path) -> ctypes.CDLL:
    """加载编译好的 CUDA 扩展"""
    ensure_cusparselt_loaded()
    
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    
    # 设置函数签名
    lib.cusparselt_search_single_m.argtypes = [
        ctypes.c_void_p,   # W_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # C_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_char_p,   # outdtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.c_int,      # topk
        ctypes.c_int,      # search_split_k
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_int),        # out_split_k
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
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
    
    return lib


# =============================================================================
# 数据准备
# =============================================================================

def quantize_int8(x: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """将 BF16/FP16 张量量化到 INT8"""
    abs_max = x.abs().max().item()
    scale = 127.0 / abs_max if abs_max > 0 else 1.0
    q = (x * scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def to_fp8_e4m3(x: torch.Tensor) -> torch.Tensor:
    """转换为 FP8E4M3"""
    return x.to(torch.float8_e4m3fn)


def prepare_and_prune_weight(
    lib: ctypes.CDLL,
    W_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备并剪枝权重矩阵。
    
    返回:
        (W_pruned, W_compressed): 剪枝后的矩阵和压缩后的矩阵
    """
    N, K = W_bf16.shape
    
    # 量化
    if dtype == "int8":
        W_q, _ = quantize_int8(W_bf16)
    elif dtype == "fp8e4m3":
        W_q = to_fp8_e4m3(W_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    # 转置为 K x N (列主序存储)
    W_t = W_q.t().contiguous()
    
    # Prune 2:4
    W_pruned = torch.empty_like(W_t)
    ret = lib.cusparselt_prune_24(
        W_t.data_ptr(),
        W_pruned.data_ptr(),
        K, N,
        dtype.encode(),
        None,
    )
    if ret != 0:
        error = lib.cusparselt_alg_search_get_last_error()
        raise RuntimeError(f"Prune 失败: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    # 获取压缩大小
    compressed_size = lib.cusparselt_get_compressed_size(K, N, dtype.encode())
    if compressed_size < 0:
        raise RuntimeError("获取压缩大小失败")
    
    # 压缩
    W_compressed = torch.empty(compressed_size, dtype=torch.uint8, device=W_t.device)
    ret = lib.cusparselt_compress(
        W_pruned.data_ptr(),
        W_compressed.data_ptr(),
        K, N,
        dtype.encode(),
        None,
    )
    if ret < 0:
        error = lib.cusparselt_alg_search_get_last_error()
        raise RuntimeError(f"Compress 失败: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    return W_pruned, W_compressed


def prepare_activation(
    A_bf16: torch.Tensor,
    dtype: str,
) -> torch.Tensor:
    """准备激活矩阵"""
    if dtype == "int8":
        A_q, _ = quantize_int8(A_bf16)
    elif dtype == "fp8e4m3":
        A_q = to_fp8_e4m3(A_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    # 转置为 K x M (列主序)
    return A_q.t().contiguous()


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    W_compressed: torch.Tensor,
    A_q: torch.Tensor,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
    search_split_k: bool = False,
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的最佳算法"""
    # 分配输出缓冲
    C_torch_dtype = torch.float32 if outdtype == "fp32" else torch.bfloat16
    C_out = torch.zeros(N, M, dtype=C_torch_dtype, device=A_q.device)
    
    # 分配输出数组
    out_alg_ids = (ctypes.c_int * topk)()
    out_split_k = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    
    # 调用 C 函数
    ret = lib.cusparselt_search_single_m(
        W_compressed.data_ptr(),
        A_q.data_ptr(),
        C_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        outdtype.encode(),
        warmup,
        repeat,
        topk,
        1 if search_split_k else 0,
        out_alg_ids,
        out_split_k,
        out_lat_us,
        out_tops,
        out_workspace,
        out_valid,
        ctypes.byref(out_num_valid),
        ctypes.byref(out_alg_count),
        None,
    )
    
    if ret != 0:
        error = lib.cusparselt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
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
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    outdtype: str,
    nk_list: List[Tuple[int, int]],
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    search_split_k: bool = False,
    verbose: bool = True,
) -> Dict:
    """运行完整的算法搜索"""
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    max_alg_count = 0
    supports_segment_k = bool(lib.cusparselt_supports_segment_k())
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        
        # 剪枝并压缩权重
        W_pruned, W_compressed = prepare_and_prune_weight(lib, W, dtype)
        
        # 准备激活
        A_q = prepare_activation(A, dtype)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": {},
        }
        
        for M in m_list:
            # 切片 (A_q 是 K x M)
            A_slice = A_q[:, :M].contiguous()
            
            out = search_single_nk(
                lib, N, K, M,
                W_compressed, A_slice,
                dtype, outdtype,
                warmup, repeat, topk,
                search_split_k,
            )
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
        
        if verbose:
            first_m = m_list[0]
            first_result = nk_results["m_results"][first_m]
            print(f"      → 算法数: {first_result['alg_count']}, 有效: {first_result['num_valid']}")
        
        results.append(nk_results)
        
        del W, A, W_pruned, W_compressed, A_q
    
    torch.cuda.empty_cache()
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
        "supports_segment_k": supports_segment_k,
        "search_split_k": search_split_k,
    }


# =============================================================================
# 结果保存
# =============================================================================

def save_outputs(
    out_dir: Path,
    model_name: str,
    dtype: str,
    outdtype: str,
    search_ret: Dict,
    warmup: int,
    repeat: int,
    verify: bool,
) -> Path:
    """保存搜索结果"""
    layout = "TNCCcol_sparse24"
    
    subdir_name = build_output_dir_name(model_name, dtype, outdtype)
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / build_result_filename("alg_search_bench", model_name, "csv")
    json_path = subdir / build_result_filename("alg_search_LUT", model_name, "json")
    
    alg_count = search_ret.get("max_alg_count", 0)
    config_count = alg_count * (6 if search_ret.get("search_split_k") else 1)
    
    # === CSV 生成 ===
    header_lines = build_csv_header_lines(
        model_name=model_name,
        dtype=dtype,
        outdtype=outdtype,
        warmup=warmup,
        repeat=repeat,
        verify=verify,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        layout=layout,
        alg_count=alg_count,
        config_count=config_count,
    )
    
    csv_lines = list(header_lines)
    csv_lines.append("M,N,K,alg_count,config_count,tops1,lat_us1,id1,sk1,ws1,tops2,lat_us2,id2,sk2,ws2,tops3,lat_us3,id3,sk3,ws3")
    
    csv_rows = []
    
    for nk_idx, nk_res in enumerate(search_ret["results"]):
        N, K = nk_res["N"], nk_res["K"]
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            values = [str(M), str(N), str(K), str(m_res.get("alg_count", 0)), str(config_count)]
            
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
            
            csv_rows.append((M, nk_idx, ",".join(values)))
    
    csv_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in csv_rows:
        csv_lines.append(line)
    
    csv_path.write_text("\n".join(csv_lines))
    
    # === JSON 生成 ===
    meta = build_search_meta(
        dtype=dtype,
        outdtype=outdtype,
        warmup=warmup,
        repeat=repeat,
        verify=verify,
        m_list=search_ret["M_list"],
        nk_list=search_ret["NK_list"],
        model_name=model_name,
        layout=layout,
        alg_count=alg_count,
        config_count=config_count,
    )
    meta["supports_segment_k"] = search_ret.get("supports_segment_k", False)
    meta["search_split_k"] = search_ret.get("search_split_k", False)
    
    nk_entries = {}
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        nk_key = f"({N},{K})"
        
        m_thresholds = []
        alg_by_m = {}
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            if results:
                m_thresholds.append(M)
                top3_info = []
                for r in results[:3]:
                    top3_info.append({
                        "alg_id": r["alg_id"],
                        "split_k": r["split_k"],
                    })
                alg_by_m[str(M)] = top3_info
        
        nk_entries[nk_key] = {
            "m_thresholds": m_thresholds,
            "alg_by_m": alg_by_m,
        }
    
    json_payload = {
        "meta": meta,
        "nk_entries": nk_entries,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))
    
    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt 算法离线搜索")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default="BitNet-2B4T", help="模型名称或路径")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--search_split_k", action="store_true", help="搜索 split-k 配置")
    p.add_argument("--out_dir", default=None, help="输出目录")
    p.add_argument("--m_list", type=str, default=None, help="M 列表，逗号分隔")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    
    model_name = build_model_name_with_dtype(args.model.split('/')[-1], args.dtype)
    
    print("=" * 60)
    print("cuSPARSELt 算法离线搜索 (2:4 稀疏)")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag}, {hw_info.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}")
    print(f"Split-K 搜索: {'开启' if args.search_split_k else '关闭'}")
    print()
    
    out_dir = Path(args.out_dir) if args.out_dir else Path("./alg_search_results")
    
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "alg_search_cusparselt.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_cuda_extension(src_path, build_dir, force=args.compile)
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_extension(so_path)
    
    if not lib.cusparselt_alg_search_is_available():
        raise RuntimeError("cuSPARSELt 不可用")
    print("✓ cuSPARSELt 可用")
    
    supports_segment_k = bool(lib.cusparselt_supports_segment_k())
    print(f"✓ Segment-K 支持: {'是' if supports_segment_k else '否'}")
    
    nk_list = get_nk_list_auto(args.model, with_names=False)
    
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    else:
        m_list = default_m_list()
    
    print()
    print(f"[3/4] 开始算法搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()
    
    ret = run_search(
        lib,
        args.dtype,
        args.outdtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        topk=3,
        search_split_k=args.search_split_k,
        verbose=True,
    )
    
    saved_dir = save_outputs(
        out_dir,
        model_name,
        args.dtype,
        args.outdtype,
        ret,
        args.warmup,
        args.repeat,
        args.verify,
    )
    
    print()
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
