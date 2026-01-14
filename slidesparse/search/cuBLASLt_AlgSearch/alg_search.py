#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuBLASLt API 调用、精确计时

固定 Layout:
- T/N + Col/Col + Col (权重 W 在左)
- W[N,K]^T_col * A[K,M]_col = C[N,M]_col

运行示例:
    python3 alg_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T
    python3 alg_search.py --dtype fp8e4m3 --outdtype bf16 --model BitNet-2B4T
    python3 alg_search.py --dtype int8 --outdtype bf16 --model /path/to/model
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
    # 硬件信息
    hw_info,
    normalize_dtype,
    # 编译与加载
    ensure_cublaslt_loaded,
    # 模型 NK 工具
    get_nk_list_auto,
    build_model_name_with_dtype,
    # 输出命名
    build_output_dir_name,
    build_result_filename,
    # 元数据
    build_search_meta,
    build_csv_header_lines,
    # dtype 检测
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
    # 默认配置
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
    """
    使用 nvcc 直接编译 CUDA 扩展为 .so 文件。
    
    Returns:
        编译生成的 .so 文件路径
    """
    build_dir.mkdir(parents=True, exist_ok=True)
    
    so_name = f"alg_search_cublaslt_{hw_info.gpu_name}_{hw_info.cc_tag}.so"
    so_path = build_dir / so_name
    
    # 检查是否需要重新编译
    if so_path.exists() and not force:
        if source_file.stat().st_mtime <= so_path.stat().st_mtime:
            if verbose:
                print(f"✓ Using existing: {so_path.name}")
            return so_path
    
    if verbose:
        print(f"🔨 Building {so_name}...")
    
    # CUDA 路径
    import os
    cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    nvcc = Path(cuda_home) / 'bin' / 'nvcc'
    
    # 编译命令
    cmd = [
        str(nvcc),
        '-std=c++17', '-O3', '-Xcompiler', '-fPIC', '--shared',
        f'-gencode=arch=compute_{hw_info.cc_major}{hw_info.cc_minor},'
        f'code=sm_{hw_info.cc_major}{hw_info.cc_minor}',
        f'-I{cuda_home}/include',
        str(source_file),
        '-L/usr/lib/x86_64-linux-gnu',
        '-lcublasLt', '-lcublas', '-lcuda',
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
    """
    加载编译好的 CUDA 扩展。
    """
    # 确保 cuBLASLt 库已加载
    ensure_cublaslt_loaded()
    
    # 加载扩展
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    
    # 设置函数签名
    lib.cublaslt_search_single_m.argtypes = [
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
        ctypes.POINTER(ctypes.c_int),        # out_alg_ids
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_float),      # out_waves_count
        ctypes.POINTER(ctypes.c_uint8),      # out_algo_data
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.POINTER(ctypes.c_int),        # out_alg_count
        ctypes.c_void_p,   # stream
    ]
    lib.cublaslt_search_single_m.restype = ctypes.c_int
    
    lib.cublaslt_alg_search_is_available.argtypes = []
    lib.cublaslt_alg_search_is_available.restype = ctypes.c_int
    
    lib.cublaslt_alg_search_get_last_error.argtypes = []
    lib.cublaslt_alg_search_get_last_error.restype = ctypes.c_char_p
    
    lib.cublaslt_alg_search_get_alignment.argtypes = [ctypes.c_char_p]
    lib.cublaslt_alg_search_get_alignment.restype = ctypes.c_int
    
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


def prepare_data(
    W_bf16: torch.Tensor,
    A_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """准备量化后的数据"""
    if dtype == "int8":
        W_q, _ = quantize_int8(W_bf16)
        A_q, _ = quantize_int8(A_bf16)
    elif dtype == "fp8e4m3":
        W_q = to_fp8_e4m3(W_bf16)
        A_q = to_fp8_e4m3(A_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    return W_q, A_q


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    W_q: torch.Tensor,
    A_q: torch.Tensor,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
    topk: int = 3,
) -> Dict[str, Any]:
    """
    搜索单个 (N, K, M) 组合的最佳算法。
    """
    # 分配输出缓冲
    C_torch_dtype = torch.float32 if outdtype == "fp32" else torch.bfloat16
    C_out = torch.zeros(M, N, dtype=C_torch_dtype, device=W_q.device)
    
    # 分配输出数组
    out_alg_ids = (ctypes.c_int * topk)()
    out_lat_us = (ctypes.c_float * topk)()
    out_tops = (ctypes.c_float * topk)()
    out_workspace = (ctypes.c_int64 * topk)()
    out_waves_count = (ctypes.c_float * topk)()
    out_algo_data = (ctypes.c_uint8 * (topk * 64))()
    out_valid = (ctypes.c_uint8 * topk)()
    out_num_valid = ctypes.c_int(0)
    out_alg_count = ctypes.c_int(0)
    
    # 调用 C 函数
    ret = lib.cublaslt_search_single_m(
        W_q.data_ptr(),
        A_q.data_ptr(),
        C_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        outdtype.encode(),
        warmup,
        repeat,
        topk,
        out_alg_ids,
        out_lat_us,
        out_tops,
        out_workspace,
        out_waves_count,
        out_algo_data,
        out_valid,
        ctypes.byref(out_num_valid),
        ctypes.byref(out_alg_count),
        None,  # 使用默认 stream
    )
    
    if ret != 0:
        error = lib.cublaslt_alg_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
    # 转换结果
    results = []
    for i in range(topk):
        if out_valid[i]:
            algo_bytes = bytes(out_algo_data[i*64:(i+1)*64])
            results.append({
                "alg_id": out_alg_ids[i],
                "lat_us": out_lat_us[i],
                "tops": out_tops[i],
                "workspace": out_workspace[i],
                "waves_count": out_waves_count[i],
                "algo_data": algo_bytes,
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
    verbose: bool = True,
) -> Dict:
    """
    运行完整的算法搜索。
    """
    layout = "TNCCcol"
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    max_alg_count = 0
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        
        # 量化
        W_q, A_q = prepare_data(W, A, dtype)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": {},
        }
        
        for M in m_list:
            # 切片
            A_slice = A_q[:M].contiguous()
            
            out = search_single_nk(
                lib, N, K, M,
                W_q, A_slice,
                dtype, outdtype,
                warmup, repeat, topk,
            )
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
        
        if verbose:
            first_m = m_list[0]
            first_result = nk_results["m_results"][first_m]
            print(f"      → 启发式返回: {first_result['alg_count']} 算法, 有效: {first_result['num_valid']}")
        
        results.append(nk_results)
        
        # 释放
        del W, A, W_q, A_q
    
    torch.cuda.empty_cache()
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
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
    """
    保存搜索结果到 CSV 和 JSON 文件。
    """
    layout = "TNCCcol"
    
    subdir_name = build_output_dir_name(model_name, dtype, outdtype)
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / build_result_filename("alg_search_bench", model_name, "csv")
    json_path = subdir / build_result_filename("alg_search_LUT", model_name, "json")
    
    alg_count = search_ret.get("max_alg_count", 0)
    config_count = alg_count  # cuBLASLt 没有 split-k
    
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
    
    # CSV 数据列
    csv_lines = list(header_lines)
    csv_lines.append("M,N,K,alg_count,config_count,tops1,lat_us1,id1,ws1,waves1,tops2,lat_us2,id2,ws2,waves2,tops3,lat_us3,id3,ws3,waves3")
    
    csv_rows = []  # [(M, nk_idx, line), ...]
    
    for nk_idx, nk_res in enumerate(search_ret["results"]):
        N, K = nk_res["N"], nk_res["K"]
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            values = [str(M), str(N), str(K), str(m_res.get("alg_count", 0)), str(m_res.get("alg_count", 0))]
            
            for k in range(3):
                if k < len(results):
                    r = results[k]
                    values.extend([
                        f"{r['tops']:.6f}",
                        f"{r['lat_us']:.3f}",
                        str(r['alg_id']),
                        str(r['workspace']),
                        f"{r['waves_count']:.4f}",
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
                top3_b64 = []
                for r in results[:3]:
                    if "algo_data" in r:
                        algo_b64 = base64.b64encode(r["algo_data"]).decode('ascii')
                        top3_b64.append(algo_b64)
                alg_by_m[str(M)] = top3_b64
        
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
    p = argparse.ArgumentParser(description="cuBLASLt 算法离线搜索")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default="BitNet-2B4T", help="模型名称或路径")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录")
    p.add_argument("--m_list", type=str, default=None, help="M 列表，逗号分隔，如 16,128,512,2048,16384")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    
    # 构建模型名称
    dtype_suffix = normalize_dtype(args.dtype)
    model_name = build_model_name_with_dtype(args.model.split('/')[-1], args.dtype)
    
    # === 显示配置信息 ===
    print("=" * 60)
    print("cuBLASLt 算法离线搜索")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag}, {hw_info.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}, warmup={args.warmup}, repeat={args.repeat}")
    print()
    
    # 输出目录
    out_dir = Path(args.out_dir) if args.out_dir else Path("./alg_search_results")
    
    # 编译 CUDA 扩展
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "alg_search_cublaslt.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_cuda_extension(src_path, build_dir, force=args.compile)
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_extension(so_path)
    
    if not lib.cublaslt_alg_search_is_available():
        raise RuntimeError("cuBLASLt 不可用")
    print("✓ cuBLASLt 可用")
    
    # 获取 NK 列表
    nk_list = get_nk_list_auto(args.model, with_names=False)
    
    # 获取 M 列表
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
