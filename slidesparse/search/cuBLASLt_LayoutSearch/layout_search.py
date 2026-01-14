#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt 布局离线搜索

架构说明：
=========
测试 8 种布局组合:
  - 转置: TT, TN, NT, NN
  - A/B 排列: RowCol, ColCol
  (D 输出固定为 ColMajor)

固定最优布局: T/N + Col/Col + Col

运行示例:
    python3 layout_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T
"""

import argparse
import ctypes
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
    ensure_cublaslt_loaded,
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
# 布局常量
# =============================================================================

NUM_LAYOUTS = 8

LAYOUT_NAMES = [
    "TT_RowCol", "TN_RowCol", "NT_RowCol", "NN_RowCol",
    "TT_ColCol", "TN_ColCol", "NT_ColCol", "NN_ColCol",
]


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
    
    so_name = f"layout_search_cublaslt_{hw_info.gpu_name}_{hw_info.cc_tag}.so"
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
    """加载编译好的 CUDA 扩展"""
    ensure_cublaslt_loaded()
    
    lib = ctypes.CDLL(str(so_path), mode=ctypes.RTLD_GLOBAL)
    
    # 设置函数签名
    lib.cublaslt_layout_search_single.argtypes = [
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # B_ptr
        ctypes.c_void_p,   # C_ptr
        ctypes.c_int64,    # M
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_char_p,   # dtype
        ctypes.c_char_p,   # outdtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.POINTER(ctypes.c_int),        # out_layout_ids
        ctypes.c_char_p,                     # out_layout_names
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_uint8),      # out_valid
        ctypes.POINTER(ctypes.c_int),        # out_num_valid
        ctypes.c_void_p,   # stream
    ]
    lib.cublaslt_layout_search_single.restype = ctypes.c_int
    
    lib.cublaslt_layout_search_is_available.argtypes = []
    lib.cublaslt_layout_search_is_available.restype = ctypes.c_int
    
    lib.cublaslt_layout_search_get_last_error.argtypes = []
    lib.cublaslt_layout_search_get_last_error.restype = ctypes.c_char_p
    
    lib.cublaslt_layout_get_name.argtypes = [ctypes.c_int]
    lib.cublaslt_layout_get_name.restype = ctypes.c_char_p
    
    lib.cublaslt_layout_get_count.argtypes = []
    lib.cublaslt_layout_get_count.restype = ctypes.c_int
    
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
    A_bf16: torch.Tensor,
    B_bf16: torch.Tensor,
    dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """准备量化后的数据"""
    if dtype == "int8":
        A_q, _ = quantize_int8(A_bf16)
        B_q, _ = quantize_int8(B_bf16)
    elif dtype == "fp8e4m3":
        A_q = to_fp8_e4m3(A_bf16)
        B_q = to_fp8_e4m3(B_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    return A_q, B_q


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    A_q: torch.Tensor,
    B_q: torch.Tensor,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的所有布局"""
    # 分配输出缓冲
    C_torch_dtype = torch.float32 if outdtype == "fp32" else torch.bfloat16
    C_out = torch.zeros(M, N, dtype=C_torch_dtype, device=A_q.device)
    
    # 分配输出数组
    out_layout_ids = (ctypes.c_int * NUM_LAYOUTS)()
    out_layout_names = ctypes.create_string_buffer(NUM_LAYOUTS * 32)
    out_lat_us = (ctypes.c_float * NUM_LAYOUTS)()
    out_tops = (ctypes.c_float * NUM_LAYOUTS)()
    out_workspace = (ctypes.c_int64 * NUM_LAYOUTS)()
    out_valid = (ctypes.c_uint8 * NUM_LAYOUTS)()
    out_num_valid = ctypes.c_int(0)
    
    # 调用 C 函数
    ret = lib.cublaslt_layout_search_single(
        A_q.data_ptr(),
        B_q.data_ptr(),
        C_out.data_ptr(),
        M, N, K,
        dtype.encode(),
        outdtype.encode(),
        warmup,
        repeat,
        out_layout_ids,
        out_layout_names,
        out_lat_us,
        out_tops,
        out_workspace,
        out_valid,
        ctypes.byref(out_num_valid),
        None,
    )
    
    if ret != 0:
        error = lib.cublaslt_layout_search_get_last_error()
        raise RuntimeError(f"搜索失败: {error.decode() if error else 'unknown error'}")
    
    # 转换结果
    results = []
    for i in range(NUM_LAYOUTS):
        name_bytes = out_layout_names.raw[i*32:(i+1)*32]
        name = name_bytes.split(b'\x00')[0].decode('utf-8')
        
        results.append({
            "layout_id": out_layout_ids[i],
            "layout_name": name if name else LAYOUT_NAMES[i],
            "lat_us": out_lat_us[i],
            "tops": out_tops[i],
            "workspace": out_workspace[i],
            "valid": bool(out_valid[i]),
        })
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    outdtype: str,
    nk_list: List[Tuple[int, int]],
    m_list: List[int],
    warmup: int,
    repeat: int,
    verbose: bool = True,
) -> Dict:
    """运行完整的布局搜索"""
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        
        # 量化
        A_q, B_q = prepare_data(A, B, dtype)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": {},
        }
        
        for M in m_list:
            A_slice = A_q[:M].contiguous()
            
            out = search_single_nk(
                lib, N, K, M,
                A_slice, B_q,
                dtype, outdtype,
                warmup, repeat,
            )
            
            nk_results["m_results"][M] = out
        
        if verbose:
            first_m = m_list[0]
            first_result = nk_results["m_results"][first_m]
            valid_layouts = [r for r in first_result["results"] if r["valid"]]
            if valid_layouts:
                best = max(valid_layouts, key=lambda x: x["tops"])
                print(f"      → 最优布局: {best['layout_name']}, {best['tops']:.2f} TOPS")
        
        results.append(nk_results)
        
        del A, B, A_q, B_q
    
    torch.cuda.empty_cache()
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
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
    subdir_name = build_output_dir_name(model_name, dtype, outdtype)
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / build_result_filename("layout_search_bench", model_name, "csv")
    json_path = subdir / build_result_filename("layout_search_summary", model_name, "json")
    
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
        layout="SEARCH",
        alg_count=NUM_LAYOUTS,
        config_count=NUM_LAYOUTS,
    )
    
    csv_lines = list(header_lines)
    csv_lines.append("M,N,K,layout,tops,lat_us,workspace,valid")
    
    csv_rows = []
    
    for nk_res in search_ret["results"]:
        N, K = nk_res["N"], nk_res["K"]
        
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            for r in results:
                values = [
                    str(M), str(N), str(K),
                    r["layout_name"],
                    f"{r['tops']:.6f}",
                    f"{r['lat_us']:.3f}",
                    str(r["workspace"]),
                    "1" if r["valid"] else "0",
                ]
                csv_rows.append((M, ",".join(values)))
    
    csv_rows.sort(key=lambda x: x[0])
    for _, line in csv_rows:
        csv_lines.append(line)
    
    csv_path.write_text("\n".join(csv_lines))
    
    # === JSON 汇总 ===
    layout_stats = {name: {"wins": 0, "total_tops": 0.0, "count": 0} for name in LAYOUT_NAMES}
    
    for nk_res in search_ret["results"]:
        for M in search_ret["M_list"]:
            m_res = nk_res["m_results"].get(M, {})
            results = m_res.get("results", [])
            
            valid_results = [r for r in results if r["valid"]]
            if valid_results:
                best = max(valid_results, key=lambda x: x["tops"])
                layout_stats[best["layout_name"]]["wins"] += 1
            
            for r in results:
                if r["valid"]:
                    layout_stats[r["layout_name"]]["total_tops"] += r["tops"]
                    layout_stats[r["layout_name"]]["count"] += 1
    
    # 计算平均
    for name in LAYOUT_NAMES:
        s = layout_stats[name]
        s["avg_tops"] = s["total_tops"] / s["count"] if s["count"] > 0 else 0.0
    
    # 排序
    ranked = sorted(layout_stats.items(), key=lambda x: x[1]["wins"], reverse=True)
    
    summary = {
        "model": model_name,
        "dtype": dtype,
        "outdtype": outdtype,
        "gpu": hw_info.gpu_full_name,
        "layout_ranking": [
            {
                "layout": name,
                "wins": stats["wins"],
                "avg_tops": stats["avg_tops"],
            }
            for name, stats in ranked
        ],
        "recommendation": ranked[0][0] if ranked else "TN_ColCol",
    }
    
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="cuBLASLt 布局离线搜索")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="输入数据类型")
    p.add_argument("--outdtype", default="bf16", choices=SUPPORTED_OUTDTYPES, help="输出数据类型")
    p.add_argument("--model", default="BitNet-2B4T", help="模型名称或路径")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录")
    p.add_argument("--m_list", type=str, default=None, help="M 列表，逗号分隔")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    
    model_name = build_model_name_with_dtype(args.model.split('/')[-1], args.dtype)
    
    print("=" * 60)
    print("cuBLASLt 布局离线搜索")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag}, {hw_info.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}")
    print()
    
    out_dir = Path(args.out_dir) if args.out_dir else Path("./layout_search_results")
    
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "layout_search_cublaslt.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_cuda_extension(src_path, build_dir, force=args.compile)
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_extension(so_path)
    
    if not lib.cublaslt_layout_search_is_available():
        raise RuntimeError("cuBLASLt 不可用")
    print("✓ cuBLASLt 可用")
    
    nk_list = get_nk_list_auto(args.model, with_names=False)
    
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    else:
        m_list = default_m_list()
    
    print()
    print(f"[3/4] 开始布局搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print(f"      布局数量: {NUM_LAYOUTS}")
    print()
    
    ret = run_search(
        lib,
        args.dtype,
        args.outdtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
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
