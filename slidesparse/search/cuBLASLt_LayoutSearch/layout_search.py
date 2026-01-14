#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt 布局离线搜索

架构说明：
=========
测试 16 种布局组合:
  - 转置: TT, TN, NT, NN (4种)
  - A/B 排列: RowCol, ColCol (2种)
  - D 输出: Col, Row (2种)

固定最优布局: T/N + Col/Col + Col

运行示例:
    python3 layout_search.py --dtype int8 --outdtype bf16 --model BitNet-2B4T
"""

import argparse
import ctypes
import sys
from pathlib import Path
from typing import Dict, List, Any

import torch

# 添加 search 目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SEARCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SEARCH_DIR))

from utils import (
    hw_info,
    # 编译与加载
    build_search_extension,
    load_search_extension,
    # 模型工具
    get_nk_list_auto,
    build_model_name_with_dtype,
    # 数据准备
    quantize_tensor,
    get_output_torch_dtype,
    # 结果保存
    save_layout_search_results,
    # 常量
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
    default_m_list,
)


# =============================================================================
# 布局常量
# =============================================================================

NUM_LAYOUTS = 16

LAYOUT_NAMES = [
    # D 输出为 ColMajor (前8种)
    "TT_RowCol_Col", "TN_RowCol_Col", "NT_RowCol_Col", "NN_RowCol_Col",
    "TT_ColCol_Col", "TN_ColCol_Col", "NT_ColCol_Col", "NN_ColCol_Col",
    # D 输出为 RowMajor (后8种)
    "TT_RowCol_Row", "TN_RowCol_Row", "NT_RowCol_Row", "NN_RowCol_Row",
    "TT_ColCol_Row", "TN_ColCol_Row", "NT_ColCol_Row", "NN_ColCol_Row",
]


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名"""
    lib.cublaslt_layout_search_single.argtypes = [
        ctypes.c_void_p,   # W_ptr
        ctypes.c_void_p,   # A_ptr
        ctypes.c_void_p,   # R_ptr
        ctypes.c_int64,    # N
        ctypes.c_int64,    # K
        ctypes.c_int64,    # M
        ctypes.c_char_p,   # dtype
        ctypes.c_char_p,   # outdtype
        ctypes.c_int,      # warmup
        ctypes.c_int,      # repeat
        ctypes.POINTER(ctypes.c_int),        # out_layout_ids
        ctypes.c_char_p,                     # out_layout_names
        ctypes.POINTER(ctypes.c_float),      # out_lat_us
        ctypes.POINTER(ctypes.c_float),      # out_tops
        ctypes.POINTER(ctypes.c_int64),      # out_workspace
        ctypes.POINTER(ctypes.c_int),        # out_best_alg_id
        ctypes.POINTER(ctypes.c_float),      # out_waves_count
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
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的所有布局"""
    # 分配输出缓冲
    R_torch_dtype = get_output_torch_dtype(outdtype)
    # 输出 R[N, M]，Column Major 在 PyTorch Row Major 中存储为 [M, N]
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
    # 分配输出数组
    out_layout_ids = (ctypes.c_int * NUM_LAYOUTS)()
    out_layout_names = ctypes.create_string_buffer(NUM_LAYOUTS * 32)
    out_lat_us = (ctypes.c_float * NUM_LAYOUTS)()
    out_tops = (ctypes.c_float * NUM_LAYOUTS)()
    out_workspace = (ctypes.c_int64 * NUM_LAYOUTS)()
    out_best_alg_id = (ctypes.c_int * NUM_LAYOUTS)()
    out_waves_count = (ctypes.c_float * NUM_LAYOUTS)()
    out_valid = (ctypes.c_uint8 * NUM_LAYOUTS)()
    out_num_valid = ctypes.c_int(0)
    
    # 调用 C 函数
    ret = lib.cublaslt_layout_search_single(
        W_q.data_ptr(),
        A_q.data_ptr(),
        R_out.data_ptr(),
        N, K, M,
        dtype.encode(),
        outdtype.encode(),
        warmup,
        repeat,
        out_layout_ids,
        out_layout_names,
        out_lat_us,
        out_tops,
        out_workspace,
        out_best_alg_id,
        out_waves_count,
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
            "best_alg_id": out_best_alg_id[i],
            "waves_count": out_waves_count[i],
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
    nk_list: List,
    m_list: List[int],
    warmup: int,
    repeat: int,
    verbose: bool = True,
) -> Dict:
    """运行完整的布局搜索"""
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    for nk_id, nk in enumerate(nk_list):
        N, K = nk[0], nk[1]
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据
        # W[N, K]: 权重矩阵
        # A[M, K]: 激活矩阵 (max_M 用于预分配，后续按需切片)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)
        
        # 量化
        W_q = quantize_tensor(W, dtype)
        A_q = quantize_tensor(A, dtype)
        
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
                W_q, A_slice,
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
                print(f"      → 最优布局: {best['layout_name']}, {best['tops']:.2f} TOPS, alg={best['best_alg_id']}")
            else:
                print(f"      → 无有效布局")
        
        results.append(nk_results)
        
        del W, A, W_q, A_q
    
    torch.cuda.empty_cache()
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
    }


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
    so_path = build_search_extension(
        name="layout_search_cublaslt",
        source_file=src_path,
        build_dir=build_dir,
        backend="cublaslt",
        force=args.compile,
    )
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_search_extension(so_path, backend="cublaslt", setup_func=setup_lib_signatures)
    
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
    
    saved_dir = save_layout_search_results(
        out_dir,
        model_name,
        args.dtype,
        args.outdtype,
        ret,
        args.warmup,
        args.repeat,
        args.verify,
        layout_names=LAYOUT_NAMES,
        is_sparse=False,
    )
    
    print()
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
