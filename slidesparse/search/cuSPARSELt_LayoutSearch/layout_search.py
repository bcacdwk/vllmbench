#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
cuSPARSELt 布局离线搜索

架构说明：
=========
测试 16 种布局组合 (2:4 稀疏矩阵乘法):
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
from typing import Dict, List, Tuple, Any

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
    quantize_int8,
    to_fp8_e4m3,
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
    "TT_RowCol_DCol", "TN_RowCol_DCol", "NT_RowCol_DCol", "NN_RowCol_DCol",
    "TT_ColCol_DCol", "TN_ColCol_DCol", "NT_ColCol_DCol", "NN_ColCol_DCol",
    # D 输出为 RowMajor (后8种)
    "TT_RowCol_DRow", "TN_RowCol_DRow", "NT_RowCol_DRow", "NN_RowCol_DRow",
    "TT_ColCol_DRow", "TN_ColCol_DRow", "NT_ColCol_DRow", "NN_ColCol_DRow",
]


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名"""
    lib.cusparselt_layout_search_single.argtypes = [
        ctypes.c_void_p,   # A_compressed_ptr
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
    lib.cusparselt_layout_search_single.restype = ctypes.c_int
    
    lib.cusparselt_layout_prune_24.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cusparselt_layout_prune_24.restype = ctypes.c_int
    
    lib.cusparselt_layout_compress.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.cusparselt_layout_compress.restype = ctypes.c_int64
    
    lib.cusparselt_layout_get_compressed_size.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.cusparselt_layout_get_compressed_size.restype = ctypes.c_int64
    
    lib.cusparselt_layout_search_is_available.argtypes = []
    lib.cusparselt_layout_search_is_available.restype = ctypes.c_int
    
    lib.cusparselt_layout_search_get_last_error.argtypes = []
    lib.cusparselt_layout_search_get_last_error.restype = ctypes.c_char_p
    
    lib.cusparselt_layout_get_name.argtypes = [ctypes.c_int]
    lib.cusparselt_layout_get_name.restype = ctypes.c_char_p
    
    lib.cusparselt_layout_get_count.argtypes = []
    lib.cusparselt_layout_get_count.restype = ctypes.c_int


# =============================================================================
# 数据准备 (cuSPARSELt 特定的压缩流程)
# =============================================================================

def prepare_and_prune_weight(
    lib: ctypes.CDLL,
    A_bf16: torch.Tensor,
    dtype: str,
    order: int = 0,  # 0=COL, 1=ROW
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备并剪枝稀疏矩阵 A (权重)。
    
    返回:
        (A_pruned, A_compressed)
    """
    rows, cols = A_bf16.shape
    
    # 量化
    if dtype == "int8":
        A_q, _ = quantize_int8(A_bf16)
    elif dtype == "fp8e4m3":
        A_q = to_fp8_e4m3(A_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    A_q = A_q.contiguous()
    
    # Prune 2:4
    A_pruned = torch.empty_like(A_q)
    ret = lib.cusparselt_layout_prune_24(
        A_q.data_ptr(),
        A_pruned.data_ptr(),
        rows, cols,
        dtype.encode(),
        order,
        None,
    )
    if ret != 0:
        error = lib.cusparselt_layout_search_get_last_error()
        raise RuntimeError(f"Prune 失败: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    # 获取压缩大小
    compressed_size = lib.cusparselt_layout_get_compressed_size(rows, cols, dtype.encode(), order)
    if compressed_size < 0:
        raise RuntimeError("获取压缩大小失败")
    
    # 压缩
    A_compressed = torch.empty(compressed_size, dtype=torch.uint8, device=A_q.device)
    ret = lib.cusparselt_layout_compress(
        A_pruned.data_ptr(),
        A_compressed.data_ptr(),
        rows, cols,
        dtype.encode(),
        order,
        None,
    )
    if ret < 0:
        error = lib.cusparselt_layout_search_get_last_error()
        raise RuntimeError(f"Compress 失败: {error.decode() if error else 'unknown'}")
    
    torch.cuda.synchronize()
    
    return A_pruned, A_compressed


def prepare_activation(
    B_bf16: torch.Tensor,
    dtype: str,
) -> torch.Tensor:
    """准备稠密矩阵 B (激活)"""
    if dtype == "int8":
        B_q, _ = quantize_int8(B_bf16)
    elif dtype == "fp8e4m3":
        B_q = to_fp8_e4m3(B_bf16)
    else:
        raise ValueError(f"不支持的数据类型: {dtype}")
    
    return B_q.contiguous()


# =============================================================================
# 搜索核心
# =============================================================================

def search_single_nk(
    lib: ctypes.CDLL,
    N: int, K: int, M: int,
    A_compressed: torch.Tensor,
    B_q: torch.Tensor,
    dtype: str,
    outdtype: str,
    warmup: int,
    repeat: int,
) -> Dict[str, Any]:
    """搜索单个 (N, K, M) 组合的所有布局"""
    # 分配输出缓冲
    C_torch_dtype = get_output_torch_dtype(outdtype)
    C_out = torch.zeros(N, M, dtype=C_torch_dtype, device=B_q.device)
    
    # 分配输出数组
    out_layout_ids = (ctypes.c_int * NUM_LAYOUTS)()
    out_layout_names = ctypes.create_string_buffer(NUM_LAYOUTS * 32)
    out_lat_us = (ctypes.c_float * NUM_LAYOUTS)()
    out_tops = (ctypes.c_float * NUM_LAYOUTS)()
    out_workspace = (ctypes.c_int64 * NUM_LAYOUTS)()
    out_valid = (ctypes.c_uint8 * NUM_LAYOUTS)()
    out_num_valid = ctypes.c_int(0)
    
    # 调用 C 函数
    ret = lib.cusparselt_layout_search_single(
        A_compressed.data_ptr(),
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
        error = lib.cusparselt_layout_search_get_last_error()
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
        # A: 稀疏矩阵 (N x K 或 K x N 取决于布局)
        # 为了简化，我们使用固定的 TN_ColCol 布局进行压缩
        # A: K x N (列主序)
        A = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, max_M, device="cuda", dtype=torch.bfloat16)
        
        # 剪枝并压缩 A (使用列主序)
        A_pruned, A_compressed = prepare_and_prune_weight(lib, A, dtype, order=0)
        
        # 准备 B
        B_q = prepare_activation(B, dtype)
        
        nk_results = {
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "m_results": {},
        }
        
        for M in m_list:
            B_slice = B_q[:, :M].contiguous()
            
            out = search_single_nk(
                lib, N, K, M,
                A_compressed, B_slice,
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
            else:
                print(f"      → 无有效布局")
        
        results.append(nk_results)
        
        del A, B, A_pruned, A_compressed, B_q
    
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
    p = argparse.ArgumentParser(description="cuSPARSELt 布局离线搜索")
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
    print("cuSPARSELt 布局离线搜索 (2:4 稀疏)")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag}, {hw_info.arch_name})")
    print(f"模型: {model_name}")
    print(f"参数: dtype={args.dtype}, outdtype={args.outdtype}")
    print()
    
    out_dir = Path(args.out_dir) if args.out_dir else Path("./layout_search_results")
    
    print("[1/4] 编译 CUDA 扩展...")
    src_path = SCRIPT_DIR / "layout_search_cusparselt.cu"
    build_dir = SCRIPT_DIR / "build"
    so_path = build_search_extension(
        name="layout_search_cusparselt",
        source_file=src_path,
        build_dir=build_dir,
        backend="cusparselt",
        force=args.compile,
    )
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_search_extension(so_path, backend="cusparselt", setup_func=setup_lib_signatures)
    
    if not lib.cusparselt_layout_search_is_available():
        raise RuntimeError("cuSPARSELt 不可用")
    print("✓ cuSPARSELt 可用")
    
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
        is_sparse=True,
    )
    
    print()
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
