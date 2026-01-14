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
import ctypes
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch

# 添加 search 目录到路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SEARCH_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SEARCH_DIR))

from utils import (
    # 硬件信息
    hw_info,
    normalize_dtype,
    # 编译与加载
    build_search_extension,
    load_search_extension,
    # 模型 NK 工具
    get_nk_list_auto,
    build_model_name_with_dtype,
    # 数据准备
    quantize_tensor,
    get_output_torch_dtype,
    # 结果保存
    save_alg_search_results,
    # 验证
    verify_gemm_result,
    # dtype 检测
    SUPPORTED_DTYPES,
    SUPPORTED_OUTDTYPES,
    # 默认配置
    default_m_list,
)


# =============================================================================
# CUDA 扩展加载
# =============================================================================

def setup_lib_signatures(lib: ctypes.CDLL) -> None:
    """设置 CUDA 扩展的函数签名。"""
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
    verify: bool = False,
    W_q_for_verify: Optional[torch.Tensor] = None,
    A_q_for_verify: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    搜索单个 (N, K, M) 组合的最佳算法。
    """
    # 分配输出缓冲
    R_torch_dtype = get_output_torch_dtype(outdtype)
    # Column Major [N, M] 在 PyTorch Row Major 中存储为 [M, N]
    R_out = torch.zeros(M, N, dtype=R_torch_dtype, device=W_q.device)
    
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
        R_out.data_ptr(),
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
    
    # 验证正确性
    verify_result = None
    if verify and W_q_for_verify is not None and A_q_for_verify is not None:
        verify_result = verify_gemm_result(
            W_q=W_q_for_verify,
            A_q=A_q_for_verify,
            R_out=R_out,
            M=M,
            is_col_major=True,  # cuBLASLt AlgSearch 固定使用 Column Major
        )
        if verify_result["critical"]:
            print(f"    [CRITICAL] M={M}: {verify_result['message']}")
        elif not verify_result["passed"]:
            print(f"    [WARN] M={M}: {verify_result['message']}")
    
    return {
        "results": results,
        "num_valid": out_num_valid.value,
        "alg_count": out_alg_count.value,
        "verify_result": verify_result,
    }


def run_search(
    lib: ctypes.CDLL,
    dtype: str,
    outdtype: str,
    nk_list: List,
    m_list: List[int],
    warmup: int,
    repeat: int,
    topk: int = 3,
    verify: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    运行完整的算法搜索。
    """
    results = []
    max_M = max(m_list)
    total_nk = len(nk_list)
    
    max_alg_count = 0
    
    # verify 统计
    verify_stats = {"total": 0, "passed": 0, "warned": 0, "critical": 0}
    
    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据
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
            # 切片
            A_slice = A_q[:M].contiguous()
            # verify 用的 A_q 切片
            A_q_slice = A_slice if verify else None
            
            out = search_single_nk(
                lib, N, K, M,
                W_q, A_slice,
                dtype, outdtype,
                warmup, repeat, topk,
                verify=verify,
                W_q_for_verify=W_q if verify else None,
                A_q_for_verify=A_q_slice,
            )
            
            nk_results["m_results"][M] = out
            
            if out["alg_count"] > max_alg_count:
                max_alg_count = out["alg_count"]
            
            # 更新 verify 统计
            if verify and out.get("verify_result"):
                vr = out["verify_result"]
                verify_stats["total"] += 1
                if vr["critical"]:
                    verify_stats["critical"] += 1
                elif vr["passed"]:
                    verify_stats["passed"] += 1
                else:
                    verify_stats["warned"] += 1
        
        if verbose:
            first_m = m_list[0]
            first_result = nk_results["m_results"][first_m]
            print(f"      → 启发式返回: {first_result['alg_count']} 算法, 有效: {first_result['num_valid']}")
        
        results.append(nk_results)
        
        # 释放
        del W, A, W_q, A_q
    
    torch.cuda.empty_cache()
    
    # 打印 verify 汇总
    if verify and verbose:
        print()
        print(f"    验证统计: 总计={verify_stats['total']}, "
              f"通过={verify_stats['passed']}, "
              f"警告={verify_stats['warned']}, "
              f"严重错误={verify_stats['critical']}")
    
    return {
        "dtype": dtype,
        "outdtype": outdtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
        "max_alg_count": max_alg_count,
    }


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
    so_path = build_search_extension(
        name="alg_search_cublaslt",
        source_file=src_path,
        build_dir=build_dir,
        backend="cublaslt",
        force=args.compile,
    )
    
    print("[2/4] 加载 CUDA 扩展...")
    lib = load_search_extension(so_path, backend="cublaslt", setup_func=setup_lib_signatures)
    
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
        verify=args.verify,
        verbose=True,
    )
    
    saved_dir = save_alg_search_results(
        out_dir,
        model_name,
        args.dtype,
        args.outdtype,
        ret,
        args.warmup,
        args.repeat,
        args.verify,
        layout="TNCCcol",
        is_sparse=False,
        has_split_k=False,
    )
    
    print()
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
