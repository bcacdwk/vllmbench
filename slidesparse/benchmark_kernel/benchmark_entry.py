#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark 统一入口

测试 cuBLASLt (Dense) vs cuSPARSELt (Sparse) 的性能差异

两种运行模式:
=============
1. Model-based 模式: 指定 --model <model_name>，从模型 checkpoint 提取真实 NK 尺寸
   - 支持 --Lmax 参数：自动生成 L=4,6,...,Lmax 的所有 slided NK
2. Square 模式: --model 不指定或指定为 "square"，M=N=K=[64, 128, ..., 16384]

核心功能:
=========
- 对 cuBLASLt dense GEMM 和 cuSPARSELt sparse GEMM 进行算法搜索
- 支持多种稀疏度配置：
  - --Lmax: 自动生成 2_4, 2_6, ..., 2_Lmax 以及 2_inf（推荐）
  - --sparsity: 手动指定稀疏度列表（兼容旧用法）
- 计算并输出加速比 (Sparse / Dense)

支持的数据类型:
===============
- FP16:    FP16 输入, FP32 计算, FP16 输出
- BF16:    BF16 输入, FP32 计算, BF16 输出
- INT8:    INT8 输入, INT32 计算, INT8 输出
- FP8E4M3: FP8 输入, FP32 计算, FP8 输出
- FP4E2M1: FP4 输入, FP32 计算, FP4 输出
- all:     测试所有支持的类型

Sparsity 参数说明:
==================
格式: {Z}_{L} 表示每 L 个元素保留 Z 个非零值
- 2_4:  K_factor = 1.00, 标准 2:4 稀疏 (50% 非零)
- 2_6:  K_factor = 1.33 (33% 非零)
- 2_8:  K_factor = 1.50 (25% 非零)
- 2_10: K_factor = 1.60 (20% 非零)
- 2_inf: K_factor = 2.00 (理论上限, 全稀疏)

用法示例:
=========
# Model-based 模式（推荐使用 --Lmax）
python benchmark_entry.py --model Qwen2.5-0.5B --dtype fp8e4m3 --Lmax 8
# 等效于 --sparsity 2_4,2_6,2_8,2_inf

# Square 模式 (不指定 --model)
python benchmark_entry.py --dtype all -- --Lmax 10

# 测试所有 dtype（使用默认 sparsity=2_4）
python benchmark_entry.py --model square --dtype all

# 手动指定稀疏度（兼容旧用法）
python benchmark_entry.py --model Llama3.2-1B --dtype fp8e4m3 --sparsity 2_4,2_8

# 只测试 cuBLASLt (无需 sparsity)
python benchmark_entry.py --model Llama3.2-1B --dtype all --backend cublaslt


python benchmark_entry.py --model Llama3.2-1B --M-quick --dtype all --backend cusparselt --sparsity 2_4,2_6

"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 添加路径
SCRIPT_DIR = Path(__file__).parent.absolute()
SLIDESPARSE_DIR = SCRIPT_DIR.parent
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
    # Sparsity 计算
    calculate_k_slide,
    get_k_expansion_factor,
    pad_to_alignment,
    get_sparsity_list_for_benchmark,
    # NK 列表
    get_nk_list_for_benchmark,
    # 文件命名与目录
    build_hw_folder_name,
    build_dtype_folder_name,
    build_output_dir,
    build_result_filename,
    # 结果整合
    compute_speedup,
    merge_benchmark_results,
)


# =============================================================================
# 子脚本路径
# =============================================================================

CUBLASLT_SCRIPT = SCRIPT_DIR / "cuBLASLt" / "alg_search.py"
CUSPARSELT_SCRIPT = SCRIPT_DIR / "cuSPARSELt" / "alg_search.py"


# =============================================================================
# 工具函数
# =============================================================================

def run_cublaslt_search(
    dtype: str,
    model: str,  # 现在始终传入 model name（可以是 "square"）
    m_list: List[int],
    warmup: int,
    repeat: int,
    compile_flag: bool,
    out_dir: Path,
) -> Tuple[int, Optional[Path]]:
    """
    运行 cuBLASLt 搜索
    
    Returns:
        (return_code, json_path): 返回码和 JSON 结果文件路径
    """
    cmd = [
        sys.executable, str(CUBLASLT_SCRIPT),
        "--dtype", dtype,
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--out_dir", str(out_dir),
        "--m_list", ",".join(str(m) for m in m_list),
    ]
    
    if compile_flag:
        cmd.append("--compile")
    
    print(f"[cuBLASLt] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # 获取结果文件路径（新结构：hw_folder/dtype_folder/file）
    if result.returncode == 0:
        result_dir = build_output_dir(out_dir, dtype)
        json_filename = build_result_filename("alg_search", model.upper(), "json")
        json_path = result_dir / json_filename
        if json_path.exists():
            return result.returncode, json_path
    
    return result.returncode, None


def run_cusparselt_search(
    dtype: str,
    model: str,  # 现在始终传入 model name（可以是 "square"）
    sparsity: str,
    m_list: List[int],
    warmup: int,
    repeat: int,
    compile_flag: bool,
    out_dir: Path,
    no_segment_k: bool = False,
    no_api_search: bool = False,
) -> Tuple[int, Optional[Path]]:
    """
    运行 cuSPARSELt 搜索
    
    Returns:
        (return_code, json_path): 返回码和 JSON 结果文件路径
    """
    cmd = [
        sys.executable, str(CUSPARSELT_SCRIPT),
        "--dtype", dtype,
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--out_dir", str(out_dir),
        "--m_list", ",".join(str(m) for m in m_list),
        "--sparsity", sparsity,
    ]
    
    if compile_flag:
        cmd.append("--compile")
    
    if no_segment_k:
        cmd.append("--no_segment_k")
    
    if no_api_search:
        cmd.append("--no_api_search")
    
    print(f"[cuSPARSELt] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    # 获取结果文件路径（新结构：hw_folder/dtype_folder/file）
    if result.returncode == 0:
        result_dir = build_output_dir(out_dir, dtype)
        json_filename = build_result_filename("alg_search", model.upper(), "json", sparsity)
        json_path = result_dir / json_filename
        if json_path.exists():
            return result.returncode, json_path
    
    return result.returncode, None


def load_json_results(json_path: Path) -> Optional[Dict]:
    """加载 JSON 结果文件"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {json_path}: {e}")
        return None


def compute_and_save_speedup(
    cublaslt_json: Path,
    cusparselt_json: Path,
    model_name: str,
    dtype: str,
    sparsity: str,
    output_dir: Path,
) -> Optional[Path]:
    """
    计算并保存加速比结果
    
    Returns:
        保存的 CSV 文件路径
    """
    cublaslt_data = load_json_results(cublaslt_json)
    cusparselt_data = load_json_results(cusparselt_json)
    
    if not cublaslt_data or not cusparselt_data:
        print("[WARN] Failed to load results for speedup calculation")
        return None
    
    hw_folder = build_hw_folder_name()
    out_dir = output_dir / hw_folder
    out_dir.mkdir(parents=True, exist_ok=True)
    
    csv_filename = build_result_filename("speedup", model_name, "csv", sparsity)
    csv_path = out_dir / csv_filename
    
    k_factor = get_k_expansion_factor(sparsity)
    
    # 构建 CSV
    lines = [
        f"# GPU: {hw_info.gpu_full_name}",
        f"# Model: {model_name}",
        f"# dtype: {dtype.upper()}",
        f"# Sparsity: {sparsity} (K_factor={k_factor:.3f})",
        f"# Time: {datetime.datetime.now().isoformat()}",
        "M,N,K,K_slide,cublaslt_tops,cusparselt_tops,speedup,cublaslt_lat_us,cusparselt_lat_us,cublaslt_alg_id,cusparselt_alg_id,cusparselt_split_k",
    ]
    
    # 构建索引: (M, N, K) -> result
    # JSON 结构: { "nk_entries": { "(N,K)": { "alg_by_m": { "M": {...} } } } }
    cublaslt_index = {}
    for nk_key, nk_data in cublaslt_data.get("nk_entries", {}).items():
        # nk_key 格式: "(N,K)" -> 解析出 N, K
        try:
            nk_str = nk_key.strip("()")
            N, K = map(int, nk_str.split(","))
        except:
            continue
        
        for m_str, m_data in nk_data.get("alg_by_m", {}).items():
            M = int(m_str)
            if m_data:
                cublaslt_index[(M, N, K)] = m_data
    
    cusparselt_index = {}
    for nk_key, nk_data in cusparselt_data.get("nk_entries", {}).items():
        # nk_key 格式: "(N,K)"
        try:
            nk_str = nk_key.strip("()")
            N, K = map(int, nk_str.split(","))
        except:
            continue
        
        K_slide = calculate_k_slide(K, sparsity, dtype=dtype)
        for m_str, m_data in nk_data.get("alg_by_m", {}).items():
            M = int(m_str)
            if m_data:
                cusparselt_index[(M, N, K_slide)] = {
                    **m_data,
                    "orig_K": K,
                }
    
    # 生成结果行
    result_rows = []
    for (M, N, K), dense in sorted(cublaslt_index.items()):
        K_slide = calculate_k_slide(K, sparsity, dtype=dtype)
        sparse_key = (M, N, K_slide)
        
        if sparse_key in cusparselt_index:
            sparse = cusparselt_index[sparse_key]
            
            dense_tops = dense.get("tops", 0)
            sparse_tops = sparse.get("tops", 0)
            speedup = compute_speedup(dense_tops, sparse_tops)
            
            dense_lat = dense.get("lat_us", 0)
            sparse_lat = sparse.get("lat_us", 0)
            dense_alg_id = dense.get("alg_id", -1)
            sparse_alg_id = sparse.get("alg_id", -1)
            sparse_split_k = sparse.get("split_k", -1)
            
            result_rows.append({
                "M": M, "N": N, "K": K, "K_slide": K_slide,
                "dense_tops": dense_tops, "sparse_tops": sparse_tops,
                "speedup": speedup,
                "dense_lat": dense_lat, "sparse_lat": sparse_lat,
                "dense_alg_id": dense_alg_id, "sparse_alg_id": sparse_alg_id,
                "sparse_split_k": sparse_split_k,
            })
    
    # 按 M, N, K 排序
    result_rows.sort(key=lambda x: (x["M"], x["N"], x["K"]))
    
    for row in result_rows:
        line = (
            f"{row['M']},{row['N']},{row['K']},{row['K_slide']},"
            f"{row['dense_tops']:.6f},{row['sparse_tops']:.6f},{row['speedup']:.4f},"
            f"{row['dense_lat']:.3f},{row['sparse_lat']:.3f},"
            f"{row['dense_alg_id']},{row['sparse_alg_id']},{row['sparse_split_k']}"
        )
        lines.append(line)
    
    csv_path.write_text("\n".join(lines))
    
    return csv_path


def print_speedup_summary(csv_path: Path):
    """打印加速比摘要"""
    try:
        lines = csv_path.read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#") and l.strip() and "," in l]
        
        if len(data_lines) < 2:  # 至少需要 header + 1 行数据
            return
        
        header = data_lines[0].split(",")
        speedup_idx = header.index("speedup") if "speedup" in header else -1
        m_idx = header.index("M") if "M" in header else 0
        
        if speedup_idx < 0:
            return
        
        speedups = []
        m_speedups = {}
        
        for line in data_lines[1:]:
            parts = line.split(",")
            if len(parts) > speedup_idx:
                try:
                    sp = float(parts[speedup_idx])
                    m = int(parts[m_idx])
                    speedups.append(sp)
                    if m not in m_speedups:
                        m_speedups[m] = []
                    m_speedups[m].append(sp)
                except ValueError:
                    continue
        
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)
            
            print()
            print("    ┌────────────────────────────────────────┐")
            print("    │          Speedup Summary               │")
            print("    ├────────────────────────────────────────┤")
            print(f"    │  Average Speedup: {avg_speedup:>7.3f}x             │")
            print(f"    │  Max Speedup:     {max_speedup:>7.3f}x             │")
            print(f"    │  Min Speedup:     {min_speedup:>7.3f}x             │")
            print("    ├────────────────────────────────────────┤")
            print("    │  Speedup by M:                         │")
            
            for m in sorted(m_speedups.keys()):
                m_avg = sum(m_speedups[m]) / len(m_speedups[m])
                print(f"    │    M={m:<6}: {m_avg:>7.3f}x                 │")
            
            print("    └────────────────────────────────────────┘")
    except Exception as e:
        print(f"[WARN] Failed to print speedup summary: {e}")


# =============================================================================
# 主流程
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="SlideSparse Kernel Benchmark 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # Model-based 模式（推荐使用 --Lmax）
  python benchmark_entry.py --model Qwen2.5-0.5B --dtype fp8e4m3 --Lmax 8
  # 等效于 --sparsity 2_4,2_6,2_8,2_inf
  
  # Square 模式 (不指定 --model)
  python benchmark_entry.py --dtype int8 --Lmax 10
  
  # 只测试标准 2:4 稀疏（默认）
  python benchmark_entry.py --model Llama3.2-1B --dtype bf16
  
  # 只测试 cuBLASLt（无需指定 sparsity）
  python benchmark_entry.py --model Qwen2.5-0.5B --dtype all --backend cublaslt
  
  # 手动指定稀疏度（兼容旧用法）
  python benchmark_entry.py --model Llama3.2-1B --dtype fp8e4m3 --sparsity 2_4,2_8
        """
    )
    
    # 模式选择
    p.add_argument("--model", type=str, default=None,
                   help="模型名称（不指定或指定 'square' 进入 Square 模式）")
    p.add_argument("--dtype", type=str, required=True,
                   choices=SUPPORTED_DTYPES + ["all"],
                   help="数据类型 (fp16, bf16, int8, fp8e4m3, fp4e2m1, all)")
    
    # 稀疏度配置（Lmax 和 sparsity 二选一）
    sparsity_group = p.add_mutually_exclusive_group()
    sparsity_group.add_argument("--Lmax", type=int, default=None,
                   help="最大 L 值。自动生成 2_4, 2_6, ..., 2_Lmax 以及 2_inf。"
                        "例如 --Lmax 8 等效于 --sparsity 2_4,2_6,2_8,2_inf")
    sparsity_group.add_argument("--sparsity", type=str, default=None,
                   help="手动指定稀疏度列表，逗号分隔 (2_4, 2_6, 2_8, 2_10, 2_inf)")
    
    p.add_argument("--backend", type=str, default="all",
                   choices=["all", "cublaslt", "cusparselt"],
                   help="测试后端")
    
    # M 列表控制
    p.add_argument("--M-quick", action="store_true", dest="m_quick",
                   help="使用快速 M 列表 (仅 Model-based 模式有效)")
    p.add_argument("--m_list", type=str, default=None,
                   help="自定义 M 列表，逗号分隔 (Square 模式: M=N=K 使用此列表)")
    
    # 测试参数
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    
    # 编译控制
    p.add_argument("--compile", action="store_true", dest="force_compile",
                   help="强制重新编译 CUDA 扩展")
    
    # cuSPARSELt 选项
    p.add_argument("--no_segment_k", action="store_true",
                   help="禁用 Segment-K 测试")
    p.add_argument("--no_api_search", action="store_true",
                   help="禁用官方 API 搜索对比")
    
    # 输出控制
    p.add_argument("--dry-run", action="store_true",
                   help="只打印命令，不执行")
    p.add_argument("--no-speedup", action="store_true", dest="no_speedup",
                   help="不计算 speedup")
    
    return p.parse_args()


def run_benchmark_for_dtype(args, dtype: str, m_list: List[int], model_name: str, sparsity_list: List[str]):
    """为单个 dtype 运行 benchmark"""
    
    # 检查 dtype 支持
    supported, reason = check_dtype_support(dtype)
    if not supported:
        print(f"[SKIP] {dtype}: {reason}")
        return
    
    # 检查 cuSPARSELt 支持
    backend = args.backend
    if backend in ("all", "cusparselt"):
        cusparselt_ok, cusparselt_reason = check_cusparselt_support()
        if not cusparselt_ok:
            if backend == "cusparselt":
                print(f"[SKIP] {dtype}: {cusparselt_reason}")
                return
            else:
                print(f"[WARN] {cusparselt_reason}")
                print("[INFO] 将只测试 cuBLASLt")
                backend = "cublaslt"
    
    # 输出目录
    out_dir = SCRIPT_DIR
    cublaslt_json_path = None
    cusparselt_json_paths = {}
    
    # 显示配置
    print()
    print(f"{'=' * 60}")
    print(f"Benchmark dtype={dtype.upper()}")
    print(f"{'=' * 60}")
    
    # 运行 cuBLASLt
    if backend in ("all", "cublaslt"):
        print()
        print(f"[{dtype}] Running cuBLASLt Dense GEMM Search")
        
        ret, cublaslt_json_path = run_cublaslt_search(
            dtype=dtype,
            model=model_name,
            m_list=m_list,
            warmup=args.warmup,
            repeat=args.repeat,
            compile_flag=args.force_compile,
            out_dir=out_dir / "cuBLASLt" / "alg_search_results",
        )
        
        if ret != 0:
            print(f"[ERROR] cuBLASLt search failed with code {ret}")
            return
    
    # 运行 cuSPARSELt
    if backend in ("all", "cusparselt"):
        for sparsity in sparsity_list:
            print()
            print(f"[{dtype}] Running cuSPARSELt Sparse GEMM Search (sparsity={sparsity})")
            
            k_factor = get_k_expansion_factor(sparsity)
            print(f"[INFO] K_factor = {k_factor:.3f}")
            
            ret, json_path = run_cusparselt_search(
                dtype=dtype,
                model=model_name,
                sparsity=sparsity,
                m_list=m_list,
                warmup=args.warmup,
                repeat=args.repeat,
                compile_flag=args.force_compile,
                out_dir=out_dir / "cuSPARSELt" / "alg_search_results",
                no_segment_k=args.no_segment_k,
                no_api_search=args.no_api_search,
            )
            
            if ret != 0:
                print(f"[ERROR] cuSPARSELt search (sparsity={sparsity}) failed")
            else:
                cusparselt_json_paths[sparsity] = json_path
    
    # 计算 speedup
    if backend == "all" and not args.no_speedup:
        speedup_dir = out_dir / "speedup_results"
        
        for sparsity, cusparselt_json in cusparselt_json_paths.items():
            if cublaslt_json_path and cusparselt_json:
                print(f"\n  [{dtype}] Computing speedup for sparsity={sparsity}...")
                
                csv_path = compute_and_save_speedup(
                    cublaslt_json=cublaslt_json_path,
                    cusparselt_json=cusparselt_json,
                    model_name=model_name,
                    dtype=dtype,
                    sparsity=sparsity,
                    output_dir=speedup_dir,
                )
                
                if csv_path:
                    print(f"    Saved: {csv_path}")
                    print_speedup_summary(csv_path)


def main():
    args = parse_args()
    
    # 确定要测试的 dtype 列表
    if args.dtype == "all":
        dtype_list = get_supported_dtypes_for_gpu()
        print(f"[INFO] dtype=all, 将测试: {dtype_list}")
    else:
        dtype_list = [args.dtype]
    
    # 统一 m_list 处理逻辑（两种模式完全一致）
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    elif args.m_quick:
        m_list = list(M_QUICK_LIST)
    else:
        m_list = list(M_QUICK_LIST)  # 默认: [16, 128, 1024, 4096, 16384]
    
    m_list = [pad_to_alignment(m) for m in m_list]
    
    # 生成稀疏度列表：--Lmax 优先，否则使用 --sparsity，默认 ["2_4"]
    if args.Lmax is not None:
        # 使用 Lmax 自动生成稀疏度列表（与参考代码一致 + 2_inf）
        sparsity_list = get_sparsity_list_for_benchmark(args.Lmax)
        print(f"[INFO] Lmax={args.Lmax}, 稀疏度列表: {sparsity_list}")
    elif args.sparsity:
        # 手动指定稀疏度（兼容旧用法）
        sparsity_list = [s.strip() for s in args.sparsity.split(",")]
    else:
        # 默认：只测试标准 2:4 稀疏
        sparsity_list = ["2_4"]
    
    # 使用 get_nk_list_for_benchmark 统一判断模式和获取 model_name
    # 这样确保模型路径查找逻辑与子脚本一致
    is_square_mode = (args.model is None or args.model.lower() == "square")
    
    if is_square_mode:
        # Square 模式
        model_name = "SQUARE"
        mode = "square"
    else:
        # Model-based 模式：使用 get_nk_list_for_benchmark 获取正确的 model_name
        # 注意：这里我们不传入 L_max，因为 L_max 只影响稀疏度列表，不影响 model_name
        _, model_name, mode = get_nk_list_for_benchmark(
            model=args.model,
            m_list=m_list if is_square_mode else None,
        )
    
    # 显示配置
    print("=" * 60)
    print("SlideSparse Kernel Benchmark")
    print("=" * 60)
    print(f"GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"Mode: {mode.upper()}")
    print(f"Model: {model_name}")
    if is_square_mode:
        print(f"M=N=K: {m_list}")
    else:
        print(f"M_list: {m_list}")
    print(f"dtype: {dtype_list}")
    print(f"Backend: {args.backend}")
    if args.backend in ("all", "cusparselt"):
        print(f"Sparsity: {sparsity_list}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print("=" * 60)
    
    if args.dry_run:
        print("[DRY-RUN] 不执行，只显示配置")
        return
    
    # 为每个 dtype 运行 benchmark
    for dtype in dtype_list:
        run_benchmark_for_dtype(args, dtype, m_list, model_name, sparsity_list)
    
    # 输出目录
    out_dir = SCRIPT_DIR
    
    print()
    print("=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print()
    print("Results saved to:")
    if args.backend in ("all", "cublaslt"):
        print(f"  - cuBLASLt:   {out_dir / 'cuBLASLt' / 'alg_search_results'}")
    if args.backend in ("all", "cusparselt"):
        print(f"  - cuSPARSELt: {out_dir / 'cuSPARSELt' / 'alg_search_results'}")
    if args.backend == "all" and not args.no_speedup:
        print(f"  - Speedup:    {out_dir / 'speedup_results'}")


if __name__ == "__main__":
    main()
