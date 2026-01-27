#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark 结果提取脚本

从 cuBLASLt 和 cuSPARSELt 的 alg_search_results 中提取 TOPS 数据，
计算加速比 (Sparse / Dense)，生成汇总 CSV。

输出目录结构:
    kernel_speedup_results/
    ├── {hw_folder}/
    │   ├── model/
    │   │   ├── {dtype}/
    │   │   │   ├── absolute_tops_Llama3.2-1B-INT8.csv
    │   │   │   ├── speedup_Llama3.2-1B-INT8_2_4.csv
    │   │   │   ├── speedup_Llama3.2-1B-INT8_2_6.csv
    │   │   │   └── ...
    │   │   └── ...
    │   └── square/
    │       └── {dtype}/
    │           └── ...
    └── ...

Usage:
    # 提取所有结果
    python3 extract_kernel_results.py
    
    # 只提取特定 dtype
    python3 extract_kernel_results.py --dtype fp8e4m3
    
    # 只提取特定模型
    python3 extract_kernel_results.py --model Llama3.2-1B-INT8
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# =============================================================================
# 路径设置
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.tools.utils import (
    Colors,
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
)

from slidesparse.benchmark_kernel.utils import (
    build_hw_folder_name,
    build_dtype_folder_name,
    build_result_filename,
    calculate_k_slide,
    get_k_expansion_factor,
    hw_info,
)


# =============================================================================
# 配置常量
# =============================================================================

# 8个模型
MODELS = [
    "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
    "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
    "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
]

# 5种精度
DTYPES = ["fp16", "bf16", "int8", "fp8e4m3", "fp4e2m1"]

# 稀疏度列表 (高稀疏 + 低稀疏)
SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10", "2_12", "2_14", "2_16", "2_inf"]

# M 值列表
M_LIST = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# 源数据目录
CUBLASLT_DIR = _SCRIPT_DIR / "cuBLASLt" / "alg_search_results"
CUSPARSELT_DIR = _SCRIPT_DIR / "cuSPARSELt" / "alg_search_results"

# 输出目录
OUTPUT_DIR = _SCRIPT_DIR / "kernel_speedup_results"


# =============================================================================
# 工具函数
# =============================================================================

def load_json_results(json_path: Path) -> Optional[Dict]:
    """加载 JSON 结果文件"""
    if not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def build_index_from_json(data: Dict) -> Dict[Tuple[int, int, int], Dict]:
    """
    从 JSON 数据构建索引
    
    Returns:
        {(M, N, K): result_dict}
    """
    index = {}
    for nk_key, nk_data in data.get("nk_entries", {}).items():
        try:
            nk_str = nk_key.strip("()")
            N, K = map(int, nk_str.split(","))
        except:
            continue
        
        for m_str, m_data in nk_data.get("alg_by_m", {}).items():
            M = int(m_str)
            if m_data:
                index[(M, N, K)] = m_data
    
    return index


def find_cublaslt_json(hw_folder: str, model_name: str, dtype: str) -> Optional[Path]:
    """查找 cuBLASLt JSON 文件"""
    dtype_folder = build_dtype_folder_name(dtype)
    json_filename = build_result_filename("alg_search", model_name, "json")
    
    json_path = CUBLASLT_DIR / hw_folder / dtype_folder / json_filename
    return json_path if json_path.exists() else None


def find_cusparselt_json(hw_folder: str, model_name: str, dtype: str, sparsity: str) -> Optional[Path]:
    """查找 cuSPARSELt JSON 文件"""
    dtype_folder = build_dtype_folder_name(dtype)
    json_filename = build_result_filename("alg_search", model_name, "json", sparsity)
    
    json_path = CUSPARSELT_DIR / hw_folder / dtype_folder / json_filename
    return json_path if json_path.exists() else None


@dataclass
class ExtractedRow:
    """一行提取的数据"""
    m_value: int
    n_value: int
    k_value: int
    cublas_tops: Optional[float]
    cublas_lat_us: Optional[float]
    cusparse_tops: Dict[str, Optional[float]]  # sparsity -> tops
    cusparse_lat_us: Dict[str, Optional[float]]  # sparsity -> lat_us
    k_slide_map: Dict[str, int]  # sparsity -> K_slide


def extract_model_data(
    hw_folder: str,
    model_name: str,
    dtype: str,
    sparsity_list: List[str],
) -> List[ExtractedRow]:
    """
    提取单个模型的所有数据
    
    Returns:
        ExtractedRow 列表
    """
    # 加载 cuBLASLt 数据
    cublas_json_path = find_cublaslt_json(hw_folder, model_name, dtype)
    cublas_data = load_json_results(cublas_json_path) if cublas_json_path else None
    cublas_index = build_index_from_json(cublas_data) if cublas_data else {}
    
    # 加载 cuSPARSELt 数据（每个稀疏度）
    cusparse_indices = {}
    for sp in sparsity_list:
        sp_json_path = find_cusparselt_json(hw_folder, model_name, dtype, sp)
        sp_data = load_json_results(sp_json_path) if sp_json_path else None
        if sp_data:
            cusparse_indices[sp] = build_index_from_json(sp_data)
    
    if not cublas_index and not cusparse_indices:
        return []
    
    # 收集所有唯一的 (M, N, K) 组合（从 cuBLASLt）
    rows = []
    for (M, N, K), cublas_result in sorted(cublas_index.items()):
        cublas_tops = cublas_result.get("tops")
        cublas_lat = cublas_result.get("lat_us")
        
        # 收集各稀疏度的结果
        cusparse_tops = {}
        cusparse_lat = {}
        k_slide_map = {}
        
        for sp in sparsity_list:
            K_slide = calculate_k_slide(K, sp, dtype=dtype)
            k_slide_map[sp] = K_slide
            
            sp_index = cusparse_indices.get(sp, {})
            sp_key = (M, N, K_slide)
            
            if sp_key in sp_index:
                sp_result = sp_index[sp_key]
                cusparse_tops[sp] = sp_result.get("tops")
                cusparse_lat[sp] = sp_result.get("lat_us")
            else:
                cusparse_tops[sp] = None
                cusparse_lat[sp] = None
        
        row = ExtractedRow(
            m_value=M,
            n_value=N,
            k_value=K,
            cublas_tops=cublas_tops,
            cublas_lat_us=cublas_lat,
            cusparse_tops=cusparse_tops,
            cusparse_lat_us=cusparse_lat,
            k_slide_map=k_slide_map,
        )
        rows.append(row)
    
    return rows


def write_absolute_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入绝对延迟 CSV（单位：微秒）
    
    注意：TOPS 在有 K_slide 的情况下没有意义，因为 cuBLAS 和 cuSPARSE 计算的操作数不同。
    因此我们使用 latency 作为绝对指标。
    
    Returns:
        有效数据行数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # 表头
        header = ["M", "N", "K", "cuBLAS_lat_us"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}_lat_us")
        f.write(",".join(header) + "\n")
        
        for row in rows:
            values = [
                str(row.m_value),
                str(row.n_value),
                str(row.k_value),
                f"{row.cublas_lat_us:.3f}" if row.cublas_lat_us is not None else "",
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_lat_us.get(sp)
                values.append(f"{sp_lat:.3f}" if sp_lat is not None else "")
            
            f.write(",".join(values) + "\n")
            
            if row.cublas_lat_us is not None:
                valid_count += 1
    
    return valid_count


def write_speedup_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入加速比 CSV（所有稀疏度在一个文件中）
    
    格式: M, N, K, cuBLAS (=1.00), cuSPARSE_2_4_speedup, cuSPARSE_2_6_speedup, ...
    
    加速比 = cuBLAS_latency / cuSPARSE_latency
    （latency 越小越好，所以 cuBLAS 除以 cuSPARSE）
    
    Returns:
        有效数据行数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # 表头
        header = ["M", "N", "K", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in rows:
            if row.cublas_lat_us is None or row.cublas_lat_us <= 0:
                continue
            
            values = [
                str(row.m_value),
                str(row.n_value),
                str(row.k_value),
                "1.00",  # cuBLAS 作为基准 = 1.00
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_lat_us.get(sp)
                if sp_lat is not None and sp_lat > 0:
                    # 加速比 = cuBLAS_lat / cuSPARSE_lat（latency 越小越好）
                    speedup = row.cublas_lat_us / sp_lat
                    values.append(f"{speedup:.2f}")
                else:
                    values.append("")
            
            f.write(",".join(values) + "\n")
            valid_count += 1
    
    return valid_count


def process_model(
    hw_folder: str,
    model_name: str,
    dtype: str,
    sparsity_list: List[str],
    output_base: Path,
    is_square: bool = False,
) -> bool:
    """
    处理单个模型的所有数据
    
    Returns:
        是否有有效数据
    """
    rows = extract_model_data(hw_folder, model_name, dtype, sparsity_list)
    
    if not rows:
        return False
    
    # 输出目录
    mode = "square" if is_square else "model"
    dtype_folder = build_dtype_folder_name(dtype)
    output_dir = output_base / hw_folder / mode / dtype_folder
    
    # 写入绝对延迟 CSV（单位：微秒）
    lat_csv = output_dir / f"absolute_latency_{model_name}.csv"
    lat_count = write_absolute_csv(rows, lat_csv, sparsity_list)
    
    # 写入加速比 CSV（所有稀疏度在同一个文件）
    speedup_csv = output_dir / f"speedup_{model_name}.csv"
    speedup_count = write_speedup_csv(rows, speedup_csv, sparsity_list)
    
    return lat_count > 0 or speedup_count > 0


def find_available_hw_folders() -> List[str]:
    """查找可用的硬件文件夹"""
    hw_folders = set()
    
    # 从 cuBLASLt 目录查找
    if CUBLASLT_DIR.exists():
        for d in CUBLASLT_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                hw_folders.add(d.name)
    
    # 从 cuSPARSELt 目录查找
    if CUSPARSELT_DIR.exists():
        for d in CUSPARSELT_DIR.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                hw_folders.add(d.name)
    
    return sorted(hw_folders)


def main():
    parser = argparse.ArgumentParser(
        description="从 kernel benchmark 结果提取 speedup"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help=f"指定 dtype，默认全部: {','.join(DTYPES)}"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型名（包括 SQUARE），默认全部"
    )
    parser.add_argument(
        "--sparsity",
        type=str,
        default=None,
        help=f"指定稀疏度列表（逗号分隔），默认: {','.join(SPARSITY_LIST)}"
    )
    
    args = parser.parse_args()
    
    # 解析参数
    dtype_list = [args.dtype] if args.dtype else DTYPES
    model_list = [args.model.upper()] if args.model else MODELS + ["SQUARE"]
    sparsity_list = args.sparsity.split(",") if args.sparsity else SPARSITY_LIST
    
    # 查找可用的硬件文件夹
    hw_folders = find_available_hw_folders()
    
    if not hw_folders:
        print_error("未找到任何结果目录")
        print_info(f"cuBLASLt 目录: {CUBLASLT_DIR}")
        print_info(f"cuSPARSELt 目录: {CUSPARSELT_DIR}")
        return 1
    
    print()
    print("=" * 70)
    print("SlideSparse Kernel Benchmark 结果提取")
    print("=" * 70)
    print()
    print_info(f"硬件文件夹: {hw_folders}")
    print_info(f"模型列表: {model_list}")
    print_info(f"精度列表: {dtype_list}")
    print_info(f"稀疏度: {sparsity_list}")
    print_info(f"输出目录: {OUTPUT_DIR}")
    print()
    
    total_success = 0
    total_skip = 0
    
    for hw_folder in hw_folders:
        print_header(f"处理硬件: {hw_folder}")
        
        for dtype in dtype_list:
            print_subheader(f"dtype: {dtype}")
            
            for model in model_list:
                is_square = (model.upper() == "SQUARE")
                
                has_data = process_model(
                    hw_folder=hw_folder,
                    model_name=model,
                    dtype=dtype,
                    sparsity_list=sparsity_list,
                    output_base=OUTPUT_DIR,
                    is_square=is_square,
                )
                
                if has_data:
                    total_success += 1
                    print_success(f"  {model}: 已生成 absolute_latency + speedup")
                else:
                    total_skip += 1
                    print_warning(f"  {model}: 无数据")
    
    print()
    print("=" * 70)
    print("提取完成!")
    print("=" * 70)
    print()
    print_success(f"成功: {total_success} 个模型")
    if total_skip > 0:
        print_warning(f"跳过: {total_skip} 个模型（无数据）")
    print()
    print_info(f"结果保存在: {OUTPUT_DIR}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
