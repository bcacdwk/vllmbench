#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark 结果提取脚本

从 cuBLASLt 和 cuSPARSELt 的 alg_search_results 中提取 TOPS 和 Latency 数据，
计算加速比 (cuBLAS / cuSPARSE)，生成汇总 CSV。

输出目录结构:
    kernel_speedup_results/
    ├── {hw_folder}/
    │   ├── tops/{dtype}/
    │   │   ├── tops_Llama3.2-1B-INT8.csv
    │   │   ├── tops_SQUARE.csv
    │   │   └── ...
    │   ├── latency/{dtype}/
    │   │   ├── latency_Llama3.2-1B-INT8.csv        # 所有 MNK
    │   │   ├── total_latency_Llama3.2-1B-INT8.csv  # 按 M 汇总（仅 model）
    │   │   ├── latency_SQUARE.csv
    │   │   └── ...
    │   └── speedup/{dtype}/
    │       ├── speedup_Llama3.2-1B-INT8.csv        # 所有 MNK
    │       ├── total_speedup_Llama3.2-1B-INT8.csv  # 按 M 汇总（仅 model）
    │       ├── speedup_SQUARE.csv
    │       └── ...
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
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

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

# 模型列表（包括可能的 BitNet）
MODELS = [
    "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
    "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
    "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
    "BitNet-2B-INT8", "BitNet-2B-FP8",
]

# 5种精度
DTYPES = ["fp16", "bf16", "int8", "fp8e4m3", "fp4e2m1"]

# 稀疏度列表 (高稀疏 + 低稀疏)
SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10", "2_12", "2_14", "2_16", "2_inf"]

# 源数据目录
CUBLASLT_DIR = _SCRIPT_DIR / "cuBLASLt" / "alg_search_results"
CUSPARSELT_DIR = _SCRIPT_DIR / "cuSPARSELt" / "alg_search_results"

# 输出目录
OUTPUT_DIR = _SCRIPT_DIR / "kernel_speedup_results"


# =============================================================================
# 工具函数
# =============================================================================

def load_json_results(json_path: Optional[Path]) -> Optional[Dict]:
    """加载 JSON 结果文件"""
    if json_path is None or not json_path.exists():
        return None
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def get_nk_list_from_json(data: Dict) -> List[Tuple[int, int]]:
    """
    从 JSON meta 中获取完整的 NK 列表
    
    Returns:
        [(N, K), ...] 列表
    """
    if not data:
        return []
    
    meta = data.get("meta", {})
    nk_list = meta.get("NK_list", [])
    
    # NK_list 格式: [[N1, K1], [N2, K2], ...]
    return [(int(nk[0]), int(nk[1])) for nk in nk_list]


def build_index_from_json(data: Dict) -> Dict[Tuple[int, int, int], Dict]:
    """
    从 JSON 数据构建索引
    
    Returns:
        {(M, N, K): result_dict}
    """
    index = {}
    if not data:
        return index
    
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


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ExtractedRow:
    """一行提取的数据（对应一个 MNK 组合）"""
    m_value: int
    n_value: int
    k_value: int
    cublas_tops: Optional[float]
    cublas_lat_us: Optional[float]
    cusparse_tops: Dict[str, Optional[float]]  # sparsity -> tops
    cusparse_lat_us: Dict[str, Optional[float]]  # sparsity -> lat_us


@dataclass
class TotalRow:
    """按 M 汇总的数据（所有 NK 相加）"""
    m_value: int
    cublas_total_lat: Optional[float]  # 所有 NK 的 cuBLAS latency 之和
    cusparse_total_lat: Dict[str, Optional[float]]  # sparsity -> 所有 NK latency 之和


# =============================================================================
# 数据提取
# =============================================================================

def extract_model_data(
    hw_folder: str,
    model_name: str,
    dtype: str,
    sparsity_list: List[str],
) -> Tuple[List[ExtractedRow], List[Tuple[int, int]]]:
    """
    提取单个模型的所有数据
    
    Returns:
        (rows, nk_list): 提取的数据行列表 和 完整的 NK 列表
    """
    # 加载 cuBLASLt 数据
    cublas_json_path = find_cublaslt_json(hw_folder, model_name, dtype)
    cublas_data = load_json_results(cublas_json_path)
    cublas_index = build_index_from_json(cublas_data)
    
    # 获取完整的 NK 列表（从 cuBLASLt JSON meta）
    nk_list = get_nk_list_from_json(cublas_data)
    
    # 加载 cuSPARSELt 数据（每个稀疏度）
    cusparse_data_map = {}  # sparsity -> json_data
    cusparse_indices = {}   # sparsity -> index
    
    for sp in sparsity_list:
        sp_json_path = find_cusparselt_json(hw_folder, model_name, dtype, sp)
        sp_data = load_json_results(sp_json_path)
        if sp_data:
            cusparse_data_map[sp] = sp_data
            cusparse_indices[sp] = build_index_from_json(sp_data)
            # 如果 cuBLASLt 没有 NK 列表，尝试从 cuSPARSELt 获取
            if not nk_list:
                nk_list = get_nk_list_from_json(sp_data)
    
    if not cublas_index and not cusparse_indices:
        return [], []
    
    # 收集所有唯一的 (M, N, K) 组合（从 cuBLASLt）
    rows = []
    for (M, N, K), cublas_result in sorted(cublas_index.items()):
        cublas_tops = cublas_result.get("tops")
        cublas_lat = cublas_result.get("lat_us")
        
        # 收集各稀疏度的结果
        cusparse_tops = {}
        cusparse_lat = {}
        
        for sp in sparsity_list:
            # 计算 K_slide
            K_slide = calculate_k_slide(K, sp, dtype=dtype)
            
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
        )
        rows.append(row)
    
    return rows, nk_list


def compute_total_rows(
    rows: List[ExtractedRow],
    nk_list: List[Tuple[int, int]],
    sparsity_list: List[str],
) -> List[TotalRow]:
    """
    计算按 M 汇总的数据（所有 NK 的 latency 相加）
    
    规则：
    - 如果某个 M 下，任意一个 NK pair 的 cuBLAS 缺失，整个 cuBLAS total 留空
    - 如果某个 M 下，任意一个 NK pair 的某稀疏度 cuSPARSE 缺失，该稀疏度 total 留空
    
    Args:
        rows: 提取的数据行
        nk_list: 完整的 NK 列表（定义了该模型应该有多少个 NK）
        sparsity_list: 稀疏度列表
    
    Returns:
        TotalRow 列表（按 M 排序）
    """
    if not nk_list or len(nk_list) == 0:
        return []
    
    # 按 M 分组
    m_to_rows: Dict[int, List[ExtractedRow]] = defaultdict(list)
    for row in rows:
        m_to_rows[row.m_value].append(row)
    
    # 构建 NK 集合用于检查完整性
    nk_set = set(nk_list)
    expected_nk_count = len(nk_list)
    
    total_rows = []
    for m in sorted(m_to_rows.keys()):
        m_rows = m_to_rows[m]
        
        # 检查该 M 下是否有完整的 NK 列表
        actual_nk_set = {(row.n_value, row.k_value) for row in m_rows}
        
        # cuBLAS total：所有 NK 都必须有数据
        cublas_total = None
        if actual_nk_set == nk_set:
            # 检查所有行的 cuBLAS latency 是否都存在
            all_cublas_valid = all(
                row.cublas_lat_us is not None and row.cublas_lat_us > 0
                for row in m_rows
            )
            if all_cublas_valid:
                cublas_total = sum(row.cublas_lat_us for row in m_rows)
        
        # cuSPARSE total：每个稀疏度独立检查
        cusparse_totals = {}
        for sp in sparsity_list:
            if actual_nk_set == nk_set:
                # 检查所有行的该稀疏度 cuSPARSE latency 是否都存在
                all_sp_valid = all(
                    row.cusparse_lat_us.get(sp) is not None and row.cusparse_lat_us.get(sp) > 0
                    for row in m_rows
                )
                if all_sp_valid:
                    cusparse_totals[sp] = sum(row.cusparse_lat_us.get(sp) for row in m_rows)
                else:
                    cusparse_totals[sp] = None
            else:
                cusparse_totals[sp] = None
        
        total_rows.append(TotalRow(
            m_value=m,
            cublas_total_lat=cublas_total,
            cusparse_total_lat=cusparse_totals,
        ))
    
    return total_rows


# =============================================================================
# CSV 写入函数
# =============================================================================

def write_tops_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入 TOPS CSV
    
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
            values = [
                str(row.m_value),
                str(row.n_value),
                str(row.k_value),
                f"{row.cublas_tops:.2f}" if row.cublas_tops is not None else "",
            ]
            
            for sp in sparsity_list:
                sp_tops = row.cusparse_tops.get(sp)
                values.append(f"{sp_tops:.2f}" if sp_tops is not None else "")
            
            f.write(",".join(values) + "\n")
            
            if row.cublas_tops is not None:
                valid_count += 1
    
    return valid_count


def write_latency_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入 Latency CSV（单位：微秒）
    
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


def write_total_latency_csv(total_rows: List[TotalRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入 Total Latency CSV（按 M 汇总，单位：微秒）
    
    Returns:
        有效数据行数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # 表头（没有 N, K 列）
        header = ["M", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in total_rows:
            values = [
                str(row.m_value),
                f"{row.cublas_total_lat:.3f}" if row.cublas_total_lat is not None else "",
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_total_lat.get(sp)
                values.append(f"{sp_lat:.3f}" if sp_lat is not None else "")
            
            f.write(",".join(values) + "\n")
            
            if row.cublas_total_lat is not None:
                valid_count += 1
    
    return valid_count


def write_speedup_csv(rows: List[ExtractedRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入 Speedup CSV
    
    加速比 = cuBLAS_latency / cuSPARSE_latency
    （latency 越小越好，所以 cuBLAS 除以 cuSPARSE，>1 表示 cuSPARSE 更快）
    
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
            # 如果 cuBLAS latency 缺失，整行 speedup 留空（只输出 MNK）
            if row.cublas_lat_us is None or row.cublas_lat_us <= 0:
                values = [
                    str(row.m_value),
                    str(row.n_value),
                    str(row.k_value),
                    "",  # cuBLAS 列留空
                ]
                for sp in sparsity_list:
                    values.append("")  # 所有 cuSPARSE 列留空
                f.write(",".join(values) + "\n")
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
                    # 加速比 = cuBLAS_lat / cuSPARSE_lat
                    speedup = row.cublas_lat_us / sp_lat
                    values.append(f"{speedup:.2f}")
                else:
                    values.append("")
            
            f.write(",".join(values) + "\n")
            valid_count += 1
    
    return valid_count


def write_total_speedup_csv(total_rows: List[TotalRow], output_path: Path, sparsity_list: List[str]) -> int:
    """
    写入 Total Speedup CSV（按 M 汇总）
    
    加速比 = cuBLAS_total_lat / cuSPARSE_total_lat
    
    Returns:
        有效数据行数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        # 表头（没有 N, K 列）
        header = ["M", "cuBLAS"]
        for sp in sparsity_list:
            header.append(f"cuSPARSE_{sp}")
        f.write(",".join(header) + "\n")
        
        for row in total_rows:
            # 如果 cuBLAS total 缺失，整行留空（只输出 M）
            if row.cublas_total_lat is None or row.cublas_total_lat <= 0:
                values = [str(row.m_value), ""]
                for sp in sparsity_list:
                    values.append("")
                f.write(",".join(values) + "\n")
                continue
            
            values = [
                str(row.m_value),
                "1.00",  # cuBLAS 作为基准 = 1.00
            ]
            
            for sp in sparsity_list:
                sp_lat = row.cusparse_total_lat.get(sp)
                if sp_lat is not None and sp_lat > 0:
                    speedup = row.cublas_total_lat / sp_lat
                    values.append(f"{speedup:.2f}")
                else:
                    values.append("")
            
            f.write(",".join(values) + "\n")
            valid_count += 1
    
    return valid_count


# =============================================================================
# 处理单个模型
# =============================================================================

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
    # 提取数据
    rows, nk_list = extract_model_data(hw_folder, model_name, dtype, sparsity_list)
    
    if not rows:
        return False
    
    dtype_folder = build_dtype_folder_name(dtype)
    
    # 1. 写入 TOPS
    tops_dir = output_base / hw_folder / "tops" / dtype_folder
    tops_csv = tops_dir / f"tops_{model_name}.csv"
    write_tops_csv(rows, tops_csv, sparsity_list)
    
    # 2. 写入 Latency
    latency_dir = output_base / hw_folder / "latency" / dtype_folder
    latency_csv = latency_dir / f"latency_{model_name}.csv"
    write_latency_csv(rows, latency_csv, sparsity_list)
    
    # 3. 写入 Speedup
    speedup_dir = output_base / hw_folder / "speedup" / dtype_folder
    speedup_csv = speedup_dir / f"speedup_{model_name}.csv"
    write_speedup_csv(rows, speedup_csv, sparsity_list)
    
    # 4. 对于非 SQUARE 模型，计算并写入 Total
    if not is_square and nk_list and len(nk_list) > 1:
        total_rows = compute_total_rows(rows, nk_list, sparsity_list)
        
        if total_rows:
            # Total Latency
            total_latency_csv = latency_dir / f"total_latency_{model_name}.csv"
            write_total_latency_csv(total_rows, total_latency_csv, sparsity_list)
            
            # Total Speedup
            total_speedup_csv = speedup_dir / f"total_speedup_{model_name}.csv"
            write_total_speedup_csv(total_rows, total_speedup_csv, sparsity_list)
    
    return True


# =============================================================================
# 查找可用数据
# =============================================================================

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


def find_available_dtypes(hw_folder: str) -> List[str]:
    """查找某硬件下可用的 dtype"""
    dtypes = set()
    
    cublas_hw_dir = CUBLASLT_DIR / hw_folder
    if cublas_hw_dir.exists():
        for d in cublas_hw_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                dtypes.add(d.name.lower())
    
    cusparse_hw_dir = CUSPARSELT_DIR / hw_folder
    if cusparse_hw_dir.exists():
        for d in cusparse_hw_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                dtypes.add(d.name.lower())
    
    # 标准化 dtype 名称
    dtype_map = {
        "fp16": "fp16",
        "bf16": "bf16",
        "int8": "int8",
        "fp8": "fp8e4m3",
        "fp8e4m3": "fp8e4m3",
        "fp4": "fp4e2m1",
        "fp4e2m1": "fp4e2m1",
    }
    
    normalized = set()
    for dt in dtypes:
        if dt in dtype_map:
            normalized.add(dtype_map[dt])
    
    return sorted(normalized)


def find_available_models(hw_folder: str, dtype: str) -> List[str]:
    """查找某硬件、某 dtype 下可用的模型"""
    models = set()
    dtype_folder = build_dtype_folder_name(dtype)
    
    # 从 cuBLASLt 查找
    cublas_dir = CUBLASLT_DIR / hw_folder / dtype_folder
    if cublas_dir.exists():
        for f in cublas_dir.iterdir():
            if f.is_file() and f.suffix == ".json" and f.name.startswith("alg_search_"):
                # alg_search_Llama3.2-1B-INT8.json -> Llama3.2-1B-INT8
                model_name = f.stem.replace("alg_search_", "")
                models.add(model_name)
    
    return sorted(models)


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 kernel benchmark 结果提取 TOPS, Latency, Speedup"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help=f"指定 dtype，默认自动检测"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="指定模型名（包括 SQUARE），默认自动检测"
    )
    parser.add_argument(
        "--sparsity",
        type=str,
        default=None,
        help=f"指定稀疏度列表（逗号分隔），默认: {','.join(SPARSITY_LIST)}"
    )
    
    args = parser.parse_args()
    
    # 解析稀疏度列表
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
    print_info(f"稀疏度: {sparsity_list}")
    print_info(f"输出目录: {OUTPUT_DIR}")
    print()
    
    total_success = 0
    total_skip = 0
    
    for hw_folder in hw_folders:
        print_header(f"处理硬件: {hw_folder}")
        
        # 确定 dtype 列表
        if args.dtype:
            dtype_list = [args.dtype]
        else:
            dtype_list = find_available_dtypes(hw_folder)
        
        if not dtype_list:
            print_warning(f"  未找到任何 dtype 数据")
            continue
        
        for dtype in dtype_list:
            print_subheader(f"dtype: {dtype}")
            
            # 确定模型列表
            if args.model:
                model_list = [args.model]
            else:
                model_list = find_available_models(hw_folder, dtype)
            
            if not model_list:
                print_warning(f"    未找到任何模型数据")
                continue
            
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
                    if is_square:
                        print_success(f"    {model}: tops + latency + speedup")
                    else:
                        print_success(f"    {model}: tops + latency + speedup + total_latency + total_speedup")
                else:
                    total_skip += 1
                    print_warning(f"    {model}: 无数据")
    
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
    print()
    print("输出结构:")
    print(f"  {OUTPUT_DIR}/{{hw}}/tops/{{dtype}}/tops_*.csv")
    print(f"  {OUTPUT_DIR}/{{hw}}/latency/{{dtype}}/latency_*.csv")
    print(f"  {OUTPUT_DIR}/{{hw}}/latency/{{dtype}}/total_latency_*.csv  (仅 model)")
    print(f"  {OUTPUT_DIR}/{{hw}}/speedup/{{dtype}}/speedup_*.csv")
    print(f"  {OUTPUT_DIR}/{{hw}}/speedup/{{dtype}}/total_speedup_*.csv  (仅 model)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
