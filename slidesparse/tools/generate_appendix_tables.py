#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Appendix 表格生成脚本

从 kernel_speedup_results 和 end2end_speedup_results 中提取数据，
生成用于论文 Appendix 的 CSV 大表。

输出文件:
    appendix_tables/
    ├── appendix_a_square_BF16.csv        # Square Kernel (5种精度)
    ├── appendix_a_square_FP16.csv
    ├── appendix_a_square_INT8.csv
    ├── appendix_a_square_FP8.csv
    ├── appendix_a_square_FP4.csv
    ├── appendix_b_model_kernel_BF16.csv  # Model Kernel (5种GEMM精度)
    ├── appendix_b_model_kernel_FP16.csv
    ├── appendix_b_model_kernel_INT8.csv
    ├── appendix_b_model_kernel_FP8.csv
    ├── appendix_b_model_kernel_FP4.csv
    ├── appendix_c_prefill_INT8.csv       # E2E Prefill (按精度分)
    ├── appendix_c_prefill_FP8.csv
    ├── appendix_d_decode_INT8.csv        # E2E Decode (按精度分)
    └── appendix_d_decode_FP8.csv

Usage:
    python3 generate_appendix_tables.py
"""

import os
import sys
import csv
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

# =============================================================================
# 配置常量
# =============================================================================

# GPU 顺序 (按架构)
GPU_ORDER = ["A100", "RTX4090", "H100", "B200", "RTX5080", "GB10"]

# GPU 目录名映射 (简称 -> 完整目录前缀)
GPU_KERNEL_DIR_MAP = {
    "A100": "A100_cc80",
    "RTX4090": "RTX4090_cc89",
    "H100": "H100_cc90",
    "B200": "B200_cc100",
    "RTX5080": "RTX5080_cc120",
    "GB10": "GB10_cc121",
}

# GEMM 精度列表
DTYPES = ["BF16", "FP16", "INT8", "FP8", "FP4"]

# dtype 目录名映射
DTYPE_DIR_MAP = {
    "BF16": "BF16",
    "FP16": "FP16",
    "INT8": "INT8",
    "FP8": "FP8",
    "FP4": "FP4",
}

# 稀疏度列表 (Kernel) - 使用新格式
KERNEL_SPARSITY_LIST = ["2:4", "2:6", "2:8", "2:10", "2:12", "2:14", "2:16", "2:∞"]
# 稀疏度映射 (新格式 -> 原CSV列名)
KERNEL_SPARSITY_MAP = {
    "2:4": "2_4", "2:6": "2_6", "2:8": "2_8", "2:10": "2_10",
    "2:12": "2_12", "2:14": "2_14", "2:16": "2_16", "2:∞": "2_inf"
}

# 稀疏度列表 (E2E) - 使用新格式
E2E_SPARSITY_LIST = ["2:4", "2:6", "2:8", "2:10"]
E2E_SPARSITY_MAP = {"2:4": "2_4", "2:6": "2_6", "2:8": "2_8", "2:10": "2_10"}

# 模型列表 (简化名，用于 Appendix B) - 按参数量排序
MODELS_SIMPLE = ["Llama3.2-1B", "BitNet-2B", "Llama3.2-3B", "Qwen2.5-7B", "Qwen2.5-14B"]

# 模型列表 (完整名，用于 E2E) - 按参数量排序，INT8 和 FP8 分开
MODELS_E2E_INT8 = [
    "Llama3.2-1B-INT8", "BitNet-2B-INT8", "Llama3.2-3B-INT8",
    "Qwen2.5-7B-INT8", "Qwen2.5-14B-INT8",
]
MODELS_E2E_FP8 = [
    "Llama3.2-1B-FP8", "BitNet-2B-FP8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-FP8", "Qwen2.5-14B-FP8",
]

# M 值上限
M_MAX_APPENDIX_A = 16384
M_MAX_APPENDIX_B = 16384
M_MAX_APPENDIX_C = 32768
# Appendix D 不限制

# 数据源目录
KERNEL_RESULTS_DIR = _SLIDESPARSE_ROOT / "benchmark_kernel" / "kernel_speedup_results"
E2E_RESULTS_DIR = _SCRIPT_DIR / "end2end_speedup_results"

# 输出目录
OUTPUT_DIR = _SCRIPT_DIR / "appendix_tables"


# =============================================================================
# 工具函数
# =============================================================================

def find_kernel_hw_dir(gpu_name: str) -> Optional[Path]:
    """查找 Kernel 结果中对应 GPU 的目录"""
    prefix = GPU_KERNEL_DIR_MAP.get(gpu_name)
    if not prefix:
        return None
    
    for d in KERNEL_RESULTS_DIR.iterdir():
        if d.is_dir() and d.name.startswith(prefix):
            return d
    return None


def read_csv_to_dict(csv_path: Path) -> List[Dict]:
    """读取 CSV 文件为字典列表"""
    if not csv_path.exists():
        return []
    
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def format_scientific(val: str) -> str:
    """格式化为科学计数法 (如 1.12e3)"""
    if not val or val.strip() == '':
        return ''
    try:
        v = float(val)
        if v == 0:
            return '0'
        return f"{v:.2e}"
    except:
        return ''


def format_speedup(val: str) -> str:
    """格式化 speedup 值 (保留2位小数)"""
    if not val or val.strip() == '':
        return ''
    try:
        return f"{float(val):.2f}"
    except:
        return ''


def get_model_simple_name(model_full: str) -> str:
    """从完整模型名获取简化名 (去掉 -INT8/-FP8)"""
    if model_full.endswith("-INT8"):
        return model_full[:-5]
    elif model_full.endswith("-FP8"):
        return model_full[:-4]
    return model_full


def get_model_precision(model_full: str) -> str:
    """从完整模型名获取精度"""
    if "-INT8" in model_full:
        return "INT8"
    elif "-FP8" in model_full:
        return "FP8"
    return ""


# =============================================================================
# Appendix A: Square Kernel Performance
# =============================================================================

def generate_appendix_a():
    """生成 Appendix A: Square Kernel 大表"""
    print_header("生成 Appendix A: Square Kernel Performance")
    
    for dtype in DTYPES:
        print_subheader(f"处理 {dtype}")
        
        rows_out = []
        
        for gpu in GPU_ORDER:
            hw_dir = find_kernel_hw_dir(gpu)
            if not hw_dir:
                print_warning(f"  {gpu}: 未找到目录")
                continue
            
            # 构建文件路径
            dtype_dir = DTYPE_DIR_MAP.get(dtype, dtype)
            latency_csv = hw_dir / "latency" / dtype_dir / "latency_SQUARE.csv"
            speedup_csv = hw_dir / "speedup" / dtype_dir / "speedup_SQUARE.csv"
            
            if not latency_csv.exists():
                print_warning(f"  {gpu}: {dtype} 无数据")
                continue
            
            # 读取数据
            latency_data = read_csv_to_dict(latency_csv)
            speedup_data = read_csv_to_dict(speedup_csv)
            
            # 建立 speedup 索引 (M -> row)
            speedup_index = {}
            for row in speedup_data:
                m = row.get('M', '')
                if m:
                    speedup_index[m] = row
            
            # 生成输出行
            row_count = 0
            for lat_row in latency_data:
                m = lat_row.get('M', '')
                if not m:
                    continue
                
                # 过滤 M 值
                try:
                    m_int = int(m)
                    if m_int > M_MAX_APPENDIX_A:
                        continue
                except:
                    continue
                
                sp_row = speedup_index.get(m, {})
                
                out_row = {
                    'GPU': gpu,
                    'M': m,
                    'cuBLASLt Latency (μs)': format_scientific(lat_row.get('cuBLAS', '')),
                }
                
                # 添加各稀疏度的 speedup
                for sp_new in KERNEL_SPARSITY_LIST:
                    sp_old = KERNEL_SPARSITY_MAP[sp_new]
                    col_name_old = f'cuSPARSE_{sp_old}'
                    out_row[sp_new] = format_speedup(sp_row.get(col_name_old, ''))
                
                rows_out.append(out_row)
                row_count += 1
            
            print_success(f"  {gpu}: {row_count} 行")
        
        # 写入 CSV
        if rows_out:
            output_path = OUTPUT_DIR / f"appendix_a_square_{dtype}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = ['GPU', 'M', 'cuBLASLt Latency (μs)'] + KERNEL_SPARSITY_LIST
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_out)
            
            print_success(f"  输出: {output_path.name} ({len(rows_out)} 行)")
        else:
            print_warning(f"  {dtype}: 无数据输出")


# =============================================================================
# Appendix B: Model-Aware Kernel Performance
# =============================================================================

def generate_appendix_b():
    """生成 Appendix B: Model Kernel 大表 (按 GEMM 精度分)"""
    print_header("生成 Appendix B: Model-Aware Kernel Performance")
    
    for dtype in DTYPES:
        print_subheader(f"处理 {dtype}")
        
        rows_out = []
        
        for gpu in GPU_ORDER:
            hw_dir = find_kernel_hw_dir(gpu)
            if not hw_dir:
                print_warning(f"  {gpu}: 未找到目录")
                continue
            
            dtype_dir = DTYPE_DIR_MAP.get(dtype, dtype)
            latency_dir = hw_dir / "latency" / dtype_dir
            speedup_dir = hw_dir / "speedup" / dtype_dir
            
            if not latency_dir.exists():
                print_warning(f"  {gpu}: {dtype} 无数据")
                continue
            
            for model_simple in MODELS_SIMPLE:
                # 尝试找到对应的 total_latency 文件
                # 文件名格式: total_latency_Llama3.2-1B-INT8.csv
                # 因为 INT8 和 FP8 的 NK 相同，我们只需要找一个
                model_file = None
                suffix_found = None
                for suffix in ["-INT8", "-FP8"]:
                    candidate = latency_dir / f"total_latency_{model_simple}{suffix}.csv"
                    if candidate.exists():
                        model_file = candidate
                        suffix_found = suffix
                        break
                
                if not model_file:
                    continue
                
                # 对应的 speedup 文件
                speedup_file = speedup_dir / f"total_speedup_{model_simple}{suffix_found}.csv"
                
                # 读取数据
                latency_data = read_csv_to_dict(model_file)
                speedup_data = read_csv_to_dict(speedup_file) if speedup_file.exists() else []
                
                # 建立 speedup 索引
                speedup_index = {}
                for row in speedup_data:
                    m = row.get('M', '')
                    if m:
                        speedup_index[m] = row
                
                # 生成输出行
                for lat_row in latency_data:
                    m = lat_row.get('M', '')
                    if not m:
                        continue
                    
                    # 过滤 M 值
                    try:
                        m_int = int(m)
                        if m_int > M_MAX_APPENDIX_B:
                            continue
                    except:
                        continue
                    
                    sp_row = speedup_index.get(m, {})
                    
                    out_row = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuBLASLt Latency (μs)': format_scientific(lat_row.get('cuBLAS', '')),
                    }
                    
                    # 添加各稀疏度的 speedup
                    for sp_new in KERNEL_SPARSITY_LIST:
                        sp_old = KERNEL_SPARSITY_MAP[sp_new]
                        col_name_old = f'cuSPARSE_{sp_old}'
                        out_row[sp_new] = format_speedup(sp_row.get(col_name_old, ''))
                    
                    rows_out.append(out_row)
            
            print_info(f"  {gpu}: 处理完成")
        
        # 写入 CSV
        if rows_out:
            output_path = OUTPUT_DIR / f"appendix_b_model_kernel_{dtype}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = ['GPU', 'Model', 'M', 'cuBLASLt Latency (μs)'] + KERNEL_SPARSITY_LIST
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_out)
            
            print_success(f"  输出: {output_path.name} ({len(rows_out)} 行)")
        else:
            print_warning(f"  {dtype}: 无数据输出")


# =============================================================================
# Appendix C: End-to-End Prefill Performance (按精度分表)
# =============================================================================

def generate_appendix_c():
    """生成 Appendix C: E2E Prefill 大表 (INT8 和 FP8 分开)"""
    print_header("生成 Appendix C: End-to-End Prefill Performance")
    
    for precision, models_list in [("INT8", MODELS_E2E_INT8), ("FP8", MODELS_E2E_FP8)]:
        print_subheader(f"处理 {precision}")
        
        rows_out = []
        
        for gpu in GPU_ORDER:
            hw_dir = E2E_RESULTS_DIR / gpu / "prefill"
            
            if not hw_dir.exists():
                print_warning(f"  {gpu}: 未找到 prefill 目录")
                continue
            
            for model_full in models_list:
                model_simple = get_model_simple_name(model_full)
                
                # 读取绝对吞吐量和加速比
                abs_csv = hw_dir / f"absolute_throughput_{model_full}.csv"
                speedup_csv = hw_dir / f"speedup_{model_full}.csv"
                
                if not abs_csv.exists():
                    continue
                
                abs_data = read_csv_to_dict(abs_csv)
                speedup_data = read_csv_to_dict(speedup_csv) if speedup_csv.exists() else []
                
                # 建立 speedup 索引
                speedup_index = {}
                for row in speedup_data:
                    m = row.get('M', '')
                    if m:
                        speedup_index[m] = row
                
                # 生成输出行
                for abs_row in abs_data:
                    m = abs_row.get('M', '')
                    if not m:
                        continue
                    
                    # 过滤 M 值
                    try:
                        m_int = int(m)
                        if m_int > M_MAX_APPENDIX_C:
                            continue
                    except:
                        continue
                    
                    sp_row = speedup_index.get(m, {})
                    
                    out_row = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuBLASLt Throughput (token/s)': format_scientific(abs_row.get('cuBLAS', '')),
                    }
                    
                    # 添加各稀疏度的 speedup
                    for sp_new in E2E_SPARSITY_LIST:
                        sp_old = E2E_SPARSITY_MAP[sp_new]
                        col_name_old = f'cusparse_{sp_old}'
                        out_row[sp_new] = format_speedup(sp_row.get(col_name_old, ''))
                    
                    rows_out.append(out_row)
            
            print_info(f"  {gpu}: 处理完成")
        
        # 写入 CSV
        if rows_out:
            output_path = OUTPUT_DIR / f"appendix_c_prefill_{precision}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = ['GPU', 'Model', 'M', 'cuBLASLt Throughput (token/s)'] + E2E_SPARSITY_LIST
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_out)
            
            print_success(f"  输出: {output_path.name} ({len(rows_out)} 行)")
        else:
            print_warning(f"  Prefill {precision}: 无数据输出")


# =============================================================================
# Appendix D: End-to-End Decode Performance (按精度分表)
# =============================================================================

def generate_appendix_d():
    """生成 Appendix D: E2E Decode 大表 (INT8 和 FP8 分开)"""
    print_header("生成 Appendix D: End-to-End Decode Performance")
    
    for precision, models_list in [("INT8", MODELS_E2E_INT8), ("FP8", MODELS_E2E_FP8)]:
        print_subheader(f"处理 {precision}")
        
        rows_out = []
        
        for gpu in GPU_ORDER:
            hw_dir = E2E_RESULTS_DIR / gpu / "decode"
            
            if not hw_dir.exists():
                print_warning(f"  {gpu}: 未找到 decode 目录")
                continue
            
            for model_full in models_list:
                model_simple = get_model_simple_name(model_full)
                
                # 读取绝对吞吐量和加速比
                abs_csv = hw_dir / f"absolute_throughput_{model_full}.csv"
                speedup_csv = hw_dir / f"speedup_{model_full}.csv"
                
                if not abs_csv.exists():
                    continue
                
                abs_data = read_csv_to_dict(abs_csv)
                speedup_data = read_csv_to_dict(speedup_csv) if speedup_csv.exists() else []
                
                # 建立 speedup 索引
                speedup_index = {}
                for row in speedup_data:
                    m = row.get('M', '')
                    if m:
                        speedup_index[m] = row
                
                # 生成输出行
                for abs_row in abs_data:
                    m = abs_row.get('M', '')
                    if not m:
                        continue
                    
                    # Decode 不限制 M 值上限
                    
                    sp_row = speedup_index.get(m, {})
                    
                    out_row = {
                        'GPU': gpu,
                        'Model': model_simple,
                        'M': m,
                        'cuBLASLt Throughput (token/s)': format_scientific(abs_row.get('cuBLAS', '')),
                    }
                    
                    # 添加各稀疏度的 speedup
                    for sp_new in E2E_SPARSITY_LIST:
                        sp_old = E2E_SPARSITY_MAP[sp_new]
                        col_name_old = f'cusparse_{sp_old}'
                        out_row[sp_new] = format_speedup(sp_row.get(col_name_old, ''))
                    
                    rows_out.append(out_row)
            
            print_info(f"  {gpu}: 处理完成")
        
        # 写入 CSV
        if rows_out:
            output_path = OUTPUT_DIR / f"appendix_d_decode_{precision}.csv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = ['GPU', 'Model', 'M', 'cuBLASLt Throughput (token/s)'] + E2E_SPARSITY_LIST
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_out)
            
            print_success(f"  输出: {output_path.name} ({len(rows_out)} 行)")
        else:
            print_warning(f"  Decode {precision}: 无数据输出")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print_header("=" * 60)
    print_header("SlideSparse Appendix 表格生成")
    print_header("=" * 60)
    
    print_info(f"Kernel 数据源: {KERNEL_RESULTS_DIR}")
    print_info(f"E2E 数据源: {E2E_RESULTS_DIR}")
    print_info(f"输出目录: {OUTPUT_DIR}")
    print()
    
    # 生成各个 Appendix
    generate_appendix_a()
    print()
    
    generate_appendix_b()
    print()
    
    generate_appendix_c()
    print()
    
    generate_appendix_d()
    print()
    
    print_header("=" * 60)
    print_success("所有 Appendix 表格生成完成!")
    print_header("=" * 60)
    
    # 列出输出文件
    print_info("\n输出文件列表:")
    if OUTPUT_DIR.exists():
        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.suffix == '.csv':
                size = f.stat().st_size
                print_info(f"  {f.name} ({size} bytes)")


if __name__ == "__main__":
    main()
