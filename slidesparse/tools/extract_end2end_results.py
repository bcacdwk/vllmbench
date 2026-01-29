#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 端到端结果提取脚本

从 throughput_benchmark_results 中提取 tokens/s 数据，
生成按硬件/stage/model 分类的汇总 CSV。

输出目录结构:
    end2end_speedup_results/
    ├── A100/
    │   ├── prefill/
    │   │   ├── absolute_throughput_Llama3.2-1B-INT8.csv
    │   │   ├── speedup_Llama3.2-1B-INT8.csv
    │   │   └── ...
    │   └── decode/
    │       └── ...
    ├── B200/
    ├── H100/
    ├── RTX4090/
    └── RTX5080/

Usage:
    # 提取所有硬件的结果
    python3 extract_end2end_results.py
    
    # 只提取特定硬件
    python3 extract_end2end_results.py --hardware B200
    python3 extract_end2end_results.py --hardware A100,H100
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


# =============================================================================
# 配置常量
# =============================================================================

# 6个硬件平台
HARDWARE_LIST = ["A100", "B200", "H100", "RTX4090", "RTX5080", "GB10"]

# 10个模型 (5个 base × 2种量化)
MODELS = [
    "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
    "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
    "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
    "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
    "BitNet-2B-INT8", "BitNet-2B-FP8",
]

# 测试阶段
STAGES = ["prefill", "decode"]

# 稀疏度列表 (cusparselt)
SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10"]

# M 值列表 (来自 prepare_for_vllm_bench.py)
M_LIST_PREFILL = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
M_LIST_DECODE = [64, 128, 256, 512]

# 源数据目录
SOURCE_DIR = _SCRIPT_DIR / "throughput_benchmark_results"

# 输出目录
OUTPUT_DIR = _SCRIPT_DIR / "end2end_speedup_results"


# =============================================================================
# 工具函数
# =============================================================================

def get_quant_from_model(model_name: str) -> str:
    """从模型名提取量化类型"""
    if "INT8" in model_name:
        return "INT8"
    elif "FP8" in model_name:
        return "FP8"
    return "UNKNOWN"


def find_hw_dirs(hw_name: str, stage: str) -> List[Path]:
    """
    查找指定硬件在指定 stage 下的所有结果目录
    
    Args:
        hw_name: 硬件名称 (如 "B200")
        stage: "prefill" 或 "decode"
    
    Returns:
        匹配的目录列表 (可能有 INT8 和 FP8 两个)
    """
    stage_dir = SOURCE_DIR / stage
    if not stage_dir.exists():
        return []
    
    matched = []
    for d in stage_dir.iterdir():
        if d.is_dir() and d.name.startswith(hw_name + "_"):
            matched.append(d)
    
    return sorted(matched)


def get_m_list(stage: str) -> List[int]:
    """获取对应 stage 的 M 值列表"""
    return M_LIST_PREFILL if stage == "prefill" else M_LIST_DECODE


def read_tokens_per_second(json_path: Path) -> Optional[float]:
    """
    从 JSON 文件读取 tokens_per_second
    
    Returns:
        tokens_per_second 值，如果文件不存在或无效返回 None
    """
    if not json_path.exists():
        return None
    
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        tps = data.get("tokens_per_second", 0)
        if tps > 0:
            return tps
        return None
    except Exception:
        return None


def find_json_for_model_m(
    hw_dir: Path,
    backend: str,
    model_name: str,
    m_value: int,
    sparsity: Optional[str] = None,
) -> Path:
    """
    构建 JSON 文件路径
    
    Args:
        hw_dir: 硬件目录 (如 B200_cc100_INT8_py312_cu129_x86_64)
        backend: "cublaslt" 或 "cusparselt"
        model_name: 模型名 (如 "Llama3.2-1B-INT8")
        m_value: M 值
        sparsity: 稀疏度 (仅 cusparselt 需要)
    
    Returns:
        JSON 文件路径
    """
    if backend == "cublaslt":
        return hw_dir / "cublaslt" / "json" / f"{model_name}_M{m_value}.json"
    elif backend == "cusparselt" and sparsity:
        return hw_dir / "cusparselt" / sparsity / "json" / f"{model_name}_M{m_value}.json"
    else:
        raise ValueError(f"Invalid backend/sparsity: {backend}/{sparsity}")


def find_hw_dir_for_model(hw_dirs: List[Path], model_name: str) -> Optional[Path]:
    """
    根据模型的量化类型找到对应的硬件目录
    
    Args:
        hw_dirs: 该硬件的所有目录列表
        model_name: 模型名
    
    Returns:
        匹配的硬件目录，找不到返回 None
    """
    quant = get_quant_from_model(model_name)
    
    for hw_dir in hw_dirs:
        dir_name = hw_dir.name
        # INT8 模型找 INT8 目录
        if quant == "INT8" and "_INT8_" in dir_name:
            return hw_dir
        # FP8 模型找 FP8E4M3 目录
        if quant == "FP8" and "_FP8" in dir_name:
            return hw_dir
    
    return None


@dataclass
class ExtractedRow:
    """一行提取的数据"""
    m_value: int
    cublas: Optional[float]
    cusparse_2_4: Optional[float]
    cusparse_2_6: Optional[float]
    cusparse_2_8: Optional[float]
    cusparse_2_10: Optional[float]


def extract_model_data(
    hw_dirs: List[Path],
    model_name: str,
    stage: str,
) -> List[ExtractedRow]:
    """
    提取单个模型的所有 M 值数据
    
    Returns:
        ExtractedRow 列表
    """
    hw_dir = find_hw_dir_for_model(hw_dirs, model_name)
    if hw_dir is None:
        # 该硬件不支持此模型的量化类型
        return []
    
    m_list = get_m_list(stage)
    rows = []
    
    for m_value in m_list:
        # cuBLAS
        cublas_json = find_json_for_model_m(hw_dir, "cublaslt", model_name, m_value)
        cublas_tps = read_tokens_per_second(cublas_json)
        
        # cuSPARSE 各稀疏度
        cusparse_values = {}
        for sp in SPARSITY_LIST:
            sp_json = find_json_for_model_m(hw_dir, "cusparselt", model_name, m_value, sp)
            sp_tps = read_tokens_per_second(sp_json)
            cusparse_values[sp] = sp_tps
        
        row = ExtractedRow(
            m_value=m_value,
            cublas=cublas_tps,
            cusparse_2_4=cusparse_values.get("2_4"),
            cusparse_2_6=cusparse_values.get("2_6"),
            cusparse_2_8=cusparse_values.get("2_8"),
            cusparse_2_10=cusparse_values.get("2_10"),
        )
        rows.append(row)
    
    return rows


def write_absolute_csv(rows: List[ExtractedRow], output_path: Path) -> int:
    """
    写入绝对吞吐量 CSV
    
    Returns:
        有效数据行数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        f.write("M,cuBLAS,cusparse_2_4,cusparse_2_6,cusparse_2_8,cusparse_2_10\n")
        
        for row in rows:
            values = [
                str(row.m_value),
                f"{row.cublas:.3f}" if row.cublas is not None else "",
                f"{row.cusparse_2_4:.3f}" if row.cusparse_2_4 is not None else "",
                f"{row.cusparse_2_6:.3f}" if row.cusparse_2_6 is not None else "",
                f"{row.cusparse_2_8:.3f}" if row.cusparse_2_8 is not None else "",
                f"{row.cusparse_2_10:.3f}" if row.cusparse_2_10 is not None else "",
            ]
            f.write(",".join(values) + "\n")
            
            # 统计有效行 (至少有 cuBLAS 数据)
            if row.cublas is not None:
                valid_count += 1
    
    return valid_count


def write_speedup_csv(rows: List[ExtractedRow], output_path: Path) -> int:
    """
    写入加速比 CSV (相对于 cuBLAS)
    
    Returns:
        有效数据行数
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w") as f:
        f.write("M,cuBLAS,cusparse_2_4,cusparse_2_6,cusparse_2_8,cusparse_2_10\n")
        
        for row in rows:
            # cuBLAS 作为基准，加速比永远是 1.00
            if row.cublas is None or row.cublas <= 0:
                # 没有基准，全部留空
                f.write(f"{row.m_value},,,,,\n")
                continue
            
            valid_count += 1
            
            def calc_speedup(val: Optional[float]) -> str:
                if val is None:
                    return ""
                return f"{val / row.cublas:.2f}"
            
            values = [
                str(row.m_value),
                "1.00",  # cuBLAS baseline
                calc_speedup(row.cusparse_2_4),
                calc_speedup(row.cusparse_2_6),
                calc_speedup(row.cusparse_2_8),
                calc_speedup(row.cusparse_2_10),
            ]
            f.write(",".join(values) + "\n")
    
    return valid_count


def process_hardware(hw_name: str) -> Tuple[int, int]:
    """
    处理单个硬件的所有数据
    
    Returns:
        (成功的模型数, 失败/跳过的模型数)
    """
    print_header(f"处理硬件: {hw_name}")
    
    success_count = 0
    skip_count = 0
    
    for stage in STAGES:
        print_subheader(f"Stage: {stage}")
        
        hw_dirs = find_hw_dirs(hw_name, stage)
        if not hw_dirs:
            print_warning(f"  未找到 {hw_name} 的 {stage} 结果目录")
            skip_count += len(MODELS)
            continue
        
        print_info(f"  找到目录: {[d.name for d in hw_dirs]}")
        
        for model_name in MODELS:
            rows = extract_model_data(hw_dirs, model_name, stage)
            
            if not rows:
                print_warning(f"  {model_name}: 无数据 (可能不支持该量化类型)")
                skip_count += 1
                continue
            
            # 输出目录
            output_dir = OUTPUT_DIR / hw_name / stage
            
            # 写入绝对吞吐量 CSV
            abs_csv = output_dir / f"absolute_throughput_{model_name}.csv"
            valid_abs = write_absolute_csv(rows, abs_csv)
            
            # 写入加速比 CSV
            speedup_csv = output_dir / f"speedup_{model_name}.csv"
            valid_speedup = write_speedup_csv(rows, speedup_csv)
            
            if valid_abs > 0:
                print_success(f"  {model_name}: {valid_abs}/{len(rows)} 行有效")
                success_count += 1
            else:
                print_warning(f"  {model_name}: 全部无效数据")
                skip_count += 1
    
    return success_count, skip_count


def main():
    parser = argparse.ArgumentParser(
        description="从 throughput_benchmark_results 提取端到端结果"
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default=None,
        help=f"指定硬件 (逗号分隔)，默认全部: {','.join(HARDWARE_LIST)}"
    )
    
    args = parser.parse_args()
    
    # 解析硬件列表
    if args.hardware:
        hw_list = [h.strip() for h in args.hardware.split(",")]
        # 验证
        for hw in hw_list:
            if hw not in HARDWARE_LIST:
                print_error(f"未知硬件: {hw}")
                print_info(f"支持的硬件: {HARDWARE_LIST}")
                sys.exit(1)
    else:
        hw_list = HARDWARE_LIST
    
    print()
    print("=" * 70)
    print("SlideSparse 端到端结果提取")
    print("=" * 70)
    print()
    print_info(f"硬件列表: {hw_list}")
    print_info(f"模型列表: {MODELS}")
    print_info(f"阶段: {STAGES}")
    print_info(f"稀疏度: {SPARSITY_LIST}")
    print_info(f"源目录: {SOURCE_DIR}")
    print_info(f"输出目录: {OUTPUT_DIR}")
    print()
    
    total_success = 0
    total_skip = 0
    
    for hw_name in hw_list:
        success, skip = process_hardware(hw_name)
        total_success += success
        total_skip += skip
    
    print()
    print("=" * 70)
    print("提取完成!")
    print("=" * 70)
    print()
    print_success(f"成功: {total_success} 个模型×阶段组合")
    if total_skip > 0:
        print_warning(f"跳过: {total_skip} 个模型×阶段组合")
    print()
    print_info(f"结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
