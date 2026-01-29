#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
CSV 转 PDF 表格工具 (双行表头版本)

将 appendix_tables 目录下的 CSV 文件转换为美观的 PDF 表格，
适用于论文 Appendix 插图。

特点:
- 竖版 (Portrait) A4 页面
- 双行表头设计
- 紧凑布局 (小字体、窄行距)
- 重复值省略 (GPU/Model 相同时留空，跨页也保持)
- 高亮 Speedup > 1.0 的单元格

Usage:
    python3 csv_to_pdf_table.py
    python3 csv_to_pdf_table.py --file appendix_a_square_FP16.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import csv

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# =============================================================================
# 配置
# =============================================================================

# 页面设置 (A4 竖向，单位：英寸)
PAGE_WIDTH = 8.27
PAGE_HEIGHT = 11.69
MARGIN = 0.2

# 表格样式
HEADER_BG_COLOR_L1 = '#1E3A5F'   # 深蓝色 (第一行表头)
HEADER_BG_COLOR_L2 = '#2E4A6F'   # 稍浅蓝色 (第二行表头)
HEADER_TEXT_COLOR = 'white'
ROW_EVEN_COLOR = '#F8F9FA'       # 浅灰色偶数行
ROW_ODD_COLOR = 'white'          # 白色奇数行
SPEEDUP_GOOD_COLOR = '#D4EDDA'   # 浅绿色 (speedup >= 1.0)
SPEEDUP_GREAT_COLOR = '#B8E6C1'  # 更绿 (speedup >= 1.5)
GRID_COLOR = '#DEE2E6'

# 字体设置 (紧凑版)
HEADER_FONT_SIZE = 5.5
CELL_FONT_SIZE = 5
TITLE_FONT_SIZE = 8

# 行高设置 (紧凑版)
ROW_HEIGHT = 0.011
HEADER_HEIGHT = 0.028  # 双行表头总高度

# 固定表格宽度 (所有表统一)
FIXED_TABLE_WIDTH = 0.72

# 每页最大行数
MAX_ROWS_PER_PAGE = 65

# 输入输出目录
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "appendix_tables"
OUTPUT_DIR = SCRIPT_DIR / "appendix_tables_pdf"


# =============================================================================
# 表格类型检测
# =============================================================================

def detect_table_type(header: List[str], filename: str) -> str:
    """
    检测表格类型
    返回: 'kernel_square', 'kernel_model', 'e2e_prefill', 'e2e_decode'
    """
    filename_lower = filename.lower()
    
    if 'appendix_a' in filename_lower or 'square' in filename_lower:
        return 'kernel_square'
    elif 'appendix_b' in filename_lower or 'model_kernel' in filename_lower:
        return 'kernel_model'
    elif 'appendix_c' in filename_lower or 'prefill' in filename_lower:
        return 'e2e_prefill'
    elif 'appendix_d' in filename_lower or 'decode' in filename_lower:
        return 'e2e_decode'
    
    # 根据列名判断
    header_str = ' '.join(header).lower()
    if 'throughput' in header_str:
        return 'e2e_prefill'
    else:
        return 'kernel_model'


def get_id_columns(table_type: str) -> List[str]:
    """获取作为标识的列 (需要省略重复值的列)"""
    if table_type == 'kernel_square':
        return ['GPU']
    else:
        return ['GPU', 'Model']


def get_speedup_columns(header: List[str]) -> List[str]:
    """获取 speedup 列 (2:4, 2:6, 等)"""
    speedup_cols = []
    for col in header:
        if col.startswith('2:'):
            speedup_cols.append(col)
    return speedup_cols


# =============================================================================
# 工具函数
# =============================================================================

def read_csv(csv_path: Path) -> Tuple[List[str], List[List[str]]]:
    """读取 CSV 文件，返回 (表头, 数据行)"""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def is_speedup_column(col_name: str) -> bool:
    """判断是否是 speedup 列"""
    return col_name.startswith('2:')


def get_cell_color(value: str, col_name: str, row_idx: int) -> str:
    """获取单元格背景色"""
    base_color = ROW_EVEN_COLOR if row_idx % 2 == 0 else ROW_ODD_COLOR
    
    if is_speedup_column(col_name) and value.strip():
        try:
            val = float(value)
            if val >= 1.5:
                return SPEEDUP_GREAT_COLOR
            elif val >= 1.0:
                return SPEEDUP_GOOD_COLOR
        except:
            pass
    
    return base_color


def build_display_rows(rows: List[List[str]], header: List[str], 
                       id_columns: List[str], start_global_idx: int) -> List[List[str]]:
    """
    构建显示行：省略重复的 ID 列值
    """
    if not hasattr(build_display_rows, 'prev_ids'):
        build_display_rows.prev_ids = {}
    
    id_indices = []
    for col_name in id_columns:
        if col_name in header:
            id_indices.append(header.index(col_name))
    
    display_rows = []
    for local_idx, row in enumerate(rows):
        new_row = list(row)
        
        for col_idx in id_indices:
            if col_idx < len(row):
                current_val = row[col_idx]
                prev_key = f"col_{col_idx}"
                
                if prev_key in build_display_rows.prev_ids and \
                   build_display_rows.prev_ids[prev_key] == current_val:
                    new_row[col_idx] = ''
                else:
                    build_display_rows.prev_ids[prev_key] = current_val
        
        display_rows.append(new_row)
    
    return display_rows


def reset_display_state():
    """重置显示状态 (新文件时调用)"""
    if hasattr(build_display_rows, 'prev_ids'):
        build_display_rows.prev_ids = {}


def calculate_col_widths(header: List[str], num_cols: int, 
                         table_width: float, table_type: str) -> List[float]:
    """计算列宽度 (固定比例分配)"""
    
    # 根据表类型设置各列的相对权重
    weights = []
    for col in header:
        if col == 'GPU':
            weights.append(1.4)
        elif col == 'Model':
            weights.append(2.0)
        elif col == 'M':
            weights.append(0.9)
        elif 'Latency' in col or 'Throughput' in col:
            weights.append(1.6)
        elif col.startswith('2:'):
            weights.append(0.9)
        else:
            weights.append(1.0)
    
    total_weight = sum(weights)
    widths = [w / total_weight * table_width for w in weights]
    
    return widths


# =============================================================================
# 绘制函数
# =============================================================================

def draw_two_row_header(ax, header: List[str], col_widths: List[float],
                        start_x: float, start_y: float, 
                        header_height: float, table_type: str):
    """
    绘制双行表头
    
    - GPU/Model: 跨两行
    - M: 根据表类型，第一行显示 Batch Size / Concurrency，第二行显示 M
    - cuBLASLt: 第一行 cuBLASLt，第二行 Latency (μs) 或 Throughput (token/s)
    - Speedup列: 第一行合并 cuSPARSELt Speedup Ratio，第二行各稀疏度
    """
    row_height = header_height / 2
    
    # 分类列
    id_cols = []  # GPU, Model
    m_col = None  # M 列
    baseline_col = None  # cuBLASLt 列
    speedup_cols = []  # 2:4, 2:6, ...
    
    for i, col in enumerate(header):
        if col in ['GPU', 'Model']:
            id_cols.append((i, col))
        elif col == 'M':
            m_col = (i, col)
        elif 'Latency' in col or 'Throughput' in col:
            baseline_col = (i, col)
        elif col.startswith('2:'):
            speedup_cols.append((i, col))
    
    y1 = start_y
    
    # === GPU/Model 列 (跨两行) ===
    for col_idx, col_name in id_cols:
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        
        rect = mpatches.FancyBboxPatch(
            (x, y1 - header_height), w, header_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        ax.text(x + w/2, y1 - header_height/2, col_name,
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE, fontweight='bold',
                color=HEADER_TEXT_COLOR)
    
    # === M 列 (双行，根据表类型) ===
    if m_col:
        col_idx, col_name = m_col
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        
        rect = mpatches.FancyBboxPatch(
            (x, y1 - header_height), w, header_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        # 第一行: 根据表类型
        if table_type == 'e2e_prefill':
            line1_text = 'Batch Size'
        elif table_type == 'e2e_decode':
            line1_text = 'Concurrency'
        else:
            line1_text = ''
        
        if line1_text:
            ax.text(x + w/2, y1 - row_height/2, line1_text,
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE - 0.5, fontweight='bold',
                    color=HEADER_TEXT_COLOR)
            # 第二行: M
            ax.text(x + w/2, y1 - header_height + row_height/2, 'M',
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='bold',
                    color=HEADER_TEXT_COLOR)
        else:
            # Kernel表: M 跨两行居中
            ax.text(x + w/2, y1 - header_height/2, 'M',
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='bold',
                    color=HEADER_TEXT_COLOR)
    
    # === cuBLASLt 列 (双行) ===
    if baseline_col:
        col_idx, col_name = baseline_col
        x = start_x + sum(col_widths[:col_idx])
        w = col_widths[col_idx]
        
        rect = mpatches.FancyBboxPatch(
            (x, y1 - header_height), w, header_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        # 第一行: cuBLASLt
        ax.text(x + w/2, y1 - row_height/2, 'cuBLASLt',
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE, fontweight='bold',
                color=HEADER_TEXT_COLOR)
        
        # 第二行: Latency (μs) 或 Throughput (token/s)
        if 'Latency' in col_name:
            unit_text = 'Latency (μs)'
        else:
            unit_text = 'Throughput (token/s)'
        
        ax.text(x + w/2, y1 - header_height + row_height/2, unit_text,
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE - 0.5, fontweight='bold',
                color=HEADER_TEXT_COLOR)
    
    # === Speedup 列 (第一行合并，第二行分开) ===
    if speedup_cols:
        first_idx = speedup_cols[0][0]
        last_idx = speedup_cols[-1][0]
        x_start = start_x + sum(col_widths[:first_idx])
        total_w = sum(col_widths[first_idx:last_idx+1])
        
        # 第一行: 合并的 "cuSPARSELt Speedup Ratio"
        rect = mpatches.FancyBboxPatch(
            (x_start, y1 - row_height), total_w, row_height,
            boxstyle="square,pad=0",
            facecolor=HEADER_BG_COLOR_L1,
            edgecolor=GRID_COLOR,
            linewidth=0.5
        )
        ax.add_patch(rect)
        
        ax.text(x_start + total_w/2, y1 - row_height/2, 
                'cuSPARSELt Speedup Ratio',
                ha='center', va='center',
                fontsize=HEADER_FONT_SIZE, fontweight='bold',
                color=HEADER_TEXT_COLOR)
        
        # 第二行: 各个稀疏度
        for col_idx, col_name in speedup_cols:
            x = start_x + sum(col_widths[:col_idx])
            w = col_widths[col_idx]
            
            rect = mpatches.FancyBboxPatch(
                (x, y1 - header_height), w, row_height,
                boxstyle="square,pad=0",
                facecolor=HEADER_BG_COLOR_L2,
                edgecolor=GRID_COLOR,
                linewidth=0.5
            )
            ax.add_patch(rect)
            
            ax.text(x + w/2, y1 - header_height + row_height/2, col_name,
                    ha='center', va='center',
                    fontsize=HEADER_FONT_SIZE, fontweight='bold',
                    color=HEADER_TEXT_COLOR)


def draw_table_page(ax, header: List[str], rows: List[List[str]], 
                    title: str, page_num: int, total_pages: int,
                    col_widths: List[float], table_type: str):
    """在一个 Axes 上绘制表格"""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    num_cols = len(header)
    num_rows = len(rows)
    
    # 表格尺寸
    table_width = sum(col_widths)
    
    # 表格起始位置 (居中)
    start_x = (1 - table_width) / 2
    start_y = 0.97
    
    # 绘制标题
    title_text = title
    if total_pages > 1:
        title_text += f" (Page {page_num}/{total_pages})"
    ax.text(0.5, 0.99, title_text, ha='center', va='top',
            fontsize=TITLE_FONT_SIZE, fontweight='bold')
    
    # 绘制双行表头
    draw_two_row_header(ax, header, col_widths, start_x, start_y, 
                        HEADER_HEIGHT, table_type)
    
    # 绘制数据行
    y = start_y - HEADER_HEIGHT
    for row_idx, row in enumerate(rows):
        x = start_x
        for col_idx in range(num_cols):
            value = row[col_idx] if col_idx < len(row) else ''
            col_name = header[col_idx] if col_idx < len(header) else ''
            
            bg_color = get_cell_color(value, col_name, row_idx)
            
            rect = mpatches.FancyBboxPatch(
                (x, y - ROW_HEIGHT), col_widths[col_idx], ROW_HEIGHT,
                boxstyle="square,pad=0",
                facecolor=bg_color,
                edgecolor=GRID_COLOR,
                linewidth=0.3
            )
            ax.add_patch(rect)
            
            ax.text(x + col_widths[col_idx]/2, y - ROW_HEIGHT/2,
                    value, ha='center', va='center',
                    fontsize=CELL_FONT_SIZE)
            x += col_widths[col_idx]
        
        y -= ROW_HEIGHT
    
    # 添加图例 (仅第一页)
    if page_num == 1:
        legend_y = y - 0.012
        ax.add_patch(mpatches.Rectangle((start_x, legend_y - 0.006), 0.01, 0.006,
                                        facecolor=SPEEDUP_GREAT_COLOR, edgecolor='gray', linewidth=0.3))
        ax.text(start_x + 0.012, legend_y - 0.003, '≥1.5×', fontsize=4, va='center')
        
        ax.add_patch(mpatches.Rectangle((start_x + 0.045, legend_y - 0.006), 0.01, 0.006,
                                        facecolor=SPEEDUP_GOOD_COLOR, edgecolor='gray', linewidth=0.3))
        ax.text(start_x + 0.057, legend_y - 0.003, '≥1.0×', fontsize=4, va='center')


def csv_to_pdf(csv_path: Path, output_path: Path):
    """将 CSV 转换为 PDF"""
    header, rows = read_csv(csv_path)
    
    if not rows:
        print(f"  跳过空文件: {csv_path.name}")
        return
    
    # 检测表格类型
    table_type = detect_table_type(header, csv_path.name)
    id_columns = get_id_columns(table_type)
    
    # 重置显示状态
    reset_display_state()
    
    # 计算需要多少页
    total_rows = len(rows)
    total_pages = (total_rows + MAX_ROWS_PER_PAGE - 1) // MAX_ROWS_PER_PAGE
    
    # 计算列宽 (使用固定表格宽度)
    col_widths = calculate_col_widths(header, len(header), FIXED_TABLE_WIDTH, table_type)
    
    # 获取标题
    title = csv_path.stem.replace('_', ' ').title()
    
    # 创建 PDF
    with PdfPages(output_path) as pdf:
        for page in range(total_pages):
            start_idx = page * MAX_ROWS_PER_PAGE
            end_idx = min(start_idx + MAX_ROWS_PER_PAGE, total_rows)
            page_rows = rows[start_idx:end_idx]
            
            # 构建显示行 (省略重复值)
            display_rows = build_display_rows(page_rows, header, id_columns, start_idx)
            
            # 创建图形 (A4 竖向)
            fig, ax = plt.subplots(figsize=(PAGE_WIDTH, PAGE_HEIGHT))
            fig.subplots_adjust(left=MARGIN/PAGE_WIDTH, right=1-MARGIN/PAGE_WIDTH,
                              top=1-MARGIN/PAGE_HEIGHT, bottom=MARGIN/PAGE_HEIGHT)
            
            draw_table_page(ax, header, display_rows, title, 
                          page + 1, total_pages, col_widths, table_type)
            
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    
    print(f"  ✓ {csv_path.name} -> {output_path.name} ({total_pages} 页, {total_rows} 行)")


def main():
    parser = argparse.ArgumentParser(description="CSV 转 PDF 表格工具")
    parser.add_argument('--file', type=str, default=None,
                        help="指定要转换的 CSV 文件名 (默认: 全部)")
    parser.add_argument('--input', type=str, default=str(INPUT_DIR),
                        help=f"输入目录 (默认: {INPUT_DIR})")
    parser.add_argument('--output', type=str, default=str(OUTPUT_DIR),
                        help=f"输出目录 (默认: {OUTPUT_DIR})")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CSV 转 PDF 表格工具 (紧凑双行表头版)")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print()
    
    # 获取要处理的文件列表
    if args.file:
        csv_files = [input_dir / args.file]
    else:
        csv_files = sorted(input_dir.glob("*.csv"))
    
    if not csv_files:
        print("未找到 CSV 文件!")
        return
    
    print(f"找到 {len(csv_files)} 个 CSV 文件")
    print()
    
    for csv_path in csv_files:
        if not csv_path.exists():
            print(f"  ✗ 文件不存在: {csv_path}")
            continue
        
        output_path = output_dir / (csv_path.stem + ".pdf")
        csv_to_pdf(csv_path, output_path)
    
    print()
    print("=" * 60)
    print("转换完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
