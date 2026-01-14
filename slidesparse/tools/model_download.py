#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Model Download Script

用于批量下载 vLLM 基线测试所需的 W8A8 量化模型。
支持 INT8 (quantized.w8a8) 和 FP8 (FP8-dynamic) 两种格式。

Usage:
    python3 model_download.py [选项]

Options:
    -a, --all           下载所有模型
    -i, --int8          仅下载 INT8 模型
    -f, --fp8           仅下载 FP8 模型
    -q, --qwen          仅下载 Qwen2.5 系列
    -l, --llama         仅下载 Llama3.2 系列
    -m, --model NAME    下载指定模型 (如: qwen2.5-7b-int8)
    -c, --check         检查已下载模型状态
    -s, --size          显示模型预估大小
    -h, --help          显示帮助信息

Examples:
    python3 model_download.py --all                    # 下载全部模型
    python3 model_download.py --int8 --qwen            # 下载 Qwen INT8 模型
    python3 model_download.py --model qwen2.5-7b-fp8   # 下载指定模型
    python3 model_download.py --check                  # 检查下载状态
"""

import sys
import argparse
from pathlib import Path

# 确保可以导入 slidesparse
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    model_registry,
    list_models,
    MODEL_SIZE_GB,
)
from slidesparse.tools.utils import (
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    CHECKPOINT_DIR,
    check_hf_cli,
    download_model,
    print_model_status,
)


def show_model_sizes():
    """显示模型预估大小"""
    print_header("模型预估大小")
    
    print("模型大小参考:")
    for size, gb in sorted(MODEL_SIZE_GB.items(), key=lambda x: x[1]):
        print(f"  - {size} 模型: ~{gb:.1f} GB")
    
    print()
    print("估算总大小 (所有模型):")
    
    int8_total = sum(
        MODEL_SIZE_GB.get(e.size.upper(), 0)
        for e in model_registry.list(quant="int8")
    )
    fp8_total = sum(
        MODEL_SIZE_GB.get(e.size.upper(), 0)
        for e in model_registry.list(quant="fp8")
    )
    
    print(f"  - INT8 全部: ~{int8_total:.1f} GB")
    print(f"  - FP8 全部:  ~{fp8_total:.1f} GB")
    print(f"  - 总计:      ~{int8_total + fp8_total:.1f} GB")


def download_models(
    quant_filter: str | None = None,
    family_filter: str | None = None,
    specific_model: str | None = None,
):
    """
    下载模型
    
    Args:
        quant_filter: 量化类型过滤 (int8, fp8)
        family_filter: 模型系列过滤 (qwen, llama)
        specific_model: 指定模型 key
    """
    # 检查 HF CLI
    if not check_hf_cli():
        print_error("HuggingFace CLI 未安装")
        print_info("请运行: pip install -U huggingface_hub")
        sys.exit(1)
    
    # 确保 checkpoints 目录存在
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 确定要下载的模型
    if specific_model:
        # 下载指定模型
        entry = model_registry.get(specific_model)
        if entry is None:
            print_error(f"模型不存在: {specific_model}")
            print_info(f"可用模型: {', '.join(list_models())}")
            sys.exit(1)
        
        models_to_download = [entry]
    else:
        # 按过滤条件获取模型列表
        models_to_download = model_registry.list(
            family=family_filter,
            quant=quant_filter,
        )
    
    if not models_to_download:
        print_warning("没有符合条件的模型")
        return
    
    # 显示下载计划
    total_gb = sum(e.estimated_gb for e in models_to_download)
    print_header(f"准备下载 {len(models_to_download)} 个模型 (~{total_gb:.1f} GB)")
    
    for entry in models_to_download:
        print(f"  - {entry.local_name} ({entry.estimated_gb:.1f} GB)")
    print()
    
    # 开始下载
    success_count = 0
    failed_models = []
    
    for entry in models_to_download:
        print_header(f"下载: {entry.local_name}")
        print_info(f"HuggingFace: {entry.hf_path}")
        print_info(f"本地目录: {CHECKPOINT_DIR / entry.local_name}")
        print()
        
        success, msg = download_model(entry.key, CHECKPOINT_DIR)
        
        if success:
            print_success(msg)
            success_count += 1
        else:
            print_error(msg)
            failed_models.append(entry.key)
        
        print()
    
    # 显示结果
    print_header("下载完成")
    print(f"成功: {success_count}/{len(models_to_download)}")
    
    if failed_models:
        print_warning(f"失败: {', '.join(failed_models)}")
    
    # 显示最终状态
    print_model_status(CHECKPOINT_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse Model Download Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
可用模型:

  INT8 模型 (quantized.w8a8):
""" + "\n".join(f"    - {key}" for key in list_models(quant="int8")) + """

  FP8 模型 (FP8-dynamic):
""" + "\n".join(f"    - {key}" for key in list_models(quant="fp8")) + """

  BitNet 模型 (BF16):
    - bitnet1.58-2b-bf16

示例:
  %(prog)s --all                    # 下载全部模型 (INT8 + FP8)
  %(prog)s --int8 --qwen            # 下载 Qwen INT8 模型
  %(prog)s --bitnet                 # 下载 BitNet BF16 模型
  %(prog)s --model qwen2.5-7b-fp8   # 下载指定模型
  %(prog)s --check                  # 检查下载状态
"""
    )
    
    # 模型选择
    model_group = parser.add_argument_group("模型选择")
    model_group.add_argument(
        "-a", "--all", action="store_true",
        help="下载所有模型 (INT8 + FP8)"
    )
    model_group.add_argument(
        "-i", "--int8", action="store_true",
        help="仅下载 INT8 模型"
    )
    model_group.add_argument(
        "-f", "--fp8", action="store_true",
        help="仅下载 FP8 模型"
    )
    model_group.add_argument(
        "-q", "--qwen", action="store_true",
        help="仅下载 Qwen2.5 系列"
    )
    model_group.add_argument(
        "-l", "--llama", action="store_true",
        help="仅下载 Llama3.2 系列"
    )
    model_group.add_argument(
        "-b", "--bitnet", action="store_true",
        help="下载 BitNet BF16 模型 (microsoft)"
    )
    model_group.add_argument(
        "-m", "--model", type=str, metavar="NAME",
        help="下载指定模型"
    )
    
    # 其他选项
    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument(
        "-c", "--check", action="store_true",
        help="检查已下载模型状态"
    )
    other_group.add_argument(
        "-s", "--size", action="store_true",
        help="显示模型预估大小"
    )
    
    args = parser.parse_args()
    
    # 显示大小信息
    if args.size:
        show_model_sizes()
        return 0
    
    # 检查模式
    if args.check:
        print_model_status(CHECKPOINT_DIR)
        return 0
    
    # 处理 BitNet 特殊情况
    if args.bitnet:
        download_models(specific_model="bitnet1.58-2b-bf16")
        return 0
    
    # 确定过滤条件
    quant_filter = None
    family_filter = None
    
    if args.int8 and not args.fp8:
        quant_filter = "int8"
    elif args.fp8 and not args.int8:
        quant_filter = "fp8"
    elif args.all:
        quant_filter = None  # 下载所有
    elif not args.model:
        # 没有指定任何选项，显示帮助
        parser.print_help()
        return 0
    
    if args.qwen and not args.llama:
        family_filter = "qwen"
    elif args.llama and not args.qwen:
        family_filter = "llama"
    
    # 执行下载
    download_models(
        quant_filter=quant_filter,
        family_filter=family_filter,
        specific_model=args.model,
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
