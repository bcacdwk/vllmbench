#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 离线权重转换入口脚本

完整流程：检查模型 → Prune → Slide → Compress → 保存

将 HuggingFace 的 compressed-tensor 格式模型（FP8/INT8）转换为
支持 cuSPARSELt 2:4 稀疏加速的格式。

Usage:
    # 处理单个模型
    python entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8
    
    # 处理指定目录
    python entry.py --input /path/to/model --Z 2 --L 8
    
    # 仅执行 prune+slide（跳过压缩）
    python entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8 --skip-compress
    
    # 处理所有已下载的 FP8 模型
    python entry.py --all --quant fp8 --Z 2 --L 8
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch

# 导入本地模块
from utils import (
    SlideSparseConfig,
    CHECKPOINT_DIR,
    CHECKPOINT_SLIDESPARSE_DIR,
    get_model_safetensors_path,
    get_all_safetensors_files,
    get_output_model_dir,
    load_model_config,
    save_model_config,
    save_slidesparse_config,
    copy_non_weight_files,
    load_safetensors,
    save_safetensors,
    is_target_layer,
    detect_weight_dtype,
    verify_ZL_sparsity,
    verify_2to4_sparsity,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
)

from prune import prune_tensor, quant_and_prune_tensor_bitnet
from slide import slide_tensor
from compress import compress_tensor, compress_tensor_fake, check_cusparselt_available

# 添加项目路径以导入 slidesparse.utils
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from slidesparse.utils import model_registry, check_model_downloaded
    HAS_MODEL_REGISTRY = True
except ImportError:
    HAS_MODEL_REGISTRY = False
    print_warning("slidesparse.utils not available, --model option disabled")


# =============================================================================
# 完整流程处理
# =============================================================================

def process_single_tensor(
    tensor: torch.Tensor,
    config: SlideSparseConfig,
    mode: str = "magnitude",
    skip_prune: bool = False,
    skip_slide: bool = False,
    skip_compress: bool = False,
    use_real_cusparselt: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    处理单个权重张量的完整流程
    
    Args:
        tensor: 输入权重 [N, K]
        config: SlideSparse 配置
        mode: 剪枝模式
        skip_prune: 跳过剪枝
        skip_slide: 跳过滑动
        skip_compress: 跳过压缩
        use_real_cusparselt: 使用真实 cuSPARSELt
        verbose: 详细输出
    
    Returns:
        {
            "tensor": 处理后的张量,
            "metadata": 元数据张量（如果有）,
            "info": 处理信息,
        }
    """
    original_shape = list(tensor.shape)
    original_dtype = detect_weight_dtype(tensor)
    
    result = {
        "original_shape": original_shape,
        "original_dtype": original_dtype,
        "stages": [],
    }
    
    current_tensor = tensor
    
    # Stage 1: Prune
    if not skip_prune:
        if verbose:
            print_info(f"  Stage 1: Pruning ({config.Z}:{config.L}, mode={mode})")
        
        current_tensor = prune_tensor(
            current_tensor, config.Z, config.L, mode
        )
        
        # 验证剪枝结果
        tensor_for_verify = current_tensor.float() if current_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8) else current_tensor
        is_valid, valid_ratio = verify_ZL_sparsity(tensor_for_verify, config.Z, config.L)
        
        result["stages"].append({
            "name": "prune",
            "shape": list(current_tensor.shape),
            "ZL_valid": is_valid,
            "ZL_valid_ratio": valid_ratio,
        })
        
        if verbose:
            status = "✓" if is_valid else f"✗ ({valid_ratio:.2%})"
            print_info(f"    {config.Z}:{config.L} validation: {status}")
    
    # Stage 2: Slide
    if not skip_slide:
        if verbose:
            print_info(f"  Stage 2: Sliding (expand_ratio={config.expand_ratio:.3f})")
        
        current_tensor, slide_meta = slide_tensor(current_tensor, config)
        
        # 验证 2:4 稀疏性
        tensor_for_verify = current_tensor.float() if current_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2, torch.int8) else current_tensor
        is_valid, violation_ratio = verify_2to4_sparsity(tensor_for_verify)
        
        result["stages"].append({
            "name": "slide",
            "shape": list(current_tensor.shape),
            "2to4_valid": is_valid,
            "2to4_violation_ratio": violation_ratio,
            "slide_metadata": slide_meta,
        })
        
        if verbose:
            status = "✓" if is_valid else f"✗ (violation: {violation_ratio:.2%})"
            print_info(f"    Shape: {original_shape} -> {list(current_tensor.shape)}")
            print_info(f"    2:4 validation: {status}")
    
    # Stage 3: Compress
    metadata_tensor = None
    if not skip_compress:
        if verbose:
            print_info(f"  Stage 3: Compressing (cuSPARSELt={'real' if use_real_cusparselt else 'fake'})")
        
        # 需要转换为 INT8
        if current_tensor.dtype != torch.int8:
            if current_tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                tensor_int8 = current_tensor.float().round().clamp(-127, 127).to(torch.int8)
            else:
                tensor_int8 = current_tensor.round().clamp(-127, 127).to(torch.int8)
        else:
            tensor_int8 = current_tensor
        
        if use_real_cusparselt:
            compressed, compress_meta = compress_tensor(tensor_int8, verbose=False)
            result["stages"].append({
                "name": "compress",
                "compressed_size": compressed.shape[0],
                "compress_metadata": compress_meta,
            })
            current_tensor = compressed
        else:
            values, meta_tensor, compress_info = compress_tensor_fake(tensor_int8, verbose=False)
            result["stages"].append({
                "name": "compress_fake",
                "values_shape": list(values.shape),
                "meta_shape": list(meta_tensor.shape),
                "compress_info": compress_info,
            })
            current_tensor = values
            metadata_tensor = meta_tensor
        
        if verbose:
            if use_real_cusparselt:
                print_info(f"    Compressed: {compressed.shape[0]} bytes")
            else:
                print_info(f"    Values: {list(values.shape)}, Meta: {list(meta_tensor.shape)}")
    
    result["tensor"] = current_tensor
    result["metadata_tensor"] = metadata_tensor
    result["final_shape"] = list(current_tensor.shape)
    
    return result


def process_model(
    input_dir: Path,
    output_dir: Path,
    config: SlideSparseConfig,
    mode: str = "magnitude",
    skip_prune: bool = False,
    skip_slide: bool = False,
    skip_compress: bool = False,
    use_real_cusparselt: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    处理完整模型
    
    Args:
        input_dir: 输入模型目录
        output_dir: 输出目录
        config: SlideSparse 配置
        mode: 剪枝模式
        skip_*: 跳过对应阶段
        use_real_cusparselt: 使用真实 cuSPARSELt
        verbose: 详细输出
    
    Returns:
        处理报告
    """
    start_time = time.time()
    
    if verbose:
        print_header(f"Processing: {input_dir.name}")
        print_info(f"Config: {config}")
        print_info(f"Mode: {mode}")
        print_info(f"Output: {output_dir}")
        print()
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制非权重文件
    if verbose:
        print_info("Copying non-weight files...")
    copied_files = copy_non_weight_files(input_dir, output_dir)
    if verbose:
        print_info(f"  Copied: {', '.join(copied_files[:5])}{'...' if len(copied_files) > 5 else ''}")
    
    # 获取所有 safetensors 文件
    safetensors_files = get_all_safetensors_files(input_dir)
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {input_dir}")
    
    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": config.to_dict(),
        "mode": mode,
        "skip_prune": skip_prune,
        "skip_slide": skip_slide,
        "skip_compress": skip_compress,
        "use_real_cusparselt": use_real_cusparselt,
        "files": [],
        "total_layers_processed": 0,
        "total_layers_skipped": 0,
    }
    
    # 处理每个 safetensors 文件
    for sf_path in safetensors_files:
        if verbose:
            print()
            print_info(f"Processing file: {sf_path.name}")
        
        file_report = {
            "file": sf_path.name,
            "layers": [],
            "skipped": [],
        }
        
        # 加载权重
        weights = load_safetensors(sf_path)
        output_weights = {}
        
        for key, tensor in weights.items():
            # 检查是否为目标层
            if not is_target_layer(key):
                output_weights[key] = tensor
                file_report["skipped"].append(key)
                continue
            
            # 检查是否为 2D 张量
            if tensor.dim() != 2:
                output_weights[key] = tensor
                file_report["skipped"].append(key)
                continue
            
            if verbose:
                dtype_str = detect_weight_dtype(tensor)
                print_info(f"\nLayer: {key}")
                print_info(f"  Input: shape={list(tensor.shape)}, dtype={dtype_str}")
            
            # 处理张量
            result = process_single_tensor(
                tensor,
                config,
                mode=mode,
                skip_prune=skip_prune,
                skip_slide=skip_slide,
                skip_compress=skip_compress,
                use_real_cusparselt=use_real_cusparselt,
                verbose=verbose,
            )
            
            # 保存处理后的张量
            output_weights[key] = result["tensor"]
            
            # 如果有元数据张量，也保存
            if result["metadata_tensor"] is not None:
                meta_key = key.replace(".weight", ".weight_meta")
                output_weights[meta_key] = result["metadata_tensor"]
            
            file_report["layers"].append({
                "key": key,
                "result": {
                    "original_shape": result["original_shape"],
                    "final_shape": result["final_shape"],
                    "stages": result["stages"],
                }
            })
        
        # 保存输出
        output_sf_path = output_dir / sf_path.name
        if verbose:
            print()
            print_info(f"Saving: {output_sf_path}")
        
        save_safetensors(output_weights, output_sf_path)
        
        report["files"].append(file_report)
        report["total_layers_processed"] += len(file_report["layers"])
        report["total_layers_skipped"] += len(file_report["skipped"])
    
    # 保存 SlideSparse 配置
    save_slidesparse_config(config, output_dir, extra_info={
        "mode": mode,
        "skip_prune": skip_prune,
        "skip_slide": skip_slide,
        "skip_compress": skip_compress,
    })
    
    elapsed_time = time.time() - start_time
    report["elapsed_time"] = elapsed_time
    
    # 保存处理报告
    report_path = output_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    if verbose:
        print()
        print_header("Summary")
        print_success(f"Processed: {report['total_layers_processed']} layers")
        print_info(f"Skipped: {report['total_layers_skipped']} layers")
        print_info(f"Time: {elapsed_time:.2f}s")
        print_info(f"Report: {report_path}")
    
    return report


# =============================================================================
# 模型查找
# =============================================================================

def find_model_dir(model_key: str) -> Optional[Path]:
    """
    根据模型 key 查找本地目录
    
    Args:
        model_key: 模型 key（如 "qwen2.5-0.5b-fp8"）或本地目录名
    
    Returns:
        模型目录路径，如果不存在返回 None
    """
    # 尝试作为目录名直接查找
    direct_path = CHECKPOINT_DIR / model_key
    if direct_path.is_dir():
        return direct_path
    
    # 尝试通过 model_registry 查找
    if HAS_MODEL_REGISTRY:
        try:
            entry = model_registry.get(model_key)
            if entry:
                local_path = CHECKPOINT_DIR / entry.local_name
                if local_path.is_dir():
                    return local_path
        except (KeyError, AttributeError):
            pass
    
    # 尝试模糊匹配
    for d in CHECKPOINT_DIR.iterdir():
        if d.is_dir() and model_key.lower() in d.name.lower():
            return d
    
    return None


def list_available_models(quant_filter: Optional[str] = None) -> List[Path]:
    """
    列出所有已下载的模型
    
    Args:
        quant_filter: 量化类型过滤（"fp8", "int8"）
    
    Returns:
        模型目录列表
    """
    models = []
    
    if not CHECKPOINT_DIR.exists():
        return models
    
    for d in sorted(CHECKPOINT_DIR.iterdir()):
        if not d.is_dir():
            continue
        
        # 检查是否有 config.json
        if not (d / "config.json").exists():
            continue
        
        # 检查是否有 safetensors 文件
        if not get_all_safetensors_files(d):
            continue
        
        # 过滤
        if quant_filter:
            name_lower = d.name.lower()
            if quant_filter == "fp8" and "fp8" not in name_lower:
                continue
            if quant_filter == "int8" and "int8" not in name_lower:
                continue
        
        models.append(d)
    
    return models


# =============================================================================
# 命令行接口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse 离线权重转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
完整流程:
  1. Prune:    应用 Z:L 稀疏约束 (如 2:8)
  2. Slide:    2:L → 2:4 滑动窗口映射 (K 扩展)
  3. Compress: cuSPARSELt 2:4 压缩 (K 减半)

示例:
  # 处理单个模型（通过 key）
  python entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8
  
  # 处理单个模型（通过路径）
  python entry.py --input /path/to/model --Z 2 --L 8
  
  # 跳过压缩（用于测试）
  python entry.py --model qwen2.5-0.5b-fp8 --Z 2 --L 8 --skip-compress
  
  # 处理所有 FP8 模型
  python entry.py --all --quant fp8 --Z 2 --L 8

维度变化 (2:8, expand_ratio=1.5):
  原始:     [N, K]
  Slide后:  [N, K×1.5]
  压缩后:   [N, K×0.75]
        """
    )
    
    # 输入选项组
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model", "-m",
        type=str,
        help="模型 key 或目录名（如 qwen2.5-0.5b-fp8）",
    )
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="输入模型目录路径",
    )
    input_group.add_argument(
        "--all",
        action="store_true",
        help="处理所有已下载的模型",
    )
    input_group.add_argument(
        "--list",
        action="store_true",
        help="列出可用模型",
    )
    
    # 配置选项
    parser.add_argument(
        "--Z",
        type=int,
        default=2,
        help="稀疏度分子（默认: 2）",
    )
    parser.add_argument(
        "--L",
        type=int,
        default=8,
        help="稀疏度分母（默认: 8）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["magnitude", "random"],
        default="magnitude",
        help="剪枝模式（默认: magnitude）",
    )
    
    # 过滤选项
    parser.add_argument(
        "--quant",
        type=str,
        choices=["fp8", "int8"],
        help="量化类型过滤（用于 --all）",
    )
    
    # 输出选项
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="输出目录（默认: checkpoints_slidesparse/）",
    )
    
    # 阶段控制
    parser.add_argument(
        "--skip-prune",
        action="store_true",
        help="跳过剪枝阶段",
    )
    parser.add_argument(
        "--skip-slide",
        action="store_true",
        help="跳过滑动阶段",
    )
    parser.add_argument(
        "--skip-compress",
        action="store_true",
        help="跳过压缩阶段",
    )
    parser.add_argument(
        "--fake-compress",
        action="store_true",
        help="使用模拟压缩（不调用 cuSPARSELt）",
    )
    
    # 其他选项
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式",
    )
    
    args = parser.parse_args()
    
    # 处理 --list
    if args.list:
        models = list_available_models(args.quant)
        print_header("Available Models")
        for m in models:
            print(f"  - {m.name}")
        print()
        print_info(f"Total: {len(models)} models")
        return 0
    
    # 创建配置
    config = SlideSparseConfig(Z=args.Z, L=args.L)
    
    # 确定要处理的模型
    if args.all:
        models = list_available_models(args.quant)
        if not models:
            print_error("No models found")
            return 1
        
        if not args.quiet:
            print_header(f"Processing {len(models)} models")
            for m in models:
                print(f"  - {m.name}")
            print()
    
    elif args.model:
        model_dir = find_model_dir(args.model)
        if model_dir is None:
            print_error(f"Model not found: {args.model}")
            print_info("Use --list to see available models")
            return 1
        models = [model_dir]
    
    elif args.input:
        input_path = Path(args.input)
        if not input_path.is_dir():
            print_error(f"Directory not found: {args.input}")
            return 1
        models = [input_path]
    
    else:
        print_error("No input specified")
        return 1
    
    # 检查 cuSPARSELt
    use_real_cusparselt = not args.fake_compress
    if use_real_cusparselt and not args.skip_compress:
        if not check_cusparselt_available():
            print_warning("cuSPARSELt not available, using fake compression")
            use_real_cusparselt = False
    
    # 处理每个模型
    success_count = 0
    failed_models = []
    
    for model_dir in models:
        try:
            # 确定输出目录
            if args.output:
                output_dir = Path(args.output) / f"{model_dir.name}-SlideSparse-{args.Z}_{args.L}"
            else:
                output_dir = get_output_model_dir(model_dir, suffix=f"-SlideSparse-{args.Z}_{args.L}")
            
            report = process_model(
                model_dir,
                output_dir,
                config,
                mode=args.mode,
                skip_prune=args.skip_prune,
                skip_slide=args.skip_slide,
                skip_compress=args.skip_compress,
                use_real_cusparselt=use_real_cusparselt,
                verbose=not args.quiet,
            )
            
            success_count += 1
            
        except Exception as e:
            print_error(f"Failed to process {model_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append(model_dir.name)
    
    # 最终总结
    if len(models) > 1 and not args.quiet:
        print()
        print_header("Final Summary")
        print_success(f"Processed: {success_count}/{len(models)} models")
        if failed_models:
            print_error(f"Failed: {', '.join(failed_models)}")
    
    return 0 if not failed_models else 1


if __name__ == "__main__":
    sys.exit(main())
