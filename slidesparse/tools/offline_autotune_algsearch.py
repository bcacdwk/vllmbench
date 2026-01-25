#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 统一离线调优与算法搜索脚本

- CUDA cuBLAS: cuBLASLt GEMM 算法搜索
- CUDA cuSPARSE: cuSPARSELt 2:4 稀疏算法搜索
- Triton Dequant: 反量化 + Bias 融合 Kernel
- Triton Quant Slide: 量化 + Slide 融合 Kernel
- Triton Quant Only: 纯量化 Kernel


模型名称约定
============
本脚本与其他 tools 脚本的模型名称约定不同：

1. 输入格式：接受 **base name**（无量化后缀）
   - 推荐：Qwen2.5-0.5B, Llama3.2-1B
   - 也可以带后缀（会被自动去除）：Qwen2.5-0.5B-INT8 → Qwen2.5-0.5B

2. 调优类型：由 --dtype 参数决定（int8/fp8/all），与输入的模型名后缀无关
   - 这是因为 INT8 和 FP8 模型的 NK 配置相同，只是量化方式不同

3. 传递给子脚本：
   - CUDA kernel: 完整 checkpoint 名（如 Qwen2.5-0.5B-INT8）
   - Triton kernel: 任意存在的 checkpoint 名

与其他脚本的区别：
- model_download.py / throughput_benchmark.py：使用 registry key（如 qwen2.5-0.5b-fp8）
- weight_convert_entry.py：使用完整目录名或 registry key
- offline_autotune_algsearch.py（本脚本）：使用 base name，自动按 --dtype 扩展


参数说明:
=========
--model:       模型名称，支持 base name 或带后缀的完整名称
               如 "Qwen2.5-0.5B" 或 "Qwen2.5-0.5B-INT8"
               注意：INT8/FP8 模型的 NK 配置相同，后缀会被忽略
--dtype:       输入数据类型，int8/fp8/all（必须指定）
               指定要调优的量化类型，与模型后缀无关
--outdtype:    输出数据类型，bf16（默认）可选 --inner-32 高精度累加
--Lmax:        最大稀疏长度
--M-quick:     快速 M 模式 [16, 128, 1024, 4096, 16384]
--m_list:      自定义 M 列表
--warmup:      预热次数（默认 25）
--repeat:      重复次数（默认 100）
--kernels:     指定要调优的 Kernel，格式为 "1,1,0,1,1"
               顺序: cuBLAS, cuSPARSE, Triton Dequant, Triton Quant Slide, Triton Quant Only
--skip-build:  跳过编译步骤


用法示例:
=========
# 调优所有 Kernel（INT8 + FP8），模型名支持 base name
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B --dtype all --M-quick

# 也可以带后缀（后缀会被忽略，实际按 --dtype 指定的类型调优）
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B-INT8 --dtype all --M-quick

# 只调优 INT8
python3 offline_autotune_algsearch.py --model Llama3.2-1B --dtype int8 --M-quick

# 只调优 CUDA Kernel（cuBLAS + cuSPARSE），使用高精度累加
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B --dtype int8 --inner-32 --kernels 1,1,0,0,0

# 只调优 Triton Kernel
python3 offline_autotune_algsearch.py --model Llama3.2-1B --dtype fp8 --kernels 0,0,1,1,1

# 多模型调优
python3 offline_autotune_algsearch.py --model Qwen2.5-0.5B,Llama3.2-1B --dtype all --M-quick
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import traceback

# 添加项目根目录到 path
_TOOLS_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _TOOLS_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    hw_info,
    DEFAULT_M_LIST,
    M_QUICK_LIST,
    get_nk_list_for_search,
    model_base_name,
    normalize_model_input,
    find_any_model_checkpoint,
    find_model_checkpoint_for_dtype,
)

from slidesparse.tools.utils import (
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
    Colors,
)


# =============================================================================
# 常量定义
# =============================================================================

# Kernel 名称列表（顺序固定）
KERNEL_NAMES = [
    "cuBLASLt GEMM",
    "cuSPARSELt GEMM",
    "Triton Dequant + Bias",
    "Triton Quant + Slide",
    "Triton Quant Only",
]

# Kernel 脚本路径
KERNEL_SCRIPTS = {
    "cublaslt": _SLIDESPARSE_ROOT / "search" / "cuBLASLt_AlgSearch" / "alg_search.py",
    "cusparselt": _SLIDESPARSE_ROOT / "search" / "cuSPARSELt_AlgSearch" / "alg_search.py",
    "triton_dequant": _SLIDESPARSE_ROOT / "csrc" / "fused_dequant_bias_triton" / "autotune_autogen_dequant_bias.py",
    "triton_quant_slide": _SLIDESPARSE_ROOT / "csrc" / "fused_quant_slide_triton" / "autotune_autogen_quant_slide.py",
    "triton_quant_only": _SLIDESPARSE_ROOT / "csrc" / "quant_only_triton" / "autotune_autogen_quant_only.py",
}

# Build 脚本路径
BUILD_SCRIPTS = {
    "cublaslt": _SLIDESPARSE_ROOT / "csrc" / "cublaslt_gemm" / "build_cublaslt.py",
    "cusparselt": _SLIDESPARSE_ROOT / "csrc" / "cusparselt_gemm" / "build_cusparselt.py",
    "compress": _SLIDESPARSE_ROOT / "weight_convert" / "build_compress.py",
}

# 默认模型列表（使用 base name）
DEFAULT_MODELS = ["Qwen2.5-0.5B", "Llama3.2-1B"]

# 默认 warmup/repeat
DEFAULT_WARMUP = 25
DEFAULT_REPEAT = 100


# =============================================================================
# 工具函数
# =============================================================================

def parse_kernel_mask(mask_str: str) -> List[bool]:
    """
    解析 Kernel mask 字符串
    
    Args:
        mask_str: 格式为 "1,1,0,1,1" 的字符串
        
    Returns:
        布尔列表，表示哪些 Kernel 需要调优
    """
    parts = mask_str.split(",")
    if len(parts) != 5:
        raise ValueError(f"Kernel mask 必须有 5 个值，当前: {mask_str}")
    return [int(p.strip()) == 1 for p in parts]


def get_dtype_for_cuda(dtype: str, inner_32: bool) -> Tuple[str, str]:
    """
    获取 CUDA Kernel 的 dtype 和 outdtype
    
    Args:
        dtype: 输入类型 (int8/fp8)
        inner_32: 是否使用高精度累加
        
    Returns:
        (dtype, outdtype)
    """
    if inner_32:
        if dtype == "int8":
            return "int8", "int32"
        else:  # fp8
            return "fp8e4m3", "fp32"
    else:
        if dtype == "int8":
            # cuBLASLt INT8 默认 int32，cuSPARSELt INT8 默认 bf16
            return "int8", "bf16"
        else:  # fp8
            return "fp8e4m3", "bf16"


def run_subprocess(cmd: List[str], name: str) -> Tuple[bool, str]:
    """
    运行子进程并捕获输出
    
    Args:
        cmd: 命令列表
        name: 进程名称（用于日志）
        
    Returns:
        (success, output)
    """
    try:
        print_info(f"执行: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 小时超时
        )
        output = result.stdout + result.stderr
        if result.returncode != 0:
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        return False, f"[{name}] 超时（超过 1 小时）"
    except Exception as e:
        return False, f"[{name}] 异常: {str(e)}\n{traceback.format_exc()}"


# =============================================================================
# Build 步骤
# =============================================================================

def run_build_step(force: bool = True) -> bool:
    """
    运行编译步骤
    
    Args:
        force: 是否强制重新编译
        
    Returns:
        是否成功
    """
    print_header("Step 0: 编译 CUDA 扩展")
    
    success_count = 0
    total_count = len(BUILD_SCRIPTS)
    
    for name, script_path in BUILD_SCRIPTS.items():
        print_subheader(f"编译 {name}")
        
        if not script_path.exists():
            print_error(f"脚本不存在: {script_path}")
            continue
        
        cmd = [sys.executable, str(script_path), "build"]
        if force:
            cmd.append("--force")
        
        success, output = run_subprocess(cmd, name)
        if success:
            print_success(f"{name} 编译成功")
            success_count += 1
        else:
            print_error(f"{name} 编译失败:")
            print(output)
    
    return success_count == total_count


# =============================================================================
# CUDA Kernel 调优
# =============================================================================

def run_cuda_tune(
    kernel_type: str,  # "cublaslt" or "cusparselt"
    dtype: str,
    outdtype: str,
    model: str,
    Lmax: Optional[int],
    m_quick: bool,
    m_list: Optional[List[int]],
    warmup: int,
    repeat: int,
) -> Tuple[bool, str]:
    """
    运行 CUDA Kernel 调优
    
    Returns:
        (success, message)
    """
    script_path = KERNEL_SCRIPTS[kernel_type]
    
    if not script_path.exists():
        return False, f"脚本不存在: {script_path}"
    
    cmd = [
        sys.executable, str(script_path),
        "--dtype", dtype,
        "--outdtype", outdtype,
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--compile",  # 确保编译
    ]
    
    if Lmax:
        cmd.extend(["--Lmax", str(Lmax)])
    
    if m_quick:
        cmd.append("--M-quick")
    elif m_list:
        cmd.extend(["--m_list", ",".join(map(str, m_list))])
    
    return run_subprocess(cmd, kernel_type)


# =============================================================================
# Triton Kernel 调优
# =============================================================================

def run_triton_tune(
    kernel_type: str,  # "triton_dequant", "triton_quant_slide", "triton_quant_only"
    model: str,
    Lmax: Optional[int],
    m_quick: bool,
    m_list: Optional[List[int]],
    warmup: int,
    repeat: int,
) -> Tuple[bool, str]:
    """
    运行 Triton Kernel 调优
    
    Returns:
        (success, message)
    """
    script_path = KERNEL_SCRIPTS[kernel_type]
    
    if not script_path.exists():
        return False, f"脚本不存在: {script_path}"
    
    cmd = [
        sys.executable, str(script_path),
        "--model", model,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
    ]
    
    if Lmax:
        cmd.extend(["--Lmax", str(Lmax)])
    
    if m_quick:
        cmd.append("--M-quick")
    elif m_list:
        cmd.extend(["--m_list", ",".join(map(str, m_list))])
    
    return run_subprocess(cmd, kernel_type)


# =============================================================================
# 主调优流程
# =============================================================================

def run_autotune(
    dtypes: List[str],
    outdtype: str,
    inner_32: bool,
    models: List[str],  # base names
    Lmax: Optional[int],
    m_quick: bool,
    m_list: Optional[List[int]],
    warmup: int,
    repeat: int,
    kernel_mask: List[bool],
    skip_build: bool,
) -> dict:
    """
    运行完整的调优流程
    
    Args:
        models: 模型 base name 列表（不带量化后缀）
    
    Returns:
        结果字典 {kernel_name: {model: (success, message)}}
    """
    results = {}
    
    # Step 0: 编译（如果没有跳过）
    if not skip_build:
        if not run_build_step(force=True):
            print_warning("部分编译失败，继续调优...")
    else:
        print_info("跳过编译步骤")
    
    # Step 1-5: 按 Kernel 顺序调优
    kernel_keys = ["cublaslt", "cusparselt", "triton_dequant", "triton_quant_slide", "triton_quant_only"]
    
    for idx, (kernel_key, kernel_name, enabled) in enumerate(zip(kernel_keys, KERNEL_NAMES, kernel_mask)):
        step_num = idx + 1
        
        if not enabled:
            print_header(f"Step {step_num}: {kernel_name} [跳过]")
            continue
        
        print_header(f"Step {step_num}: {kernel_name}")
        results[kernel_key] = {}
        
        for base_name in models:
            print_subheader(f"模型: {base_name}")
            
            # 找到任意一个存在的 checkpoint 来获取 NK 配置
            # （INT8 和 FP8 的 NK 相同，只需找到一个即可）
            ckpt_path, ckpt_name = find_any_model_checkpoint(base_name)
            if ckpt_path is None:
                print_error(f"未找到模型 '{base_name}' 的 checkpoint 目录")
                results[kernel_key][base_name] = (False, f"未找到 checkpoint")
                continue
            
            # 获取 NK 配置
            try:
                nk_list, _ = get_nk_list_for_search(ckpt_name, Lmax)
                print_info(f"NK 组合数: {len(nk_list)} (from {ckpt_name})")
            except ValueError as e:
                print_error(f"模型验证失败: {e}")
                results[kernel_key][base_name] = (False, str(e))
                continue
            
            # CUDA Kernel：按 dtype 分别调优
            if kernel_key in ["cublaslt", "cusparselt"]:
                for dtype in dtypes:
                    # 查找该 dtype 对应的 checkpoint（用于命名输出文件）
                    dtype_ckpt = find_model_checkpoint_for_dtype(base_name, dtype)
                    if dtype_ckpt is None:
                        print_warning(f"未找到 {base_name} 的 {dtype.upper()} checkpoint，跳过")
                        continue
                    model_name_for_tune = dtype_ckpt.name  # 如 "Qwen2.5-0.5B-INT8"
                    
                    actual_dtype, actual_outdtype = get_dtype_for_cuda(dtype, inner_32)
                    
                    # cuBLASLt INT8 强制 int32
                    if kernel_key == "cublaslt" and dtype == "int8":
                        actual_outdtype = "int32"
                    
                    print_info(f"dtype={actual_dtype}, outdtype={actual_outdtype}")
                    
                    success, output = run_cuda_tune(
                        kernel_key,
                        actual_dtype,
                        actual_outdtype,
                        model_name_for_tune,  # 传递完整模型名
                        Lmax,
                        m_quick,
                        m_list,
                        warmup,
                        repeat,
                    )
                    
                    key = f"{base_name}_{dtype}"
                    results[kernel_key][key] = (success, output)
                    
                    if success:
                        print_success(f"{kernel_name} ({dtype}) 完成")
                    else:
                        print_error(f"{kernel_name} ({dtype}) 失败:")
                        print(output[-2000:] if len(output) > 2000 else output)
            
            # Triton Kernel：使用任意存在的 checkpoint 名
            else:
                # Triton kernel 对 dtype 不敏感，使用任意找到的 checkpoint 名
                success, output = run_triton_tune(
                    kernel_key,
                    ckpt_name,  # 使用找到的 checkpoint 名
                    Lmax,
                    m_quick,
                    m_list,
                    warmup,
                    repeat,
                )
                
                results[kernel_key][base_name] = (success, output)
                
                if success:
                    print_success(f"{kernel_name} 完成")
                else:
                    print_error(f"{kernel_name} 失败:")
                    print(output[-2000:] if len(output) > 2000 else output)
    
    return results


def print_summary(results: dict, kernel_mask: List[bool]) -> None:
    """打印调优总结"""
    print_header("调优总结")
    
    kernel_keys = ["cublaslt", "cusparselt", "triton_dequant", "triton_quant_slide", "triton_quant_only"]
    
    success_total = 0
    fail_total = 0
    skip_total = 0
    
    for idx, (kernel_key, kernel_name, enabled) in enumerate(zip(kernel_keys, KERNEL_NAMES, kernel_mask)):
        if not enabled:
            print(f"  {kernel_name}: {Colors.YELLOW}[跳过]{Colors.NC}")
            skip_total += 1
            continue
        
        if kernel_key not in results:
            print(f"  {kernel_name}: {Colors.RED}[未执行]{Colors.NC}")
            fail_total += 1
            continue
        
        kernel_results = results[kernel_key]
        success_count = sum(1 for s, _ in kernel_results.values() if s)
        fail_count = len(kernel_results) - success_count
        
        success_total += success_count
        fail_total += fail_count
        
        if fail_count == 0:
            status = f"{Colors.GREEN}[全部成功]{Colors.NC} ({success_count}/{len(kernel_results)})"
        elif success_count == 0:
            status = f"{Colors.RED}[全部失败]{Colors.NC} ({fail_count}/{len(kernel_results)})"
        else:
            status = f"{Colors.YELLOW}[部分成功]{Colors.NC} ({success_count}/{len(kernel_results)})"
        
        print(f"  {kernel_name}: {status}")
    
    print()
    print(f"总计: 成功 {success_total}, 失败 {fail_total}, 跳过 {skip_total}")


# =============================================================================
# 主函数
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="SlideSparse 统一离线调优与算法搜索脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # 必须参数
    parser.add_argument(
        "--dtype", required=True, choices=["int8", "fp8", "all"],
        help="输入数据类型: int8, fp8, 或 all（两者都调优）"
    )
    
    # CUDA 特有参数
    parser.add_argument(
        "--outdtype", default="bf16", choices=["bf16", "fp32", "int32"],
        help="输出数据类型（默认 bf16，cuBLAS+INT8 会自动 fallback 到 int32）"
    )
    parser.add_argument(
        "--inner-32", action="store_true", dest="inner_32",
        help="使用高精度累加: FP8→FP32, INT8→INT32（仅对 CUDA Kernel 生效）"
    )
    
    # 公共参数
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型 base name（如 Qwen2.5-0.5B）或带后缀名称（后缀会被忽略）。逗号分隔多个模型。"
    )
    parser.add_argument(
        "--Lmax", type=int, default=None,
        help="最大稀疏长度 L（会生成 L=4,6,...,Lmax 的所有 NK）"
    )
    parser.add_argument(
        "--M-quick", action="store_true", dest="m_quick",
        help="M-quick 模式: 使用固定 M 列表 [16, 128, 1024, 4096, 16384]"
    )
    parser.add_argument(
        "--m_list", type=str, default=None,
        help="自定义 M 列表，逗号分隔（如 16,128,512,2048,16384）"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP,
        help=f"预热次数（默认 {DEFAULT_WARMUP}）"
    )
    parser.add_argument(
        "--repeat", type=int, default=DEFAULT_REPEAT,
        help=f"重复次数（默认 {DEFAULT_REPEAT}）"
    )
    
    # Kernel 选择
    parser.add_argument(
        "--kernels", type=str, default="1,1,1,1,1",
        help='要调优的 Kernel，格式 "1,1,0,1,1"（顺序: cuBLAS,cuSPARSE,Dequant,QuantSlide,QuantOnly）'
    )
    
    # 其他选项
    parser.add_argument(
        "--skip-build", action="store_true", dest="skip_build",
        help="跳过编译步骤（假设已编译）"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="仅显示配置信息，不执行调优"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 解析参数
    if args.dtype == "all":
        dtypes = ["int8", "fp8"]
    else:
        dtypes = [args.dtype]
    
    # 解析模型列表，并标准化为 base name
    if args.model:
        raw_models = [m.strip() for m in args.model.split(",")]
    else:
        raw_models = DEFAULT_MODELS
        print_warning(f"未指定模型，使用默认: {raw_models}")
    
    # 标准化模型名称：提取 base name，验证存在性
    models = []
    model_hints = {}  # 记录用户指定的量化类型（如果有）
    for raw in raw_models:
        try:
            base, quant_hint = normalize_model_input(raw)
            if base not in models:  # 去重
                models.append(base)
                if quant_hint:
                    model_hints[base] = quant_hint
            # 如果用户输入了带后缀的名称，提示会被忽略
            if raw != base:
                print_info(f"模型 '{raw}' → base name '{base}' (后缀将被忽略，按 --dtype 调优)")
        except ValueError as e:
            print_error(str(e))
            return 1
    
    # 解析 M 列表
    m_list = None
    if args.m_list:
        m_list = [int(x.strip()) for x in args.m_list.split(",")]
    
    # 解析 Kernel mask
    try:
        kernel_mask = parse_kernel_mask(args.kernels)
    except ValueError as e:
        print_error(str(e))
        return 1
    
    # 显示配置信息
    print_header("SlideSparse 统一离线调优")
    print(f"  GPU:           {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"  Python:        {hw_info.python_tag}")
    print(f"  CUDA:          {hw_info.cuda_tag}")
    print(f"  Arch:          {hw_info.arch_tag}")
    print()
    print(f"  数据类型:      {dtypes}")
    print(f"  输出类型:      {args.outdtype}")
    print(f"  高精度累加:    {'是' if args.inner_32 else '否'}")
    print(f"  模型 (base):   {models}")
    print(f"  Lmax:          {args.Lmax or '未指定'}")
    print(f"  M-quick:       {'是' if args.m_quick else '否'}")
    print(f"  M 列表:        {m_list or ('M_QUICK_LIST' if args.m_quick else 'DEFAULT_M_LIST')}")
    print(f"  Warmup/Repeat: {args.warmup}/{args.repeat}")
    print()
    print("  Kernel 调优:")
    for name, enabled in zip(KERNEL_NAMES, kernel_mask):
        status = f"{Colors.GREEN}✓{Colors.NC}" if enabled else f"{Colors.RED}✗{Colors.NC}"
        print(f"    {status} {name}")
    
    if args.info:
        return 0
    
    # 运行调优
    results = run_autotune(
        dtypes=dtypes,
        outdtype=args.outdtype,
        inner_32=args.inner_32,
        models=models,
        Lmax=args.Lmax,
        m_quick=args.m_quick,
        m_list=m_list,
        warmup=args.warmup,
        repeat=args.repeat,
        kernel_mask=kernel_mask,
        skip_build=args.skip_build,
    )
    
    # 打印总结
    print_summary(results, kernel_mask)
    
    # 检查是否有失败
    has_failure = any(
        not success
        for kernel_results in results.values()
        for success, _ in kernel_results.values()
    )
    
    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
