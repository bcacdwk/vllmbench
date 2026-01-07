#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuSPARSELt Layout 性能测试

测试不同 layout 组合的 SpMM 性能，支持 INT8 和 FP8。

Layout 组合说明：
================
测试 4 种主要 layout 配置 × 2 种输出顺序 = 8 种组合：
1. T/N + Col/Col + (Col/Row)  - opW=T, opA=N, orderW=Col, orderA=Col
2. N/T + Row/Row + (Col/Row)  - opW=N, opA=T, orderW=Row, orderA=Row
3. N/N + Row/Col + (Col/Row)  - opW=N, opA=N, orderW=Row, orderA=Col
4. T/T + Col/Row + (Col/Row)  - opW=T, opA=T, orderW=Col, orderA=Row

运行示例:
CUDA_VISIBLE_DEVICES=0 python3 layout_benchmark.py --dtype int8 --compile
CUDA_VISIBLE_DEVICES=0 python3 layout_benchmark.py --dtype fp8e4m3

"""

import argparse
import ctypes
import ctypes.util
import datetime
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.cpp_extension import load


# === CUDA 版本信息获取 ===

def get_nvidia_smi_cuda_version() -> str:
    """获取 nvidia-smi 显示的 CUDA Version（驱动支持的最高 CUDA 版本）"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    try:
        driver_version = torch.cuda.driver_version()
        if driver_version:
            major = driver_version // 1000
            minor = (driver_version % 1000) // 10
            return f"{major}.{minor}"
    except Exception:
        pass
    
    return "unknown"


def get_cuda_runtime_version() -> str:
    """获取 CUDA runtime 版本"""
    try:
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def get_gpu_short_name() -> str:
    """获取 GPU 名称简化版（如 A100, H100, B200）"""
    prop = torch.cuda.get_device_properties(0)
    full_name = prop.name
    short_name = full_name
    
    nvidia_pos = full_name.find("NVIDIA ")
    if nvidia_pos != -1:
        short_name = full_name[nvidia_pos + 7:]
    
    for sep in [" ", "-"]:
        end_pos = short_name.find(sep)
        if end_pos != -1:
            short_name = short_name[:end_pos]
            break
    
    if not short_name:
        short_name = full_name
        for c in [" ", "-", "/"]:
            short_name = short_name.replace(c, "_")
    
    return short_name


# === cuSPARSELt 动态库加载 ===
_CUSPARSELT_LOADED = False

def ensure_cusparselt_loaded() -> None:
    """优先加载系统或环境变量指定的 cuSPARSELt"""
    global _CUSPARSELT_LOADED
    if _CUSPARSELT_LOADED:
        return

    preferred_paths = []
    env_path = os.environ.get("CUSPARSELT_PATH")
    if env_path:
        preferred_paths.append(env_path)

    preferred_paths.extend([
        "/usr/lib/aarch64-linux-gnu/libcusparseLt.so.0",
        "/usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
        "/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
        "/usr/lib/x86_64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
        "/usr/local/cuda/lib64/libcusparseLt.so.0",
    ])
    found = ctypes.util.find_library("cusparseLt")
    if found:
        preferred_paths.append(found)

    for path in dict.fromkeys(preferred_paths):
        if not path:
            continue
        try:
            lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            getattr(lib, "cusparseLtMatmulAlgSelectionDestroy")
            _CUSPARSELT_LOADED = True
            return
        except (OSError, AttributeError):
            continue

    raise OSError(
        "无法找到兼容的 libcusparseLt，请设置 CUSPARSELT_PATH 或安装 CUDA 12.9+。"
    )


# === 架构检测 ===

ARCH_INFO = {
    8: ("Ampere", "ampere"),
    9: ("Hopper", "hopper"),
    10: ("Blackwell", "blackwell"),
    12: ("Blackwell", "blackwell"),
}

def detect_arch() -> Tuple[str, str, str]:
    """检测 GPU 架构，返回 (arch_name, arch_suffix, sm_code)"""
    prop = torch.cuda.get_device_properties(0)
    major = prop.major
    sm_code = f"sm_{major}{prop.minor}"
    
    if major in ARCH_INFO:
        name, suffix = ARCH_INFO[major]
        return name, suffix, sm_code
    
    return f"SM{major}{prop.minor}", f"sm{major}{prop.minor}", sm_code


# === 扩展编译/加载 ===

def load_extension(verbose: bool = True, force_compile: bool = False) -> object:
    """加载 CUDA 扩展"""
    if verbose:
        print("[1/4] 加载 cuSPARSELt 库...", end=" ", flush=True)
    ensure_cusparselt_loaded()
    if verbose:
        print("✓", flush=True)
    
    gpu_short_name = get_gpu_short_name()
    _, _, sm_code = detect_arch()
    prop = torch.cuda.get_device_properties(0)
    ext_name = f"layout_benchmark_cusparselt_{gpu_short_name}_cc{prop.major}{prop.minor}"
    so_pattern = f"{ext_name}*.so"
    
    src_path = Path(__file__).parent / "layout_benchmark_cusparselt.cu"
    build_dir = Path(__file__).parent / "build_so_files"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    existing_so = list(build_dir.glob(so_pattern))
    need_compile = force_compile
    if not need_compile:
        if not existing_so:
            need_compile = True
        else:
            need_compile = src_path.stat().st_mtime > existing_so[0].stat().st_mtime
    
    if not need_compile and existing_so:
        if verbose:
            print(f"[2/4] 加载 {gpu_short_name} 扩展...", end=" ", flush=True)
        import importlib.util
        spec = importlib.util.spec_from_file_location(ext_name, str(existing_so[0]))
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        if verbose:
            print(f"✓ ({existing_so[0].name})", flush=True)
        return ext
    else:
        if verbose:
            reason = "强制" if force_compile else ("首次" if not existing_so else "源文件已更新")
            print(f"[2/4] 编译 {gpu_short_name} 扩展（{reason}）...", end=" ", flush=True)
        
        ext = load(
            name=ext_name,
            sources=[str(src_path)],
            extra_cuda_cflags=["-O3", f"-arch={sm_code}"],
            extra_ldflags=["-lcusparseLt", "-lnvrtc", "-ldl"],
            verbose=False,
            build_directory=str(build_dir),
            with_cuda=True,
        )
        
        for pattern in [".ninja_deps", ".ninja_log", "build.ninja", "*.o"]:
            for f in build_dir.glob(pattern):
                f.unlink(missing_ok=True)
        
        if verbose:
            print("✓", flush=True)
        return ext


# === Layout 配置 ===

# 4 种主要 layout 配置
LAYOUT_CONFIGS = [
    {"name": "TN+CC", "opW": "T", "opA": "N", "orderW": "Col", "orderA": "Col"},
    {"name": "NT+RR", "opW": "N", "opA": "T", "orderW": "Row", "orderA": "Row"},
    {"name": "NN+RC", "opW": "N", "opA": "N", "orderW": "Row", "orderA": "Col"},
    {"name": "TT+CR", "opW": "T", "opA": "T", "orderW": "Col", "orderA": "Row"},
]

# 输出矩阵顺序
OUTPUT_ORDERS = ["Col", "Row"]


# === 默认配置 ===

def default_nk_list() -> List[Tuple[int, int, str]]:
    """返回 (N, K, name) 列表"""
    return [
        (3840, 2560, "Wqkv"),
        (2560, 2560, "Wo"),
        (13824, 2560, "W13"),
        (2560, 6912, "W2"),
    ]


def default_m_list() -> List[int]:
    return [16, 256, 2048, 16384]


# === 支持的 dtype ===
SUPPORTED_DTYPES = ["int8", "fp8e4m3"]


# === 测试运行 ===

def run_layout_benchmark(ext, dtype: str, nk_list: List[Tuple[int, int, str]], 
                         m_list: List[int], warmup: int, repeat: int,
                         verbose: bool = True) -> Dict:
    """
    运行所有 layout 组合的性能测试。
    
    返回结构：
    {
        "dtype": str,
        "nk_list": [...],
        "m_list": [...],
        "results": {
            "(N,K)": {
                "M": {
                    "layout_name+orderR": {
                        "supported": bool,
                        "max_alg_id": int,
                        "top3": [(alg_id, lat_us, tops), ...]
                    }
                }
            }
        }
    }
    """
    results = {}
    total_nk = len(nk_list)
    
    for nk_idx, (N, K, nk_name) in enumerate(nk_list):
        if verbose:
            print(f"  NK {nk_idx+1}/{total_nk}: ({N}, {K}) - {nk_name}")
        
        nk_key = f"({N},{K})"
        results[nk_key] = {"name": nk_name}
        
        for M in m_list:
            if verbose:
                print(f"    M={M}:", end=" ", flush=True)
            
            m_results = {}
            
            for layout_cfg in LAYOUT_CONFIGS:
                for orderR in OUTPUT_ORDERS:
                    config_name = f"{layout_cfg['name']}+{orderR}"
                    
                    try:
                        # 调用 C++ 扩展进行测试
                        out = ext.test_layout(
                            N, K, M,
                            layout_cfg["opW"],
                            layout_cfg["opA"],
                            layout_cfg["orderW"],
                            layout_cfg["orderA"],
                            orderR,
                            dtype,
                            warmup,
                            repeat,
                        )
                        
                        m_results[config_name] = {
                            "supported": out["supported"],
                            "alg_count": out.get("alg_count", 0),
                            "config_count": out.get("config_count", 0),
                            "best_tops": out.get("best_tops", 0.0),
                            "best_lat_us": out.get("best_lat_us", 0.0),
                            "best_id": out.get("best_id", -1),
                            "best_ws": out.get("best_ws", 0),
                            "best_split_k": out.get("best_split_k", 1),
                        }
                        
                    except Exception as e:
                        m_results[config_name] = {
                            "supported": False,
                            "error": str(e),
                            "alg_count": 0,
                            "config_count": 0,
                            "best_tops": 0.0,
                            "best_lat_us": 0.0,
                            "best_id": -1,
                            "best_ws": 0,
                            "best_split_k": 1,
                        }
            
            results[nk_key][str(M)] = m_results
            
            # 统计支持的配置数
            supported_count = sum(1 for v in m_results.values() if v.get("supported", False))
            if verbose:
                print(f"{supported_count}/8 layouts ✓")
    
    return {
        "dtype": dtype,
        "nk_list": [(n, k, name) for n, k, name in nk_list],
        "m_list": m_list,
        "results": results,
    }


# === 结果保存 ===

def save_outputs(out_dir: Path, gpu_short_name: str, arch_name: str, dtype: str,
                 benchmark_ret: Dict, warmup: int, repeat: int) -> Path:
    """保存测试结果到 CSV 和 JSON 文件"""
    prop = torch.cuda.get_device_properties(0)
    
    subdir_name = f"{gpu_short_name}_cc{prop.major}{prop.minor}_{dtype.upper()}"
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    csv_path = subdir / "layout_benchmark_results.csv"
    json_path = subdir / "layout_benchmark_LUT.json"
    
    cuda_driver_ver = get_nvidia_smi_cuda_version()
    cuda_runtime_ver = get_cuda_runtime_version()
    
    meta = {
        "gpu_name": prop.name,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "arch_name": arch_name,
        "dtype": dtype,
        "warmup": warmup,
        "repeat": repeat,
        "torch_version": torch.__version__,
        "cuda_version_driver": cuda_driver_ver,
        "cuda_version_runtime": cuda_runtime_ver,
        "time": datetime.datetime.now().isoformat(),
        "m_list": benchmark_ret["m_list"],
        "nk_list": benchmark_ret["nk_list"],
        "layout_configs": [cfg["name"] for cfg in LAYOUT_CONFIGS],
        "output_orders": OUTPUT_ORDERS,
    }

    # === CSV 生成 ===
    lines = []
    lines.append(f"# GPU: {prop.name}")
    lines.append(f"# CC: {prop.major}.{prop.minor}")
    lines.append(f"# dtype: {dtype}, warmup={warmup}, repeat={repeat}")
    lines.append(f"# Layouts: {[cfg['name'] for cfg in LAYOUT_CONFIGS]}")
    # CSV列顺序: tops, lat, id, ws, split_k (best结果)
    lines.append("M,N,K,layout,orderR,supported,alg_count,config_count,best_tops,best_lat_us,best_id,best_ws,best_split_k")

    for nk_key, nk_data in benchmark_ret["results"].items():
        # 解析 nk_key
        N, K = map(int, nk_key.strip("()").split(","))
        
        for m_key, m_data in nk_data.items():
            if m_key == "name":
                continue
            M = int(m_key)
            
            for config_name, result in m_data.items():
                # 解析 config_name: "TN+CC+Col"
                parts = config_name.rsplit("+", 1)
                layout_name = parts[0]
                orderR = parts[1] if len(parts) > 1 else "Col"
                
                supported = 1 if result.get("supported", False) else 0
                alg_count = result.get("alg_count", 0)
                config_count = result.get("config_count", 0)
                
                best_tops = result.get("best_tops", 0.0)
                best_lat = result.get("best_lat_us", 0.0)
                best_id = result.get("best_id", -1)
                best_ws = result.get("best_ws", 0)
                best_split_k = result.get("best_split_k", 1)
                
                lines.append(f"{M},{N},{K},{layout_name},{orderR},{supported},{alg_count},{config_count},{best_tops:.6f},{best_lat:.3f},{best_id},{best_ws},{best_split_k}")
    
    csv_path.write_text("\n".join(lines))

    # === JSON 生成 ===
    json_payload = {
        "meta": meta,
        "results": benchmark_ret["results"],
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))

    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir


# === dtype 兼容性预测试 ===

def probe_dtype_support(ext, dtype: str) -> Tuple[bool, str]:
    """探测 dtype 是否被当前 GPU 支持"""
    N, K, M = 32, 32, 16
    
    try:
        out = ext.test_layout(
            N, K, M,
            "T", "N", "Col", "Col", "Col",  # TN+CC+Col
            dtype,
            1, 1,  # warmup, repeat
        )
        
        if out.get("supported", False):
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 不支持此 layout"
            
    except Exception as e:
        return False, f"dtype={dtype} 测试失败: {str(e)}"
    finally:
        torch.cuda.empty_cache()


def check_dtype_support(ext, dtype: str, arch_name: str, verbose: bool = True) -> None:
    """检查 dtype 是否被支持"""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"不支持的数据类型: {dtype}\n"
            f"支持的类型: {', '.join(SUPPORTED_DTYPES)}"
        )
    
    if verbose:
        print(f"[预测试] 检测 dtype={dtype} 在 {arch_name} 上的支持情况...", end=" ", flush=True)
    
    supported, message = probe_dtype_support(ext, dtype)
    
    if supported:
        if verbose:
            print("✓", flush=True)
    else:
        if verbose:
            print("✗", flush=True)
        raise ValueError(
            f"数据类型 {dtype.upper()} 在当前 GPU ({arch_name}) 上不可用。\n"
            f"原因: {message}\n"
        )


# === 主流程 ===

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt Layout 性能测试")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="数据类型")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--compile", action="store_true", help="强制重新编译 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    print("="*60)
    print("cuSPARSELt Layout 性能测试")
    print("="*60)
    
    arch_name, arch_suffix, sm_code = detect_arch()
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor}, {arch_name})")
    print(f"参数: dtype={args.dtype}, warmup={args.warmup}, repeat={args.repeat}")
    print()

    out_dir = Path(args.out_dir) if args.out_dir else Path("./layout_benchmark_results")
    
    gpu_short_name = get_gpu_short_name()
    print(f"GPU 简称: {gpu_short_name}")
    print()

    ext = load_extension(verbose=True, force_compile=args.compile)

    try:
        check_dtype_support(ext, args.dtype, arch_name, verbose=True)
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        return
    
    print()

    nk_list = default_nk_list()
    m_list = default_m_list()
    
    print(f"[3/4] 开始 Layout 性能测试...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print(f"      Layout 配置: {len(LAYOUT_CONFIGS)} × {len(OUTPUT_ORDERS)} = {len(LAYOUT_CONFIGS) * len(OUTPUT_ORDERS)} 种")
    print()

    ret = run_layout_benchmark(
        ext,
        args.dtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        verbose=True,
    )
    
    saved_dir = save_outputs(
        out_dir,
        gpu_short_name,
        arch_name,
        args.dtype,
        ret,
        args.warmup,
        args.repeat,
    )
    
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
