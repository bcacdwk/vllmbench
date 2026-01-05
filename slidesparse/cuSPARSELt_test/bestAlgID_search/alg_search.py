#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cuSPARSELt 算法离线搜索

架构说明：
=========
- Python 端：负责外层 NK 循环、参数解析、GPU 检测、数据生成、结果落盘
- C++ 端：负责内层 M 循环、算法枚举、cuSPARSELt API 调用、精确计时

1. Python 控制外层 NK 循环，方便做进度条、异常捕获、断点续跑
2. C++ 控制内层 M 循环和算法枚举，避免跨进程通信开销
3. JSON 在 Python 端生成
4. torch.utils.cpp_extension.load 自动检测 GPU 架构，无需手动指定 -arch

显存优化：
- 预分配最大尺寸的张量，复用 buffer 避免反复 malloc/free
- 搜索结束后显式释放并调用 empty_cache()

运行示例:
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype int8 --verify --compile
CUDA_VISIBLE_DEVICES=0 python3 alg_search.py --dtype fp8e4m3 --verify

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
            # 解析 "CUDA Version: 13.0" 这样的格式
            for line in result.stdout.split('\n'):
                if "CUDA Version" in line:
                    # 格式通常是 "... CUDA Version: 13.0 ..."
                    import re
                    match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                    if match:
                        return match.group(1)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # 备选方案：通过 cudaDriverGetVersion API
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
    """获取 CUDA runtime 版本（PyTorch 编译时使用的版本）"""
    try:
        # torch.version.cuda 是 PyTorch 编译时使用的 CUDA toolkit 版本
        return torch.version.cuda or "unknown"
    except Exception:
        return "unknown"


def get_gpu_short_name() -> str:
    """
    获取 GPU 名称（简化版，如 A100, H100, B200）。
    
    常见格式处理:
    - "NVIDIA A100-SXM4-40GB" -> "A100"
    - "NVIDIA H100 PCIe" -> "H100"
    - "NVIDIA B200" -> "B200"
    """
    prop = torch.cuda.get_device_properties(0)
    full_name = prop.name
    short_name = full_name
    
    # 移除 "NVIDIA " 前缀
    nvidia_pos = full_name.find("NVIDIA ")
    if nvidia_pos != -1:
        short_name = full_name[nvidia_pos + 7:]
    
    # 提取第一个空格或连字符之前的部分作为 GPU 型号
    for sep in [" ", "-"]:
        end_pos = short_name.find(sep)
        if end_pos != -1:
            short_name = short_name[:end_pos]
            break
    
    # 如果提取失败，使用清理后的完整名称（替换空格和特殊字符）
    if not short_name:
        short_name = full_name
        for c in [" ", "-", "/"]:
            short_name = short_name.replace(c, "_")
    
    return short_name


# === cuSPARSELt 动态库加载（必须在加载自定义 .so 之前完成）===
_CUSPARSELT_LOADED = False

def ensure_cusparselt_loaded() -> None:
    """优先加载系统或环境变量指定的 cuSPARSELt，避免符号冲突。"""
    global _CUSPARSELT_LOADED
    if _CUSPARSELT_LOADED:
        return

    preferred_paths = []
    env_path = os.environ.get("CUSPARSELT_PATH")
    if env_path:
        preferred_paths.append(env_path)

    arch = platform.machine()
    preferred_paths.extend(
        [
            "/usr/lib/aarch64-linux-gnu/libcusparseLt.so.0",
            "/usr/lib/aarch64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
            "/usr/lib/x86_64-linux-gnu/libcusparseLt.so.0",
            "/usr/lib/x86_64-linux-gnu/libcusparseLt/13/libcusparseLt.so.0",
            "/usr/local/cuda/lib64/libcusparseLt.so.0",
        ]
    )
    found = ctypes.util.find_library("cusparseLt")
    if found:
        preferred_paths.append(found)

    for path in dict.fromkeys(preferred_paths):  # 去重但保持优先级
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

# === 扩展编译/加载（跨架构支持）===


def load_extension(verbose: bool = True, force_compile: bool = False) -> object:
    """
    加载 CUDA 扩展（支持不同 GPU 设备）。
    
    根据当前 GPU 设备加载或编译对应的 .so 文件。
    不同设备的 .so 文件可以在同一目录共存：
    - alg_search_A100_cusparselt.so
    - alg_search_H100_cusparselt.so
    - alg_search_B200_cusparselt.so
    
    Args:
        verbose: 是否显示进度信息
        force_compile: 是否强制重新编译当前设备
    
    Returns:
        编译好的扩展模块
    """
    if verbose:
        print("[1/4] 加载 cuSPARSELt 库...", end=" ", flush=True)
    ensure_cusparselt_loaded()
    if verbose:
        print("✓", flush=True)
    
    # 获取 GPU 设备简称和架构信息
    gpu_short_name = get_gpu_short_name()
    _, _, sm_code = detect_arch()
    prop = torch.cuda.get_device_properties(0)
    ext_name = f"alg_search_cusparselt_{gpu_short_name}_cc{prop.major}{prop.minor}"
    so_pattern = f"{ext_name}*.so"
    
    src_path = Path(__file__).parent / "alg_search_cusparselt.cu"
    build_dir = Path(__file__).parent / "build_so_files"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查当前设备的 .so 是否存在且比源文件新
    existing_so = list(build_dir.glob(so_pattern))
    need_compile = force_compile
    if not need_compile:
        if not existing_so:
            need_compile = True
        else:
            # 源文件比 .so 新则需要重编译
            need_compile = src_path.stat().st_mtime > existing_so[0].stat().st_mtime
    
    if not need_compile and existing_so:
        # 直接加载已有的 .so
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
        # 编译（torch.utils.cpp_extension.load 会自动覆盖旧文件）
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
        
        # 清理编译中间文件，只保留 .so
        for pattern in [".ninja_deps", ".ninja_log", "build.ninja", "*.o"]:
            for f in build_dir.glob(pattern):
                f.unlink(missing_ok=True)
        
        if verbose:
            print("✓", flush=True)
        return ext

# === 架构检测 ===

# 架构名称映射：compute capability major -> (arch_name, arch_suffix)
ARCH_INFO = {
    8: ("Ampere", "ampere"),       # A100, A10, A30 等
    9: ("Hopper", "hopper"),       # H100, H200 等
    10: ("Blackwell", "blackwell"), # B100, B200 等
    12: ("Blackwell", "blackwell"), # GB10 等 (CC 12.x 也是 Blackwell 家族)
}

def detect_arch() -> Tuple[str, str, str]:
    """
    检测 GPU 架构。
    
    返回:
        (arch_name, arch_suffix, sm_code) 其中:
        - arch_name: "Ampere", "Hopper", "Blackwell" 等（用于显示）
        - arch_suffix: "ampere", "hopper", "blackwell" 等（用于文件命名）
        - sm_code: "sm_80", "sm_90" 等（用于 nvcc 编译）
    """
    prop = torch.cuda.get_device_properties(0)
    major = prop.major
    sm_code = f"sm_{major}{prop.minor}"
    
    if major in ARCH_INFO:
        name, suffix = ARCH_INFO[major]
        return name, suffix, sm_code
    
    # 未知架构
    return f"SM{major}{prop.minor}", f"sm{major}{prop.minor}", sm_code

# === 默认 NK/M 列表 ===

def default_nk_list() -> List[Tuple[int, int]]:
    return [
        (3840, 2560),  # Wqkv
        (2560, 2560),  # Wo
        (13824, 2560), # W13
        (2560, 6912),  # W2
    ]


def default_m_list() -> List[int]:
    # pow2 序列从 16 到 16384，覆盖 decode 和 prefill 的各种 batch size
    return [16, 64, 128, 512, 2048, 8192, 16384]


# === JSON 查询辅助函数 ===

def lookup_best_alg(json_data: Dict, N: int, K: int, M: int) -> Optional[int]:
    """
    从 JSON 数据中查询最佳算法 ID。
    
    查询逻辑：
    1. 用 (N, K) 在 nk_entries 中找到对应条目
    2. 在 m_thresholds 中找到 <= query_M 的最大值
    3. 返回该 M 对应的 alg_by_m[m][0]（最佳算法）
    
    Args:
        json_data: 加载的 JSON 数据
        N: 稀疏矩阵 W 的行数
        K: 共享维度
        M: 稠密矩阵 A 的行数（查询的 batch size）
    
    Returns:
        最佳算法 ID，如果找不到返回 None
    """
    nk_key = f"({N},{K})"
    nk_entries = json_data.get("nk_entries", {})
    
    if nk_key not in nk_entries:
        return None
    
    entry = nk_entries[nk_key]
    m_thresholds = entry.get("m_thresholds", [])
    alg_by_m = entry.get("alg_by_m", {})
    
    if not m_thresholds:
        return None
    
    # 找到 <= M 的最大阈值
    selected_m = None
    for threshold in m_thresholds:
        if threshold <= M:
            selected_m = threshold
        else:
            break
    
    if selected_m is None:
        # M 比所有阈值都小，使用最小的阈值
        selected_m = m_thresholds[0]
    
    m_key = str(selected_m)
    if m_key in alg_by_m:
        # 简化格式: alg_by_m[m_key] = [best_id, 2nd_id, 3rd_id]
        alg_list = alg_by_m[m_key]
        if isinstance(alg_list, list) and len(alg_list) > 0:
            return alg_list[0]
    
    return None


# === 运行一次 layout 的搜索 ===

def run_search(ext, dtype: str, nk_list: List[Tuple[int, int]], m_list: List[int],
               warmup: int, repeat: int, verify: bool, verbose: bool = True) -> Dict:
    """
    运行算法搜索。
    
    固定布局: T/N + C/C + C (权重W在左，稀疏矩阵，Column Major 输出)
    每个 NK 组合生成新的随机数据
    """
    layout = "TNCCcol"  # 固定布局: TN+CC+Col
    results = []
    max_M = max(m_list)
    blacklist = []  # 不再需要屏蔽任何算法
    total_nk = len(nk_list)

    for nk_id, (N, K) in enumerate(nk_list):
        if verbose:
            print(f"    NK {nk_id+1}/{total_nk}: ({N}, {K})", flush=True)
        
        # 生成随机数据（每个 NK 新分配，简洁明了）
        max_M = max(m_list)
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(max_M, K, device="cuda", dtype=torch.bfloat16)

        # 先做 2:4 剪枝
        W_pruned = ext.prune_24(W, layout)

        # 调用搜索
        out = ext.search_topk(
            W_pruned,
            A,
            m_list,
            layout,
            dtype,
            warmup,
            repeat,
            verify,
            blacklist,
            3,
        )
        
        # 显示压缩算法ID和每个 M 的有效算法数
        compress_alg_id = out["compress_alg_id"]
        num_valid_per_m = out["num_valid_algs_per_M"].cpu().tolist()
        
        if verbose:
            print(f"      → 最大有效算法ID: {compress_alg_id}，正在通过 id={compress_alg_id} 进行压缩")
            # 显示每个 M 的有效算法数（取第一个作为代表，应该都一样）
            first_valid = num_valid_per_m[0] if num_valid_per_m else 0
            print(f"      → 每个 M 的有效算法数: {first_valid} ✓")
        
        results.append({
            "nk_id": nk_id,
            "N": N,
            "K": K,
            "raw": out,
        })
        
        # 释放当前 NK 的张量
        del W, A, W_pruned
    
    torch.cuda.empty_cache()
    
    return {
        "dtype": dtype,
        "results": results,
        "M_list": m_list,
        "NK_list": nk_list,
    }

# === 落盘工具 ===

def save_outputs(out_dir: Path, gpu_short_name: str, arch_name: str, dtype: str,
                 search_ret: Dict, warmup: int, repeat: int, verify: bool) -> Path:
    """
    保存搜索结果到 CSV 和 JSON 文件。
    
    文件夹命名: {GPU}_{cc}_{dtype}，如 A100_cc80_INT8
    文件命名: alg_id_benchmark_results.csv, alg_id_LUT.json
    
    CSV 排序规则：先按 M 升序，M 相同时按 nk_list 传入顺序排序。
    
    JSON 格式设计用于两步查询：
    1. 先按 (N, K) 查找对应的 nk_entry
    2. 在 nk_entry 的 m_thresholds 中找到 <= 目标 M 的最大阈值，使用其 best_alg_id
    
    Returns:
        保存结果的子目录路径
    """
    layout = "TNCCcol"  # 固定布局，仅用于元数据记录
    prop = torch.cuda.get_device_properties(0)
    
    # 子目录命名: {GPU}_{cc}_{dtype}
    subdir_name = f"{gpu_short_name}_cc{prop.major}{prop.minor}_{dtype.upper()}"
    subdir = out_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)
    
    # 固定文件名
    csv_path = subdir / "alg_id_benchmark_results.csv"
    json_path = subdir / "alg_id_LUT.json"
    
    # 获取 CUDA 版本信息
    cuda_driver_ver = get_nvidia_smi_cuda_version()  # nvidia-smi 显示的 CUDA 版本
    cuda_runtime_ver = get_cuda_runtime_version()    # PyTorch 编译时的 CUDA 版本
    
    # 获取最大有效算法 ID（所有 MNK 组合都相同）
    max_alg_id = search_ret["results"][0]["raw"]["compress_alg_id"] if search_ret["results"] else -1
    
    meta = {
        "gpu_name": prop.name,
        "compute_capability": f"{prop.major}.{prop.minor}",
        "arch_name": arch_name,
        "layout": layout,
        "dtype": dtype,
        "max_alg_id": max_alg_id,
        "warmup": warmup,
        "repeat": repeat,
        "verify": verify,
        "torch_version": torch.__version__,
        "cuda_version_driver": cuda_driver_ver,
        "cuda_version_runtime": cuda_runtime_ver,
        "time": datetime.datetime.now().isoformat(),
        "M_list": search_ret["M_list"],
        "NK_list": search_ret["NK_list"],
    }

    # === CSV 生成（按 M 升序，M 相同时按 nk_list 顺序）===
    lines = []
    header_info = [
        f"# GPU: {prop.name}",
        f"# CC: {prop.major}.{prop.minor}",
        f"# max_alg_id: {max_alg_id}",
        f"# torch: {torch.__version__}",
        f"# CUDA driver: {cuda_driver_ver}, runtime: {cuda_runtime_ver}",
        f"# layout: {layout}, dtype: {dtype}, warmup={warmup}, repeat={repeat}, verify={verify}",
        f"# M_list: {search_ret['M_list']}",
        f"# NK_list: {search_ret['NK_list']}",
    ]
    lines.extend(header_info)
    lines.append("M,N,K,best_id1,lat_us1,tops1,best_id2,lat_us2,tops2,best_id3,lat_us3,tops3")

    # 收集所有数据行，用于排序
    csv_rows = []  # [(M, nk_idx, csv_line_str), ...]
    
    for nk_idx, res in enumerate(search_ret["results"]):
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        topk_lat = raw["topk_lat_us"].cpu()
        topk_tops = raw["topk_tops"].cpu()
        valid = raw["valid_mask"].cpu()

        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            lats = topk_lat[m_i]
            tops = topk_tops[m_i]
            vmask = valid[m_i]

            csv_values = [str(M), str(res["N"]), str(res["K"])]
            for k in range(3):
                if vmask[k]:
                    csv_values.extend([
                        str(int(algs[k].item())),
                        f"{float(lats[k].item()):.3f}",
                        f"{float(tops[k].item()):.6f}",
                    ])
                else:
                    csv_values.extend(["", "", ""])
            csv_rows.append((M, nk_idx, ",".join(csv_values)))

    # 排序：先按 M 升序，M 相同时按 nk_idx（即 nk_list 顺序）
    csv_rows.sort(key=lambda x: (x[0], x[1]))
    for _, _, line in csv_rows:
        lines.append(line)

    csv_path.write_text("\n".join(lines))

    # === JSON 生成（简化版：只保留 top3 算法 ID）===
    # 格式设计：
    # {
    #   "meta": {...},
    #   "nk_entries": {
    #     "(N,K)": {
    #       "m_thresholds": [m1, m2, ...],  # 升序排列的 M 值
    #       "alg_by_m": {
    #         "m1": [best_id, 2nd_id, 3rd_id],
    #         "m2": [...],
    #         ...
    #       }
    #     }
    #   }
    # }
    # 
    # 查询逻辑：
    # 1. 用 (N, K) 找到 nk_entry
    # 2. 在 m_thresholds 中找到 <= query_M 的最大值 m_key
    # 3. 返回 alg_by_m[m_key][0] 作为最佳算法
    
    nk_entries = {}
    
    for nk_idx, res in enumerate(search_ret["results"]):
        N, K = res["N"], res["K"]
        nk_key = f"({N},{K})"
        
        raw = res["raw"]
        topk_id = raw["topk_alg_id"].cpu()
        valid = raw["valid_mask"].cpu()

        m_thresholds = []
        alg_by_m = {}
        
        for m_i, M in enumerate(search_ret["M_list"]):
            algs = topk_id[m_i]
            vmask = valid[m_i]

            # 只有当有有效结果时才记录
            if vmask[0]:
                m_thresholds.append(M)
                # 简化格式：只记录 top3 的 alg_id
                top3_ids = [int(algs[k].item()) for k in range(3) if vmask[k]]
                alg_by_m[str(M)] = top3_ids
        
        nk_entries[nk_key] = {
            "m_thresholds": m_thresholds,
            "alg_by_m": alg_by_m,
        }

    json_payload = {
        "meta": meta,
        "nk_entries": nk_entries,
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))

    print(f"已生成: {csv_path}")
    print(f"已生成: {json_path}")
    
    return subdir

# === dtype 兼容性预测试 ===

# 支持的 dtype 列表
SUPPORTED_DTYPES = ["int8", "fp8e4m3"]


def probe_dtype_support(ext, dtype: str, layout: str = "TNCCcol") -> Tuple[bool, str]:
    """
    通过实际调用 cuSPARSELt 来探测 dtype 是否被当前 GPU 支持。
    
    使用最小尺寸（32x32）的矩阵进行快速测试，避免硬编码的架构判断。
    
    Args:
        ext: 编译好的 CUDA 扩展模块
        dtype: 要测试的数据类型
        layout: 布局类型
    
    Returns:
        (supported, message) 其中:
        - supported: 是否支持
        - message: 成功或失败的详细信息
    """
    # 最小测试尺寸（满足 cuSPARSELt 对齐要求：N 需 32 对齐，K/M 需 16 对齐）
    N, K, M = 32, 32, 16
    
    try:
        # 创建小型测试矩阵
        W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        
        # 执行 2:4 剪枝
        W_pruned = ext.prune_24(W, layout)
        
        # 尝试搜索（只做 1 次 warmup，1 次 repeat，不验证）
        out = ext.search_topk(
            W_pruned, A,
            [M],           # M_list
            layout,
            dtype,
            1,             # warmup
            1,             # repeat
            False,         # verify
            [],            # blacklist
            1,             # topk
        )
        
        # 检查是否有有效结果
        valid_mask = out["valid_mask"].cpu()
        if valid_mask.sum().item() > 0:
            return True, f"dtype={dtype} 测试通过 ✓"
        else:
            return False, f"dtype={dtype} 无有效算法（可能不支持）"
            
    except Exception as e:
        error_msg = str(e)
        # 提取关键错误信息
        if "CUSPARSE_STATUS" in error_msg:
            return False, f"cuSPARSELt 不支持 dtype={dtype}: {error_msg}"
        elif "不支持的数据类型" in error_msg:
            return False, f"dtype={dtype} 不被支持"
        else:
            return False, f"dtype={dtype} 测试失败: {error_msg}"
    finally:
        # 清理
        torch.cuda.empty_cache()


def check_dtype_support(ext, dtype: str, arch_name: str, verbose: bool = True) -> None:
    """
    检查 dtype 是否被当前 GPU 支持（通过实际测试）。
    
    Args:
        ext: 编译好的 CUDA 扩展模块
        dtype: 要测试的数据类型
        arch_name: 架构名称（用于显示）
        verbose: 是否显示详细信息
    
    Raises:
        ValueError: 如果 dtype 不被支持
    """
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
            print(f"✓", flush=True)
    else:
        if verbose:
            print(f"✗", flush=True)
        raise ValueError(
            f"数据类型 {dtype.upper()} 在当前 GPU ({arch_name}) 上不可用。\n"
            f"原因: {message}\n"
        )


# === 主流程 ===

def parse_args():
    p = argparse.ArgumentParser(description="cuSPARSELt 算法离线搜索 v1.0")
    p.add_argument("--dtype", default="int8", choices=SUPPORTED_DTYPES, help="数据类型")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=100)
    p.add_argument("--verify", action="store_true", help="开启正确性校验")
    p.add_argument("--compile", action="store_true", help="强制重新编译当前架构的 CUDA 扩展")
    p.add_argument("--out_dir", default=None, help="输出目录，默认 ./alg_search_results/<timestamp>")
    return p.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")

    # === 显示配置信息 ===
    print("="*60)
    print("cuSPARSELt 算法离线搜索 v1.0")
    print("="*60)
    
    arch_name, arch_suffix, sm_code = detect_arch()
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name} (CC {prop.major}.{prop.minor}, {arch_name})")
    print(f"参数: dtype={args.dtype}, warmup={args.warmup}, repeat={args.repeat}")
    if args.verify:
        print("注意: 已开启 verify 模式，会降低搜索速度")
    print()

    # 输出根目录（默认为 ./alg_search_results）
    out_dir = Path(args.out_dir) if args.out_dir else Path("./alg_search_results")
    
    # 获取简化的 GPU 名称
    gpu_short_name = get_gpu_short_name()
    print(f"GPU 简称: {gpu_short_name}")
    print()

    ext = load_extension(verbose=True, force_compile=args.compile)

    # === 预测试 dtype 兼容性（通过实际调用 cuSPARSELt）===
    try:
        check_dtype_support(ext, args.dtype, arch_name, verbose=True)
    except ValueError as e:
        print(f"\n❌ 错误: {e}")
        print("")
        return
    
    print()

    nk_list = default_nk_list()
    m_list = default_m_list()
    
    print(f"[3/4] 开始算法搜索...")
    print(f"      NK 组合: {len(nk_list)} 个, M 列表: {m_list}")
    print()

    ret = run_search(
        ext,
        args.dtype,
        nk_list,
        m_list,
        args.warmup,
        args.repeat,
        args.verify,
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
        args.verify,
    )
    
    print(f"[4/4] 完成! 结果已保存到:")
    print(f"      - {saved_dir}")
    print("="*60)
if __name__ == "__main__":
    main()
