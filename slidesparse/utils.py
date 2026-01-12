#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 文件名统一工具库

提供统一的硬件信息获取、文件命名和模块加载功能。

命名规范
========
所有生成的文件名遵循统一格式：
    {prefix}_{GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}.{ext}

dtype 部分是可选的，支持三种情况：
1. 单个 dtype:   cublaslt_gemm_H100_cc90_FP8E4M3_py312_cu124_x86_64.so
2. 多个 dtype:   cublaslt_gemm_H100_cc90_FP8_INT8_py312_cu124_x86_64.so
3. 无 dtype:     cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so

示例：
    cublaslt_gemm_B200_cc100_py312_cu129_x86_64.so       # 支持多种类型的 GEMM
    dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py  # 特定类型
    alg_id_LUT_A100_cc80_INT8_py311_cu121_x86_64.json   # 特定类型

组件说明：
    - prefix:    用途前缀（cublaslt_gemm, cusparselt_gemm, dequant_bias_tuned 等）
    - GPU:       GPU 简称（H100, A100, B200, GB10 等）
    - CC:        Compute Capability（cc90, cc100, cc121 等）
    - dtype:     数据类型（可选，单个或多个：FP8E4M3, INT8, BF16, FP32 等）
    - PyVer:     Python 版本（py312, py311 等）
    - CUDAVer:   CUDA 版本（cu129, cu124 等）
    - Arch:      系统架构（x86_64, aarch64 等）

主要功能
========
1. HardwareInfo: 硬件信息单例类，缓存所有硬件信息
2. FileNameBuilder: 文件名构建器
3. FileFinder: 文件查找器，支持模糊匹配
4. ModuleLoader: 模块加载器，支持 .py 和 .so

使用示例
========
>>> from slidesparse.utils import hw_info, build_filename, find_file, load_module
>>>
>>> # 获取硬件信息
>>> print(hw_info.gpu_name)  # "H100"
>>> print(hw_info.cc_tag)    # "cc90"
>>>
>>> # 构建文件名（无 dtype，用于支持多类型的扩展）
>>> name = build_filename("cublaslt_gemm", ext=".so")
>>> # -> "cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so"
>>>
>>> # 构建文件名（带单个 dtype）
>>> name = build_filename("dequant_bias_tuned", dtype="BF16", ext=".py")
>>> # -> "dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py"
>>>
>>> # 构建文件名（带多个 dtype）
>>> name = build_filename("gemm_kernel", dtype=["FP8", "INT8"], ext=".so")
>>> # -> "gemm_kernel_H100_cc90_FP8_INT8_py312_cu124_x86_64.so"
>>>
>>> # 查找文件
>>> path = find_file("cublaslt_gemm", search_dir=build_dir)
>>>
>>> # 加载模块
>>> module = load_module("cublaslt_gemm", search_dir=build_dir)
"""

import importlib
import importlib.util
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import cached_property

# 延迟导入 torch
_torch = None


def _get_torch():
    """延迟导入 torch，避免在不需要时加载"""
    global _torch
    if _torch is None:
        try:
            import torch
            _torch = torch
        except ImportError:
            raise ImportError("PyTorch is required but not installed")
    return _torch


# =============================================================================
# 数据类型标准化
# =============================================================================

# 数据类型别名映射（输入 -> 标准名称）
DTYPE_ALIASES = {
    # FP8 变体
    "fp8": "FP8E4M3",
    "fp8e4m3": "FP8E4M3",
    "fp8_e4m3": "FP8E4M3",
    "FP8": "FP8E4M3",
    "FP8E4M3": "FP8E4M3",
    "e4m3": "FP8E4M3",
    "fp8e5m2": "FP8E5M2",
    "fp8_e5m2": "FP8E5M2",
    "FP8E5M2": "FP8E5M2",
    "e5m2": "FP8E5M2",
    # INT8
    "int8": "INT8",
    "INT8": "INT8",
    "i8": "INT8",
    # BF16
    "bf16": "BF16",
    "BF16": "BF16",
    "bfloat16": "BF16",
    # FP16
    "fp16": "FP16",
    "FP16": "FP16",
    "float16": "FP16",
    "half": "FP16",
    # FP32
    "fp32": "FP32",
    "FP32": "FP32",
    "float32": "FP32",
    "float": "FP32",
}


def normalize_dtype(dtype: str) -> str:
    """
    标准化数据类型名称
    
    Args:
        dtype: 输入的数据类型名称（大小写不敏感）
        
    Returns:
        标准化的数据类型名称
        
    Raises:
        ValueError: 未知的数据类型
        
    Examples:
        >>> normalize_dtype("fp8")
        'FP8E4M3'
        >>> normalize_dtype("int8")
        'INT8'
    """
    key = dtype.lower().replace("-", "_").replace(" ", "")
    if key in DTYPE_ALIASES:
        return DTYPE_ALIASES[key]
    # 尝试原始输入
    if dtype in DTYPE_ALIASES.values():
        return dtype
    raise ValueError(f"未知的数据类型: {dtype}. 支持的类型: {set(DTYPE_ALIASES.values())}")


# =============================================================================
# 硬件信息类
# =============================================================================

@dataclass
class HardwareInfo:
    """
    硬件信息单例类
    
    缓存所有硬件相关信息，避免重复查询。
    所有属性使用 cached_property 实现懒加载。
    
    Attributes:
        gpu_name: GPU 简称（H100, A100 等）
        gpu_full_name: GPU 完整名称
        cc_major: Compute Capability 主版本
        cc_minor: Compute Capability 次版本
        cc_tag: CC 标签（cc90, cc100 等）
        python_tag: Python 版本标签（py312 等）
        cuda_tag: CUDA 版本标签（cu129 等）
        arch_tag: 系统架构标签（x86_64 等）
    """
    
    _instance: Optional['HardwareInfo'] = field(default=None, repr=False, init=False)
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    # -------------------------------------------------------------------------
    # GPU 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def gpu_full_name(self) -> str:
        """GPU 完整名称"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.name
        # 备选：nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return "unknown"
    
    @cached_property
    def gpu_name(self) -> str:
        """
        GPU 简称（H100, A100, B200, RTX5080 等）
        
        处理常见格式:
        - "NVIDIA A100-SXM4-40GB" -> "A100"
        - "NVIDIA H100 PCIe" -> "H100"
        - "NVIDIA GeForce RTX 5080" -> "RTX5080"
        - "NVIDIA GeForce RTX 4090" -> "RTX4090"
        - "NVIDIA GeForce GTX 1080 Ti" -> "GTX1080Ti"
        - "NVIDIA TITAN RTX" -> "TitanRTX"
        """
        full_name = self.gpu_full_name
        if full_name == "unknown":
            return "unknown"
        
        # 移除 "NVIDIA " 前缀
        name = full_name
        if name.startswith("NVIDIA "):
            name = name[7:]
        
        # 处理 GeForce RTX/GTX 系列: "GeForce RTX 5080" -> "RTX5080"
        if name.startswith("GeForce "):
            name = name[8:]  # 移除 "GeForce "
            # 现在 name 可能是 "RTX 5080" 或 "GTX 1080 Ti"
            # 提取 RTX/GTX 前缀和型号
            parts = name.split()
            if len(parts) >= 2 and parts[0] in ("RTX", "GTX"):
                prefix = parts[0]  # RTX 或 GTX
                model = "".join(parts[1:])  # 5080 或 1080Ti
                return f"{prefix}{model}"
            # 其他 GeForce 情况
            return "".join(parts)
        
        # 处理 TITAN 系列: "TITAN RTX" -> "TitanRTX"
        if name.startswith("TITAN "):
            return "Titan" + name[6:].replace(" ", "")
        
        # 数据中心卡: "A100-SXM4-40GB" -> "A100", "H100 PCIe" -> "H100"
        # 提取第一个空格或连字符之前的部分
        for sep in [" ", "-"]:
            end_pos = name.find(sep)
            if end_pos != -1:
                name = name[:end_pos]
                break
        
        # 清理特殊字符
        if not name:
            name = full_name
            for c in [" ", "-", "/"]:
                name = name.replace(c, "_")
        
        return name
    
    @cached_property
    def cc_major(self) -> int:
        """Compute Capability 主版本"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.major
        return 0
    
    @cached_property
    def cc_minor(self) -> int:
        """Compute Capability 次版本"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.minor
        return 0
    
    @cached_property
    def cc_tag(self) -> str:
        """CC 标签（cc90, cc100, cc121 等）"""
        return f"cc{self.cc_major}{self.cc_minor}"
    
    @cached_property
    def sm_code(self) -> str:
        """SM 代码（sm_90, sm_100 等）"""
        return f"sm_{self.cc_major}{self.cc_minor}"
    
    @cached_property
    def gpu_memory_gb(self) -> float:
        """GPU 显存大小 (GB)"""
        torch = _get_torch()
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            return prop.total_memory / (1024 ** 3)
        return 0.0
    
    # -------------------------------------------------------------------------
    # Python 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def python_version(self) -> Tuple[int, int]:
        """Python 版本 (major, minor)"""
        return (sys.version_info.major, sys.version_info.minor)
    
    @cached_property
    def python_tag(self) -> str:
        """Python 版本标签（py312, py311 等）"""
        return f"py{self.python_version[0]}{self.python_version[1]}"
    
    # -------------------------------------------------------------------------
    # CUDA 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def cuda_runtime_version(self) -> str:
        """CUDA Runtime 版本（PyTorch 编译时使用的版本）"""
        torch = _get_torch()
        try:
            return torch.version.cuda or "unknown"
        except Exception:
            return "unknown"
    
    @cached_property
    def cuda_driver_version(self) -> str:
        """CUDA Driver 版本（nvidia-smi 显示的版本）"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if "CUDA Version" in line:
                        match = re.search(r'CUDA Version:\s*(\d+\.\d+)', line)
                        if match:
                            return match.group(1)
        except Exception:
            pass
        return "unknown"
    
    @cached_property
    def cuda_tag(self) -> str:
        """
        CUDA 版本标签（cu129, cu124 等）
        
        优先使用 Runtime 版本，因为这是实际编译时使用的版本。
        """
        version = self.cuda_runtime_version
        if version == "unknown":
            version = self.cuda_driver_version
        if version == "unknown":
            return "cu000"
        # "12.9" -> "cu129", "12.4" -> "cu124"
        parts = version.split(".")
        if len(parts) >= 2:
            major = parts[0]
            minor = parts[1].split(".")[0]  # 处理 "12.4.1" 这种情况
            return f"cu{major}{minor}"
        return f"cu{version.replace('.', '')}"
    
    # -------------------------------------------------------------------------
    # 系统架构
    # -------------------------------------------------------------------------
    
    @cached_property
    def arch_raw(self) -> str:
        """原始系统架构"""
        return platform.machine()
    
    @cached_property
    def arch_tag(self) -> str:
        """系统架构标签（x86_64, aarch64 等）"""
        machine = self.arch_raw
        if machine in ("x86_64", "AMD64"):
            return "x86_64"
        elif machine in ("aarch64", "arm64"):
            return "aarch64"
        return machine.lower()
    
    # -------------------------------------------------------------------------
    # 驱动信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def driver_version(self) -> str:
        """NVIDIA 驱动版本"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        except Exception:
            pass
        return "unknown"
    
    # -------------------------------------------------------------------------
    # PyTorch 信息
    # -------------------------------------------------------------------------
    
    @cached_property
    def pytorch_version(self) -> str:
        """PyTorch 版本"""
        torch = _get_torch()
        return torch.__version__
    
    # -------------------------------------------------------------------------
    # 架构检测
    # -------------------------------------------------------------------------
    
    # 架构名称映射
    ARCH_INFO = {
        7: ("Volta", "volta"),         # V100 等
        8: ("Ampere", "ampere"),       # A100, A10, A30 等
        9: ("Hopper", "hopper"),       # H100, H200 等
        10: ("Blackwell", "blackwell"), # B100, B200 等
        12: ("Blackwell", "blackwell"), # GB10 等 (CC 12.x 也是 Blackwell 家族)
    }
    
    @cached_property
    def arch_name(self) -> str:
        """架构名称（Ampere, Hopper, Blackwell 等）"""
        if self.cc_major in self.ARCH_INFO:
            return self.ARCH_INFO[self.cc_major][0]
        return f"SM{self.cc_major}{self.cc_minor}"
    
    @cached_property
    def arch_suffix(self) -> str:
        """架构后缀（ampere, hopper, blackwell 等）"""
        if self.cc_major in self.ARCH_INFO:
            return self.ARCH_INFO[self.cc_major][1]
        return f"sm{self.cc_major}{self.cc_minor}"
    
    # -------------------------------------------------------------------------
    # 功能检测
    # -------------------------------------------------------------------------
    
    @cached_property
    def supports_fp8(self) -> bool:
        """是否支持原生 FP8（CC >= 8.9，Ada/Hopper+）"""
        return (self.cc_major, self.cc_minor) >= (8, 9)
    
    @cached_property
    def supports_int8(self) -> bool:
        """是否支持原生 INT8（CC >= 8.0，Ampere+）"""
        return self.cc_major >= 8
    
    @cached_property
    def triton_supported(self) -> Tuple[bool, str]:
        """
        检查 Triton 是否支持当前架构
        
        Returns:
            (supported, reason)
        """
        # 已知不被支持的架构
        UNSUPPORTED = {
            (12, 1): "GB10 (sm_121a) is not yet supported by Triton/ptxas",
        }
        
        if (self.cc_major, self.cc_minor) in UNSUPPORTED:
            return False, UNSUPPORTED[(self.cc_major, self.cc_minor)]
        
        return True, "Architecture is supported"
    
    @cached_property
    def needs_eager_mode(self) -> bool:
        """是否需要使用 eager mode（禁用 torch.compile）"""
        return not self.triton_supported[0]
    
    # -------------------------------------------------------------------------
    # 汇总信息
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "gpu": {
                "name": self.gpu_name,
                "full_name": self.gpu_full_name,
                "memory_gb": round(self.gpu_memory_gb, 1),
            },
            "compute_capability": {
                "major": self.cc_major,
                "minor": self.cc_minor,
                "tag": self.cc_tag,
                "sm_code": self.sm_code,
            },
            "cuda": {
                "runtime_version": self.cuda_runtime_version,
                "driver_version": self.cuda_driver_version,
                "tag": self.cuda_tag,
            },
            "python": {
                "version": f"{self.python_version[0]}.{self.python_version[1]}",
                "tag": self.python_tag,
            },
            "system": {
                "arch": self.arch_tag,
                "driver_version": self.driver_version,
            },
            "architecture": {
                "name": self.arch_name,
                "suffix": self.arch_suffix,
            },
            "capabilities": {
                "supports_fp8": self.supports_fp8,
                "supports_int8": self.supports_int8,
                "triton_supported": self.triton_supported[0],
            },
            "pytorch_version": self.pytorch_version,
        }
    
    def print_info(self):
        """打印硬件信息"""
        print("=" * 60)
        print("SlideSparse Hardware Info")
        print("=" * 60)
        print(f"GPU:           {self.gpu_full_name}")
        print(f"GPU (short):   {self.gpu_name}")
        print(f"Memory:        {self.gpu_memory_gb:.1f} GB")
        print(f"CC:            {self.cc_tag} ({self.sm_code})")
        print(f"Architecture:  {self.arch_name}")
        print(f"Python:        {self.python_tag}")
        print(f"CUDA Runtime:  {self.cuda_runtime_version}")
        print(f"CUDA Driver:   {self.cuda_driver_version}")
        print(f"CUDA Tag:      {self.cuda_tag}")
        print(f"System Arch:   {self.arch_tag}")
        print(f"Driver:        {self.driver_version}")
        print(f"PyTorch:       {self.pytorch_version}")
        print("-" * 60)
        print(f"FP8 Support:   {self.supports_fp8}")
        print(f"INT8 Support:  {self.supports_int8}")
        print(f"Triton:        {'✓' if self.triton_supported[0] else '✗ ' + self.triton_supported[1]}")
        print("=" * 60)


# 全局单例
hw_info = HardwareInfo()


# =============================================================================
# 文件名构建
# =============================================================================

def build_filename(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    ext: str = "",
    *,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
    python_tag: Optional[str] = None,
    cuda_tag: Optional[str] = None,
    arch_tag: Optional[str] = None,
) -> str:
    """
    构建标准化文件名
    
    格式: {prefix}_{GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}.{ext}
    
    dtype 部分是可选的，支持三种情况：
    - None: 不包含 dtype，用于支持多种类型的扩展
    - str: 单个 dtype
    - List[str]: 多个 dtype，按顺序连接
    
    Args:
        prefix: 用途前缀（cublaslt_gemm, cusparselt_gemm, dequant_bias_tuned 等）
        dtype: 数据类型（单个字符串、字符串列表、或 None）
        ext: 文件扩展名（.so, .py, .json 等），不包含点时自动添加
        gpu_name: GPU 名称，默认自动检测
        cc_tag: CC 标签，默认自动检测
        python_tag: Python 版本标签，默认自动检测
        cuda_tag: CUDA 版本标签，默认自动检测
        arch_tag: 系统架构标签，默认自动检测
        
    Returns:
        标准化的文件名
        
    Examples:
        # 无 dtype（支持多种类型的 GEMM 扩展）
        >>> build_filename("cublaslt_gemm", ext=".so")
        'cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so'
        
        # 单个 dtype
        >>> build_filename("dequant_bias_tuned", dtype="BF16", ext=".py")
        'dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py'
        
        # 多个 dtype
        >>> build_filename("gemm_kernel", dtype=["FP8", "INT8"], ext=".so")
        'gemm_kernel_H100_cc90_FP8_INT8_py312_cu124_x86_64.so'
    """
    # 使用提供的值或从硬件信息获取
    _gpu = gpu_name or hw_info.gpu_name
    _cc = cc_tag or hw_info.cc_tag
    _py = python_tag or hw_info.python_tag
    _cuda = cuda_tag or hw_info.cuda_tag
    _arch = arch_tag or hw_info.arch_tag
    
    # 构建组件列表
    components = [prefix, _gpu, _cc]
    
    # 添加数据类型（如果提供）
    if dtype:
        if isinstance(dtype, str):
            # 单个 dtype
            components.append(normalize_dtype(dtype))
        elif isinstance(dtype, (list, tuple)):
            # 多个 dtype，逐个标准化后添加
            for d in dtype:
                components.append(normalize_dtype(d))
    
    # 添加其余组件
    components.extend([_py, _cuda, _arch])
    
    # 连接组件
    name = "_".join(components)
    
    # 处理扩展名
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        name += ext
    
    return name


def build_stem(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    **kwargs
) -> str:
    """
    构建文件名主干（不含扩展名）
    
    等同于 build_filename(..., ext="")
    """
    return build_filename(prefix, dtype=dtype, ext="", **kwargs)


def build_dir_name(
    prefix: Optional[str] = None,
    dtype: Optional[str] = None,
    *,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
) -> str:
    """
    构建目录名（用于按 GPU+CC+dtype 分类的场景）
    
    格式: {GPU}_{CC}_{dtype} 或带 prefix 时 {prefix}_{GPU}_{CC}_{dtype}
    
    Args:
        prefix: 可选前缀
        dtype: 数据类型（必需）
        gpu_name: GPU 名称，默认自动检测
        cc_tag: CC 标签，默认自动检测
        
    Examples:
        >>> build_dir_name(dtype="FP8E4M3")
        'H100_cc90_FP8E4M3'
        
        >>> build_dir_name(prefix="results", dtype="INT8")
        'results_H100_cc90_INT8'
    """
    _gpu = gpu_name or hw_info.gpu_name
    _cc = cc_tag or hw_info.cc_tag
    
    components = []
    if prefix:
        components.append(prefix)
    components.extend([_gpu, _cc])
    
    if dtype:
        components.append(normalize_dtype(dtype))
    
    return "_".join(components)


# =============================================================================
# 文件查找
# =============================================================================

def find_file(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    search_dir: Union[str, Path] = ".",
    ext: Optional[str] = None,
    *,
    exact: bool = True,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
    python_tag: Optional[str] = None,
    cuda_tag: Optional[str] = None,
    arch_tag: Optional[str] = None,
) -> Optional[Path]:
    """
    查找符合命名规范的文件
    
    Args:
        prefix: 用途前缀
        dtype: 数据类型（单个字符串、字符串列表、或 None）
        search_dir: 搜索目录
        ext: 文件扩展名（None 表示任意扩展名）
        exact: True 表示精确匹配，False 表示模糊匹配（忽略某些组件）
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        python_tag: Python 版本标签覆盖
        cuda_tag: CUDA 版本标签覆盖
        arch_tag: 系统架构标签覆盖
        
    Returns:
        找到的文件路径，未找到返回 None
        
    Examples:
        >>> find_file("cublaslt_gemm", search_dir="build", ext=".so")
        PosixPath('build/cublaslt_gemm_H100_cc90_py312_cu124_x86_64.so')
        
        >>> find_file("dequant_bias_tuned", dtype="BF16", search_dir="build", ext=".py")
        PosixPath('build/dequant_bias_tuned_H100_cc90_BF16_py312_cu124_x86_64.py')
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return None
    
    if exact:
        # 精确匹配：构建完整文件名
        if ext:
            filename = build_filename(
                prefix, dtype=dtype, ext=ext,
                gpu_name=gpu_name, cc_tag=cc_tag,
                python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
            )
            target = search_dir / filename
            return target if target.exists() else None
        else:
            # 尝试常见扩展名
            stem = build_stem(
                prefix, dtype=dtype,
                gpu_name=gpu_name, cc_tag=cc_tag,
                python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
            )
            for ext_try in [".so", ".py", ".json", ".csv", ""]:
                target = search_dir / (stem + ext_try)
                if target.exists():
                    return target
            return None
    else:
        # 模糊匹配：使用 glob 模式
        _gpu = gpu_name or hw_info.gpu_name
        _cc = cc_tag or hw_info.cc_tag
        
        # 构建 dtype 模式
        if dtype is None:
            dtype_pattern = "*"
        elif isinstance(dtype, str):
            dtype_pattern = normalize_dtype(dtype)
        else:
            # 多个 dtype 连接
            dtype_pattern = "_".join(normalize_dtype(d) for d in dtype)
        
        # 模糊匹配模式：prefix_GPU_CC_[dtype_]*_py*_cu*_arch
        pattern = f"{prefix}_{_gpu}_{_cc}_{dtype_pattern}_*" if dtype else f"{prefix}_{_gpu}_{_cc}_*"
        if ext:
            if not ext.startswith("."):
                ext = "." + ext
            pattern += ext
        
        matches = list(search_dir.glob(pattern))
        return matches[0] if matches else None


def find_files(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    search_dir: Union[str, Path] = ".",
    ext: Optional[str] = None,
    **kwargs
) -> List[Path]:
    """
    查找所有符合条件的文件
    
    参数同 find_file，但返回所有匹配的文件列表。
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return []
    
    _gpu = kwargs.get("gpu_name") or hw_info.gpu_name
    _cc = kwargs.get("cc_tag") or hw_info.cc_tag
    
    # 构建 dtype 模式
    if dtype is None:
        dtype_pattern = "*"
    elif isinstance(dtype, str):
        dtype_pattern = normalize_dtype(dtype)
    else:
        dtype_pattern = "_".join(normalize_dtype(d) for d in dtype)
    
    # 模糊匹配模式
    pattern = f"{prefix}_{_gpu}_{_cc}_{dtype_pattern}_*" if dtype else f"{prefix}_{_gpu}_{_cc}_*"
    if ext:
        if not ext.startswith("."):
            ext = "." + ext
        pattern += ext
    
    return sorted(search_dir.glob(pattern))


def find_dir(
    dtype: Optional[str] = None,
    search_dir: Union[str, Path] = ".",
    *,
    prefix: Optional[str] = None,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
) -> Optional[Path]:
    """
    查找符合命名规范的目录
    
    格式: {GPU}_{CC}_{dtype} 或 {prefix}_{GPU}_{CC}_{dtype}
    
    Args:
        dtype: 数据类型
        search_dir: 搜索目录
        prefix: 可选前缀
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        
    Returns:
        找到的目录路径，未找到返回 None
    """
    search_dir = Path(search_dir)
    
    if not search_dir.exists():
        return None
    
    dir_name = build_dir_name(
        prefix=prefix, dtype=dtype,
        gpu_name=gpu_name, cc_tag=cc_tag
    )
    
    target = search_dir / dir_name
    return target if target.is_dir() else None


# =============================================================================
# 模块加载
# =============================================================================

# 模块缓存
_module_cache: Dict[str, Any] = {}


def load_module(
    prefix: str,
    dtype: Optional[Union[str, List[str]]] = None,
    search_dir: Union[str, Path] = ".",
    *,
    ext: Optional[str] = None,
    cache: bool = True,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
    python_tag: Optional[str] = None,
    cuda_tag: Optional[str] = None,
    arch_tag: Optional[str] = None,
) -> Any:
    """
    加载 Python 模块（.py 或 .so）
    
    自动根据当前硬件信息构建模块名并加载。
    
    Args:
        prefix: 模块前缀
        dtype: 数据类型（单个字符串、字符串列表、或 None）
        search_dir: 搜索目录
        ext: 文件扩展名（None 表示自动检测 .so 或 .py）
        cache: 是否缓存模块
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        python_tag: Python 版本标签覆盖
        cuda_tag: CUDA 版本标签覆盖
        arch_tag: 系统架构标签覆盖
        
    Returns:
        加载的 Python 模块
        
    Raises:
        FileNotFoundError: 模块文件不存在
        ImportError: 模块加载失败
        
    Examples:
        # 无 dtype（支持多类型的 GEMM 扩展）
        >>> module = load_module("cublaslt_gemm", search_dir="build")
        >>> module.gemm(...)
        
        # 带 dtype
        >>> module = load_module("dequant_bias_tuned", dtype="BF16", search_dir="build")
    """
    search_dir = Path(search_dir)
    
    # 构建缓存键
    dtype_key = str(dtype) if dtype else "None"
    cache_key = f"{prefix}_{dtype_key}_{search_dir}_{gpu_name}_{cc_tag}_{python_tag}_{cuda_tag}_{arch_tag}"
    
    if cache and cache_key in _module_cache:
        return _module_cache[cache_key]
    
    # 查找模块文件
    module_path = None
    
    if ext:
        module_path = find_file(
            prefix, dtype=dtype, search_dir=search_dir, ext=ext,
            gpu_name=gpu_name, cc_tag=cc_tag,
            python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
        )
    else:
        # 优先 .so，然后 .py
        for try_ext in [".so", ".py"]:
            module_path = find_file(
                prefix, dtype=dtype, search_dir=search_dir, ext=try_ext,
                gpu_name=gpu_name, cc_tag=cc_tag,
                python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
            )
            if module_path:
                break
    
    if not module_path:
        expected_name = build_filename(
            prefix, dtype=dtype, ext=ext or ".so/.py",
            gpu_name=gpu_name, cc_tag=cc_tag,
            python_tag=python_tag, cuda_tag=cuda_tag, arch_tag=arch_tag
        )
        raise FileNotFoundError(
            f"模块不存在: {expected_name}\n"
            f"搜索路径: {search_dir.absolute()}\n"
        )
    
    # 添加目录到 sys.path
    if str(search_dir.absolute()) not in sys.path:
        sys.path.insert(0, str(search_dir.absolute()))
    
    # 加载模块
    module_name = module_path.stem
    
    if module_path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块: {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        # .so 文件
        module = importlib.import_module(module_name)
    
    if cache:
        _module_cache[cache_key] = module
    
    return module


def clear_module_cache():
    """清除模块缓存"""
    global _module_cache
    _module_cache.clear()


# =============================================================================
# 文件保存
# =============================================================================

def save_json(
    data: Any,
    prefix: str,
    dtype: Optional[str] = None,
    save_dir: Union[str, Path] = ".",
    *,
    indent: int = 2,
    **kwargs
) -> Path:
    """
    保存数据为 JSON 文件
    
    Args:
        data: 要保存的数据
        prefix: 文件前缀
        dtype: 数据类型
        save_dir: 保存目录
        indent: JSON 缩进
        **kwargs: 传递给 build_filename 的参数
        
    Returns:
        保存的文件路径
    """
    import json
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = build_filename(prefix, dtype=dtype, ext=".json", **kwargs)
    filepath = save_dir / filename
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    return filepath


def load_json(
    prefix: str,
    dtype: Optional[str] = None,
    search_dir: Union[str, Path] = ".",
    **kwargs
) -> Any:
    """
    加载 JSON 文件
    
    Args:
        prefix: 文件前缀
        dtype: 数据类型
        search_dir: 搜索目录
        **kwargs: 传递给 find_file 的参数
        
    Returns:
        加载的数据
        
    Raises:
        FileNotFoundError: 文件不存在
    """
    import json
    
    filepath = find_file(prefix, dtype=dtype, search_dir=search_dir, ext=".json", **kwargs)
    
    if not filepath:
        expected_name = build_filename(prefix, dtype=dtype, ext=".json", **kwargs)
        raise FileNotFoundError(f"JSON 文件不存在: {expected_name} in {search_dir}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    data: List[Dict[str, Any]],
    prefix: str,
    dtype: Optional[str] = None,
    save_dir: Union[str, Path] = ".",
    **kwargs
) -> Path:
    """
    保存数据为 CSV 文件
    
    Args:
        data: 字典列表形式的数据
        prefix: 文件前缀
        dtype: 数据类型
        save_dir: 保存目录
        **kwargs: 传递给 build_filename 的参数
        
    Returns:
        保存的文件路径
    """
    import csv
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    filename = build_filename(prefix, dtype=dtype, ext=".csv", **kwargs)
    filepath = save_dir / filename
    
    if not data:
        filepath.touch()
        return filepath
    
    fieldnames = list(data[0].keys())
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    return filepath


# =============================================================================
# 目录管理
# =============================================================================

def ensure_result_dir(
    base_dir: Union[str, Path],
    dtype: Optional[str] = None,
    *,
    prefix: Optional[str] = None,
    gpu_name: Optional[str] = None,
    cc_tag: Optional[str] = None,
) -> Path:
    """
    确保结果目录存在并返回路径
    
    创建格式为 {GPU}_{CC}_{dtype} 的子目录。
    
    Args:
        base_dir: 基础目录
        dtype: 数据类型
        prefix: 可选前缀
        gpu_name: GPU 名称覆盖
        cc_tag: CC 标签覆盖
        
    Returns:
        创建/已存在的目录路径
        
    Examples:
        >>> result_dir = ensure_result_dir("results", dtype="FP8E4M3")
        >>> # Creates: results/H100_cc90_FP8E4M3/
    """
    base_dir = Path(base_dir)
    dir_name = build_dir_name(prefix=prefix, dtype=dtype, gpu_name=gpu_name, cc_tag=cc_tag)
    result_dir = base_dir / dir_name
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir


# =============================================================================
# 便捷函数
# =============================================================================

def get_gpu_name() -> str:
    """获取 GPU 简称"""
    return hw_info.gpu_name


def get_gpu_cc() -> str:
    """获取 CC 标签"""
    return hw_info.cc_tag


def get_python_version_tag() -> str:
    """获取 Python 版本标签"""
    return hw_info.python_tag


def get_cuda_ver() -> str:
    """获取 CUDA 版本标签"""
    return hw_info.cuda_tag


def get_arch_tag() -> str:
    """获取系统架构标签"""
    return hw_info.arch_tag


def get_sm_code() -> str:
    """获取 SM 代码"""
    return hw_info.sm_code


def print_system_info():
    """打印系统信息"""
    hw_info.print_info()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 数据类型
    "normalize_dtype",
    "DTYPE_ALIASES",
    
    # 硬件信息
    "HardwareInfo",
    "hw_info",
    
    # 文件名构建
    "build_filename",
    "build_stem",
    "build_dir_name",
    
    # 文件查找
    "find_file",
    "find_files",
    "find_dir",
    
    # 模块加载
    "load_module",
    "clear_module_cache",
    
    # 数据保存/加载
    "save_json",
    "load_json",
    "save_csv",
    
    # 目录管理
    "ensure_result_dir",
    
    # 便捷函数
    "get_gpu_name",
    "get_gpu_cc",
    "get_python_version_tag",
    "get_cuda_ver",
    "get_arch_tag",
    "get_sm_code",
    "print_system_info",
]


# =============================================================================
# CLI
# =============================================================================

def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SlideSparse 统一工具库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 显示硬件信息
    python -m slidesparse.utils info
    
    # 生成文件名
    python -m slidesparse.utils name cuBLASLt --dtype FP8E4M3 --ext .so
    
    # 查找文件
    python -m slidesparse.utils find cuBLASLt --dtype FP8E4M3 --dir build
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="命令")
    
    # info 命令
    info_parser = subparsers.add_parser("info", help="显示硬件信息")
    info_parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    
    # name 命令
    name_parser = subparsers.add_parser("name", help="生成文件名")
    name_parser.add_argument("prefix", help="文件前缀")
    name_parser.add_argument("--dtype", help="数据类型")
    name_parser.add_argument("--ext", default="", help="文件扩展名")
    
    # find 命令
    find_parser = subparsers.add_parser("find", help="查找文件")
    find_parser.add_argument("prefix", help="文件前缀")
    find_parser.add_argument("--dtype", help="数据类型")
    find_parser.add_argument("--dir", default=".", help="搜索目录")
    find_parser.add_argument("--ext", help="文件扩展名")
    
    args = parser.parse_args()
    
    if args.command == "info":
        if args.json:
            import json
            print(json.dumps(hw_info.to_dict(), indent=2, ensure_ascii=False))
        else:
            hw_info.print_info()
    
    elif args.command == "name":
        name = build_filename(args.prefix, dtype=args.dtype, ext=args.ext)
        print(name)
    
    elif args.command == "find":
        result = find_file(args.prefix, dtype=args.dtype, search_dir=args.dir, ext=args.ext)
        if result:
            print(result)
        else:
            print(f"未找到匹配的文件", file=sys.stderr)
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
