#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 测试工具库

提供统一的测试基础设施，包括：
- 测试状态和结果数据类
- 测试装饰器和运行器
- 环境检测工具
- 模型查找工具
- CUDA 内存管理
- 性能测试工具

环境变量设计（两层控制）:
========================

第一层：是否启用 SlideSparse
    DISABLE_SLIDESPARSE=1  →  禁用 SlideSparse，使用 vLLM 原生路径
    DISABLE_SLIDESPARSE=0  →  启用 SlideSparse（默认）

第二层：Kernel 后端选择（三选一，互斥）
    USE_CUBLASLT=1         →  cuBLASLt kernel
    USE_CUSPARSELT=1       →  cuSPARSELt kernel (TODO)
    默认（两者都不设置）   →  CUTLASS fallback

附加选项：
    INNER_DTYPE_FP32=1     →  GEMM 输出使用 FP32（仅 cuBLASLt/cuSPARSELt 时生效）

命令行参数映射：
===============
    --disable-slidesparse  →  DISABLE_SLIDESPARSE=1（baseline，不走 SlideSparse）
    --use-cutlass          →  USE_CUBLASLT=0, USE_CUSPARSELT=0（默认）
    --use-cublaslt         →  USE_CUBLASLT=1
    --use-cusparselt       →  USE_CUSPARSELT=1
    --inner-fp32           →  INNER_DTYPE_FP32=1
"""

import os
import sys
import time
import functools
import traceback
import argparse
import statistics
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from pathlib import Path


# ============================================================================
# 路径设置
# ============================================================================

# utils.py 现在位于 slidesparse/test/utils.py
# 目录结构: PROJECT_ROOT/slidesparse/test/utils.py
TEST_DIR = Path(__file__).parent.absolute()  # slidesparse/test
SLIDESPARSE_DIR = TEST_DIR.parent            # slidesparse
PROJECT_ROOT = SLIDESPARSE_DIR.parent        # vllmbench (项目根目录)
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# ANSI 颜色
# ============================================================================

class Colors:
    """终端颜色"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color
    
    @classmethod
    def red(cls, text: str) -> str:
        return f"{cls.RED}{text}{cls.NC}"
    
    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.NC}"
    
    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.NC}"
    
    @classmethod
    def blue(cls, text: str) -> str:
        return f"{cls.BLUE}{text}{cls.NC}"
    
    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls.CYAN}{text}{cls.NC}"
    
    @classmethod
    def magenta(cls, text: str) -> str:
        return f"{cls.MAGENTA}{text}{cls.NC}"
    
    @classmethod
    def bold(cls, text: str) -> str:
        return f"{cls.BOLD}{text}{cls.NC}"


# ============================================================================
# 测试状态枚举
# ============================================================================

class TestStatus(Enum):
    """测试状态"""
    PASSED = "✓"
    FAILED = "✗"
    SKIPPED = "⊘"
    WARNING = "⚠"


# ============================================================================
# 测试结果数据类
# ============================================================================

@dataclass
class TestResult:
    """单个测试的结果"""
    name: str
    status: TestStatus
    message: str = ""
    duration: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status_char = self.status.value
        if self.status == TestStatus.PASSED:
            status_str = Colors.green(f"{status_char} {self.name}")
        elif self.status == TestStatus.FAILED:
            status_str = Colors.red(f"{status_char} {self.name}")
        elif self.status == TestStatus.SKIPPED:
            status_str = Colors.yellow(f"{status_char} {self.name}")
        else:
            status_str = Colors.yellow(f"{status_char} {self.name}")
        
        if self.message:
            status_str += f": {self.message}"
        if self.duration > 0:
            status_str += f" ({self.duration:.2f}s)"
        return status_str


@dataclass  
class TestSuiteResult:
    """测试套件的结果"""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)
    
    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)
    
    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success(self) -> bool:
        return self.failed == 0
    
    def add(self, result: TestResult):
        self.results.append(result)
    
    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"测试套件: {self.name}",
            "-" * 60,
        ]
        
        passed_str = Colors.green(f"通过: {self.passed}")
        failed_str = Colors.red(f"失败: {self.failed}") if self.failed > 0 else f"失败: {self.failed}"
        skipped_str = Colors.yellow(f"跳过: {self.skipped}") if self.skipped > 0 else f"跳过: {self.skipped}"
        
        lines.append(f"{passed_str}  {failed_str}  {skipped_str}")
        lines.append(f"耗时: {self.duration:.2f}s")
        lines.append("-" * 60)
        
        for r in self.results:
            lines.append(f"  {r}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# 测试装饰器
# ============================================================================

def test_case(name: str = None, skip_if: Callable[[], Tuple[bool, str]] = None):
    """
    测试用例装饰器
    
    Args:
        name: 测试名称（默认使用函数名）
        skip_if: 跳过条件函数，返回 (should_skip, reason)
    
    Example:
        @test_case("导入测试")
        def test_import():
            import slidesparse
            return True, "导入成功"
            
        @test_case(skip_if=lambda: (not torch.cuda.is_available(), "需要 CUDA"))
        def test_cuda_kernel():
            ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> TestResult:
            test_name = name or func.__name__
            
            # 检查跳过条件
            if skip_if:
                should_skip, reason = skip_if()
                if should_skip:
                    return TestResult(
                        name=test_name,
                        status=TestStatus.SKIPPED,
                        message=reason
                    )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 处理返回值
                if isinstance(result, TestResult):
                    result.duration = duration
                    return result
                elif isinstance(result, tuple) and len(result) >= 2:
                    success, message = result[0], result[1]
                    details = result[2] if len(result) > 2 else {}
                    return TestResult(
                        name=test_name,
                        status=TestStatus.PASSED if success else TestStatus.FAILED,
                        message=message,
                        duration=duration,
                        details=details
                    )
                elif isinstance(result, bool):
                    return TestResult(
                        name=test_name,
                        status=TestStatus.PASSED if result else TestStatus.FAILED,
                        duration=duration
                    )
                else:
                    return TestResult(
                        name=test_name,
                        status=TestStatus.PASSED,
                        duration=duration
                    )
                    
            except Exception as e:
                duration = time.time() - start_time
                return TestResult(
                    name=test_name,
                    status=TestStatus.FAILED,
                    message=str(e),
                    duration=duration,
                    details={"traceback": traceback.format_exc()}
                )
        
        return wrapper
    return decorator


# ============================================================================
# 环境检测工具
# ============================================================================

class EnvironmentChecker:
    """环境检测工具类"""
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
    
    @classmethod
    def has_cuda(cls) -> bool:
        """检查 CUDA 是否可用"""
        if "has_cuda" not in cls._cache:
            try:
                import torch
                cls._cache["has_cuda"] = torch.cuda.is_available()
            except ImportError:
                cls._cache["has_cuda"] = False
        return cls._cache["has_cuda"]
    
    @classmethod
    def cuda_device_name(cls) -> str:
        """获取 CUDA 设备名称"""
        if not cls.has_cuda():
            return "N/A"
        if "cuda_device_name" not in cls._cache:
            import torch
            cls._cache["cuda_device_name"] = torch.cuda.get_device_name(0)
        return cls._cache["cuda_device_name"]
    
    @classmethod
    def cuda_compute_capability(cls) -> Tuple[int, int]:
        """获取 CUDA 计算能力"""
        if not cls.has_cuda():
            return (0, 0)
        if "cuda_cc" not in cls._cache:
            import torch
            cls._cache["cuda_cc"] = torch.cuda.get_device_capability(0)
        return cls._cache["cuda_cc"]
    
    @classmethod
    def supports_fp8(cls) -> bool:
        """检查是否支持 FP8 (sm_89+)"""
        cc = cls.cuda_compute_capability()
        return cc >= (8, 9)
    
    @classmethod
    def is_slidesparse_enabled(cls) -> bool:
        """检查 SlideSparse 是否启用"""
        if "slidesparse_enabled" not in cls._cache:
            try:
                from slidesparse.core.config import is_slidesparse_enabled
                cls._cache["slidesparse_enabled"] = is_slidesparse_enabled()
            except ImportError:
                cls._cache["slidesparse_enabled"] = False
        return cls._cache["slidesparse_enabled"]
    
    @classmethod
    def is_cublaslt_enabled(cls) -> bool:
        """检查 cuBLASLt kernel 是否启用"""
        if "cublaslt_enabled" not in cls._cache:
            try:
                from slidesparse.core.config import is_cublaslt_enabled
                cls._cache["cublaslt_enabled"] = is_cublaslt_enabled()
            except ImportError:
                cls._cache["cublaslt_enabled"] = False
        return cls._cache["cublaslt_enabled"]
    
    @classmethod
    def is_cusparselt_enabled(cls) -> bool:
        """检查 cuSPARSELt kernel 是否启用"""
        if "cusparselt_enabled" not in cls._cache:
            try:
                from slidesparse.core.config import is_cusparselt_enabled
                cls._cache["cusparselt_enabled"] = is_cusparselt_enabled()
            except ImportError:
                cls._cache["cusparselt_enabled"] = False
        return cls._cache["cusparselt_enabled"]
    
    @classmethod
    def is_inner_dtype_fp32(cls) -> bool:
        """检查 INNER_DTYPE_FP32 是否启用"""
        if "inner_dtype_fp32" not in cls._cache:
            try:
                from slidesparse.core.config import is_inner_dtype_fp32
                cls._cache["inner_dtype_fp32"] = is_inner_dtype_fp32()
            except ImportError:
                cls._cache["inner_dtype_fp32"] = False
        return cls._cache["inner_dtype_fp32"]
    
    @classmethod
    def get_kernel_name(cls) -> str:
        """获取当前选择的 kernel 名称"""
        if not cls.is_slidesparse_enabled():
            return "vLLM 原生 (CUTLASS)"
        if cls.is_cublaslt_enabled():
            inner = "FP32" if cls.is_inner_dtype_fp32() else "BF16"
            return f"cuBLASLt ({inner})"
        if cls.is_cusparselt_enabled():
            inner = "FP32" if cls.is_inner_dtype_fp32() else "BF16"
            return f"cuSPARSELt ({inner})"
        return "CUTLASS (fallback)"
    
    @classmethod
    def get_env_info(cls) -> Dict[str, Any]:
        """获取完整环境信息"""
        info = {
            "has_cuda": cls.has_cuda(),
            "cuda_device": cls.cuda_device_name(),
            "cuda_cc": cls.cuda_compute_capability(),
            "supports_fp8": cls.supports_fp8(),
            "slidesparse_enabled": cls.is_slidesparse_enabled(),
            "cublaslt_enabled": cls.is_cublaslt_enabled(),
            "cusparselt_enabled": cls.is_cusparselt_enabled(),
            "inner_dtype_fp32": cls.is_inner_dtype_fp32(),
            "kernel_name": cls.get_kernel_name(),
        }
        
        # Python 版本
        info["python_version"] = sys.version.split()[0]
        
        # PyTorch 版本
        try:
            import torch
            info["torch_version"] = torch.__version__
        except ImportError:
            info["torch_version"] = "N/A"
        
        # vLLM 版本
        try:
            import vllm
            info["vllm_version"] = vllm.__version__
        except ImportError:
            info["vllm_version"] = "N/A"
        
        return info
    
    @classmethod
    def print_env_info(cls):
        """打印环境信息"""
        info = cls.get_env_info()
        print("=" * 70)
        print(Colors.bold("环境信息"))
        print("-" * 70)
        print(f"  Python: {info['python_version']}")
        print(f"  PyTorch: {info['torch_version']}")
        print(f"  vLLM: {info['vllm_version']}")
        print(f"  CUDA: {'可用' if info['has_cuda'] else '不可用'}")
        if info['has_cuda']:
            print(f"  GPU: {info['cuda_device']}")
            print(f"  Compute Capability: sm_{info['cuda_cc'][0]}{info['cuda_cc'][1]}")
            fp8_status = Colors.green("支持") if info['supports_fp8'] else Colors.red("不支持")
            print(f"  FP8: {fp8_status}")
        
        print("-" * 70)
        
        # SlideSparse 状态
        if info['slidesparse_enabled']:
            slidesparse_status = Colors.green("启用")
        else:
            slidesparse_status = Colors.yellow("禁用 (vLLM 原生路径)")
        print(f"  SlideSparse: {slidesparse_status}")
        
        if info['slidesparse_enabled']:
            # Kernel 选择（三选一）
            if info['cublaslt_enabled']:
                kernel_status = Colors.cyan("cuBLASLt")
            elif info['cusparselt_enabled']:
                kernel_status = Colors.magenta("cuSPARSELt")
            else:
                kernel_status = Colors.yellow("CUTLASS (fallback)")
            print(f"  Kernel 后端: {kernel_status}")
            
            # Inner dtype（仅 cuBLASLt/cuSPARSELt 时显示）
            if info['cublaslt_enabled'] or info['cusparselt_enabled']:
                inner_dtype = "FP32" if info['inner_dtype_fp32'] else "BF16"
                print(f"  GEMM Inner Dtype: {inner_dtype}")
        
        print("=" * 70)


# ============================================================================
# 模型查找工具
# ============================================================================

class ModelFinder:
    """模型查找工具类"""
    
    # 支持的模型类型
    MODEL_TYPES = {
        "FP8": ["FP8", "fp8", "Fp8"],
        "INT8": ["INT8", "int8", "Int8", "W8A8", "w8a8"],
    }
    
    # 推荐的小模型列表（按优先级排序）
    SMALL_MODELS = [
        "Qwen2.5-0.5B",
        "Qwen2.5-1.5B", 
        "Llama3.2-1B",
        "Qwen2.5-3B",
        "Llama3.2-3B",
    ]
    
    @classmethod
    def get_checkpoints_dir(cls) -> Path:
        """获取 checkpoints 目录路径"""
        return PROJECT_ROOT / "checkpoints"
    
    @classmethod
    def find_models(cls, model_type: str = None) -> List[Path]:
        """
        查找可用模型
        
        Args:
            model_type: 模型类型 ("FP8", "INT8", None=全部)
        
        Returns:
            模型路径列表
        """
        checkpoints_dir = cls.get_checkpoints_dir()
        if not checkpoints_dir.exists():
            return []
        
        models = []
        type_keywords = []
        if model_type and model_type in cls.MODEL_TYPES:
            type_keywords = cls.MODEL_TYPES[model_type]
        
        for item in checkpoints_dir.iterdir():
            if not item.is_dir():
                continue
            
            # 检查是否匹配类型
            if type_keywords:
                if not any(kw in item.name for kw in type_keywords):
                    continue
            
            # 检查是否有必要的文件
            if (item / "config.json").exists():
                models.append(item)
        
        return models
    
    @classmethod
    def find_small_model(cls, model_type: str = "FP8") -> Optional[Path]:
        """
        查找推荐的小模型
        
        Args:
            model_type: 模型类型
        
        Returns:
            模型路径或 None
        """
        models = cls.find_models(model_type)
        if not models:
            return None
        
        # 按推荐顺序查找
        for base_name in cls.SMALL_MODELS:
            for model_path in models:
                if base_name in model_path.name:
                    return model_path
        
        # 返回第一个找到的
        return models[0] if models else None
    
    @classmethod
    def get_test_models(cls, model_type: str = "FP8", max_count: int = 2) -> List[Path]:
        """
        获取测试用模型列表
        
        优先返回 Qwen2.5-0.5B 和 Llama3.2-1B
        """
        priority = ["Qwen2.5-0.5B", "Llama3.2-1B"]
        models = cls.find_models(model_type)
        
        result = []
        # 先添加优先模型
        for name in priority:
            for model in models:
                if name in model.name and model not in result:
                    result.append(model)
                    if len(result) >= max_count:
                        break
            if len(result) >= max_count:
                break
        
        # 如果不够，补充其他模型
        for model in models:
            if model not in result:
                result.append(model)
            if len(result) >= max_count:
                break
        
        return result


# ============================================================================
# 资源管理工具
# ============================================================================

@contextmanager
def cuda_memory_manager():
    """CUDA 内存管理上下文"""
    try:
        yield
    finally:
        if EnvironmentChecker.has_cuda():
            import torch
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


@contextmanager
def suppress_vllm_logs():
    """抑制 vLLM 的日志输出"""
    old_level = os.environ.get("VLLM_LOGGING_LEVEL", "")
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    try:
        yield
    finally:
        if old_level:
            os.environ["VLLM_LOGGING_LEVEL"] = old_level
        else:
            os.environ.pop("VLLM_LOGGING_LEVEL", None)


# ============================================================================
# 测试运行器
# ============================================================================

class TestRunner:
    """测试运行器"""
    
    def __init__(self, suite_name: str, verbose: bool = True):
        self.suite_name = suite_name
        self.verbose = verbose
        self.result = TestSuiteResult(name=suite_name)
    
    def run_test(self, test_func: Callable, *args, **kwargs) -> TestResult:
        """运行单个测试"""
        if self.verbose:
            print(f"\n  运行: {test_func.__name__}...")
        
        result = test_func(*args, **kwargs)
        self.result.add(result)
        
        if self.verbose:
            print(f"    {result}")
        
        return result
    
    def run_all(self, tests: List[Callable]) -> TestSuiteResult:
        """运行所有测试"""
        self.result.start_time = time.time()
        
        if self.verbose:
            print("=" * 60)
            print(Colors.bold(f"测试套件: {self.suite_name}"))
            print("=" * 60)
        
        for test in tests:
            self.run_test(test)
        
        self.result.end_time = time.time()
        
        if self.verbose:
            print("\n" + self.result.summary())
        
        return self.result


# ============================================================================
# 性能测试工具
# ============================================================================

class Benchmarker:
    """性能测试工具"""
    
    @staticmethod
    def benchmark(
        func: Callable,
        warmup: int = 10,
        repeat: int = 100,
        synchronize: bool = True
    ) -> Tuple[float, float]:
        """
        执行基准测试
        
        Returns:
            (平均时间 ms, 标准差 ms)
        """
        import torch
        
        # Warmup
        for _ in range(warmup):
            func()
        
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(repeat):
            start = time.perf_counter()
            func()
            if synchronize and torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        
        mean_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        
        return mean_time, std_time
    
    @staticmethod
    def compute_tflops(M: int, N: int, K: int, time_ms: float) -> float:
        """计算 GEMM TFLOPS"""
        if time_ms <= 0:
            return 0.0
        flops = 2 * M * N * K
        return flops / (time_ms * 1e-3) / 1e12


# ============================================================================
# 跳过条件
# ============================================================================

def skip_if_no_cuda() -> Tuple[bool, str]:
    """如果没有 CUDA 则跳过"""
    return (not EnvironmentChecker.has_cuda(), "需要 CUDA")


def skip_if_no_fp8() -> Tuple[bool, str]:
    """如果不支持 FP8 则跳过"""
    if not EnvironmentChecker.has_cuda():
        return (True, "需要 CUDA")
    if not EnvironmentChecker.supports_fp8():
        cc = EnvironmentChecker.cuda_compute_capability()
        return (True, f"需要 sm_89+, 当前 sm_{cc[0]}{cc[1]}")
    return (False, "")


def skip_if_no_model(model_type: str = "FP8") -> Tuple[bool, str]:
    """如果没有指定类型的模型则跳过"""
    model = ModelFinder.find_small_model(model_type)
    if model is None:
        return (True, f"未找到 {model_type} 模型")
    return (False, "")


def skip_if_slidesparse_disabled() -> Tuple[bool, str]:
    """如果 SlideSparse 禁用则跳过"""
    if not EnvironmentChecker.is_slidesparse_enabled():
        return (True, "SlideSparse 未启用 (DISABLE_SLIDESPARSE=1)")
    return (False, "")


def skip_if_cublaslt_disabled() -> Tuple[bool, str]:
    """如果 cuBLASLt 禁用则跳过"""
    if not EnvironmentChecker.is_cublaslt_enabled():
        return (True, "cuBLASLt 未启用 (设置 USE_CUBLASLT=1)")
    return (False, "")


def skip_if_cusparselt_disabled() -> Tuple[bool, str]:
    """如果 cuSPARSELt 禁用则跳过"""
    if not EnvironmentChecker.is_cusparselt_enabled():
        return (True, "cuSPARSELt 未启用 (设置 USE_CUSPARSELT=1)")
    return (False, "")


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_common_args(description: str) -> argparse.ArgumentParser:
    """
    解析通用命令行参数
    
    环境变量设计（两层）：
    ====================
    
    第一层：是否启用 SlideSparse
        --disable-slidesparse  →  DISABLE_SLIDESPARSE=1（baseline）
        默认                   →  DISABLE_SLIDESPARSE=0（启用 SlideSparse）
    
    第二层：Kernel 后端选择（三选一，互斥）
        --use-cutlass    →  CUTLASS fallback（默认）
        --use-cublaslt   →  USE_CUBLASLT=1
        --use-cusparselt →  USE_CUSPARSELT=1
    
    附加选项：
        --inner-fp32     →  INNER_DTYPE_FP32=1
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 默认: SlideSparse + CUTLASS fallback
  %(prog)s --disable-slidesparse    # vLLM 原生路径 (baseline)
  %(prog)s --use-cublaslt           # SlideSparse + cuBLASLt
  %(prog)s --use-cublaslt --inner-fp32  # cuBLASLt + FP32 累加
  %(prog)s --use-cusparselt         # SlideSparse + cuSPARSELt (TODO)
        """
    )
    
    # 第一层：SlideSparse 开关
    parser.add_argument(
        "--disable-slidesparse", 
        action="store_true",
        help="禁用 SlideSparse，使用 vLLM 原生路径 (baseline)"
    )
    
    # 第二层：Kernel 后端选择（互斥组）
    kernel_group = parser.add_mutually_exclusive_group()
    kernel_group.add_argument(
        "--use-cutlass", 
        action="store_true",
        help="使用 CUTLASS fallback（默认）"
    )
    kernel_group.add_argument(
        "--use-cublaslt", 
        action="store_true",
        help="使用 cuBLASLt kernel"
    )
    kernel_group.add_argument(
        "--use-cusparselt", 
        action="store_true",
        help="使用 cuSPARSELt kernel (TODO)"
    )
    
    # 附加选项
    parser.add_argument(
        "--inner-fp32", 
        action="store_true", 
        help="GEMM 输出使用 FP32（仅 cuBLASLt/cuSPARSELt 时生效）"
    )
    
    return parser


def apply_env_args(args: argparse.Namespace) -> None:
    """
    应用环境变量参数
    
    根据命令行参数设置对应的环境变量
    """
    # 第一层：DISABLE_SLIDESPARSE
    if getattr(args, 'disable_slidesparse', False):
        os.environ["DISABLE_SLIDESPARSE"] = "1"
        # baseline 模式下，清除其他环境变量
        os.environ.pop("USE_CUBLASLT", None)
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("INNER_DTYPE_FP32", None)
    else:
        os.environ["DISABLE_SLIDESPARSE"] = "0"
        
        # 第二层：Kernel 后端选择
        if getattr(args, 'use_cublaslt', False):
            os.environ["USE_CUBLASLT"] = "1"
            os.environ.pop("USE_CUSPARSELT", None)
        elif getattr(args, 'use_cusparselt', False):
            os.environ["USE_CUSPARSELT"] = "1"
            os.environ.pop("USE_CUBLASLT", None)
        else:
            # 默认或 --use-cutlass: CUTLASS fallback
            os.environ.pop("USE_CUBLASLT", None)
            os.environ.pop("USE_CUSPARSELT", None)
        
        # 附加选项：INNER_DTYPE_FP32
        if getattr(args, 'inner_fp32', False):
            os.environ["INNER_DTYPE_FP32"] = "1"
        else:
            os.environ.pop("INNER_DTYPE_FP32", None)
    
    # 清除缓存以重新读取环境变量
    EnvironmentChecker.clear_cache()


def get_backend_name(use_cublaslt: bool = False, use_cusparselt: bool = False, 
                     inner_fp32: bool = False) -> str:
    """
    根据参数获取后端名称
    
    用于测试输出显示
    """
    if use_cublaslt:
        suffix = " (FP32累加)" if inner_fp32 else ""
        return f"SlideSparse + cuBLASLt{suffix}"
    elif use_cusparselt:
        suffix = " (FP32累加)" if inner_fp32 else ""
        return f"SlideSparse + cuSPARSELt{suffix}"
    else:
        return "SlideSparse + CUTLASS"


def set_env_for_baseline() -> Dict[str, Optional[str]]:
    """
    设置环境变量为 baseline (vLLM 原生路径)
    
    Returns:
        保存的原环境变量，用于恢复
    """
    saved = {
        "DISABLE_SLIDESPARSE": os.environ.get("DISABLE_SLIDESPARSE"),
        "USE_CUBLASLT": os.environ.get("USE_CUBLASLT"),
        "USE_CUSPARSELT": os.environ.get("USE_CUSPARSELT"),
        "INNER_DTYPE_FP32": os.environ.get("INNER_DTYPE_FP32"),
    }
    
    os.environ["DISABLE_SLIDESPARSE"] = "1"
    os.environ.pop("USE_CUBLASLT", None)
    os.environ.pop("USE_CUSPARSELT", None)
    os.environ.pop("INNER_DTYPE_FP32", None)
    
    EnvironmentChecker.clear_cache()
    return saved


def set_env_for_test(use_cublaslt: bool = False, use_cusparselt: bool = False,
                     inner_fp32: bool = False) -> Dict[str, Optional[str]]:
    """
    设置环境变量为测试配置
    
    Args:
        use_cublaslt: 使用 cuBLASLt kernel
        use_cusparselt: 使用 cuSPARSELt kernel
        inner_fp32: 使用 FP32 累加
    
    Returns:
        保存的原环境变量，用于恢复
    """
    saved = {
        "DISABLE_SLIDESPARSE": os.environ.get("DISABLE_SLIDESPARSE"),
        "USE_CUBLASLT": os.environ.get("USE_CUBLASLT"),
        "USE_CUSPARSELT": os.environ.get("USE_CUSPARSELT"),
        "INNER_DTYPE_FP32": os.environ.get("INNER_DTYPE_FP32"),
    }
    
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    if use_cublaslt:
        os.environ["USE_CUBLASLT"] = "1"
        os.environ.pop("USE_CUSPARSELT", None)
    elif use_cusparselt:
        os.environ["USE_CUSPARSELT"] = "1"
        os.environ.pop("USE_CUBLASLT", None)
    else:
        os.environ.pop("USE_CUBLASLT", None)
        os.environ.pop("USE_CUSPARSELT", None)
    
    if inner_fp32:
        os.environ["INNER_DTYPE_FP32"] = "1"
    else:
        os.environ.pop("INNER_DTYPE_FP32", None)
    
    EnvironmentChecker.clear_cache()
    return saved


def restore_env(saved: Dict[str, Optional[str]]) -> None:
    """
    恢复环境变量
    
    Args:
        saved: 之前保存的环境变量
    """
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    
    EnvironmentChecker.clear_cache()
