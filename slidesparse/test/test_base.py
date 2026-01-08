#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 测试基础设施

提供统一的测试工具类和辅助函数，确保测试的:
- 泛用性: 支持不同模型、不同 GPU、不同配置
- 鲁棒性: 完善的错误处理、资源清理、超时控制
- 可扩展性: 易于添加新测试用例、新模型支持

"""

import os
import sys
import time
import json
import functools
import traceback
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from pathlib import Path

# 添加项目根目录到 Python 路径
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
        status_str = f"{self.status.value} {self.name}"
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
            f"通过: {self.passed}  失败: {self.failed}  跳过: {self.skipped}",
            f"耗时: {self.duration:.2f}s",
            "-" * 60,
        ]
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
        """检查是否支持 FP8"""
        cc = cls.cuda_compute_capability()
        # sm_89 (Ada) 或更高支持 FP8
        return cc >= (8, 9)
    
    @classmethod
    def is_cublaslt_enabled(cls) -> bool:
        """检查 cuBLASLt 是否启用"""
        if "cublaslt_enabled" not in cls._cache:
            try:
                from slidesparse.core.cublaslt_config import is_cublaslt_enabled
                cls._cache["cublaslt_enabled"] = is_cublaslt_enabled()
            except ImportError:
                cls._cache["cublaslt_enabled"] = False
        return cls._cache["cublaslt_enabled"]
    
    @classmethod
    def get_env_info(cls) -> Dict[str, Any]:
        """获取完整环境信息"""
        info = {
            "has_cuda": cls.has_cuda(),
            "cuda_device": cls.cuda_device_name(),
            "cuda_cc": cls.cuda_compute_capability(),
            "supports_fp8": cls.supports_fp8(),
            "cublaslt_enabled": cls.is_cublaslt_enabled(),
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
        print("=" * 60)
        print("环境信息")
        print("-" * 60)
        print(f"  Python: {info['python_version']}")
        print(f"  PyTorch: {info['torch_version']}")
        print(f"  vLLM: {info['vllm_version']}")
        print(f"  CUDA: {'可用' if info['has_cuda'] else '不可用'}")
        if info['has_cuda']:
            print(f"  GPU: {info['cuda_device']}")
            print(f"  Compute Capability: {info['cuda_cc']}")
            print(f"  FP8 支持: {'是' if info['supports_fp8'] else '否'}")
        print(f"  cuBLASLt: {'启用' if info['cublaslt_enabled'] else '禁用'}")
        print("=" * 60)


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
    def get_model_info(cls, model_path: Path) -> Dict[str, Any]:
        """获取模型信息"""
        info = {"path": str(model_path), "name": model_path.name}
        
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                info["architecture"] = config.get("architectures", ["Unknown"])[0]
                info["hidden_size"] = config.get("hidden_size", 0)
                info["num_layers"] = config.get("num_hidden_layers", 0)
            except Exception:
                pass
        
        return info


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
def timeout_context(seconds: float, message: str = "操作超时"):
    """超时控制上下文 (仅 Unix)"""
    import signal
    
    def handler(signum, frame):
        raise TimeoutError(message)
    
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(int(seconds))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows 不支持 SIGALRM
        yield


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
            print(f"测试套件: {self.suite_name}")
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
        
        Args:
            func: 要测试的函数
            warmup: 预热次数
            repeat: 重复次数
            synchronize: 是否同步 CUDA
        
        Returns:
            (平均时间 ms, 标准差 ms)
        """
        import torch
        import statistics
        
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
    def compute_tflops(flops: int, time_ms: float) -> float:
        """计算 TFLOPS"""
        return flops / (time_ms * 1e-3) / 1e12


# ============================================================================
# 断言工具
# ============================================================================

class Assertions:
    """断言工具类"""
    
    @staticmethod
    def assert_close(
        actual,
        expected,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        msg: str = ""
    ) -> Tuple[bool, str]:
        """断言两个张量接近"""
        import torch
        
        if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
            max_diff = (actual - expected).abs().max().item()
            mean_diff = (actual - expected).abs().mean().item()
            return False, f"{msg} max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
        return True, f"{msg} 通过"
    
    @staticmethod
    def assert_equal(actual, expected, msg: str = "") -> Tuple[bool, str]:
        """断言相等"""
        if actual != expected:
            return False, f"{msg} 期望 {expected}, 实际 {actual}"
        return True, f"{msg} 通过"
    
    @staticmethod
    def assert_true(condition: bool, msg: str = "") -> Tuple[bool, str]:
        """断言为真"""
        if not condition:
            return False, f"{msg} 条件不满足"
        return True, f"{msg} 通过"


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
        return (True, f"需要 sm_89+, 当前 {EnvironmentChecker.cuda_compute_capability()}")
    return (False, "")


def skip_if_no_model(model_type: str = "FP8") -> Tuple[bool, str]:
    """如果没有指定类型的模型则跳过"""
    model = ModelFinder.find_small_model(model_type)
    if model is None:
        return (True, f"未找到 {model_type} 模型")
    return (False, "")


def skip_if_cublaslt_disabled() -> Tuple[bool, str]:
    """如果 cuBLASLt 禁用则跳过"""
    if not EnvironmentChecker.is_cublaslt_enabled():
        return (True, "cuBLASLt 未启用 (设置 VLLM_USE_CUBLASLT=1)")
    return (False, "")
