#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse vLLM Throughput Benchmark 脚本 (官方vllm bench)

用于精确测试 W8A8 量化模型在不同 Backend/Sparsity/M 下的 Prefill/Decode 性能。

测试维度层级:
  Model → Backend → Sparsity → Stage → M

Backend 支持:
  - cutlass:    SlideSparse CUTLASS fallback (baseline)
  - cublaslt:   SlideSparse cuBLASLt dense GEMM
  - cusparselt: SlideSparse cuSPARSELt 2:N sparse GEMM (需要预稀疏化的 checkpoint)

核心设计思想:
  - Prefill 测试: 控制 M_prefill = max_num_seqs × prompt_length，最小化 Decode 开销
  - Decode 测试:  控制 M_decode = max_num_seqs，最小化 Prefill 开销
  - 动态计算 max-model-len 以最大化 KV Cache 利用率 (Tight Fit 策略)
  - 禁用 Chunked Prefill 以获得纯净的性能数据

Usage 示例:
    # 默认测试 (使用 DEFAULT_MODEL_LIST)
    python3 throughput_benchmark.py
    
    # 测试特定模型的所有 backend
    python3 throughput_benchmark.py --model qwen2.5-0.5b-fp8
    
    # 只测试 cutlass backend (baseline)
    python3 throughput_benchmark.py --model fp8 --backend cutlass
    
    # 只测试 cublaslt backend
    python3 throughput_benchmark.py --model fp8 --backend cublaslt
    
    # 只测试 cusparselt 特定稀疏度
    python3 throughput_benchmark.py --model fp8 --backend cusparselt --sparsity 2_4,2_8
    
    # 快速测试 (少量 M 值)
    python3 throughput_benchmark.py --model qwen2.5-0.5b-fp8 --M quick
    
    # Dry-run 验证
    python3 throughput_benchmark.py --model fp8 --dry-run
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
import signal
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# 确保可以导入 slidesparse
_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import (
    HardwareInfo,
    model_registry,
    check_quant_support,
    get_model_local_path,
    hw_info,
    build_stem,
    extract_model_name,
)
from slidesparse.tools.utils import (
    Colors,
    print_header,
    print_subheader,
    print_info,
    print_success,
    print_warning,
    print_error,
    strip_ansi,
    CHECKPOINT_DIR,
    get_vllm_env_vars,
    check_triton_support_and_warn,
    print_hardware_info,
    get_hw_folder_name,
    build_backend_result_dir,
    get_checkpoint_path,
    auto_convert_sparse_checkpoint,
)


# ============================================================================
# Backend 支持检测
# ============================================================================

def check_backend_support(backend: str, quant: str) -> Tuple[bool, str]:
    """
    检查指定 backend 是否支持当前 GPU 和量化类型
    
    Args:
        backend: "cutlass" / "cublaslt" / "cusparselt"
        quant: "fp8" / "int8"
    
    Returns:
        (supported, reason)
    """
    quant_upper = quant.upper()
    
    if backend == "cutlass":
        # CUTLASS 是 SlideSparse 的 fallback 路径（内部调用 vLLM 的 cutlass_scaled_mm）
        if quant_upper == "INT8":
            supported, reason = hw_info.supports_vllm_cutlass_int8
            if not supported:
                return False, f"vLLM CUTLASS INT8 不支持: {reason}"
        elif quant_upper == "FP8":
            supported, reason = hw_info.supports_vllm_cutlass_fp8
            if not supported:
                return False, f"vLLM CUTLASS FP8 不支持: {reason}"
        return True, "OK"
    
    elif backend == "cublaslt":
        # cuBLASLt 支持 sm_70+
        supported, reason = hw_info.supports_cublaslt
        if not supported:
            return False, f"cuBLASLt 不支持: {reason}"
        return True, "OK"
    
    elif backend == "cusparselt":
        # cuSPARSELt 支持 sm_80+
        supported, reason = hw_info.supports_cusparselt
        if not supported:
            return False, f"cuSPARSELt 不支持: {reason}"
        return True, "OK"
    
    return False, f"未知 backend: {backend}"


# ============================================================================
# 全局配置参数
# ============================================================================

# 默认测试的模型列表 (不加 --model 参数时使用)
DEFAULT_MODEL_LIST = [
    "llama3.2-1b-int8",
    "llama3.2-1b-fp8",
]

# Prefill 测试配置
DEFAULT_M_LIST_PREFILL = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
N_PREFILL = 128  # Prefill 重复次数

# Decode 测试配置
DEFAULT_M_LIST_DECODE = [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]
N_DECODE = 256  # Decode 生成的 token 数

# 快速测试 M 列表
QUICK_M_LIST = [16, 128, 256]

# Prompt length 配置
PROMPT_LENGTH_CAP_PREFILL = 1024  # Prefill 模式下 prompt_length 的上限
PROMPT_LENGTH_FIXED_DECODE = 16   # Decode 模式下固定的 prompt_length

# max-model-len 计算的 Buffer
MODEL_LEN_BUFFER = 16

# 默认稀疏度列表 (仅 cusparselt)
DEFAULT_SPARSITY_LIST = ["2_4", "2_6", "2_8", "2_10", "2_12"]

# 支持的 Backend（顺序即默认测试顺序）
DEFAULT_BACKEND_LIST = ["cutlass", "cublaslt", "cusparselt"]

# 日志级别 (WARNING 减少日志开销，需要调试时改为 INFO)
VLLM_LOG_LEVEL = "WARNING"

# GPU 配置
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
GPU_MEMORY_UTILIZATION = 0.8

# 全局状态 (用于信号处理)
_CURRENT_OUTPUT_DIR: Path | None = None
_GLOBAL_LOG_FILE: Path | None = None


# ============================================================================
# 日志管理
# ============================================================================

class TeeLogger:
    """
    同时输出到控制台和日志文件的 Logger
    
    使用方式:
        with TeeLogger(log_file) as logger:
            # 所有 print 输出都会被记录
            print("test")
    """
    
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.original_stdout = None
        self.original_stderr = None
        self.file = None
    
    def __enter__(self):
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.file = open(self.log_file, "a", encoding="utf-8")
        sys.stdout = _TeeStream(self.original_stdout, self.file)
        sys.stderr = _TeeStream(self.original_stderr, self.file)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        if self.file:
            self.file.close()
        return False


class _TeeStream:
    """同时写入两个流的包装器"""
    
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
    
    def write(self, data):
        self.stream1.write(data)
        # 写入文件时去除 ANSI 颜色码
        self.stream2.write(strip_ansi(data))
    
    def flush(self):
        self.stream1.flush()
        self.stream2.flush()
    
    def isatty(self):
        return self.stream1.isatty()


def create_log_file(result_base: Path, args) -> Path:
    """
    创建日志文件
    
    Args:
        result_base: 结果目录基础路径
        args: 命令行参数
    
    Returns:
        日志文件路径
    """
    logs_dir = result_base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用时间戳作为文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"benchmark_{timestamp}.log"
    
    # 写入头部信息
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"SlideSparse vLLM Throughput Benchmark Log\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        # 原始命令行
        f.write("原始命令:\n")
        f.write(f"  {' '.join(sys.argv)}\n\n")
        
        # 解析后的参数
        f.write("命令行参数:\n")
        for key, value in vars(args).items():
            f.write(f"  --{key.replace('_', '-')}: {value}\n")
        f.write("\n")
        
        # 硬件信息
        f.write("硬件信息:\n")
        f.write(f"  GPU: {hw_info.gpu_name}\n")
        f.write(f"  Compute Capability: {hw_info.cc_tag}\n")
        f.write(f"  VRAM: {hw_info.gpu_memory_gb:.1f} GB\n")
        f.write(f"  CUDA: {hw_info.cuda_runtime_version}\n")
        f.write(f"  Python: {hw_info.python_tag}\n")
        f.write("\n")
        
        # Backend 环境变量（运行时设置，此处记录初始状态）
        f.write("Backend 环境变量 (初始状态):\n")
        f.write(f"  DISABLE_SLIDESPARSE: {os.environ.get('DISABLE_SLIDESPARSE', '未设置')}\n")
        f.write(f"  USE_CUBLASLT: {os.environ.get('USE_CUBLASLT', '未设置')}\n")
        f.write(f"  USE_CUSPARSELT: {os.environ.get('USE_CUSPARSELT', '未设置')}\n")
        f.write(f"  SPARSITY: {os.environ.get('SPARSITY', '未设置')}\n")
        f.write(f"  INNER_DTYPE_32: {os.environ.get('INNER_DTYPE_32', '未设置')}\n")
        f.write("\n")
        f.write("=" * 70 + "\n\n")
    
    return log_file


# ============================================================================
# 信号处理
# ============================================================================

def _signal_handler(signum, frame):
    """处理中断信号 (SIGINT/SIGTERM)"""
    print()
    print("=" * 60)
    print("测试被中断!")
    if _CURRENT_OUTPUT_DIR is not None:
        print(f"当前结果目录: {_CURRENT_OUTPUT_DIR}")
    print("=" * 60)
    sys.exit(130)


def _setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class TestParams:
    """测试参数"""
    prompt_length: int
    max_num_seqs: int
    num_prompts: int
    output_len: int
    max_model_len: int
    n_prefill: int
    n_decode: int
    m_prefill: int
    m_decode: int


@dataclass
class BenchmarkConfig:
    """Benchmark 配置"""
    models: List[str]
    backends: List[str]
    sparsities: List[str]
    stages: List[str]
    m_list_prefill: List[int]
    m_list_decode: List[int]
    n_repeat: Optional[int]
    inner_32: bool
    enforce_eager: bool
    dry_run: bool
    gpu_memory_util: float
    gpu_id: str


# ============================================================================
# 环境变量管理
# ============================================================================

def set_backend_env(
    backend: str,
    sparsity: Optional[str] = None,
    inner_32: bool = False,
    model_name: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    设置 Backend 对应的环境变量
    
    Args:
        backend: "cutlass" / "cublaslt" / "cusparselt"
        sparsity: 稀疏配置 (仅 cusparselt)
        inner_32: 是否使用高精度累加
        model_name: 模型名称，用于加载 model-specific 的 tuned kernels
    
    Returns:
        保存的原环境变量，用于恢复
    """
    saved = {
        "DISABLE_SLIDESPARSE": os.environ.get("DISABLE_SLIDESPARSE"),
        "USE_CUBLASLT": os.environ.get("USE_CUBLASLT"),
        "USE_CUSPARSELT": os.environ.get("USE_CUSPARSELT"),
        "INNER_DTYPE_32": os.environ.get("INNER_DTYPE_32"),
        "SPARSITY": os.environ.get("SPARSITY"),
        "SLIDESPARSE_MODEL_NAME": os.environ.get("SLIDESPARSE_MODEL_NAME"),
        "SLIDESPARSE_MODEL_NAME_WITH_SLIDE": os.environ.get("SLIDESPARSE_MODEL_NAME_WITH_SLIDE"),
    }
    
    # 所有 backend 都通过 SlideSparse 转发（DISABLE_SLIDESPARSE=0）
    # - cutlass:    不设置 USE_CUBLASLT/USE_CUSPARSELT → SlideSparse CUTLASS fallback
    # - cublaslt:   USE_CUBLASLT=1
    # - cusparselt: USE_CUSPARSELT=1 + SPARSITY
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    if backend == "cutlass":
        # CUTLASS: 清除其他 backend 的环境变量，使用 SlideSparse 默认的 CUTLASS 路径
        os.environ.pop("USE_CUBLASLT", None)
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("SPARSITY", None)
        os.environ.pop("SLIDESPARSE_MODEL_NAME", None)  # CUTLASS 不需要
        os.environ.pop("SLIDESPARSE_MODEL_NAME_WITH_SLIDE", None)
    elif backend == "cublaslt":
        os.environ["USE_CUBLASLT"] = "1"
        os.environ.pop("USE_CUSPARSELT", None)
        os.environ.pop("SPARSITY", None)
        # cuBLASLt 需要 model_name 来加载 tuned kernels
        if model_name:
            # 设置完整 checkpoint 名 (可能带 -SlideSparse- 后缀)
            os.environ["SLIDESPARSE_MODEL_NAME_WITH_SLIDE"] = model_name
            # 设置基础模型名（去除 -SlideSparse-2_L 后缀），用于 kernel 查找
            base_model = extract_model_name(model_name)
            os.environ["SLIDESPARSE_MODEL_NAME"] = base_model
    elif backend == "cusparselt":
        os.environ["USE_CUSPARSELT"] = "1"
        os.environ.pop("USE_CUBLASLT", None)
        if sparsity:
            os.environ["SPARSITY"] = sparsity
        # cuSPARSELt 需要 model_name 来加载 tuned kernels
        if model_name:
            # 设置完整 checkpoint 名 (可能带 -SlideSparse- 后缀)
            os.environ["SLIDESPARSE_MODEL_NAME_WITH_SLIDE"] = model_name
            # 设置基础模型名（去除 -SlideSparse-2_L 后缀），用于 kernel 查找
            base_model = extract_model_name(model_name)
            os.environ["SLIDESPARSE_MODEL_NAME"] = base_model
    
    if inner_32:
        os.environ["INNER_DTYPE_32"] = "1"
    else:
        os.environ.pop("INNER_DTYPE_32", None)
    
    return saved


def restore_env(saved: Dict[str, Optional[str]]) -> None:
    """恢复环境变量"""
    for key, value in saved.items():
        if value is not None:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)


# ============================================================================
# 辅助函数
# ============================================================================

def _truncate(s: str, max_len: int = 45) -> str:
    """截断字符串以适应显示框"""
    return s[:max_len-3] + "..." if len(s) > max_len else s


def parse_m_list(m_str: Optional[str], stage: str) -> List[int]:
    """
    解析 M 值列表字符串
    
    Args:
        m_str: M 值字符串 (逗号分隔) 或 "quick" 或 None
        stage: "prefill" 或 "decode"
    
    Returns:
        M 值列表
    """
    if m_str is None:
        return DEFAULT_M_LIST_PREFILL if stage == "prefill" else DEFAULT_M_LIST_DECODE
    
    if m_str.lower() == "quick":
        return QUICK_M_LIST
    
    return [int(x.strip()) for x in m_str.split(",")]


def parse_model_list(model_arg: Optional[str]) -> List[str]:
    """
    解析模型列表参数
    
    Args:
        model_arg: --model 参数值
    
    Returns:
        模型 key 列表
    """
    if model_arg is None:
        return DEFAULT_MODEL_LIST
    
    model_arg_lower = model_arg.lower()
    
    if model_arg_lower == "all":
        # 返回所有模型
        return list(model_registry.keys())
    
    if model_arg_lower in ("fp8", "int8"):
        # 按量化类型筛选
        return [e.key for e in model_registry.list(quant=model_arg_lower)]
    
    # 具体模型名
    if model_registry.get(model_arg_lower):
        return [model_arg_lower]
    
    # 尝试原样返回
    return [model_arg]


def parse_backend_list(backend_arg: Optional[str]) -> List[str]:
    """解析 backend 列表参数"""
    if backend_arg is None or backend_arg.lower() == "all":
        return DEFAULT_BACKEND_LIST.copy()
    
    backends = [b.strip().lower() for b in backend_arg.split(",")]
    for b in backends:
        if b not in DEFAULT_BACKEND_LIST:
            print_warning(f"未知 backend: {b}，将被忽略")
    return [b for b in backends if b in DEFAULT_BACKEND_LIST]


def parse_sparsity_list(sparsity_arg: Optional[str]) -> List[str]:
    """解析 sparsity 列表参数"""
    if sparsity_arg is None:
        return DEFAULT_SPARSITY_LIST.copy()
    
    return [s.strip() for s in sparsity_arg.split(",")]


def parse_stage_list(stage_arg: Optional[str]) -> List[str]:
    """解析 stage 列表参数"""
    if stage_arg is None or stage_arg.lower() == "all":
        return ["prefill", "decode"]
    
    return [stage_arg.lower()]


def calculate_test_params(m_value: int, test_mode: str, n_repeat: Optional[int] = None) -> TestParams:
    """
    根据测试模式和 M 值计算所有测试参数
    
    Args:
        m_value: M 值
        test_mode: 测试模式 (prefill/decode)
        n_repeat: 重复次数 (覆盖默认值)
        
    Returns:
        TestParams 数据类实例
    """
    if test_mode == "prefill":
        # Prefill 测试: M_prefill = max_num_seqs × prompt_length
        n_prefill_val = n_repeat if n_repeat else N_PREFILL
        
        if m_value <= PROMPT_LENGTH_CAP_PREFILL:
            prompt_length = m_value
            max_num_seqs = 1
        else:
            prompt_length = PROMPT_LENGTH_CAP_PREFILL
            # 使用 ceiling 除法，确保 M_prefill >= m_value
            max_num_seqs = (m_value + prompt_length - 1) // prompt_length
        
        num_prompts = n_prefill_val * max_num_seqs
        output_len = 1  # 最小化 Decode
        max_model_len = prompt_length + output_len + MODEL_LEN_BUFFER
        
        m_prefill = max_num_seqs * prompt_length
        m_decode = max_num_seqs
        
        return TestParams(
            prompt_length=prompt_length,
            max_num_seqs=max_num_seqs,
            num_prompts=num_prompts,
            output_len=output_len,
            max_model_len=max_model_len,
            n_prefill=n_prefill_val,
            n_decode=0,
            m_prefill=m_prefill,
            m_decode=m_decode,
        )
    else:
        # Decode 测试: M_decode = max_num_seqs (batch size)
        n_decode_val = n_repeat if n_repeat else N_DECODE
        
        prompt_length = PROMPT_LENGTH_FIXED_DECODE
        max_num_seqs = m_value
        num_prompts = max_num_seqs
        output_len = n_decode_val
        max_model_len = prompt_length + output_len + MODEL_LEN_BUFFER
        
        m_prefill = max_num_seqs * prompt_length
        m_decode = max_num_seqs
        
        return TestParams(
            prompt_length=prompt_length,
            max_num_seqs=max_num_seqs,
            num_prompts=num_prompts,
            output_len=output_len,
            max_model_len=max_model_len,
            n_prefill=1,
            n_decode=n_decode_val,
            m_prefill=m_prefill,
            m_decode=m_decode,
        )


# ============================================================================
# 核心测试函数
# ============================================================================

def run_single_m_test(
    model_key: str,
    m_value: int,
    test_mode: str,
    backend: str,
    result_json_dir: Path,
    log_file: Path,
    checkpoint_path: Path,
    *,
    sparsity: Optional[str] = None,
    n_repeat: Optional[int] = None,
    inner_32: bool = False,
    gpu_memory_util: float = GPU_MEMORY_UTILIZATION,
    gpu_id: str = GPU_ID,
    enforce_eager: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    运行单个 M 值的吞吐测试
    
    Returns:
        成功返回 True，失败返回 False
    """
    # 获取模型信息
    entry = model_registry.get(model_key)
    if entry is None:
        print_error(f"模型不存在: {model_key}")
        return False
    
    # 计算测试参数
    params = calculate_test_params(m_value, test_mode, n_repeat)
    
    # 结果文件名
    result_file = result_json_dir / f"{entry.local_name}_M{m_value}.json"
    
    # - max_num_batched_tokens: 设置为目标 M 值，控制每次迭代处理的最大 token 数
    #   - Prefill: M = max_num_seqs * prompt_length
    #   - Decode:  M = max_num_seqs (batch size)
    if test_mode == "prefill":
        max_num_batched_tokens = params.m_prefill  # = max_num_seqs * prompt_length
    else:
        max_num_batched_tokens = params.m_decode   # = max_num_seqs
    
    # vLLM 配置约束:
    # 1. max_model_len >= prompt_len + output_len (否则无法处理请求)
    # 2. max_num_batched_tokens >= max_model_len (配置验证要求)
    #
    # 实际 M 值控制逻辑:
    # - max_num_batched_tokens 只是"允许的上限"，不是强制值
    # - 实际 M 由 max_num_seqs 和请求状态决定:
    #   - Prefill: M = max_num_seqs * prompt_len ✅
    #   - Decode:  M = max_num_seqs (每序列生成 1 token) ✅
    # - --no-enable-chunked-prefill 确保 prompt 不被分片
    #
    # 因此，即使 max_num_batched_tokens > target_M，实际 M 仍然等于 target_M
    min_model_len = params.prompt_length + params.output_len
    effective_max_model_len = min_model_len
    
    # 如果 max_num_batched_tokens < min_model_len，需要放宽以通过配置验证
    if max_num_batched_tokens < min_model_len:
        max_num_batched_tokens = min_model_len
    
    # 构建 backend 显示名
    if backend == "cutlass":
        backend_display = "CUTLASS (SlideSparse fallback)"
    elif backend == "cusparselt" and sparsity:
        backend_display = f"cuSPARSELt ({sparsity.replace('_', ':')})"
    else:
        backend_display = "cuBLASLt"
    
    # cuBLASLt INT8 输出固定为 INT32
    if backend == "cublaslt" and entry.quant.lower() == "int8":
        backend_display += " [INT32 output]"
    elif inner_32:
        backend_display += " [inner32]"
    
    # 显示测试参数
    print()
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    测试参数                                  │")
    print("├─────────────────────────────────────────────────────────────┤")
    print(f"│ 模型:     {_truncate(entry.local_name, 48):<48}│")
    print(f"│ Backend:  {_truncate(backend_display, 48):<48}│")
    print(f"│ 阶段:     {test_mode:<48}│")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ GEMM M 维度 (精确控制):")
    print(f"│   目标 M        = {m_value}")
    print(f"│   M_prefill     = {params.m_prefill} (= {params.max_num_seqs} x {params.prompt_length})")
    print(f"│   M_decode      = {params.m_decode}")
    print(f"│   batched_tokens = {max_num_batched_tokens} (控制 M 的关键参数)")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ vLLM 参数:")
    print(f"│   --input-len              = {params.prompt_length}")
    print(f"│   --output-len             = {params.output_len}")
    print(f"│   --num-prompts            = {params.num_prompts}")
    print(f"│   --max-num-seqs           = {params.max_num_seqs}")
    print(f"│   --max-model-len          = {effective_max_model_len}")
    print(f"│   --max-num-batched-tokens = {max_num_batched_tokens}")
    print(f"│   --no-enable-chunked-prefill")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│ 迭代次数:")
    print(f"│   N_prefill = {params.n_prefill}")
    print(f"│   N_decode  = {params.n_decode}")
    if enforce_eager:
        print("├─────────────────────────────────────────────────────────────┤")
        print("│ 编译模式: --enforce-eager")
    print("└─────────────────────────────────────────────────────────────┘")
    print()
    
    # 设置环境变量（从 checkpoint_path 提取 model_name）
    model_name = checkpoint_path.name if checkpoint_path else None
    saved_env = set_backend_env(backend, sparsity, inner_32, model_name)
    
    try:
        # 构建环境变量
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.update(get_vllm_env_vars(log_level=VLLM_LOG_LEVEL))
        
        # slidesparse 没有通过 pip 安装，需要添加项目根目录到 PYTHONPATH
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{_PROJECT_ROOT}:{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = str(_PROJECT_ROOT)
        
        # 构建命令
        # 关键参数说明:
        # - --max-num-batched-tokens: 精确控制每次迭代的 M 值
        # - --max-num-seqs: 控制每次迭代的最大序列数
        # - --no-enable-chunked-prefill: 禁用 chunked prefill，确保 prompt 不被分片
        cmd = [
            "vllm", "bench", "throughput",
            "--model", str(checkpoint_path),
            "--dataset-name", "random",
            "--input-len", str(params.prompt_length),
            "--output-len", str(params.output_len),
            "--num-prompts", str(params.num_prompts),
            "--max-num-seqs", str(params.max_num_seqs),
            "--max-model-len", str(effective_max_model_len),
            "--max-num-batched-tokens", str(max_num_batched_tokens),
            "--no-enable-chunked-prefill",  # 禁用 chunked prefill 以精确控制 M
            "--gpu-memory-utilization", str(gpu_memory_util),
            "--disable-log-stats",
            "--output-json", str(result_file),
        ]
        
        if enforce_eager:
            cmd.append("--enforce-eager")
        
        # 记录到日志
        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f"========== M={m_value} ==========\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Backend: {backend_display}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Params: prompt_len={params.prompt_length}, output_len={params.output_len}, ")
            f.write(f"num_prompts={params.num_prompts}, max_num_seqs={params.max_num_seqs}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write("\n")
        
        # Dry-run 模式
        if dry_run:
            print_info("[DRY-RUN] 将执行的命令:")
            env_str = f"CUDA_VISIBLE_DEVICES={gpu_id} DISABLE_SLIDESPARSE=0"
            if backend == "cutlass":
                pass  # CUTLASS: 不需要额外环境变量
            elif backend == "cublaslt":
                env_str += " USE_CUBLASLT=1"
            elif backend == "cusparselt":
                env_str += f" USE_CUSPARSELT=1 SPARSITY={sparsity}"
            if inner_32:
                env_str += " INNER_DTYPE_32=1"
            print(f"{env_str} {' '.join(cmd)}")
            print()
            # 生成模拟结果
            with open(result_file, "w") as f:
                json.dump({
                    "requests_per_second": 0,
                    "tokens_per_second": 0,
                    "elapsed_time": 0,
                    "num_requests": 0,
                }, f)
            return True
        
        # 执行测试
        print_info("开始测试...")
        start_time = datetime.now()
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 打印输出到控制台（会被 TeeLogger 捕获到全局日志）
        if result.stdout:
            print("\n─── STDOUT ───")
            print(result.stdout)
        if result.stderr:
            print("\n─── STDERR ───")
            print(result.stderr)
        
        # 同时记录到 per-model 日志
        with open(log_file, "a", encoding="utf-8") as f:
            if result.stdout:
                f.write("STDOUT:\n")
                f.write(strip_ansi(result.stdout))
                f.write("\n")
            if result.stderr:
                f.write("STDERR:\n")
                f.write(strip_ansi(result.stderr))
                f.write("\n")
        
        if result.returncode == 0 and result_file.exists():
            print_success(f"测试完成! 耗时: {duration:.1f}s")
            
            # 解析并显示结果
            with open(result_file, "r") as f:
                data = json.load(f)
            
            req_per_s = data.get("requests_per_second", 0)
            tok_per_s = data.get("tokens_per_second", 0)
            elapsed = data.get("elapsed_time", 0)
            num_req = data.get("num_requests", 0)
            
            print()
            print(f"{Colors.GREEN}测试结果:{Colors.NC}")
            print(f"  Requests/s:   {req_per_s:.2f}")
            print(f"  Tokens/s:     {tok_per_s:.2f}")
            print(f"  Total Reqs:   {num_req}")
            print(f"  Elapsed:      {elapsed:.2f}s")
            
            # 计算分析
            if test_mode == "prefill" and params.n_prefill > 0:
                total_prefill_tokens = params.m_prefill * params.n_prefill
                if elapsed > 0:
                    prefill_tps = total_prefill_tokens / elapsed
                    print()
                    print("  [Prefill 分析]")
                    print(f"  Total Prefill Tokens: {total_prefill_tokens}")
                    print(f"  Prefill Tokens/s:     {prefill_tps:.2f}")
            elif test_mode == "decode" and params.n_decode > 0:
                decode_tokens = params.m_decode * params.n_decode
                if elapsed > 0:
                    decode_tps = decode_tokens / elapsed
                    print()
                    print("  [Decode 分析]")
                    print(f"  Total Decode Tokens:  {decode_tokens}")
                    print(f"  Decode Tokens/s:      {decode_tps:.2f}")
            
            return True
        else:
            print_error(f"测试失败: M={m_value} (exit code: {result.returncode})")
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"ERROR: Test failed for M={m_value}\n")
            
            return False
            
    except Exception as e:
        print_error(f"执行异常: {e}")
        return False
    finally:
        restore_env(saved_env)


def generate_model_csv(
    model_name: str,
    m_list: List[int],
    test_mode: str,
    result_json_dir: Path,
    output_dir: Path,
    n_repeat: Optional[int] = None,
):
    """生成单个模型的 CSV 结果"""
    csv_file = output_dir / f"{model_name}_{test_mode}.csv"
    
    print()
    print_subheader(f"生成 CSV: {model_name}")
    
    # CSV 表头
    if test_mode == "prefill":
        header = "M_prefill,prompt_len,max_num_seqs,num_prompts,N_prefill,requests_per_s,tokens_per_s,elapsed_time_s"
    else:
        header = "M_decode,prompt_len,max_num_seqs,num_prompts,N_decode,output_len,requests_per_s,tokens_per_s,elapsed_time_s"
    
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        
        for m_value in m_list:
            result_file = result_json_dir / f"{model_name}_M{m_value}.json"
            params = calculate_test_params(m_value, test_mode, n_repeat)
            
            if result_file.exists():
                try:
                    with open(result_file, "r") as rf:
                        data = json.load(rf)
                    req_s = data.get("requests_per_second", 0)
                    tok_s = data.get("tokens_per_second", 0)
                    elapsed = data.get("elapsed_time", 0)
                except Exception:
                    # JSON 解析失败，标记为 -1
                    req_s = tok_s = elapsed = -1
            else:
                # 测试失败（result_file 不存在），标记为 -1
                req_s = tok_s = elapsed = -1
            
            # 写入 CSV（包括失败的测试）
            if test_mode == "prefill":
                f.write(f"{m_value},{params.prompt_length},{params.max_num_seqs},"
                       f"{params.num_prompts},{params.n_prefill},"
                       f"{req_s:.4f},{tok_s:.4f},{elapsed:.4f}\n")
            else:
                f.write(f"{m_value},{params.prompt_length},{params.max_num_seqs},"
                       f"{params.num_prompts},{params.n_decode},{params.output_len},"
                       f"{req_s:.4f},{tok_s:.4f},{elapsed:.4f}\n")
    
    print_success(f"CSV 保存到: {csv_file}")
    
    # 显示 CSV 预览
    print()
    print("预览:")
    print("-" * 60)
    with open(csv_file, "r") as f:
        print(f.read())
    print("-" * 60)


# ============================================================================
# 高级测试函数
# ============================================================================

def run_stage_benchmark(
    model_key: str,
    backend: str,
    stage: str,
    m_list: List[int],
    checkpoint_path: Path,
    *,
    sparsity: Optional[str] = None,
    n_repeat: Optional[int] = None,
    inner_32: bool = False,
    enforce_eager: bool = False,
    gpu_memory_util: float = GPU_MEMORY_UTILIZATION,
    gpu_id: str = GPU_ID,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    运行单个 (model, backend, sparsity, stage) 组合的所有 M 值测试
    
    Returns:
        (success_count, fail_count)
    """
    global _CURRENT_OUTPUT_DIR
    
    entry = model_registry.get(model_key)
    if entry is None:
        print_error(f"模型不存在: {model_key}")
        return (0, 1)
    
    # 构建输出目录
    output_dir = build_backend_result_dir("throughput_benchmark", stage, backend, entry.quant.upper(), sparsity)
    _CURRENT_OUTPUT_DIR = output_dir
    
    result_json_dir = output_dir / "json"
    result_json_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "benchmark.log"
    
    # 构建标题
    if backend == "cutlass":
        title = f"{entry.local_name} | CUTLASS | {stage}"
    elif backend == "cusparselt" and sparsity:
        title = f"{entry.local_name} | cuSPARSELt ({sparsity}) | {stage}"
    else:
        title = f"{entry.local_name} | cuBLASLt | {stage}"
    
    print_header(title)
    print_info(f"Checkpoint: {checkpoint_path}")
    print_info(f"Output: {output_dir}")
    
    success_count = 0
    fail_count = 0
    
    for i, m_value in enumerate(m_list, 1):
        print()
        print("=" * 60)
        print(f"[{i}/{len(m_list)}] 测试 M={m_value}")
        print("=" * 60)
        
        success = run_single_m_test(
            model_key, m_value, stage, backend,
            result_json_dir, log_file, checkpoint_path,
            sparsity=sparsity,
            n_repeat=n_repeat,
            inner_32=inner_32,
            gpu_memory_util=gpu_memory_util,
            gpu_id=gpu_id,
            enforce_eager=enforce_eager,
            dry_run=dry_run,
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 生成 CSV 结果
    generate_model_csv(
        entry.local_name, m_list, stage,
        result_json_dir, output_dir, n_repeat
    )
    
    print()
    print_info(f"完成: {success_count} 成功, {fail_count} 失败")
    
    return (success_count, fail_count)


def run_full_benchmark(config: BenchmarkConfig) -> Tuple[int, int]:
    """
    运行完整的 Benchmark
    
    遍历: Model → Backend → Sparsity → Stage → M
    
    Returns:
        (total_success, total_fail)
    """
    total_success = 0
    total_fail = 0
    
    for model_key in config.models:
        entry = model_registry.get(model_key)
        if entry is None:
            print_warning(f"模型不存在，跳过: {model_key}")
            total_fail += 1
            continue
        
        # 检查硬件支持
        supported, msg = check_quant_support(entry.quant)
        if not supported:
            print_warning(f"硬件不支持 {entry.quant.upper()}，跳过: {entry.local_name}")
            print_warning(f"  原因: {msg}")
            continue
        
        for backend in config.backends:
            # ========== 预检测: Backend 支持 ==========
            # 在尝试运行测试之前，检查当前 GPU 是否支持该 backend + quant 组合
            backend_supported, backend_reason = check_backend_support(backend, entry.quant)
            if not backend_supported:
                print_warning(f"Backend 不支持，跳过: {entry.local_name} + {backend}")
                print_warning(f"  原因: {backend_reason}")
                continue
            
            if backend == "cusparselt":
                # cuSPARSELt: 遍历所有 sparsity
                for sparsity in config.sparsities:
                    # 获取 sparse checkpoint
                    checkpoint_path = get_checkpoint_path(model_key, backend, sparsity)
                    if checkpoint_path is None:
                        print_warning(f"Sparse checkpoint 不存在，跳过: {entry.local_name} ({sparsity})")
                        continue
                    
                    for stage in config.stages:
                        m_list = config.m_list_prefill if stage == "prefill" else config.m_list_decode
                        
                        success, fail = run_stage_benchmark(
                            model_key, backend, stage, m_list, checkpoint_path,
                            sparsity=sparsity,
                            n_repeat=config.n_repeat,
                            inner_32=config.inner_32,
                            enforce_eager=config.enforce_eager,
                            gpu_memory_util=config.gpu_memory_util,
                            gpu_id=config.gpu_id,
                            dry_run=config.dry_run,
                        )
                        total_success += success
                        total_fail += fail
            else:
                # cutlass / cuBLASLt: 使用 dense checkpoint
                checkpoint_path = get_checkpoint_path(model_key, "cublaslt", None)
                if checkpoint_path is None:
                    print_warning(f"Dense checkpoint 不存在，跳过: {entry.local_name}")
                    continue
                
                for stage in config.stages:
                    m_list = config.m_list_prefill if stage == "prefill" else config.m_list_decode
                    
                    success, fail = run_stage_benchmark(
                        model_key, backend, stage, m_list, checkpoint_path,
                        sparsity=None,
                        n_repeat=config.n_repeat,
                        inner_32=config.inner_32,
                        enforce_eager=config.enforce_eager,
                        gpu_memory_util=config.gpu_memory_util,
                        gpu_id=config.gpu_id,
                        dry_run=config.dry_run,
                    )
                    total_success += success
                    total_fail += fail
    
    return (total_success, total_fail)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse vLLM Throughput Benchmark (重构版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试维度层级:
  Model → Backend → Sparsity → Stage → M

Backend 说明:
  cutlass    - SlideSparse CUTLASS fallback (作为 baseline)
  cublaslt   - SlideSparse cuBLASLt dense GEMM (使用原始 checkpoint)
  cusparselt - SlideSparse cuSPARSELt sparse GEMM (使用预稀疏化 checkpoint)

Sparsity 说明:
  2_4  - 2:4 稀疏 (50% 稀疏率)
  2_6  - 2:6 稀疏 (67% 稀疏率)
  2_8  - 2:8 稀疏 (75% 稀疏率)
  2_10 - 2:10 稀疏 (80% 稀疏率)
  2_12 - 2:12 稀疏 (83% 稀疏率)

示例:
  python3 throughput_benchmark.py                     # 使用默认模型列表
  python3 throughput_benchmark.py --model qwen2.5-0.5b-fp8  # 测试特定模型
  python3 throughput_benchmark.py --model fp8         # 测试所有 FP8 模型
  python3 throughput_benchmark.py --model all         # 测试所有模型
  python3 throughput_benchmark.py --backend cutlass   # 只测试 CUTLASS (baseline)
  python3 throughput_benchmark.py --backend cublaslt  # 只测试 cuBLASLt
  python3 throughput_benchmark.py --backend cusparselt --sparsity 2_8
  python3 throughput_benchmark.py --stage prefill     # 只测试 Prefill
  python3 throughput_benchmark.py --M quick           # 快速测试
  python3 throughput_benchmark.py --dry-run           # Dry-run 验证
"""
    )
    
    # 模型选择
    model_group = parser.add_argument_group("模型选择")
    model_group.add_argument(
        "-m", "--model", type=str, metavar="NAME",
        help="模型选择: 具体模型名 / fp8 / int8 / all (默认: DEFAULT_MODEL_LIST)"
    )
    
    # Backend 选择
    backend_group = parser.add_argument_group("Backend 选择")
    backend_group.add_argument(
        "-b", "--backend", type=str, metavar="NAME",
        help="Backend: cutlass / cublaslt / cusparselt / all (默认: all)"
    )
    backend_group.add_argument(
        "--sparsity", type=str, metavar="LIST",
        help=f"Sparsity 列表 (仅 cusparselt): 逗号分隔 (默认: {','.join(DEFAULT_SPARSITY_LIST)})"
    )
    
    # Stage 选择
    stage_group = parser.add_argument_group("Stage 选择")
    stage_group.add_argument(
        "-s", "--stage", type=str, metavar="NAME",
        help="Stage: prefill / decode / all (默认: all)"
    )
    
    # 参数覆盖
    param_group = parser.add_argument_group("参数覆盖")
    param_group.add_argument(
        "--M", type=str, metavar="LIST",
        help="M 值列表: 逗号分隔 / quick (默认: 按 stage 使用 DEFAULT_M_LIST)"
    )
    param_group.add_argument(
        "--N", type=int, metavar="NUM",
        help="覆盖重复次数"
    )
    param_group.add_argument(
        "--inner-32", action="store_true",
        help="高精度累加 (FP8→FP32, INT8→INT32)"
    )
    
    # 编译选项
    compile_group = parser.add_argument_group("编译选项")
    compile_group.add_argument(
        "--eager", action="store_true",
        help="强制使用 eager mode"
    )
    
    # 硬件选项
    hw_group = parser.add_argument_group("硬件选项")
    hw_group.add_argument(
        "--gpu-id", type=str, default=GPU_ID,
        help=f"GPU ID (默认: {GPU_ID})"
    )
    hw_group.add_argument(
        "--gpu-mem", type=float, default=GPU_MEMORY_UTILIZATION,
        help=f"GPU 内存利用率 (默认: {GPU_MEMORY_UTILIZATION})"
    )
    
    # 其他选项
    other_group = parser.add_argument_group("其他选项")
    other_group.add_argument(
        "--dry-run", action="store_true",
        help="只显示命令不执行"
    )
    other_group.add_argument(
        "--list-models", action="store_true",
        help="列出所有可用模型"
    )
    
    args = parser.parse_args()
    
    # 列出模型
    if args.list_models:
        print_header("可用模型列表")
        for entry in model_registry.list():
            local_path = get_model_local_path(entry.key, CHECKPOINT_DIR)
            status = "✓" if local_path.exists() else "✗"
            print(f"  {status} {entry.key:<25} ({entry.local_name})")
        return 0
    
    # 检查 vllm 是否安装
    if not shutil.which("vllm"):
        print_error("vllm 未安装或不在 PATH 中")
        return 1
    
    # 解析参数
    models = parse_model_list(args.model)
    backends = parse_backend_list(args.backend)
    sparsities = parse_sparsity_list(args.sparsity)
    stages = parse_stage_list(args.stage)
    
    # M 列表 (根据 stage 分别解析)
    m_list_prefill = parse_m_list(args.M, "prefill")
    m_list_decode = parse_m_list(args.M, "decode")
    
    # 确定是否需要 eager mode
    enforce_eager = args.eager
    if not args.eager:
        if not check_triton_support_and_warn():
            print_warning("检测到不支持 torch.compile 的 GPU 架构")
            print_warning("自动启用 eager mode")
            enforce_eager = True
    
    # 构建配置
    config = BenchmarkConfig(
        models=models,
        backends=backends,
        sparsities=sparsities,
        stages=stages,
        m_list_prefill=m_list_prefill,
        m_list_decode=m_list_decode,
        n_repeat=args.N,
        inner_32=args.inner_32,
        enforce_eager=enforce_eager,
        dry_run=args.dry_run,
        gpu_memory_util=args.gpu_mem,
        gpu_id=args.gpu_id,
    )
    
    # 设置信号处理
    _setup_signal_handlers()
    
    # 显示配置信息
    print_header("SlideSparse vLLM Throughput Benchmark")
    print()
    print_hardware_info()
    print()
    
    print("测试配置:")
    print(f"  模型:             {models}")
    print(f"  Backends:         {backends}")
    if "cusparselt" in backends:
        print(f"  Sparsities:       {sparsities}")
    print(f"  Stages:           {stages}")
    print(f"  M_prefill:        {m_list_prefill}")
    print(f"  M_decode:         {m_list_decode}")
    if args.N:
        print(f"  N_repeat:         {args.N}")
    if args.inner_32:
        print(f"  Inner dtype:      FP32/INT32")
    print(f"  GPU 内存利用率:   {args.gpu_mem}")
    if enforce_eager:
        print(f"  编译模式:         Eager")
    if args.dry_run:
        print(f"  模式:             DRY-RUN")
    print()
    print("输出目录结构:")
    print("  throughput_benchmark_results/{stage}/{hw_folder}/{backend}/[{sparsity}/]")
    print("=" * 60)
    
    # 创建日志文件
    result_base = _SCRIPT_DIR / "throughput_benchmark_results"
    result_base.mkdir(parents=True, exist_ok=True)
    log_file = create_log_file(result_base, args)
    global _GLOBAL_LOG_FILE
    _GLOBAL_LOG_FILE = log_file
    
    print_info(f"日志文件: {log_file}")
    print()
    
    # 使用 TeeLogger 同时输出到控制台和日志文件
    with TeeLogger(log_file):
        # 执行测试
        total_success, total_fail = run_full_benchmark(config)
        
        # 显示结果
        print()
        print_header("Benchmark 完成!")
        print()
        print(f"总计: {Colors.GREEN}{total_success} 成功{Colors.NC}, ", end="")
        if total_fail > 0:
            print(f"{Colors.RED}{total_fail} 失败{Colors.NC}")
        else:
            print(f"{total_fail} 失败")
        print("=" * 60)
    
    print()
    print_info(f"日志已保存: {log_file}")
    
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
