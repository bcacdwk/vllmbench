#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
BitNet 端到端 Benchmark 准备脚本

自动化执行 BitNet 模型从下载到性能测试的完整流程，设计用于无人值守的长时间运行。

流水线任务：
============
Task 1: 基础模型准备（下载 BF16 + 量化为 INT8/FP8）
Task 2: SlideSparse 转换（prune+slide，生成 2_4/2_6/2_8/2_10 共 8 个模型）
Task 3: 离线调优（粗调优 + 细调优）
Task 4: 完整 Prefill Benchmark
Task 5: 完整 Decode Benchmark
Task 6: Kernel 测试 - cuBLASLt
Task 7: Kernel 测试 - cuSPARSELt 高稀疏 (2_4~2_10)
Task 8: Kernel 测试 - cuSPARSELt 低稀疏 (2_12~2_inf)

Usage:
    # 执行所有任务
    python3 prepare_for_bitnet_bench.py --task 1,1,1,1,1,1,1,1
    
    # 只执行模型准备和转换
    python3 prepare_for_bitnet_bench.py --task 1,1,0,0,0,0,0,0
    
    # 跳过下载，只做 benchmark
    python3 prepare_for_bitnet_bench.py --task 0,0,0,0,1,1,0,0
    
    # 只执行 Kernel 测试
    python3 prepare_for_bitnet_bench.py --task 0,0,0,0,0,1,1,1
    
    # 查看任务配置（不执行）
    python3 prepare_for_bitnet_bench.py --info

运行方式：
    # 1. 创建 tmux 会话并进入
    tmux new -s bitnet_bench

    # 2. 在 tmux 里运行脚本
    cd /root/vllmbench/slidesparse/tools && python3 prepare_for_bitnet_bench.py --task 1,1,1,1,1,1,1,1 --gpu 0
    
    # 3. 脚本开始运行后，按 Ctrl+B 后按 D 退出保持运行
    # 或者
    tmux detach

    # 4. 重新连接 tmux 会话
    tmux attach -t bitnet_bench

    # 5. 结束会话
    tmux kill-session -t bitnet_bench
"""

import argparse
import os
import sys
import subprocess
import signal
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# 路径设置
# =============================================================================

_SCRIPT_DIR = Path(__file__).parent
_SLIDESPARSE_ROOT = _SCRIPT_DIR.parent
_PROJECT_ROOT = _SLIDESPARSE_ROOT.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from slidesparse.utils import hw_info
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
)

# Checkpoints 目录
CHECKPOINT_SLIDESPARSE_DIR = _PROJECT_ROOT / "checkpoints_slidesparse"


# =============================================================================
# 任务配置（BitNet 专用）
# =============================================================================

@dataclass
class BitNetTaskConfig:
    """BitNet 任务配置"""
    
    # Task 1: 基础模型准备
    DOWNLOAD_MODEL: str = "bitnet1.58-2b-bf16"  # BF16 源模型
    QUANT_DTYPES: List[str] = field(default_factory=lambda: ["int8", "fp8_e4m3"])
    QUANT_Z: int = 2  # 纯量化参数
    QUANT_L: int = 2  # L=Z 时跳过剪枝，只做量化
    
    # Task 2: SlideSparse 转换
    CONVERT_BASE_MODELS: List[str] = field(default_factory=lambda: [
        "BitNet-2B-INT8", "BitNet-2B-FP8"
    ])
    CONVERT_SPARSITIES: List[Tuple[int, int]] = field(default_factory=lambda: [
        (2, 4), (2, 6), (2, 8), (2, 10)
    ])
    
    # Task 3: 离线调优（删除 65536）
    TUNE_MODEL: str = "BitNet-2B"
    TUNE_COARSE_M_LIST: List[int] = field(default_factory=lambda: [256, 1024, 4096, 16384, 32768])
    TUNE_COARSE_KERNELS: str = "1,0,0,0,1"  # cuBLASLt + Triton quant_only
    TUNE_FINE_M_LIST: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
    ])
    TUNE_FINE_KERNELS: str = "0,1,1,1,0"  # cuSPARSELt + Triton Dequant/QuantSlide
    TUNE_DTYPE: str = "all"
    TUNE_LMAX: int = 10
    TUNE_WARMUP: int = 25
    TUNE_REPEAT: int = 50
    
    # Task 4: Prefill Benchmark
    BENCH_PREFILL_MODELS: List[str] = field(default_factory=lambda: [
        "bitnet1.58-2b-int8", "bitnet1.58-2b-fp8"
    ])
    BENCH_PREFILL_M_LIST: List[int] = field(default_factory=lambda: [
        512, 1024, 2048, 4096, 8192, 16384, 32768
    ])
    BENCH_PREFILL_BACKENDS: str = "cublaslt,cusparselt"  # 无 cutlass
    BENCH_PREFILL_SPARSITIES: List[str] = field(default_factory=lambda: ["2_4", "2_6", "2_8", "2_10"])
    BENCH_PREFILL_STAGES: str = "prefill"
    
    # Task 5: Decode Benchmark
    BENCH_DECODE_MODELS: List[str] = field(default_factory=lambda: [
        "bitnet1.58-2b-int8", "bitnet1.58-2b-fp8"
    ])
    BENCH_DECODE_M_LIST: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    BENCH_DECODE_BACKENDS: str = "cublaslt,cusparselt"
    BENCH_DECODE_SPARSITIES: List[str] = field(default_factory=lambda: ["2_4", "2_6", "2_8", "2_10"])
    BENCH_DECODE_STAGES: str = "decode"
    
    # Task 6/7/8: Kernel Benchmark
    KERNEL_MODEL: str = "BitNet-2B"  # 只测这一个模型
    KERNEL_M_LIST: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384
    ])
    KERNEL_DTYPE: str = "all"
    KERNEL_WARMUP: int = 25
    KERNEL_REPEAT: int = 50
    KERNEL_HIGH_SPARSITY: List[str] = field(default_factory=lambda: ["2_4", "2_6", "2_8", "2_10"])
    KERNEL_LOW_SPARSITY: List[str] = field(default_factory=lambda: ["2_12", "2_14", "2_16", "2_inf"])


# 全局配置实例
CONFIG = BitNetTaskConfig()

# 任务名称
TASK_NAMES = [
    "基础模型准备 (下载 + 量化)",
    "SlideSparse 转换 (prune + slide)",
    "离线调优 (粗调优 + 细调优)",
    "完整 Prefill Benchmark",
    "完整 Decode Benchmark",
    "Kernel: cuBLASLt",
    "Kernel: cuSPARSELt 高稀疏 (2_4~2_10)",
    "Kernel: cuSPARSELt 低稀疏 (2_12~2_inf)",
]

# 脚本路径
SCRIPTS = {
    "download": _SCRIPT_DIR / "model_download.py",
    "convert": _SLIDESPARSE_ROOT / "weight_convert" / "weight_convert_entry.py",
    "tune": _SCRIPT_DIR / "offline_autotune_algsearch.py",
    "benchmark": _SCRIPT_DIR / "throughput_benchmark.py",
    "kernel_bench": _SLIDESPARSE_ROOT / "benchmark_kernel" / "benchmark_entry.py",
}


# =============================================================================
# 进程保护
# =============================================================================

def setup_process_protection():
    """设置进程保护（防止被 OOM Killer 杀掉）"""
    try:
        with open(f'/proc/{os.getpid()}/oom_score_adj', 'w') as f:
            f.write('-1000')
        print_info("已设置 OOM 保护 (oom_score_adj=-1000)")
    except (PermissionError, FileNotFoundError):
        print_warning("无法设置 OOM 保护（需要 root 权限）")


def setup_gpu_environment(gpu_id: str = "0"):
    """设置 GPU 环境变量"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print_info(f"已设置 CUDA_VISIBLE_DEVICES={gpu_id}")


# =============================================================================
# 日志管理
# =============================================================================

class TeeLogger:
    """同时输出到控制台和日志文件的 Logger"""
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.terminal = sys.stdout
        self.log = open(log_file, 'a', encoding='utf-8', buffering=1)
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(strip_ansi(message))
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


class LogManager:
    """日志管理器"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = log_dir / f"bitnet_bench_{timestamp}.log"
        self.status_file = log_dir / f"bitnet_bench_{timestamp}_status.json"
        
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_logger = None
        
    def start(self):
        """开始日志记录"""
        with open(self.main_log, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("BitNet Benchmark Log\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write("Hardware:\n")
            f.write(f"  GPU: {hw_info.gpu_full_name} ({hw_info.cc_tag})\n")
            f.write(f"  Python: {hw_info.python_tag}\n")
            f.write(f"  CUDA: {hw_info.cuda_tag}\n")
            f.write(f"  Arch: {hw_info.arch_tag}\n")
            f.write("\n")
        
        self._tee_logger = TeeLogger(self.main_log)
        sys.stdout = self._tee_logger
        sys.stderr = self._tee_logger
        
        print_info(f"日志文件: {self.main_log}")
        
    def stop(self):
        """停止日志记录"""
        if self._tee_logger:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            self._tee_logger.close()
            
    def save_status(self, status: Dict[str, Any]):
        """保存任务状态"""
        import json
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=2, ensure_ascii=False)
            
    def log_task_start(self, task_id: int, task_name: str):
        """记录任务开始"""
        print()
        print("=" * 70)
        print(f"TASK {task_id + 1}: {task_name}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        print()
        
    def log_task_end(self, task_id: int, task_name: str, success: bool, duration: float):
        """记录任务结束"""
        print()
        print("-" * 70)
        status = f"{Colors.GREEN}SUCCESS{Colors.NC}" if success else f"{Colors.RED}FAILED{Colors.NC}"
        print(f"TASK {task_id + 1}: {task_name} - {status}")
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print("-" * 70)
        print()


# =============================================================================
# 命令执行
# =============================================================================

def run_command(
    cmd: List[str],
    name: str,
    timeout: Optional[int] = None,
    cwd: Optional[Path] = None,
) -> Tuple[bool, str, float]:
    """
    执行命令并捕获输出
    
    Args:
        cmd: 命令列表
        name: 命令名称
        timeout: 超时时间（秒）
        cwd: 工作目录
        
    Returns:
        (success, output, duration)
    """
    start_time = time.time()
    
    print_info(f"执行: {' '.join(cmd)}")
    if cwd:
        print_info(f"工作目录: {cwd}")
    print()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        
        output_lines = []
        while True:
            line = process.stdout.readline()
            if line:
                print(line, end='')
                output_lines.append(line)
            elif process.poll() is not None:
                break
                
        remaining = process.stdout.read()
        if remaining:
            print(remaining, end='')
            output_lines.append(remaining)
            
        duration = time.time() - start_time
        success = process.returncode == 0
        output = ''.join(output_lines)
        
        return success, output, duration
        
    except subprocess.TimeoutExpired:
        process.kill()
        duration = time.time() - start_time
        return False, f"[TIMEOUT] 超过 {timeout} 秒", duration
        
    except Exception as e:
        duration = time.time() - start_time
        return False, f"[ERROR] {str(e)}\n{traceback.format_exc()}", duration


# =============================================================================
# 任务执行器
# =============================================================================

class TaskRunner:
    """任务执行器"""
    
    def __init__(self, log_manager: LogManager, global_results_ref: Dict):
        self.log_manager = log_manager
        self.results = global_results_ref
        
    def run_task_1_prepare_base_models(self) -> bool:
        """Task 1: 基础模型准备（下载 + 量化）"""
        
        # Step 1: 下载 BF16 模型
        print_subheader("Step 1: 下载 BitNet BF16 模型")
        
        bf16_dir = CHECKPOINT_DIR / "BitNet-2B-BF16"
        if bf16_dir.exists() and (bf16_dir / "config.json").exists():
            print_info("BitNet-2B-BF16 已存在，跳过下载")
        else:
            cmd = [
                sys.executable, str(SCRIPTS["download"]),
                "--model", CONFIG.DOWNLOAD_MODEL,
            ]
            success, output, duration = run_command(cmd, "download bitnet bf16")
            if not success:
                print_error("下载 BitNet BF16 失败")
                return False
            print_success(f"下载完成 ({duration:.1f}s)")
        
        # Step 2: 量化为 INT8 和 FP8
        total_success = 0
        total_fail = 0
        
        for output_dtype in CONFIG.QUANT_DTYPES:
            dtype_upper = output_dtype.upper().replace("_E4M3", "").replace("_", "")
            target_name = f"BitNet-2B-{dtype_upper}"
            target_dir = CHECKPOINT_DIR / target_name
            
            print_subheader(f"Step 2: 量化为 {target_name}")
            
            # 检查是否已存在
            if target_dir.exists() and (target_dir / "config.json").exists():
                print_info(f"{target_name} 已存在，跳过量化")
                total_success += 1
                continue
            
            # 使用 weight_convert_entry.py 进行量化
            # --bitnet + Z=2, L=2 会触发纯量化模式（L <= Z 时跳过剪枝）
            # 输出目录会自动生成为 BitNet-2B-BF16-SlideSparse-2_2，需要手动重命名
            temp_output_name = f"BitNet-2B-BF16-SlideSparse-{CONFIG.QUANT_Z}_{CONFIG.QUANT_L}"
            temp_output_dir = CHECKPOINT_SLIDESPARSE_DIR / temp_output_name
            
            cmd = [
                sys.executable, str(SCRIPTS["convert"]),
                "--model", "BitNet-2B-BF16",  # 使用目录名
                "--bitnet",
                "--output-dtype", output_dtype,
                "--Z", str(CONFIG.QUANT_Z),
                "--L", str(CONFIG.QUANT_L),
                "--skip-slide",
            ]
            
            success, output, duration = run_command(
                cmd, f"quant {target_name}",
                cwd=SCRIPTS["convert"].parent
            )
            
            if success:
                # 重命名输出目录到 checkpoints/
                if temp_output_dir.exists():
                    import shutil
                    if target_dir.exists():
                        shutil.rmtree(target_dir)
                    shutil.move(str(temp_output_dir), str(target_dir))
                    print_success(f"{target_name} 量化完成并移动到 {target_dir} ({duration:.1f}s)")
                    total_success += 1
                else:
                    print_error(f"{target_name} 量化输出目录不存在: {temp_output_dir}")
                    total_fail += 1
            else:
                print_error(f"{target_name} 量化失败")
                total_fail += 1
        
        print()
        print_info(f"基础模型准备统计: 成功 {total_success}, 失败 {total_fail}")
        
        # 检查基础模型是否都已生成
        if not self._verify_base_models():
            print_error("基础模型验证失败，停止后续任务")
            return False
        
        return total_fail == 0
    
    def run_task_2_slidesparse_convert(self) -> bool:
        """Task 2: SlideSparse 转换（prune + slide）"""
        
        total_success = 0
        total_fail = 0
        total_skip = 0
        
        for base_model in CONFIG.CONVERT_BASE_MODELS:
            for Z, L in CONFIG.CONVERT_SPARSITIES:
                target_name = f"{base_model}-SlideSparse-{Z}_{L}"
                target_dir = CHECKPOINT_SLIDESPARSE_DIR / target_name
                
                # 检查是否已存在
                if target_dir.exists() and (target_dir / "config.json").exists():
                    print_info(f"跳过已存在: {target_name}")
                    total_skip += 1
                    continue
                
                print_subheader(f"转换: {base_model} -> SlideSparse-{Z}_{L}")
                
                # 直接使用目录名作为 --model 参数
                cmd = [
                    sys.executable, str(SCRIPTS["convert"]),
                    "--model", base_model,  # 直接用目录名，如 BitNet-2B-INT8
                    "--Z", str(Z),
                    "--L", str(L),
                ]
                
                success, output, duration = run_command(
                    cmd, f"convert {target_name}",
                    cwd=SCRIPTS["convert"].parent
                )
                
                if success:
                    print_success(f"{target_name} 转换完成 ({duration:.1f}s)")
                    total_success += 1
                else:
                    print_error(f"{target_name} 转换失败")
                    total_fail += 1
        
        print()
        print_info(f"SlideSparse 转换统计: 成功 {total_success}, 跳过 {total_skip}, 失败 {total_fail}")
        
        # 检查所有 SlideSparse 模型是否都已生成
        if not self._verify_slidesparse_models():
            print_error("SlideSparse 模型验证失败，停止后续任务")
            return False
        
        return total_fail == 0
    
    def _verify_base_models(self) -> bool:
        """验证基础模型是否都已生成"""
        print_subheader("验证基础模型")
        
        all_ok = True
        for base_model in CONFIG.CONVERT_BASE_MODELS:
            model_dir = CHECKPOINT_DIR / base_model
            config_file = model_dir / "config.json"
            
            if model_dir.exists() and config_file.exists():
                print_success(f"  ✓ {base_model}")
            else:
                print_error(f"  ✗ {base_model} 不存在或不完整")
                all_ok = False
        
        return all_ok
    
    def _verify_slidesparse_models(self) -> bool:
        """验证 SlideSparse 模型是否都已生成"""
        print_subheader("验证 SlideSparse 模型")
        
        all_ok = True
        for base_model in CONFIG.CONVERT_BASE_MODELS:
            for Z, L in CONFIG.CONVERT_SPARSITIES:
                target_name = f"{base_model}-SlideSparse-{Z}_{L}"
                target_dir = CHECKPOINT_SLIDESPARSE_DIR / target_name
                config_file = target_dir / "config.json"
                
                if target_dir.exists() and config_file.exists():
                    print_success(f"  ✓ {target_name}")
                else:
                    print_error(f"  ✗ {target_name} 不存在或不完整")
                    all_ok = False
        
        return all_ok
    
    def run_task_3_offline_tune(self) -> bool:
        """Task 3: 离线调优（粗调优 + 细调优）"""
        
        # 粗调优
        print_subheader("粗调优: cuBLASLt + Triton quant_only")
        
        m_list_str = ",".join(map(str, CONFIG.TUNE_COARSE_M_LIST))
        cmd = [
            sys.executable, str(SCRIPTS["tune"]),
            "--model", CONFIG.TUNE_MODEL,
            "--dtype", CONFIG.TUNE_DTYPE,
            "--m_list", m_list_str,
            "--Lmax", str(CONFIG.TUNE_LMAX),
            "--warmup", str(CONFIG.TUNE_WARMUP),
            "--repeat", str(CONFIG.TUNE_REPEAT),
            "--kernels", CONFIG.TUNE_COARSE_KERNELS,
        ]
        
        success_coarse, output, duration = run_command(cmd, "coarse tune")
        if success_coarse:
            print_success(f"粗调优完成 ({duration:.1f}s)")
        else:
            print_error("粗调优失败")
        
        # 细调优
        print_subheader("细调优: cuSPARSELt + Triton Dequant/QuantSlide")
        
        m_list_str = ",".join(map(str, CONFIG.TUNE_FINE_M_LIST))
        cmd = [
            sys.executable, str(SCRIPTS["tune"]),
            "--model", CONFIG.TUNE_MODEL,
            "--dtype", CONFIG.TUNE_DTYPE,
            "--m_list", m_list_str,
            "--Lmax", str(CONFIG.TUNE_LMAX),
            "--warmup", str(CONFIG.TUNE_WARMUP),
            "--repeat", str(CONFIG.TUNE_REPEAT),
            "--kernels", CONFIG.TUNE_FINE_KERNELS,
        ]
        
        success_fine, output, duration = run_command(cmd, "fine tune")
        if success_fine:
            print_success(f"细调优完成 ({duration:.1f}s)")
        else:
            print_error("细调优失败")
        
        return success_coarse and success_fine
    
    def run_task_4_bench_prefill(self) -> bool:
        """Task 4: 完整 Prefill Benchmark"""
        
        total_success = 0
        total_fail = 0
        
        m_list_str = ",".join(map(str, CONFIG.BENCH_PREFILL_M_LIST))
        
        for model_key in CONFIG.BENCH_PREFILL_MODELS:
            print_subheader(f"Prefill Benchmark: {model_key}")
            
            cmd = [
                sys.executable, str(SCRIPTS["benchmark"]),
                "--model", model_key,
                "--backend", CONFIG.BENCH_PREFILL_BACKENDS,
                "--stage", CONFIG.BENCH_PREFILL_STAGES,
                "--sparsity", ",".join(CONFIG.BENCH_PREFILL_SPARSITIES),
                "--M", m_list_str,
            ]
            
            success, output, duration = run_command(cmd, f"prefill {model_key}")
            
            if success:
                print_success(f"{model_key} Prefill 完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model_key} Prefill 失败")
                total_fail += 1
        
        print()
        print_info(f"Prefill 统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_5_bench_decode(self) -> bool:
        """Task 5: 完整 Decode Benchmark"""
        
        total_success = 0
        total_fail = 0
        
        m_list_str = ",".join(map(str, CONFIG.BENCH_DECODE_M_LIST))
        
        for model_key in CONFIG.BENCH_DECODE_MODELS:
            print_subheader(f"Decode Benchmark: {model_key}")
            
            cmd = [
                sys.executable, str(SCRIPTS["benchmark"]),
                "--model", model_key,
                "--backend", CONFIG.BENCH_DECODE_BACKENDS,
                "--stage", CONFIG.BENCH_DECODE_STAGES,
                "--sparsity", ",".join(CONFIG.BENCH_DECODE_SPARSITIES),
                "--M", m_list_str,
            ]
            
            success, output, duration = run_command(cmd, f"decode {model_key}")
            
            if success:
                print_success(f"{model_key} Decode 完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model_key} Decode 失败")
                total_fail += 1
        
        print()
        print_info(f"Decode 统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def _build_kernel_base_cmd(self, backend: str) -> List[str]:
        """构建 Kernel benchmark 基础命令"""
        m_list_str = ",".join(map(str, CONFIG.KERNEL_M_LIST))
        return [
            sys.executable, str(SCRIPTS["kernel_bench"]),
            "--dtype", CONFIG.KERNEL_DTYPE,
            "--warmup", str(CONFIG.KERNEL_WARMUP),
            "--repeat", str(CONFIG.KERNEL_REPEAT),
            "--m_list", m_list_str,
            "--backend", backend,
            "--model", CONFIG.KERNEL_MODEL,
        ]
    
    def run_task_6_kernel_cublaslt(self) -> bool:
        """Task 6: Kernel - cuBLASLt"""
        
        print_subheader(f"cuBLASLt Kernel: {CONFIG.KERNEL_MODEL}")
        
        cmd = self._build_kernel_base_cmd("cublaslt")
        
        success, output, duration = run_command(cmd, "cublaslt kernel")
        
        if success:
            print_success(f"cuBLASLt Kernel 测试完成 ({duration:.1f}s)")
        else:
            print_error("cuBLASLt Kernel 测试失败")
        
        return success
    
    def run_task_7_kernel_cusparselt_high(self) -> bool:
        """Task 7: Kernel - cuSPARSELt 高稀疏"""
        
        print_subheader(f"cuSPARSELt 高稀疏 Kernel: {CONFIG.KERNEL_MODEL}")
        
        sparsity_str = ",".join(CONFIG.KERNEL_HIGH_SPARSITY)
        
        cmd = self._build_kernel_base_cmd("cusparselt")
        cmd.extend(["--sparsity", sparsity_str])
        
        success, output, duration = run_command(cmd, "cusparselt high kernel")
        
        if success:
            print_success(f"cuSPARSELt 高稀疏 Kernel 测试完成 ({duration:.1f}s)")
        else:
            print_error("cuSPARSELt 高稀疏 Kernel 测试失败")
        
        return success
    
    def run_task_8_kernel_cusparselt_low(self) -> bool:
        """Task 8: Kernel - cuSPARSELt 低稀疏"""
        
        print_subheader(f"cuSPARSELt 低稀疏 Kernel: {CONFIG.KERNEL_MODEL}")
        
        sparsity_str = ",".join(CONFIG.KERNEL_LOW_SPARSITY)
        
        cmd = self._build_kernel_base_cmd("cusparselt")
        cmd.extend(["--sparsity", sparsity_str])
        
        success, output, duration = run_command(cmd, "cusparselt low kernel")
        
        if success:
            print_success(f"cuSPARSELt 低稀疏 Kernel 测试完成 ({duration:.1f}s)")
        else:
            print_error("cuSPARSELt 低稀疏 Kernel 测试失败")
        
        return success
    
    def run_all(self, task_mask: List[bool]) -> None:
        """执行所有任务"""
        task_runners = [
            self.run_task_1_prepare_base_models,
            self.run_task_2_slidesparse_convert,
            self.run_task_3_offline_tune,
            self.run_task_4_bench_prefill,
            self.run_task_5_bench_decode,
            self.run_task_6_kernel_cublaslt,
            self.run_task_7_kernel_cusparselt_high,
            self.run_task_8_kernel_cusparselt_low,
        ]
        
        for i, (enabled, runner, name) in enumerate(zip(task_mask, task_runners, TASK_NAMES)):
            if not enabled:
                print_info(f"跳过 Task {i+1}: {name}")
                self.results[i] = {"skipped": True}
                continue
                
            self.log_manager.log_task_start(i, name)
            
            start_time = time.time()
            try:
                success = runner()
            except Exception as e:
                print_error(f"Task {i+1} 异常: {e}")
                traceback.print_exc()
                success = False
                
            duration = time.time() - start_time
            
            self.results[i] = {
                "success": success,
                "duration": duration,
                "skipped": False,
            }
            
            self.log_manager.log_task_end(i, name, success, duration)
            
            # 保存状态（方便中途查看进度）
            self.log_manager.save_status({
                "current_task": i,
                "results": self.results,
                "last_update": datetime.now().isoformat(),
            })


# =============================================================================
# 信号处理
# =============================================================================

_GLOBAL_LOG_MANAGER: Optional[LogManager] = None
_GLOBAL_RESULTS: Dict = {}


def signal_handler(signum, frame):
    """处理中断信号"""
    print()
    print("=" * 70)
    print(f"{Colors.YELLOW}收到中断信号 (signal {signum}){Colors.NC}")
    print("=" * 70)
    
    if _GLOBAL_LOG_MANAGER:
        _GLOBAL_LOG_MANAGER.save_status({
            "interrupted": True,
            "signal": signum,
            "results": _GLOBAL_RESULTS,
            "timestamp": datetime.now().isoformat(),
        })
        print_info(f"状态已保存: {_GLOBAL_LOG_MANAGER.status_file}")
        _GLOBAL_LOG_MANAGER.stop()
        
    sys.exit(130)


# =============================================================================
# 主函数
# =============================================================================

def parse_task_mask(mask_str: str) -> List[bool]:
    """解析任务 mask 字符串"""
    parts = mask_str.split(",")
    if len(parts) != 8:
        raise ValueError(f"任务 mask 必须有 8 个值（当前 {len(parts)} 个）: {mask_str}")
    return [int(p.strip()) == 1 for p in parts]


def print_config_info():
    """打印配置信息"""
    print_header("任务配置")
    print()
    
    print(f"{Colors.CYAN}Task 1: 基础模型准备{Colors.NC}")
    print(f"  下载: {CONFIG.DOWNLOAD_MODEL}")
    print(f"  量化类型: {CONFIG.QUANT_DTYPES}")
    print(f"  纯量化参数: Z={CONFIG.QUANT_Z}, L={CONFIG.QUANT_L}")
    print()
    
    print(f"{Colors.CYAN}Task 2: SlideSparse 转换{Colors.NC}")
    print(f"  基础模型: {CONFIG.CONVERT_BASE_MODELS}")
    print(f"  稀疏度: {CONFIG.CONVERT_SPARSITIES}")
    print()
    
    print(f"{Colors.CYAN}Task 3: 离线调优{Colors.NC}")
    print(f"  模型: {CONFIG.TUNE_MODEL}")
    print(f"  粗调优 M 列表: {CONFIG.TUNE_COARSE_M_LIST}")
    print(f"  粗调优 Kernels: {CONFIG.TUNE_COARSE_KERNELS}")
    print(f"  细调优 M 列表: {CONFIG.TUNE_FINE_M_LIST}")
    print(f"  细调优 Kernels: {CONFIG.TUNE_FINE_KERNELS}")
    print()
    
    print(f"{Colors.CYAN}Task 4: Prefill Benchmark{Colors.NC}")
    print(f"  模型: {CONFIG.BENCH_PREFILL_MODELS}")
    print(f"  M 列表: {CONFIG.BENCH_PREFILL_M_LIST}")
    print(f"  Backend: {CONFIG.BENCH_PREFILL_BACKENDS}")
    print(f"  稀疏度: {CONFIG.BENCH_PREFILL_SPARSITIES}")
    print()
    
    print(f"{Colors.CYAN}Task 5: Decode Benchmark{Colors.NC}")
    print(f"  模型: {CONFIG.BENCH_DECODE_MODELS}")
    print(f"  M 列表: {CONFIG.BENCH_DECODE_M_LIST}")
    print(f"  Backend: {CONFIG.BENCH_DECODE_BACKENDS}")
    print(f"  稀疏度: {CONFIG.BENCH_DECODE_SPARSITIES}")
    print()
    
    print(f"{Colors.CYAN}Task 6/7/8: Kernel Benchmark{Colors.NC}")
    print(f"  模型: {CONFIG.KERNEL_MODEL}")
    print(f"  M 列表: {CONFIG.KERNEL_M_LIST}")
    print(f"  高稀疏: {CONFIG.KERNEL_HIGH_SPARSITY}")
    print(f"  低稀疏: {CONFIG.KERNEL_LOW_SPARSITY}")
    print()


def main():
    global _GLOBAL_LOG_MANAGER, _GLOBAL_RESULTS
    
    parser = argparse.ArgumentParser(
        description="BitNet 端到端 Benchmark 准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--task", type=str, default="1,1,1,1,1,1,1,1",
        help='任务 mask，格式 "1,1,1,1,1,1,1,1" (默认: 全部执行)'
    )
    parser.add_argument(
        "--gpu", type=str, default="0",
        help="使用的 GPU ID (默认: 0)"
    )
    parser.add_argument(
        "--info", action="store_true",
        help="仅显示配置信息，不执行"
    )
    parser.add_argument(
        "--no-protect", action="store_true",
        help="禁用进程保护"
    )
    
    args = parser.parse_args()
    
    # 解析任务 mask
    try:
        task_mask = parse_task_mask(args.task)
    except ValueError as e:
        print_error(str(e))
        return 1
    
    # 显示配置信息
    print_header("BitNet 端到端 Benchmark 准备脚本")
    print()
    print(f"  GPU:     {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"  Python:  {hw_info.python_tag}")
    print(f"  CUDA:    {hw_info.cuda_tag}")
    print()
    print("  任务列表:")
    for i, (name, enabled) in enumerate(zip(TASK_NAMES, task_mask)):
        status = f"{Colors.GREEN}✓{Colors.NC}" if enabled else f"{Colors.RED}✗{Colors.NC}"
        print(f"    {status} Task {i+1}: {name}")
    print()
    
    if args.info:
        print_config_info()
        return 0
    
    # 设置 GPU 环境
    setup_gpu_environment(args.gpu)
    
    # 设置进程保护
    if not args.no_protect:
        setup_process_protection()
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 创建日志管理器
    log_manager = LogManager(_SCRIPT_DIR)
    _GLOBAL_LOG_MANAGER = log_manager
    log_manager.start()
    
    try:
        # 执行任务
        runner = TaskRunner(log_manager, _GLOBAL_RESULTS)
        runner.run_all(task_mask)
        results = _GLOBAL_RESULTS
        
        # 打印最终总结
        print()
        print_header("最终总结")
        print()
        
        success_count = 0
        fail_count = 0
        skip_count = 0
        total_duration = 0
        
        for i, (name, enabled) in enumerate(zip(TASK_NAMES, task_mask)):
            r = results.get(i, {})
            
            if r.get("skipped"):
                print(f"  Task {i+1}: {name} - {Colors.YELLOW}SKIPPED{Colors.NC}")
                skip_count += 1
            elif r.get("success"):
                d = r.get("duration", 0)
                total_duration += d
                print(f"  Task {i+1}: {name} - {Colors.GREEN}SUCCESS{Colors.NC} ({d:.1f}s)")
                success_count += 1
            else:
                d = r.get("duration", 0)
                total_duration += d
                print(f"  Task {i+1}: {name} - {Colors.RED}FAILED{Colors.NC} ({d:.1f}s)")
                fail_count += 1
                
        print()
        print(f"  总计: {Colors.GREEN}{success_count} 成功{Colors.NC}, ", end="")
        if fail_count > 0:
            print(f"{Colors.RED}{fail_count} 失败{Colors.NC}, ", end="")
        else:
            print(f"{fail_count} 失败, ", end="")
        print(f"{skip_count} 跳过")
        print(f"  总耗时: {total_duration:.1f} 秒 ({total_duration/3600:.2f} 小时)")
        print()
        
        # 保存最终状态
        log_manager.save_status({
            "completed": True,
            "results": results,
            "summary": {
                "success": success_count,
                "failed": fail_count,
                "skipped": skip_count,
                "total_duration": total_duration,
            },
            "timestamp": datetime.now().isoformat(),
        })
        
        print_info(f"日志文件: {log_manager.main_log}")
        print_info(f"状态文件: {log_manager.status_file}")
        
        return 0 if fail_count == 0 else 1
        
    finally:
        log_manager.stop()


if __name__ == "__main__":
    sys.exit(main())
