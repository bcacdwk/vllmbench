#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse Kernel Benchmark 准备脚本

自动化执行 cuBLASLt/cuSPARSELt kernel-level benchmark 的完整流程。

流水线任务：
============
Task 1: cuBLASLt Model 测试（8个模型，5种精度）
Task 2: cuBLASLt Square 测试（方阵，5种精度）
Task 3: cuSPARSELt Model 高稀疏测试（2_4, 2_6, 2_8, 2_10）
Task 4: cuSPARSELt Square 高稀疏测试（2_4, 2_6, 2_8, 2_10）
Task 5: cuSPARSELt Model 低稀疏测试（2_12, 2_14, 2_16, 2_inf）
Task 6: cuSPARSELt Square 低稀疏测试（2_12, 2_14, 2_16, 2_inf）

M 列表：[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
精度：fp16, bf16, int8, fp8e4m3, fp4e2m1 (all)
8个模型：
    - Llama3.2-1B-INT8, Llama3.2-1B-FP8
    - Llama3.2-3B-INT8, Llama3.2-3B-FP8
    - Qwen2.5-7B-INT8, Qwen2.5-7B-FP8
    - Qwen2.5-14B-INT8, Qwen2.5-14B-FP8

Usage:
    # 执行所有任务
    python3 prepare_for_kernel_bench.py --task 1,1,1,1,1,1
    
    # 只执行 cuBLASLt 测试
    python3 prepare_for_kernel_bench.py --task 1,1,0,0,0,0
    
    # 只执行高稀疏 cuSPARSELt 测试
    python3 prepare_for_kernel_bench.py --task 0,0,1,1,0,0
    
    # 查看任务配置（不执行）
    python3 prepare_for_kernel_bench.py --info
    
    # 在 tmux 中运行
    tmux new -s kernel_bench
    cd /root/vllmbench/slidesparse/benchmark_kernel && python3 prepare_for_kernel_bench.py --task 1,1,1,1,1,1 --gpu 0
    # Ctrl+B 后 按 D 退出保持运行
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
)


# =============================================================================
# 任务配置（写死在此，一般不需要修改）
# =============================================================================

@dataclass
class TaskConfig:
    """任务配置"""
    
    # 通用 M 列表
    M_LIST: List[int] = field(default_factory=lambda: [
        64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536
    ])
    
    # 8个模型（4个 base × 2种量化）
    MODELS: List[str] = field(default_factory=lambda: [
        "Llama3.2-1B-INT8", "Llama3.2-1B-FP8",
        "Llama3.2-3B-INT8", "Llama3.2-3B-FP8",
        "Qwen2.5-7B-INT8", "Qwen2.5-7B-FP8",
        "Qwen2.5-14B-INT8", "Qwen2.5-14B-FP8",
    ])
    
    # 高稀疏度配置（标准 + 中等）
    HIGH_SPARSITY: List[str] = field(default_factory=lambda: [
        "2_4", "2_6", "2_8", "2_10"
    ])
    
    # 低稀疏度配置（深度稀疏）
    LOW_SPARSITY: List[str] = field(default_factory=lambda: [
        "2_12", "2_14", "2_16", "2_inf"
    ])
    
    # 精度
    DTYPE: str = "all"  # fp16, bf16, int8, fp8e4m3, fp4e2m1
    
    # 测试参数
    WARMUP: int = 25
    REPEAT: int = 50


# 全局配置实例
CONFIG = TaskConfig()

# 任务名称
TASK_NAMES = [
    "cuBLASLt Model 测试",
    "cuBLASLt Square 测试",
    "cuSPARSELt Model 高稀疏 (2_4~2_10)",
    "cuSPARSELt Square 高稀疏 (2_4~2_10)",
    "cuSPARSELt Model 低稀疏 (2_12~2_inf)",
    "cuSPARSELt Square 低稀疏 (2_12~2_inf)",
]

# 脚本路径
BENCHMARK_ENTRY = _SCRIPT_DIR / "benchmark_entry.py"


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


def setup_gpu_environment(gpu_id: str = "0", memory_fraction: float = 0.95):
    """
    设置 GPU 环境变量
    
    Args:
        gpu_id: GPU 设备 ID
        memory_fraction: GPU 显存使用比例 (0.0-1.0)，默认 95%
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print_info(f"已设置 CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # 设置 PyTorch CUDA 内存分配策略
    # PYTORCH_CUDA_ALLOC_CONF 可以控制内存分配行为
    # max_split_size_mb 防止内存碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # 设置 GPU 显存使用比例（通过 torch 设置）
    try:
        import torch
        if torch.cuda.is_available():
            # 设置显存使用比例
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print_info(f"已设置 GPU 显存使用比例: {memory_fraction*100:.0f}%")
    except Exception as e:
        print_warning(f"无法设置 GPU 显存比例: {e}")


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
        self.log_dir = log_dir / "kernel_bench_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_log = self.log_dir / f"kernel_bench_{timestamp}.log"
        self.status_file = self.log_dir / f"kernel_bench_{timestamp}_status.json"
        
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._tee_logger = None
        
    def start(self):
        """开始日志记录"""
        with open(self.main_log, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("SlideSparse Kernel Benchmark Log\n")
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
    cwd: Optional[Path] = None,
) -> Tuple[bool, str, float]:
    """
    执行命令并实时输出
    
    Args:
        cmd: 命令及参数列表
        name: 命令描述名称（用于日志）
        cwd: 工作目录
    
    Returns:
        (success, output, duration)
    """
    start_time = time.time()
    
    print_info(f"执行: {' '.join(cmd)}")
    if cwd:
        print_info(f"工作目录: {cwd}")
    print()
    
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        
        assert process.stdout is not None  # 类型检查
        
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
        
    except Exception as e:
        duration = time.time() - start_time
        if process is not None:
            try:
                process.kill()
            except Exception:
                pass
        return False, f"[ERROR] {str(e)}\n{traceback.format_exc()}", duration


# =============================================================================
# 任务执行器
# =============================================================================

class TaskRunner:
    """任务执行器"""
    
    def __init__(self, log_manager: LogManager, global_results_ref: Dict):
        self.log_manager = log_manager
        self.results = global_results_ref
        
    def _build_base_cmd(self, backend: str) -> List[str]:
        """构建基础命令"""
        m_list_str = ",".join(map(str, CONFIG.M_LIST))
        return [
            sys.executable, str(BENCHMARK_ENTRY),
            "--dtype", CONFIG.DTYPE,
            "--warmup", str(CONFIG.WARMUP),
            "--repeat", str(CONFIG.REPEAT),
            "--m_list", m_list_str,
            "--backend", backend,
        ]
    
    def run_task_1_cublaslt_model(self) -> bool:
        """Task 1: cuBLASLt Model 测试"""
        total_success = 0
        total_fail = 0
        
        for model in CONFIG.MODELS:
            print_subheader(f"cuBLASLt Model: {model}")
            
            cmd = self._build_base_cmd("cublaslt")
            cmd.extend(["--model", model])
            
            success, output, duration = run_command(cmd, f"cublaslt {model}")
            
            if success:
                print_success(f"{model} 完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model} 失败")
                total_fail += 1
                
        print()
        print_info(f"cuBLASLt Model 统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_2_cublaslt_square(self) -> bool:
        """Task 2: cuBLASLt Square 测试"""
        print_subheader("cuBLASLt Square 测试")
        
        cmd = self._build_base_cmd("cublaslt")
        cmd.extend(["--model", "square"])
        
        success, output, duration = run_command(cmd, "cublaslt square")
        
        if success:
            print_success(f"Square 测试完成 ({duration:.1f}s)")
        else:
            print_error("Square 测试失败")
            
        return success
    
    def run_task_3_cusparselt_model_high(self) -> bool:
        """Task 3: cuSPARSELt Model 高稀疏测试"""
        total_success = 0
        total_fail = 0
        
        sparsity_str = ",".join(CONFIG.HIGH_SPARSITY)
        
        for model in CONFIG.MODELS:
            print_subheader(f"cuSPARSELt Model 高稀疏: {model}")
            
            cmd = self._build_base_cmd("cusparselt")
            cmd.extend([
                "--model", model,
                "--sparsity", sparsity_str,
            ])
            
            success, output, duration = run_command(cmd, f"cusparselt high {model}")
            
            if success:
                print_success(f"{model} 高稀疏完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model} 高稀疏失败")
                total_fail += 1
                
        print()
        print_info(f"cuSPARSELt Model 高稀疏统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_4_cusparselt_square_high(self) -> bool:
        """Task 4: cuSPARSELt Square 高稀疏测试"""
        print_subheader("cuSPARSELt Square 高稀疏测试")
        
        sparsity_str = ",".join(CONFIG.HIGH_SPARSITY)
        
        cmd = self._build_base_cmd("cusparselt")
        cmd.extend([
            "--model", "square",
            "--sparsity", sparsity_str,
        ])
        
        success, output, duration = run_command(cmd, "cusparselt square high")
        
        if success:
            print_success(f"Square 高稀疏测试完成 ({duration:.1f}s)")
        else:
            print_error("Square 高稀疏测试失败")
            
        return success
    
    def run_task_5_cusparselt_model_low(self) -> bool:
        """Task 5: cuSPARSELt Model 低稀疏测试"""
        total_success = 0
        total_fail = 0
        
        sparsity_str = ",".join(CONFIG.LOW_SPARSITY)
        
        for model in CONFIG.MODELS:
            print_subheader(f"cuSPARSELt Model 低稀疏: {model}")
            
            cmd = self._build_base_cmd("cusparselt")
            cmd.extend([
                "--model", model,
                "--sparsity", sparsity_str,
            ])
            
            success, output, duration = run_command(cmd, f"cusparselt low {model}")
            
            if success:
                print_success(f"{model} 低稀疏完成 ({duration:.1f}s)")
                total_success += 1
            else:
                print_error(f"{model} 低稀疏失败")
                total_fail += 1
                
        print()
        print_info(f"cuSPARSELt Model 低稀疏统计: 成功 {total_success}, 失败 {total_fail}")
        return total_fail == 0
    
    def run_task_6_cusparselt_square_low(self) -> bool:
        """Task 6: cuSPARSELt Square 低稀疏测试"""
        print_subheader("cuSPARSELt Square 低稀疏测试")
        
        sparsity_str = ",".join(CONFIG.LOW_SPARSITY)
        
        cmd = self._build_base_cmd("cusparselt")
        cmd.extend([
            "--model", "square",
            "--sparsity", sparsity_str,
        ])
        
        success, output, duration = run_command(cmd, "cusparselt square low")
        
        if success:
            print_success(f"Square 低稀疏测试完成 ({duration:.1f}s)")
        else:
            print_error("Square 低稀疏测试失败")
            
        return success
    
    def run_all(self, task_mask: List[bool]) -> None:
        """执行所有任务"""
        task_runners = [
            self.run_task_1_cublaslt_model,
            self.run_task_2_cublaslt_square,
            self.run_task_3_cusparselt_model_high,
            self.run_task_4_cusparselt_square_high,
            self.run_task_5_cusparselt_model_low,
            self.run_task_6_cusparselt_square_low,
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
    if len(parts) != 6:
        raise ValueError(f"任务 mask 必须有 6 个值（当前 {len(parts)} 个）: {mask_str}")
    return [int(p.strip()) == 1 for p in parts]


def print_config_info():
    """打印配置信息"""
    print_header("任务配置")
    print()
    
    print(f"{Colors.CYAN}通用配置:{Colors.NC}")
    print(f"  M 列表: {CONFIG.M_LIST}")
    print(f"  精度: {CONFIG.DTYPE}")
    print(f"  Warmup/Repeat: {CONFIG.WARMUP}/{CONFIG.REPEAT}")
    print()
    
    print(f"{Colors.CYAN}模型列表 (8个):{Colors.NC}")
    for model in CONFIG.MODELS:
        print(f"  - {model}")
    print()
    
    print(f"{Colors.CYAN}Task 1: cuBLASLt Model 测试{Colors.NC}")
    print(f"  8个模型 × 5种精度")
    print()
    
    print(f"{Colors.CYAN}Task 2: cuBLASLt Square 测试{Colors.NC}")
    print(f"  方阵测试 (M=N=K)")
    print()
    
    print(f"{Colors.CYAN}Task 3: cuSPARSELt Model 高稀疏{Colors.NC}")
    print(f"  稀疏度: {CONFIG.HIGH_SPARSITY}")
    print()
    
    print(f"{Colors.CYAN}Task 4: cuSPARSELt Square 高稀疏{Colors.NC}")
    print(f"  稀疏度: {CONFIG.HIGH_SPARSITY}")
    print()
    
    print(f"{Colors.CYAN}Task 5: cuSPARSELt Model 低稀疏{Colors.NC}")
    print(f"  稀疏度: {CONFIG.LOW_SPARSITY}")
    print()
    
    print(f"{Colors.CYAN}Task 6: cuSPARSELt Square 低稀疏{Colors.NC}")
    print(f"  稀疏度: {CONFIG.LOW_SPARSITY}")
    print()
    
    # 估算存储需求
    print(f"{Colors.YELLOW}⚠ 内存估算:{Colors.NC}")
    max_m = max(CONFIG.M_LIST)
    # 最大矩阵: max_m × max_m × 2bytes (BF16) × 3 (W, A, R)
    max_mem_gb = (max_m * max_m * 2 * 3) / (1024**3)
    print(f"  最大单矩阵 (M={max_m}): ~{max_m*max_m*2/1024/1024:.1f} MB (BF16)")
    print(f"  最大总内存需求 (W+A+R): ~{max_mem_gb:.1f} GB")
    print()


def main():
    global _GLOBAL_LOG_MANAGER, _GLOBAL_RESULTS
    
    parser = argparse.ArgumentParser(
        description="SlideSparse Kernel Benchmark 准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--task", type=str, default="1,1,1,1,1,1",
        help='任务 mask，格式 "1,1,1,1,1,1" (默认: 全部执行)'
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
    parser.add_argument(
        "--memory-fraction", type=float, default=0.95,
        help="GPU 显存使用比例 (0.0-1.0)，默认: 0.95 (95%)"
    )
    
    args = parser.parse_args()
    
    try:
        task_mask = parse_task_mask(args.task)
    except ValueError as e:
        print_error(str(e))
        return 1
    
    print_header("SlideSparse Kernel Benchmark 准备脚本")
    print()
    print(f"  GPU:     {hw_info.gpu_full_name} ({hw_info.cc_tag})")
    print(f"  Python:  {hw_info.python_tag}")
    print(f"  CUDA:    {hw_info.cuda_tag}")
    print(f"  显存比例: {args.memory_fraction*100:.0f}%")
    print()
    print("  任务列表:")
    for i, (name, enabled) in enumerate(zip(TASK_NAMES, task_mask)):
        status = f"{Colors.GREEN}✓{Colors.NC}" if enabled else f"{Colors.RED}✗{Colors.NC}"
        print(f"    {status} Task {i+1}: {name}")
    print()
    
    if args.info:
        print_config_info()
        return 0
    
    setup_gpu_environment(args.gpu, args.memory_fraction)
    
    if not args.no_protect:
        setup_process_protection()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    log_manager = LogManager(_SCRIPT_DIR)
    _GLOBAL_LOG_MANAGER = log_manager
    log_manager.start()
    
    try:
        runner = TaskRunner(log_manager, _GLOBAL_RESULTS)
        runner.run_all(task_mask)
        results = _GLOBAL_RESULTS
        
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
        
        # 输出结果目录信息
        print()
        print_info("结果保存位置:")
        print(f"  - cuBLASLt:   {_SCRIPT_DIR / 'cuBLASLt' / 'alg_search_results'}")
        print(f"  - cuSPARSELt: {_SCRIPT_DIR / 'cuSPARSELt' / 'alg_search_results'}")
        
        return 0 if fail_count == 0 else 1
        
    finally:
        log_manager.stop()


if __name__ == "__main__":
    sys.exit(main())
