#!/usr/bin/env python3
"""
重试失败的 Throughput Benchmark 测试

特点:
1. 每个测试独立进程执行，完全隔离
2. 测试完成后强制 kill 所有相关进程，彻底释放显存
3. 测试之间等待一段时间，确保资源完全释放
4. 支持 dry-run 模式预览命令

用法:
    python3 retry_failed_benchmarks.py              # 执行所有失败的测试
    python3 retry_failed_benchmarks.py --dry-run    # 预览命令不执行
    python3 retry_failed_benchmarks.py --wait 10 --gpu-id 1    # 测试间等待 10 秒
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

# ============================================================================
# 配置
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
BENCHMARK_SCRIPT = SCRIPT_DIR / "throughput_benchmark.py"
LOG_DIR = SCRIPT_DIR / "retry_logs"

# 测试之间等待时间（秒）
DEFAULT_WAIT_TIME = 5

# 默认 GPU ID
DEFAULT_GPU_ID = "1"

# ============================================================================
# GPU 内存利用率配置（根据 M 值和阶段动态调整）
# ============================================================================

def get_optimal_gpu_mem_util(model: str, stage: str, m_value: int) -> float:
    """
    根据模型、阶段和 M 值返回最优的 gpu_memory_utilization
    
    原理：
    - gpu_mem_util 控制 KV Cache 能用多少显存
    - 越大的 M 需要越多的 Activation 空间，因此需要降低 gpu_mem_util
    - Decode 阶段 M 值小，可以给更多空间给 KV Cache
    - torch.compile autotuning 需要额外 2-5GB 临时空间
    
    Qwen2.5-7B 特殊：模型权重 ~8GB，只剩 ~8GB 给其他用途
    """
    
    # Decode 阶段：M 值小（64-512），Activation 开销小
    if stage == "decode":
        if m_value <= 256:
            return 0.85
        elif m_value <= 512:
            return 0.80
        else:
            return 0.75
    
    # Prefill 阶段：根据 M 值调整
    # 对于 Qwen 7B，需要更保守
    is_qwen_7b = "qwen2.5-7b" in model.lower()
    
    if is_qwen_7b:
        # Qwen 7B 特殊处理：模型大，显存紧张
        if m_value <= 4096:
            return 0.80
        elif m_value <= 8192:
            return 0.70
        elif m_value <= 16384:
            return 0.60
        elif m_value <= 32768:
            return 0.50  # 极限尝试
        else:  # 65536
            return 0.45  # 非常激进，可能还是 OOM
    else:
        # Llama 1B/3B：模型小，显存充裕
        if m_value <= 16384:
            return 0.85
        elif m_value <= 32768:
            return 0.80
        else:  # 65536
            return 0.70

# ============================================================================
# 失败测试列表 (更新于 2026-01-26，已移除成功的测试)
# ============================================================================

# 格式: (model, stage, backend, sparsity, m_value)
# 每个条目只测一个 M 值，细粒度控制
# sparsity 为 None 表示 cuBLASLt

# ========== 已成功的测试 (1-13) - 已移除 ==========
# ✅ llama3.2-1b-int8 | cusparselt (2_4, 2_6) | prefill | M=65536
# ✅ llama3.2-1b-fp8 | cusparselt (2_4, 2_6, 2_8, 2_10) | prefill | M=65536
# ✅ llama3.2-3b-int8 | cusparselt (2_4, 2_6, 2_8, 2_10) | prefill | M=65536
# ✅ llama3.2-3b-fp8 | cusparselt (2_4, 2_8, 2_10) | prefill | M=65536

FAILED_TESTS = [
    # ========== Qwen2.5-7B-INT8 Prefill (细粒度，每个 M 单独测) ==========
    
    # cuBLASLt
    ("qwen2.5-7b-int8", "prefill", "cublaslt", None, 32768),
    ("qwen2.5-7b-int8", "prefill", "cublaslt", None, 65536),
    
    # cuSPARSELt 2_4 (16384 已成功，测 32768 和 65536)
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_4", 32768),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_4", 65536),
    
    # cuSPARSELt 2_6
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_6", 8192),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_6", 16384),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_6", 32768),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_6", 65536),
    
    # cuSPARSELt 2_8
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_8", 8192),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_8", 16384),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_8", 32768),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_8", 65536),
    
    # cuSPARSELt 2_10
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_10", 8192),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_10", 16384),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_10", 32768),
    ("qwen2.5-7b-int8", "prefill", "cusparselt", "2_10", 65536),
    
    # ========== Qwen2.5-7B-FP8 Prefill ==========
    
    # cuBLASLt
    ("qwen2.5-7b-fp8", "prefill", "cublaslt", None, 32768),
    ("qwen2.5-7b-fp8", "prefill", "cublaslt", None, 65536),
    
    # cuSPARSELt 2_4
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_4", 65536),
    
    # cuSPARSELt 2_6
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_6", 32768),
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_6", 65536),
    
    # cuSPARSELt 2_8
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_8", 32768),
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_8", 65536),
    
    # cuSPARSELt 2_10
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_10", 32768),
    ("qwen2.5-7b-fp8", "prefill", "cusparselt", "2_10", 65536),
    
    # ========== Qwen2.5-7B-INT8 Decode (细粒度) ==========
    ("qwen2.5-7b-int8", "decode", "cusparselt", "2_6", 512),
    ("qwen2.5-7b-int8", "decode", "cusparselt", "2_8", 128),
    ("qwen2.5-7b-int8", "decode", "cusparselt", "2_8", 256),
    ("qwen2.5-7b-int8", "decode", "cusparselt", "2_10", 128),
    ("qwen2.5-7b-int8", "decode", "cusparselt", "2_10", 512),
]

# ============================================================================
# 颜色输出
# ============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(msg):
    print(f"\n{Colors.CYAN}{'='*70}")
    print(f" {msg}")
    print(f"{'='*70}{Colors.NC}\n")

def print_info(msg):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def print_success(msg):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def print_error(msg):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")

# ============================================================================
# 进程管理
# ============================================================================

def kill_vllm_processes():
    """强制 kill 所有 vllm 相关进程"""
    try:
        # Kill vllm 进程
        subprocess.run(
            ["pkill", "-9", "-f", "vllm"],
            capture_output=True,
            timeout=5
        )
    except Exception:
        pass
    
    try:
        # Kill python 进程中包含 benchmark 的
        subprocess.run(
            ["pkill", "-9", "-f", "throughput_benchmark"],
            capture_output=True,
            timeout=5
        )
    except Exception:
        pass
    
    # 清理 CUDA 缓存
    try:
        subprocess.run(
            ["python3", "-c", "import torch; torch.cuda.empty_cache()"],
            capture_output=True,
            timeout=10
        )
    except Exception:
        pass

def wait_for_gpu_release(seconds: int):
    """等待 GPU 资源释放"""
    print_info(f"等待 {seconds} 秒让 GPU 资源完全释放...")
    for i in range(seconds, 0, -1):
        print(f"\r  剩余 {i} 秒...", end="", flush=True)
        time.sleep(1)
    print("\r  完成!          ")

# ============================================================================
# 测试执行
# ============================================================================

# 重试配置
MAX_RETRIES = 3          # 最大重试次数
GPU_MEM_STEP = 0.05      # 每次重试降低的 gpu_mem_util
MIN_GPU_MEM = 0.30       # 最低 gpu_mem_util（低于此值放弃）

def build_command_with_gpu_mem(
    model: str, stage: str, backend: str, sparsity: str | None, 
    m_value: int, gpu_mem_util: float, gpu_id: str = DEFAULT_GPU_ID
) -> list[str]:
    """构建测试命令（指定 gpu_mem_util）"""
    cmd = [
        "python3", str(BENCHMARK_SCRIPT),
        "--model", model,
        "--stage", stage,
        "--backend", backend,
        "--M", str(m_value),
        "--gpu-mem", str(gpu_mem_util),
        "--gpu-id", gpu_id,
    ]
    
    if sparsity:
        cmd.extend(["--sparsity", sparsity])
    
    return cmd

def is_oom_error(stdout: str, stderr: str) -> bool:
    """检查是否是 OOM 错误"""
    combined = stdout + stderr
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "torch.OutOfMemoryError",
        "CUDA error: out of memory",
    ]
    return any(pattern in combined for pattern in oom_patterns)

def is_kv_cache_error(stdout: str, stderr: str) -> bool:
    """检查是否是 KV Cache 空间不足错误（降低 gpu_mem 无用）"""
    combined = stdout + stderr
    kv_patterns = [
        "Not enough memory to allocate KV cache",
        "Cannot allocate KV cache",
        "insufficient KV cache",
        "KV cache blocks",
    ]
    return any(pattern.lower() in combined.lower() for pattern in kv_patterns)

def run_single_attempt(
    cmd: list[str],
    gpu_mem_util: float,
    attempt: int = 1,
) -> tuple[bool, str, str, float]:
    """
    运行单次测试尝试
    
    Returns:
        (success, stdout, stderr, duration)
    """
    print_info(f"尝试 {attempt}: gpu_mem={gpu_mem_util:.2f}")
    print_info(f"命令: {' '.join(cmd)}")
    print()
    
    start_time = datetime.now()
    
    process = None
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        stdout, stderr = process.communicate(timeout=600)  # 10 分钟超时
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        return (process.returncode == 0, stdout, stderr, duration)
        
    except subprocess.TimeoutExpired:
        # 超时后强制终止进程
        if process:
            process.kill()
            process.wait()  # 等待进程完全退出
        return (False, "", "测试超时 (>10分钟)", 600.0)
    except Exception as e:
        if process:
            process.kill()
            process.wait()
        return (False, "", str(e), 0.0)

def run_single_test(
    model: str,
    stage: str,
    backend: str,
    sparsity: str | None,
    m_value: int,
    test_num: int,
    total_tests: int,
    dry_run: bool = False,
    log_file: Path | None = None,
    gpu_id: str = DEFAULT_GPU_ID,
) -> bool:
    """运行单个测试（带重试机制）"""
    
    # 构建描述
    if sparsity:
        desc = f"{model} | {backend} ({sparsity}) | {stage} | M={m_value}"
    else:
        desc = f"{model} | {backend} | {stage} | M={m_value}"
    
    print_header(f"[{test_num}/{total_tests}] {desc}")
    
    # 获取初始 gpu_mem_util
    initial_gpu_mem = get_optimal_gpu_mem_util(model, stage, m_value)
    
    if dry_run:
        cmd = build_command_with_gpu_mem(model, stage, backend, sparsity, m_value, initial_gpu_mem, gpu_id)
        print_info("[DRY-RUN] 命令:")
        print(f"  {' '.join(cmd)}")
        print_info(f"[DRY-RUN] 如果失败，将以 {GPU_MEM_STEP:.0%} 步长重试，最多 {MAX_RETRIES} 次")
        return True
    
    # 重试循环
    current_gpu_mem = initial_gpu_mem
    
    for attempt in range(1, MAX_RETRIES + 1):
        # 构建命令
        cmd = build_command_with_gpu_mem(model, stage, backend, sparsity, m_value, current_gpu_mem, gpu_id)
        
        # 执行测试
        success, stdout, stderr, duration = run_single_attempt(cmd, current_gpu_mem, attempt)
        
        # 写入日志
        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\n")
                f.write(f"Test: {desc}\n")
                f.write(f"Attempt: {attempt}/{MAX_RETRIES}\n")
                f.write(f"GPU Mem Util: {current_gpu_mem:.2f}\n")
                f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duration: {duration:.1f}s\n")
                f.write(f"Success: {success}\n")
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"\nSTDOUT:\n{stdout}\n")
                if stderr:
                    f.write(f"\nSTDERR:\n{stderr}\n")
        
        # 检查结果
        if success and not is_oom_error(stdout, stderr):
            print_success(f"测试通过! 尝试 {attempt}, gpu_mem={current_gpu_mem:.2f}, 耗时: {duration:.1f}s")
            return True
        
        # 失败处理
        if is_oom_error(stdout, stderr):
            print_error(f"尝试 {attempt} OOM! gpu_mem={current_gpu_mem:.2f}, 耗时: {duration:.1f}s")
            
            # 检查是否是 KV Cache 不足（降低 gpu_mem 无用）
            if is_kv_cache_error(stdout, stderr):
                print_warning("检测到 KV Cache 空间不足，继续降低 gpu_mem 无效，放弃重试")
                break
            
            # 准备下一次尝试
            if attempt < MAX_RETRIES:
                next_gpu_mem = current_gpu_mem - GPU_MEM_STEP
                
                if next_gpu_mem < MIN_GPU_MEM:
                    print_warning(f"gpu_mem 已达下限 ({MIN_GPU_MEM:.2f})，放弃重试")
                    break
                
                print_info(f"将降低 gpu_mem: {current_gpu_mem:.2f} → {next_gpu_mem:.2f}")
                current_gpu_mem = next_gpu_mem
                
                # 清理并等待
                kill_vllm_processes()
                wait_for_gpu_release(5)
        else:
            # 非 OOM 错误，不重试
            print_error(f"尝试 {attempt} 失败 (非 OOM)! 耗时: {duration:.1f}s")
            if stderr:
                stderr_lines = stderr.strip().split('\n')
                print_error("错误信息 (最后 5 行):")
                for line in stderr_lines[-5:]:
                    print(f"  {line}")
            break
    
    print_error(f"测试最终失败: {desc}")
    return False

# ============================================================================
# 主函数
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="重试失败的 Throughput Benchmark 测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只显示命令不执行"
    )
    parser.add_argument(
        "--wait", type=int, default=DEFAULT_WAIT_TIME,
        help=f"测试之间等待时间（秒）(默认: {DEFAULT_WAIT_TIME})"
    )
    parser.add_argument(
        "--start", type=int, default=1,
        help="从第几个测试开始 (默认: 1)"
    )
    parser.add_argument(
        "--only-prefill", action="store_true",
        help="只重试 Prefill 测试"
    )
    parser.add_argument(
        "--only-decode", action="store_true",
        help="只重试 Decode 测试"
    )
    parser.add_argument(
        "--only-model", type=str,
        help="只重试指定模型 (如: qwen2.5-7b-int8)"
    )
    parser.add_argument(
        "--gpu-id", type=str, default=DEFAULT_GPU_ID,
        help=f"GPU ID (默认: {DEFAULT_GPU_ID})"
    )
    
    args = parser.parse_args()
    
    # 过滤测试列表
    tests = FAILED_TESTS.copy()
    
    if args.only_prefill:
        tests = [t for t in tests if t[1] == "prefill"]
    elif args.only_decode:
        tests = [t for t in tests if t[1] == "decode"]
    
    if args.only_model:
        tests = [t for t in tests if t[0] == args.only_model.lower()]
    
    if not tests:
        print_warning("没有符合条件的测试!")
        return 0
    
    # 创建日志目录
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"retry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    print_header("SlideSparse Benchmark 重试脚本")
    print_info(f"总测试数: {len(tests)}")
    print_info(f"测试间等待: {args.wait} 秒")
    print_info(f"起始测试: {args.start}")
    print_info(f"GPU ID: {args.gpu_id}")
    if not args.dry_run:
        print_info(f"日志文件: {log_file}")
    print()
    
    # 显示测试列表
    print("将执行以下测试:")
    print("-" * 80)
    print(f"  {'#':<4} {'Model':<18} {'Backend':<18} {'Stage':<8} {'M':<8} {'gpu_mem'}")
    print("-" * 80)
    for i, (model, stage, backend, sparsity, m_value) in enumerate(tests, 1):
        skip = " (跳过)" if i < args.start else ""
        sp_str = f" ({sparsity})" if sparsity else ""
        backend_str = f"{backend}{sp_str}"
        gpu_mem = get_optimal_gpu_mem_util(model, stage, m_value)
        print(f"  {i:<4} {model:<18} {backend_str:<18} {stage:<8} {m_value:<8} {gpu_mem:.2f}{skip}")
    print("-" * 80)
    print()
    
    if args.dry_run:
        print_warning("[DRY-RUN 模式] 以下命令将被执行:")
        print()
    
    # 执行测试
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    total = len(tests)
    
    for i, (model, stage, backend, sparsity, m_value) in enumerate(tests, 1):
        if i < args.start:
            skipped_count += 1
            continue
        
        # 清理之前的进程（第一个测试只清理不等待）
        if not args.dry_run:
            kill_vllm_processes()
            if i > args.start:
                wait_for_gpu_release(args.wait)
        
        # 运行测试
        success = run_single_test(
            model, stage, backend, sparsity, m_value,
            test_num=i,
            total_tests=total,
            dry_run=args.dry_run,
            log_file=None if args.dry_run else log_file,
            gpu_id=args.gpu_id,
        )
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # 最终清理
    if not args.dry_run:
        kill_vllm_processes()
    
    # 打印总结
    print_header("测试总结")
    print(f"  总测试数: {total}")
    print(f"  跳过:     {skipped_count}")
    print(f"  成功:     {Colors.GREEN}{success_count}{Colors.NC}")
    print(f"  失败:     {Colors.RED}{fail_count}{Colors.NC}")
    
    if not args.dry_run:
        print()
        print_info(f"详细日志: {log_file}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
