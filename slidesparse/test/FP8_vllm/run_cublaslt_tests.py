#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt 一键测试入口

功能:
  1. 依次运行 4 个测试脚本（使用 cuBLASLt 后端）
  2. 汇总测试结果

使用方法:
  python3 run_cublaslt_tests.py               # cuBLASLt + BF16（默认）
  python3 run_cublaslt_tests.py --inner-fp32  # cuBLASLt + FP32


"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

# 脚本所在目录
SCRIPT_DIR = Path(__file__).parent.absolute()

# 颜色定义
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


def print_banner(backend_name: str):
    """打印横幅"""
    print()
    print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
    print(f"{Colors.BOLD}                    SlideSparse cuBLASLt 一键测试{Colors.NC}")
    print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
    print(f"  {Colors.BLUE}后端:{Colors.NC} {backend_name}")
    print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
    print()


def print_step(step: str, title: str):
    """打印步骤标题"""
    print()
    print(f"{Colors.YELLOW}{'━' * 80}{Colors.NC}")
    print(f"{Colors.BOLD}[{step}] {title}{Colors.NC}")
    print(f"{Colors.YELLOW}{'━' * 80}{Colors.NC}")
    print()


def run_test(script: str, args: list[str]) -> bool:
    """运行测试脚本"""
    script_path = SCRIPT_DIR / script
    cmd = [sys.executable, str(script_path)] + args
    
    print(f"{Colors.CYAN}>>> 运行: python3 {script} {' '.join(args)}{Colors.NC}")
    print()
    
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    
    if result.returncode == 0:
        print(f"\n{Colors.GREEN}✓ {script} 完成{Colors.NC}")
        return True
    else:
        print(f"\n{Colors.RED}✗ {script} 失败{Colors.NC}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="SlideSparse cuBLASLt 一键测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python3 run_cublaslt_tests.py               # cuBLASLt + BF16（默认）
  python3 run_cublaslt_tests.py --inner-fp32  # cuBLASLt + FP32
        """
    )
    parser.add_argument("--inner-fp32", action="store_true", 
                        help="使用 FP32 中间累加")
    args = parser.parse_args()
    
    # 确定后端名称和测试参数
    backend_name = "cuBLASLt + FP32" if args.inner_fp32 else "cuBLASLt + BF16"
    test_args = ["--use-cublaslt"]
    if args.inner_fp32:
        test_args.append("--inner-fp32")
    
    print_banner(backend_name)
    
    start_time = time.time()
    failed_tests = []
    
    # 测试配置: (步骤, 脚本, 参数)
    tests = [
        ("1/4", "test_01_bridge.py",     "桥接与集成测试",     []),
        ("2/4", "test_02_kernel.py",     "Kernel 正确性测试",  test_args),
        ("3/4", "test_03_inference.py",  "端到端推理对比",     test_args),
        ("4/4", "test_04_throughput.py", "吞吐量对比测试",     test_args),
    ]
    
    for step, script, title, script_args in tests:
        print_step(step, f"{title} ({script})")
        if not run_test(script, script_args):
            failed_tests.append(script)
    
    # 汇总结果
    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    
    print()
    print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
    print(f"{Colors.BOLD}                              测试汇总{Colors.NC}")
    print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
    print(f"  {Colors.BLUE}后端:{Colors.NC}     {backend_name}")
    print(f"  {Colors.BLUE}耗时:{Colors.NC}     {minutes}分{seconds}秒")
    print()
    
    if not failed_tests:
        print(f"  {Colors.GREEN}{Colors.BOLD}✓ 全部 {len(tests)} 个测试通过！{Colors.NC}")
        print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
        print()
        return 0
    else:
        print(f"  {Colors.RED}{Colors.BOLD}✗ {len(failed_tests)} 个测试失败:{Colors.NC}")
        for t in failed_tests:
            print(f"    {Colors.RED}- {t}{Colors.NC}")
        print(f"{Colors.CYAN}{'═' * 80}{Colors.NC}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
