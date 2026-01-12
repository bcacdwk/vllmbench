#!/bin/bash
# ============================================================================
# SlideSparse 测试运行脚本
# ============================================================================
#
# 使用方法:
#   ./run_all.sh                      # 运行所有测试 (默认: SlideSparse + 外挂 CUTLASS)
#   ./run_all.sh --use-cublaslt       # 测试 cuBLASLt kernel
#   ./run_all.sh --disable-slidesparse # 测试 vLLM 原生路径 (baseline)
#   ./run_all.sh --inner-fp32         # cuBLASLt + FP32 中间精度
#   ./run_all.sh 01                   # 只运行 01_bridge 测试
#   ./run_all.sh 02 03                # 运行 02_kernel 和 03_inference
#
# 三种测试路径:
#   1. vLLM 原生路径 (baseline): --disable-slidesparse
#   2. SlideSparse + 外挂 CUTLASS (对照组): 默认行为
#   3. SlideSparse + cuBLASLt (实验组): --use-cublaslt
#
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# 默认配置
DISABLE_SLIDESPARSE=""
USE_CUBLASLT=""
INNER_FP32=""
SPECIFIC_TESTS=()

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --disable-slidesparse)
            DISABLE_SLIDESPARSE=1
            shift
            ;;
        --use-cublaslt)
            USE_CUBLASLT=1
            shift
            ;;
        --ext-cutlass)
            # 兼容旧参数，等同于默认行为
            USE_CUBLASLT=""
            shift
            ;;
        --inner-fp32)
            INNER_FP32=1
            shift
            ;;
        01|02|03|04)
            SPECIFIC_TESTS+=("$1")
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [TEST_NUMBERS...]"
            echo ""
            echo "Options:"
            echo "  --disable-slidesparse  禁用 SlideSparse hook，测试 vLLM 原生路径 (baseline)"
            echo "  --use-cublaslt         启用 cuBLASLt kernel (USE_CUBLASLT=1)"
            echo "  --ext-cutlass          使用外挂 CUTLASS 路径（等同于默认行为）"
            echo "  --inner-fp32           GEMM 输出使用 FP32 (INNER_DTYPE_FP32=1)"
            echo "  -h, --help             显示帮助"
            echo ""
            echo "Test numbers:"
            echo "  01  桥接与集成测试"
            echo "  02  Kernel 正确性测试"
            echo "  03  推理测试"
            echo "  04  吞吐量测试"
            echo ""
            echo "Examples:"
            echo "  $0                              # 默认: SlideSparse + 外挂 CUTLASS"
            echo "  $0 --use-cublaslt               # SlideSparse + cuBLASLt"
            echo "  $0 --disable-slidesparse        # vLLM 原生路径 (baseline)"
            echo "  $0 --use-cublaslt --inner-fp32  # cuBLASLt + FP32"
            echo "  $0 01 02                        # 只运行 01 和 02"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 设置环境变量
if [ -n "$DISABLE_SLIDESPARSE" ]; then
    export DISABLE_SLIDESPARSE=1
    unset USE_CUBLASLT
    unset INNER_DTYPE_FP32
else
    export DISABLE_SLIDESPARSE=0
    if [ -n "$USE_CUBLASLT" ]; then
        export USE_CUBLASLT=1
    else
        export USE_CUBLASLT=0
    fi
    if [ -n "$INNER_FP32" ]; then
        export INNER_DTYPE_FP32=1
    fi
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 打印配置
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}SlideSparse 测试套件${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
echo -e "配置:"
echo -e "  DISABLE_SLIDESPARSE: ${DISABLE_SLIDESPARSE:-0}"
if [ "${DISABLE_SLIDESPARSE:-0}" = "1" ]; then
    echo -e "  测试路径:            ${YELLOW}vLLM 原生路径 (baseline)${NC}"
else
    echo -e "  USE_CUBLASLT:        ${USE_CUBLASLT:-0}"
    echo -e "  INNER_DTYPE_FP32:    ${INNER_DTYPE_FP32:-0}"
    if [ "${USE_CUBLASLT:-0}" = "1" ]; then
        echo -e "  测试路径:            ${GREEN}SlideSparse + cuBLASLt${NC}"
    else
        echo -e "  测试路径:            ${CYAN}SlideSparse + 外挂 CUTLASS${NC}"
    fi
fi
echo ""

# 测试文件列表
declare -A TESTS=(
    ["01"]="01_bridge.py"
    ["02"]="02_kernel.py"
    ["03"]="03_inference.py"
    ["04"]="04_throughput.py"
)

declare -A TEST_NAMES=(
    ["01"]="桥接与集成测试"
    ["02"]="Kernel 正确性测试"
    ["03"]="推理测试"
    ["04"]="吞吐量测试"
)

# 确定要运行的测试
if [ ${#SPECIFIC_TESTS[@]} -eq 0 ]; then
    TESTS_TO_RUN=("01" "02" "03" "04")
else
    TESTS_TO_RUN=("${SPECIFIC_TESTS[@]}")
fi

# 构建传递给 Python 脚本的参数
PYTHON_ARGS=""
if [ -n "$DISABLE_SLIDESPARSE" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --disable-slidesparse"
elif [ -n "$USE_CUBLASLT" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --use-cublaslt"
fi
if [ -n "$INNER_FP32" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --inner-fp32"
fi

# 运行测试
PASSED=0
FAILED=0
RESULTS=()

for test_num in "${TESTS_TO_RUN[@]}"; do
    test_file="${TESTS[$test_num]}"
    test_name="${TEST_NAMES[$test_num]}"
    test_path="${SCRIPT_DIR}/${test_file}"
    
    echo -e "${CYAN}----------------------------------------${NC}"
    echo -e "${CYAN}[$test_num] $test_name${NC}"
    echo -e "${CYAN}----------------------------------------${NC}"
    
    # 构建命令
    CMD="python3 $test_path $PYTHON_ARGS"
    
    # 运行测试
    if $CMD; then
        PASSED=$((PASSED + 1))
        RESULTS+=("${GREEN}✓ [$test_num] $test_name${NC}")
    else
        FAILED=$((FAILED + 1))
        RESULTS+=("${RED}✗ [$test_num] $test_name${NC}")
    fi
    
    echo ""
done

# 打印汇总
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}测试结果汇总${NC}"
echo -e "${BOLD}========================================${NC}"
echo -e "通过: ${GREEN}$PASSED${NC}"
echo -e "失败: ${RED}$FAILED${NC}"
echo ""
for result in "${RESULTS[@]}"; do
    echo -e "  $result"
done
echo -e "${BOLD}========================================${NC}"

if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
