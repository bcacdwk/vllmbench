#!/bin/bash
# ============================================================================
# SlideSparse cuBLASLt 测试运行脚本
# ============================================================================
#
# 使用方法:
#   ./run_all.sh                    # 运行所有测试 (默认 CUTLASS)
#   ./run_all.sh --cublaslt         # 启用 cuBLASLt 运行所有测试
#   ./run_all.sh --inner-fp32       # 启用 FP32 中间精度
#   ./run_all.sh --full             # 运行完整测试 (包括性能对比)
#   ./run_all.sh 01                 # 只运行 01_bridge 测试
#   ./run_all.sh 02 03              # 运行 02_kernel 和 03_inference
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
USE_CUBLASLT=""
INNER_FP32=""
FULL_TEST=""
QUIET=""
SPECIFIC_TESTS=()

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cublaslt)
            USE_CUBLASLT=1
            shift
            ;;
        --inner-fp32)
            INNER_FP32=1
            shift
            ;;
        --full)
            FULL_TEST=1
            shift
            ;;
        -q|--quiet)
            QUIET="-q"
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
            echo "  --cublaslt    启用 USE_CUBLASLT=1"
            echo "  --inner-fp32  启用 INNER_DTYPE_FP32=1"
            echo "  --full        运行完整测试 (包括性能对比)"
            echo "  -q, --quiet   静默模式"
            echo "  -h, --help    显示帮助"
            echo ""
            echo "Test numbers:"
            echo "  01  桥接与集成测试"
            echo "  02  Kernel 正确性测试"
            echo "  03  推理测试"
            echo "  04  吞吐量测试"
            echo ""
            echo "Examples:"
            echo "  $0                    # 运行所有测试"
            echo "  $0 --cublaslt         # 启用 cuBLASLt"
            echo "  $0 01 02              # 只运行 01 和 02"
            echo "  $0 --cublaslt --full  # 完整 cuBLASLt 测试"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 设置环境变量
if [ -n "$USE_CUBLASLT" ]; then
    export USE_CUBLASLT=1
fi

if [ -n "$INNER_FP32" ]; then
    export INNER_DTYPE_FP32=1
fi

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 打印配置
echo -e "${BOLD}========================================${NC}"
echo -e "${BOLD}SlideSparse cuBLASLt 测试套件${NC}"
echo -e "${BOLD}========================================${NC}"
echo ""
echo -e "配置:"
echo -e "  USE_CUBLASLT:     ${USE_CUBLASLT:-0}"
echo -e "  INNER_DTYPE_FP32: ${INNER_FP32:-0}"
echo -e "  FULL_TEST:        ${FULL_TEST:-0}"
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
    CMD="python3 $test_path"
    
    if [ -n "$QUIET" ]; then
        CMD="$CMD $QUIET"
    fi
    
    if [ -n "$FULL_TEST" ] && [[ "$test_num" == "02" || "$test_num" == "04" ]]; then
        CMD="$CMD --full"
    fi
    
    if [ -n "$FULL_TEST" ] && [[ "$test_num" == "03" ]]; then
        CMD="$CMD --compare"
    fi
    
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
