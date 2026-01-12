#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# run_all_tests.sh - SlideSparse FP8 一键测试脚本
#
# 功能:
#   1. 编译 cuBLASLt Extension（如果需要）
#   2. 依次运行 4 个测试脚本
#
# 使用方法:
#   chmod +x /root/vllmbench/slidesparse/test/FP8/run_all_tests.sh
#   ./run_all_tests.sh                      # 默认: CUTLASS fallback
#   ./run_all_tests.sh --use-cublaslt       # cuBLASLt + BF16
#   ./run_all_tests.sh --use-cublaslt --inner-fp32  # cuBLASLt + FP32
#   ./run_all_tests.sh --use-cusparselt     # cuSPARSELt
#   
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUBLASLT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")/csrc/cublaslt_gemm"

# 解析参数
USE_CUBLASLT=0
USE_CUSPARSELT=0
INNER_FP32=0
TEST_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --use-cublaslt)
            USE_CUBLASLT=1
            TEST_ARGS="$TEST_ARGS --use-cublaslt"
            shift
            ;;
        --use-cusparselt)
            USE_CUSPARSELT=1
            TEST_ARGS="$TEST_ARGS --use-cusparselt"
            shift
            ;;
        --inner-fp32)
            INNER_FP32=1
            TEST_ARGS="$TEST_ARGS --inner-fp32"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  (无参数)                    CUTLASS fallback"
            echo "  --use-cublaslt              cuBLASLt + BF16"
            echo "  --use-cublaslt --inner-fp32 cuBLASLt + FP32"
            echo "  --use-cusparselt            cuSPARSELt"
            echo "  -h, --help                  显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            exit 1
            ;;
    esac
done

# 确定后端名称
if [[ $USE_CUBLASLT -eq 1 ]]; then
    if [[ $INNER_FP32 -eq 1 ]]; then
        BACKEND_NAME="cuBLASLt + FP32"
    else
        BACKEND_NAME="cuBLASLt + BF16"
    fi
elif [[ $USE_CUSPARSELT -eq 1 ]]; then
    BACKEND_NAME="cuSPARSELt"
else
    BACKEND_NAME="CUTLASS fallback"
fi

# 打印横幅
print_banner() {
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                    SlideSparse FP8 一键测试脚本${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BLUE}后端:${NC} ${BACKEND_NAME}"
    echo -e "  ${BLUE}参数:${NC} ${TEST_ARGS:-"(无)"}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

# 打印步骤标题
print_step() {
    local step_num=$1
    local step_name=$2
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}[${step_num}] ${step_name}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# 运行测试并记录结果
run_test() {
    local test_script=$1
    local test_name=$2
    
    echo -e "${CYAN}>>> 运行: ${test_name}${NC}"
    echo ""
    
    if python3 "${SCRIPT_DIR}/${test_script}" $TEST_ARGS; then
        echo ""
        echo -e "${GREEN}✓ ${test_name} 完成${NC}"
        return 0
    else
        echo ""
        echo -e "${RED}✗ ${test_name} 失败${NC}"
        return 1
    fi
}

# 主流程
main() {
    local start_time=$(date +%s)
    local failed_tests=()
    
    print_banner
    
    # Step 0: 编译 cuBLASLt Extension（如果使用 cuBLASLt）
    if [[ $USE_CUBLASLT -eq 1 ]]; then
        print_step "0/4" "编译 cuBLASLt Extension"
        
        if [[ -f "${CUBLASLT_DIR}/setup_cublaslt.py" ]]; then
            echo -e "${CYAN}>>> 运行: python3 setup_cublaslt.py build${NC}"
            echo ""
            cd "${CUBLASLT_DIR}"
            if python3 setup_cublaslt.py build; then
                echo ""
                echo -e "${GREEN}✓ cuBLASLt Extension 编译完成${NC}"
            else
                echo ""
                echo -e "${RED}✗ cuBLASLt Extension 编译失败${NC}"
                exit 1
            fi
            cd "${SCRIPT_DIR}"
        else
            echo -e "${YELLOW}⚠ 未找到 setup_cublaslt.py，跳过编译${NC}"
        fi
    fi
    
    # Step 1: 桥接与集成测试
    print_step "1/4" "桥接与集成测试 (test_01_bridge.py)"
    if ! run_test "test_01_bridge.py" "桥接与集成测试"; then
        failed_tests+=("test_01_bridge.py")
    fi
    
    # Step 2: Kernel 正确性测试
    print_step "2/4" "Kernel 正确性测试 (test_02_kernel.py)"
    if ! run_test "test_02_kernel.py" "Kernel 正确性测试"; then
        failed_tests+=("test_02_kernel.py")
    fi
    
    # Step 3: 端到端推理对比
    print_step "3/4" "端到端推理对比 (test_03_inference.py)"
    if ! run_test "test_03_inference.py" "端到端推理对比"; then
        failed_tests+=("test_03_inference.py")
    fi
    
    # Step 4: 吞吐量对比测试
    print_step "4/4" "吞吐量对比测试 (test_04_throughput.py)"
    if ! run_test "test_04_throughput.py" "吞吐量对比测试"; then
        failed_tests+=("test_04_throughput.py")
    fi
    
    # 汇总结果
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    echo ""
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}                              测试汇总${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${BLUE}后端:${NC}     ${BACKEND_NAME}"
    echo -e "  ${BLUE}耗时:${NC}     ${minutes}分${seconds}秒"
    echo ""
    
    if [[ ${#failed_tests[@]} -eq 0 ]]; then
        echo -e "  ${GREEN}${BOLD}✓ 全部 4 个测试通过！${NC}"
        echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
        echo ""
        exit 0
    else
        echo -e "  ${RED}${BOLD}✗ ${#failed_tests[@]} 个测试失败:${NC}"
        for test in "${failed_tests[@]}"; do
            echo -e "    ${RED}- ${test}${NC}"
        done
        echo -e "${CYAN}════════════════════════════════════════════════════════════════════════════════${NC}"
        echo ""
        exit 1
    fi
}

# 执行主流程
main
