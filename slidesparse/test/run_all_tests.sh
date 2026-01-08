#!/bin/bash
# ============================================================================
# SlideSparse cuBLASLt 测试脚本
# ============================================================================
# 运行所有 cuBLASLt 集成测试
#
# 使用方法:
#   ./slidesparse/test/run_all_tests.sh              # 运行所有测试 (禁用 cuBLASLt)
#   ./slidesparse/test/run_all_tests.sh --cublaslt   # 运行所有测试 (启用 cuBLASLt)
#   ./slidesparse/test/run_all_tests.sh --import     # 仅运行导入测试
#   ./slidesparse/test/run_all_tests.sh --integration # 仅运行集成测试
#   ./slidesparse/test/run_all_tests.sh --kernel     # 仅运行 kernel 测试
#   ./slidesparse/test/run_all_tests.sh --throughput # 仅运行吞吐量测试
#   ./slidesparse/test/run_all_tests.sh --inference  # 仅运行推理测试
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 默认配置
USE_CUBLASLT=0
RUN_IMPORT=0
RUN_INTEGRATION=0
RUN_KERNEL=0
RUN_THROUGHPUT=0
RUN_INFERENCE=0
RUN_ALL=1

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cublaslt)
            USE_CUBLASLT=1
            shift
            ;;
        --import)
            RUN_IMPORT=1
            RUN_ALL=0
            shift
            ;;
        --integration)
            RUN_INTEGRATION=1
            RUN_ALL=0
            shift
            ;;
        --kernel)
            RUN_KERNEL=1
            RUN_ALL=0
            shift
            ;;
        --throughput)
            RUN_THROUGHPUT=1
            RUN_ALL=0
            shift
            ;;
        --inference)
            RUN_INFERENCE=1
            RUN_ALL=0
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --cublaslt    Enable cuBLASLt backend (VLLM_USE_CUBLASLT=1)"
            echo "  --import      Run import test only"
            echo "  --integration Run integration test only"
            echo "  --kernel      Run kernel correctness test only"
            echo "  --throughput  Run throughput test only"
            echo "  --inference   Run inference test only"
            echo "  -h, --help    Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 设置环境变量
if [ "$USE_CUBLASLT" = "1" ]; then
    export VLLM_USE_CUBLASLT=1
    echo -e "${YELLOW}cuBLASLt 后端已启用 (VLLM_USE_CUBLASLT=1)${NC}"
else
    export VLLM_USE_CUBLASLT=0
    echo -e "${BLUE}cuBLASLt 后端已禁用 (使用原生 cutlass)${NC}"
fi

cd "$PROJECT_ROOT"

echo ""
echo "=============================================="
echo "  SlideSparse cuBLASLt 测试套件"
echo "=============================================="
echo ""

PASSED=0
FAILED=0

run_test() {
    local test_name=$1
    local test_script=$2
    
    echo -e "${BLUE}运行测试: ${test_name}${NC}"
    echo "----------------------------------------"
    
    if python3 "$test_script"; then
        echo -e "${GREEN}✓ ${test_name} 通过${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ ${test_name} 失败${NC}"
        ((FAILED++))
    fi
    
    echo ""
}

# 运行测试
if [ "$RUN_ALL" = "1" ] || [ "$RUN_IMPORT" = "1" ]; then
    run_test "导入测试" "slidesparse/test/test_cublaslt_import.py"
fi

if [ "$RUN_ALL" = "1" ] || [ "$RUN_INTEGRATION" = "1" ]; then
    run_test "vLLM 集成测试" "slidesparse/test/test_cublaslt_vllm_integration.py"
fi

if [ "$RUN_ALL" = "1" ] || [ "$RUN_KERNEL" = "1" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ] || nvidia-smi &>/dev/null; then
        run_test "Kernel 正确性测试" "slidesparse/test/test_cublaslt_kernel_correctness.py"
    else
        echo -e "${YELLOW}⚠ 跳过 Kernel 测试 (无 GPU)${NC}"
    fi
fi

if [ "$RUN_ALL" = "1" ] || [ "$RUN_THROUGHPUT" = "1" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ] || nvidia-smi &>/dev/null; then
        run_test "吞吐量测试" "slidesparse/test/test_cublaslt_throughput.py"
    else
        echo -e "${YELLOW}⚠ 跳过吞吐量测试 (无 GPU)${NC}"
    fi
fi

if [ "$RUN_INFERENCE" = "1" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ] || nvidia-smi &>/dev/null; then
        run_test "推理测试" "slidesparse/test/test_cublaslt_inference.py"
    else
        echo -e "${YELLOW}⚠ 跳过推理测试 (无 GPU)${NC}"
    fi
fi

# 总结
echo "=============================================="
echo "  测试结果总结"
echo "=============================================="
echo -e "  ${GREEN}通过: ${PASSED}${NC}"
echo -e "  ${RED}失败: ${FAILED}${NC}"
echo "=============================================="

if [ "$FAILED" = "0" ]; then
    echo -e "${GREEN}✓ 所有测试通过!${NC}"
    exit 0
else
    echo -e "${RED}✗ 部分测试失败${NC}"
    exit 1
fi
