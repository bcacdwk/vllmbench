#!/bin/bash
# 运行所有 cuBLASLt 测试
# Usage: ./run_all_tests.sh [--no-cublaslt] [--quiet]
#
# 默认启用 cuBLASLt 后端，使用 --no-cublaslt 可禁用

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# 解析参数 (默认启用 cuBLASLt)
QUIET=""
CUBLASLT="1"
for arg in "$@"; do
    case $arg in
        --quiet|-q)
            QUIET="-q"
            ;;
        --no-cublaslt)
            CUBLASLT=""
            ;;
    esac
done

# 设置环境变量
if [ -n "$CUBLASLT" ]; then
    export VLLM_USE_CUBLASLT=1
    echo "=========================================="
    echo "cuBLASLt 后端: 启用"
    echo "=========================================="
else
    echo "=========================================="
    echo "cuBLASLt 后端: 禁用 (使用原生 cutlass)"
    echo "=========================================="
fi

echo ""
echo "=========================================="
echo "运行 cuBLASLt 测试套件"
echo "=========================================="
echo ""

# 测试文件列表
TESTS=(
    "slidesparse/test/test_cublaslt_01_import.py"
    "slidesparse/test/test_cublaslt_02_vllm_integration.py"
    "slidesparse/test/test_cublaslt_03_inference.py"
    "slidesparse/test/test_cublaslt_04_kernel_correctness.py"
    "slidesparse/test/test_cublaslt_05_throughput.py"
)

PASSED=0
FAILED=0
RESULTS=()

for test in "${TESTS[@]}"; do
    echo "----------------------------------------"
    echo "运行: $test"
    echo "----------------------------------------"
    
    if python3 "$test" $QUIET; then
        PASSED=$((PASSED + 1))
        RESULTS+=("✓ $test")
    else
        FAILED=$((FAILED + 1))
        RESULTS+=("✗ $test")
    fi
    echo ""
done

echo "=========================================="
echo "测试结果汇总"
echo "=========================================="
echo "通过: $PASSED"
echo "失败: $FAILED"
echo ""
for result in "${RESULTS[@]}"; do
    echo "  $result"
done
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
