#!/bin/bash
# 编译自定义 CUDA Kernel

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="$SCRIPT_DIR/cuda"

echo "============================================"
echo "编译自定义 CUDA Kernel"
echo "============================================"

cd "$CUDA_DIR"

# 检测 GPU 架构
GPU_ARCH=${GPU_ARCH:-"80"}
echo "Target GPU Architecture: sm_${GPU_ARCH}"

# 编译 custom_gemm.so
echo ""
echo ">>> Compiling custom_gemm.cu..."
nvcc -shared -o libcustom_gemm.so custom_gemm.cu \
    -lcublas \
    -I/usr/local/cuda/include \
    -L/usr/local/cuda/lib64 \
    --compiler-options '-fPIC' \
    -arch=sm_${GPU_ARCH} \
    -O3

echo "✅ libcustom_gemm.so created"

# 编译测试版本 (可选)
if [ "$1" == "--with-test" ]; then
    echo ""
    echo ">>> Compiling test binary..."
    nvcc -o test_custom_gemm custom_gemm.cu \
        -DTEST_MAIN \
        -lcublas \
        -I/usr/local/cuda/include \
        -L/usr/local/cuda/lib64 \
        -arch=sm_${GPU_ARCH} \
        -O3
    echo "✅ test_custom_gemm created"
    
    echo ""
    echo ">>> Running test..."
    ./test_custom_gemm
fi

echo ""
echo "============================================"
echo "编译完成！"
echo "============================================"
echo "生成的文件："
ls -la "$CUDA_DIR"/*.so 2>/dev/null || echo "  (no .so files)"
ls -la "$CUDA_DIR"/test_* 2>/dev/null || echo "  (no test binaries)"
