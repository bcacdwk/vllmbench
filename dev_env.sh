#!/bin/bash
# =============================================================================
# vLLM 开发环境一键启动脚本
# =============================================================================

set -e

# 配置
IMAGE_NAME="vllm-dev:v0.13.0"
CONTAINER_NAME="vllm-kernel-dev"
VLLM_SRC="/home/v-hanshao/vllmbench"
GPU_SRC="/home/v-hanshao/GPU"
HF_CACHE="${HOME}/.cache/huggingface"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}>>> $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# =============================================================================
# 主菜单
# =============================================================================
show_menu() {
    echo ""
    echo "============================================"
    echo "  vLLM Kernel 开发环境"
    echo "============================================"
    echo "1) 构建开发镜像"
    echo "2) 启动开发容器"
    echo "3) 进入已运行的容器"
    echo "4) 停止容器"
    echo "5) 安装 vLLM (可编辑模式)"
    echo "6) 运行 Benchmark"
    echo "7) 编译自定义 Kernel"
    echo "0) 退出"
    echo "============================================"
    read -p "请选择操作 [0-7]: " choice
}

# =============================================================================
# 功能函数
# =============================================================================

build_image() {
    print_step "构建开发镜像..."
    cd "$VLLM_SRC"
    docker build -t "$IMAGE_NAME" -f Dockerfile.dev .
    echo -e "${GREEN}✅ 镜像构建完成: $IMAGE_NAME${NC}"
}

start_container() {
    # 检查是否已存在
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_warning "容器 $CONTAINER_NAME 已存在"
        read -p "是否删除并重新创建? [y/N]: " confirm
        if [[ "$confirm" =~ ^[Yy]$ ]]; then
            docker rm -f "$CONTAINER_NAME"
        else
            echo "使用 '进入已运行的容器' 选项"
            return
        fi
    fi

    print_step "启动开发容器..."
    docker run --gpus all -d --ipc=host \
        --name "$CONTAINER_NAME" \
        -v "$VLLM_SRC":/root/vllmbench \
        -v "$GPU_SRC":/root/GPU \
        -v "$HF_CACHE":/root/.cache/huggingface \
        -e "HF_TOKEN=${HF_TOKEN:-}" \
        -e "PYTHONPATH=/root/vllmbench/custom_kernels" \
        "$IMAGE_NAME" \
        sleep infinity

    echo -e "${GREEN}✅ 容器已启动: $CONTAINER_NAME${NC}"
    echo ""
    echo "进入容器: docker exec -it $CONTAINER_NAME /bin/bash"
}

enter_container() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "容器 $CONTAINER_NAME 未运行"
        return
    fi
    print_step "进入容器..."
    docker exec -it "$CONTAINER_NAME" /bin/bash
}

stop_container() {
    print_step "停止容器..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
    echo -e "${GREEN}✅ 容器已停止${NC}"
}

install_vllm() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "容器 $CONTAINER_NAME 未运行，请先启动容器"
        return
    fi

    echo ""
    echo "选择安装模式:"
    echo "1) 快速安装 (使用预编译 wheel，推荐)"
    echo "2) 完整编译 (首次约 15-30 分钟)"
    read -p "请选择 [1/2]: " mode

    if [[ "$mode" == "1" ]]; then
        print_step "快速安装 vLLM (使用预编译 wheel)..."
        docker exec -it "$CONTAINER_NAME" bash -c "cd /root/vllmbench && VLLM_USE_PRECOMPILED=1 pip install -e ."
    else
        print_step "完整编译 vLLM..."
        docker exec -it "$CONTAINER_NAME" bash -c "cd /root/vllmbench && pip install -e ."
    fi

    echo -e "${GREEN}✅ vLLM 安装完成${NC}"
}

run_benchmark() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "容器 $CONTAINER_NAME 未运行"
        return
    fi

    echo ""
    echo "Benchmark 选项:"
    echo "1) 吞吐量测试 (默认)"
    echo "2) 延迟测试"
    echo "3) 使用自定义 Kernel 的吞吐量测试"
    read -p "请选择 [1-3]: " bench_type

    MODEL=${MODEL:-"Qwen/Qwen2.5-0.5B"}
    INPUT_LEN=${INPUT_LEN:-128}
    OUTPUT_LEN=${OUTPUT_LEN:-64}
    NUM_PROMPTS=${NUM_PROMPTS:-10}

    case $bench_type in
        1)
            print_step "运行吞吐量测试..."
            docker exec -it "$CONTAINER_NAME" bash -c \
                "vllm bench throughput --model $MODEL --input-len $INPUT_LEN --output-len $OUTPUT_LEN --num-prompts $NUM_PROMPTS"
            ;;
        2)
            print_step "运行延迟测试..."
            docker exec -it "$CONTAINER_NAME" bash -c \
                "vllm bench latency --model $MODEL --input-len $INPUT_LEN --output-len $OUTPUT_LEN"
            ;;
        3)
            print_step "使用自定义 Kernel 运行吞吐量测试..."
            docker exec -it "$CONTAINER_NAME" bash -c \
                "VLLM_USE_CUSTOM_GEMM=1 vllm bench throughput --model $MODEL --input-len $INPUT_LEN --output-len $OUTPUT_LEN --num-prompts $NUM_PROMPTS"
            ;;
        *)
            print_error "无效选择"
            ;;
    esac
}

compile_kernels() {
    if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        print_error "容器 $CONTAINER_NAME 未运行"
        return
    fi

    print_step "编译自定义 CUDA Kernel..."
    docker exec -it "$CONTAINER_NAME" bash -c \
        "cd /root/vllmbench/custom_kernels && chmod +x compile.sh && ./compile.sh"
}

# =============================================================================
# 主循环
# =============================================================================
while true; do
    show_menu
    case $choice in
        1) build_image ;;
        2) start_container ;;
        3) enter_container ;;
        4) stop_container ;;
        5) install_vllm ;;
        6) run_benchmark ;;
        7) compile_kernels ;;
        0) echo "Bye!"; exit 0 ;;
        *) print_error "无效选择" ;;
    esac
done
