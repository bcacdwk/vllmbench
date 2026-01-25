# =============================================================================
# vLLM Kernel 开发专用镜像
#
# 设计思路：
# 1. 基座：继承 vLLM 官方镜像，复用已有的 PyTorch、Triton 和 uv 环境。
# 2. 补全：安装 nvcc 和 cuda-libraries-dev，补全编译 .cu 文件所需的完整头文件。
# 3. 劫持：卸载预装的 vLLM 包，为挂载本地源码并执行 pip install -e . 腾出空间。
# 4. 加速：利用 uv 包管理器和 ccache 缓存，最大化缩短构建和重编时间。
# =============================================================================
FROM vllm/vllm-openai:v0.13.0

USER root
ENV DEBIAN_FRONTEND=noninteractive

# -----------------------------------------------------------------------------
# 1. 基础配置：APT 缓存与源
# -----------------------------------------------------------------------------
# 配置 APT 保持下载的包缓存，加速后续重复构建
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# -----------------------------------------------------------------------------
# 2. 安装 CUDA 开发环境与系统工具
# -----------------------------------------------------------------------------
# 安装编译所需的编译器、CMake、Ninja 以及 CUDA 核心头文件
# 特别补充 libnccl-dev 以确保分布式通信库能正确链接
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    git vim curl wget tmux \
    build-essential cmake ninja-build pkg-config \
    cuda-nvcc-12-9 \
    cuda-libraries-dev-12-9 \
    libnccl-dev \
    ccache \
    fonts-noto-cjk fonts-wqy-zenhei \
    # 性能分析工具 (按需保留)
    nsight-compute-2025.3.0 \
    nsight-systems-2025.3.2

# 启用 ccache 以加速 C++ 代码的增量编译
ENV PATH="/usr/lib/ccache:${PATH}"
ENV CCACHE_DIR=/root/.ccache

# -----------------------------------------------------------------------------
# 3. 安装依赖库：cuSPARSELt (自定义 Kernel 依赖)
# -----------------------------------------------------------------------------
# 根据架构自动选择安装 cuSPARSELt 0.8.1
RUN export ARCH=$(dpkg --print-architecture) && \
    apt-get update && \
    apt-get remove -y libcusparselt0 libcusparselt-dev || true && \
    if [ "${ARCH}" = "amd64" ]; then \
        CUSPARSE_URL="https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-repo-ubuntu2204-0.8.1_0.8.1-1_amd64.deb"; \
    elif [ "${ARCH}" = "arm64" ]; then \
        CUSPARSE_URL="https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb"; \
    else \
        echo "Unsupported architecture: ${ARCH}" && exit 1; \
    fi && \
    wget -q ${CUSPARSE_URL} -O /tmp/cusparselt-repo.deb && \
    dpkg -i /tmp/cusparselt-repo.deb && \
    cp /var/cusparselt-local-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-get install -y --no-install-recommends cusparselt-cuda-12 && \
    rm -f /tmp/cusparselt-repo.deb && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "✅ cuSPARSELt 0.8.1 installed"

# -----------------------------------------------------------------------------
# 4. 环境清理：卸载冲突包
# -----------------------------------------------------------------------------
# 使用 uv (比 pip 快) 卸载镜像预置的 vLLM，防止 Python 导入路径混淆
# 这一步至关重要，确保后续 pip install -e . 生效
RUN uv pip uninstall --system vllm

# 卸载 pip 安装的 cusparselt，强制使用系统级安装的版本
RUN uv pip uninstall --system nvidia-cusparselt-cu12 || true

# -----------------------------------------------------------------------------
# 5. 编译参数调优
# -----------------------------------------------------------------------------
# 限制目标 CUDA 架构，显著减少 JIT 编译时间 (根据实际 GPU 型号调整)
# 8.0=A100, 9.0=H100/H200, 10.0=B100/B200
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0"

# 控制编译并发数：
# MAX_JOBS 控制 Ninja 并行文件数，NVCC_THREADS 控制编译器内部线程
# 同时限制以防止内存溢出 (OOM)
ENV MAX_JOBS=4
ENV NVCC_THREADS=8

# 修正动态库搜索路径
# 优先加载 /usr/local/nvidia 以兼容云环境可能挂载的宿主机驱动
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# -----------------------------------------------------------------------------
# 6. vLLM 预编译模式配置
# -----------------------------------------------------------------------------
# 启用预编译模式：仅编译 Python 层修改，C++ 核心库直接下载官方编译好的 Wheel
ENV VLLM_USE_PRECOMPILED=1

# 指定 v0.13.0 版本的官方 Commit Hash (来源: git rev-parse HEAD)
ENV VLLM_PRECOMPILED_WHEEL_COMMIT=72506c98349d6bcd32b4e33eec7b5513453c1502

# 指定 CUDA 版本变体 (推荐使用 cu12 以保证兼容性)
ENV VLLM_PRECOMPILED_WHEEL_VARIANT=cu12

# -----------------------------------------------------------------------------
# 7. 工作空间与工具配置
# -----------------------------------------------------------------------------
WORKDIR /root/vllmbench

# 复制 vllmbench 源代码到镜像中
COPY . /root/vllmbench/

# 配置 uv 行为：增加超时容错，使用 Copy 模式避免 Docker 缓存层的硬链接错误
ENV UV_HTTP_TIMEOUT=500
ENV UV_LINK_MODE=copy

CMD ["sleep", "infinity"]