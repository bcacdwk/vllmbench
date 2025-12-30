# vLLM 框架概述 (Framework Overview)

本文档旨在帮助你快速了解 vLLM 项目的整体目录结构、各个文件夹的用途，以及如何运行和测试这个项目。

---

## 1. 项目目录结构概览

```
vllmbench/
├── vllm/                   # 🔥 核心推理框架（最重要）
├── benchmarks/             # 性能基准测试脚本
├── tests/                  # 测试用例（非常全面）
├── examples/               # 使用示例
├── csrc/                   # C++/CUDA 源代码
├── docs/                   # 文档
├── tools/                  # 辅助工具
├── custom_kernels/         # 自定义kernel示例
├── requirements/           # 依赖配置
├── cmake/                  # CMake 构建配置
├── .buildkite/             # CI/CD 配置
├── .github/                # GitHub Actions 配置
└── ...                     # 其他配置文件
```

---

## 2. 各目录详细说明

### 2.1 `vllm/` - 核心推理框架 ⭐⭐⭐

这是整个 vLLM 项目的核心，包含了所有推理相关的代码。内部组织非常复杂，详细介绍请参考 [framework_vllmcore.md](./framework_vllmcore.md)。

简要概述：
- `vllm/entrypoints/` - 入口点，包含 LLM 类和 API 服务器
- `vllm/engine/` - LLM 引擎实现
- `vllm/model_executor/` - 模型执行器，包含模型定义和量化层
- `vllm/attention/` - 注意力机制实现
- `vllm/v1/` - V1 版本的新架构实现

### 2.2 `benchmarks/` - 性能基准测试 ⭐⭐

用于性能测试和评估的脚本集合。

```
benchmarks/
├── benchmark_throughput.py       # 吞吐量测试（已移至 CLI）
├── benchmark_serving.py          # 在线服务测试
├── benchmark_latency.py          # 延迟测试
├── benchmark_prefix_caching.py   # 前缀缓存测试
├── backend_request_func.py       # 请求后端函数
├── benchmark_utils.py            # 基准测试工具
├── kernels/                      # kernel 级别的 benchmark
├── fused_kernels/               # 融合 kernel benchmark
├── cutlass_benchmarks/          # CUTLASS benchmark
└── ...
```

**注意**: 现在推荐使用 vLLM CLI 来运行 benchmark：
```bash
# 吞吐量测试
vllm bench throughput --help

# 服务测试  
vllm bench serve --help

# 延迟测试
vllm bench latency --help
```

### 2.3 `tests/` - 测试用例 ⭐⭐

非常全面的测试集合，涵盖了几乎所有功能：

```
tests/
├── basic_correctness/      # 基础正确性测试
├── models/                 # 模型测试
├── kernels/               # kernel 测试
├── quantization/          # 量化测试
├── distributed/           # 分布式测试
├── entrypoints/           # 入口点测试
├── engine/                # 引擎测试
├── samplers/              # 采样器测试
├── lora/                  # LoRA 测试
├── multimodal/            # 多模态测试
├── v1/                    # V1 架构测试
├── conftest.py            # pytest 配置
└── ...
```

**运行测试示例**：
```bash
# 运行特定测试
pytest tests/models/test_llama.py -v

# 运行所有 kernel 测试
pytest tests/kernels/ -v

# 运行量化相关测试
pytest tests/quantization/ -v
```

### 2.4 `examples/` - 使用示例 ⭐⭐

包含各种使用场景的示例代码：

```
examples/
├── offline_inference/           # 离线推理示例
│   ├── basic/                   # 基础示例
│   │   ├── generate.py          # 文本生成
│   │   ├── chat.py              # 对话
│   │   ├── embed.py             # 嵌入
│   │   └── ...
│   ├── vision_language.py       # 视觉语言模型
│   ├── spec_decode.py           # 投机解码
│   ├── lora_with_quantization_inference.py  # LoRA + 量化
│   └── ...
├── online_serving/              # 在线服务示例
├── pooling/                     # 池化示例
├── template_*.jinja             # 聊天模板
└── tool_chat_template_*.jinja   # 工具调用模板
```

### 2.5 `docs/` - 文档

官方文档的源文件，使用 MkDocs 构建：

```
docs/
├── getting_started/    # 入门指南
├── usage/              # 使用说明
├── models/             # 支持的模型
├── configuration/      # 配置说明
├── deployment/         # 部署指南
├── benchmarking/       # 性能测试文档
├── contributing/       # 贡献指南
└── ...
```

**官方文档网站**: https://docs.vllm.ai/en/stable/usage/

### 2.6 `csrc/` - C++/CUDA 源代码

底层高性能 kernel 的实现：

```
csrc/
├── attention/              # 注意力 kernel
├── quantization/           # 量化 kernel
├── moe/                    # MoE kernel
├── cutlass_extensions/     # CUTLASS 扩展
├── mamba/                  # Mamba 模型 kernel
├── sparse/                 # 稀疏计算 kernel
├── activation_kernels.cu   # 激活函数 kernel
├── layernorm_kernels.cu    # LayerNorm kernel
├── pos_encoding_kernels.cu # 位置编码 kernel
├── torch_bindings.cpp      # PyTorch 绑定
└── ...
```

**注意**: 这些是编译后供 Python 调用的 CUDA/C++ 实现，通常不需要直接修改。

### 2.7 `tools/` - 辅助工具

开发和运维相关的工具：

```
tools/
├── profiler/              # 性能分析工具
├── ep_kernels/            # Expert Parallelism kernels
├── pre_commit/            # 代码检查钩子
├── flashinfer-build.sh    # FlashInfer 构建脚本
├── install_deepgemm.sh    # DeepGEMM 安装脚本
├── install_gdrcopy.sh     # GDRCopy 安装脚本
├── check_repo.sh          # 仓库检查脚本
└── ...
```

### 2.8 `custom_kernels/` - 自定义 Kernel 示例

这是一个自定义 kernel 的示例目录（项目作者添加）：

```
custom_kernels/
├── cuda/           # CUDA kernel 示例
├── triton/         # Triton kernel 示例
├── compile.sh      # 编译脚本
└── patch_example.py # 补丁示例
```

---

## 3. 如何运行 vLLM

### 3.1 安装

**方式一：从 PyPI 安装**
```bash
pip install vllm
```

**方式二：从源码安装**
```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 3.2 基本推理示例

```python
from vllm import LLM, SamplingParams

# 创建 LLM 实例
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256
)

# 生成
prompts = ["你好，请介绍一下你自己。"]
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
```

### 3.3 运行 benchmark

```bash
# 使用 vLLM CLI 运行吞吐量测试
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100

# 服务基准测试
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name sharegpt \
    --request-rate 10
```

### 3.4 启动 API 服务器

```bash
# 启动 OpenAI 兼容的 API 服务器
vllm serve meta-llama/Llama-3.2-1B-Instruct \
    --host 0.0.0.0 \
    --port 8000

# 或者使用 Python
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct
```

---

## 4. 模型下载与配置

### 4.1 从 HuggingFace 下载模型

vLLM 直接支持 HuggingFace 模型格式。你可以通过以下方式获取模型：

**方式一：自动下载**
```python
# vLLM 会自动从 HuggingFace 下载模型
llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct")
```

**方式二：手动下载**
```bash
# 使用 huggingface-cli
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct

# 或使用 Python
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-3.2-1B-Instruct", local_dir="./models/llama-3.2")
```

**方式三：使用本地路径**
```python
llm = LLM(model="/path/to/your/model")
```

### 4.2 常用模型推荐

| 模型系列 | HuggingFace 路径 | 说明 |
|---------|-----------------|------|
| Llama 3.2 | `meta-llama/Llama-3.2-1B-Instruct` | Meta 最新轻量模型 |
| Llama 3.1 | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 主流开源模型 |
| Qwen 2.5 | `Qwen/Qwen2.5-7B-Instruct` | 阿里千问模型 |
| DeepSeek | `deepseek-ai/deepseek-llm-7b-chat` | DeepSeek 模型 |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.2` | Mistral AI 模型 |

### 4.3 处理不同格式的模型

**SafeTensors 格式（推荐）**
```python
# 直接使用，vLLM 原生支持
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
```

**PyTorch 格式 (.pt/.bin)**
```python
# 同样直接支持
llm = LLM(model="/path/to/pytorch/model")
```

**GGUF 格式**
```python
# vLLM 支持 GGUF 量化模型
llm = LLM(model="/path/to/model.gguf")
```

### 4.4 量化模型

**FP8 量化**
```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8"
)
```

**AWQ 量化**
```python
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-AWQ",
    quantization="awq"
)
```

**GPTQ 量化**
```python
llm = LLM(
    model="TheBloke/Llama-2-7B-Chat-GPTQ",
    quantization="gptq"
)
```

---

## 5. 关键配置参数

### 5.1 模型配置

```python
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    
    # 数据类型
    dtype="auto",  # auto, float16, bfloat16, float32
    
    # 量化方法
    quantization=None,  # None, "fp8", "awq", "gptq", "squeezellm"
    
    # 张量并行
    tensor_parallel_size=1,  # GPU 数量
    
    # 信任远程代码
    trust_remote_code=False,
    
    # GPU 内存利用率
    gpu_memory_utilization=0.9,
    
    # 最大模型长度（上下文长度）
    max_model_len=None,  # None 表示使用模型默认值
    
    # 是否启用前缀缓存
    enable_prefix_caching=False,
)
```

### 5.2 采样参数

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    # 生成控制
    max_tokens=256,           # 最大生成 token 数
    temperature=0.8,          # 温度，越高越随机
    top_p=0.95,               # nucleus sampling
    top_k=50,                 # top-k sampling
    
    # 惩罚项
    presence_penalty=0.0,     # 存在惩罚
    frequency_penalty=0.0,    # 频率惩罚
    repetition_penalty=1.0,   # 重复惩罚
    
    # 停止条件
    stop=None,                # 停止词列表
    stop_token_ids=None,      # 停止 token ID 列表
    ignore_eos=False,         # 是否忽略 EOS
    
    # 输出控制
    n=1,                      # 每个 prompt 生成几个结果
    best_of=None,             # 采样最佳
    logprobs=None,            # 返回 logprobs 数量
)
```

---

## 6. 推理入口与调用链

vLLM 的推理入口主要有以下几种：

### 6.1 离线批量推理（Offline Inference）

```
用户代码
  │
  ▼
LLM.generate()                    # vllm/entrypoints/llm.py
  │
  ▼
LLMEngine                         # vllm/v1/engine/llm_engine.py
  │
  ▼
EngineCoreClient                  # 引擎核心客户端
  │
  ▼
GPUModelRunner.execute_model()    # vllm/v1/worker/gpu_model_runner.py
  │
  ▼
Model.forward()                   # vllm/model_executor/models/*.py
```

### 6.2 在线服务（Online Serving）

```
HTTP 请求
  │
  ▼
OpenAI API Server                 # vllm/entrypoints/openai/api_server.py
  │
  ▼
AsyncLLMEngine                    # 异步引擎
  │
  ▼
... (同上)
```

### 6.3 CLI 入口

```bash
# 主要的 CLI 命令
vllm serve        # 启动服务器
vllm bench        # 运行 benchmark
vllm chat         # 交互式对话
```

---

## 7. 小结

本文档介绍了 vLLM 项目的整体结构。如需深入了解：

- **核心框架细节** → 请参考 [framework_vllmcore.md](./framework_vllmcore.md)
- **线性层与 GEMM** → 请参考 [framework_lineargemm.md](./framework_lineargemm.md)

vLLM 的设计理念是通过 PagedAttention、连续批处理和 CUDA Graph 等技术，实现高吞吐、低延迟的大模型推理。整个项目结构清晰，模块化程度高，便于二次开发和定制。
