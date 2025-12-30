# vLLM 核心框架详解 (Framework vLLM Core)

本文档深入介绍 vLLM 核心推理框架 `vllm/` 目录的组织结构，并梳理典型模型（如 Llama/Qwen2）的调用链。

---

## 1. vllm/ 目录结构总览

```
vllm/
├── entrypoints/        # 🔵 入口点（API、CLI、LLM类）
├── engine/             # 🔵 推理引擎（V0/Legacy）
├── v1/                 # 🔵 V1 新架构（推荐）
├── model_executor/     # 🔴 模型执行器（核心）
├── attention/          # 🔴 注意力机制
├── distributed/        # 分布式相关
├── config/             # 配置类
├── inputs/             # 输入处理
├── outputs.py          # 输出定义
├── sampling_params.py  # 采样参数
├── sequence.py         # 序列定义
├── lora/               # LoRA 支持
├── multimodal/         # 多模态支持
├── tokenizers/         # 分词器
├── transformers_utils/  # Transformers 工具
├── platforms/          # 平台适配（CUDA/ROCm/CPU等）
├── compilation/        # 编译优化
├── triton_utils/       # Triton 工具
├── plugins/            # 插件系统
├── utils/              # 通用工具
└── _custom_ops.py      # 自定义算子绑定
```

---

## 2. 核心模块详解

### 2.1 `entrypoints/` - 入口点

所有用户接口的入口：

```
entrypoints/
├── llm.py                  # ⭐ LLM 类 - 离线推理主入口
├── api_server.py           # FastAPI 服务器
├── openai/                 # OpenAI 兼容 API
│   ├── api_server.py       # OpenAI API 服务器
│   └── ...
├── cli/                    # CLI 命令
│   ├── main.py             # CLI 主入口
│   ├── benchmark/          # benchmark 命令
│   └── serve.py            # serve 命令
├── chat_utils.py           # 聊天工具
└── ...
```

**LLM 类的主要方法**：
```python
class LLM:
    def __init__(self, model, ...):         # 初始化
    def generate(self, prompts, ...):       # 文本生成
    def chat(self, messages, ...):          # 对话生成
    def encode(self, prompts, ...):         # 编码（Embedding）
    def embed(self, prompts, ...):          # 嵌入生成
```

### 2.2 `engine/` - 推理引擎（Legacy）

V0 版本的引擎实现（现已指向 V1）：

```
engine/
├── __init__.py
├── llm_engine.py           # 现在指向 v1 版本
├── async_llm_engine.py     # 异步引擎
├── arg_utils.py            # 参数解析
└── protocol.py             # 协议定义
```

**当前状态**：`llm_engine.py` 现在实际上导入自 `v1`：
```python
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
LLMEngine = V1LLMEngine
```

### 2.3 `v1/` - V1 新架构 ⭐

vLLM 的新一代架构，推荐使用：

```
v1/
├── engine/                 # V1 引擎
│   ├── llm_engine.py       # ⭐ LLMEngine 主类
│   ├── core_client.py      # 引擎核心客户端
│   ├── input_processor.py  # 输入处理
│   └── output_processor.py # 输出处理
├── worker/                 # Worker 实现
│   ├── gpu_model_runner.py # ⭐ GPU 模型运行器
│   ├── gpu_worker.py       # GPU Worker
│   ├── cpu_model_runner.py # CPU 模型运行器
│   └── worker_base.py      # Worker 基类
├── attention/              # V1 注意力
├── sample/                 # 采样器
├── spec_decode/            # 投机解码
├── outputs.py              # 输出定义
└── ...
```

### 2.4 `model_executor/` - 模型执行器 ⭐⭐⭐

这是整个推理框架的核心，包含模型定义和执行逻辑：

```
model_executor/
├── models/                 # 🔴 所有支持的模型实现
│   ├── llama.py            # ⭐ Llama 模型
│   ├── qwen2.py            # ⭐ Qwen2 模型
│   ├── mixtral.py          # Mixtral MoE 模型
│   ├── deepseek_v2.py      # DeepSeek V2
│   ├── registry.py         # 模型注册表
│   ├── interfaces.py       # 模型接口定义
│   └── ...（200+ 模型文件）
├── layers/                 # 🔴 模型层实现
│   ├── linear.py           # ⭐ 线性层（含量化）
│   ├── activation.py       # 激活函数
│   ├── layernorm.py        # LayerNorm
│   ├── rotary_embedding/   # RoPE 位置编码
│   ├── vocab_parallel_embedding.py  # 词嵌入
│   ├── logits_processor.py # Logits 处理
│   ├── fused_moe/          # 融合 MoE 层
│   └── quantization/       # 🔴 量化实现
│       ├── fp8.py          # ⭐ FP8 量化
│       ├── awq.py          # AWQ 量化
│       ├── gptq.py         # GPTQ 量化
│       ├── base_config.py  # 量化基类
│       └── utils/          # 量化工具
├── model_loader/           # 模型加载器
├── custom_op.py            # 自定义算子
└── parameter.py            # 参数定义
```

### 2.5 `attention/` - 注意力机制

```
attention/
├── layer.py                # 注意力层封装
├── selector.py             # 后端选择器
├── backends/               # 注意力后端
│   ├── abstract.py         # 抽象基类
│   ├── flash_attn.py       # FlashAttention
│   ├── flashinfer.py       # FlashInfer
│   ├── xformers.py         # xFormers
│   └── ...
├── ops/                    # 注意力操作
└── utils/                  # 工具函数
```

### 2.6 `config/` - 配置类

所有配置相关的定义：

```
config/
├── __init__.py             # 导出所有配置类
├── model.py                # 模型配置
├── cache.py                # KV Cache 配置
├── parallel.py             # 并行配置
├── scheduler.py            # 调度器配置
├── vllm.py                 # VllmConfig 主配置
└── ...
```

---

## 3. 典型调用链分析（Llama/Qwen2）

### 3.1 完整调用链图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         用户代码入口                                      │
│  llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")                           │
│  outputs = llm.generate(prompts, sampling_params)                       │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LLM 类 (vllm/entrypoints/llm.py)                                       │
│                                                                         │
│  def __init__(self, model, ...):                                        │
│      engine_args = EngineArgs(model=model, ...)                         │
│      self.llm_engine = LLMEngine.from_engine_args(engine_args)         │
│                                                                         │
│  def generate(self, prompts, sampling_params):                          │
│      self._validate_and_add_requests(prompts, params)                   │
│      outputs = self._run_engine()  # 循环调用 engine.step()             │
│      return outputs                                                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LLMEngine (vllm/v1/engine/llm_engine.py)                               │
│                                                                         │
│  def __init__(...):                                                     │
│      self.input_processor = InputProcessor(...)                         │
│      self.output_processor = OutputProcessor(...)                       │
│      self.engine_core = EngineCoreClient.make_client(...)              │
│                                                                         │
│  def step(self):                                                        │
│      engine_core_outputs = self.engine_core.step()  # 调用核心引擎      │
│      return self.output_processor.process(...)                          │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  EngineCoreClient → EngineCore (vllm/v1/engine/core_client.py)          │
│                                                                         │
│  内部维护 model_executor，负责调度和管理请求                              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  GPUModelRunner (vllm/v1/worker/gpu_model_runner.py)                    │
│                                                                         │
│  def execute_model(self, scheduler_output):                             │
│      # 1. 准备输入                                                       │
│      model_input = self._prepare_inputs(...)                            │
│      # 2. 准备注意力元数据                                                │
│      attn_metadata = self._prepare_attention_metadata(...)              │
│      # 3. 执行模型前向传播                                                │
│      with set_forward_context(...):                                     │
│          hidden_states = self.model(                                    │
│              input_ids=model_input.input_ids,                           │
│              positions=model_input.positions,                           │
│              ...                                                        │
│          )                                                              │
│      # 4. 计算 logits 并采样                                             │
│      logits = self.model.compute_logits(hidden_states)                  │
│      sampler_output = self.sampler(logits, sampling_metadata)           │
│      return sampler_output                                              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Model Forward (以 Qwen2ForCausalLM 为例)                                │
│  vllm/model_executor/models/qwen2.py                                    │
│                                                                         │
│  class Qwen2ForCausalLM:                                                │
│      def forward(self, input_ids, positions, ...):                      │
│          hidden_states = self.model(input_ids, positions, ...)          │
│          return hidden_states                                           │
│                                                                         │
│  class Qwen2Model:                                                      │
│      def forward(self, input_ids, positions, ...):                      │
│          # 1. Embedding                                                 │
│          hidden_states = self.embed_tokens(input_ids)                   │
│          residual = None                                                │
│          # 2. 循环所有 Decoder Layer                                     │
│          for layer in self.layers:                                      │
│              hidden_states, residual = layer(positions, hidden_states,  │
│                                              residual)                  │
│          # 3. 最终 LayerNorm                                             │
│          hidden_states, _ = self.norm(hidden_states, residual)          │
│          return hidden_states                                           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Qwen2DecoderLayer.forward()                                            │
│                                                                         │
│  def forward(self, positions, hidden_states, residual):                 │
│      # Self Attention                                                   │
│      if residual is None:                                               │
│          residual = hidden_states                                       │
│          hidden_states = self.input_layernorm(hidden_states)            │
│      else:                                                              │
│          hidden_states, residual = self.input_layernorm(hidden_states,  │
│                                                         residual)       │
│      hidden_states = self.self_attn(positions, hidden_states)           │
│                                                                         │
│      # MLP                                                              │
│      hidden_states, residual = self.post_attention_layernorm(           │
│          hidden_states, residual)                                       │
│      hidden_states = self.mlp(hidden_states)                            │
│      return hidden_states, residual                                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
            ┌───────────────────┴───────────────────┐
            │                                       │
            ▼                                       ▼
┌───────────────────────────────┐   ┌───────────────────────────────────┐
│  Qwen2Attention.forward()     │   │  Qwen2MLP.forward()                │
│                               │   │                                   │
│  # QKV 投影                    │   │  # gate_up_proj (W13)             │
│  qkv, _ = self.qkv_proj(x)    │   │  gate_up, _ = self.gate_up_proj(x)│
│  q, k, v = qkv.split(...)     │   │  x = self.act_fn(gate_up)         │
│  # RoPE                        │   │  # down_proj (W2)                 │
│  q, k = self.rotary_emb(...)  │   │  x, _ = self.down_proj(x)         │
│  # Attention                   │   │  return x                         │
│  attn_output = self.attn(qkv) │   │                                   │
│  # O 投影                      │   │                                   │
│  output, _ = self.o_proj(...)  │   │                                   │
│  return output                 │   │                                   │
└───────────────────────────────┘   └───────────────────────────────────┘
```

### 3.2 关键文件列表

| 层级 | 文件路径 | 说明 |
|-----|---------|------|
| 入口 | `vllm/entrypoints/llm.py` | LLM 类定义 |
| 引擎 | `vllm/v1/engine/llm_engine.py` | V1 LLMEngine |
| 运行器 | `vllm/v1/worker/gpu_model_runner.py` | GPU 模型运行器 |
| 模型 | `vllm/model_executor/models/qwen2.py` | Qwen2 模型 |
| 模型 | `vllm/model_executor/models/llama.py` | Llama 模型 |
| 线性层 | `vllm/model_executor/layers/linear.py` | 线性层定义 |
| 注意力 | `vllm/attention/layer.py` | 注意力层 |
| 量化 | `vllm/model_executor/layers/quantization/fp8.py` | FP8 量化 |

---

## 4. 模型定义详解（Llama/Qwen2）

### 4.1 模型类层次结构

```
nn.Module
    │
    ├── LlamaForCausalLM / Qwen2ForCausalLM    # 顶层模型
    │       │
    │       ├── LlamaModel / Qwen2Model        # 主体模型
    │       │       │
    │       │       ├── VocabParallelEmbedding  # 词嵌入
    │       │       ├── LlamaDecoderLayer[]     # Decoder 层列表
    │       │       │       │
    │       │       │       ├── LlamaAttention   # 注意力
    │       │       │       │   ├── QKVParallelLinear  # Wqkv
    │       │       │       │   ├── RowParallelLinear  # Wo
    │       │       │       │   └── Attention          # 注意力计算
    │       │       │       │
    │       │       │       ├── LlamaMLP         # MLP
    │       │       │       │   ├── MergedColumnParallelLinear  # W13
    │       │       │       │   └── RowParallelLinear           # W2
    │       │       │       │
    │       │       │       ├── RMSNorm (input)
    │       │       │       └── RMSNorm (post_attn)
    │       │       │
    │       │       └── RMSNorm (final)
    │       │
    │       ├── ParallelLMHead                  # LM Head
    │       └── LogitsProcessor                 # Logits 处理
```

### 4.2 四个关键线性层

在 Llama/Qwen2 这类 Dense 模型中，每层有 4 个关键的线性投影：

| 层名 | 类型 | 输入维度 | 输出维度 | 说明 |
|-----|------|---------|---------|------|
| `qkv_proj` | QKVParallelLinear | hidden_size | (q+k+v)_size | Q/K/V 投影合并 |
| `o_proj` | RowParallelLinear | head_dim * num_heads | hidden_size | 输出投影 |
| `gate_up_proj` | MergedColumnParallelLinear | hidden_size | intermediate_size * 2 | Gate + Up 合并 |
| `down_proj` | RowParallelLinear | intermediate_size | hidden_size | Down 投影 |

### 4.3 代码示例：Qwen2MLP

```python
# vllm/model_executor/models/qwen2.py

class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # gate_up_proj 合并了 gate_proj 和 up_proj
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # [gate_size, up_size]
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        # down_proj
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)   # GEMM: W13
        x = self.act_fn(gate_up)            # SiLU 激活
        x, _ = self.down_proj(x)            # GEMM: W2
        return x
```

### 4.4 代码示例：Qwen2Attention

```python
# vllm/model_executor/models/qwen2.py

class Qwen2Attention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # QKV 合并投影
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(...)
        self.attn = Attention(...)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)  # GEMM: Wqkv
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)  # RoPE
        attn_output = self.attn(q, k, v)          # Attention
        output, _ = self.o_proj(attn_output)     # GEMM: Wo
        return output
```

---

## 5. 线性层实现（Linear Layers）

### 5.1 线性层类层次结构

```
LinearBase (CustomOp)
    │
    ├── ReplicatedLinear          # 复制线性层
    ├── ColumnParallelLinear      # 列并行线性层
    │   ├── MergedColumnParallelLinear  # 合并列并行（用于 MLP）
    │   └── QKVParallelLinear           # QKV 并行（用于 Attention）
    └── RowParallelLinear         # 行并行线性层
```

### 5.2 LinearBase 基类

```python
# vllm/model_executor/layers/linear.py

class LinearBase(CustomOp):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,  # 量化配置
        prefix: str = "",
        ...
    ):
        # 根据 quant_config 选择量化方法
        if quant_config is None:
            self.quant_method = UnquantizedLinearMethod()
        else:
            self.quant_method = quant_config.get_quant_method(self, prefix=prefix)
```

### 5.3 Forward 流程

```python
# ColumnParallelLinear.forward()
def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None
    
    # Matrix multiply - 核心 GEMM 调用
    assert self.quant_method is not None
    output_parallel = self.quant_method.apply(self, input_, bias)
    
    if self.gather_output and self.tp_size > 1:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    
    return output, output_bias
```

---

## 6. 引擎配置与参数传递

### 6.1 配置类层次

```
VllmConfig                          # 顶层配置
    ├── ModelConfig                 # 模型配置
    ├── CacheConfig                 # KV Cache 配置
    ├── ParallelConfig              # 并行配置
    ├── SchedulerConfig             # 调度器配置
    ├── DeviceConfig                # 设备配置
    ├── LoRAConfig                  # LoRA 配置（可选）
    ├── MultiModalConfig            # 多模态配置（可选）
    ├── SpeculativeConfig           # 投机解码配置（可选）
    └── ObservabilityConfig         # 可观测性配置
```

### 6.2 参数流向

```
用户参数 (model, dtype, quantization, ...)
         │
         ▼
    EngineArgs                      # vllm/engine/arg_utils.py
         │
         ▼
    VllmConfig.from_engine_args()   # 创建完整配置
         │
         ├──→ ModelConfig           # 传给模型加载器
         ├──→ CacheConfig           # 传给 KV Cache 管理
         ├──→ ParallelConfig        # 传给分布式管理
         └──→ quant_config          # 传给量化层
```

---

## 7. 小结

vLLM 的核心架构可以概括为：

1. **入口层** (`entrypoints/`): 提供用户友好的 API
2. **引擎层** (`engine/`, `v1/engine/`): 管理请求调度和生命周期
3. **执行层** (`v1/worker/`): 在 GPU 上执行模型推理
4. **模型层** (`model_executor/models/`): 具体模型实现
5. **算子层** (`model_executor/layers/`): 底层计算算子

对于想要修改线性层 GEMM 的场景，需要重点关注：
- `vllm/model_executor/layers/linear.py` - 线性层定义
- `vllm/model_executor/layers/quantization/*.py` - 量化方法
- `vllm/_custom_ops.py` - 底层算子绑定

详细的 GEMM 调用链请参考 [framework_lineargemm.md](./framework_lineargemm.md)。
