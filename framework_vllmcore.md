# vLLM 核心框架详解 (Framework vLLM Core)

本文档深入介绍 vLLM 核心推理框架 `vllm/` 目录的组织结构，并梳理典型模型（如 Llama/Qwen2）的完整调用链。本文档旨在帮助开发者深入理解 vLLM 的内部架构，以便进行二次开发、性能优化或添加新功能。

---

## 0. 概述：vLLM 的分层架构

vLLM 采用清晰的分层架构设计，从用户接口到底层计算分为多个层次：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           用户层 (User Layer)                            │
│  LLM 类、OpenAI API、CLI 命令                                            │
│  vllm/entrypoints/                                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                          引擎层 (Engine Layer)                           │
│  LLMEngine、AsyncLLMEngine、请求调度、KV Cache 管理                      │
│  vllm/v1/engine/                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                         执行层 (Executor Layer)                          │
│  GPUModelRunner、Worker、批处理管理                                      │
│  vllm/v1/worker/                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                          模型层 (Model Layer)                            │
│  模型定义（200+ 模型）、Transformer 层实现                               │
│  vllm/model_executor/models/                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                          算子层 (Operator Layer)                         │
│  线性层、注意力层、LayerNorm、激活函数、量化                              │
│  vllm/model_executor/layers/                                            │
├─────────────────────────────────────────────────────────────────────────┤
│                         内核层 (Kernel Layer)                            │
│  CUDA/Triton Kernel、FlashAttention、PagedAttention                     │
│  csrc/、vllm/attention/backends/                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心设计原则

1. **模块化**: 每个模块职责单一，便于维护和扩展
2. **可配置性**: 通过 Config 类统一管理所有配置
3. **可扩展性**: 插件系统支持自定义模型和算子
4. **高性能**: CUDA Graph、量化、批处理等优化
5. **兼容性**: 支持多种硬件平台（CUDA、ROCm、CPU）

---

## 1. vllm/ 目录结构总览

```
vllm/
├── __init__.py             # 包初始化，导出公共 API
├── entrypoints/            # 🔵 入口点（API、CLI、LLM类）
├── engine/                 # 🔵 推理引擎（Legacy，现指向 V1）
├── v1/                     # 🔵 V1 新架构（当前主要实现）
├── model_executor/         # 🔴 模型执行器（核心）
├── attention/              # 🔴 注意力机制
├── distributed/            # 分布式相关
├── config/                 # 配置类
├── inputs/                 # 输入处理
├── outputs.py              # 输出定义
├── sampling_params.py      # 采样参数
├── pooling_params.py       # 池化参数
├── sequence.py             # 序列定义
├── lora/                   # LoRA 支持
├── multimodal/             # 多模态支持
├── tokenizers/             # 分词器
├── transformers_utils/     # Transformers 工具
├── platforms/              # 平台适配（CUDA/ROCm/CPU等）
├── compilation/            # 编译优化（CUDA Graph 等）
├── triton_utils/           # Triton 工具
├── plugins/                # 插件系统
├── utils/                  # 通用工具
├── _custom_ops.py          # 自定义算子绑定
├── forward_context.py      # 前向传播上下文
├── envs.py                 # 环境变量
└── logger.py               # 日志系统
```

---

## 2. 核心模块详解

### 2.1 `entrypoints/` - 入口点 ⭐⭐⭐

所有用户接口的入口，是与 vLLM 交互的第一层：

```
entrypoints/
├── __init__.py             # 导出 LLM 类等
├── llm.py                  # ⭐ LLM 类 - 离线推理主入口
├── api_server.py           # FastAPI 服务器（通用）
├── openai/                 # OpenAI 兼容 API
│   ├── api_server.py       # ⭐ OpenAI API 服务器
│   ├── serving_chat.py     # Chat Completion 处理
│   ├── serving_completion.py # Text Completion 处理
│   ├── serving_embedding.py  # Embedding 处理
│   ├── protocol.py         # API 协议定义
│   └── ...
├── cli/                    # CLI 命令
│   ├── main.py             # CLI 主入口 (vllm 命令)
│   ├── serve.py            # vllm serve 命令
│   ├── benchmark/          # vllm bench 子命令
│   │   ├── throughput.py   # 吞吐量测试
│   │   ├── latency.py      # 延迟测试
│   │   └── serve.py        # 服务测试
│   └── ...
├── chat_utils.py           # 聊天工具函数
├── score_utils.py          # 评分工具
├── utils.py                # 通用工具
├── launcher.py             # 启动器
└── context.py              # 上下文管理
```

#### LLM 类详解 (`vllm/entrypoints/llm.py`)

这是用户使用 vLLM 进行离线推理的主要入口：

```python
# vllm/entrypoints/llm.py (简化版)

class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.
    
    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache).
    """
    
    def __init__(
        self,
        model: str,                              # 模型路径或 HuggingFace ID
        *,
        tokenizer: str | None = None,            # 可选的 tokenizer 路径
        tokenizer_mode: str = "auto",            # tokenizer 模式
        skip_tokenizer_init: bool = False,       # 是否跳过 tokenizer 初始化
        trust_remote_code: bool = False,         # 是否信任远程代码
        tensor_parallel_size: int = 1,           # 张量并行 GPU 数量
        dtype: str = "auto",                     # 数据类型
        quantization: str | None = None,         # 量化方法
        gpu_memory_utilization: float = 0.9,     # GPU 内存利用率
        swap_space: float = 4,                   # 交换空间 (GiB)
        enforce_eager: bool = False,             # 强制 eager 模式
        **kwargs,
    ) -> None:
        """LLM constructor."""
        
        # 1. 创建引擎参数
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            ...
        )
        
        # 2. 创建 LLMEngine (实际是 V1 版本)
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args,
            usage_context=UsageContext.LLM_CLASS
        )
        
        # 3. 初始化请求计数器和其他状态
        self.request_counter = Counter()
        self.model_config = self.llm_engine.model_config
        self.input_processor = self.llm_engine.input_processor

    def generate(
        self,
        prompts: PromptType | Sequence[PromptType],
        sampling_params: SamplingParams | Sequence[SamplingParams] | None = None,
        *,
        use_tqdm: bool = True,
        lora_request: LoRARequest | None = None,
    ) -> list[RequestOutput]:
        """Generates the completions for the input prompts.
        
        Args:
            prompts: The prompts to the LLM.
            sampling_params: The sampling parameters for text generation.
            use_tqdm: Whether to show progress bar.
            lora_request: LoRA request to use for generation.
            
        Returns:
            A list of RequestOutput objects containing the generated texts.
        """
        # 1. 验证模型类型
        if self.model_config.runner_type != "generate":
            raise ValueError("LLM.generate() is only supported for generative models.")
        
        # 2. 使用默认采样参数（如果未提供）
        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()
        
        # 3. 添加所有请求到引擎
        self._validate_and_add_requests(
            prompts=prompts,
            params=sampling_params,
            lora_request=lora_request,
        )
        
        # 4. 运行引擎，循环调用 step() 直到所有请求完成
        outputs = self._run_engine(use_tqdm=use_tqdm)
        
        return outputs

    def _run_engine(self, *, use_tqdm: bool = True) -> list[RequestOutput]:
        """Run the engine until all requests are completed."""
        outputs = []
        
        # 循环直到所有请求完成
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
        
        # 按请求 ID 排序
        return sorted(outputs, key=lambda x: int(x.request_id))

    def chat(
        self,
        messages: list[dict],
        sampling_params: SamplingParams | None = None,
        *,
        chat_template: str | None = None,
        add_generation_prompt: bool = True,
    ) -> list[RequestOutput]:
        """Generate responses for a chat conversation.
        
        Converts the chat conversation to a text prompt using the tokenizer
        and calls the generate() method.
        """
        # 1. 预处理聊天消息，应用聊天模板
        prompts = self.preprocess_chat(
            messages=messages,
            chat_template=chat_template,
            add_generation_prompt=add_generation_prompt,
        )
        
        # 2. 调用 generate
        return self.generate(prompts, sampling_params=sampling_params)

    def embed(self, prompts: PromptType | Sequence[PromptType], ...) -> list[EmbeddingRequestOutput]:
        """Generate embedding vectors for each prompt."""
        # 用于 embedding 模型
        ...
    
    def classify(self, prompts: ...) -> list[ClassificationRequestOutput]:
        """Generate class logits for each prompt."""
        # 用于分类模型
        ...
```

#### API 服务器 (`vllm/entrypoints/openai/api_server.py`)

提供 OpenAI 兼容的 HTTP API：

```python
# 简化的 API 服务器结构

app = FastAPI()

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """处理 Chat Completion 请求"""
    # 1. 验证请求
    # 2. 转换为内部格式
    # 3. 调用 AsyncLLMEngine
    # 4. 返回响应（支持流式）
    ...

@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """处理 Text Completion 请求"""
    ...

@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingRequest):
    """处理 Embedding 请求"""
    ...
```

### 2.2 `engine/` - 推理引擎（Legacy）

V0 版本的引擎实现（现已重定向到 V1）：

```
engine/
├── __init__.py             # 导出 LLMEngine
├── llm_engine.py           # ⚠️ 现在导入自 v1
├── async_llm_engine.py     # 异步引擎包装器
├── arg_utils.py            # EngineArgs 参数解析
└── protocol.py             # 协议定义
```

**当前状态**：`engine/llm_engine.py` 实际上从 V1 导入：
```python
# vllm/engine/llm_engine.py (当前)
from vllm.v1.engine.llm_engine import LLMEngine

# 这意味着 from vllm.engine import LLMEngine 
# 实际获取的是 V1 版本的引擎
```

### 2.3 `v1/` - V1 新架构 ⭐⭐⭐

vLLM 的新一代架构，是当前的默认实现：

```
v1/
├── engine/                      # V1 引擎
│   ├── __init__.py              # 导出 EngineCoreRequest 等
│   ├── llm_engine.py            # ⭐ LLMEngine 主类
│   ├── core_client.py           # 引擎核心客户端
│   ├── input_processor.py       # 输入处理器
│   ├── output_processor.py      # 输出处理器
│   ├── parallel_sampling.py     # 并行采样支持 (n>1)
│   └── async_llm_engine.py      # 异步引擎
│
├── worker/                      # Worker 实现
│   ├── gpu_model_runner.py      # ⭐ GPU 模型运行器 (核心)
│   ├── gpu_worker.py            # GPU Worker
│   ├── gpu_input_batch.py       # 输入批次管理
│   ├── cpu_model_runner.py      # CPU 模型运行器
│   ├── worker_base.py           # Worker 基类
│   ├── lora_model_runner_mixin.py # LoRA 支持
│   └── ...
│
├── core/                        # 核心调度逻辑
│   ├── sched/                   # 调度器
│   │   ├── scheduler.py         # 调度器实现
│   │   └── output.py            # 调度输出
│   └── kv_cache_manager.py      # KV Cache 管理
│
├── attention/                   # V1 注意力
│   └── backends/                # 注意力后端
│       ├── flash_attn.py        # FlashAttention
│       ├── flashinfer.py        # FlashInfer
│       ├── triton_attn.py       # Triton 实现
│       ├── flex_attention.py    # Flex Attention
│       └── utils.py             # 工具函数
│
├── sample/                      # 采样器
│   ├── sampler.py               # 采样实现
│   ├── metadata.py              # 采样元数据
│   ├── logits_processor/        # Logits 处理器
│   └── rejection_sampler.py     # 拒绝采样（投机解码用）
│
├── spec_decode/                 # 投机解码
│   ├── eagle.py                 # EAGLE 投机解码
│   ├── medusa.py                # Medusa 投机解码
│   ├── ngram_proposer.py        # N-gram 提议器
│   └── suffix_decoding.py       # 后缀解码
│
├── kv_cache_interface.py        # KV Cache 接口
├── kv_offload/                  # KV Cache 卸载
├── outputs.py                   # 输出定义
├── request.py                   # 请求定义
└── metrics/                     # 指标收集
```

#### V1 LLMEngine 详解 (`vllm/v1/engine/llm_engine.py`)

```python
# vllm/v1/engine/llm_engine.py (简化版)

class LLMEngine:
    """V1 LLMEngine - 当前推荐的推理引擎实现。"""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        ...
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        
        # 1. 初始化 Tokenizer
        if not self.model_config.skip_tokenizer_init:
            tokenizer = cached_tokenizer_from_config(self.model_config)
        
        # 2. 创建输入处理器
        self.input_processor = InputProcessor(self.vllm_config, tokenizer)
        
        # 3. 创建输出处理器（负责 detokenization）
        self.output_processor = OutputProcessor(
            self.tokenizer,
            log_stats=self.log_stats,
        )
        
        # 4. 创建引擎核心客户端
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
        )

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs, ...) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""
        # 1. 从 engine_args 创建 VllmConfig
        vllm_config = engine_args.create_engine_config(usage_context)
        
        # 2. 获取执行器类
        executor_class = Executor.get_class(vllm_config)
        
        # 3. 创建引擎
        return cls(vllm_config=vllm_config, executor_class=executor_class, ...)

    def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | PromptType,
        params: SamplingParams | PoolingParams,
        ...
    ) -> None:
        """Add a request to the engine."""
        # 1. 处理原始输入
        if isinstance(prompt, EngineCoreRequest):
            request = prompt
        else:
            request = self.input_processor.process_inputs(
                request_id, prompt, params, ...
            )
        
        # 2. 添加到输出处理器（用于跟踪）
        self.output_processor.add_request(request, ...)
        
        # 3. 添加到引擎核心
        self.engine_core.add_request(request)

    def step(self) -> list[RequestOutput | PoolingRequestOutput]:
        """Perform one decoding iteration."""
        # 1. 从引擎核心获取输出
        outputs = self.engine_core.get_output()
        
        # 2. 处理输出（detokenization 等）
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            ...
        )
        
        # 3. 中止已完成的请求
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)
        
        return processed_outputs.request_outputs
```

#### GPUModelRunner 详解 (`vllm/v1/worker/gpu_model_runner.py`)

这是实际执行模型推理的核心类：

```python
# vllm/v1/worker/gpu_model_runner.py (简化版)

class GPUModelRunner:
    """GPU Model Runner - 在 GPU 上执行模型推理"""
    
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        
        # 模型相关
        self.model: nn.Module  # 在 load_model() 中设置
        
        # KV Cache
        self.kv_caches: list[torch.Tensor] = []
        
        # 采样器
        self.sampler = Sampler(...)
        
        # 投机解码（如果启用）
        if self.speculative_config:
            self.drafter = ...  # EAGLE/Medusa/NGram
            self.rejection_sampler = RejectionSampler(self.sampler)
        
        # 请求状态缓存
        self.requests: dict[str, CachedRequestState] = {}
        
        # 输入批次管理
        self.input_batch = InputBatch(...)
        
        # 预分配的 GPU 缓冲区
        self.input_ids = torch.zeros(max_num_tokens, dtype=torch.int32, device=device)
        self.positions = torch.zeros(max_num_tokens, dtype=torch.int64, device=device)
        ...

    def load_model(self) -> None:
        """Load the model onto the device."""
        loader = get_model_loader(self.load_config)
        self.model = loader.load_model(self.vllm_config)
        
        # 设置 LoRA（如果有）
        if self.lora_config:
            self.set_lora_state(...)

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        """Execute model forward pass and sampling.
        
        这是推理的核心方法，每个 step 调用一次。
        """
        # 1. 更新内部状态
        self._update_states(scheduler_output)
        
        # 2. 准备输入
        num_scheduled_tokens = np.array([
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in self.input_batch.req_id_to_index
        ])
        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens,
        )
        
        # 3. 构建注意力元数据
        attn_metadata = self._prepare_attention_metadata(...)
        
        # 4. 执行模型前向传播
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=self.input_ids[:total_num_tokens],
                positions=self.positions[:total_num_tokens],
                intermediate_tensors=intermediate_tensors,
                ...
            )
        
        # 5. 计算 logits
        selected_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(selected_hidden_states)
        
        # 6. 采样
        sampling_metadata = self._prepare_sampling_metadata(...)
        
        if spec_decode_metadata is not None:
            # 投机解码：使用拒绝采样
            sampler_output = self.rejection_sampler(
                spec_decode_metadata,
                draft_probs,
                logits,
                sampling_metadata,
            )
        else:
            # 普通采样
            sampler_output = self.sampler(logits, sampling_metadata)
        
        # 7. 返回结果
        return ModelRunnerOutput(
            sampled_token_ids=sampler_output.sampled_token_ids,
            logprobs=sampler_output.logprobs,
            ...
        )

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        """准备模型输入张量"""
        # 填充 input_ids, positions 等
        ...

    def _prepare_attention_metadata(self, ...):
        """准备注意力元数据（用于 PagedAttention）"""
        # 包括 block table, sequence lengths 等
        ...
```

### 2.4 `model_executor/` - 模型执行器 ⭐⭐⭐

这是整个推理框架的核心，包含模型定义和执行逻辑：

```
model_executor/
├── models/                      # 🔴 所有支持的模型实现
│   ├── llama.py                 # ⭐ Llama 模型
│   ├── qwen2.py                 # ⭐ Qwen2 模型
│   ├── mixtral.py               # Mixtral MoE 模型
│   ├── deepseek_v2.py           # DeepSeek V2
│   ├── gpt2.py                  # GPT-2
│   ├── phi3.py                  # Phi-3
│   ├── gemma.py                 # Gemma
│   ├── mamba.py                 # Mamba (状态空间模型)
│   ├── qwen2_vl.py              # Qwen2-VL (视觉语言)
│   ├── llava.py                 # LLaVA (视觉语言)
│   ├── whisper.py               # Whisper (音频)
│   ├── registry.py              # ⭐ 模型注册表
│   ├── interfaces.py            # 模型接口定义
│   ├── interfaces_base.py       # 基础接口
│   └── utils.py                 # 模型工具函数
│   └── ...（200+ 模型文件）
│
├── layers/                      # 🔴 模型层实现
│   ├── linear.py                # ⭐ 线性层（含量化支持）
│   ├── activation.py            # 激活函数 (SiLU, GELU, etc.)
│   ├── layernorm.py             # LayerNorm 实现
│   ├── vocab_parallel_embedding.py  # 词嵌入层
│   ├── logits_processor.py      # Logits 处理器
│   ├── sampler.py               # 采样层
│   ├── pooler.py                # 池化层
│   │
│   ├── rotary_embedding/        # RoPE 位置编码
│   │   ├── __init__.py          # 导出 get_rope()
│   │   └── base.py              # RotaryEmbedding 实现
│   │
│   ├── fused_moe/               # 融合 MoE 层
│   │   ├── layer.py             # FusedMoE 主类
│   │   ├── fused_moe.py         # 融合内核调用
│   │   └── config.py            # MoE 配置
│   │
│   └── quantization/            # 🔴 量化实现
│       ├── __init__.py          # 导出量化方法
│       ├── base_config.py       # QuantizationConfig 基类
│       ├── fp8.py               # ⭐ FP8 量化
│       ├── awq.py               # AWQ 量化
│       ├── awq_marlin.py        # AWQ Marlin 格式
│       ├── gptq.py              # GPTQ 量化
│       ├── gptq_marlin.py       # GPTQ Marlin 格式（高效 GPTQ）
│       ├── bitsandbytes.py      # BitsAndBytes 量化
│       ├── gguf.py              # GGUF 格式支持
│       ├── compressed_tensors/  # CompressedTensors 支持
│       └── utils/               # 量化工具
│           ├── fp8_utils.py     # FP8 工具
│           ├── w8a8_utils.py    # W8A8 工具
│           └── marlin_utils.py  # Marlin 工具
│
├── model_loader/                # 模型加载器
│   ├── loader.py                # 主加载器
│   ├── weight_utils.py          # 权重工具
│   └── tensorizer.py            # Tensorizer 支持
│
├── custom_op.py                 # CustomOp 基类
├── parameter.py                 # 参数定义
├── guided_decoding/             # 引导解码
└── utils.py                     # 工具函数
```

#### 模型注册表 (`registry.py`)

vLLM 使用注册表模式管理支持的模型：

```python
# vllm/model_executor/models/registry.py

# 支持的模型列表（部分）
_TEXT_GENERATION_MODELS = {
    # 语言模型
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "MistralForCausalLM": ("llama", "LlamaForCausalLM"),  # 使用 Llama 实现
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "Phi3ForCausalLM": ("phi3", "Phi3ForCausalLM"),
    "GemmaForCausalLM": ("gemma", "GemmaForCausalLM"),
    
    # 视觉语言模型
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    
    # 嵌入模型
    "BertModel": ("bert", "BertEmbeddingModel"),
    
    # 状态空间模型
    "MambaForCausalLM": ("mamba", "MambaForCausalLM"),
    
    # ... 200+ 其他模型
}

def get_model_architecture(config) -> tuple[str, str]:
    """Get the module and class name for a model config."""
    architectures = getattr(config, "architectures", [])
    for arch in architectures:
        if arch in _TRANSFORMERS_MODELS:
            return _TRANSFORMERS_MODELS[arch]
    raise ValueError(f"Model architecture {architectures} not supported")
```

### 2.5 `attention/` - 注意力机制 ⭐⭐⭐

```
attention/
├── __init__.py              # 导出 Attention 类
├── layer.py                 # ⭐ Attention 层封装
├── selector.py              # 后端自动选择器
├── ops/                     # 注意力操作
│   ├── paged_attn.py        # PagedAttention 操作
│   └── prefix_prefill.py    # 前缀预填充
│
├── backends/                # 注意力后端实现
│   ├── abstract.py          # 抽象基类
│   ├── registry.py          # 后端注册表
│   └── utils.py             # 工具函数
│
├── layers/                  # 特殊注意力层
│   └── encoder_only_attention.py # 仅编码器注意力
│
└── utils/                   # 工具
```

**注意**: V1 架构的注意力后端位于 `vllm/v1/attention/backends/`，包含：
- `flash_attn.py` - FlashAttention
- `flashinfer.py` - FlashInfer  
- `triton_attn.py` - Triton 实现
- `flex_attention.py` - Flex Attention
- 以及 ROCm、TPU 等平台特定的实现

#### Attention 层 (`layer.py`)

```python
# vllm/attention/layer.py (简化版)

class Attention(nn.Module):
    """Multi-head attention layer with paged attention support."""
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        prefix: str = "",
        **kwargs,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        
        # 根据配置选择最佳注意力后端
        self.impl = get_attn_backend(
            num_heads=num_heads,
            head_size=head_size,
            num_kv_heads=self.num_kv_heads,
            dtype=cache_config.dtype if cache_config else torch.float16,
            **kwargs,
        )
    
    def forward(
        self,
        query: torch.Tensor,           # [num_tokens, num_heads * head_size]
        key: torch.Tensor,             # [num_tokens, num_kv_heads * head_size]
        value: torch.Tensor,           # [num_tokens, num_kv_heads * head_size]
        kv_cache: torch.Tensor | None, # KV Cache 张量
        attn_metadata: AttentionMetadata,  # 注意力元数据
    ) -> torch.Tensor:
        """Forward pass with paged attention."""
        return self.impl.forward(
            query, key, value, kv_cache, attn_metadata, self.k_scale, self.v_scale
        )
```

#### 注意力后端选择 (`selector.py`)

vLLM 自动选择最佳的注意力后端：

```python
# 后端选择优先级（简化）
def get_attn_backend(...) -> AttentionBackend:
    """Select the best attention backend for the current configuration."""
    
    # 1. FlashInfer (如果可用且合适)
    if is_flashinfer_available() and ...:
        return FlashInferBackend(...)
    
    # 2. FlashAttention (最常用)
    if is_flash_attn_available() and head_size in [64, 80, 96, 128, 256]:
        return FlashAttentionBackend(...)
    
    # 3. xFormers (备选)
    if is_xformers_available():
        return XFormersBackend(...)
    
    # 4. PyTorch SDPA (fallback)
    return TorchSDPABackend(...)
```

### 2.6 `config/` - 配置类

所有配置相关的定义：

```
config/
├── __init__.py             # 导出所有配置类
├── vllm.py                 # ⭐ VllmConfig 主配置
├── model.py                # ModelConfig 模型配置
├── cache.py                # CacheConfig KV Cache 配置
├── parallel.py             # ParallelConfig 并行配置
├── scheduler.py            # SchedulerConfig 调度器配置
├── device.py               # DeviceConfig 设备配置
├── lora.py                 # LoRAConfig LoRA 配置
├── speculative.py          # SpeculativeConfig 投机解码配置
├── compilation.py          # CompilationConfig 编译配置
└── ...
```

#### VllmConfig (`config/vllm.py`)

```python
# vllm/config/vllm.py

@dataclass
class VllmConfig:
    """Top-level configuration for vLLM."""
    
    model_config: ModelConfig           # 模型相关配置
    cache_config: CacheConfig           # KV Cache 配置
    parallel_config: ParallelConfig     # 并行配置
    scheduler_config: SchedulerConfig   # 调度器配置
    device_config: DeviceConfig         # 设备配置
    load_config: LoadConfig             # 加载配置
    lora_config: LoRAConfig | None      # LoRA 配置
    multimodal_config: MultiModalConfig | None  # 多模态配置
    speculative_config: SpeculativeConfig | None  # 投机解码配置
    observability_config: ObservabilityConfig     # 可观测性配置
    compilation_config: CompilationConfig         # 编译配置

# ModelConfig 示例
@dataclass
class ModelConfig:
    model: str                          # 模型路径或 HuggingFace ID
    tokenizer: str | None               # Tokenizer 路径
    dtype: torch.dtype                  # 数据类型
    trust_remote_code: bool             # 是否信任远程代码
    max_model_len: int                  # 最大上下文长度
    quantization: str | None            # 量化方法
    revision: str | None                # 模型版本
    ...
```

### 2.7 `distributed/` - 分布式支持

```
distributed/
├── __init__.py              # 导出分布式工具
├── parallel_state.py        # ⭐ 并行状态管理
├── communication_op.py      # 通信操作
├── utils.py                 # 工具函数
│
├── kv_transfer/             # KV Cache 传输（用于分离式推理）
│   ├── kv_connector/        # KV 连接器
│   └── ...
│
└── eplb/                    # 专家并行负载均衡
```

#### 并行状态管理 (`parallel_state.py`)

```python
# vllm/distributed/parallel_state.py

# 全局并行状态
_TENSOR_MODEL_PARALLEL_GROUP = None
_PIPELINE_MODEL_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP = None

def get_tensor_model_parallel_rank() -> int:
    """获取当前进程的张量并行 rank"""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return 0
    return torch.distributed.get_rank(_TENSOR_MODEL_PARALLEL_GROUP)

def get_tensor_model_parallel_world_size() -> int:
    """获取张量并行世界大小"""
    if _TENSOR_MODEL_PARALLEL_GROUP is None:
        return 1
    return torch.distributed.get_world_size(_TENSOR_MODEL_PARALLEL_GROUP)

def tensor_model_parallel_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
    """张量并行 all-reduce 操作"""
    if get_tensor_model_parallel_world_size() == 1:
        return tensor
    return all_reduce(tensor, group=_TENSOR_MODEL_PARALLEL_GROUP)
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

## 7. 小结与关键路径

### 7.1 架构层次总结

vLLM 的核心架构可以概括为六个层次：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 1: 入口层 (entrypoints/)                                          │
│  - LLM 类：离线推理                                                       │
│  - OpenAI API Server：在线服务                                           │
│  - CLI：命令行工具                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 2: 引擎层 (v1/engine/)                                            │
│  - LLMEngine：请求管理和生命周期                                          │
│  - InputProcessor：输入预处理和 tokenization                             │
│  - OutputProcessor：输出后处理和 detokenization                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 3: 调度层 (v1/core/)                                              │
│  - Scheduler：请求调度                                                    │
│  - KV Cache Manager：KV Cache 分配和管理                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 4: 执行层 (v1/worker/)                                            │
│  - GPUModelRunner：GPU 模型运行                                          │
│  - InputBatch：批次管理                                                   │
│  - Sampler：采样                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 5: 模型层 (model_executor/models/)                                │
│  - 200+ 模型实现                                                          │
│  - Transformer 层组装                                                     │
├─────────────────────────────────────────────────────────────────────────┤
│  Layer 6: 算子层 (model_executor/layers/, csrc/)                         │
│  - 线性层（含量化）                                                       │
│  - 注意力层                                                               │
│  - 激活函数、LayerNorm 等                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 关键文件速查

| 目的 | 文件路径 | 说明 |
|------|---------|------|
| **用户入口** | | |
| 离线推理 | `vllm/entrypoints/llm.py` | LLM 类 |
| 在线服务 | `vllm/entrypoints/openai/api_server.py` | API 服务器 |
| CLI | `vllm/entrypoints/cli/main.py` | 命令行入口 |
| **引擎核心** | | |
| V1 引擎 | `vllm/v1/engine/llm_engine.py` | LLMEngine |
| 核心客户端 | `vllm/v1/engine/core_client.py` | EngineCoreClient |
| **执行器** | | |
| GPU 执行 | `vllm/v1/worker/gpu_model_runner.py` | GPUModelRunner |
| 采样器 | `vllm/v1/sample/sampler.py` | Sampler |
| **模型定义** | | |
| Llama | `vllm/model_executor/models/llama.py` | LlamaForCausalLM |
| Qwen2 | `vllm/model_executor/models/qwen2.py` | Qwen2ForCausalLM |
| 模型注册 | `vllm/model_executor/models/registry.py` | 模型注册表 |
| **层实现** | | |
| 线性层 | `vllm/model_executor/layers/linear.py` | Linear 层 |
| FP8 量化 | `vllm/model_executor/layers/quantization/fp8.py` | FP8 实现 |
| 注意力 | `vllm/attention/layer.py` | Attention 层 |
| **配置** | | |
| 主配置 | `vllm/config/vllm.py` | VllmConfig |
| 参数解析 | `vllm/engine/arg_utils.py` | EngineArgs |

### 7.3 核心数据流

```
用户输入 (prompts)
    │
    ▼
┌─────────────────────────────────────┐
│ Tokenization (InputProcessor)       │
│ "Hello" → [15496, 995]              │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Scheduling (Scheduler)              │
│ - 请求排队                           │
│ - KV Cache 分配                     │
│ - 批次组织                          │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Model Forward (GPUModelRunner)      │
│ 1. Embedding                        │
│ 2. N × Decoder Layer               │
│    - Attention (qkv → attn → o)    │
│    - MLP (gate_up → act → down)    │
│ 3. Final Norm                       │
│ 4. LM Head → Logits                │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Sampling (Sampler)                  │
│ Logits → Token IDs                  │
│ [3.2, 1.5, ...] → [15496]          │
└───────────────────┬─────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│ Detokenization (OutputProcessor)    │
│ [15496, 995, ...] → "Hello world"  │
└─────────────────────────────────────┘
```

### 7.4 性能优化关键点

| 优化技术 | 位置 | 说明 |
|---------|------|------|
| **PagedAttention** | `vllm/attention/` | KV Cache 分页管理 |
| **连续批处理** | `vllm/v1/core/sched/` | 动态请求调度 |
| **CUDA Graph** | `vllm/compilation/` | 减少内核启动开销 |
| **量化推理** | `vllm/model_executor/layers/quantization/` | FP8/AWQ/GPTQ |
| **张量并行** | `vllm/distributed/` | 多 GPU 推理 |
| **FlashAttention** | `vllm/attention/backends/` | 高效注意力计算 |
| **投机解码** | `vllm/v1/spec_decode/` | 加速生成 |
| **前缀缓存** | `vllm/v1/core/` | 共享前缀 KV Cache |

### 7.5 二次开发指南

**添加新模型**：
1. 在 `vllm/model_executor/models/` 创建模型文件
2. 继承适当的基类（如 `nn.Module`）
3. 在 `registry.py` 注册模型
4. 实现 `forward()` 和 `compute_logits()` 方法

**添加新量化方法**：
1. 在 `vllm/model_executor/layers/quantization/` 创建文件
2. 继承 `QuantizationConfig` 和 `QuantizeMethodBase`
3. 实现 `create_weights()` 和 `apply()` 方法
4. 在 `__init__.py` 注册

**修改线性层 GEMM**：
1. 查看 `vllm/model_executor/layers/linear.py`
2. 修改 `UnquantizedLinearMethod.apply()` 或创建新的 LinearMethod
3. 对于 CUDA kernel，修改 `csrc/` 下的相关文件

---

## 8. 扩展阅读

- **线性层与 GEMM 详解** → [framework_lineargemm.md](./framework_lineargemm.md)
- **项目整体结构** → [framework_overview.md](./framework_overview.md)
- **官方文档** → https://docs.vllm.ai/en/stable/
- **PagedAttention 论文** → https://arxiv.org/abs/2309.06180
