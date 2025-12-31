# SlideSparse 集成 vLLM 框架详解

本文档详细阐述 SlideSparse 稀疏加速方法的理论原理、工具开发流程、以及在 vLLM 框架中的工程实现方案。本文档旨在为后续的开发工作提供完整的技术指导和实施路线图。

---

## 目录

1. [SlideSparse 理论概述与创新点](#1-slidesparse-理论概述与创新点)
2. [工具开发流程](#2-工具开发流程)
3. [工程实现流程](#3-工程实现流程)
4. [关键工作：Model Loader 权重加载](#4-关键工作model-loader-权重加载)
5. [关键工作：Kernel 替换与前向传播](#5-关键工作kernel-替换与前向传播)
6. [测试与验证](#6-测试与验证)
7. [附录：关键文件路径速查表](#7-附录关键文件路径速查表)

---

## 1. SlideSparse 理论概述与创新点

### 1.1 背景与动机

在大语言模型（LLM）推理中，**GEMM（通用矩阵乘法）占据了约 70-80% 的计算时间**。稀疏计算是加速 GEMM 的重要途径之一。NVIDIA 的 Ampere 及后续架构 GPU 提供了 2:4 结构化稀疏的硬件加速支持，可以实现理论上 2 倍的计算吞吐提升。

然而，现有的 2:4 稀疏方案存在以下限制：
- **稀疏度固定**：必须严格满足每 4 个元素中有 2 个为零（50% 稀疏度）
- **精度损失**：强制剪枝可能导致较大的精度下降
- **灵活性不足**：无法适应不同稀疏度的模型

**SlideSparse** 提出了一种创新的解决方案，通过权重和激活的重排策略，使**更低稀疏度、更粗粒度**的稀疏模型也能利用 2:4 结构化稀疏硬件加速。

### 1.2 SlideSparse 核心原理

#### 1.2.1 松弛结构化稀疏 (Relaxed Structured Sparse)

SlideSparse 定义了一种更宽松的稀疏格式：**Z:L' 稀疏**，其中：
- L' = L + k×(L-Z)，k = 0, 1, 2, 3...
- L 是硬件支持的稀疏窗口大小（如 2:4 中的 4）
- Z 是窗口内的零元素个数（如 2:4 中的 2）
- k 是滑框次数

例如，2:8 稀疏（25% 稀疏度）可以通过 SlideSparse 映射到 2:4 硬件进行加速。

#### 1.2.2 重叠滑动窗口机制 (Overlapping Sliding Windows)

SlideSparse 的核心机制是通过**重叠滑动窗口**将原始权重矩阵分解为多个子权重：

```
原始权重序列：    1  2  3  4  5  6  7  8
                 ↓  重叠滑动窗口分解
拓展后的序列：    1  2  3  4 | 3  4  5  6 | 5  6  7  8
                 └─窗口1─┘   └─窗口2─┘   └─窗口3─┘

窗口参数：
- 窗口大小 (Window Size) = L = 4
- 滑框步长 (Stride) = L - Z = 2
- 重叠宽度 (Overlap) = Z = 2
```

#### 1.2.3 贪心残差分配 (Greedy Residual Allocation)

SlideSparse 采用贪心残差分配策略，确保：
1. 所有零元素被完全覆盖
2. 非零元素被分配到符合 Z:L 稀疏格式的位置
3. 最小化拓展后的总长度

**示例分析**（2:8 稀疏 → 2:4 硬件）：
```
原始稀疏：2:8（25% 稀疏度，每 8 个元素有 2 个零，6 个非零）
所需窗口数：⌈6/2⌉ = 3 组 2:4 稀疏
拓展总长度：3 × 4 = 12
预期延时：(6/2 × 4) / 2 / 8 = 75%（相比 dense 计算）
```

#### 1.2.4 通用性公式

对于硬件支持 Z:L 稀疏加速（L 个元素中至少 Z 个零，最多 N 个非零，Z + N = L）：

| 参数 | 公式 | 说明 |
|------|------|------|
| 加速比 | L / N | 如 2:4 → 2x，3:4 → 4x |
| 支持的稀疏格式 | Z:L'，L' = L + k×N | k = 0, 1, 2, 3... |
| 窗口大小 | L（或 L 的因数） | |
| 滑框步长 | N = L - Z | |
| 重叠宽度 | Z | |
| 延时比例 | (L' - Z) / L' | |

### 1.3 SlideSparse 创新点总结

1. **任意稀疏度适配**：打破 2:4 稀疏的 50% 稀疏度限制，支持 2:4, 2:6, 2:8, 2:10, 2:12 等多种稀疏格式，以及 1:2, 1:3, 1:4, 1:5 等更细粒度的稀疏

2. **理论最优利用率**：确保 X% 的稀疏度一定能减少 X% 的计算延时，完全享受稀疏度带来的"零值跳跃"吞吐收益

3. **硬件兼容性**：无需修改硬件，复用现有的 NVIDIA 2:4 稀疏硬件加速单元

4. **端到端加速**：通过算子融合（fused_quant_slide + sparse_GEMM + fused_dequant_transpose），最小化额外开销

### 1.4 SlideSparse 与 vLLM 集成的目标

本项目的目标是将 SlideSparse 集成到 vLLM 推理框架中，实现：

1. **吞吐量提升**：在 Prefill 和 Decode 阶段验证 token/s 的提升百分比
2. **模型支持**：支持 Qwen3、Llama3.2 等 dense 小模型的端到端加速
3. **量化兼容**：支持 FP8 和 INT8 两种 GEMM 精度
4. **规模覆盖**：测试 ~1B、~3B、~7B、~14B 四种典型模型尺寸

**测试模型计划**：
- FP8：Llama3.2-1B、Llama3.2-3B、Qwen3-8B、Qwen3-14B
- INT8：需寻找 W8A8 版本或自行量化

---

## 2. 工具开发流程

SlideSparse 的实现需要开发一系列离线和在线工具。本章节详细梳理每个工具的功能、输入输出和实现要点。

### 2.1 工具概览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              离线工具链 (Offline)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐ │
│   │ 权重 Prune   │ ──► │ 权重 Slide   │ ──► │ 权重压缩     │ ──► │ 算法搜索  │ │
│   │   (Python)   │     │   (Python)   │     │   (CUDA)     │     │ (Python)  │ │
│   │              │     │              │     │              │     │           │ │
│   │ Dense [N,K]  │     │ Sparse [N,K] │     │ Slided [N,K']│     │ json 配置 │ │
│   │    ↓         │     │    ↓         │     │    ↓         │     │           │ │
│   │ Sparse [N,K] │     │ Slided [N,K']│     │ Compressed   │     │           │ │
│   └──────────────┘     └──────────────┘     └──────────────┘     └───────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              在线工具链 (Online)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────────────┐     ┌──────────────────┐     ┌────────────────────┐ │
│   │ Fused Quant + Slide  │ ──► │ Sparse GEMM      │ ──► │ Fused Transpose +  │ │
│   │      (Triton)        │     │    (CUDA)        │     │ Dequant (Triton)   │ │
│   │                      │     │                  │     │                    │ │
│   │ BF16 [M,K]           │     │ FP8/INT8 [M,K']  │     │ INT32/FP32 [N,M]   │ │
│   │    ↓                 │     │    ↓             │     │    ↓               │ │
│   │ FP8/INT8 [M,K']      │     │ INT32/FP32 [N,M] │     │ BF16 [M,N]         │ │
│   └──────────────────────┘     └──────────────────┘     └────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 离线工具详解

#### 2.2.1 权重剪枝工具 (Weight Pruning)

**功能**：将 Dense 权重剪枝为指定稀疏度的结构化稀疏权重

**实现语言**：Python

**输入**：
- Dense 权重张量：`[N, K]` 维度（N 为输出维度，K 为输入维度）
- 稀疏粒度参数：`2:2N`（每 2N 个连续元素中至少有 2 个零）
- 剪枝模式：`random` 或 `magnitude`

**输出**：
- 稀疏权重张量：`[N, K]` 维度，满足指定的结构化稀疏约束

**剪枝策略**：
```python
# magnitude 模式：按绝对值大小剪枝
def magnitude_prune(weight, sparsity_ratio, group_size):
    """
    对权重进行基于幅度的结构化剪枝
    
    Args:
        weight: [N, K] 权重张量
        sparsity_ratio: 目标稀疏度（如 0.25 表示 2:8）
        group_size: 稀疏组大小（如 8 表示 2:8）
    
    Returns:
        pruned_weight: 剪枝后的权重
        mask: 稀疏掩码
    """
    N, K = weight.shape
    # 确保 K 能被 group_size 整除
    assert K % group_size == 0
    
    # 重塑为 [N, K/group_size, group_size]
    weight_grouped = weight.view(N, -1, group_size)
    
    # 计算每组中的 top-k 非零位置
    num_zeros_per_group = int(sparsity_ratio * group_size)
    num_nonzeros = group_size - num_zeros_per_group
    
    # 按绝对值排序，保留最大的 num_nonzeros 个
    _, indices = torch.topk(weight_grouped.abs(), num_nonzeros, dim=-1)
    
    # 生成掩码
    mask = torch.zeros_like(weight_grouped)
    mask.scatter_(-1, indices, 1)
    
    # 应用掩码
    pruned_weight = weight_grouped * mask
    return pruned_weight.view(N, K), mask.view(N, K)
```

**需要处理的线性层**：
| 线性层名称 | 模型中的位置 | 典型维度 |
|-----------|------------|---------|
| Wqkv | `self_attn.qkv_proj` | `[3*H, H]` 或 `[(Q+2*KV), H]` |
| Wo | `self_attn.o_proj` | `[H, H]` |
| W13 | `mlp.gate_up_proj` | `[2*I, H]` |
| W2 | `mlp.down_proj` | `[H, I]` |

#### 2.2.2 权重滑动工具 (Weight Sliding)

**功能**：将稀疏权重通过滑动窗口机制拓展为符合 2:4 硬件格式的权重

**实现语言**：Python

**输入**：
- 稀疏权重张量：`[N, K]`，满足 2:L 稀疏约束
- 目标硬件格式：2:4

**输出**：
- 滑动后的权重张量：`[N, K']`，其中 `K' = K * expand_ratio`

**核心算法**：
```python
def slide_weight(sparse_weight, src_sparsity=(2, 8), tgt_sparsity=(2, 4)):
    """
    将稀疏权重进行滑动拓展
    
    Args:
        sparse_weight: [N, K] 稀疏权重
        src_sparsity: (Z, L') 源稀疏格式
        tgt_sparsity: (Z, L) 目标硬件稀疏格式
    
    Returns:
        slided_weight: [N, K'] 滑动后的权重
    """
    Z, L_src = src_sparsity
    Z_tgt, L_tgt = tgt_sparsity
    
    N, K = sparse_weight.shape
    
    # 确保 K 能被 L_src 整除
    if K % L_src != 0:
        # 需要先 padding
        pad_size = L_src - (K % L_src)
        sparse_weight = F.pad(sparse_weight, (0, pad_size))
        K = K + pad_size
    
    # 计算拓展参数
    stride = L_tgt - Z_tgt  # 滑动步长 = 非零元素个数
    num_windows = (L_src - Z) // stride  # 窗口数 = 非零元素数 / 每窗口非零数
    expand_length = num_windows * L_tgt
    
    # 重塑为组
    weight_grouped = sparse_weight.view(N, -1, L_src)  # [N, num_groups, L_src]
    num_groups = weight_grouped.shape[1]
    
    # 滑动拓展
    slided_groups = []
    for i in range(num_windows):
        start = i * stride
        end = start + L_tgt
        slided_groups.append(weight_grouped[:, :, start:end])
    
    # 拼接：[N, num_groups, num_windows * L_tgt]
    slided_weight = torch.cat(slided_groups, dim=-1)
    slided_weight = slided_weight.view(N, -1)
    
    return slided_weight
```

**Padding 策略**：
- K 首先需要被 padding 到能够被 `4 * L` 整除
- 这是 cuSPARSELt 库的对齐要求

#### 2.2.3 权重压缩工具 (Weight Compression)

**功能**：调用 cuSPARSELt 库将 2:4 稀疏权重压缩为硬件可识别的格式

**实现语言**：CUDA / Python (调用 cuSPARSELt)

**输入**：
- 滑动后的权重：`[N, K']`，符合 2:4 稀疏格式
- 数据类型：FP8 / INT8

**输出**：
- 压缩后的权重：`CompressedTensor` 类型
- 元数据：压缩索引信息

**关键 API**：
```python
import torch
import cusparselt

def compress_weight_cusparselt(weight, dtype=torch.float8_e4m3fn):
    """
    使用 cuSPARSELt 压缩 2:4 稀疏权重
    
    Args:
        weight: [N, K] 2:4 稀疏权重
        dtype: 目标数据类型
    
    Returns:
        compressed_weight: 压缩后的权重张量
        metadata: 压缩元数据
    """
    # 转换数据类型
    weight = weight.to(dtype).contiguous()
    
    # 调用 cuSPARSELt 压缩
    # cuSPARSELt 要求权重为列主序 [K, N]
    weight_col_major = weight.t().contiguous()
    
    compressed_weight, metadata = cusparselt.compress_sparse_matrix(
        weight_col_major, 
        sparsity_type='2:4'
    )
    
    return compressed_weight, metadata
```

**存储格式**：
- 压缩后的权重需要注册为 `CompressedTensor` 类型
- 存储到 `.pt` 或 `.safetensors` 权重文件中
- 替换原有 Transformer Block 的线性层权重

#### 2.2.4 算法搜索工具 (Algorithm Tuning)

**功能**：离线搜索 cuSPARSELt GEMM 的最优算法 ID

**实现语言**：Python (调用 cuSPARSELt)

**输入**：
- 模型线性层的 `[N, K]` 尺寸信息
- 不同的 batch size `M` 范围

**输出**：
- JSON 配置文件：记录每个 `(M, N, K)` 组合的最优算法 ID

**搜索流程**：
```python
import json
import cusparselt

def search_optimal_algorithms(model_config, m_range, device='cuda'):
    """
    搜索最优的 cuSPARSELt 算法配置
    
    Args:
        model_config: 包含线性层 NK 尺寸的配置
        m_range: M 值的搜索范围
        device: 目标设备
    
    Returns:
        best_algorithms: {(M, N, K): best_algo_id} 字典
    """
    best_algorithms = {}
    
    for layer_name, (N, K) in model_config.items():
        for M in m_range:
            # 创建测试输入
            A = torch.randn(M, K, device=device, dtype=torch.float8_e4m3fn)
            B = torch.randn(N, K, device=device, dtype=torch.float8_e4m3fn)
            
            # 搜索所有算法
            best_time = float('inf')
            best_algo = 0
            
            for algo_id in range(cusparselt.get_num_algorithms()):
                try:
                    # 计时
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    
                    start.record()
                    for _ in range(100):
                        _ = cusparselt.sparse_mm(A, B, algo_id=algo_id)
                    end.record()
                    torch.cuda.synchronize()
                    
                    elapsed = start.elapsed_time(end) / 100
                    if elapsed < best_time:
                        best_time = elapsed
                        best_algo = algo_id
                except:
                    continue
            
            best_algorithms[(M, N, K)] = best_algo
    
    return best_algorithms

def save_algorithm_config(best_algorithms, output_path):
    """保存算法配置到 JSON 文件"""
    # 转换 key 为字符串
    config = {f"{m},{n},{k}": algo for (m, n, k), algo in best_algorithms.items()}
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
```

### 2.3 在线工具详解

#### 2.3.1 融合量化+滑动算子 (Fused Quant + Slide)

**功能**：将 BF16 输入激活进行在线量化（FP8/INT8）和滑动拓展

**实现语言**：Triton

**输入**：
- 输入激活：`[M, K]`，BF16 精度
- 量化 scale：动态计算或预设

**输出**：
- 量化+滑动后的激活：`[M, K']`，FP8/INT8 精度

**设计原则**：
- 将 slide 操作融合在 quant 流程中，掩盖额外的读写开销
- 利用 Triton 的灵活性实现自定义内存访问模式

**Triton Kernel 框架**：
```python
import triton
import triton.language as tl

@triton.jit
def fused_quant_slide_kernel(
    input_ptr,        # 输入指针
    output_ptr,       # 输出指针
    scale_ptr,        # scale 指针
    M, K, K_expanded, # 维度信息
    src_L: tl.constexpr,  # 源稀疏窗口大小
    tgt_L: tl.constexpr,  # 目标稀疏窗口大小
    stride: tl.constexpr,  # 滑动步长
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    融合的量化 + 滑动 Kernel
    
    流程：
    1. 加载 BF16 输入块
    2. 计算/加载 scale
    3. 量化为 FP8/INT8
    4. 按滑动模式重排写出
    """
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # 计算块的起始位置
    m_start = pid_m * BLOCK_M
    k_start = pid_k * BLOCK_K
    
    # 加载输入块
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k = k_start + tl.arange(0, BLOCK_K)
    
    mask_m = offs_m < M
    mask_k = offs_k < K
    mask = mask_m[:, None] & mask_k[None, :]
    
    # 加载 BF16 输入
    x = tl.load(input_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask)
    
    # 动态量化：计算 scale
    x_max = tl.max(tl.abs(x))
    scale = x_max / 448.0  # FP8 E4M3 最大值
    
    # 量化
    x_quant = (x / scale).to(tl.float8e4nv)
    
    # 滑动重排：计算输出位置
    # 这里需要根据滑动窗口逻辑计算新的偏移
    num_windows = src_L // stride
    
    for w in range(num_windows):
        src_start = w * stride
        dst_start = w * tgt_L
        
        # 选择当前窗口的数据
        window_offs = src_start + tl.arange(0, tgt_L)
        window_mask = window_offs < src_L
        
        # 计算输出偏移
        out_offs_k = k_start // src_L * (num_windows * tgt_L) + dst_start + tl.arange(0, tgt_L)
        
        # 写出
        # ... 实际实现需要更复杂的索引计算
    
    # 存储 scale
    tl.store(scale_ptr + pid_m, scale)
```

**性能考虑**：
- 使用 block-level 量化（per-token 或 per-block）减少精度损失
- 优化内存访问模式，尽量合并读写
- 利用 Triton autotune 搜索最优配置

#### 2.3.2 结构化稀疏 GEMM (Sparse GEMM)

**功能**：调用 cuSPARSELt 执行 2:4 结构化稀疏矩阵乘法

**实现语言**：CUDA (调用 cuSPARSELt)

**输入**：
- 量化+滑动后的激活：`[M, K']`，FP8/INT8
- 压缩后的权重：CompressedTensor
- 算法 ID：从 JSON 配置查表获取

**输出**：
- 乘法结果：`[N, M]`（行主序）

**API 限制说明**：
- cuSPARSELt 当前版本只能输出行主序的 `[N, M]` 结果
- 需要后续进行转置操作

**调用接口**：
```python
def sparse_gemm_cusparselt(
    activation,      # [M, K'] FP8/INT8
    weight,          # CompressedTensor
    algo_id,         # 最优算法 ID
    scale_a,         # 激活 scale
    scale_b,         # 权重 scale
):
    """
    调用 cuSPARSELt 执行结构化稀疏 GEMM
    
    Returns:
        output: [N, M] 行主序结果，INT32/FP32
    """
    # 查表获取算法 ID
    M, K = activation.shape
    N = weight.shape[0]
    
    # 调用 cuSPARSELt
    output = cusparselt.sparse_matmul(
        activation,
        weight,
        algorithm=algo_id,
        out_dtype=torch.int32 if activation.dtype == torch.int8 else torch.float32
    )
    
    return output  # [N, M]
```

#### 2.3.3 融合转置+反量化算子 (Fused Transpose + Dequant)

**功能**：将 GEMM 输出从 `[N, M]` 转置为 `[M, N]`，并反量化为 BF16

**实现语言**：Triton

**输入**：
- GEMM 输出：`[N, M]`，INT32/FP32
- Scale 参数：scale_a, scale_b

**输出**：
- 反量化结果：`[M, N]`，BF16

**Triton Kernel 框架**：
```python
@triton.jit
def fused_transpose_dequant_kernel(
    input_ptr,        # [N, M] 输入
    output_ptr,       # [M, N] 输出
    scale_a_ptr,      # 激活 scale
    scale_b_ptr,      # 权重 scale
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    融合的转置 + 反量化 Kernel
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算块位置
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # 从 [N, M] 读取（转置访问）
    # input[n, m] -> output[m, n]
    x = tl.load(
        input_ptr + offs_n[None, :] * M + offs_m[:, None],
        mask=mask_n[None, :] & mask_m[:, None]
    )
    
    # 加载 scales
    scale_a = tl.load(scale_a_ptr)
    scale_b = tl.load(scale_b_ptr)
    
    # 反量化
    x_dequant = x.to(tl.float32) * scale_a * scale_b
    x_bf16 = x_dequant.to(tl.bfloat16)
    
    # 写出 [M, N]
    tl.store(
        output_ptr + offs_m[:, None] * N + offs_n[None, :],
        x_bf16,
        mask=mask
    )
```

### 2.4 工具开发检查清单

| 工具 | 语言 | 输入 | 输出 | 状态 |
|------|------|------|------|------|
| 权重剪枝 | Python | Dense `[N,K]` | Sparse `[N,K]` | 待开发 |
| 权重滑动 | Python | Sparse `[N,K]` | Slided `[N,K']` | 待开发 |
| 权重压缩 | CUDA | Slided `[N,K']` | Compressed | 待开发 |
| 算法搜索 | Python | `(M,N,K)` | JSON config | 待开发 |
| Fused Quant+Slide | Triton | BF16 `[M,K]` | FP8/INT8 `[M,K']` | 待开发 |
| Sparse GEMM | CUDA | `[M,K']` + Compressed | `[N,M]` | 待开发 |
| Fused Transpose+Dequant | Triton | `[N,M]` INT32/FP32 | `[M,N]` BF16 | 待开发 |

---

## 3. 工程实现流程

本章节详细描述基于 vLLM 框架实现 SlideSparse 端到端推理加速的工程流程。

### 3.1 环境准备阶段（已完成）

#### 3.1.1 代码框架和 Docker 环境

**已完成的工作**：
- Fork vLLM 0.13.0 稳定分支作为本地 main 分支
- 基于 `vllm/vllm-openai:v0.13.0` Docker 镜像（Ubuntu 22.04、CUDA 12.9、PyTorch 2.9、Flash-Attention）
- 通过 Dockerfile 添加 cuSPARSELt 依赖：
  ```dockerfile
  # 安装 cuSPARSELt
  RUN apt-get update && apt-get install -y libcusparselt0 libcusparselt-dev
  ```
- 卸载预装的 vllm（`pip uninstall vllm`）
- 使用开发模式安装（`pip install -e .`）建立源码软链接

**当前状态**：开发环境已就绪，可以直接修改 `vllm/` 目录下的源代码进行测试。

### 3.2 Baseline 验证阶段

#### 3.2.1 吞吐测试脚本验证

**目标**：确保能够在 vLLM 源码环境中运行 HuggingFace 开源模型，获取正确的输出和吞吐数据。

**测试命令**：
```bash
# 吞吐量测试 - Llama3.2-1B
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100 \
    --dtype auto \
    --tensor-parallel-size 1

# 吞吐量测试 - Qwen3-8B (FP8)
vllm bench throughput \
    --model Qwen/Qwen3-8B \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100 \
    --quantization fp8 \
    --dtype float16
```

**关键文件**：
- CLI 入口：`vllm/entrypoints/cli/benchmark/throughput.py`
- 底层脚本：`benchmarks/benchmark_throughput.py`

**输出指标**：
- Prefill 阶段：tokens/s
- Decode 阶段：tokens/s
- 端到端吞吐量：requests/s

#### 3.2.2 模型对话验证

**目标**：确保模型能够生成正确、连贯的对话输出。

**测试代码**：
```python
from vllm import LLM, SamplingParams

# 创建 LLM 实例
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    dtype="auto",
    tensor_parallel_size=1,
)

# 对话测试
messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

outputs = llm.chat(
    messages,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=256)
)

print(outputs[0].outputs[0].text)
```

### 3.3 目标模型选择与准备

#### 3.3.1 FP8 模型矩阵

| 模型 | HuggingFace 路径 | 参数量 | hidden_size | intermediate_size |
|------|-----------------|--------|-------------|-------------------|
| Llama3.2-1B | `meta-llama/Llama-3.2-1B-Instruct` | 1B | 2048 | 8192 |
| Llama3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 3B | 3072 | 8192 |
| Qwen3-8B | `Qwen/Qwen3-8B` | 8B | 4096 | 12288 |
| Qwen3-14B | `Qwen/Qwen3-14B` | 14B | 5120 | 13696 |

**FP8 量化要求**：
- 模型原始激活必须是 BF16（不能是原生 FP8）
- 这样才有 quant/dequant 操作可以融合 slide

**vLLM FP8 启用方式**：
```python
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    quantization="fp8",
    dtype="float16",  # 或 "bfloat16"
)
```

#### 3.3.2 INT8 模型准备

**选项 A**：寻找现有的 W8A8 量化模型
- 搜索 HuggingFace 上的 AWQ/GPTQ W8A8 版本
- 例如：`TheBloke/Llama-2-7B-Chat-AWQ`

**选项 B**：自行量化 BF16 模型
- 使用 vLLM 内置的动态量化
- 或使用 llm-compressor 等工具进行离线量化

**vLLM INT8 相关配置**：
```python
# 使用 CompressedTensors 加载 W8A8 模型
llm = LLM(
    model="path/to/w8a8_model",
    quantization="compressed-tensors",
)
```

### 3.4 GEMM Kernel 分析

#### 3.4.1 确认当前 GEMM 实现

**目标**：确定 vLLM 在 FP8/INT8 GEMM 中调用的具体 kernel。

**分析路径**（基于 `framework_lineargemm.md`）：

```
线性层 forward
    │
    ▼
self.quant_method.apply(...)
    │
    ├── FP8: Fp8LinearMethod.apply()
    │   └── Fp8LinearOp.apply()
    │       ├── 量化: ops.scaled_fp8_quant()
    │       └── GEMM: cutlass_scaled_mm() 或 triton_scaled_mm()
    │
    └── INT8: 取决于具体量化配置
        └── 可能是 CUTLASS INT8 GEMM 或 cuBLAS INT8
```

**关键文件**：
- FP8 量化方法：`vllm/model_executor/layers/quantization/fp8.py`
- GEMM 调用：`vllm/_custom_ops.py`
- CUTLASS 封装：`csrc/quantization/cutlass_extensions/`

#### 3.4.2 Baseline GEMM 类型确认

| 量化类型 | GEMM 实现 | 数据类型 | 输出类型 |
|---------|----------|---------|---------|
| FP8 动态 | CUTLASS scaled_mm | FP8×FP8 | FP16/BF16 |
| FP8 静态 | CUTLASS scaled_mm | FP8×FP8 | FP16/BF16 |
| INT8 W8A8 | cuBLASLt INT8 或 CUTLASS | INT8×INT8 | INT32→FP16 |

**如果是闭源 kernel**：需要替换为开源的 cuBLASLt 实现作为已知的 baseline。

### 3.5 线性层维度分析

#### 3.5.1 获取模型线性层尺寸

**脚本示例**：
```python
from transformers import AutoConfig

def get_linear_dimensions(model_name):
    """获取模型的线性层 NK 尺寸"""
    config = AutoConfig.from_pretrained(model_name)
    
    H = config.hidden_size
    I = config.intermediate_size
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    head_dim = H // num_heads
    
    dimensions = {
        'Wqkv': (num_heads * head_dim + 2 * num_kv_heads * head_dim, H),
        'Wo': (H, num_heads * head_dim),
        'W13': (2 * I, H),
        'W2': (H, I),
    }
    
    print(f"Model: {model_name}")
    print(f"hidden_size (H): {H}")
    print(f"intermediate_size (I): {I}")
    print(f"num_heads: {num_heads}, num_kv_heads: {num_kv_heads}")
    print(f"\nLinear layer dimensions [N, K]:")
    for name, (N, K) in dimensions.items():
        print(f"  {name}: [{N}, {K}]")
    
    return dimensions

# 示例
get_linear_dimensions("meta-llama/Llama-3.2-1B-Instruct")
```

#### 3.5.2 典型模型线性层尺寸

| 模型 | Wqkv [N,K] | Wo [N,K] | W13 [N,K] | W2 [N,K] |
|------|-----------|----------|-----------|----------|
| Llama3.2-1B | [2560, 2048] | [2048, 2048] | [16384, 2048] | [2048, 8192] |
| Llama3.2-3B | [4096, 3072] | [3072, 3072] | [16384, 3072] | [3072, 8192] |
| Qwen3-8B | [6144, 4096] | [4096, 4096] | [24576, 4096] | [4096, 12288] |

### 3.6 实现流程总览

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SlideSparse vLLM 集成流程                           │
└─────────────────────────────────────────────────────────────────────────────────┘

Phase 1: 环境准备 (已完成)
├── Fork vLLM 0.13.0
├── Docker 环境配置
└── cuSPARSELt 依赖安装

Phase 2: Baseline 验证
├── 运行吞吐测试
├── 验证模型输出正确性
└── 记录 Baseline 性能数据

Phase 3: 离线权重处理
├── 权重剪枝脚本开发
├── 权重滑动脚本开发
├── cuSPARSELt 压缩脚本开发
├── 算法搜索脚本开发
└── 生成处理后的权重文件

Phase 4: Model Loader 修改 (关键)
├── 创建 SlideSparse 权重加载器
├── 注册 CompressedTensor 类型
└── 修改权重加载流程

Phase 5: Kernel 替换 (关键)
├── 开发 Fused Quant+Slide Triton kernel
├── 封装 cuSPARSELt Sparse GEMM
├── 开发 Fused Transpose+Dequant Triton kernel
└── 修改线性层 forward 路径

Phase 6: 集成测试
├── 端到端吞吐测试
├── 精度验证
└── 性能分析和优化

Phase 7: 实验与分析
├── 不同稀疏度 (2:4, 2:6, 2:8, ...) 测试
├── 不同模型尺寸测试
├── 加速比 vs 精度损失曲线
└── PPL 精度评估
```

---

## 4. 关键工作：Model Loader 权重加载

本章节详细描述如何修改 vLLM 的模型加载流程，以支持 SlideSparse 处理后的权重文件。

### 4.1 vLLM 模型加载架构分析

#### 4.1.1 加载流程概览

```
用户调用 LLM(model="xxx")
    │
    ▼
vllm/entrypoints/llm.py: LLM.__init__()
    │
    ▼
LLMEngine.from_engine_args()
    │
    ▼
vllm/model_executor/model_loader/__init__.py: get_model()
    │
    ├── get_model_loader(load_config)  # 根据 load_format 选择加载器
    │   └── 返回 DefaultModelLoader / BitsAndBytesModelLoader / ...
    │
    └── loader.load_model(vllm_config, model_config)
        │
        ├── initialize_model()          # 初始化模型结构
        │   └── model_class(vllm_config, prefix)
        │
        ├── load_weights(model, model_config)  # 加载权重
        │   └── 遍历权重文件，调用 weight_loader
        │
        └── process_weights_after_loading()    # 后处理（量化等）
            └── quant_method.process_weights_after_loading()
```

#### 4.1.2 关键类和文件

| 类/函数 | 文件路径 | 说明 |
|--------|---------|------|
| `get_model` | `vllm/model_executor/model_loader/__init__.py` | 模型加载入口 |
| `get_model_loader` | `vllm/model_executor/model_loader/__init__.py` | 加载器工厂 |
| `BaseModelLoader` | `vllm/model_executor/model_loader/base_loader.py` | 加载器基类 |
| `DefaultModelLoader` | `vllm/model_executor/model_loader/default_loader.py` | 默认加载器 |
| `initialize_model` | `vllm/model_executor/model_loader/utils.py` | 模型初始化 |
| `process_weights_after_loading` | `vllm/model_executor/model_loader/utils.py` | 权重后处理 |

### 4.2 SlideSparse 权重加载方案

#### 4.2.1 方案选择

**方案 A：创建自定义 ModelLoader（推荐）**
- 继承 `BaseModelLoader`
- 注册为新的 `load_format`（如 `"slidesparse"`）
- 在加载权重前执行离线处理脚本

**方案 B：预处理权重文件 + 自定义 QuantizationConfig**
- 离线生成处理后的权重文件
- 创建 `SlideSparseConfig` 继承 `QuantizationConfig`
- 在 `process_weights_after_loading` 中处理

**推荐方案 A**，因为：
1. 权重处理在加载阶段完成，更清晰
2. 不影响现有的量化流程
3. 可以复用现有的权重迭代器

#### 4.2.2 详细实现方案

**步骤 1：创建 SlideSparse 权重预处理脚本**

创建脚本文件：`tools/slidesparse/preprocess_weights.py`

```python
"""
SlideSparse 权重预处理脚本

使用方法:
    python tools/slidesparse/preprocess_weights.py \
        --input-model meta-llama/Llama-3.2-1B-Instruct \
        --output-dir ./slidesparse_weights/llama-3.2-1b \
        --sparsity 2:8 \
        --prune-mode magnitude
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

from slidesparse_utils import (
    prune_weight,
    slide_weight,
    compress_weight_cusparselt,
)


def preprocess_model_weights(
    input_model: str,
    output_dir: str,
    sparsity: tuple[int, int],  # (Z, L) 如 (2, 8)
    prune_mode: str = "magnitude",
    target_layers: list[str] = None,
):
    """
    预处理模型权重：剪枝 -> 滑动 -> 压缩
    
    Args:
        input_model: 输入模型路径或 HuggingFace ID
        output_dir: 输出目录
        sparsity: 稀疏格式 (Z, L)
        prune_mode: 剪枝模式 'magnitude' 或 'random'
        target_layers: 需要处理的层名称列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认处理的线性层
    if target_layers is None:
        target_layers = [
            "qkv_proj",      # Wqkv
            "o_proj",        # Wo
            "gate_up_proj",  # W13
            "down_proj",     # W2
        ]
    
    # 加载原始模型
    print(f"Loading model from {input_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        input_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    
    processed_weights = {}
    metadata = {
        "sparsity": f"{sparsity[0]}:{sparsity[1]}",
        "prune_mode": prune_mode,
        "processed_layers": [],
    }
    
    # 遍历模型参数
    for name, param in model.named_parameters():
        weight = param.data
        
        # 检查是否是需要处理的线性层权重
        is_target = any(layer in name for layer in target_layers)
        is_weight = name.endswith(".weight")
        
        if is_target and is_weight and len(weight.shape) == 2:
            print(f"Processing {name}...")
            
            # 1. 剪枝
            pruned_weight, mask = prune_weight(
                weight,
                sparsity_ratio=sparsity[0] / sparsity[1],
                group_size=sparsity[1],
                mode=prune_mode,
            )
            
            # 2. 滑动
            slided_weight = slide_weight(
                pruned_weight,
                src_sparsity=sparsity,
                tgt_sparsity=(2, 4),
            )
            
            # 3. 压缩
            compressed_weight, compress_meta = compress_weight_cusparselt(
                slided_weight,
                dtype=torch.float8_e4m3fn,
            )
            
            # 保存处理后的权重
            processed_weights[name] = compressed_weight
            processed_weights[f"{name}.compress_meta"] = compress_meta
            
            # 记录元数据
            metadata["processed_layers"].append({
                "name": name,
                "original_shape": list(weight.shape),
                "slided_shape": list(slided_weight.shape),
                "compressed_shape": list(compressed_weight.shape),
            })
        else:
            # 其他参数保持不变
            processed_weights[name] = weight
    
    # 保存处理后的权重
    output_path = os.path.join(output_dir, "model.safetensors")
    print(f"Saving processed weights to {output_path}...")
    save_file(processed_weights, output_path)
    
    # 保存元数据
    metadata_path = os.path.join(output_dir, "slidesparse_config.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 复制 tokenizer 和 config
    model.config.save_pretrained(output_dir)
    
    print("Done!")
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sparsity", default="2:8")
    parser.add_argument("--prune-mode", default="magnitude")
    args = parser.parse_args()
    
    Z, L = map(int, args.sparsity.split(":"))
    preprocess_model_weights(
        args.input_model,
        args.output_dir,
        (Z, L),
        args.prune_mode,
    )
```

**步骤 2：创建 SlideSparse ModelLoader**

创建文件：`vllm/model_executor/model_loader/slidesparse_loader.py`

```python
"""
SlideSparse Model Loader

用于加载经过 SlideSparse 预处理的权重文件。
"""

import json
import os
from collections.abc import Generator
from typing import Any

import torch
from torch import nn

from vllm.config import ModelConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    safetensors_weights_iterator,
)

logger = init_logger(__name__)


class SlideSparseModelLoader(BaseModelLoader):
    """Model loader for SlideSparse preprocessed weights."""
    
    def __init__(self, load_config):
        super().__init__(load_config)
        self.slidesparse_config: dict[str, Any] | None = None
    
    def _load_slidesparse_config(self, model_path: str) -> dict[str, Any]:
        """加载 SlideSparse 配置文件"""
        config_path = os.path.join(model_path, "slidesparse_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        return {}
    
    def download_model(self, model_config: ModelConfig) -> None:
        """下载模型（如果需要）"""
        # SlideSparse 模型通常是本地预处理的
        # 如果需要下载，复用 DefaultModelLoader 的逻辑
        default_loader = DefaultModelLoader(self.load_config)
        default_loader.download_model(model_config)
    
    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """加载 SlideSparse 预处理后的权重"""
        model_path = model_config.model
        
        # 加载 SlideSparse 配置
        self.slidesparse_config = self._load_slidesparse_config(model_path)
        
        if self.slidesparse_config:
            logger.info(
                "Loading SlideSparse model with sparsity: %s",
                self.slidesparse_config.get("sparsity", "unknown")
            )
        
        # 获取权重文件
        weight_files = self._get_weight_files(model_path)
        
        # 使用权重迭代器加载
        for name, loaded_weight in safetensors_weights_iterator(weight_files):
            # 跳过压缩元数据
            if name.endswith(".compress_meta"):
                continue
            
            # 检查是否是 SlideSparse 处理过的权重
            is_slidesparse_weight = self._is_slidesparse_weight(name)
            
            if is_slidesparse_weight:
                # 加载对应的压缩元数据
                meta_name = f"{name}.compress_meta"
                # 获取元数据（如果存在）
                self._load_slidesparse_weight(model, name, loaded_weight)
            else:
                # 普通权重，使用标准加载
                self._load_standard_weight(model, name, loaded_weight)
    
    def _is_slidesparse_weight(self, name: str) -> bool:
        """检查是否是 SlideSparse 处理过的权重"""
        if not self.slidesparse_config:
            return False
        
        processed_layers = self.slidesparse_config.get("processed_layers", [])
        return any(layer["name"] == name for layer in processed_layers)
    
    def _load_slidesparse_weight(
        self,
        model: nn.Module,
        name: str,
        weight: torch.Tensor,
    ) -> None:
        """加载 SlideSparse 权重到模型"""
        # 找到对应的模块
        param = self._get_parameter(model, name)
        if param is None:
            logger.warning(f"Parameter {name} not found in model")
            return
        
        # 验证形状兼容性
        # 注意：SlideSparse 权重的 K 维度会扩展
        # 需要在模型初始化时已经调整好
        if param.shape != weight.shape:
            logger.warning(
                f"Shape mismatch for {name}: "
                f"model {param.shape} vs weight {weight.shape}"
            )
            # 尝试适配
            weight = self._adapt_weight_shape(param, weight)
        
        # 加载权重
        param.data.copy_(weight)
        
        # 设置 SlideSparse 标记
        param.slidesparse = True
    
    def _load_standard_weight(
        self,
        model: nn.Module,
        name: str,
        weight: torch.Tensor,
    ) -> None:
        """加载标准权重"""
        param = self._get_parameter(model, name)
        if param is not None:
            param.data.copy_(weight)
    
    def _get_parameter(self, model: nn.Module, name: str) -> torch.nn.Parameter | None:
        """根据名称获取模型参数"""
        parts = name.split(".")
        module = model
        for part in parts[:-1]:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        if hasattr(module, parts[-1]):
            return getattr(module, parts[-1])
        return None
    
    def _get_weight_files(self, model_path: str) -> list[str]:
        """获取权重文件列表"""
        import glob
        patterns = ["*.safetensors", "*.pt", "*.bin"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(model_path, pattern)))
        return files
    
    def _adapt_weight_shape(
        self,
        param: torch.nn.Parameter,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """适配权重形状"""
        # 这里需要根据具体情况处理形状适配
        # 例如：切片、填充等
        return weight
```

**步骤 3：注册 SlideSparse ModelLoader**

修改文件：`vllm/model_executor/model_loader/__init__.py`

```python
# 在文件开头添加导入
from vllm.model_executor.model_loader.slidesparse_loader import SlideSparseModelLoader

# 在 _LOAD_FORMAT_TO_MODEL_LOADER 字典中添加
_LOAD_FORMAT_TO_MODEL_LOADER: dict[str, type[BaseModelLoader]] = {
    # ... 现有的加载器 ...
    "slidesparse": SlideSparseModelLoader,  # 添加这一行
}

# 在 LoadFormats 类型定义中添加
LoadFormats = Literal[
    # ... 现有的格式 ...
    "slidesparse",  # 添加这一行
]
```

### 4.3 SlideSparse 量化配置

除了 ModelLoader，还需要创建对应的 QuantizationConfig 以支持在 forward 阶段使用 SlideSparse kernel。

**步骤 4：创建 SlideSparse QuantizationConfig**

创建文件：`vllm/model_executor/layers/quantization/slidesparse.py`

```python
"""
SlideSparse Quantization Configuration

定义 SlideSparse 的量化配置和线性层方法。
"""

from typing import Any, Optional

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

logger = init_logger(__name__)


class SlideSparseConfig(QuantizationConfig):
    """Configuration for SlideSparse quantization."""
    
    def __init__(
        self,
        sparsity: str = "2:8",
        activation_dtype: str = "fp8",  # "fp8" 或 "int8"
        algo_config_path: str | None = None,
    ):
        self.sparsity = sparsity
        Z, L = map(int, sparsity.split(":"))
        self.sparsity_z = Z
        self.sparsity_l = L
        
        self.activation_dtype = activation_dtype
        self.algo_config_path = algo_config_path
        self.algo_config: dict[str, int] = {}
        
        if algo_config_path:
            self._load_algo_config(algo_config_path)
    
    def _load_algo_config(self, path: str) -> None:
        """加载算法配置"""
        import json
        with open(path) as f:
            self.algo_config = json.load(f)
    
    @classmethod
    def get_name(cls) -> str:
        return "slidesparse"
    
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]
    
    @classmethod
    def get_min_capability(cls) -> int:
        return 80  # 需要 Ampere 或更新架构
    
    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["slidesparse_config.json"]
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SlideSparseConfig":
        return cls(
            sparsity=config.get("sparsity", "2:8"),
            activation_dtype=config.get("activation_dtype", "fp8"),
            algo_config_path=config.get("algo_config_path"),
        )
    
    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> Optional["SlideSparseLinearMethod"]:
        if isinstance(layer, LinearBase):
            return SlideSparseLinearMethod(self)
        return None
    
    def get_scaled_act_names(self) -> list[str]:
        return []


class SlideSparseLinearMethod(LinearMethodBase):
    """Linear method for SlideSparse.
    
    实现 SlideSparse 的线性层计算：
    1. Fused Quant + Slide
    2. Sparse GEMM (cuSPARSELt)
    3. Fused Transpose + Dequant
    """
    
    def __init__(self, quant_config: SlideSparseConfig):
        self.quant_config = quant_config
    
    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """创建 SlideSparse 权重参数"""
        # 计算扩展后的输入维度
        Z = self.quant_config.sparsity_z
        L = self.quant_config.sparsity_l
        stride = 4 - 2  # 2:4 硬件的步长
        num_windows = (L - Z) // stride
        expand_ratio = (num_windows * 4) / L
        
        expanded_input_size = int(input_size_per_partition * expand_ratio)
        
        # 创建压缩权重参数
        # 注意：实际形状需要根据 cuSPARSELt 的压缩格式确定
        output_size_total = sum(output_partition_sizes)
        
        weight = torch.nn.Parameter(
            torch.empty(
                output_size_total,
                expanded_input_size // 2,  # 2:4 压缩后大小减半
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        
        # 权重 scale
        weight_scale = torch.nn.Parameter(
            torch.ones(output_size_total, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("weight_scale", weight_scale)
        
        # 设置属性
        layer.input_size_expanded = expanded_input_size
        layer.sparsity_config = self.quant_config
    
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行 SlideSparse 线性变换
        
        流程：
        1. Fused Quant + Slide: BF16 [M, K] -> FP8/INT8 [M, K']
        2. Sparse GEMM: FP8 [M, K'] × Compressed [N, K'/2] -> [N, M]
        3. Fused Transpose + Dequant: [N, M] -> BF16 [M, N]
        """
        # 导入 SlideSparse kernel
        from vllm.model_executor.layers.quantization.slidesparse_kernels import (
            fused_quant_slide,
            sparse_gemm_cusparselt,
            fused_transpose_dequant,
        )
        
        # 1. Fused Quant + Slide
        x_quant, scale_a = fused_quant_slide(
            x,
            src_sparsity=(self.quant_config.sparsity_z, self.quant_config.sparsity_l),
            tgt_sparsity=(2, 4),
            dtype=self._get_quant_dtype(),
        )
        
        # 2. Sparse GEMM
        # 查找最优算法 ID
        M, K_expanded = x_quant.shape
        N = layer.weight.shape[0]
        algo_id = self._get_algo_id(M, N, K_expanded)
        
        output_gemm = sparse_gemm_cusparselt(
            x_quant,
            layer.weight,
            algo_id=algo_id,
        )  # [N, M]
        
        # 3. Fused Transpose + Dequant
        output = fused_transpose_dequant(
            output_gemm,
            scale_a,
            layer.weight_scale,
            out_dtype=x.dtype,
        )  # [M, N]
        
        # 添加 bias
        if bias is not None:
            output = output + bias
        
        return output
    
    def _get_quant_dtype(self) -> torch.dtype:
        """获取量化数据类型"""
        if self.quant_config.activation_dtype == "fp8":
            return torch.float8_e4m3fn
        elif self.quant_config.activation_dtype == "int8":
            return torch.int8
        else:
            raise ValueError(f"Unsupported dtype: {self.quant_config.activation_dtype}")
    
    def _get_algo_id(self, M: int, N: int, K: int) -> int:
        """从配置中获取最优算法 ID"""
        key = f"{M},{N},{K}"
        if key in self.quant_config.algo_config:
            return self.quant_config.algo_config[key]
        # 默认算法
        return 0
```

**步骤 5：注册 SlideSparse 量化配置**

修改文件：`vllm/model_executor/layers/quantization/__init__.py`

```python
# 在 QuantizationMethods 中添加
QuantizationMethods = Literal[
    # ... 现有方法 ...
    "slidesparse",  # 添加这一行
]

# 在 get_quantization_config 函数中添加导入和映射
def get_quantization_config(quantization: str) -> type[QuantizationConfig]:
    # ... 现有导入 ...
    from .slidesparse import SlideSparseConfig  # 添加这一行
    
    method_to_config: dict[str, type[QuantizationConfig]] = {
        # ... 现有映射 ...
        "slidesparse": SlideSparseConfig,  # 添加这一行
    }
    # ...
```

### 4.4 Model Loader 实现检查清单

| 任务 | 文件 | 状态 |
|------|------|------|
| 创建权重预处理脚本 | `tools/slidesparse/preprocess_weights.py` | 待开发 |
| 创建 slidesparse_utils 工具模块 | `tools/slidesparse/slidesparse_utils.py` | 待开发 |
| 创建 SlideSparseModelLoader | `vllm/model_executor/model_loader/slidesparse_loader.py` | 待开发 |
| 注册 ModelLoader | `vllm/model_executor/model_loader/__init__.py` | 待修改 |
| 创建 SlideSparseConfig | `vllm/model_executor/layers/quantization/slidesparse.py` | 待开发 |
| 注册 QuantizationConfig | `vllm/model_executor/layers/quantization/__init__.py` | 待修改 |

---

## 5. 关键工作：Kernel 替换与前向传播

本章节详细描述如何修改 vLLM 的前向传播流程，将标准的 `quant + dense_GEMM + dequant` 替换为 SlideSparse 的 `fused_quant_slide + sparse_GEMM + fused_transpose_dequant`。

### 5.1 vLLM 线性层前向传播分析

#### 5.1.1 当前 GEMM 调用链

根据 `framework_lineargemm.md` 的分析，vLLM 的线性层前向传播调用链如下：

```
模型 forward (如 Qwen2MLP.forward)
    │
    ▼
self.gate_up_proj(x)  # MergedColumnParallelLinear
    │
    ▼
ColumnParallelLinear.forward()
文件: vllm/model_executor/layers/linear.py (行 557-575)
    │
    ▼
self.quant_method.apply(self, input_, bias)  # 核心 GEMM 调用
    │
    ├── UnquantizedLinearMethod.apply()      # 无量化
    │   └── dispatch_unquantized_gemm()
    │       └── torch.nn.functional.linear() # cuBLAS
    │
    ├── Fp8LinearMethod.apply()              # FP8 量化
    │   └── Fp8LinearOp.apply()
    │       ├── ops.scaled_fp8_quant()       # 量化
    │       └── cutlass_scaled_mm()          # GEMM + Dequant
    │
    └── SlideSparseLinearMethod.apply()      # SlideSparse (新增)
        ├── fused_quant_slide()              # Quant + Slide
        ├── sparse_gemm_cusparselt()         # Sparse GEMM
        └── fused_transpose_dequant()        # Transpose + Dequant
```

#### 5.1.2 关键文件和函数

| 组件 | 文件 | 函数/类 | 行号 |
|------|------|--------|------|
| 线性层 forward | `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear.forward()` | 557-575 |
| 线性层 forward | `vllm/model_executor/layers/linear.py` | `RowParallelLinear.forward()` | 1388-1416 |
| FP8 量化方法 | `vllm/model_executor/layers/quantization/fp8.py` | `Fp8LinearMethod.apply()` | ~610-687 |
| FP8 量化操作 | `vllm/_custom_ops.py` | `scaled_fp8_quant()` | 1678-1735 |
| CUTLASS GEMM | `vllm/_custom_ops.py` | `cutlass_scaled_mm()` | 828-876 |
| 稀疏压缩 | `vllm/_custom_ops.py` | `cutlass_sparse_compress()` | 920-958 |
| 稀疏 GEMM | `vllm/_custom_ops.py` | `cutlass_scaled_sparse_mm()` | 961-1000+ |

#### 5.1.3 vLLM 已有的稀疏支持

vLLM 已经实现了 2:4 稀疏的基础支持：

```python
# vllm/_custom_ops.py

def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    压缩 2:4 稀疏矩阵
    
    输入: a [M, K]，满足 2:4 稀疏
    输出:
        - a_nzs [M, K/2]: 非零元素
        - a_meta [M, K/8]: 稀疏元数据（每 4 个非零对应 1 字节）
    """
    return torch.ops._C.cutlass_sparse_compress(a)

def cutlass_scaled_sparse_mm(
    a: torch.Tensor,
    bt_nzs: torch.Tensor,
    bt_meta: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    执行稀疏矩阵乘法
    
    输入:
        - a [M, K]: 稠密激活
        - bt_nzs [N, K/2]: 压缩后的权重非零元素
        - bt_meta [N, K/8]: 权重稀疏元数据
        - scale_a, scale_b: 量化 scale
    
    输出: [M, N]
    """
    ...
```

**重要发现**：vLLM 使用 CUTLASS 实现稀疏 GEMM，这与 cuSPARSELt 是不同的后端。需要评估使用哪个：
- **CUTLASS Sparse**: 已集成在 vLLM 中，更易使用
- **cuSPARSELt**: 可能有更好的性能，需要额外集成

### 5.2 SlideSparse Kernel 实现

#### 5.2.1 Kernel 文件组织

建议创建以下文件结构：

```
vllm/model_executor/layers/quantization/
├── slidesparse.py                      # SlideSparse 配置和线性方法
└── slidesparse_kernels/
    ├── __init__.py                     # 导出所有 kernel
    ├── fused_quant_slide.py            # Triton: 融合量化+滑动
    ├── sparse_gemm.py                  # CUDA/CUTLASS: 稀疏 GEMM 封装
    └── fused_transpose_dequant.py      # Triton: 融合转置+反量化
```

#### 5.2.2 Fused Quant + Slide Kernel

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/fused_quant_slide.py`

```python
"""
Fused Quantization + Slide Triton Kernel

将 BF16 输入激活进行量化（FP8/INT8）并同时执行滑动拓展。
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'K', 'K_expanded'],
)
@triton.jit
def fused_quant_slide_kernel(
    # 输入输出指针
    input_ptr,          # [M, K] BF16 输入
    output_ptr,         # [M, K_expanded] FP8 输出
    scale_ptr,          # [M] 或 [1] per-token/per-tensor scale
    # 维度信息
    M,                  # batch × seq_len
    K,                  # 原始输入维度
    K_expanded,         # 滑动拓展后的维度
    # 稀疏参数
    src_L: tl.constexpr,     # 源稀疏窗口大小 (如 8)
    tgt_L: tl.constexpr,     # 目标稀疏窗口大小 (如 4)
    stride: tl.constexpr,    # 滑动步长 (如 2)
    num_windows: tl.constexpr,  # 窗口数
    # 步长
    stride_im,          # input stride for M
    stride_ik,          # input stride for K
    stride_om,          # output stride for M
    stride_ok,          # output stride for K_expanded
    # 配置
    use_per_token_scale: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    融合量化+滑动 Kernel
    
    算法:
    1. 对每个源稀疏组 (src_L 个元素)
    2. 计算动态 scale (如果需要)
    3. 量化为 FP8
    4. 按滑动窗口模式写出到目标位置
    """
    # 程序 ID
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    
    # 计算块起始位置
    m_start = pid_m * BLOCK_M
    k_start = pid_k * BLOCK_K
    
    # 计算 K 维度在源稀疏组中的位置
    # k_start 对应的源稀疏组起始
    src_group_start = (k_start // src_L) * src_L
    
    # 加载输入块
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_k_src = src_group_start + tl.arange(0, src_L)
    
    mask_m = offs_m < M
    mask_k = offs_k_src < K
    
    # 加载源数据 [BLOCK_M, src_L]
    x = tl.load(
        input_ptr + offs_m[:, None] * stride_im + offs_k_src[None, :] * stride_ik,
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    )
    
    # 计算 scale
    if use_per_token_scale:
        # Per-token: 每行计算 max
        x_max = tl.max(tl.abs(x), axis=1)  # [BLOCK_M]
        scale = x_max / 448.0  # FP8 E4M3 最大值
        scale = tl.where(scale > 0, scale, 1.0)
        
        # 保存 scale
        tl.store(
            scale_ptr + offs_m,
            scale,
            mask=mask_m,
        )
    else:
        # Per-tensor: 使用预设 scale
        scale = tl.load(scale_ptr)
    
    # 量化
    if use_per_token_scale:
        x_quant = x / scale[:, None]
    else:
        x_quant = x / scale
    
    # 裁剪到 FP8 范围
    x_quant = tl.clamp(x_quant, -448.0, 448.0)
    
    # 滑动写出
    # 计算目标位置
    dst_group_start = (k_start // src_L) * (num_windows * tgt_L)
    
    for w in range(num_windows):
        # 源窗口范围: [w * stride, w * stride + tgt_L)
        src_start = w * stride
        # 目标位置: dst_group_start + w * tgt_L
        dst_start = dst_group_start + w * tgt_L
        
        # 提取窗口数据
        offs_window = tl.arange(0, tgt_L)
        window_data = tl.load(
            x_quant + offs_m[:, None] * src_L + (src_start + offs_window)[None, :],
            mask=mask_m[:, None] & ((src_start + offs_window)[None, :] < src_L),
            other=0.0,
        )
        
        # 写出到目标位置
        offs_k_dst = dst_start + offs_window
        tl.store(
            output_ptr + offs_m[:, None] * stride_om + offs_k_dst[None, :] * stride_ok,
            window_data,
            mask=mask_m[:, None] & (offs_k_dst[None, :] < K_expanded),
        )


def fused_quant_slide(
    input: torch.Tensor,
    src_sparsity: tuple[int, int],  # (Z, L') 如 (2, 8)
    tgt_sparsity: tuple[int, int],  # (Z, L) 如 (2, 4)
    dtype: torch.dtype = torch.float8_e4m3fn,
    use_per_token_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    融合的量化 + 滑动操作
    
    Args:
        input: [M, K] BF16 输入激活
        src_sparsity: 源稀疏格式 (Z, L')
        tgt_sparsity: 目标硬件稀疏格式 (Z, L)
        dtype: 输出数据类型
        use_per_token_scale: 是否使用 per-token 量化
    
    Returns:
        output: [M, K'] 量化后的激活
        scale: [M] 或 [1] 量化 scale
    """
    assert input.ndim == 2
    M, K = input.shape
    
    Z_src, L_src = src_sparsity
    Z_tgt, L_tgt = tgt_sparsity
    
    # 计算滑动参数
    stride = L_tgt - Z_tgt  # 非零元素个数
    num_windows = (L_src - Z_src) // stride
    
    # 计算拓展后的维度
    num_groups = K // L_src
    K_expanded = num_groups * num_windows * L_tgt
    
    # 分配输出
    output = torch.empty((M, K_expanded), device=input.device, dtype=dtype)
    
    if use_per_token_scale:
        scale = torch.empty((M,), device=input.device, dtype=torch.float32)
    else:
        # 计算全局 scale
        scale = torch.max(torch.abs(input)) / 448.0
        scale = scale.unsqueeze(0)
    
    # 配置 grid
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(K, L_src),  # 每个源稀疏组一个
        )
    
    # 调用 kernel
    fused_quant_slide_kernel[grid](
        input,
        output,
        scale,
        M, K, K_expanded,
        L_src, L_tgt, stride, num_windows,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        use_per_token_scale,
    )
    
    return output, scale
```

#### 5.2.3 Sparse GEMM 封装

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/sparse_gemm.py`

```python
"""
Sparse GEMM Wrapper

封装 CUTLASS 或 cuSPARSELt 的稀疏 GEMM 调用。
"""

import torch
from vllm import _custom_ops as ops


def sparse_gemm_cutlass(
    activation: torch.Tensor,      # [M, K'] FP8 量化+滑动后的激活
    weight_nzs: torch.Tensor,      # [N, K'/2] 压缩后的权重非零元素
    weight_meta: torch.Tensor,     # [N, K'/8] 权重稀疏元数据
    scale_a: torch.Tensor,         # 激活 scale
    scale_b: torch.Tensor,         # 权重 scale
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    使用 CUTLASS 执行稀疏 GEMM
    
    注意: CUTLASS 稀疏 GEMM 期望权重是压缩过的，激活是稠密的
    
    Args:
        activation: [M, K'] 稠密激活（已经滑动拓展）
        weight_nzs: [N, K'/2] 压缩后的权重
        weight_meta: [N, K'/8] 稀疏元数据
        scale_a, scale_b: 量化 scale
        out_dtype: 输出数据类型
        bias: 可选偏置
    
    Returns:
        output: [M, N] 输出
    """
    # 调用 vLLM 已有的 CUTLASS 稀疏 GEMM
    output = ops.cutlass_scaled_sparse_mm(
        activation,
        weight_nzs,
        weight_meta,
        scale_a,
        scale_b,
        out_dtype,
        bias,
    )
    
    return output


def sparse_gemm_cusparselt(
    activation: torch.Tensor,      # [M, K'] FP8/INT8
    weight_compressed: torch.Tensor,  # cuSPARSELt 压缩格式
    algo_id: int = 0,
) -> torch.Tensor:
    """
    使用 cuSPARSELt 执行稀疏 GEMM
    
    注意: 这需要额外的 cuSPARSELt 集成
    
    Args:
        activation: [M, K'] 量化后的激活
        weight_compressed: cuSPARSELt 压缩格式的权重
        algo_id: 算法 ID
    
    Returns:
        output: [N, M] 行主序输出（需要后续转置）
    """
    # TODO: 实现 cuSPARSELt 调用
    # 这需要:
    # 1. 添加 cuSPARSELt Python 绑定
    # 2. 注册 torch.ops._C 函数
    
    raise NotImplementedError(
        "cuSPARSELt integration not yet implemented. "
        "Please use CUTLASS sparse GEMM for now."
    )


def compress_weight_for_sparse_gemm(
    weight: torch.Tensor,
    backend: str = "cutlass",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    压缩权重以用于稀疏 GEMM
    
    Args:
        weight: [N, K] 满足 2:4 稀疏的权重
        backend: "cutlass" 或 "cusparselt"
    
    Returns:
        weight_nzs: 压缩后的非零元素
        weight_meta: 稀疏元数据
    """
    if backend == "cutlass":
        # CUTLASS 压缩需要转置后的权重
        weight_t = weight.t().contiguous()
        weight_nzs, weight_meta = ops.cutlass_sparse_compress(weight_t)
        return weight_nzs, weight_meta
    else:
        raise NotImplementedError(f"Backend {backend} not supported")
```

#### 5.2.4 Fused Transpose + Dequant Kernel

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/fused_transpose_dequant.py`

```python
"""
Fused Transpose + Dequantization Triton Kernel

将 [N, M] 的 GEMM 输出转置为 [M, N] 并反量化为 BF16。
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_transpose_dequant_kernel(
    # 输入输出指针
    input_ptr,          # [N, M] INT32/FP32 输入 (GEMM 输出)
    output_ptr,         # [M, N] BF16 输出
    scale_a_ptr,        # 激活 scale
    scale_b_ptr,        # 权重 scale
    # 维度
    M,
    N,
    # 步长
    stride_in,          # input stride for N
    stride_im,          # input stride for M
    stride_om,          # output stride for M
    stride_on,          # output stride for N
    # scale 类型
    use_per_token_scale_a: tl.constexpr,
    use_per_channel_scale_b: tl.constexpr,
    # 配置
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    融合转置+反量化 Kernel
    
    执行:
    output[m, n] = input[n, m] * scale_a[m] * scale_b[n]
    """
    # 程序 ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算块起始位置
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    # 偏移
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    
    # 从 [N, M] 读取（转置访问）
    # input[n, m] -> 读取位置 offs_n[:, None] * stride_in + offs_m[None, :] * stride_im
    x = tl.load(
        input_ptr + offs_n[None, :] * stride_in + offs_m[:, None] * stride_im,
        mask=mask_n[None, :] & mask_m[:, None],
        other=0.0,
    )
    # x 现在是 [BLOCK_M, BLOCK_N]
    
    # 转置: x[m, n] = input[n, m]
    # 上面的读取已经完成了转置
    
    # 加载 scale
    if use_per_token_scale_a:
        scale_a = tl.load(scale_a_ptr + offs_m, mask=mask_m)  # [BLOCK_M]
    else:
        scale_a = tl.load(scale_a_ptr)  # scalar
    
    if use_per_channel_scale_b:
        scale_b = tl.load(scale_b_ptr + offs_n, mask=mask_n)  # [BLOCK_N]
    else:
        scale_b = tl.load(scale_b_ptr)  # scalar
    
    # 反量化
    x_fp32 = x.to(tl.float32)
    if use_per_token_scale_a:
        x_fp32 = x_fp32 * scale_a[:, None]
    else:
        x_fp32 = x_fp32 * scale_a
    
    if use_per_channel_scale_b:
        x_fp32 = x_fp32 * scale_b[None, :]
    else:
        x_fp32 = x_fp32 * scale_b
    
    # 转换为 BF16
    x_bf16 = x_fp32.to(tl.bfloat16)
    
    # 写出 [M, N]
    tl.store(
        output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        x_bf16,
        mask=mask,
    )


def fused_transpose_dequant(
    input: torch.Tensor,           # [N, M] GEMM 输出
    scale_a: torch.Tensor,         # 激活 scale
    scale_b: torch.Tensor,         # 权重 scale
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    融合的转置 + 反量化操作
    
    Args:
        input: [N, M] GEMM 输出 (INT32 或 FP32)
        scale_a: 激活量化 scale
        scale_b: 权重量化 scale
        out_dtype: 输出数据类型
    
    Returns:
        output: [M, N] 反量化后的输出
    """
    assert input.ndim == 2
    N, M = input.shape
    
    # 判断 scale 类型
    use_per_token_scale_a = scale_a.numel() == M
    use_per_channel_scale_b = scale_b.numel() == N
    
    # 分配输出
    output = torch.empty((M, N), device=input.device, dtype=out_dtype)
    
    # 配置 grid
    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
    
    # 调用 kernel
    fused_transpose_dequant_kernel[grid](
        input,
        output,
        scale_a,
        scale_b,
        M, N,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        use_per_token_scale_a,
        use_per_channel_scale_b,
    )
    
    return output
```

#### 5.2.5 Kernel 模块初始化

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/__init__.py`

```python
"""
SlideSparse Kernels Module

导出所有 SlideSparse 相关的 kernel 函数。
"""

from .fused_quant_slide import fused_quant_slide
from .sparse_gemm import (
    sparse_gemm_cutlass,
    sparse_gemm_cusparselt,
    compress_weight_for_sparse_gemm,
)
from .fused_transpose_dequant import fused_transpose_dequant

__all__ = [
    "fused_quant_slide",
    "sparse_gemm_cutlass",
    "sparse_gemm_cusparselt",
    "compress_weight_for_sparse_gemm",
    "fused_transpose_dequant",
]
```

### 5.3 修改 SlideSparseLinearMethod.apply()

更新 `vllm/model_executor/layers/quantization/slidesparse.py` 中的 `apply` 方法：

```python
class SlideSparseLinearMethod(LinearMethodBase):
    """Linear method for SlideSparse."""
    
    def __init__(self, quant_config: SlideSparseConfig):
        self.quant_config = quant_config
        # 选择 GEMM 后端
        self.gemm_backend = quant_config.gemm_backend  # "cutlass" 或 "cusparselt"
    
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        执行 SlideSparse 线性变换
        
        完整流程:
        1. Fused Quant + Slide: BF16 [M, K] -> FP8 [M, K']
        2. Sparse GEMM: FP8 [M, K'] × Compressed [N, K'/2] -> FP32 [M, N]
        3. Dequant: FP32 [M, N] -> BF16 [M, N]
        
        注意: 使用 CUTLASS sparse GEMM 时，输出直接是 [M, N]，不需要转置
        """
        from vllm.model_executor.layers.quantization.slidesparse_kernels import (
            fused_quant_slide,
            sparse_gemm_cutlass,
        )
        
        # 获取稀疏参数
        Z = self.quant_config.sparsity_z
        L = self.quant_config.sparsity_l
        
        # 1. Fused Quant + Slide
        x_quant, scale_a = fused_quant_slide(
            x,
            src_sparsity=(Z, L),
            tgt_sparsity=(2, 4),
            dtype=self._get_quant_dtype(),
            use_per_token_scale=True,
        )
        
        # 2. Sparse GEMM (使用 CUTLASS)
        # CUTLASS sparse GEMM 输出直接是 [M, N]
        output = sparse_gemm_cutlass(
            x_quant,
            layer.weight_nzs,      # 压缩后的权重
            layer.weight_meta,     # 稀疏元数据
            scale_a,
            layer.weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        
        return output
    
    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """
        权重加载后的处理
        
        将加载的权重压缩为稀疏格式
        """
        from vllm.model_executor.layers.quantization.slidesparse_kernels import (
            compress_weight_for_sparse_gemm,
        )
        
        # 压缩权重
        weight = layer.weight.data
        weight_nzs, weight_meta = compress_weight_for_sparse_gemm(
            weight,
            backend="cutlass",
        )
        
        # 替换权重参数
        del layer.weight
        layer.register_buffer("weight_nzs", weight_nzs)
        layer.register_buffer("weight_meta", weight_meta)
```

### 5.4 启用 SlideSparse 的方式

#### 5.4.1 通过量化配置启用

```python
from vllm import LLM

# 方式 1: 使用预处理后的模型
llm = LLM(
    model="./slidesparse_weights/llama-3.2-1b",
    quantization="slidesparse",
    load_format="slidesparse",
)

# 方式 2: 使用配置文件
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    quantization="slidesparse",
    quantization_param_path="./slidesparse_config.json",
)
```

#### 5.4.2 运行时条件分支（可选方案）

如果需要在运行时根据条件选择是否使用 SlideSparse，可以添加环境变量控制：

```python
# 在 linear.py 或模型文件中
import os

USE_SLIDESPARSE = os.getenv("VLLM_USE_SLIDESPARSE", "0") == "1"

class SomeLayer(nn.Module):
    def forward(self, x):
        if USE_SLIDESPARSE and hasattr(self.linear, 'weight_nzs'):
            # 使用 SlideSparse 路径
            return self.slidesparse_forward(x)
        else:
            # 使用标准路径
            return self.standard_forward(x)
```

### 5.5 Kernel 替换检查清单

| 任务 | 文件 | 状态 |
|------|------|------|
| 创建 fused_quant_slide kernel | `slidesparse_kernels/fused_quant_slide.py` | 待开发 |
| 创建 sparse_gemm 封装 | `slidesparse_kernels/sparse_gemm.py` | 待开发 |
| 创建 fused_transpose_dequant kernel | `slidesparse_kernels/fused_transpose_dequant.py` | 待开发 |
| 创建 kernel 模块 __init__.py | `slidesparse_kernels/__init__.py` | 待开发 |
| 更新 SlideSparseLinearMethod.apply() | `slidesparse.py` | 待开发 |
| 添加 process_weights_after_loading | `slidesparse.py` | 待开发 |
| Triton autotune 调优 | 各 kernel 文件 | 待测试 |
| 单元测试 | `tests/kernels/test_slidesparse.py` | 待开发 |

---

## 6. 测试与验证

### 6.1 单元测试

#### 6.1.1 Kernel 测试

创建文件：`tests/kernels/test_slidesparse.py`

```python
"""
SlideSparse Kernel 单元测试
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.slidesparse_kernels import (
    fused_quant_slide,
    sparse_gemm_cutlass,
    fused_transpose_dequant,
    compress_weight_for_sparse_gemm,
)


class TestFusedQuantSlide:
    """测试融合量化+滑动 kernel"""
    
    @pytest.mark.parametrize("M", [1, 32, 128])
    @pytest.mark.parametrize("K", [256, 512, 1024])
    @pytest.mark.parametrize("sparsity", [(2, 8), (2, 6)])
    def test_output_shape(self, M, K, sparsity):
        """测试输出形状正确性"""
        Z, L = sparsity
        stride = 4 - 2  # 2:4 目标
        num_windows = (L - Z) // stride
        K_expanded = (K // L) * num_windows * 4
        
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        output, scale = fused_quant_slide(x, sparsity, (2, 4))
        
        assert output.shape == (M, K_expanded)
        assert output.dtype == torch.float8_e4m3fn
    
    def test_quantization_accuracy(self):
        """测试量化精度"""
        M, K = 32, 256
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        output, scale = fused_quant_slide(x, (2, 8), (2, 4), use_per_token_scale=False)
        
        # 反量化应该近似原始值（考虑滑动重排）
        # 这需要更复杂的验证逻辑


class TestSparseGemm:
    """测试稀疏 GEMM"""
    
    def test_correctness(self):
        """测试计算正确性"""
        M, K, N = 32, 512, 256
        
        # 创建 2:4 稀疏权重
        weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
        # 强制 2:4 稀疏
        weight = prune_to_2_4(weight)
        
        # 压缩权重
        weight_nzs, weight_meta = compress_weight_for_sparse_gemm(weight)
        
        # 创建激活
        activation = torch.randn(M, K, dtype=torch.float8_e4m3fn, device="cuda")
        scale_a = torch.ones(1, device="cuda")
        scale_b = torch.ones(1, device="cuda")
        
        # 执行稀疏 GEMM
        output = sparse_gemm_cutlass(
            activation, weight_nzs, weight_meta,
            scale_a, scale_b, torch.bfloat16
        )
        
        assert output.shape == (M, N)


def prune_to_2_4(weight: torch.Tensor) -> torch.Tensor:
    """将权重剪枝为 2:4 稀疏"""
    N, K = weight.shape
    assert K % 4 == 0
    
    weight_grouped = weight.view(N, -1, 4)
    _, indices = torch.topk(weight_grouped.abs(), 2, dim=-1)
    mask = torch.zeros_like(weight_grouped)
    mask.scatter_(-1, indices, 1)
    
    return (weight_grouped * mask).view(N, K)
```

### 6.2 端到端测试

#### 6.2.1 吞吐量测试脚本

```bash
#!/bin/bash
# test_slidesparse_throughput.sh

# 预处理权重
python tools/slidesparse/preprocess_weights.py \
    --input-model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./slidesparse_weights/llama-3.2-1b \
    --sparsity 2:8 \
    --prune-mode magnitude

# 运行吞吐测试
vllm bench throughput \
    --model ./slidesparse_weights/llama-3.2-1b \
    --quantization slidesparse \
    --load-format slidesparse \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100

# 对比 baseline
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --quantization fp8 \
    --input-len 128 \
    --output-len 128 \
    --num-prompts 100
```

### 6.3 精度评估

#### 6.3.1 PPL 测试

```python
"""
SlideSparse 精度评估 - PPL 测试
"""

from lm_eval import evaluator
from lm_eval.models.vllm_causallms import VLLM

def evaluate_ppl(model_path, quantization=None, load_format="auto"):
    """评估模型的 PPL"""
    model = VLLM(
        pretrained=model_path,
        quantization=quantization,
        load_format=load_format,
    )
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=["wikitext"],
        num_fewshot=0,
    )
    
    return results["results"]["wikitext"]["word_perplexity"]


# 测试
baseline_ppl = evaluate_ppl("meta-llama/Llama-3.2-1B-Instruct", "fp8")
slidesparse_ppl = evaluate_ppl(
    "./slidesparse_weights/llama-3.2-1b",
    "slidesparse",
    "slidesparse"
)

print(f"Baseline PPL: {baseline_ppl:.4f}")
print(f"SlideSparse PPL: {slidesparse_ppl:.4f}")
print(f"Degradation: {(slidesparse_ppl - baseline_ppl) / baseline_ppl * 100:.2f}%")
```

---

## 7. 附录：关键文件路径速查表

### 7.1 vLLM 核心文件

| 功能 | 文件路径 |
|------|---------|
| **入口点** | |
| LLM 类 | `vllm/entrypoints/llm.py` |
| CLI 入口 | `vllm/entrypoints/cli/main.py` |
| 吞吐测试 | `vllm/entrypoints/cli/benchmark/throughput.py` |
| **引擎** | |
| V1 引擎 | `vllm/v1/engine/llm_engine.py` |
| GPU 模型运行器 | `vllm/v1/worker/gpu_model_runner.py` |
| **模型加载** | |
| 加载器入口 | `vllm/model_executor/model_loader/__init__.py` |
| 基类 | `vllm/model_executor/model_loader/base_loader.py` |
| 默认加载器 | `vllm/model_executor/model_loader/default_loader.py` |
| 工具函数 | `vllm/model_executor/model_loader/utils.py` |
| **模型定义** | |
| Llama | `vllm/model_executor/models/llama.py` |
| Qwen2 | `vllm/model_executor/models/qwen2.py` |
| 模型注册 | `vllm/model_executor/models/registry.py` |
| **线性层** | |
| 线性层定义 | `vllm/model_executor/layers/linear.py` |
| **量化** | |
| 量化配置入口 | `vllm/model_executor/layers/quantization/__init__.py` |
| 基类 | `vllm/model_executor/layers/quantization/base_config.py` |
| FP8 量化 | `vllm/model_executor/layers/quantization/fp8.py` |
| **自定义算子** | |
| 算子绑定 | `vllm/_custom_ops.py` |
| **CUDA 源码** | |
| 量化 kernel | `csrc/quantization/` |
| 稀疏 kernel | `csrc/sparse/` |

### 7.2 SlideSparse 新增文件（计划）

| 功能 | 文件路径 |
|------|---------|
| **工具脚本** | |
| 权重预处理 | `tools/slidesparse/preprocess_weights.py` |
| 工具函数 | `tools/slidesparse/slidesparse_utils.py` |
| 算法搜索 | `tools/slidesparse/search_algorithms.py` |
| **加载器** | |
| SlideSparse 加载器 | `vllm/model_executor/model_loader/slidesparse_loader.py` |
| **量化配置** | |
| SlideSparse 配置 | `vllm/model_executor/layers/quantization/slidesparse.py` |
| **Kernel** | |
| 模块入口 | `vllm/model_executor/layers/quantization/slidesparse_kernels/__init__.py` |
| 融合量化+滑动 | `vllm/model_executor/layers/quantization/slidesparse_kernels/fused_quant_slide.py` |
| 稀疏 GEMM 封装 | `vllm/model_executor/layers/quantization/slidesparse_kernels/sparse_gemm.py` |
| 融合转置+反量化 | `vllm/model_executor/layers/quantization/slidesparse_kernels/fused_transpose_dequant.py` |
| **测试** | |
| Kernel 测试 | `tests/kernels/test_slidesparse.py` |
| 端到端测试 | `tests/quantization/test_slidesparse.py` |

### 7.3 关键函数签名速查

```python
# vllm/model_executor/layers/linear.py
class ColumnParallelLinear:
    def forward(self, input_) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        """执行列并行线性变换"""
        ...

# vllm/model_executor/layers/quantization/fp8.py
class Fp8LinearMethod:
    def apply(self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """执行 FP8 量化的线性变换"""
        ...

# vllm/_custom_ops.py
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    ...
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 量化"""
    ...

def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """CUTLASS scaled matrix multiplication"""
    ...

def cutlass_sparse_compress(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """压缩 2:4 稀疏矩阵"""
    ...

def cutlass_scaled_sparse_mm(
    a: torch.Tensor,
    bt_nzs: torch.Tensor,
    bt_meta: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """稀疏矩阵乘法"""
    ...
```

---

## 结语

本文档详细描述了 SlideSparse 在 vLLM 框架中的集成方案，包括：

1. **理论基础**：SlideSparse 的核心原理和创新点
2. **工具开发**：离线权重处理和在线 kernel 的开发流程
3. **工程实现**：基于 vLLM 的端到端实现方案
4. **关键工作**：Model Loader 和 Kernel 替换的详细指导

后续开发应按照文档中的检查清单逐步推进，建议优先级：

1. **高优先级**：离线工具开发（权重剪枝、滑动、压缩）
2. **高优先级**：Triton kernel 开发（fused_quant_slide、fused_transpose_dequant）
3. **中优先级**：SlideSparse 量化配置和 LinearMethod
4. **中优先级**：Model Loader 集成
5. **后续**：性能优化、更多模型支持、cuSPARSELt 集成



