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

