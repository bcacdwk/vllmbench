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

**剪枝算法**：

**magnitude 模式（按绝对值大小剪枝）**：
1. 将权重张量 `[N, K]` 重塑为 `[N, K/group_size, group_size]`，形成多个稀疏组
2. 在每个组内，计算每个元素的绝对值
3. 使用 `torch.topk` 选出绝对值最大的 `num_nonzeros` 个位置
4. 生成二值掩码（被选中位置为1，其余为0）
5. 将掩码与原权重相乘，得到剪枝后的稀疏权重

**random 模式（随机剪枝）**：
1. 同样将权重重塑为组
2. 在每个组内随机选择 `num_nonzeros` 个位置保留
3. 其余位置置零

**关键参数计算**：
- `num_zeros_per_group = int(sparsity_ratio * group_size)` —— 每组需要置零的元素数
- `num_nonzeros = group_size - num_zeros_per_group` —— 每组保留的非零元素数

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

**滑动拓展算法**：

**滑动拓展流程**：
1. **Padding 预处理**：确保 K 能被源稀疏窗口大小 `L_src` 整除，必要时在末尾补零
2. **参数计算**：
   - `stride = L_tgt - Z_tgt`（滑动步长 = 目标窗口中非零元素个数 = 2）
   - `num_windows = (L_src - Z) // stride`（窗口数 = 源窗口非零元素数 / 每窗口非零数）
3. **权重分组**：将权重重塑为 `[N, num_groups, L_src]`
4. **滑动提取**：对每个窗口 `i`，提取位置 `[i*stride, i*stride+L_tgt)` 的元素
5. **拼接输出**：将所有窗口的结果拼接，形成 `[N, K']` 的扩展权重

**示例（2:8 → 2:4）**：
- 源稀疏：L_src=8, Z=2（每8个元素有2个零，6个非零）
- 目标稀疏：L_tgt=4, Z_tgt=2（每4个元素有2个零，2个非零）
- stride = 4 - 2 = 2
- num_windows = (8 - 2) / 2 = 3 个窗口
- 扩展比例：3 × 4 / 8 = 1.5x

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

**关键 API 调用思路**：

**压缩流程**：
1. 将权重转换为目标数据类型（FP8/INT8），并确保内存连续
2. 由于 cuSPARSELt 要求列主序输入，需要对权重进行转置：`weight_col_major = weight.t().contiguous()`
3. 调用 cuSPARSELt 的压缩函数，指定稀疏类型为 `'2:4'`
4. 返回压缩后的权重张量和元数据（包含稀疏索引信息）

**注意事项**：
- cuSPARSELt 的压缩会将权重大小减半（2:4 稀疏每 4 个元素只存储 2 个非零值）
- 元数据张量用于记录非零元素的位置信息
- vLLM 已有类似的 CUTLASS 稀疏压缩实现可供参考：`vllm/_custom_ops.py` 中的 `cutlass_sparse_compress()` 函数

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

**搜索算法**：

**搜索算法**：
1. **遍历模型线性层**：获取每个线性层的 `(N, K)` 尺寸
2. **遍历 M 值范围**：针对不同的 batch size（如 1, 4, 8, 16, 32, ...）
3. **算法搜索**：
   - 创建测试输入张量 A `[M, K]` 和权重张量 B `[N, K]`
   - 遍历 cuSPARSELt 支持的所有算法 ID
   - 对每个算法执行多次（如 100 次）计时测试
   - 记录最快算法的 ID
4. **保存配置**：将 `{(M,N,K): best_algo_id}` 字典序列化为 JSON 文件

**JSON 文件格式示例**：
```json
{
  "1,4096,4096": 0,
  "4,4096,4096": 2,
  "16,4096,4096": 1,
  "32,4096,4096": 3
}
```

**使用方式**：在线推理时，根据当前的 MNK 尺寸查表获取最优算法 ID

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

**Triton Kernel 设计要点**：

**核心设计目标**：将 slide 操作融合在 quant 流程中，掩盖额外的读写开销

**实现流程**：
1. **块级并行**：按 `(BLOCK_M, BLOCK_K)` 划分输入，每个 Triton program 处理一个块
2. **加载输入**：从 `[M, K]` BF16 输入中加载当前块的数据
3. **动态量化**：
   - 计算当前块（或整行）的最大绝对值 `x_max`
   - 计算 scale：`scale = x_max / 448.0`（448 是 FP8 E4M3 的最大表示值）
   - 量化：`x_quant = (x / scale).to(FP8)`
4. **滑动重排**：
   - 计算当前块对应的源稀疏组和目标位置
   - 对每个滑动窗口，从量化结果中提取 `L_tgt` 个元素
   - 写入到扩展后的目标位置
5. **存储 scale**：per-token 模式下保存每行的 scale 值

**性能优化要点**：
- 使用 `@triton.autotune` 装饰器搜索最优的 `BLOCK_M`、`BLOCK_K` 配置
- 合并内存访问，尽量使用连续的内存读写
- 考虑 per-token 和 per-tensor 两种量化粒度的实现

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

**调用接口设计**：

**GEMM 调用流程**：
1. 根据当前激活的 `(M, K')` 尺寸和权重的 `N` 维度，构建查询 key
2. 从预先生成的 JSON 配置中查表获取最优算法 ID
3. 调用 cuSPARSELt 的稀疏矩阵乘法 API
4. 返回 `[N, M]` 行主序的乘法结果

**重要说明**：
- cuSPARSELt 当前版本输出为行主序 `[N, M]`，后续需要转置
- 输出数据类型取决于输入：INT8 输入产生 INT32 累加结果，FP8 输入产生 FP32 累加结果
- vLLM 已有的 CUTLASS 稀疏 GEMM 可作为替代方案：`vllm/_custom_ops.py` 中的 `cutlass_scaled_sparse_mm()` 函数，该函数直接输出 `[M, N]` 格式，无需转置

#### 2.3.3 融合转置+反量化算子 (Fused Transpose + Dequant)

**功能**：将 GEMM 输出从 `[N, M]` 转置为 `[M, N]`，并反量化为 BF16

**实现语言**：Triton

**输入**：
- GEMM 输出：`[N, M]`，INT32/FP32
- Scale 参数：scale_a, scale_b

**输出**：
- 反量化结果：`[M, N]`，BF16

**Triton Kernel 设计要点**：

**核心功能**：将 `[N, M]` 的 GEMM 输出转置为 `[M, N]`，并同时完成反量化

**实现流程**：
1. **块级并行**：按 `(BLOCK_M, BLOCK_N)` 划分输出，每个 program 处理一个输出块
2. **转置读取**：从输入 `[N, M]` 读取时，使用转置的索引模式：`input[n, m]` → `output[m, n]`
3. **加载 scale**：
   - `scale_a`：激活量化的 scale（per-token 时为 `[M]` 维度）
   - `scale_b`：权重量化的 scale（per-channel 时为 `[N]` 维度）
4. **反量化计算**：`x_dequant = x.to(float32) * scale_a * scale_b`
5. **类型转换**：将 FP32 结果转换为 BF16 输出
6. **存储输出**：按 `[M, N]` 行主序写出

**广播规则**：
- per-token scale_a：广播到每行的所有列
- per-channel scale_b：广播到每列的所有行
- 计算时需要正确处理 scale 的维度广播

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

**脚本功能**：将原始 HuggingFace 模型的权重进行 剪枝 → 滑动 → 压缩 的完整预处理流程

**命令行使用方式**：
```bash
python tools/slidesparse/preprocess_weights.py \
    --input-model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./slidesparse_weights/llama-3.2-1b \
    --sparsity 2:8 \
    --prune-mode magnitude
```

**实现要点**：
1. **加载原始模型**：使用 `transformers.AutoModelForCausalLM.from_pretrained()` 加载 HuggingFace 模型
2. **遍历线性层权重**：识别 `qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj` 四种线性层
3. **依次执行处理**：
   - 调用 `prune_weight()` 进行结构化剪枝
   - 调用 `slide_weight()` 进行滑动拓展
   - 调用 `compress_weight_cusparselt()` 进行 2:4 稀疏压缩
4. **保存输出**：
   - 使用 `safetensors.torch.save_file()` 保存处理后的权重到 `model.safetensors`
   - 保存元数据到 `slidesparse_config.json`，记录稀疏格式、处理过的层名称、形状变化等信息
   - 复制原模型的 `config.json` 和 tokenizer 文件

**步骤 2：创建 SlideSparse ModelLoader**

创建文件：`vllm/model_executor/model_loader/slidesparse_loader.py`

**类设计**：继承 `BaseModelLoader`（定义于 `vllm/model_executor/model_loader/base_loader.py`）

**核心方法设计**：

1. **`__init__(self, load_config)`**：
   - 调用父类初始化
   - 初始化 `slidesparse_config` 属性用于存储配置

2. **`_load_slidesparse_config(self, model_path)`**：
   - 读取模型目录下的 `slidesparse_config.json` 配置文件
   - 解析稀疏格式、处理过的层列表等信息

3. **`download_model(self, model_config)`**：
   - SlideSparse 模型通常是本地预处理的，无需下载
   - 如需下载，可复用 `DefaultModelLoader` 的逻辑

4. **`load_weights(self, model, model_config)`**：
   - 从 `slidesparse_config.json` 加载配置
   - 使用 `weight_utils.safetensors_weights_iterator()` 遍历权重文件
   - 对于 SlideSparse 处理过的权重（通过配置识别），执行特殊加载逻辑
   - 对于普通权重，使用标准加载方式

5. **`_is_slidesparse_weight(self, name)`**：
   - 检查权重名称是否在配置的 `processed_layers` 列表中

6. **`_load_slidesparse_weight(self, model, name, weight)`**：
   - 找到模型中对应的参数
   - 验证形状兼容性（注意 K 维度扩展）
   - 复制权重数据并设置 `slidesparse` 标记

**关键依赖**：
- `vllm/model_executor/model_loader/base_loader.py`：`BaseModelLoader` 基类
- `vllm/model_executor/model_loader/weight_utils.py`：`safetensors_weights_iterator()` 函数

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

**类设计**：
- `SlideSparseConfig`：继承 `QuantizationConfig`（定义于 `vllm/model_executor/layers/quantization/base_config.py`）
- `SlideSparseLinearMethod`：继承 `LinearMethodBase`（定义于 `vllm/model_executor/layers/linear.py`）

**SlideSparseConfig 核心属性**：
- `sparsity`：稀疏格式字符串，如 `"2:8"`
- `sparsity_z` / `sparsity_l`：解析后的 Z 和 L 值
- `activation_dtype`：激活量化类型，`"fp8"` 或 `"int8"`
- `algo_config_path`：算法配置 JSON 文件路径
- `algo_config`：加载后的算法配置字典

**SlideSparseConfig 核心方法**：
- `get_name()` → 返回 `"slidesparse"`
- `get_supported_act_dtypes()` → 返回 `[torch.bfloat16, torch.float16]`
- `get_min_capability()` → 返回 `80`（需要 Ampere 或更新架构）
- `get_quant_method(layer, prefix)` → 对 `LinearBase` 实例返回 `SlideSparseLinearMethod`

**SlideSparseLinearMethod 核心方法**：

1. **`create_weights()`**：创建 SlideSparse 权重参数
   - 计算扩展比例：`expand_ratio = (num_windows * 4) / L`
   - 创建压缩权重：形状为 `[output_size, expanded_input_size // 2]`（2:4 压缩减半）
   - 创建权重 scale：形状为 `[output_size]`
   - 在 layer 上设置 `input_size_expanded` 和 `sparsity_config` 属性

2. **`apply(layer, x, bias)`**：执行 SlideSparse 线性变换
   - 步骤 1：调用 `fused_quant_slide()` → BF16 `[M, K]` → FP8 `[M, K']`
   - 步骤 2：查表获取算法 ID，调用 `sparse_gemm_cusparselt()` → `[M, N]`
   - 步骤 3：调用 `fused_transpose_dequant()` → BF16 `[M, N]`
   - 添加 bias（如有）

**步骤 5：注册 SlideSparse 量化配置**

修改文件：`vllm/model_executor/layers/quantization/__init__.py`

**修改内容**：
1. 在 `QuantizationMethods` 类型定义中添加 `"slidesparse"`
2. 在 `get_quantization_config()` 函数中：
   - 添加导入：`from .slidesparse import SlideSparseConfig`
   - 在 `method_to_config` 字典中添加：`"slidesparse": SlideSparseConfig`

**注意**：vLLM 还提供了 `register_quantization_config()` 装饰器（行 50-94），可以用于动态注册自定义量化配置，无需修改源文件。

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

**实现要点**：

**Triton Kernel 设计**：
1. 使用 `@triton.autotune` 装饰器搜索最优的 `BLOCK_M`、`BLOCK_K` 配置
2. 按源稀疏组进行并行划分：`grid = (cdiv(M, BLOCK_M), cdiv(K, src_L))`
3. 每个 program 处理一个 `[BLOCK_M, src_L]` 大小的块

**Kernel 内部流程**：
1. **加载输入**：从 BF16 输入中加载当前源稀疏组的数据
2. **计算 scale**：
   - per-token 模式：计算每行的 `max(abs(x))`，除以 448.0（FP8 E4M3 最大值）
   - per-tensor 模式：使用预计算的全局 scale
3. **量化**：`x_quant = x / (scale + eps)`，裁剪到 `[-448, 448]` 范围
4. **滑动写出**：对每个窗口 `w`：
   - 提取源位置 `[w*stride, w*stride+tgt_L)` 的数据
   - 写入目标位置 `dst_group_start + w*tgt_L`

**Python 封装函数** `fused_quant_slide()`：
- 输入：`input [M, K]` BF16，稀疏格式参数
- 输出：`output [M, K']` FP8，`scale [M]` 或 `[1]`
- 自动计算扩展维度 `K' = (K / L_src) * num_windows * L_tgt`

#### 5.2.3 Sparse GEMM 封装

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/sparse_gemm.py`

**功能**：封装 CUTLASS 或 cuSPARSELt 的稀疏 GEMM 调用

**核心函数**：

1. **`sparse_gemm_cutlass()`**：
   - 直接调用 vLLM 已有的 `ops.cutlass_scaled_sparse_mm()`（定义于 `vllm/_custom_ops.py` 第 961-1005 行）
   - 输入：稠密激活 `[M, K']`、压缩权重 `[N, K'/2]`、稀疏元数据 `[N, K'/8]`、scales
   - 输出：`[M, N]` 格式，无需转置
   - 这是**推荐的实现方式**，因为已经集成在 vLLM 中

2. **`sparse_gemm_cusparselt()`**：
   - 需要额外实现 cuSPARSELt 的 C++ 绑定
   - 输出为行主序 `[N, M]`，需要后续转置
   - 实现步骤：
     - 在 `csrc/quantization/sparse/` 目录创建 C++ 绑定代码
     - 在 `csrc/torch_bindings.cpp` 中注册 `torch.ops._C` 函数
     - 在 `vllm/_custom_ops.py` 中添加 Python 封装

3. **`compress_weight_for_sparse_gemm()`**：
   - 调用 `ops.cutlass_sparse_compress(weight.t().contiguous())`（定义于 `vllm/_custom_ops.py` 第 920-958 行）
   - 输入：满足 2:4 稀疏的权重 `[N, K]`
   - 输出：非零元素 `[K, N/2]` 和稀疏元数据 `[K, N/8]`

**vLLM 已有 API 参考**（`vllm/_custom_ops.py`）：
- `cutlass_sparse_compress(a)` → `(a_nzs, a_meta)`
- `cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype, bias)`

#### 5.2.4 Fused Transpose + Dequant Kernel

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/fused_transpose_dequant.py`

**实现要点**：

**Triton Kernel 设计**：
1. 使用 `@triton.autotune` 搜索最优的 `BLOCK_M`、`BLOCK_N` 配置
2. 按输出维度划分：`grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))`

**Kernel 内部流程**：
1. **转置读取**：从 `[N, M]` 输入读取，使用转置索引 `input[n, m]`
2. **加载 scale**：
   - `scale_a`：per-token 时为 `[M]`，per-tensor 时为 scalar
   - `scale_b`：per-channel 时为 `[N]`，per-tensor 时为 scalar
3. **反量化**：`x_dequant = x.to(float32) * scale_a * scale_b`
4. **写出**：将 BF16 结果写入 `[M, N]` 输出

**Python 封装函数** `fused_transpose_dequant()`：
- 输入：GEMM 输出 `[N, M]` (INT32/FP32)、`scale_a`、`scale_b`
- 输出：`[M, N]` BF16
- 自动检测 scale 类型（根据 numel）

**注意**：如果使用 CUTLASS sparse GEMM（输出直接是 `[M, N]`），则不需要转置操作，可以简化为纯反量化 kernel。

#### 5.2.5 Kernel 模块初始化

创建文件：`vllm/model_executor/layers/quantization/slidesparse_kernels/__init__.py`

**导出函数列表**：
- `fused_quant_slide`：融合量化+滑动
- `sparse_gemm_cutlass`：CUTLASS 稀疏 GEMM
- `sparse_gemm_cusparselt`：cuSPARSELt 稀疏 GEMM（待实现）
- `compress_weight_for_sparse_gemm`：权重压缩
- `fused_transpose_dequant`：融合转置+反量化

### 5.3 SlideSparseLinearMethod 完整设计

更新 `vllm/model_executor/layers/quantization/slidesparse.py` 中的方法：

**`apply()` 方法设计**：

1. **Fused Quant + Slide**：调用 `fused_quant_slide(x, src_sparsity, tgt_sparsity)`
   - 输入：BF16 `[M, K]`
   - 输出：FP8 `[M, K']` 和 scale

2. **Sparse GEMM**：调用 `sparse_gemm_cutlass(x_quant, weight_nzs, weight_meta, scale_a, scale_b)`
   - 使用 CUTLASS 稀疏 GEMM（输出直接是 `[M, N]`）
   - 无需额外转置

3. **添加 bias**（如有）

**`process_weights_after_loading()` 方法**：
- 在权重加载后调用
- 调用 `compress_weight_for_sparse_gemm()` 压缩权重
- 删除原 `layer.weight`，注册 `weight_nzs` 和 `weight_meta` buffer

### 5.4 启用 SlideSparse 的方式

#### 5.4.1 通过量化配置启用

**方式 1：使用预处理后的模型**
```python
from vllm import LLM

llm = LLM(
    model="./slidesparse_weights/llama-3.2-1b",
    quantization="slidesparse",
    load_format="slidesparse",
)
```

**方式 2：使用配置文件**
```python
llm = LLM(
    model="meta-llama/Llama-3.2-1B-Instruct",
    quantization="slidesparse",
    quantization_param_path="./slidesparse_config.json",
)
```

#### 5.4.2 运行时条件分支（可选方案）

如果需要在运行时根据条件选择是否使用 SlideSparse，可以添加环境变量控制：

```bash
export VLLM_USE_SLIDESPARSE=1
```

在代码中检查该环境变量，根据条件选择执行路径。

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

**测试类 `TestFusedQuantSlide`**：
- `test_output_shape`：验证不同 M、K、sparsity 组合的输出形状正确性
- `test_quantization_accuracy`：验证量化精度在可接受范围内

**测试类 `TestSparseGemm`**：
- `test_correctness`：验证稀疏 GEMM 的计算正确性
- 对比稀疏 GEMM 和 dense GEMM 的结果（考虑量化误差）

**辅助函数**：
- `prune_to_2_4(weight)`：将权重强制剪枝为 2:4 稀疏格式

**测试命令**：
```bash
pytest tests/kernels/test_slidesparse.py -v
```

### 6.2 端到端测试

#### 6.2.1 吞吐量测试

**测试流程**：
1. 使用预处理脚本生成 SlideSparse 权重
2. 运行 SlideSparse 模型的吞吐测试
3. 运行 baseline（FP8）模型的吞吐测试
4. 对比加速比

**测试命令**：
```bash
# 预处理权重
python tools/slidesparse/preprocess_weights.py \
    --input-model meta-llama/Llama-3.2-1B-Instruct \
    --output-dir ./slidesparse_weights/llama-3.2-1b \
    --sparsity 2:8

# SlideSparse 吞吐测试
vllm bench throughput \
    --model ./slidesparse_weights/llama-3.2-1b \
    --quantization slidesparse \
    --load-format slidesparse \
    --input-len 128 --output-len 128 --num-prompts 100

# Baseline 吞吐测试
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --quantization fp8 \
    --input-len 128 --output-len 128 --num-prompts 100
```

### 6.3 精度评估

#### 6.3.1 PPL 测试

**使用 lm-eval 进行 PPL 评估**：

**评估流程**：
1. 使用 `lm_eval.models.vllm_causallms.VLLM` 加载模型
2. 在 `wikitext` 数据集上评估 word_perplexity
3. 对比 baseline 和 SlideSparse 模型的 PPL

**评估命令**（使用 lm-evaluation-harness）：
```bash
# Baseline PPL
lm_eval --model vllm \
    --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,quantization=fp8 \
    --tasks wikitext --num_fewshot 0

# SlideSparse PPL
lm_eval --model vllm \
    --model_args pretrained=./slidesparse_weights/llama-3.2-1b,quantization=slidesparse,load_format=slidesparse \
    --tasks wikitext --num_fewshot 0
```

**预期结果**：
- PPL 增加应在可接受范围内（如 < 5%）
- 不同稀疏度（2:6, 2:8, 2:10）会有不同的精度损失

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

**vllm/model_executor/layers/linear.py**：
- `ColumnParallelLinear.forward(input_)` —— 行 557-575
- `RowParallelLinear.forward(input_)` —— 行 1388-1416
- `LinearBase` 基类 —— 行 243-288
- `UnquantizedLinearMethod.apply(layer, x, bias)` —— 行 234-240

**vllm/model_executor/layers/quantization/fp8.py**：
- `Fp8LinearMethod.apply(layer, x, bias)` —— 行 610-687
- `Fp8LinearMethod.process_weights_after_loading(layer)` —— 行 ~500-605

**vllm/_custom_ops.py**：
- `scaled_fp8_quant(input, scale, ...)` —— 行 1678-1735
- `cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)` —— 行 828-876
- `cutlass_sparse_compress(a)` —— 行 920-958
- `cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype, bias)` —— 行 961-1005

**vllm/model_executor/model_loader/__init__.py**：
- `get_model(vllm_config, model_config)` —— 行 126-132
- `get_model_loader(load_config)` —— 行 118-123
- `register_model_loader(load_format)` —— 行 65-115

**vllm/model_executor/layers/quantization/__init__.py**：
- `get_quantization_config(quantization)` —— 行 97-171
- `register_quantization_config(quantization)` —— 行 50-94

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



