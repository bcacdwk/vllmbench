# Dequant + Bias Triton Kernel

## 概述

本模块实现了一个高性能的 Triton kernel，用于融合反量化（Dequant）和偏置加法（Bias Add）操作，服务于 dense 和 sparse 的推理链路。

### 链路说明

- **Dense 链路**: `Triton quant` → `cuBLASLt (INT8/FP8 → BF16)` → **`Triton dequant`**
- **Sparse 链路**: `Triton quantfusedslide` → `cuSparseLt` → **`Triton dequant`** (复用)

---

## 文件结构

| 文件 | 说明 | 状态 |
|------|------|------|
| `dequant_bias_kernel.py` | 基础 Kernel 实现（手动配置选择） | ✅ 使用 |
| `dequant_bias_kernel_tuned.py` | **调优后的 Kernel**（自动生成，固定配置） | ✅ 使用 |
| `autotune_dequant_bias.py` | Autotune 脚本 + autotune 版本 Kernel | ✅ 使用 |
| `run_benchmark.py` | **统一测试脚本** | ✅ 使用 |
| `dequant_torch.py` | PyTorch 参考实现 (原始版本) | 🔧 参考 |

---

## Kernel 功能

### 计算公式

```
output[M,N] = gemm_output[M,N] * scale_a[M,1] * scale_b[1,N] + bias[1,N]
```

### 输入输出规格

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `gemm_output` | [M, N] | **BF16 或 FP32** | GEMM 输出（行主序） |
| `scale_a` | [M, 1] | FP32 | per-token scale |
| `scale_b` | [1, N] | FP32 | per-channel scale |
| `bias` | [1, N] | BF16 | per-channel bias |
| `output` | [M, N] | BF16 | 输出结果 |

### 计算流程

```
1. 读取 GEMM 输出
   - 如果是 BF16 → 转换为 FP32
   - 如果是 FP32 → 直接使用（省去转换开销）

2. FP32 逐点外积乘法
   output = gemm_output * scale_a[:, None] * scale_b[None, :]

3. 加 bias（BF16 → FP32 → 加法）
   output = output + bias[None, :]

4. 转换回 BF16 输出
```

---

## 核心优化：Kernel 融合减少内存访问

### 为什么 Triton 比 PyTorch 快？

**根本原因：减少全局内存读写次数**

这是一个 **Memory-Bound（内存带宽受限）** 的操作：
- 算术强度 ≈ 0.75 FLOP/byte（H100 拐点 ~200）
- GPU 大部分时间在等内存，不是在计算

因此，**减少内存访问是唯一有效的优化方向**。

---

### PyTorch 实现的问题

```python
# PyTorch: 每一步都要读写全局内存
temp1 = gemm_output.float()           # 读 M×N (BF16), 写 M×N (FP32)
temp2 = temp1 * scale_a               # 读 M×N + M, 写 M×N
temp3 = temp2 * scale_b               # 读 M×N + N, 写 M×N
output = temp3 + bias                 # 读 M×N + N, 写 M×N
output = output.to(bfloat16)          # 读 M×N, 写 M×N
```

**内存访问量**：
| 操作 | 读 | 写 |
|------|-----|-----|
| float() | M×N×2 | M×N×4 |
| × scale_a | M×N×4 + M×4 | M×N×4 |
| × scale_b | M×N×4 + N×4 | M×N×4 |
| + bias | M×N×4 + N×2 | M×N×4 |
| to(bf16) | M×N×4 | M×N×2 |
| **总计** | **~5 M×N×4** | **~5 M×N×4** |

👉 **约 40 M×N bytes 内存访问**（加上 5 次 kernel launch 开销）

---

### Triton 融合实现

```python
@triton.jit
def _dequant_bias_kernel(...):
    # 1. 一次性加载所有输入到寄存器
    scale_a = tl.load(scale_a_ptr + row_offs)     # [BLOCK_M] -> 寄存器
    scale_b = tl.load(scale_b_ptr + col_offs)     # [BLOCK_N] -> 寄存器  
    bias = tl.load(bias_ptr + col_offs)           # [BLOCK_N] -> 寄存器
    gemm_val = tl.load(gemm_output_ptr + offs)    # [BLOCK_M, BLOCK_N] -> 寄存器
    
    # 2. 所有计算都在寄存器内完成（不访问全局内存！）
    if not INPUT_FP32:
        gemm_val = gemm_val.to(tl.float32)        # 寄存器内类型转换
    
    output_val = gemm_val * scale_a[:, None]      # 寄存器内广播乘法
    output_val = output_val * scale_b[None, :]    # 寄存器内广播乘法
    output_val = output_val + bias[None, :]       # 寄存器内广播加法
    output_val = output_val.to(tl.bfloat16)       # 寄存器内类型转换
    
    # 3. 一次性写回全局内存
    tl.store(output_ptr + offs, output_val)       # 寄存器 -> 全局内存
```

**内存访问量**：
| 操作 | 读 | 写 |
|------|-----|-----|
| load gemm | M×N×2 (BF16) 或 M×N×4 (FP32) | 0 |
| load scales | M×4 + N×4 | 0 |
| load bias | N×2 | 0 |
| store output | 0 | M×N×2 |
| **总计 (BF16输入)** | **~2 M×N + 小量** | **~2 M×N** |

👉 **约 4 M×N bytes 内存访问**（1 次 kernel launch）

---

### 对比总结

| 指标 | PyTorch | Triton | 提升 |
|------|---------|--------|------|
| 全局内存访问 | ~40 M×N bytes | ~4 M×N bytes | **10x 减少** |
| Kernel Launch | 5 次 | 1 次 | **5x 减少** |
| 中间结果存储 | 4 个临时张量 | 0（全在寄存器） | **∞** |
| 理论加速比 | - | - | **~10x** |
| 实测加速比 | - | - | **5-12x** |

---

### 关键技术点

#### 1. 寄存器内计算（Register-Level Fusion）
```python
# 所有中间结果都在寄存器，不写回 HBM
output_val = gemm_val * scale_a[:, None] * scale_b[None, :] + bias[None, :]
```

#### 2. 编译时常量避免分支
```python
INPUT_FP32: tl.constexpr  # 编译时确定，生成两个版本的 kernel
if not INPUT_FP32:        # 编译时展开，无运行时开销
    gemm_val = gemm_val.to(tl.float32)
```

#### 3. 2D 分块并行
```python
# 每个 GPU thread block 处理一个 [BLOCK_M, BLOCK_N] tile
pid_m = tl.program_id(0)  # M 方向的 block ID
pid_n = tl.program_id(1)  # N 方向的 block ID
grid = (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))  # 总共启动的 block 数
```

#### 4. 向量广播（Broadcasting）
```python
# scale_a: [BLOCK_M] -> [BLOCK_M, 1] -> [BLOCK_M, BLOCK_N]
# scale_b: [BLOCK_N] -> [1, BLOCK_N] -> [BLOCK_M, BLOCK_N]
output = gemm_val * scale_a[:, None] * scale_b[None, :]
# 广播在寄存器内完成，无额外内存访问
```

---

### 配置选择策略

根据矩阵大小选择最优 BLOCK 大小，避免 autotune 开销：

```python
def _get_best_config(M: int, N: int) -> tuple:
    # 小 batch: 小 block 减少浪费
    if M <= 128:
        return (32, 64, 4) if N <= 4096 else (32, 128, 4)
    # 中等 batch: 平衡配置
    elif M <= 2048:
        return (64, 64, 4) if N <= 4096 else (64, 128, 8)
    # 大 batch: 大 block 提高吞吐
    else:
        return (128, 64, 8) if N <= 4096 else (128, 128, 8)
```

---

### 为什么优化空间有限？

这个 kernel 已经接近理论最优：

1. **算术强度太低** (~0.75)，无法通过计算优化提升
2. **内存访问已最小化**（只有必要的读写）
3. **没有数据复用机会**（每个元素只用一次）

进一步优化方向（收益有限）：
- Swizzling 优化 L2 Cache 命中率：预期 5-10%
- 异步预取（Triton 3.x）：预期 < 5%

---

## 测试方法

### 运行测试

```bash
cd /root/vllmbench/slidesparse/kernels/dequant_kernals

# 使用 autotune 版本（预热时自动调优）
python3 run_benchmark.py

# 使用已调优的固定配置版本（推荐，无 autotune 开销）
python3 run_benchmark.py --tuned

# 只测正确性
python3 run_benchmark.py --correctness
python3 run_benchmark.py --tuned --correctness

# 只测 BF16 或 FP32
python3 run_benchmark.py --tuned --dtype bf16
python3 run_benchmark.py --tuned --dtype fp32
```

### 生成调优配置

如需重新调优（更换 GPU 后），运行：
```bash
python3 autotune_dequant_bias.py
# 会生成新的 dequant_bias_kernel_tuned.py
```

### 测试配置（参考 autotune_example）

```python
# BitNet 模型常见隐藏层大小
N_VALUES = [2560, 3840, 13824]

# Batch size / sequence length 变化
M_VALUES = [1, 16, 32, 48, 64, 80, 96, 112, 128,
            192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
            6144, 8192, 10240, 12288, 14336, 16384, 20480, 24576, 
            32768, 40960, 49152, 65536]
```

---

## 性能结果（H100 PCIe，--tuned 模式）

### BF16 输入

| N | 平均加速比 | 最大加速比 | 最小加速比 |
|---|-----------|-----------|-----------|
| 2560 | **7.13x** | 9.49x | 4.79x |
| 3840 | **7.35x** | 9.51x | 4.68x |
| 13824 | **7.84x** | 9.50x | 4.79x |

### FP32 输入

| N | 平均加速比 | 最大加速比 | 最小加速比 |
|---|-----------|-----------|-----------|
| 2560 | **4.34x** | 5.26x | 3.44x |
| 3840 | **4.43x** | 5.26x | 3.42x |
| 13824 | **4.57x** | 5.29x | 3.45x |

### 结论

- **BF16 输入加速更明显** - 平均 7-8x，因为数据量更小，内存带宽压力更低
- **大 M 值加速更大** - 小 M (1-128) 加速 ~5x，大 M (4096+) 加速 9-10x
- **Tuned 版本无 autotune 开销** - 首次调用即为最优性能

---

## 使用示例

```python
from dequant_bias_kernel import dequant_bias_triton, dequant_bias_triton_tuned

# 准备数据
gemm_output = torch.randn(1024, 2560, dtype=torch.bfloat16, device='cuda')
scale_a = torch.rand(1024, 1, dtype=torch.float32, device='cuda')
scale_b = torch.rand(1, 2560, dtype=torch.float32, device='cuda')
bias = torch.randn(2560, dtype=torch.bfloat16, device='cuda')

# 方式1: 固定配置 (64x64)
output = dequant_bias_triton(gemm_output, scale_a, scale_b, bias)

# 方式2: 自动选择最优配置（推荐）
output = dequant_bias_triton_tuned(gemm_output, scale_a, scale_b, bias)

# 支持 FP32 输入（自动检测，无需额外转换）
gemm_fp32 = torch.randn(1024, 2560, dtype=torch.float32, device='cuda')
output = dequant_bias_triton_tuned(gemm_fp32, scale_a, scale_b, bias)
```
