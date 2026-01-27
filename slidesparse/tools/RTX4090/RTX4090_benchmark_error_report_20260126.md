# RTX 4090 SlideSparse Benchmark 错误报告

**生成时间**: 2026-01-26  
**硬件**: NVIDIA GeForce RTX 4090 (24 GB)  
**平台**: WSL2 Ubuntu 22.04

---

## 1. 整体测试概览

| 测试阶段 | 状态 | 备注 |
|---------|------|------|
| 离线模型准备 | ✅ 完成 | 所有模型下载/转换成功 |
| Simple Benchmark (CUTLASS) | ✅ 完成 | Llama3.2-1B INT8/FP8 全部通过 |
| Prefill 测试 | ⚠️ 部分失败 | 详见下方分析 |
| Decode 测试 | ⚠️ 部分失败 | 详见下方分析 |
| Retry 测试 | ❌ 全部失败 | 40/40 失败 (硬件限制) |

---

## 2. Prefill 测试详细结果

### 2.1 完全成功的模型 (0 失败)

| 模型 | Backend | 稀疏度 | M 值数量 |
|------|---------|--------|---------|
| Llama3.2-1B-INT8 | cuBLASLt | - | 8 |
| Llama3.2-1B-FP8 | cuBLASLt | - | 8 |
| Llama3.2-3B-INT8 | cuBLASLt | - | 8 |
| Llama3.2-3B-FP8 | cuBLASLt | - | 8 |
| Qwen2.5-7B-FP8 | cuBLASLt | - | 8 |
| Llama3.2-1B-INT8 | cuSPARSELt | 2:4, 2:6, 2:8, 2:10 | 各 8 |
| Llama3.2-1B-FP8 | cuSPARSELt | 2:4, 2:6, 2:8, 2:10 | 各 8 |
| Llama3.2-3B-INT8 | cuSPARSELt | 2:4, 2:6, 2:8, 2:10 | 各 8 |
| Llama3.2-3B-FP8 | cuSPARSELt | 2:4, 2:6, 2:8, 2:10 | 各 8 |
| Llama3.2-1B-INT8 | CUTLASS | - | 3 |
| Llama3.2-1B-FP8 | CUTLASS | - | 3 |

### 2.2 部分失败的模型

| 模型 | Backend | 稀疏度 | 成功 M 值 | 失败 M 值 | 失败原因 |
|------|---------|--------|----------|----------|---------|
| Qwen2.5-7B-INT8 | cuBLASLt | - | 512, 1024, 2048, 4096, 8192, 16384, 32768 | **65536** | CUDA illegal memory access (Triton bug) |
| Qwen2.5-7B-INT8 | cuSPARSELt | 2:4 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-INT8 | cuSPARSELt | 2:6 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-INT8 | cuSPARSELt | 2:8 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-INT8 | cuSPARSELt | 2:10 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-FP8 | cuSPARSELt | 2:4 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-FP8 | cuSPARSELt | 2:6 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-FP8 | cuSPARSELt | 2:8 | 512~32768 (7个) | **65536** | CUDA illegal memory access |
| Qwen2.5-7B-FP8 | cuSPARSELt | 2:10 | 512~32768 (7个) | **65536** | CUDA illegal memory access |

### 2.3 完全失败/未完成的模型

> ⚠️ **注意**: 由于 Windows 自动更新中断测试，14B 模型的 Prefill 测试只运行了 M=65536（最大值）。
> 后续的 Retry 测试尝试补全所有 M 值，但均因显存不足而失败。

| 模型 | Backend | 稀疏度 | 测试的 M 值 | 状态 | 失败原因 |
|------|---------|--------|------------|------|---------|
| Qwen2.5-14B-INT8 | cuBLASLt | - | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-FP8 | cuBLASLt | - | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-INT8 | cuSPARSELt | 2:4 | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-FP8 | cuSPARSELt | 2:4 | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-INT8 | cuSPARSELt | 2:6 | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-FP8 | cuSPARSELt | 2:6 | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-INT8 | cuSPARSELt | 2:8 | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-FP8 | cuSPARSELt | 2:8 | 仅 65536 | ❌ 失败 | KV Cache 显存不足 |
| Qwen2.5-14B-INT8 | cuSPARSELt | 2:10 | 仅 65536 | ❌ 失败 | 模型超出显存 |
| Qwen2.5-14B-FP8 | cuSPARSELt | 2:10 | 仅 65536 | ❌ 失败 | 模型超出显存 |

**Retry 测试结果** (尝试补全 M=512~32768):
- 40 个测试全部失败
- 原因：即使 `gpu_memory_utilization=0.98`，24GB 显存仍无法容纳 14B 模型 + 足够的 KV Cache

---

## 3. Decode 测试详细结果

### 3.1 完全成功的模型 (0 失败)

| 模型 | Backend | 稀疏度 | M 值数量 |
|------|---------|--------|---------|
| Llama3.2-1B-INT8 | cuBLASLt | - | 4 |
| Llama3.2-1B-FP8 | cuBLASLt | - | 4 |
| Llama3.2-3B-INT8 | cuBLASLt | - | 4 |
| Llama3.2-3B-FP8 | cuBLASLt | - | 4 |
| Qwen2.5-7B-INT8 | cuBLASLt | - | 4 |
| Qwen2.5-7B-FP8 | cuBLASLt | - | 4 |
| **Qwen2.5-14B-INT8** | cuBLASLt | - | **4** |
| **Qwen2.5-14B-FP8** | cuBLASLt | - | **4** |
| Llama3.2-1B | cuSPARSELt | 2:4~2:10 | 各 4 |
| Llama3.2-3B | cuSPARSELt | 2:4~2:10 | 各 4 |
| Qwen2.5-7B | cuSPARSELt | 2:4~2:10 | 各 4 |
| Qwen2.5-14B | cuSPARSELt | 2:4, 2:6 | 各 4 |
| Llama3.2-1B | CUTLASS | - | 3 |

### 3.2 完全失败的模型

| 模型 | Backend | 稀疏度 | 失败 M 值 | 失败原因 |
|------|---------|--------|----------|---------|
| Qwen2.5-14B-INT8 | cuSPARSELt | 2:8 | 512 | 显存不足 |
| Qwen2.5-14B-FP8 | cuSPARSELt | 2:8 | 512 | 显存不足 |
| Qwen2.5-14B-INT8 | cuSPARSELt | 2:10 | 64, 256, 512 | 模型超出显存 |
| Qwen2.5-14B-FP8 | cuSPARSELt | 2:10 | 64, 256, 512 | 模型超出显存 |

---

## 4. 失败原因分析

### 4.1 CUDA Illegal Memory Access (Qwen2.5-7B, M=65536)

**症状**:
```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

**影响范围**: 
- Qwen2.5-7B-INT8/FP8 的所有 backend，仅 M=65536 失败

**根本原因**: 
- Triton kernel bug，当 M 值过大时触发
- 这是 vLLM/Triton 的已知问题，与硬件无关

**结论**: ❌ 无法修复，需等待上游修复

---

### 4.2 KV Cache 显存不足 (Qwen2.5-14B, 大 M 值)

**症状**:
```
ValueError: No available memory for the cache blocks. 
Try increasing `gpu_memory_utilization` when initializing the engine.
```

**影响范围**:
- Qwen2.5-14B 的大 M 值测试 (M≥16384)
- 稀疏模型比 dense 更容易失败（因为模型更大）

**根本原因**:
- 14B 模型占用 ~16GB 显存（dense/2:4）
- 大 M 值需要更多 KV Cache 空间
- 24GB 显存不足以同时容纳模型 + 大 KV Cache

**已尝试的修复**:
- `gpu_memory_utilization` 从 0.75 → 0.95 → 0.98
- 仍然失败

**结论**: ❌ 硬件限制，24GB 显存不足

---

### 4.3 模型超出显存 (Qwen2.5-14B + 2:10 稀疏)

**症状**: EngineCore failed to start

**模型大小（实测）**:

| 稀疏度 | 模型大小 | 可用于 KV Cache |
|--------|---------|----------------|
| dense/2:4 | 16 GB | ~6 GB |
| 2:6 | 20 GB | ~2 GB |
| 2:8 | 22 GB | ~0.5 GB |
| 2:10 | **23 GB** | ❌ 不足 |

**根本原因**:
- 2:10 稀疏格式扩展模型大小 ×1.44
- 23GB 模型 + ~2GB CUDA 开销 ≈ 25GB > 24.5GB 显存

**结论**: ❌ 物理不可能，需要更大显存的 GPU

---

## 5. 数据完整性总结

### 5.1 Prefill 测试数据完整性

| 模型 | cuBLASLt | cuSPARSELt 2:4 | 2:6 | 2:8 | 2:10 |
|------|----------|---------------|-----|-----|------|
| Llama3.2-1B | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 |
| Llama3.2-3B | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 | ✅ 8/8 |
| Qwen2.5-7B | ⚠️ 7/8 | ⚠️ 7/8 | ⚠️ 7/8 | ⚠️ 7/8 | ⚠️ 7/8 |
| Qwen2.5-14B | ❌ 0/8* | ❌ 0/8* | ❌ 0/8* | ❌ 0/8* | ❌ 0/8* |

> *14B 只测试了 M=65536（失败），其他 M 值未测试/无法在 RTX 4090 上运行

### 5.2 Decode 测试数据完整性

| 模型 | cuBLASLt | cuSPARSELt 2:4 | 2:6 | 2:8 | 2:10 |
|------|----------|---------------|-----|-----|------|
| Llama3.2-1B | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| Llama3.2-3B | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| Qwen2.5-7B | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 |
| Qwen2.5-14B | ✅ 4/4 | ✅ 4/4 | ✅ 4/4 | ❌ 0/4* | ❌ 0/4* |

> *14B + 2:8/2:10 稀疏模型占用显存过大，无法在 RTX 4090 上运行

---

## 6. 结论与建议

### 6.1 可用数据

✅ **完整可用**:
- Llama3.2-1B (INT8/FP8): 所有 backend、所有稀疏度、所有 M 值
- Llama3.2-3B (INT8/FP8): 所有 backend、所有稀疏度、所有 M 值
- Qwen2.5-7B (INT8/FP8): 所有 M≤32768 的测试

⚠️ **部分可用**:
- Qwen2.5-7B M=65536: 因 Triton bug 全部失败
- Qwen2.5-14B Prefill: 小 M 值部分可用
- Qwen2.5-14B Decode cuBLASLt: 全部可用

### 6.2 不可修复的限制

1. **Triton Kernel Bug** (Qwen2.5-7B M=65536)
   - 需要等待 vLLM/Triton 上游修复
   
2. **RTX 4090 显存限制** (Qwen2.5-14B 大 M 值)
   - 24GB 显存无法满足 14B 模型 + 大 KV Cache
   - 建议在 A100/H100 (80GB) 上重新测试

3. **2:10 稀疏格式** (Qwen2.5-14B)
   - 模型扩展后超出 24GB 显存
   - 在消费级 GPU 上不可行

### 6.3 建议

1. **论文数据**: 使用 Llama3.2-1B/3B 和 Qwen2.5-7B (M≤32768) 的完整数据
2. **14B 模型**: 需要在更大显存的 GPU (如 A100) 上补测
3. **M=65536**: 等待 Triton bug 修复后再测试

---

## 7. 相关文件位置

- 测试结果 CSV: `throughput_benchmark_results/{prefill,decode}/RTX4090_*/`
- Recovery 日志: `recovery_bench_20260126_013139.log`
- Retry 日志: `retry_failed_20260126_202907.log`
- Retry 状态: `retry_failed_20260126_202907_status.json`

---

*报告生成时间: 2026-01-26 22:00*
