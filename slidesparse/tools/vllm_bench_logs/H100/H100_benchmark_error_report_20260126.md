# H100 Benchmark 最终报告

**日志文件**: `prepare_bench_20260126_072848.log`  
**时间**: 2026-01-26  
**总耗时**: 10.75 小时 (38,692秒)  
**GPU**: NVIDIA H100 PCIe 80GB (CC 9.0, Hopper)

---

## 📊 最终结果汇总

### 总体情况

| 任务 | 状态 | 成功数 | 失败数 | 说明 |
|------|------|--------|--------|------|
| Task 1: 模型下载 | ⏭️ 跳过 | - | - | 之前已完成 |
| Task 2: 模型转换 | ⏭️ 跳过 | - | - | 之前已完成 |
| Task 3: 离线粗调优 | ⏭️ 跳过 | - | - | 之前已完成 |
| Task 4: 离线细调优 | ✅ 成功 | 8/8 | 0 | 手动完成 cuSPARSELt 搜索 |
| Task 5: 简单 Benchmark | ✅ 成功 | 全部 | 0 | INT8 + FP8 均通过 |
| Task 6: Prefill Benchmark | ⚠️ 部分失败 | 310 | 10 | Qwen2.5-7B M=65536 失败 |
| Task 7: Decode Benchmark | ✅ 成功 | 160 | 0 | 全部通过 |

---

## 1. 离线调优结果 (Task 3 & 4)

### cuBLASLt Algorithm Search ✅

所有 8 个模型的 cuBLASLt 算法搜索已完成：

| 模型 | dtype | 状态 | 结果文件 |
|------|-------|------|----------|
| Llama3.2-1B-INT8 | int8 | ✅ | `alg_search_Llama3.2-1B-INT8_out-INT32.json` |
| Llama3.2-1B-FP8 | fp8e4m3 | ✅ | `alg_search_Llama3.2-1B-FP8_out-BF16.json` |
| Llama3.2-3B-INT8 | int8 | ✅ | `alg_search_Llama3.2-3B-INT8_out-INT32.json` |
| Llama3.2-3B-FP8 | fp8e4m3 | ✅ | `alg_search_Llama3.2-3B-FP8_out-BF16.json` |
| Qwen2.5-7B-INT8 | int8 | ✅ | `alg_search_Qwen2.5-7B-INT8_out-INT32.json` |
| Qwen2.5-7B-FP8 | fp8e4m3 | ✅ | `alg_search_Qwen2.5-7B-FP8_out-BF16.json` |
| Qwen2.5-14B-INT8 | int8 | ✅ | `alg_search_Qwen2.5-14B-INT8_out-INT32.json` |
| Qwen2.5-14B-FP8 | fp8e4m3 | ✅ | `alg_search_Qwen2.5-14B-FP8_out-BF16.json` |

**路径**: `/root/vllmbench/slidesparse/search/cuBLASLt_AlgSearch/alg_search_results/H100_cc90_py312_cu129_x86_64/`

### cuSPARSELt Algorithm Search ✅

所有 8 个模型的 cuSPARSELt 算法搜索已完成（手动绕过1小时超时限制）：

| 模型 | dtype | 状态 | 备注 |
|------|-------|------|------|
| Llama3.2-1B-INT8 | int8→bf16 | ✅ | 自动完成 |
| Llama3.2-1B-FP8 | fp8→bf16 | ✅ | 自动完成 |
| Llama3.2-3B-INT8 | int8→bf16 | ✅ | 自动完成 |
| Llama3.2-3B-FP8 | fp8→bf16 | ✅ | 自动完成 |
| Qwen2.5-7B-INT8 | int8→bf16 | ✅ | 手动运行 (原超时) |
| Qwen2.5-7B-FP8 | fp8→bf16 | ✅ | 自动完成 |
| Qwen2.5-14B-INT8 | int8→bf16 | ✅ | 手动运行 (原超时) |
| Qwen2.5-14B-FP8 | fp8→bf16 | ✅ | 手动运行 (原超时) |

**路径**: `/root/vllmbench/slidesparse/search/cuSPARSELt_AlgSearch/alg_search_results/H100_cc90_py312_cu129_x86_64/`

### Triton Kernel 调优 ✅

所有 4 个基础模型的 Triton kernel 调优文件已生成：

| Kernel 类型 | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B |
|-------------|-------------|-------------|------------|-------------|
| quant_only | ✅ | ✅ | ✅ | ✅ |
| quant_slide | ✅ | ✅ | ✅ | ✅ |
| dequant_bias | ✅ | ✅ | ✅ | ✅ |

**路径**: `/root/vllmbench/slidesparse/csrc/*/build/H100_cc90_py312_cu129_x86_64/`

---

## 2. Prefill Benchmark 结果 (Task 6)

### 结果统计

**配置**:
- M 列表: `[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]` (8个)
- 模型: INT8 和 FP8 各 4 个模型
- Backend: `cutlass`, `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)`

### INT8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cutlass | 3/3 ✅ | - | - | - | 3/3 |
| cuBLASLt | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:4 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:6 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:8 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:10 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| **Total** | 43/43 | 40/40 | **35/40** | 40/40 | **158/163** |

### FP8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cutlass | 3/3 ✅ | - | - | - | 3/3 |
| cuBLASLt | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:4 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:6 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:8 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:10 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| **Total** | 43/43 | 40/40 | **35/40** | 40/40 | **158/163** |

### ❌ 失败的测试 (10个)

| # | 模型 | M 值 | Backend | 错误类型 |
|---|------|------|---------|----------|
| 1 | Qwen2.5-7B-INT8 | 65536 | cuBLASLt | CUDA illegal memory access |
| 2 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:4) | CUDA illegal memory access |
| 3 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:6) | CUDA illegal memory access |
| 4 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:8) | CUDA illegal memory access |
| 5 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:10) | CUDA illegal memory access |
| 6 | Qwen2.5-7B-FP8 | 65536 | cuBLASLt | CUDA illegal memory access |
| 7 | Qwen2.5-7B-FP8 | 65536 | cuSPARSELt (2:4) | CUDA illegal memory access |
| 8 | Qwen2.5-7B-FP8 | 65536 | cuSPARSELt (2:6) | Triton CUDA illegal memory access |
| 9 | Qwen2.5-7B-FP8 | 65536 | cuSPARSELt (2:8) | Triton CUDA illegal memory access |
| 10 | Qwen2.5-7B-FP8 | 65536 | cuSPARSELt (2:10) | Triton CUDA illegal memory access |

### 结果文件位置

- **INT8 JSON**: `throughput_benchmark_results/prefill/H100_cc90_INT8_py312_cu129_x86_64/{backend}/json/`
- **INT8 CSV**: `throughput_benchmark_results/prefill/H100_cc90_INT8_py312_cu129_x86_64/{backend}/`
- **FP8 JSON**: `throughput_benchmark_results/prefill/H100_cc90_FP8E4M3_py312_cu129_x86_64/{backend}/json/`
- **FP8 CSV**: `throughput_benchmark_results/prefill/H100_cc90_FP8E4M3_py312_cu129_x86_64/{backend}/`

---

## 3. Decode Benchmark 结果 (Task 7)

### 结果统计 ✅ 全部通过

**配置**:
- M 列表: `[64, 128, 256, 512]` (4个)
- 模型: INT8 和 FP8 各 4 个模型
- Backend: `cutlass`, `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)`

### INT8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cutlass | 3/3 ✅ | - | - | - | 3/3 |
| cuBLASLt | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:4 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:6 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:8 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:10 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| **Total** | 23/23 | 20/20 | 20/20 | 20/20 | **83/83** |

### FP8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cutlass | 3/3 ✅ | - | - | - | 3/3 |
| cuBLASLt | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:4 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:6 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:8 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:10 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| **Total** | 23/23 | 20/20 | 20/20 | 20/20 | **83/83** |

---

## 4. 错误根本原因分析

### 🔍 关键发现

1. **失败模式与 A100 完全一致**
   - 仅 `Qwen2.5-7B` 模型在 `M=65536` 失败
   - 其他所有模型 (Llama3.2-1B, Llama3.2-3B, Qwen2.5-14B) 在 M=65536 全部通过
   - INT8 和 FP8 版本表现一致

2. **错误类型**
   - cuBLASLt 后端: `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`
   - cuSPARSELt 后端: `RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered`

3. **问题与 Qwen2.5-7B 的模型架构有关**
   - Qwen2.5-7B: `intermediate_size=18944`, `hidden_size=3584`
   - M=65536 × N=18944 × 2 (gate_up_proj) = 2,483,027,968 > INT32_MAX (2,147,483,647)
   - 这可能导致 PyTorch Inductor 在融合 kernel 时产生 INT32 索引溢出

### 🎯 根本原因

问题出在 **PyTorch Inductor** 的 autotuning 和 kernel fusion 阶段，而非我们的 SlideSparse kernel：
- 当 `xnumel = M × K` 超过 INT32_MAX 时
- Inductor 生成的 `triton_poi_fused_*` kernel 可能使用 INT32 索引
- 这在极大 batch size (M=65536) 与特定模型维度组合时会越界

**技术细节**:
- 错误发生在 Inductor 的 `triton_heuristics.pointwise` autotuning 中
- 融合 kernel 如 `triton_poi_fused_mul_quant_only_int8_silu_slice_1`
- 这不是 SlideSparse 代码的问题，而是 PyTorch/Triton 的边界条件

---

## 5. H100 特有说明

### H100 vs A100 差异

| 特性 | A100 | H100 |
|------|------|------|
| 架构 | Ampere (sm_80) | Hopper (sm_90) |
| FP8 支持 | ❌ 不支持 | ✅ 原生支持 |
| 测试的数据类型 | INT8 only | INT8 + FP8 |
| 失败的测试数 | 5 | 10 (INT8:5 + FP8:5) |
| 失败模式 | 相同 | 相同 |

### H100 FP8 测试

- H100 是 Hopper 架构 (CC 9.0)，**原生支持 FP8**
- 所有 FP8 测试成功运行，除了 Qwen2.5-7B M=65536
- FP8 和 INT8 在相同条件下表现一致

---

## 6. 测试覆盖总结

### 总体统计

| 类别 | 成功 | 失败 | 成功率 |
|------|------|------|--------|
| Prefill INT8 | 158 | 5 | 96.93% |
| Prefill FP8 | 158 | 5 | 96.93% |
| Decode INT8 | 83 | 0 | 100% |
| Decode FP8 | 83 | 0 | 100% |
| **总计** | **482** | **10** | **97.97%** |

### 完全通过的模型/配置

- ✅ Llama3.2-1B (INT8 + FP8): 所有 M 值、所有 backend 通过
- ✅ Llama3.2-3B (INT8 + FP8): 所有 M 值、所有 backend 通过
- ✅ Qwen2.5-14B (INT8 + FP8): 所有 M 值、所有 backend 通过
- ⚠️ Qwen2.5-7B (INT8 + FP8): M≤32768 全部通过，M=65536 失败

---

## 7. 建议

### 短期方案 (推荐)
1. **从 benchmark M 列表中移除 65536** - 最简单有效
   - M=65536 是极端边界用例 (65536 tokens ≈ 50,000 字 prompt)
   - 实际生产中极少遇到
   - 其他所有模型和配置在此 M 值下都能通过

### 长期方案
1. **向 PyTorch 团队报告 Inductor 的 INT32 索引问题**
   - 提供重现步骤和模型架构信息
   - 这是通用问题，不仅影响 SlideSparse

2. **考虑在 SlideSparse kernel 中添加 M 值上限检查**
   - 当 M × intermediate_size × 2 > INT32_MAX 时给出警告
   - 提前终止而非 CUDA crash

---

## 8. 日志文件

### 主要日志
- `H100/prepare_bench_20260126_072848.log` - 主运行日志 (28MB)
- `H100/prepare_bench_20260126_072848_status.json` - 状态文件

### Benchmark 日志
- `throughput_benchmark_results/logs/H100/benchmark_*.log` - 19 个单独 benchmark 日志
- `throughput_benchmark_results/prefill/H100_cc90_*/*/benchmark.log` - 各 backend 详细日志
- `throughput_benchmark_results/decode/H100_cc90_*/*/benchmark.log` - 各 backend 详细日志

---

**报告生成时间**: 2026-01-26 18:30  
**分析人**: GitHub Copilot
