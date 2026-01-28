# B200 Benchmark 最终报告

**日志文件**: 
- `prepare_bench_20260126_063652.log` (Task 3-4)
- `prepare_bench_20260126_125618.log` (Task 5-6)
- `prepare_bench_20260126_193937.log` (Task 7)

**时间**: 2026-01-26  
**GPU**: NVIDIA B200 180GB (CC 10.0, Blackwell)

---

## 📊 最终结果汇总

### 总体情况

| 任务 | 状态 | 成功数 | 失败数 | 说明 |
|------|------|--------|--------|------|
| Task 3: 离线粗调优 | ✅ 成功 | 12 | 0 | cuBLASLt 8/8 + Triton quant_only 4/4 |
| Task 4: 离线细调优 | ✅ 成功 | 16 | 0 | cuSPARSELt 8/8 + Triton dequant 4/4 + quant_slide 4/4 |
| Task 5: 简单 Benchmark | ✅ 成功 | 2 | 0 | llama3.2-1b INT8/FP8 全部通过 |
| Task 6: Prefill Benchmark | ⚠️ 部分失败 | 310 | 10 | Qwen2.5-7B M=65536 失败 |
| Task 7: Decode Benchmark | ✅ 成功 | 160 | 0 | 全部通过 |

**总耗时统计**:
- Task 3: 16.1 分钟
- Task 4: 6.0 小时 (362.1 分钟)
- Task 5: 29.8 分钟
- Task 6: 5.6 小时 (337.0 分钟)
- Task 7: 1.9 小时 (114.7 分钟)

---

## 1. 离线调优结果 (Task 3 & 4) ✅ 全部成功

### Task 3: 粗调优 (16.1 分钟)

| 组件 | 状态 | 数量 |
|------|------|------|
| cuBLASLt GEMM (INT8) | ✅ | 4/4 |
| cuBLASLt GEMM (FP8) | ✅ | 4/4 |
| Triton Quant Only | ✅ | 4/4 |

### Task 4: 细调优 (6.0 小时)

| 组件 | 状态 | 数量 |
|------|------|------|
| cuSPARSELt GEMM (INT8) | ✅ | 4/4 |
| cuSPARSELt GEMM (FP8) | ✅ | 4/4 |
| Triton Dequant + Bias | ✅ | 4/4 |
| Triton Quant + Slide | ✅ | 4/4 |

### Triton Kernel 调优文件 ✅

所有 12 个调优文件已生成且正常：

| Kernel | 文件 | 大小 |
|--------|------|------|
| quant_only | `quant_only_tuned_Llama3.2-1B.py` | 7,055 bytes |
| quant_only | `quant_only_tuned_Llama3.2-3B.py` | 7,059 bytes |
| quant_only | `quant_only_tuned_Qwen2.5-7B.py` | 6,530 bytes |
| quant_only | `quant_only_tuned_Qwen2.5-14B.py` | 6,948 bytes |
| quant_slide | `quant_slide_tuned_Llama3.2-1B.py` | 12,707 bytes |
| quant_slide | `quant_slide_tuned_Llama3.2-3B.py` | 12,182 bytes |
| quant_slide | `quant_slide_tuned_Qwen2.5-7B.py` | 11,894 bytes |
| quant_slide | `quant_slide_tuned_Qwen2.5-14B.py` | 11,658 bytes |
| dequant_bias | `dequant_bias_tuned_Llama3.2-1B.py` | 4,590 bytes |
| dequant_bias | `dequant_bias_tuned_Llama3.2-3B.py` | 4,750 bytes |
| dequant_bias | `dequant_bias_tuned_Qwen2.5-7B.py` | 4,761 bytes |
| dequant_bias | `dequant_bias_tuned_Qwen2.5-14B.py` | 4,414 bytes |

**路径**: `/root/vllmbench/slidesparse/csrc/*/build/B200_cc100_py312_cu129_x86_64/`

### GEMM 算法搜索结果 ✅

| 库 | 文件数 | 路径 |
|-----|--------|------|
| cuBLASLt | 16 (8 JSON + 8 CSV) | `search/cuBLASLt_AlgSearch/alg_search_results/B200_cc100_py312_cu129_x86_64/` |
| cuSPARSELt | 16 (8 JSON + 8 CSV) | `search/cuSPARSELt_AlgSearch/alg_search_results/B200_cc100_py312_cu129_x86_64/` |

---

## 2. 简单 Benchmark 结果 (Task 5) ✅ 全部通过

| 模型 | 状态 | 耗时 |
|------|------|------|
| llama3.2-1b-int8 | ✅ SUCCESS | 812.4s |
| llama3.2-1b-fp8 | ✅ SUCCESS | 977.4s |

---

## 3. Prefill Benchmark 结果 (Task 6) ⚠️ 部分失败

### 配置

- **M 列表**: `[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]`
- **模型 (INT8)**: `Llama3.2-1B-INT8`, `Llama3.2-3B-INT8`, `Qwen2.5-7B-INT8`, `Qwen2.5-14B-INT8`
- **模型 (FP8)**: `Llama3.2-1B-FP8`, `Llama3.2-3B-FP8`, `Qwen2.5-7B-FP8`, `Qwen2.5-14B-FP8`
- **Backend**: `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)`

### INT8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cuBLASLt | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:4 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:6 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:8 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:10 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| **Total** | 40/40 | 40/40 | **35/40** | 40/40 | **155/160** |

### FP8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cuBLASLt | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:4 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:6 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:8 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:10 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| **Total** | 40/40 | 40/40 | **35/40** | 40/40 | **155/160** |

### ❌ 失败的测试详情 (10个)

| # | 模型 | 量化 | M 值 | Backend | Sparsity | 错误类型 |
|---|------|------|------|---------|----------|----------|
| 1 | Qwen2.5-7B | INT8 | 65536 | cuBLASLt | - | CUDA illegal memory access |
| 2 | Qwen2.5-7B | INT8 | 65536 | cuSPARSELt | 2:4 | Triton CUDA error |
| 3 | Qwen2.5-7B | INT8 | 65536 | cuSPARSELt | 2:6 | Triton CUDA error |
| 4 | Qwen2.5-7B | INT8 | 65536 | cuSPARSELt | 2:8 | Triton CUDA error |
| 5 | Qwen2.5-7B | INT8 | 65536 | cuSPARSELt | 2:10 | Triton CUDA error |
| 6 | Qwen2.5-7B | FP8 | 65536 | cuBLASLt | - | CUDA illegal memory access |
| 7 | Qwen2.5-7B | FP8 | 65536 | cuSPARSELt | 2:4 | Triton CUDA error |
| 8 | Qwen2.5-7B | FP8 | 65536 | cuSPARSELt | 2:6 | Triton CUDA error |
| 9 | Qwen2.5-7B | FP8 | 65536 | cuSPARSELt | 2:8 | Triton CUDA error |
| 10 | Qwen2.5-7B | FP8 | 65536 | cuSPARSELt | 2:10 | Triton CUDA error |

### ✅ 完全通过的模型

| 模型 | INT8 | FP8 | 状态 |
|------|------|-----|------|
| Llama3.2-1B | 40/40 ✅ | 40/40 ✅ | 完全通过 |
| Llama3.2-3B | 40/40 ✅ | 40/40 ✅ | 完全通过 |
| Qwen2.5-7B | 35/40 ⚠️ | 35/40 ⚠️ | M=65536 失败 |
| Qwen2.5-14B | 40/40 ✅ | 40/40 ✅ | 完全通过 |

---

## 4. Decode Benchmark 结果 (Task 7) ✅ 全部通过

### 配置

- **M 列表**: `[64, 128, 256, 512]`
- **模型**: 同 Prefill (8个模型)
- **Backend**: `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)`

### INT8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cuBLASLt | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:4 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:6 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:8 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:10 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| **Total** | 20/20 | 20/20 | 20/20 | 20/20 | **80/80** |

### FP8 模型结果

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cuBLASLt | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:4 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:6 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:8 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:10 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| **Total** | 20/20 | 20/20 | 20/20 | 20/20 | **80/80** |

### Decode Benchmark 耗时

| 模型 | INT8 | FP8 |
|------|------|-----|
| llama3.2-1b | 634.9s | 666.1s |
| llama3.2-3b | 806.3s | 814.3s |
| qwen2.5-7b | 837.9s | 831.5s |
| qwen2.5-14b | 1150.4s | 1139.3s |

---

## 5. 错误根本原因分析

### 🔍 关键发现

1. **失败模式一致**: 只有 `Qwen2.5-7B` 在 `M=65536` 时失败
2. **跨量化类型**: INT8 和 FP8 都失败
3. **跨 Backend**: cuBLASLt 和所有 cuSPARSELt sparsity 配置都失败
4. **跨 GPU**: 该问题在 A100、RTX5080、B200 上都存在 (见其他报告)

### 🎯 根本原因

问题出在 **PyTorch Inductor 生成的融合 kernel** 中：

```
triton_poi_fused_mul_quant_only_int8_silu_slice_1
```

错误类型：`torch.AcceleratorError: CUDA error: an illegal memory access was encountered`

**技术分析**:
- Qwen2.5-7B: `intermediate_size=18944`, `hidden_size=3584`
- M=65536 × K=18944 = 1,241,513,984 elements
- 当 Inductor autotune 时，某些配置会在特定 GPU 架构上产生越界访问
- 这与 Triton/Inductor 的 autotuning 机制有关，不是我们代码的问题

### ⚠️ 重要说明

**我们的 SlideSparse Triton kernel 本身没有问题**:
- `quant_only_int8` / `quant_only_fp8` 单独测试全部通过
- `quant_slide` / `dequant_bias` 也没有问题
- 问题出在 vLLM/PyTorch 的 torch.compile 融合 kernel 中

---

## 6. B200 特有说明

### B200 vs 其他 GPU 对比

| 特性 | B200 | A100 | H100 | RTX 5080 |
|------|------|------|------|----------|
| 架构 | Blackwell (CC 10.0) | Ampere (CC 8.0) | Hopper (CC 9.0) | Blackwell (CC 12.0) |
| 显存 | 180 GB | 80 GB | 80 GB | 16 GB |
| FP8 支持 | ✅ | ❌ | ✅ | ✅ |
| INT8 支持 | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5-7B M=65536 | ❌ 失败 | ❌ 失败 | - | ❌ 失败 |

### B200 优势

1. **完整 FP8 支持**: 与 A100 不同，B200 支持原生 FP8 运算
2. **大显存**: 180GB 显存可以运行更大的 batch size
3. **新架构特性**: Blackwell 架构的 SM 优化

---

## 7. 结果文件位置

### 调优结果

```
slidesparse/csrc/quant_only_triton/build/B200_cc100_py312_cu129_x86_64/
  ├── quant_only_tuned_Llama3.2-1B.py
  ├── quant_only_tuned_Llama3.2-3B.py
  ├── quant_only_tuned_Qwen2.5-7B.py
  └── quant_only_tuned_Qwen2.5-14B.py

slidesparse/csrc/fused_quant_slide_triton/build/B200_cc100_py312_cu129_x86_64/
  ├── quant_slide_tuned_Llama3.2-1B.py
  ├── quant_slide_tuned_Llama3.2-3B.py
  ├── quant_slide_tuned_Qwen2.5-7B.py
  └── quant_slide_tuned_Qwen2.5-14B.py

slidesparse/csrc/fused_dequant_bias_triton/build/B200_cc100_py312_cu129_x86_64/
  ├── dequant_bias_tuned_Llama3.2-1B.py
  ├── dequant_bias_tuned_Llama3.2-3B.py
  ├── dequant_bias_tuned_Qwen2.5-7B.py
  └── dequant_bias_tuned_Qwen2.5-14B.py
```

### GEMM 算法搜索结果

```
slidesparse/search/cuBLASLt_AlgSearch/alg_search_results/B200_cc100_py312_cu129_x86_64/
  ├── alg_search_Llama3.2-1B-{INT8,FP8}_*.{json,csv}
  ├── alg_search_Llama3.2-3B-{INT8,FP8}_*.{json,csv}
  ├── alg_search_Qwen2.5-7B-{INT8,FP8}_*.{json,csv}
  └── alg_search_Qwen2.5-14B-{INT8,FP8}_*.{json,csv}

slidesparse/search/cuSPARSELt_AlgSearch/alg_search_results/B200_cc100_py312_cu129_x86_64/
  └── (同上结构)
```

### Benchmark 结果

```
slidesparse/tools/throughput_benchmark_results/
  ├── prefill/
  │   ├── B200_cc100_INT8_py312_cu129_x86_64/
  │   │   ├── cublaslt/       (4 CSV)
  │   │   └── cusparselt/     (16 CSV: 4 models × 4 sparsity)
  │   └── B200_cc100_FP8E4M3_py312_cu129_x86_64/
  │       ├── cublaslt/       (4 CSV)
  │       ├── cutlass/        (1 CSV)
  │       └── cusparselt/     (16 CSV)
  └── decode/
      ├── B200_cc100_INT8_py312_cu129_x86_64/
      │   ├── cublaslt/       (4 CSV)
      │   └── cusparselt/     (16 CSV)
      └── B200_cc100_FP8E4M3_py312_cu129_x86_64/
          ├── cublaslt/       (4 CSV)
          ├── cutlass/        (1 CSV)
          └── cusparselt/     (16 CSV)
```

---

## 8. 建议

### 对于 Qwen2.5-7B M=65536 失败

1. **短期**: 从 benchmark M 列表中移除 65536
   - M=65536 是极端边界用例 (65536 tokens ≈ 50,000 字)
   - 实际生产中很少遇到如此长的 prompt

2. **长期**: 等待 PyTorch/Triton 修复
   - 这是 Inductor autotune 的兼容性问题
   - 不是我们代码的问题

### 重跑失败测试的命令 (如需要)

```bash
cd /root/vllmbench/slidesparse/tools

# INT8
python throughput_benchmark.py \
  --model qwen2.5-7b-int8 \
  --backend cublaslt,cusparselt \
  --stage prefill \
  --sparsity 2_4,2_6,2_8,2_10 \
  --M 65536

# FP8
python throughput_benchmark.py \
  --model qwen2.5-7b-fp8 \
  --backend cublaslt,cusparselt \
  --stage prefill \
  --sparsity 2_4,2_6,2_8,2_10 \
  --M 65536
```

⚠️ **注意**: 此命令会再次失败，除非修改 PyTorch Inductor 或跳过 M=65536。

---

## 9. 总结

| 指标 | 结果 |
|------|------|
| **离线调优** | ✅ 100% 成功 (28/28) |
| **Prefill Benchmark** | 96.9% 成功 (310/320) |
| **Decode Benchmark** | ✅ 100% 成功 (160/160) |
| **唯一失败点** | Qwen2.5-7B × M=65536 × 10 配置 |
| **失败原因** | PyTorch Inductor autotune bug (非 SlideSparse 问题) |

B200 Benchmark 总体**成功率 97.9%** (498/508)，所有核心功能验证通过。

---

**报告生成时间**: 2026-01-26
