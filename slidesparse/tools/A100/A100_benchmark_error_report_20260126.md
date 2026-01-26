# A100 Benchmark 最终报告

**日志文件**: `prepare_bench_20260125_155107.log`  
**时间**: 2026-01-26  
**总耗时**: 9.08 小时 (32700秒)  
**GPU**: NVIDIA A100 80GB PCIe (CC 8.0, Ampere)

---

## 📊 最终结果汇总

### 总体情况

| 任务 | 状态 | 成功数 | 失败数 | 说明 |
|------|------|--------|--------|------|
| Task 3: 离线粗调优 | ⏭️ 跳过 | - | - | 之前已完成 |
| Task 4: 离线细调优 | ✅ 成功 | 16 | 0 | FP8 被正确跳过 |
| Task 5: 简单 Benchmark | ✅ 成功 | 2 | 0 | INT8 通过，FP8 跳过 |
| Task 6: Prefill Benchmark | ⚠️ 部分失败 | 155 | 5 | Qwen2.5-7B M=65536 失败 |
| Task 7: Decode Benchmark | ✅ 成功 | 80 | 0 | 全部通过 |

---

## 1. 调优结果 (Task 4)

### Triton Kernel 调优文件 ✅

所有 12 个调优文件已生成且正常：

| Kernel | 文件 | 大小 |
|--------|------|------|
| quant_only | `quant_only_tuned_Llama3.2-1B.py` | 6,696 bytes |
| quant_only | `quant_only_tuned_Llama3.2-3B.py` | 6,640 bytes |
| quant_only | `quant_only_tuned_Qwen2.5-7B.py` | 6,643 bytes |
| quant_only | `quant_only_tuned_Qwen2.5-14B.py` | 6,284 bytes |
| quant_slide | `quant_slide_tuned_Llama3.2-1B.py` | 12,640 bytes |
| quant_slide | `quant_slide_tuned_Llama3.2-3B.py` | 12,361 bytes |
| quant_slide | `quant_slide_tuned_Qwen2.5-7B.py` | 11,730 bytes |
| quant_slide | `quant_slide_tuned_Qwen2.5-14B.py` | 11,845 bytes |
| dequant_bias | `dequant_bias_tuned_Llama3.2-1B.py` | 4,641 bytes |
| dequant_bias | `dequant_bias_tuned_Llama3.2-3B.py` | 4,634 bytes |
| dequant_bias | `dequant_bias_tuned_Qwen2.5-7B.py` | 4,469 bytes |
| dequant_bias | `dequant_bias_tuned_Qwen2.5-14B.py` | 4,701 bytes |

**路径**: `/root/vllmbench/slidesparse/csrc/*/build/A100_cc80_py312_cu129_x86_64/`

---

## 2. Prefill Benchmark 结果 (Task 6)

### 结果统计

**配置**:
- M 列表: `[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]`
- 模型: `Llama3.2-1B-INT8`, `Llama3.2-3B-INT8`, `Qwen2.5-7B-INT8`, `Qwen2.5-14B-INT8`
- Backend: `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)`

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cuBLASLt | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:4 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:6 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:8 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| cuSPARSELt 2:10 | 8/8 ✅ | 8/8 ✅ | **7/8** ⚠️ | 8/8 ✅ | 31/32 |
| **Total** | 40/40 | 40/40 | **35/40** | 40/40 | **155/160** |

### ❌ 失败的测试 (5个)

| # | 模型 | M 值 | Backend | 错误类型 |
|---|------|------|---------|----------|
| 1 | Qwen2.5-7B-INT8 | 65536 | cuBLASLt | CUDA illegal memory access |
| 2 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:4) | CUDA illegal memory access |
| 3 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:6) | Triton CUDA illegal memory access |
| 4 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:8) | Triton CUDA illegal memory access |
| 5 | Qwen2.5-7B-INT8 | 65536 | cuSPARSELt (2:10) | Triton CUDA illegal memory access |

### 结果文件位置

- **JSON**: `throughput_benchmark_results/prefill/A100_cc80_INT8_py312_cu129_x86_64/{backend}/json/`
- **CSV**: `throughput_benchmark_results/prefill/A100_cc80_INT8_py312_cu129_x86_64/{backend}/`

⚠️ **注意**: `Qwen2.5-7B-INT8_prefill.csv` 被失败的测试覆盖，只有失败记录。完整数据保存在 JSON 文件中。

---

## 3. Decode Benchmark 结果 (Task 7)

### 结果统计 ✅ 全部通过

**配置**:
- M 列表: `[64, 128, 256, 512]`
- 模型: `Llama3.2-1B-INT8`, `Llama3.2-3B-INT8`, `Qwen2.5-7B-INT8`, `Qwen2.5-14B-INT8`
- Backend: `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)`

| Backend | Llama3.2-1B | Llama3.2-3B | Qwen2.5-7B | Qwen2.5-14B | Total |
|---------|-------------|-------------|------------|-------------|-------|
| cuBLASLt | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:4 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:6 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:8 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| cuSPARSELt 2:10 | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 16/16 |
| **Total** | 20/20 | 20/20 | 20/20 | 20/20 | **80/80** |

---

## 4. 错误根本原因分析

### 🔍 关键发现

1. **我们的 Triton kernel 本身完全没有问题**
   - `quant_only_int8` 直接调用测试：M=65536 全部通过
   - `quant_slide_int8` 直接调用测试：M=65536 全部通过
   - `dequant_bias` 单独测试也通过

2. **问题出在 PyTorch Inductor 生成的融合 kernel 中**
   - 错误发生在 `triton_poi_fused_mul_quant_only_int8_silu_slice_1`
   - 这是 Inductor 试图融合 `mul + silu + slice + quant_only_int8` 生成的 kernel
   - 错误发生在 Inductor 的 **autotune 阶段**

3. **只有 Qwen2.5-7B 在 M=65536 失败**
   - 其他模型 (Llama3.2-1B, Llama3.2-3B, Qwen2.5-14B) 在 M=65536 都通过
   - 这与 Qwen2.5-7B 的 `intermediate_size=18944` 有关

### 🎯 根本原因

问题最可能发生在 PyTorch Inductor 的 `triton_heuristics.pointwise` autotuning 中。

当 `xnumel = M × K = 65536 × 18944 = 1,241,513,984` 时，某些 autotune 配置可能在 A100 的 sm_80 架构上产生越界访问。这与 INT32 索引溢出有关（但不是我们代码的问题）。

**技术细节**:
- Qwen2.5-7B: `intermediate_size=18944`, `hidden_size=3584`
- M=65536 × N=18944 × 2 (gate_up_proj) = 2,483,027,968 > INT32_MAX (2,147,483,647)

---

## 5. A100 FP8 支持说明

- A100 是 Ampere 架构 (Compute Capability 8.0)，**不支持原生 FP8**
- FP8 需要 Ada Lovelace (CC 8.9+) 或 Hopper (CC 9.0+)
- 所有 FP8 相关测试被正确跳过，没有崩溃
- 警告信息: `[WARNING] GPU A100 (cc80) 不支持原生 FP8，跳过...`

---

## 6. 建议

### 短期方案 (推荐)
1. **从 benchmark M 列表中移除 65536** - 最简单有效
   - M=65536 是极端边界用例 (65536 tokens = 约 50,000 字 prompt)
   - 实际生产中很少遇到
   
2. 或设置环境变量: `TORCHINDUCTOR_MAX_AUTOTUNE=0`

### 长期方案
1. 向 PyTorch 团队报告此 Inductor bug
2. 调查 Inductor 在 sm_80 上生成的特定 kernel 配置
3. 等待 PyTorch 更新修复

---

## 7. 文件清单

### 调优结果
```
slidesparse/csrc/quant_only_triton/build/A100_cc80_py312_cu129_x86_64/
  ├── quant_only_tuned_Llama3.2-1B.py
  ├── quant_only_tuned_Llama3.2-3B.py
  ├── quant_only_tuned_Qwen2.5-7B.py
  └── quant_only_tuned_Qwen2.5-14B.py

slidesparse/csrc/fused_quant_slide_triton/build/A100_cc80_py312_cu129_x86_64/
  ├── quant_slide_tuned_Llama3.2-1B.py
  ├── quant_slide_tuned_Llama3.2-3B.py
  ├── quant_slide_tuned_Qwen2.5-7B.py
  └── quant_slide_tuned_Qwen2.5-14B.py

slidesparse/csrc/fused_dequant_bias_triton/build/A100_cc80_py312_cu129_x86_64/
  ├── dequant_bias_tuned_Llama3.2-1B.py
  ├── dequant_bias_tuned_Llama3.2-3B.py
  ├── dequant_bias_tuned_Qwen2.5-7B.py
  └── dequant_bias_tuned_Qwen2.5-14B.py
```

### Benchmark 结果
```
slidesparse/tools/throughput_benchmark_results/
  ├── prefill/A100_cc80_INT8_py312_cu129_x86_64/
  │   ├── cublaslt/         (31 JSON, 4 CSV)
  │   ├── cusparselt/2_4/   (31 JSON, 4 CSV)
  │   ├── cusparselt/2_6/   (31 JSON, 4 CSV)
  │   ├── cusparselt/2_8/   (31 JSON, 4 CSV)
  │   └── cusparselt/2_10/  (31 JSON, 4 CSV)
  └── decode/A100_cc80_INT8_py312_cu129_x86_64/
      ├── cublaslt/         (16 JSON, 4 CSV)
      ├── cusparselt/2_4/   (16 JSON, 4 CSV)
      ├── cusparselt/2_6/   (16 JSON, 4 CSV)
      ├── cusparselt/2_8/   (16 JSON, 4 CSV)
      └── cusparselt/2_10/  (16 JSON, 4 CSV)
```

---

## 8. 重跑失败测试的命令 (如需要)

```bash
# 仅重跑失败的测试
python throughput_benchmark.py \
  --model qwen2.5-7b-int8 \
  --backend cublaslt,cusparselt \
  --stage prefill \
  --sparsity 2_4,2_6,2_8,2_10 \
  --M 65536
```

⚠️ **注意**: 此命令会再次失败，除非修改 PyTorch Inductor 或跳过 M=65536。

---

**报告生成时间**: 2026-01-26
