# B200 BitNet Benchmark 最终报告

**日志文件**: `bitnet_bench_20260128_123640.log`  
**时间**: 2026-01-28 15:16  
**总耗时**: 2.67 小时 (9600.6秒)  
**GPU**: NVIDIA B200 (CC 10.0, Blackwell)

---

## 📊 最终结果汇总

### 🎉 总体情况: 全部成功！

| 任务 | 状态 | 耗时 | 说明 |
|------|------|------|------|
| Task 1: 基础模型准备 | ✅ 成功 | 88.5s | 下载 BF16 + 量化 INT8/FP8 |
| Task 2: SlideSparse 转换 | ✅ 成功 | 540.7s | 8个模型全部转换成功 |
| Task 3: 离线调优 | ✅ 成功 | 1930.5s | 粗调优 + 细调优 |
| Task 4: Prefill Benchmark | ✅ 成功 | 3317.2s | INT8 + FP8 全部通过 |
| Task 5: Decode Benchmark | ✅ 成功 | 1637.7s | INT8 + FP8 全部通过 |
| Task 6: Kernel cuBLASLt | ✅ 成功 | 27.8s | INT8 + FP8 全部通过 |
| Task 7: Kernel cuSPARSELt 高稀疏 | ✅ 成功 | 947.8s | 2_4, 2_6, 2_8, 2_10 |
| Task 8: Kernel cuSPARSELt 低稀疏 | ✅ 成功 | 1110.4s | 2_12, 2_14, 2_16, 2_inf |

**统计**: `8 成功, 0 失败, 0 跳过`

---

## 1. 模型准备结果 (Task 1 & 2)

### 基础模型 ✅

| 模型 | 路径 | 状态 | 耗时 |
|------|------|------|------|
| BitNet-2B-BF16 | `checkpoints/BitNet-2B-BF16/` | ✅ 下载成功 | 11.1s |
| BitNet-2B-INT8 | `checkpoints/BitNet-2B-INT8/` | ✅ 量化成功 | 33.9s |
| BitNet-2B-FP8 | `checkpoints/BitNet-2B-FP8/` | ✅ 量化成功 | 34.2s |

### SlideSparse 模型 ✅

| 基础模型 | 2:4 | 2:6 | 2:8 | 2:10 |
|----------|-----|-----|-----|------|
| BitNet-2B-INT8 | ✅ 64.6s | ✅ 71.3s | ✅ 70.0s | ✅ 71.2s |
| BitNet-2B-FP8 | ✅ 62.5s | ✅ 69.0s | ✅ 65.1s | ✅ 66.9s |

**路径**: `checkpoints_slidesparse/{模型名}-SlideSparse-{Z}_{L}/`

---

## 2. 离线调优结果 (Task 3)

### 粗调优 (cuBLASLt + Triton quant_only) ✅

- M 列表: `[256, 1024, 4096, 16384, 32768]`
- 耗时: 183.0s
- 结果:
  - ✅ cuBLASLt GEMM (int8) 完成
  - ✅ cuBLASLt GEMM (fp8) 完成
  - ✅ Triton Quant Only 完成

### 细调优 (cuSPARSELt + Triton Dequant/QuantSlide) ✅

- M 列表: `[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]`
- 耗时: 1747.5s
- 结果:
  - ✅ cuSPARSELt GEMM (int8) 完成
  - ✅ cuSPARSELt GEMM (fp8) 完成
  - ✅ Triton Dequant + Bias 完成
  - ✅ Triton Quant + Slide 完成

### ✅ B200 FP8 支持

B200 是 Blackwell 架构 (CC 10.0)，**完全支持原生 FP8**！INT8 和 FP8 调优均成功完成。

---

## 3. Prefill Benchmark 结果 (Task 4)

### 配置

- **模型**: `bitnet1.58-2b-int8`, `bitnet1.58-2b-fp8`
- **M 列表**: `[512, 1024, 2048, 4096, 8192, 16384, 32768]` (7个)
- **Backend**: `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)` (5个)

### 结果统计

| 模型 | cuBLASLt | cuSPARSELt 2:4 | cuSPARSELt 2:6 | cuSPARSELt 2:8 | cuSPARSELt 2:10 | Total |
|------|----------|----------------|----------------|----------------|-----------------|-------|
| **BitNet-2B-INT8** | 7/7 ✅ | 7/7 ✅ | 7/7 ✅ | 7/7 ✅ | 7/7 ✅ | **35/35** |
| **BitNet-2B-FP8** | 7/7 ✅ | 7/7 ✅ | 7/7 ✅ | 7/7 ✅ | 7/7 ✅ | **35/35** |

**Prefill 总计**: 70/70 ✅ (100% 通过)

### 性能数据 (BitNet-2B-INT8 cuBLASLt)

| M | requests/s | tokens/s | 耗时 |
|---|------------|----------|------|
| 512 | 30.03 | 15,406 | 4.3s |
| 1024 | 31.21 | 31,992 | 4.1s |
| 2048 | 62.53 | 64,090 | 4.1s |
| 4096 | 122.17 | 125,222 | 4.2s |
| 8192 | 224.06 | 229,658 | 4.6s |
| 16384 | 262.47 | 269,036 | 7.8s |
| 32768 | 275.39 | 282,280 | 14.9s |

### 结果文件

```
throughput_benchmark_results/prefill/B200_cc100_INT8_py312_cu129_x86_64/
├── cublaslt/BitNet-2B-INT8_prefill.csv
└── cusparselt/
    ├── 2_4/BitNet-2B-INT8_prefill.csv
    ├── 2_6/BitNet-2B-INT8_prefill.csv
    ├── 2_8/BitNet-2B-INT8_prefill.csv
    └── 2_10/BitNet-2B-INT8_prefill.csv

throughput_benchmark_results/prefill/B200_cc100_FP8E4M3_py312_cu129_x86_64/
├── cublaslt/BitNet-2B-FP8_prefill.csv
└── cusparselt/
    ├── 2_4/BitNet-2B-FP8_prefill.csv
    ├── 2_6/BitNet-2B-FP8_prefill.csv
    ├── 2_8/BitNet-2B-FP8_prefill.csv
    └── 2_10/BitNet-2B-FP8_prefill.csv
```

---

## 4. Decode Benchmark 结果 (Task 5)

### 配置

- **模型**: `bitnet1.58-2b-int8`, `bitnet1.58-2b-fp8`
- **M 列表**: `[64, 128, 256, 512]` (4个)
- **Backend**: `cuBLASLt`, `cuSPARSELt (2:4, 2:6, 2:8, 2:10)` (5个)

### 结果统计

| 模型 | cuBLASLt | cuSPARSELt 2:4 | cuSPARSELt 2:6 | cuSPARSELt 2:8 | cuSPARSELt 2:10 | Total |
|------|----------|----------------|----------------|----------------|-----------------|-------|
| **BitNet-2B-INT8** | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | **20/20** |
| **BitNet-2B-FP8** | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | 4/4 ✅ | **20/20** |

**Decode 总计**: 40/40 ✅ (100% 通过)

### 性能数据 (BitNet-2B-INT8 cuBLASLt)

| M | requests/s | tokens/s | 耗时 |
|---|------------|----------|------|
| 64 | 36.64 | 9,965 | 1.7s |
| 128 | 63.20 | 17,191 | 2.0s |
| 256 | 97.43 | 26,502 | 2.6s |
| 512 | 114.45 | 31,131 | 4.5s |

### 结果文件

```
throughput_benchmark_results/decode/B200_cc100_INT8_py312_cu129_x86_64/
├── cublaslt/BitNet-2B-INT8_decode.csv
└── cusparselt/
    ├── 2_4/BitNet-2B-INT8_decode.csv
    ├── 2_6/BitNet-2B-INT8_decode.csv
    ├── 2_8/BitNet-2B-INT8_decode.csv
    └── 2_10/BitNet-2B-INT8_decode.csv

throughput_benchmark_results/decode/B200_cc100_FP8E4M3_py312_cu129_x86_64/
├── cublaslt/BitNet-2B-FP8_decode.csv
└── cusparselt/
    ├── 2_4/BitNet-2B-FP8_decode.csv
    ├── 2_6/BitNet-2B-FP8_decode.csv
    ├── 2_8/BitNet-2B-FP8_decode.csv
    └── 2_10/BitNet-2B-FP8_decode.csv
```

---

## 5. Kernel Benchmark 结果 (Task 6/7/8)

### 配置

- **模型**: `BitNet-2B` (INT8 + FP8)
- **M 列表**: `[64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]` (9个)
- **数据类型**: `int8`, `fp8e4m3`

### Task 6: cuBLASLt ✅

- 耗时: 27.8s
- INT8: 全部通过 (36/36 成功, 100%)
- FP8: 全部通过 (36/36 成功, 100%)

### Task 7: cuSPARSELt 高稀疏 ✅

- 稀疏度: `2_4, 2_6, 2_8, 2_10`
- 耗时: 947.8s
- INT8: 全部通过 (144/144 成功, 100%)
- FP8: 全部通过 (144/144 成功, 100%)

### Task 8: cuSPARSELt 低稀疏 ✅

- 稀疏度: `2_12, 2_14, 2_16, 2_inf`
- 耗时: 1110.4s
- INT8: 全部通过 (144/144 成功, 100%)
- FP8: 全部通过 (144/144 成功, 100%)

### 结果文件

```
benchmark_kernel/
├── cuBLASLt/alg_search_results/B200_cc100_py312_cu129_x86_64/
│   ├── INT8/alg_search_BitNet-2B-INT8.csv
│   └── FP8/alg_search_BitNet-2B-FP8.csv
└── cuSPARSELt/alg_search_results/B200_cc100_py312_cu129_x86_64/
    ├── INT8/
    │   ├── alg_search_BitNet-2B-INT8_2_4.csv
    │   ├── alg_search_BitNet-2B-INT8_2_6.csv
    │   ├── alg_search_BitNet-2B-INT8_2_8.csv
    │   ├── alg_search_BitNet-2B-INT8_2_10.csv
    │   ├── alg_search_BitNet-2B-INT8_2_12.csv
    │   ├── alg_search_BitNet-2B-INT8_2_14.csv
    │   ├── alg_search_BitNet-2B-INT8_2_16.csv
    │   └── alg_search_BitNet-2B-INT8_2_inf.csv
    └── FP8/
        └── (同上结构)
```

---

## 6. 错误/失败统计

### ❌ 失败的测试: 无

🎉 **所有 8 个任务全部成功完成，没有任何失败！**

---

## 7. 完整测试矩阵

### 模型列表

| 模型名称 | 数据类型 | 稀疏度 | 状态 |
|----------|----------|--------|------|
| BitNet-2B-BF16 | BF16 | - | ✅ 源模型 |
| BitNet-2B-INT8 | INT8 | - | ✅ 量化完成 |
| BitNet-2B-FP8 | FP8 | - | ✅ 量化完成 |
| BitNet-2B-INT8-SlideSparse-2_4 | INT8 | 2:4 | ✅ 转换完成 |
| BitNet-2B-INT8-SlideSparse-2_6 | INT8 | 2:6 | ✅ 转换完成 |
| BitNet-2B-INT8-SlideSparse-2_8 | INT8 | 2:8 | ✅ 转换完成 |
| BitNet-2B-INT8-SlideSparse-2_10 | INT8 | 2:10 | ✅ 转换完成 |
| BitNet-2B-FP8-SlideSparse-2_4 | FP8 | 2:4 | ✅ 转换完成 |
| BitNet-2B-FP8-SlideSparse-2_6 | FP8 | 2:6 | ✅ 转换完成 |
| BitNet-2B-FP8-SlideSparse-2_8 | FP8 | 2:8 | ✅ 转换完成 |
| BitNet-2B-FP8-SlideSparse-2_10 | FP8 | 2:10 | ✅ 转换完成 |

### Benchmark 测试点总数

| 测试类型 | 模型 × Backend × M | 总测试点 | 通过 | 失败 |
|----------|-------------------|----------|------|------|
| Prefill | 2 × 5 × 7 | 70 | 70 | 0 |
| Decode | 2 × 5 × 4 | 40 | 40 | 0 |
| Kernel cuBLASLt | 2 × 9 × 4 | 72 | 72 | 0 |
| Kernel cuSPARSELt 高 | 2 × 4 × 9 × 4 | 288 | 288 | 0 |
| Kernel cuSPARSELt 低 | 2 × 4 × 9 × 4 | 288 | 288 | 0 |
| **合计** | - | **758** | **758** | **0** |

---

## 8. 总结

### ✅ 测试通过率: 100%

- **GPU**: NVIDIA B200 (Blackwell, CC 10.0) 完全支持 FP8
- **模型准备**: 3 个基础模型 + 8 个 SlideSparse 模型 = 11 个模型全部成功
- **离线调优**: 粗调优 + 细调优全部完成
- **端到端 Benchmark**: Prefill + Decode 全部通过
- **Kernel Benchmark**: cuBLASLt + cuSPARSELt (高低稀疏) 全部通过

### 📁 日志文件位置

- 主日志: `slidesparse/tools/bitnet_bench_20260128_123640.log`
- 状态文件: `slidesparse/tools/bitnet_bench_20260128_123640_status.json`

### ⏱️ 耗时分布

```
Task 1 (模型准备):     88.5s   (1.5 min)   0.9%
Task 2 (SlideSparse):  540.7s  (9.0 min)   5.6%
Task 3 (离线调优):     1930.5s (32.2 min)  20.1%
Task 4 (Prefill):      3317.2s (55.3 min)  34.6%
Task 5 (Decode):       1637.7s (27.3 min)  17.1%
Task 6 (cuBLASLt):     27.8s   (0.5 min)   0.3%
Task 7 (cuSPARSELt高): 947.8s  (15.8 min)  9.9%
Task 8 (cuSPARSELt低): 1110.4s (18.5 min)  11.6%
─────────────────────────────────────────────────
总计:                  9600.6s (2.67 hours) 100%
```

---

*报告生成时间: 2026-01-28 15:16*
