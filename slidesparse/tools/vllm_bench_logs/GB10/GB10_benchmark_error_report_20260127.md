# GB10 SlideSparse Benchmark 报告

**日期**: 2026-01-27  
**硬件**: NVIDIA GB10 (Blackwell, cc121, sm_121a)  
**显存**: 119.7 GB  
**CUDA**: 12.9 / Driver 13.0  
**Python**: 3.12  
**Architecture**: aarch64  

---

## 一、测试概述

### 1.1 执行的任务

| 任务 | 名称 | 状态 | 耗时 |
|------|------|------|------|
| Task 1 | 模型下载 | ⏭️ SKIPPED | - |
| Task 2 | 模型转换 (SlideSparse) | ⏭️ SKIPPED | - |
| Task 3 | 离线粗调优 (cuBLAS + quant_only) | ⏭️ SKIPPED | - |
| Task 4 | 离线细调优 (cuSPARSE + Triton) | ⏭️ SKIPPED | - |
| Task 5 | 简单端到端 Benchmark | ⏭️ SKIPPED | - |
| Task 6 | 完整 Prefill Benchmark | ❌ FAILED | 34.2 小时 |
| Task 7 | 完整 Decode Benchmark | ✅ SUCCESS | 4.2 小时 |

**总运行时间**: 38.41 小时

### 1.2 测试矩阵

- **模型**: 8 个
  - Llama3.2-1B (INT8, FP8)
  - Llama3.2-3B (INT8, FP8)  
  - Qwen2.5-7B (INT8, FP8)
  - Qwen2.5-14B (INT8, FP8)

- **Backend**: cuBLASLt, cuSPARSELt

- **稀疏度**: 2:4, 2:6, 2:8, 2:10

- **Prefill M 值**: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536

- **Decode M 值**: 64, 128, 256, 512

---

## 二、离线调优 (Task 3 & 4)

Task 3 和 Task 4 在本次运行中被跳过，使用的是之前已完成的调优结果。

调优结果存储位置:
- cuBLASLt: `/root/vllmbench/slidesparse/search/cuBLASLt_AlgSearch/alg_search_results/GB10_cc121_py312_cu129_aarch64/`
- cuSPARSELt: `/root/vllmbench/slidesparse/search/cuSPARSELt_AlgSearch/alg_search_results/GB10_cc121_py312_cu129_aarch64/`
- Triton kernels: `/root/vllmbench/slidesparse/csrc/*/build/GB10_cc121_py312_cu129_aarch64/`

**状态**: ✅ 调优结果完整可用

---

## 三、Prefill Benchmark (Task 6) 详细分析

### 3.1 总体结果

| 模型 | 状态 | 成功/总数 | 失败 M 值 | 耗时 |
|------|------|----------|----------|------|
| llama3.2-1b-int8 | ✅ 完全通过 | 40/40 | - | 4192s (70min) |
| llama3.2-1b-fp8 | ✅ 完全通过 | 40/40 | - | 4057s (68min) |
| llama3.2-3b-int8 | ✅ 完全通过 | 40/40 | - | 9405s (157min) |
| llama3.2-3b-fp8 | ✅ 完全通过 | 40/40 | - | 8356s (139min) |
| **qwen2.5-7b-int8** | ⚠️ 部分失败 | 35/40 | M=65536 | - |
| **qwen2.5-7b-fp8** | ⚠️ 部分失败 | 35/40 | M=65536 | - |
| qwen2.5-14b-int8 | ✅ 完全通过 | 40/40 | - | 34597s (576min) |
| qwen2.5-14b-fp8 | ✅ 完全通过 | 40/40 | - | 38559s (643min) |

**总计**: 310/320 测试通过 (96.9%)

### 3.2 失败详情

#### 3.2.1 qwen2.5-7b-int8 失败记录

| Backend | 稀疏度 | M 值 | 状态 |
|---------|--------|------|------|
| cuBLASLt | dense | 65536 | ❌ FAILED |
| cuSPARSELt | 2:4 | 65536 | ❌ FAILED |
| cuSPARSELt | 2:6 | 65536 | ❌ FAILED |
| cuSPARSELt | 2:8 | 65536 | ❌ FAILED |
| cuSPARSELt | 2:10 | 65536 | ❌ FAILED |

#### 3.2.2 qwen2.5-7b-fp8 失败记录

| Backend | 稀疏度 | M 值 | 状态 |
|---------|--------|------|------|
| cuBLASLt | dense | 65536 | ❌ FAILED |
| cuSPARSELt | 2:4 | 65536 | ❌ FAILED |
| cuSPARSELt | 2:6 | 65536 | ❌ FAILED |
| cuSPARSELt | 2:8 | 65536 | ❌ FAILED |
| cuSPARSELt | 2:10 | 65536 | ❌ FAILED |

### 3.3 错误原因分析

**错误类型**: `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`

**触发位置**: 
```
File ".../quant_only_tuned_Qwen2.5-7B.py", line 195, in quant_only_int8_triton
    out = torch.zeros(M_padded, K_padded, dtype=torch.int8, device=x.device)
```

**根本原因**:
- 当 M=65536 时，Qwen2.5-7B 模型的 Triton `quant_only` 内核触发了 CUDA 非法内存访问
- 这可能是由于:
  1. Triton 内核在处理超大 batch size 时的 grid 维度超限
  2. 内存分配或索引计算溢出
  3. GB10 (Blackwell) 架构特定的兼容性问题

**影响范围**:
- 仅影响 **Qwen2.5-7B** 模型（INT8 和 FP8）
- 仅影响 **M=65536** 的测试用例
- Llama3.2-1B/3B 和 Qwen2.5-14B 在 M=65536 下均正常工作

**建议**:
1. 检查 Qwen2.5-7B 的 `quant_only` Triton 内核实现
2. 对于生产环境，建议将 Qwen2.5-7B 的最大 batch size 限制在 32768 以内
3. 或者在 M=65536 时跳过 Triton 内核，使用纯 CUDA 实现

---

## 四、Decode Benchmark (Task 7) 详细分析

### 4.1 总体结果

| 模型 | 状态 | 成功/总数 | 耗时 |
|------|------|----------|------|
| llama3.2-1b-int8 | ✅ 完全通过 | 20/20 | 717s (12min) |
| llama3.2-1b-fp8 | ✅ 完全通过 | 20/20 | 720s (12min) |
| llama3.2-3b-int8 | ✅ 完全通过 | 20/20 | 1234s (21min) |
| llama3.2-3b-fp8 | ✅ 完全通过 | 20/20 | 1097s (18min) |
| qwen2.5-7b-int8 | ✅ 完全通过 | 20/20 | 2036s (34min) |
| qwen2.5-7b-fp8 | ✅ 完全通过 | 20/20 | 1956s (33min) |
| qwen2.5-14b-int8 | ✅ 完全通过 | 20/20 | 3786s (63min) |
| qwen2.5-14b-fp8 | ✅ 完全通过 | 20/20 | 3705s (62min) |

**总计**: 160/160 测试通过 (100%)

### 4.2 性能概览

Decode 阶段所有测试均成功完成，无任何错误。

---

## 五、结果文件位置

### 5.1 Prefill 结果
```
/root/vllmbench/slidesparse/tools/throughput_benchmark_results/prefill/
├── GB10_cc121_INT8_py312_cu129_aarch64/
│   ├── cublaslt/
│   └── cusparselt/{2_4,2_6,2_8,2_10}/
└── GB10_cc121_FP8E4M3_py312_cu129_aarch64/
    ├── cublaslt/
    └── cusparselt/{2_4,2_6,2_8,2_10}/
```

### 5.2 Decode 结果
```
/root/vllmbench/slidesparse/tools/throughput_benchmark_results/decode/
├── GB10_cc121_INT8_py312_cu129_aarch64/
│   ├── cublaslt/
│   └── cusparselt/{2_4,2_6,2_8,2_10}/
└── GB10_cc121_FP8E4M3_py312_cu129_aarch64/
    ├── cublaslt/
    └── cusparselt/{2_4,2_6,2_8,2_10}/
```

### 5.3 日志文件
- 主日志: `/root/vllmbench/slidesparse/tools/prepare_bench_20260126_063900.log`
- 状态文件: `/root/vllmbench/slidesparse/tools/prepare_bench_20260126_063900_status.json`

---

## 六、已知问题与解决方案

### 6.1 GB10 (Blackwell) 架构兼容性

**问题**: Triton 3.5.0 内置的 ptxas 不支持 sm_121a (GB10)

**解决方案**: 已在环境中设置 `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` 使用系统 CUDA 12.9 的 ptxas

**修改文件**:
- `/root/vllmbench/slidesparse/tools/prepare_for_vllm_bench.py` - 自动检测并设置
- `/root/vllmbench/Dockerfile` - 添加环境变量

### 6.2 Qwen2.5-7B M=65536 OOM/CUDA Error

**问题**: Qwen2.5-7B 在 M=65536 时触发 CUDA illegal memory access

**临时解决方案**: 
- 对于 Qwen2.5-7B，建议将 Prefill M 值限制在 ≤32768

**待修复**: 
- 检查 `quant_only_int8_triton` 内核在大 M 值时的行为
- 可能需要调整 Triton 内核的 grid 配置或添加 M 值检查

---

## 七、总结

| 指标 | 数值 |
|------|------|
| **Prefill 测试通过率** | 96.9% (310/320) |
| **Decode 测试通过率** | 100% (160/160) |
| **整体测试通过率** | 97.9% (470/480) |
| **失败模型** | Qwen2.5-7B (仅 M=65536) |
| **总运行时间** | 38.41 小时 |

**结论**: GB10 上的 SlideSparse benchmark 基本成功完成。唯一的失败点是 Qwen2.5-7B 模型在 M=65536 时的 Triton 内核问题，这是一个边界条件问题，不影响常规使用场景（M ≤ 32768）。

---

*报告生成时间: 2026-01-27 21:10 PST*
