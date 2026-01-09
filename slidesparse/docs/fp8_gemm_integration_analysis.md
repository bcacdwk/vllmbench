# SlideSparse Phase 3: FP8 GEMM 集成技术分析

> 本文档详细分析 vLLM 中 FP8 GEMM 的实现架构，包括 compressed-tensors 格式、CUTLASS 内核实现以及 cuBLASLt 替换方案。

---

## 目录

1. [Compressed-Tensors 转发架构分析](#1-compressed-tensors-转发架构分析)
2. [当前 CUTLASS FP8 GEMM 实现详解](#2-当前-cutlass-fp8-gemm-实现详解)
3. [cuBLASLt 替换方案与注意事项](#3-cublaslt-替换方案与注意事项)
4. [实现计划与代码示例](#4-实现计划与代码示例)

---

## 1. Compressed-Tensors 转发架构分析

### 1.1 为什么使用 Compressed-Tensors 而不是原生 FP8/INT8

**核心原因：Compressed-Tensors 是一个元格式（Meta-Format）**

HuggingFace 上的量化模型（如 RedHat 的 W8A8、FP8-dynamic 模型）使用 `compressed-tensors` 作为量化配置格式。这不是一个具体的量化实现，而是一个**配置解析层**，它会：

1. **读取模型的 `config.json`** 中的量化配置
2. **自动检测量化类型**（FP8、INT8、W4A16 等）
3. **选择对应的 Scheme**（`CompressedTensorsW8A8Fp8`、`CompressedTensorsW8A8Int8` 等）

```python
# vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py
class CompressedTensorsConfig(QuantizationConfig):
    def get_scheme(self, layer, layer_name):
        # 根据 layer 类型和配置选择 scheme
        scheme = self._get_scheme_from_parts(...)
        
        # ✅ 我们的 cuBLASLt 包装点
        scheme = wrap_scheme_with_cublaslt(scheme)
        return scheme
```

**典型的模型配置示例**（`config.json`）：
```json
{
  "quantization_config": {
    "quant_method": "compressed-tensors",
    "config_groups": {
      "group_0": {
        "weights": {
          "num_bits": 8,
          "type": "float",
          "strategy": "channel"
        },
        "input_activations": {
          "num_bits": 8,
          "type": "float",
          "strategy": "token"
        }
      }
    }
  }
}
```

### 1.2 当前 SlideSparse cuBLASLt 转发架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    用户加载 FP8 量化模型                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│              CompressedTensorsConfig.get_scheme()                   │
│                                                                     │
│   1. 解析 config_groups 配置                                         │
│   2. 调用 _get_scheme_from_parts() → CompressedTensorsW8A8Fp8       │
│   3. ✅ wrap_scheme_with_cublaslt(scheme) 包装                       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CuBLASLtSchemeWrapper                            │
│                                                                     │
│   - _original_scheme: CompressedTensorsW8A8Fp8                      │
│   - create_weights()      → 委托给原始 scheme                        │
│   - process_weights_after_loading() → 委托给原始 scheme              │
│   - apply_weights()       → 调用 CuBLASLtFp8LinearOp                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CuBLASLtFp8LinearOp.apply()                      │
│                                                                     │
│   当前实现（USE_REAL_CUBLASLT=False）:                               │
│   - 调用 vLLM 原生 Fp8LinearOp（使用 CUTLASS）                       │
│                                                                     │
│   目标实现（USE_REAL_CUBLASLT=True）:                                │
│   - 调用 cuBLASLt FP8 matmul API                                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 转发架构验证

**当前架构已确认正确：**

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 环境变量检测 | ✅ | `VLLM_USE_CUBLASLT=1` 或 `SLIDESPARSE_USE_CUBLASLT=1` |
| Scheme 包装 | ✅ | `wrap_scheme_with_cublaslt()` 在 `get_scheme()` 中调用 |
| 权重创建 | ✅ | 委托给原始 scheme，保持兼容性 |
| 权重加载 | ✅ | 委托给原始 scheme，safetensor 格式正常加载 |
| 推理路径 | ✅ | `apply_weights()` 正确调用 `CuBLASLtFp8LinearOp` |

---

## 2. 当前 CUTLASS FP8 GEMM 实现详解

### 2.1 完整调用链

```
CompressedTensorsW8A8Fp8.apply_weights()
    │
    ├─→ QuantFP8.apply()  [输入量化，可选]
    │       └─→ per-token / per-tensor 量化
    │
    └─→ Fp8LinearOp.apply()
            │
            ├─→ cutlass_w8a8_scaled_mm()  [CUDA 12.0+, SM90+]
            │       └─→ ops.cutlass_scaled_mm()
            │               └─→ cutlass_scaled_mm_sm90_fp8()
            │
            ├─→ ops.scaled_fp8_quant()  [Flash-attention 路径]
            │
            └─→ torch._scaled_mm()  [Fallback]
```

### 2.2 输入量化过程（QuantFP8）

```python
# vllm/model_executor/layers/quantization/input_quant_fp8.py
class QuantFP8(CustomOp):
    """FP8 输入量化，支持三种策略"""
    
    # 量化公式: x_fp8 = x / scale
    # scale 计算: scale = max(|x|) / fp8_max
    
    def __init__(self, quant_config):
        self.strategy = quant_config.input_strategy
        # "tensor" - 整个 tensor 共享一个 scale
        # "token"  - 每行（token）一个 scale  
        # "group"  - 每 group_size 个元素一个 scale
```

**量化策略说明：**

| 策略 | scale 形状 | 说明 |
|------|-----------|------|
| `per-tensor` | `[1]` | 整个输入共享一个 scale，精度最低但最快 |
| `per-token` | `[M, 1]` | 每行一个 scale，LLM 推理的典型配置 |
| `per-channel` | `[1, K]` | 每列一个 scale，权重量化常用 |

### 2.3 GEMM Layout 设计

**vLLM CUTLASS FP8 GEMM Layout:**

```
问题定义: C[M,N] = A[M,K] × B[K,N]

实际存储（CUTLASS 内部）:
    A: RowMajor    [M, K]  - 每行连续存储
    B: ColumnMajor [K, N]  - 等价于 [N, K]^T，每列连续存储  
    C: RowMajor    [M, N]
    D: RowMajor    [M, N]

在 vLLM 中:
    input (x):   [batch, hidden_dim] = [M, K]  RowMajor
    weight (w):  [out_features, in_features] = [N, K]  实际存储
                 传给 CUTLASS 时作为 ColumnMajor [K, N]
    output:      [batch, out_features] = [M, N]  RowMajor
```

**swap_ab 机制（性能优化）：**

```cpp
// csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh

// 当 M 很小时（decode 阶段），交换 A 和 B 以获得更好的性能
template <bool swap_ab = false>
void cutlass_scaled_mm_sm90_fp8_dispatch(...) {
    // swap_ab=true 时:
    //   实际计算: D^T = B^T × A^T
    //   等价于:   D   = A × B
    //   但利用了 B 的更好的内存访问模式
}

// 选择逻辑
if (M <= 64) {
    // 小 M 场景（decode），使用 swap_ab
    cutlass_scaled_mm_sm90_fp8_dispatch<true>(...);
} else {
    // 大 M 场景（prefill），不使用 swap_ab
    cutlass_scaled_mm_sm90_fp8_dispatch<false>(...);
}
```

### 2.4 Epilogue（融合后处理）详解

**CUTLASS 3.x Epilogue 计算公式：**

```
D = scale_a * (scale_b * Accumulator) + bias

具体展开（ScaledEpilogueBias）:
    Compute0: tmp = scale_b * Accum      (逐元素或逐行)
    Compute1: D = scale_a * tmp + bias   (逐元素或逐列)
```

**Epilogue 类型定义：**

```cpp
// csrc/cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp

// 基础 Epilogue（无 bias）
struct ScaledEpilogue {
    // scale_a: ColOrScalarLoad - 每列一个 scale 或全局 scalar
    // scale_b: RowOrScalarLoad - 每行一个 scale 或全局 scalar
    
    using EVTCompute = Sm90EVT<
        Compute1,           // D = scale_a * tmp
        ScaleA,             // ColOrScalarLoad
        Sm90EVT<
            Compute0,       // tmp = scale_b * Accum
            ScaleB,         // RowOrScalarLoad
            Accum           // 累加器输出
        >
    >;
};

// 带 Bias 的 Epilogue
struct ScaledEpilogueBias {
    // bias: RowLoad - 每行一个 bias（广播到所有列）
    
    using EVTCompute = Sm90EVT<
        Compute1,           // D = scale_a * tmp + bias
        ScaleA,             // ColOrScalarLoad
        Sm90EVT<
            Compute0,       // tmp = scale_b * Accum
            ScaleB,         // RowOrScalarLoad
            Accum
        >,
        Bias                // RowLoad
    >;
};

// swap_ab 场景的 Bias（列广播）
struct ScaledEpilogueColumnBias {
    // 当 swap_ab=true 时，bias 需要列方向加载
    using Bias = ColLoad<float>;
};
```

**Scale 加载模式：**

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| `ScalarLoad` | 全局单一 scale | per-tensor 量化 |
| `RowLoad` | 每行一个 scale | per-token (activation) |
| `ColLoad` | 每列一个 scale | per-channel (weight) |
| `RowOrScalarLoad` | 运行时选择 | 兼容两种模式 |
| `ColOrScalarLoad` | 运行时选择 | 兼容两种模式 |

### 2.5 Kernel 选择策略

```cpp
// csrc/quantization/w8a8/cutlass/c3x/scaled_mm_sm90_fp8_dispatch.cuh

template <typename OutType, bool swap_ab, bool with_bias>
void cutlass_gemm_sm90_fp8_dispatch(int M, int N, int K, ...) {
    
    // 根据问题规模选择最优 kernel 配置
    if (M <= 16) {
        // 极小 M：使用 M16N128K128 tile
        using TileShape = Shape<_16, _128, _128>;
        using ClusterShape = Shape<_1, _2, _1>;
        
    } else if (M <= 64) {
        // 小 M：使用 M64N128K128 tile
        using TileShape = Shape<_64, _128, _128>;
        using ClusterShape = Shape<_1, _2, _1>;
        
    } else if (M <= 128) {
        // 中等 M：使用 M128N128K128 tile
        using TileShape = Shape<_128, _128, _128>;
        using ClusterShape = Shape<_1, _1, _1>;
        
    } else if (N >= 8192) {
        // 大 N（宽矩阵）：使用专门配置
        using TileShape = Shape<_128, _256, _64>;
        using ClusterShape = Shape<_2, _1, _1>;
        
    } else {
        // 默认配置
        using TileShape = Shape<_128, _128, _128>;
        using ClusterShape = Shape<_2, _1, _1>;
    }
}
```

---

## 3. cuBLASLt 替换方案与注意事项

### 3.1 cuBLASLt FP8 GEMM 核心概念

**基本计算公式（Tensorwide Scaling）：**

```
D = scaleD * (α * scaleA * scaleB * op(A) × op(B) + β * scaleC * C)
```

**vLLM 场景简化（无 scaleC/scaleD，β=0）：**

```
D = α * scaleA * scaleB * op(A) × op(B) + bias
```

### 3.2 Layout 对应关系

| vLLM/CUTLASS | cuBLASLt | 说明 |
|--------------|----------|------|
| A: RowMajor | A: ColumnMajor + CUBLAS_OP_T | 转置后等价 |
| B: ColumnMajor | B: ColumnMajor + CUBLAS_OP_N | 直接对应 |
| C/D: RowMajor | C/D: ColumnMajor + 转置 | 需要额外处理 |

**推荐做法：使用 TN 格式**

```cpp
// cuBLASLt 最优配置（Ada/Hopper）
CUBLAS_OP_T  // A 转置
CUBLAS_OP_N  // B 不转置
```

### 3.3 Scale 处理方式对比

**CUTLASS 方式：**
- `scale_a`：per-token（列向量）或 scalar
- `scale_b`：per-channel（行向量）或 scalar
- 融合在 Epilogue 中计算

**cuBLASLt 方式：**
- `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`：指向 scaleA
- `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER`：指向 scaleB
- 支持两种 Scale Mode：
  - `CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F`：per-tensor（默认）
  - `CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F`：per-row/col（SM90+）

### 3.4 Bias 处理

**cuBLASLt Epilogue 选项：**

```cpp
// 设置 epilogue 类型
cublasLtMatmulDescSetAttribute(
    matmulDesc,
    CUBLASLT_MATMUL_DESC_EPILOGUE,
    &epilogue,  // CUBLASLT_EPILOGUE_BIAS
    sizeof(epilogue)
);

// 设置 bias 指针
cublasLtMatmulDescSetAttribute(
    matmulDesc,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER,
    &bias_ptr,
    sizeof(bias_ptr)
);

// 设置 bias 数据类型（可选，默认与输出相同）
cublasLtMatmulDescSetAttribute(
    matmulDesc,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
    &bias_type,  // CUDA_R_32F for FP8 kernels
    sizeof(bias_type)
);
```

**重要限制：**
- Bias 向量长度必须等于输出矩阵行数（M）
- Bias 被广播到所有列
- FP8 kernel 的 bias 类型通常是 `CUDA_R_16BF` 或 `CUDA_R_32F`

### 3.5 完整 cuBLASLt FP8 GEMM 实现框架

```cpp
// 伪代码示例
cublasStatus_t cublaslt_fp8_gemm(
    int M, int N, int K,
    const void* A,        // FP8 input [M, K]
    const void* B,        // FP8 weight [K, N] (stored as [N, K]^T)
    void* D,              // Output [M, N]
    const float* scale_a, // per-token scale [M] or scalar
    const float* scale_b, // per-channel scale [N] or scalar
    const float* bias,    // optional bias [M]
    bool is_scale_a_scalar,
    bool is_scale_b_scalar,
    cudaStream_t stream
) {
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    
    // 1. 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cudaDataType_t scaleType = CUDA_R_32F;
    cublasLtMatmulDescCreate(&matmulDesc, computeType, scaleType);
    
    // 2. 设置转置操作（TN 格式）
    cublasOperation_t opA = CUBLAS_OP_T;
    cublasOperation_t opB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    
    // 3. 设置 Scale 指针
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a));
    cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b));
    
    // 4. 设置 Scale Mode（per-tensor vs per-row/col）
    if (!is_scale_a_scalar || !is_scale_b_scalar) {
        // Outer vector scaling（SM90+ only）
        int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
        if (!is_scale_a_scalar) {
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        }
        if (!is_scale_b_scalar) {
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        }
    }
    
    // 5. 设置 Bias（如果有）
    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        
        cudaDataType_t biasType = CUDA_R_32F;  // 或 CUDA_R_16BF
        cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
    }
    
    // 6. 创建矩阵布局
    cublasLtMatrixLayout_t Adesc, Bdesc, Ddesc;
    
    // A: [K, M] ColumnMajor (因为 opA=T, 实际读取 [M, K] RowMajor)
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, M, K);
    
    // B: [K, N] ColumnMajor
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, N, K);
    
    // D: [M, N] (需要根据输出类型设置)
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, M, N, M);
    
    // 7. 获取算法启发式
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    
    size_t workspaceSize = 64 * 1024 * 1024;  // 64 MB
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc, Ddesc, Ddesc, preference, 1, &heuristicResult, &returnedResults);
    
    // 8. 执行矩阵乘法
    float alpha = 1.0f;
    float beta = 0.0f;
    void* workspace = nullptr;
    cudaMalloc(&workspace, heuristicResult.workspaceSize);
    
    cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        A, Adesc,
        B, Bdesc,
        &beta,
        D, Ddesc,  // C = D for in-place
        D, Ddesc,
        &heuristicResult.algo,
        workspace, heuristicResult.workspaceSize,
        stream
    );
    
    // 清理
    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(matmulDesc);
    cublasLtDestroy(handle);
    
    return CUBLAS_STATUS_SUCCESS;
}
```

### 3.6 关键注意事项

#### 3.6.1 数据类型支持

| Atype | Btype | Ctype | Dtype | 支持 |
|-------|-------|-------|-------|------|
| E4M3 | E4M3 | BF16 | BF16 | ✅ |
| E4M3 | E4M3 | FP16 | FP16 | ✅ |
| E4M3 | E4M3 | FP32 | FP32 | ✅ |
| E5M2 | E4M3 | BF16 | BF16 | ✅ |
| E4M3 | E5M2 | BF16 | BF16 | ✅ |

#### 3.6.2 Scale Mode 限制（Outer Vector Scaling）

```cpp
// SM90 (Hopper) 独有功能
CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F

// 限制:
// 1. 仅支持 SM90+
// 2. scaleD 不支持（输出必须是 FP16/BF16/FP32）
// 3. scaleA 长度 = M, scaleB 长度 = N
```

#### 3.6.3 对齐要求

- 所有矩阵指针必须 16 字节对齐
- 维度 M, K 最好是 16 的倍数
- workspace 必须 256 字节对齐

#### 3.6.4 性能优化建议

1. **复用 Handle 和 Descriptor**：创建开销大，应全局复用
2. **缓存 Heuristic 结果**：相同问题规模可复用算法选择
3. **Workspace 预分配**：避免运行时分配
4. **使用 Fast Accumulation**：`CUBLASLT_MATMUL_DESC_FAST_ACCUM = 1`

---

## 4. 实现计划与代码示例

### 4.1 修改 CuBLASLtFp8LinearOp

```python
# slidesparse/core/cublaslt_linear_method.py

class CuBLASLtFp8LinearOp:
    USE_REAL_CUBLASLT = True  # 启用真实 cuBLASLt
    
    def __init__(self):
        # 初始化 cuBLASLt handle（单例）
        self._handle = self._get_or_create_handle()
        
    @classmethod
    def _get_or_create_handle(cls):
        # 全局 handle 缓存
        if not hasattr(cls, '_global_handle'):
            cls._global_handle = cublaslt_create_handle()
        return cls._global_handle
    
    def apply(
        self,
        x: torch.Tensor,           # [M, K] FP8
        weight: torch.Tensor,      # [N, K] FP8
        x_scale: torch.Tensor,     # [M] or [1]
        weight_scale: torch.Tensor,# [N] or [1]
        bias: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        
        M, K = x.shape
        N = weight.shape[0]
        
        # 确定 scale mode
        is_x_scale_scalar = (x_scale.numel() == 1)
        is_w_scale_scalar = (weight_scale.numel() == 1)
        
        # 调用 cuBLASLt kernel
        output = cublaslt_fp8_gemm(
            self._handle,
            x.data_ptr(),
            weight.data_ptr(),
            x_scale.data_ptr(),
            weight_scale.data_ptr(),
            bias.data_ptr() if bias is not None else None,
            M, N, K,
            is_x_scale_scalar,
            is_w_scale_scalar,
            output_dtype,
            torch.cuda.current_stream().cuda_stream,
        )
        
        return output
```

### 4.2 CUDA 绑定实现

需要在 `csrc/` 目录下添加 cuBLASLt wrapper：

```cpp
// csrc/quantization/cublaslt_fp8_gemm.cu

#include <cublasLt.h>
#include <torch/extension.h>

// 全局 handle 管理
class CublasLtHandlePool {
public:
    static cublasLtHandle_t get() {
        static thread_local cublasLtHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasLtCreate(&handle);
        }
        return handle;
    }
};

torch::Tensor cublaslt_fp8_gemm(
    torch::Tensor A,        // [M, K] FP8
    torch::Tensor B,        // [N, K] FP8 (stored transposed)
    torch::Tensor scale_a,  // [M] or [1]
    torch::Tensor scale_b,  // [N] or [1]
    c10::optional<torch::Tensor> bias,
    bool is_scale_a_scalar,
    bool is_scale_b_scalar,
    torch::Dtype output_dtype
) {
    // 实现参见 3.5 节框架
}

// Python 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublaslt_fp8_gemm", &cublaslt_fp8_gemm, "cuBLASLt FP8 GEMM");
}
```

### 4.3 CMake 集成

```cmake
# cmake/cublaslt_extension.cmake

find_package(CUDAToolkit REQUIRED)

add_library(cublaslt_fp8_gemm SHARED
    csrc/quantization/cublaslt_fp8_gemm.cu
)

target_link_libraries(cublaslt_fp8_gemm
    CUDA::cublasLt
    ${TORCH_LIBRARIES}
)
```

---

## 5. 测试与验证计划

### 5.1 正确性测试

```python
def test_cublaslt_fp8_gemm_correctness():
    """对比 cuBLASLt 与 CUTLASS 结果"""
    M, N, K = 1024, 4096, 4096
    
    # 随机生成 FP8 输入
    x = torch.randn(M, K, device='cuda').to(torch.float8_e4m3fn)
    w = torch.randn(N, K, device='cuda').to(torch.float8_e4m3fn)
    scale_x = torch.rand(M, device='cuda')
    scale_w = torch.rand(N, device='cuda')
    bias = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    
    # CUTLASS 参考实现
    ref = cutlass_scaled_mm(x, w, scale_x, scale_w, bias)
    
    # cuBLASLt 实现
    out = cublaslt_fp8_gemm(x, w, scale_x, scale_w, bias)
    
    # 验证（FP8 计算允许一定误差）
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)
```

### 5.2 性能测试

```python
def benchmark_cublaslt_vs_cutlass():
    """性能对比测试"""
    configs = [
        (1, 4096, 4096),     # decode
        (32, 4096, 4096),    # small batch
        (128, 4096, 4096),   # medium batch
        (1024, 4096, 4096),  # large batch
        (4096, 4096, 4096),  # prefill
    ]
    
    for M, N, K in configs:
        # 预热
        for _ in range(10):
            cutlass_run(M, N, K)
            cublaslt_run(M, N, K)
        
        # 测量
        cutlass_time = benchmark(cutlass_run, M, N, K, iters=100)
        cublaslt_time = benchmark(cublaslt_run, M, N, K, iters=100)
        
        print(f"[{M}, {N}, {K}] CUTLASS: {cutlass_time:.2f}ms, cuBLASLt: {cublaslt_time:.2f}ms")
```

---

## 6. 总结

### 6.1 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| 环境变量切换 | ✅ 完成 | `VLLM_USE_CUBLASLT=1` |
| Scheme 包装架构 | ✅ 完成 | `CuBLASLtSchemeWrapper` |
| 权重加载兼容 | ✅ 完成 | 委托给原始 scheme |
| cuBLASLt 真实实现 | 🔄 待开发 | 本文档提供框架 |

### 6.2 开发优先级

1. **高优先级**：实现基础 cuBLASLt FP8 GEMM（per-tensor scale）
2. **中优先级**：支持 per-token/per-channel scale（Outer Vector Scaling）
3. **低优先级**：Bias 融合、GELU/ReLU 融合

### 6.3 风险与挑战

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| Layout 不匹配 | 结果错误 | 仔细验证转置逻辑 |
| Scale mode 限制 | SM89 不支持 outer vector | 回退到 per-tensor |
| 性能回退 | 小 M 场景可能更慢 | 根据规模动态选择后端 |
| cuBLAS 版本兼容 | API 差异 | 版本检测 + 条件编译 |

---

## 7. 当前外挂方法的委托详细分析

本章详细说明当前 `CuBLASLtFp8LinearMethod` / `CuBLASLtSchemeWrapper` 中各步骤的委托情况。

### 7.1 FP8 委托链路总览

| 步骤 | 当前状态 | 外挂函数 | 转发目标 | vLLM 具体位置 |
|------|----------|----------|----------|---------------|
| **权重加载** | ✅ 委托 | `create_weights()` | `original_scheme.create_weights()` | `compressed_tensors_w8a8_fp8.py:84-130` |
| **权重处理** | ✅ 委托 | `process_weights_after_loading()` | `original_scheme.process_weights_after_loading()` | `compressed_tensors_w8a8_fp8.py:132-172` |
| **输入加载** | N/A | - | 由 PyTorch 自动处理 | - |
| **输入量化** | ✅ 委托 | `apply()` → `_fp8_linear_op.apply()` | `Fp8LinearOp.quant_fp8()` | `w8a8_utils.py:462-467` → `QuantFP8` |
| **GEMM+反量化** | ✅ 委托 | `apply()` → `_fp8_linear_op.apply()` | `cutlass_w8a8_scaled_mm()` | `w8a8_utils.py:150-165` |
| **输出返回** | ✅ 委托 | `apply()` 返回 | `_fp8_linear_op.apply()` 返回 | BF16 tensor |

### 7.2 FP8 各步骤详细说明

#### 7.2.1 权重加载 (`create_weights`)

```
CuBLASLtFp8LinearMethod.create_weights()
    │
    └─→ self.original_scheme.create_weights()  # 完全委托
            │
            └─→ CompressedTensorsW8A8Fp8.create_weights()
                    │
                    ├─→ create_fp8_weight_parameter()      # 创建 FP8 权重 [N, K]
                    ├─→ create_fp8_scale_parameter()       # 创建 weight_scale
                    └─→ create_fp8_input_scale() (可选)    # 静态量化时创建 input_scale
```

**当前状态**：完全委托给原始 scheme，不做任何修改。
**后续计划**：如需自定义权重格式，需要在此处介入。

#### 7.2.2 权重处理 (`process_weights_after_loading`)

```
CuBLASLtFp8LinearMethod.process_weights_after_loading()
    │
    └─→ self.original_scheme.process_weights_after_loading()  # 完全委托
            │
            └─→ CompressedTensorsW8A8Fp8.process_weights_after_loading()
                    │
                    ├─→ process_fp8_weight_tensor_strategy()   # per-tensor
                    ├─→ process_fp8_weight_channel_strategy()  # per-channel ← Qwen 使用
                    └─→ process_fp8_weight_block_strategy()    # block
                    │
                    └─→ weight = weight.t()  # 关键：权重转置为 [K, N]
```

**关键处理**：
- 根据策略处理 weight_scale（per-tensor/per-channel/block）
- **权重转置**：从 `[N, K]` 转为 `[K, N]`
- 将 weight 和 weight_scale 转为 `torch.nn.Parameter`

**当前状态**：完全委托。
**注意**：cuBLASLt 需要特定 layout，可能需要修改此处。

#### 7.2.3 输入量化

```
CuBLASLtFp8LinearOp.apply()
    │
    └─→ self._fp8_linear_op.apply()  # 委托给 vLLM 的 Fp8LinearOp
            │
            └─→ Fp8LinearOp.apply() [w8a8_utils.py:440-490]
                    │
                    └─→ self.quant_fp8(input_2d, input_scale, input_scale_ub)
                            │
                            └─→ QuantFP8.__call__() [input_quant_fp8.py]
                                    │
                                    └─→ ops.scaled_fp8_quant(input, scale)
                                            │
                                            └─→ 返回 (qinput, x_scale)
                                                     FP8      FP32
```

**量化公式**：`qinput = input / scale`，其中 `scale = max(|input|) / fp8_max`

**Scale 形状（Qwen FP8 配置）**：
- `x_scale`: `[M, 1]` (per-token)
- `weight_scale`: `[N, 1]` (per-channel)

**当前状态**：完全委托给 `QuantFP8`。
**后续计划**：如需自定义量化，在 `CuBLASLtFp8LinearOp.apply()` 中直接调用自己的量化函数。

#### 7.2.4 GEMM + 反量化

```
Fp8LinearOp.apply() [w8a8_utils.py:480-490]
    │
    ├─→ dispatch_w8a8_scaled_mm(preferred_backend, ...)  # 选择后端
    │       │
    │       └─→ 返回 cutlass_w8a8_scaled_mm (CUDA + SM90)
    │
    └─→ cutlass_w8a8_scaled_mm() [w8a8_utils.py:150-165]
            │
            └─→ ops.cutlass_scaled_mm(qinput, weight, scale_a, scale_b, bias)
                    │
                    └─→ C++ 调用: cutlass_scaled_mm_sm90_fp8()
                            │
                            └─→ output = scale_a * (scale_b * (qinput @ weight.T)) + bias
```

**GEMM 参数**：
| 参数 | 形状 | 说明 |
|------|------|------|
| `qinput` | `[M, K]` FP8 | 量化后的输入 |
| `weight` | `[K, N]` FP8 | 转置后的权重 |
| `scale_a` | `[M, 1]` FP32 | 输入 scale (per-token) |
| `scale_b` | `[N, 1]` FP32 | 权重 scale (per-channel) |
| `bias` | `[N]` BF16 | 可选偏置 |
| `output` | `[M, N]` BF16 | 输出 |

**当前状态**：通过 `_fp8_linear_op.apply()` 间接委托给 cutlass。
**替换点**：这里是 cuBLASLt 替换的关键位置。

### 7.3 INT8 委托链路总览

| 步骤 | 当前状态 | 外挂函数 | 转发目标 | vLLM 具体位置 |
|------|----------|----------|----------|---------------|
| **权重加载** | ❌ 未支持 | - | - | `compressed_tensors_w8a8_int8.py:43-96` |
| **权重处理** | ❌ 未支持 | - | - | `cutlass.py:34-109` (kernel.process_weights) |
| **输入量化** | ❌ 未支持 | - | `ops.scaled_int8_quant()` | `cutlass.py:127-129` |
| **GEMM+反量化** | ❌ 未支持 | - | `ops.cutlass_scaled_mm()` | `cutlass.py:144-147` |

### 7.4 INT8 详细说明

当前 `wrap_scheme_with_cublaslt()` **不支持 INT8**，仅检测 `W8A8Fp8`：

```python
# cublaslt_linear_method.py:287
if "W8A8Fp8" in scheme_name:
    return CuBLASLtFp8LinearMethod(original_scheme)
else:
    # INT8 会走这里，返回原始 scheme
    return original_scheme
```

**INT8 的架构差异**：

1. **Scheme 类**：`CompressedTensorsW8A8Int8`
2. **Kernel 选择**：使用 `ScaledMMLinearKernel` 架构
   - `CutlassScaledMMLinearKernel` (CUDA)
   - `TorchScaledMMLinearKernel` (fallback)
3. **量化函数**：`ops.scaled_int8_quant()` vs FP8 的 `ops.scaled_fp8_quant()`
4. **非对称支持**：INT8 支持 asymmetric quantization (AZP)

**INT8 关键函数位置**：

| 函数 | 文件 | 说明 |
|------|------|------|
| `create_weights` | `compressed_tensors_w8a8_int8.py:43-96` | 创建 INT8 权重和 scale |
| `process_weights_after_loading` | `cutlass.py:34-109` | 权重转置、scale 处理、AZP 计算 |
| `apply_weights` | `cutlass.py:115-147` | INT8 量化 + GEMM |
| `scaled_int8_quant` | `_custom_ops` | INT8 动态/静态量化 |
| `cutlass_scaled_mm_azp` | `_custom_ops` | 带 AZP 的 INT8 GEMM |

### 7.5 后续接管计划

#### FP8 接管步骤

1. **保持委托**：`create_weights()`, `process_weights_after_loading()`
2. **自主实现**：
   - 在 `CuBLASLtFp8LinearOp.apply()` 中：
     - 直接调用自定义的 FP8 量化函数（或复用 `QuantFP8`）
     - 调用 cuBLASLt FP8 GEMM kernel
     - 返回 BF16 输出

```python
# CuBLASLtFp8LinearOp._apply_cublaslt() 的目标实现
def _apply_cublaslt(self, input, weight, weight_scale, ...):
    # 1. 输入量化（可复用 QuantFP8 或自实现）
    qinput, x_scale = self.quant_fp8(input, input_scale)
    
    # 2. cuBLASLt GEMM（替换 cutlass_scaled_mm）
    output = cublaslt_fp8_gemm(
        qinput, weight, x_scale, weight_scale, bias
    )
    
    # 3. 返回 BF16 输出
    return output
```

#### INT8 支持计划

1. 创建 `CuBLASLtInt8LinearMethod` 类
2. 在 `wrap_scheme_with_cublaslt()` 中添加 INT8 检测
3. 实现 INT8 版本的 `apply_weights()`

---

## 8. 测试框架设计

### 8.1 测试脚本架构

测试脚本 `test_cublaslt_00_kernel.py` 设计为可独立运行，直接测试 GEMM kernel 的正确性和性能。

```
测试流程:
┌─────────────────────────────────────────────────────────────┐
│  1. 生成测试数据                                             │
│     - input_bf16 [M, K] - BF16 格式                         │
│     - weight_fp8 [N, K] - FP8 格式（转置后 [K, N]）           │
│     - weight_scale [N, 1] - per-channel                     │
│     - bias [N] - 可选                                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 调用被测 kernel（通过 CuBLASLtFp8LinearOp）               │
│     - 内部会进行 BF16 → FP8 量化（per-token dynamic）        │
│     - 执行 FP8 GEMM（当前是 cutlass，将替换为 cuBLASLt）      │
│     - 返回 BF16 输出                                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 计算参考结果（模拟完整 FP8 GEMM 流程）                    │
│     - 输入量化：input_bf16 → input_fp8, x_scale             │
│     - FP8 矩阵乘：input_fp8 @ weight_fp8_t                   │
│     - 反量化：result * x_scale * weight_scale               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 比较结果和测量性能                                       │
│     - 正确性：torch.allclose(output, reference)             │
│     - 性能：测量吞吐量 (TFLOPS)                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 测试接口设计

为支持独立测试，在 `CuBLASLtFp8LinearOp` 中已添加专用接口：

```python
class CuBLASLtFp8LinearOp:
    def apply_for_test(
        self,
        input: torch.Tensor,           # [M, K] BF16，会被量化
        weight: torch.Tensor,          # [K, N] FP8，已量化已转置，column-major
        weight_scale: torch.Tensor,    # [N, 1] FP32
        out_dtype: torch.dtype = torch.bfloat16,
        input_scale: torch.Tensor | None = None,  # None = dynamic quant
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        专用于测试的接口，跳过 layer 对象依赖
        """
        ...
```

### 8.3 测试脚本使用

```bash
# 运行单个规模测试
python slidesparse/test/test_cublaslt_00_kernel.py --m 256 --n 896 --k 896

# 运行完整测试套件
python slidesparse/test/test_cublaslt_00_kernel.py --all
```

### 8.4 测试结果（RTX 5080, SM 12.0）

| M | N | K | Bias | Time(ms) | TFLOPS | Status |
|---:|---:|---:|:---:|---:|---:|:---:|
| 1 | 896 | 896 | No | 0.050 | 0.03 | ✅ |
| 1 | 4864 | 896 | No | 0.049 | 0.18 | ✅ |
| 32 | 896 | 896 | No | 0.049 | 1.05 | ✅ |
| 32 | 4864 | 896 | No | 0.049 | 5.68 | ✅ |
| 128 | 4864 | 896 | No | 0.049 | 22.76 | ✅ |
| 128 | 896 | 4864 | Yes | 0.105 | 10.66 | ✅ |
| 512 | 4864 | 896 | No | 0.061 | 73.64 | ✅ |
| 1024 | 4864 | 896 | No | 0.090 | 98.87 | ✅ |
| 2048 | 2048 | 2048 | No | 0.181 | 95.14 | ✅ |
| 4096 | 4096 | 4096 | No | 1.232 | 111.52 | ✅ |

**测试环境说明**：
- 当前底层 kernel 为 CUTLASS `cutlass_scaled_mm`
- 替换为 cuBLASLt 后，需重新运行测试验证正确性和性能差异
- 误差容限：rtol=0.05, atol=0.05（FP8 量化误差正常范围）

---

## 9. vLLM GEMM 后端分发机制深度分析

本章详细解答关于 FlashInfer、Padding、Triton 回退等关键问题。

### 9.1 FlashInfer 是什么？什么时候会被调用？

#### 9.1.1 FlashInfer 简介

**FlashInfer** 是由 NVIDIA 和社区开发的**高性能 Attention 和 GEMM 内核库**，专门针对 LLM 推理优化。它提供了：

1. **FlashAttention 变体**：高效的注意力计算
2. **FP8 GEMM**：基于 `bmm_fp8` 的批量矩阵乘法
3. **MoE 相关算子**：Fused MoE、AlltoAll 等

**关键代码位置**：`vllm/utils/flashinfer.py`

```python
# flashinfer_scaled_fp8_mm 的实现
def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,  # [M, K] FP8
    b: torch.Tensor,  # [K, N] FP8
    scale_a: torch.Tensor,  # scalar only!
    scale_b: torch.Tensor,  # scalar only!
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # ⚠️ 重要限制：只支持 per-tensor scale
    assert scale_a.numel() == 1 and scale_b.numel() == 1
    
    output = bmm_fp8(
        a.unsqueeze(0),
        b.unsqueeze(0),
        scale_a,
        scale_b,
        out_dtype,
        "auto",
    ).view(a.shape[0], b.shape[1])
    
    if bias is not None:
        output = output + bias
    return output
```

#### 9.1.2 FlashInfer 的启用条件

在 `Fp8LinearOp.__init__()` 中（[w8a8_utils.py:405-416](vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L405-L416)）：

```python
class Fp8LinearOp:
    def __init__(self, ...):
        if current_platform.is_rocm():
            self.preferred_backend = "rocm"
        elif current_platform.is_cuda() and cutlass_fp8_supported():
            # 关键条件：CC >= 100 (Blackwell B100/B200) 且 flashinfer 可用
            if has_flashinfer() and current_platform.has_device_capability(100):
                self.preferred_backend = "flashinfer"
            else:
                self.preferred_backend = "cutlass"
        else:
            self.preferred_backend = "torch"
```

**FlashInfer FP8 GEMM 启用条件**：
| 条件 | 说明 |
|------|------|
| `has_flashinfer()` | FlashInfer Python 包已安装 |
| `has_device_capability(100)` | **SM >= 100 (Blackwell)** |
| `per_tensor_weights and per_tensor_activations` | **必须是 per-tensor 量化** |

#### 9.1.3 为什么 RTX 5080 不走 FlashInfer？

**RTX 5080 是 SM 12.0 (Blackwell Consumer)**，满足 CC >= 100 的条件。但是，让我们看 `dispatch_w8a8_scaled_mm` 的逻辑（[w8a8_utils.py:363-378](vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L363-L378)）：

```python
def dispatch_w8a8_scaled_mm(
    preferred_backend: str, per_tensor_weights: bool, per_tensor_activations: bool
) -> Callable[..., torch.Tensor]:
    
    # 情况 1: per-tensor W 和 per-tensor A
    if per_tensor_weights and per_tensor_activations:
        if preferred_backend == "flashinfer":
            return flashinfer_w8a8_scaled_mm  # ✅ 走 FlashInfer
        if preferred_backend == "cutlass":
            return cutlass_w8a8_scaled_mm
        ...
    
    # 情况 2: per-channel W 或 per-token A（我们的 Qwen FP8 配置）
    # cutlass_scaled_mm supports per tensor/channel W and per tensor/token A
    if preferred_backend == "cutlass" or preferred_backend == "flashinfer":
        return cutlass_w8a8_scaled_mm  # ⚠️ 回退到 CUTLASS！
```

**结论**：
- **Qwen FP8 模型使用 per-channel weight + per-token activation**
- FlashInfer 的 `bmm_fp8` **只支持 per-tensor scale**
- 因此即使 preferred_backend="flashinfer"，也会**回退到 CUTLASS**

**简单总结**：

| 显卡 | SM | preferred_backend | 实际使用 | 原因 |
|------|-----|-------------------|---------|------|
| H100 | 90 | cutlass | CUTLASS | SM < 100 |
| B100/B200 | 100 | flashinfer | CUTLASS | per-channel/per-token 不支持 |
| RTX 5080 | 120 | flashinfer | CUTLASS | per-channel/per-token 不支持 |
| 任意 | - | - | FlashInfer | 仅当 per-tensor W + per-tensor A |

### 9.2 Padding 机制详解

#### 9.2.1 为什么需要 Padding？

CUTLASS 和 cuBLASLt 的 FP8 GEMM 内核对矩阵维度有对齐要求：
- **M, K, N 最好是 16 的倍数**
- 不对齐会导致性能下降或调用 fallback 路径

#### 9.2.2 vLLM 的 Padding 策略

在 `Fp8LinearOp.__init__()` 中（[w8a8_utils.py:420-430](vllm/model_executor/layers/quantization/utils/w8a8_utils.py#L420-L430)）：

```python
class Fp8LinearOp:
    def __init__(self, ..., pad_output: bool | None = None):
        # pad_output 的默认值逻辑
        if pad_output is None:
            config = get_current_vllm_config().compilation_config
            pad_output = (
                # 条件1: 没有使用 torch.compile
                config.mode < CompilationMode.VLLM_COMPILE
                # 条件2: 使用 torch 后端（不是 cutlass/flashinfer）
                and self.preferred_backend == "torch"
            )
        
        # 如果需要 padding，pad 到 17（而不是 16）
        # 这是因为 torch._scaled_mm 在 batch > 16 时性能更好
        self.output_padding = 17 if pad_output else None
```

**关键结论**：

| preferred_backend | torch.compile | 是否 Padding |
|-------------------|---------------|--------------|
| cutlass | 否 | ❌ 不 Padding |
| cutlass | 是 | ❌ 不 Padding |
| flashinfer | 否 | ❌ 不 Padding |
| flashinfer | 是 | ❌ 不 Padding |
| torch | 否 | ✅ Padding 到 17 |
| torch | 是 | ❌ 不 Padding（会破坏动态 shape） |

#### 9.2.3 Padding 在哪里发生？

Padding 发生在 `QuantFP8` 的量化过程中，**不是在 GEMM 阶段**：

```python
# input_quant_fp8.py
class QuantFP8(CustomOp):
    def __init__(self, ..., num_token_padding: int | None = None):
        self.num_token_padding = num_token_padding
    
    def __call__(self, input, scale, scale_ub):
        if self.num_token_padding:
            # 对输入进行 padding
            input = pad_to(input, self.num_token_padding)
        return ops.scaled_fp8_quant(input, scale)
```

然后在 GEMM 输出时通过 `torch.narrow` 截取有效部分：

```python
def torch_per_tensor_w8a8_scaled_mm(...):
    output = torch._scaled_mm(qinput, weight, ...)
    # 截取有效输出（去除 padding 部分）
    return torch.narrow(output, 0, 0, qinput.shape[0]).view(*output_shape)
```

#### 9.2.4 如果 M=31 会怎样？

**对于 CUTLASS/cuBLASLt 后端**（我们的情况）：

1. **不会进行 Padding**：`output_padding = None`
2. **CUTLASS 可以处理任意 M**：通过 tile masking 处理边界
3. **性能可能略有下降**：非对齐访问会导致部分 tile 浪费

**对于 torch 后端**（fallback）：
1. **会 Padding 到 17**：如果 M < 17
2. **然后 narrow 回 M**：输出时截取

### 9.3 Triton 回退机制

#### 9.3.1 何时会回退到 Triton？

在 `ops.cutlass_scaled_mm()` 函数中（[_custom_ops.py:863-875](vllm/_custom_ops.py#L863-L875)）：

```python
def cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    # ...
    
    # 关键检查：b 的维度是否对 16 对齐
    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    
    if current_platform.is_rocm() or not cutlass_compatible_b:
        # 回退到 Triton 实现
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )
        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        # 使用 CUTLASS
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
    
    return out.view(*target_shape)
```

#### 9.3.2 这里的 B 是什么？

**B 是权重矩阵（Weight）**，不是激活（Activation）！

```python
# 在 cutlass_w8a8_scaled_mm 中的调用
ops.cutlass_scaled_mm(
    qinput,   # A: 激活 [M, K]
    weight,   # B: 权重 [K, N]（已转置）
    ...
)
```

**权重在加载时已经转置**，所以：
- 原始权重：`[N, K]`（out_features, in_features）
- 转置后权重：`[K, N]`
- `b.shape[0] = K`，`b.shape[1] = N`

#### 9.3.3 权重是否对齐？

**通常权重是对齐的**，因为：
- `K = hidden_dim`（如 896, 4096 等）通常是 16 的倍数
- `N = out_features`（如 896, 4864 等）通常也是 16 的倍数

**但如果不对齐**（例如某些特殊模型），就会回退到 Triton。

#### 9.3.4 M 不对齐会影响吗？

**M 不对齐不会导致回退到 Triton**！

检查的是 `b.shape`（权重的 K 和 N），不是 `a.shape`（激活的 M）。

CUTLASS 内核通过 tile masking 处理 M 边界，所以：
- M=31 → 使用 CUTLASS
- M=1 → 使用 CUTLASS
- 只有 K 或 N 不对齐才回退到 Triton

### 9.4 torch.ops._C.cutlass_scaled_mm 的来源

#### 9.4.1 绑定位置

`torch.ops._C.cutlass_scaled_mm` 是通过 PyTorch 的 C++ 扩展机制注册的。

**声明**（[csrc/torch_bindings.cpp:436-439](csrc/torch_bindings.cpp#L436-L439)）：
```cpp
ops.def(
    "cutlass_scaled_mm(Tensor! out, Tensor a,"
    " Tensor b, Tensor a_scales, Tensor b_scales, Tensor? bias) -> ()");
ops.impl("cutlass_scaled_mm", torch::kCUDA, &cutlass_scaled_mm);
```

**实现入口**（[csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu:176-231](csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu#L176-L231)）：
```cpp
void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias) {
    // 根据 SM 版本分发到不同实现
    int32_t version_num = get_sm_version_num();
    
    if (version_num >= 120) {
        cutlass_scaled_mm_sm120(c, a, b, a_scales, b_scales, bias);  // Blackwell Consumer
    } else if (version_num >= 100) {
        cutlass_scaled_mm_sm100(c, a, b, a_scales, b_scales, bias);  // Blackwell Datacenter
    } else if (version_num >= 90) {
        cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);   // Hopper
    } else if (version_num == 89) {
        cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);   // Ada Lovelace
    } else if (version_num >= 80) {
        cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);   // Ampere
    }
    // ...
}
```

#### 9.4.2 CUTLASS 是什么？

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) 是 NVIDIA 开源的**高性能 GEMM 模板库**：

- GitHub: https://github.com/NVIDIA/cutlass
- 提供各种数据类型的 GEMM 实现（FP8, INT8, FP16, BF16 等）
- 针对每代 GPU 架构优化（SM75/80/89/90/100/120）
- vLLM 使用 CUTLASS 作为默认的 FP8 GEMM 后端

**你可以把它理解为**：一个开源的、高性能的 GEMM 黑盒，vLLM 已经封装好了。

### 9.5 当前 CuBLASLtFp8LinearOp.apply 的完整性分析

#### 9.5.1 当前实现回顾

当前的 `CuBLASLtFp8LinearOp.apply()` 已经是完整的 **quant + GEMM + dequant** 流程：

```python
def apply(self, input, weight, weight_scale, out_dtype, input_scale, input_scale_ub, bias):
    # 1. 展平输入
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]
    
    # 2. 量化（使用自己的 QuantFP8 实例）
    if input.dtype != current_platform.fp8_dtype():
        qinput, x_scale = self.quant_fp8(input_2d, input_scale, input_scale_ub)
    else:
        qinput, x_scale = input_2d, input_scale
    
    # 3. GEMM + 反量化（当前调用 cutlass，后续替换为 cuBLASLt）
    return cublaslt_w8a8_scaled_mm(
        qinput=qinput,
        weight=weight,
        out_dtype=out_dtype,
        scale_a=x_scale,
        scale_b=weight_scale,
        bias=bias,
        output_shape=output_shape,
    )
```

#### 9.5.2 是否有问题？

**当前实现是正确的**，与 vLLM 原生 `Fp8LinearOp.apply()` 逻辑一致。

**潜在问题和注意事项**：

| 问题 | 当前状态 | 说明 |
|------|----------|------|
| Padding | ✅ 无问题 | 我们使用 `num_token_padding=None`，不做 padding |
| Scale mode 检测 | ⚠️ 可改进 | 当前直接调用 CUTLASS，没有检测 per-tensor/per-token |
| 后端分发 | ⚠️ 简化 | 跳过了 `dispatch_w8a8_scaled_mm()`，直接用 cutlass |

#### 9.5.3 后续替换位置

**只需要修改 `cublaslt_w8a8_scaled_mm` 函数**即可完成 cuBLASLt 替换：

```python
def cublaslt_w8a8_scaled_mm(*, qinput, weight, out_dtype, scale_a, scale_b, bias, output_shape):
    """
    当前实现：调用 cutlass_scaled_mm（验证架构正确性）
    后续实现：替换为真正的 cuBLASLt kernel
    """
    # TODO: Phase 3 完成后替换为真正的 cuBLASLt kernel
    # output = ops.cublaslt_scaled_mm(qinput, weight, scale_a, scale_b, bias)
    
    # 当前：调用 cutlass
    output = ops.cutlass_scaled_mm(
        qinput, weight, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias
    )
    return output.view(*output_shape)
```

#### 9.5.4 替换时需要注意的点

1. **Layout 一致性**：
   - CUTLASS 的 B 是 column-major `[K, N]`（stride: K=1, N=K）
   - cuBLASLt 也使用 column-major，但可能需要调整 leading dimension

2. **Scale 处理**：
   - `scale_a`: per-token `[M, 1]` 或 per-tensor `[1]`
   - `scale_b`: per-channel `[N, 1]` 或 per-tensor `[1]`
   - cuBLASLt 的 per-row/col scale 需要 `OUTER_VEC_32F` mode（SM90+ only）

3. **输出格式**：
   - 确保 cuBLASLt 输出与 CUTLASS 一致（BF16，相同 shape）

4. **对齐要求**：
   - cuBLASLt 对指针对齐有更严格要求（16 字节）
   - 需要检查 qinput、weight 是否满足

---

## 10. 总结与下一步

### 10.1 关键发现

1. **FlashInfer FP8 GEMM 只支持 per-tensor scale**，我们的 Qwen FP8（per-channel W + per-token A）不会使用它
2. **Padding 只对 torch 后端生效**，CUTLASS 不需要 padding
3. **Triton 回退只看权重维度**（K 和 N），M 不对齐不会触发回退
4. **当前 `CuBLASLtFp8LinearOp.apply()` 实现完整正确**，只需替换 `cublaslt_w8a8_scaled_mm` 即可

### 10.2 下一步操作

1. 在 `csrc/` 目录实现真正的 cuBLASLt FP8 GEMM wrapper
2. 替换 `cublaslt_w8a8_scaled_mm` 中的 `ops.cutlass_scaled_mm` 调用
3. 运行测试验证正确性和性能

---

## 11. cuBLASLt 集成关键问题分析

本章针对 cuBLASLt 替换 CUTLASS 的关键技术问题进行详细分析。

### 11.1 问题概览与实现计划

| 问题编号 | 问题描述 | 状态 |
|---------|---------|------|
| Q1 | Layout 分析：CUTLASS 的 A/W/Output 布局 | ✅ 已分析 |
| Q2 | cuBLASLt T/N+C/C 格式与 vLLM 的对接 | ✅ 已分析 |
| Q3 | Scale 维度与反量化机制 | ✅ 已分析 |
| Q4 | Bias 广播方向 | ✅ 已分析 |
| Q5 | cuBLASLtMatmul API 调用要点 | ✅ 已分析 |

---

### 11.2 Q1: CUTLASS 的 A/W/Output Layout 分析

#### 11.2.1 Safetensor 原始存储格式

从 checkpoint 分析（Qwen2.5-0.5B-FP8）：

```
Weight 原始格式:
    down_proj.weight:       [896, 4864]   FP8    → [N, K] 行主序
    down_proj.weight_scale: [896, 1]      BF16   → [N, 1] per-channel
    gate_proj.weight:       [4864, 896]   FP8    → [N, K] 行主序
    gate_proj.weight_scale: [4864, 1]     BF16   → [N, 1] per-channel

Bias 格式（仅 QKV proj 有 bias）:
    q_proj.bias: [896]  BF16 → [N] 1D 向量
    k_proj.bias: [128]  BF16 → [N] 1D 向量
```

**关键发现**：
- Weight: `[N, K]` 行主序（N=out_features, K=in_features）
- weight_scale: `[N, 1]` per-channel（**不是 `[1, K]`**，你之前的猜测需要修正）
- Bias: `[N]` 1D 向量

#### 11.2.2 vLLM 权重处理流程

在 `compressed_tensors_w8a8_fp8.py` 的 `process_weights_after_loading()` 中：

```python
# 第 145/151 行：关键的转置操作
if self.strategy == QuantizationStrategy.TENSOR:
    weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(...)
    weight = weight.t()   # [N, K] → [K, N]

elif self.strategy == QuantizationStrategy.CHANNEL:
    weight, weight_scale, input_scale = process_fp8_weight_channel_strategy(...)
    weight = weight.t()   # [N, K] → [K, N]
```

**转置后的权重格式**：
- `weight`: `[K, N]`，但在 PyTorch 中 `.t()` 后 **stride 变化**
- 原始 `[N, K]` 行主序的 stride 是 `(K, 1)`
- `.t()` 后变成 `[K, N]`，stride 是 `(1, K)` → **这就是列主序！**

#### 11.2.3 CUTLASS 期望的 Layout

从 `scaled_mm_entry.cu` 第 186-188 行的检查：

```cpp
// Check for strides and alignment
TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
TORCH_CHECK(b.stride(0) == 1);                      // Column-major
```

从 `scaled_mm.cuh` 第 73-75 行的定义：

```cpp
ElementAB, cutlass::layout::RowMajor, AlignmentAB,     // A: RowMajor
ElementAB, cutlass::layout::ColumnMajor, AlignmentAB,  // B: ColumnMajor
```

**CUTLASS 期望的输入**：

| 矩阵 | 形状 | Layout | stride | 说明 |
|------|------|--------|--------|------|
| A (input) | `[M, K]` | RowMajor | `(K, 1)` | 激活，每行连续 |
| B (weight) | `[K, N]` | ColumnMajor | `(1, K)` | 权重，每列连续 |
| C (output) | `[M, N]` | RowMajor | `(N, 1)` | 输出，每行连续 |

**PyTorch 视角**：
- `A [M, K]` 行主序 → stride `(K, 1)` ✅
- `B [K, N]` 列主序 = `[N, K].t()` → stride `(1, K)` ✅
- `C [M, N]` 行主序 → stride `(N, 1)` ✅

#### 11.2.4 CUTLASS 计算公式确认

```
CUTLASS 计算: C[M,N] = A[M,K] × B[K,N]
```

其中 B 是列主序存储的 `[K, N]`，等价于 PyTorch 中 `weight.t()`。

---

### 11.3 Q2: cuBLASLt T/N+C/C 格式对接

#### 11.3.1 你的需求确认

你需要使用 **W 在左，A 在右** 的计算顺序：
```
cuBLASLt 计算: D = W × A^T
```

并且要求 **T/N + C/C/C** 格式（A 转置，B 不转置，全部列主序）。

#### 11.3.2 Layout 推导

**设定**：
- PyTorch 传入的 `A [M, K]` 行主序，stride `(K, 1)`
- PyTorch 传入的 `W [K, N]` 列主序（即 `[N, K].t()`），stride `(1, K)`

**cuBLASLt 用列主序读取行主序 = 隐式转置**：

| 矩阵 | PyTorch 存储 | cuBLASLt 读取方式 | 实际读到的 |
|------|-------------|------------------|-----------|
| A `[M, K]` row | 内存: `M×K` 连续 | 列主序读 | `A^T [K, M]` |
| W `[K, N]` col | 内存: `K×N` 连续 | 列主序读 | `W [K, N]` |

**计算过程**：

```
cuBLASLt T/N 配置:
    opA = CUBLAS_OP_T  → 对 "列主序读到的 A^T" 再转置 → 得到 A [M, K]
    opB = CUBLAS_OP_N  → 对 "列主序读到的 W" 不转置 → 得到 W [K, N]
    
等等，这和你想要的 W×A^T 不一样！
```

**重新理解你的需求**：

你说的 "W 在左，A 在右" 是指 cuBLASLt API 的参数顺序，而不是数学上的矩阵乘法顺序。

让我重新推导：

```
你想要的最终计算（数学上）: Output[M, N] = A[M, K] × W^T[K, N]

但 vLLM 传给你的 W 已经是 [K, N] 列主序了（已转置过）！

所以实际计算: Output[M, N] = A[M, K] × W_transposed[K, N]
```

#### 11.3.3 正确的 cuBLASLt 配置

**方案：让 cuBLASLt 计算 D = A × B**

cuBLASLt 默认是列主序，用行主序数据时需要技巧：

```
A: [M, K] 行主序，stride (K, 1)
   → cuBLASLt 用列主序读 → 读成 [K, M]
   → opA = T 转置回来 → 得到 [M, K]

B: [K, N] 列主序，stride (1, K)  
   → cuBLASLt 用列主序读 → 正好读成 [K, N]
   → opB = N 不转置 → 得到 [K, N]

D: [M, N] 行主序
   → cuBLASLt 用列主序写 → 写成 [N, M]^T
   → 但行主序的 [M, N] 存储等于列主序的 [N, M] ✅
```

**但是你要求 "W 在左，A 在右"！**

这需要交换 A 和 B 的位置，利用 `(A×B)^T = B^T × A^T` 的性质：

```
cuBLASLt 参数顺序: D' = B' × A'  （B'在左，A'在右）

设：
    B' = A^T = [K, M] （用列主序读行主序的 A[M,K]）
    A' = W   = [K, N] （列主序的 W）
    
那么：
    D' = A^T × W = [K, M]^T × [K, N]  ???  维度不对！
```

**正确理解**：你想要的是 cuBLASLt 计算 `D^T = W^T × A^T`，然后结果自动变成 `D`：

```
关系: D = A × W  等价于  D^T = W^T × A^T

所以:
    opA = T (对 cuBLAS 的第一个矩阵 W)
    opB = T (对 cuBLAS 的第二个矩阵 A)
    
但这是 T/T 配置，不是你要的 T/N！
```

#### 11.3.4 T/N + C/C/C 配置的正确用法

让我重新理解你的意图。T/N 意味着：
- 第一个矩阵参数：转置
- 第二个矩阵参数：不转置

```
cublasLtMatmul 的标准计算: D = α × op(A) × op(B) + β × C

设:
    第一个矩阵参数 = W_stored [N, K] 行主序
    第二个矩阵参数 = A_stored [M, K] 行主序
    opA = T → W_stored^T = [K, N]
    opB = N → A_stored   = [M, K]  但维度不匹配！
```

**问题**：`op(A) × op(B) = [K, N] × [M, K]` 维度不对！

**正确方案**：交换输入顺序

```
设:
    第一个矩阵参数 = W_stored [N, K] 行主序，传给 cuBLAS 时告诉它是 [K, N] 列主序
    第二个矩阵参数 = A_stored [M, K] 行主序，传给 cuBLAS 时告诉它是 [K, M] 列主序
    opA = N → [K, N]
    opB = T → [K, M]^T = [M, K]  维度还是不对！
```

**最终正确配置（N/T + C/C/C）**：

```
目标: D[M, N] = A[M, K] × W[K, N]

cuBLASLt 配置（利用行主序 = 列主序转置的特性）:
    实际传入:
        A_ptr: 指向 W_stored 的内存（行主序 [N, K]）
        B_ptr: 指向 A_stored 的内存（行主序 [M, K]）
        C_ptr/D_ptr: 输出内存
    
    告诉 cuBLASLt（列主序视角）:
        A: [K, N] 列主序（实际是行主序 [N, K] 的另一种解读）
        B: [K, M] 列主序（实际是行主序 [M, K] 的另一种解读）
        opA = N → A 不变，[K, N]
        opB = T → B 转置，[K, M]^T = [M, K]
        
    计算: D = op(A) × op(B) = [K, N] × [M, K]  维度还是不对！
```

**我明白了！你需要的是 D^T 的计算**：

```
目标: D[M, N] = A[M, K] × W[K, N]
等价: D^T[N, M] = W^T[N, K] × A^T[K, M]

cuBLASLt 配置:
    传入:
        A_ptr → W_stored [N, K] 行主序 → cuBLAS 列主序读为 [K, N]
        B_ptr → A_stored [M, K] 行主序 → cuBLAS 列主序读为 [K, M]
        
    opA = T → [K, N]^T = [N, K]
    opB = N → [K, M]
    
    计算: D' = op(A) × op(B) = [N, K] × [K, M] = [N, M] ✅
    
    输出:
        D_ptr → 列主序写 [N, M] → 行主序读为 [M, N] ✅
```

**最终答案**：

```cpp
// cuBLASLt T/N + Col/Col/Col 配置
cublasOperation_t opA = CUBLAS_OP_T;   // 对 W（第一个参数）转置
cublasOperation_t opB = CUBLAS_OP_N;   // 对 A（第二个参数）不转置

// 矩阵布局
// W: 行主序 [N, K]，传给 cuBLASLt 声明为列主序 [K, N]，lda = K
// A: 行主序 [M, K]，传给 cuBLASLt 声明为列主序 [K, M]，ldb = K
// D: 列主序写出 [N, M]，等于行主序 [M, N]，ldc = N

// 计算: D' = W^T × A = [K, N]^T × [K, M] = [N, K] × [K, M] = [N, M]
// 输出: 列主序 [N, M] = 行主序 [M, N] ✅
```

---

### 11.4 Q3: Scale 维度与反量化机制

#### 11.4.1 实际的 Scale 维度（从 checkpoint 确认）

```
scale_a (input_scale):  per-token dynamic → [M, 1] FP32
scale_b (weight_scale): per-channel       → [N, 1] FP32 (不是 [1, K]!)
```

**你之前的猜测需要修正**：weight_scale 是 `[N, 1]` 不是 `[1, K]`。

#### 11.4.2 CUTLASS 的反量化公式

从 `scaled_mm_epilogues_c3x.hpp` 分析：

```cpp
// ScaledEpilogue 的计算
using ScaleA = ColOrScalarLoad<float>;  // 列方向加载 → [M, 1] 广播到 [M, N]
using ScaleB = RowOrScalarLoad<float>;  // 行方向加载 → [N, 1].T = [1, N] 广播到 [M, N]

// EVTCompute0: tmp = ScaleB × Accum
// EVTCompute1: D = ScaleA × tmp

// 展开: D = ScaleA × (ScaleB × Accum)
//         = ScaleA[M,1] ⊗ (ScaleB[1,N] ⊗ Accum[M,N])
//         = (ScaleA[M,1] ⊗ ScaleB[1,N]) ⊗ Accum[M,N]  (广播逐元素乘)
```

**CUTLASS 反量化公式**：

```
D[M,N] = scale_a[M,1] ⊙ scale_b[1,N] ⊙ (qA[M,K] × qW[K,N])

其中:
    qA: 量化后的激活 [M, K] FP8
    qW: 量化后的权重 [K, N] FP8  
    scale_a: [M, 1] 广播到 [M, N]
    scale_b: [N, 1]^T = [1, N] 广播到 [M, N]
    ⊙: 广播逐元素乘法
```

#### 11.4.3 cuBLASLt 的 Outer Vector Scaling（SM90+）

从官方文档 3.1.4.3 节：

```
Outer Vector Scaling for FP8 Data Types:

D_ij = α × scale_A^i × scale_B^j × Σ(a_il × b_lj) + β × scale_C × C_ij

其中:
    scale_A: 长度为 M 的向量，每行一个 scale
    scale_B: 长度为 N 的向量，每列一个 scale
```

**启用方法**：

```cpp
// 设置 outer vector scaling mode
int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));

// 设置 scale 指针
float* scaleA = ...;  // 长度 M
float* scaleB = ...;  // 长度 N
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA, sizeof(scaleA));
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB, sizeof(scaleB));
```

#### 11.4.4 Scale 适配问题

**CUTLASS 的 scale 语义**（A在左）：
- scale_a: 对应 A (input)，维度 [M, 1]
- scale_b: 对应 B (weight)，维度 [N, 1]

**cuBLASLt 的 scale 语义（W在左，A在右）**：

由于我们交换了 A 和 B 的位置（W 作为 cuBLASLt 的第一个参数），需要相应调整：

```
cuBLASLt 计算: D'[N, M] = W'[N, K] × A'[K, M]

其中:
    W' = W^T (通过 opA=T 实现)
    A' = A^T (通过列主序读行主序实现)

scale 对应:
    cuBLASLt 的 scale_A → 对应 W → 维度 [N] (因为 op(W) 的行数是 N)
    cuBLASLt 的 scale_B → 对应 A → 维度 [M] (因为 op(A) 的列数是 M)
```

**关键适配**：需要交换传入 cuBLASLt 的 scale：

```python
# vLLM 传来的:
#   scale_a: [M, 1] → 对应 input
#   scale_b: [N, 1] → 对应 weight

# 传给 cuBLASLt (W在左):
#   cublaslt_scale_A → scale_b.squeeze() → [N]  (weight scale)
#   cublaslt_scale_B → scale_a.squeeze() → [M]  (input scale)
```

---

### 11.5 Q4: Bias 广播方向

#### 11.5.1 Bias 的存储格式

从 checkpoint 确认：
```
bias: [N] 1D 向量（N = out_features）
```

#### 11.5.2 CUTLASS 的 Bias 处理

从 `scaled_mm_epilogues_c3x.hpp`：

```cpp
// ScaledEpilogueBias 中
using Bias = RowLoad<ElementD>;  // 行方向加载

// 计算公式:
// D = ScaleA × (ScaleB × Accum) + Bias
// 其中 Bias 是 RowLoad，广播到每一行
```

**Bias 广播方向**：`[1, N]` 广播到 `[M, N]`，即**沿 N 维度（列方向）广播**。

每一行（每个 token）加上相同的 bias 向量。

#### 11.5.3 cuBLASLt 的 Bias 处理

```cpp
// 设置 epilogue 为 BIAS
cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));

// 设置 bias 指针和类型
void* biasPtr = ...;  // [N] 向量
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_BIAS_POINTER, &biasPtr, sizeof(biasPtr));

cudaDataType_t biasType = CUDA_R_16BF;  // BF16
cublasLtMatmulDescSetAttribute(matmulDesc, 
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
```

**注意**：由于我们计算的是 `D'[N, M]` 然后存储为行主序 `[M, N]`，bias 的广播方向也需要考虑。

实际上，cuBLASLt 的 bias 是加在输出矩阵的**列方向**（因为它用列主序）。
- 输出 `D'[N, M]` 列主序
- bias `[N]` 加到每一列
- 等价于行主序 `D[M, N]` 中，bias `[N]` 加到每一行 ✅

---

### 11.6 Q5: cuBLASLtMatmul API 调用要点

#### 11.6.1 完整的 API 调用框架

```cpp
#include <cublasLt.h>

cublasStatus_t cublaslt_fp8_gemm_impl(
    cublasLtHandle_t handle,
    int M, int N, int K,
    const void* W_ptr,        // 行主序 [N, K] FP8
    const void* A_ptr,        // 行主序 [M, K] FP8
    void* D_ptr,              // 行主序 [M, N] BF16
    const float* scale_w,     // [N] weight scale
    const float* scale_a,     // [M] input scale
    const void* bias,         // [N] bias (可选)
    cudaDataType_t biasType,
    cudaStream_t stream
) {
    // 1. 创建 matmul 描述符
    cublasLtMatmulDesc_t matmulDesc;
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cublasLtMatmulDescCreate(&matmulDesc, computeType, CUDA_R_32F);
    
    // 2. 设置转置操作
    cublasOperation_t opA = CUBLAS_OP_T;   // W 转置
    cublasOperation_t opB = CUBLAS_OP_N;   // A 不转置
    cublasLtMatmulDescSetAttribute(matmulDesc, 
        CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
    cublasLtMatmulDescSetAttribute(matmulDesc, 
        CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));
    
    // 3. 设置 outer vector scaling (SM90+)
    int8_t fastAccuMode = 1;
    cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccuMode, sizeof(fastAccuMode));
    
    // Scale 模式和指针
    int32_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    cublasLtMatmulDescSetAttribute(matmulDesc, 
        CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    cublasLtMatmulDescSetAttribute(matmulDesc, 
        CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    cublasLtMatmulDescSetAttribute(matmulDesc, 
        CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_w, sizeof(scale_w));
    cublasLtMatmulDescSetAttribute(matmulDesc, 
        CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_a, sizeof(scale_a));
    
    // 4. 设置 Bias (如果有)
    if (bias != nullptr) {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(matmulDesc, 
            CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue));
        cublasLtMatmulDescSetAttribute(matmulDesc, 
            CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias));
        cublasLtMatmulDescSetAttribute(matmulDesc, 
            CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &biasType, sizeof(biasType));
    }
    
    // 5. 创建矩阵布局
    // W: 行主序 [N, K] → 声明为列主序 [K, N]，lda = K
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, N, K);
    
    // A: 行主序 [M, K] → 声明为列主序 [K, M]，ldb = K
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K);
    
    // D: 列主序 [N, M]，ldc = N → 读为行主序 [M, N]
    cublasLtMatrixLayout_t Ddesc;
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, N, M, N);
    
    // 6. 获取最优算法
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    
    size_t workspaceSize = 64 * 1024 * 1024;  // 64 MB
    cublasLtMatmulPreferenceSetAttribute(preference, 
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, 
        Adesc, Bdesc, Ddesc, Ddesc,
        preference, 1, &heuristicResult, &returnedResults);
    
    // 7. 分配 workspace
    void* workspace = nullptr;
    cudaMalloc(&workspace, heuristicResult.workspaceSize);
    
    // 8. 执行 GEMM
    float alpha = 1.0f, beta = 0.0f;
    cublasLtMatmul(
        handle, matmulDesc,
        &alpha,
        W_ptr, Adesc,   // 第一个矩阵: W
        A_ptr, Bdesc,   // 第二个矩阵: A
        &beta,
        D_ptr, Ddesc,   // C (unused, beta=0)
        D_ptr, Ddesc,   // D (output)
        &heuristicResult.algo,
        workspace, heuristicResult.workspaceSize,
        stream
    );
    
    // 9. 清理
    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(matmulDesc);
    
    return CUBLAS_STATUS_SUCCESS;
}
```

#### 11.6.2 关键注意事项

| 要点 | 说明 |
|------|------|
| **矩阵参数顺序** | 第一个是 W（weight），第二个是 A（activation） |
| **op 配置** | opA=T（转置 W），opB=N（A 不转置） |
| **Layout 声明** | 行主序数据声明为列主序，维度交换 |
| **Scale 交换** | cuBLASLt 的 scale_A → weight_scale，scale_B → input_scale |
| **Bias 类型** | 需要与输出类型匹配或兼容 |
| **Workspace** | 预分配足够空间，推荐 64MB |
| **Handle 复用** | 全局缓存 handle，避免重复创建 |
| **Algorithm 缓存** | 相同问题规模可复用启发式结果 |

#### 11.6.3 Python 侧适配

在 `cublaslt_w8a8_scaled_mm` 中：

```python
def cublaslt_w8a8_scaled_mm(
    *,
    qinput: torch.Tensor,     # [M, K] FP8 行主序
    weight: torch.Tensor,     # [K, N] FP8 "列主序"（实际是 .t() 后的 view）
    out_dtype: torch.dtype,
    scale_a: torch.Tensor,    # [M, 1] input scale
    scale_b: torch.Tensor,    # [N, 1] weight scale
    bias: torch.Tensor,       # [N] 或 None
    output_shape: list,
    **kwargs,
) -> torch.Tensor:
    """
    cuBLASLt FP8 Scaled MM
    
    关键理解：
    - vLLM 传来的 weight 是 [K,N] 但 stride=(1,K)，是 .t() 后的 view
    - 物理内存实际是 [N,K] 行主序存储
    - 我们需要再 .t() 消除这个假转置，让 stride 和物理内存一致
    """
    M, K = qinput.shape
    N = weight.shape[1]  # weight 当前 shape 是 [K, N]
    
    # 关键：消除 .t() 造成的 stride 不一致
    # weight.t() 将 [K,N] stride=(1,K) 变回 [N,K] stride=(K,1)
    # 这样 stride 就和物理内存布局（行主序 [N,K]）一致了
    weight_row_major = weight.t()  # [N, K] 行主序，无需 contiguous（本身就是连续的）
    
    # 调用 cuBLASLt (注意 scale 顺序交换)
    output = ops.cublaslt_scaled_mm(
        W=weight_row_major,        # [N, K] 行主序
        A=qinput,                  # [M, K] 行主序
        scale_W=scale_b.squeeze(), # [N] weight scale
        scale_A=scale_a.squeeze(), # [M] input scale
        bias=bias,                 # [N] bias
        out_dtype=out_dtype,
    )
    
    return output.view(*output_shape)
```

#### 11.6.4 关于 Bias 广播方向的澄清

**cuBLASLt 计算流程**：

```
1. 输入:
   W: [N,K] 行主序 → cuBLASLt 列主序读为 [K,N]
   A: [M,K] 行主序 → cuBLASLt 列主序读为 [K,M]

2. 计算:
   opA=T: [K,N]^T = [N,K]
   opB=N: [K,M]
   D' = [N,K] × [K,M] = [N,M]  (列主序结果)

3. Bias 广播:
   bias: [N] 向量
   在列主序 [N,M] 中，bias 加到"每一列"（列主序视角）
   即: D'[i,j] += bias[i], 对所有 i∈[0,N), j∈[0,M)
   
4. 输出:
   列主序 [N,M] 写入内存
   按行主序解读 = [M,N] ✅
   
   从行主序 [M,N] 视角看:
   D[j,i] = D'[i,j] 包含了 bias[i]
   即每一行 j 的第 i 列都加了 bias[i]
   这就是"bias 沿 N 维度广播"的正确行为 ✅
```

**总结**：
- bias `[N]` 在 cuBLASLt 中会自动正确广播
- 无需额外处理，直接传入即可

---

*文档版本：v1.5*  
*更新日期：2025-01*  
*作者：SlideSparse Team*
