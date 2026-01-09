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

*文档版本：v1.2*  
*更新日期：2025-01*  
*作者：SlideSparse Team*
