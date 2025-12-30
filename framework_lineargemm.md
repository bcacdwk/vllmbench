# vLLM 线性层与 GEMM 调用链详解 (Framework Linear & GEMM)

本文档专门分析 vLLM 中线性层的实现，包括四个关键投影层（Wqkv、Wo、W13、W2）的位置，以及量化、反量化和 GEMM 乘法的具体调用位置。

---

## 1. 背景：你要做的事情

你计划替换 vLLM 推理过程中的以下三个步骤的算子：
1. **Quant（量化）**: 将 FP16/BF16 激活值量化为 FP8
2. **GEMM（矩阵乘法）**: 执行量化后的矩阵乘法
3. **Dequant（反量化）**: 将结果反量化回 FP16/BF16

需要找到这些操作发生的位置，并替换为自定义的 Triton/CUDA Kernel。

---

## 2. 四个关键线性层的位置

### 2.1 线性层定义总览

| 线性层 | 对应模型层 | 类名 | 文件位置 |
|-------|----------|------|---------|
| **Wqkv** | `self_attn.qkv_proj` | `QKVParallelLinear` | `vllm/model_executor/layers/linear.py` |
| **Wo** | `self_attn.o_proj` | `RowParallelLinear` | `vllm/model_executor/layers/linear.py` |
| **W13** | `mlp.gate_up_proj` | `MergedColumnParallelLinear` | `vllm/model_executor/layers/linear.py` |
| **W2** | `mlp.down_proj` | `RowParallelLinear` | `vllm/model_executor/layers/linear.py` |

### 2.2 在 Qwen2 模型中的使用位置

**文件**: `vllm/model_executor/models/qwen2.py`

```python
# Attention 层中
class Qwen2Attention(nn.Module):
    def __init__(self, ...):
        # Wqkv - QKV 合并投影
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Wo - 输出投影
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

# MLP 层中
class Qwen2MLP(nn.Module):
    def __init__(self, ...):
        # W13 - Gate 和 Up 合并投影
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        # W2 - Down 投影
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
```

### 2.3 在 Llama 模型中的使用位置

**文件**: `vllm/model_executor/models/llama.py`

结构与 Qwen2 完全相同，只是类名不同：
- `LlamaAttention` 使用 `QKVParallelLinear` (Wqkv) 和 `RowParallelLinear` (Wo)
- `LlamaMLP` 使用 `MergedColumnParallelLinear` (W13) 和 `RowParallelLinear` (W2)

---

## 3. 线性层类的详细定义

**文件位置**: `vllm/model_executor/layers/linear.py`

### 3.1 类继承结构

```
LinearBase (CustomOp)
    │
    ├── ReplicatedLinear              # 行 867-410
    │
    ├── ColumnParallelLinear          # 行 413-583
    │   ├── MergedColumnParallelLinear # 行 586-864 (W13)
    │   └── QKVParallelLinear          # 行 867-1239 (Wqkv)
    │
    └── RowParallelLinear             # 行 1241-1425 (Wo, W2)
```

### 3.2 核心代码：forward 方法中的 GEMM 调用

所有线性层的 forward 方法最终都调用 `self.quant_method.apply()`：

```python
# ColumnParallelLinear.forward() (行 557-575)
def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None
    
    # ⭐ 核心 GEMM 调用 - 通过量化方法
    assert self.quant_method is not None
    output_parallel = self.quant_method.apply(self, input_, bias)
    
    if self.gather_output and self.tp_size > 1:
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    return output, output_bias

# RowParallelLinear.forward() (行 1388-1416)
def forward(self, input_):
    # ... 输入处理 ...
    
    # ⭐ 核心 GEMM 调用 - 通过量化方法
    assert self.quant_method is not None
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    
    if self.reduce_results and self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    return output, output_bias
```

---

## 4. 量化方法与 GEMM 实现

### 4.1 量化方法类型

根据 `quant_config` 的不同，`quant_method` 可能是以下类型：

| quant_method 类 | 量化类型 | 文件位置 |
|----------------|---------|---------|
| `UnquantizedLinearMethod` | 无量化 | `vllm/model_executor/layers/linear.py` |
| `Fp8LinearMethod` | FP8 量化 | `vllm/model_executor/layers/quantization/fp8.py` |
| `AWQLinearMethod` | AWQ 量化 | `vllm/model_executor/layers/quantization/awq.py` |
| `GPTQLinearMethod` | GPTQ 量化 | `vllm/model_executor/layers/quantization/gptq.py` |
| `MarlinLinearMethod` | Marlin 量化 | `vllm/model_executor/layers/quantization/gptq_marlin.py` |

### 4.2 无量化情况 (UnquantizedLinearMethod)

**文件**: `vllm/model_executor/layers/linear.py` (行 196-240)

```python
class UnquantizedLinearMethod(LinearMethodBase):
    """Linear method without quantization."""

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ⭐ GEMM 调用 - 最终调用 torch.nn.functional.linear
        return dispatch_unquantized_gemm()(layer, x, layer.weight, bias)
```

`dispatch_unquantized_gemm()` 根据平台选择不同实现：

**文件**: `vllm/model_executor/layers/utils.py` (行 245-251)

```python
def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    if current_platform.is_rocm():
        return rocm_unquantized_gemm      # ROCm 优化
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm       # CPU 优化
    else:
        return default_unquantized_gemm   # 默认 CUDA

def default_unquantized_gemm(layer, x, weight, bias=None):
    # ⭐ 最终的 GEMM 调用
    return torch.nn.functional.linear(x, weight, bias)
```

---

## 5. FP8 量化详解 ⭐⭐⭐

这是最重要的部分，详细介绍 FP8 量化的 quant-gemm-dequant 流程。

### 5.1 FP8 量化配置

**文件**: `vllm/model_executor/layers/quantization/fp8.py`

```python
class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,  # 权重是否已量化
        activation_scheme: str = "dynamic",          # 激活量化方案
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,  # 块量化大小
    ):
        ...
```

### 5.2 Fp8LinearMethod 类

**文件**: `vllm/model_executor/layers/quantization/fp8.py` (行 366-688)

```python
class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8."""

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.weight_block_size is not None
        
        # 根据配置选择不同的 GEMM 实现
        if self.block_quant:
            self.w8a8_block_fp8_linear = W8A8BlockFp8LinearOp(...)
        else:
            self.fp8_linear = Fp8LinearOp(...)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ⭐ FP8 GEMM 的核心入口
        
        if self.use_marlin:
            # Marlin FP8 kernel
            return apply_fp8_marlin_linear(...)
        
        if self.block_quant:
            # 块量化 FP8 GEMM
            return self.w8a8_block_fp8_linear.apply(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale_inv,
                input_scale=layer.input_scale,
                bias=bias,
            )
        
        # Per-tensor FP8 GEMM
        return self.fp8_linear.apply(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias,
        )
```

### 5.3 Fp8LinearOp 类（Per-tensor 量化）

**文件**: `vllm/model_executor/layers/quantization/utils/w8a8_utils.py`

```python
class Fp8LinearOp:
    """Per-tensor FP8 linear operation."""

    def __init__(
        self,
        act_quant_static: bool,
        act_quant_group_shape: GroupShape,
    ):
        self.act_quant_static = act_quant_static
        self.act_quant_group_shape = act_quant_group_shape
        self.quant_fp8 = QuantFP8()  # 激活量化器

    def apply(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: torch.dtype,
        input_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ⭐ 1. 激活量化 (Quant)
        if self.act_quant_static:
            # 静态量化：使用预计算的 scale
            qinput, scale_a = ops.scaled_fp8_quant(input, input_scale)
        else:
            # 动态量化
            qinput, scale_a = self.quant_fp8.apply(input)

        # ⭐ 2. GEMM 乘法
        output = self.gemm_op(
            qinput=qinput,
            weight=weight,
            out_dtype=out_dtype,
            scale_a=scale_a,
            scale_b=weight_scale,
            bias=bias,
            output_shape=[...],
        )
        
        return output
```

### 5.4 量化函数 (scaled_fp8_quant)

**文件**: `vllm/_custom_ops.py` (行 1678-1735)

```python
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.
    """
    # ...
    if scale is None:
        if use_per_token_if_dynamic:
            # ⭐ 动态 per-token 量化
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
        else:
            # ⭐ 动态 per-tensor 量化
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        # ⭐ 静态量化
        torch.ops._C.static_scaled_fp8_quant(output, input, scale)

    return output, scale
```

### 5.5 CUTLASS GEMM 函数

**文件**: `vllm/_custom_ops.py` (行 828-876)

```python
def cutlass_scaled_mm(
    a: torch.Tensor,           # 量化后的激活 (FP8)
    b: torch.Tensor,           # 量化后的权重 (FP8)
    scale_a: torch.Tensor,     # 激活 scale
    scale_b: torch.Tensor,     # 权重 scale
    out_dtype: torch.dtype,    # 输出数据类型
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused GEMM with dequantization:
    output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)
    """
    # ... 维度处理 ...
    
    if current_platform.is_rocm() or not cutlass_compatible_b:
        # Triton fallback
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (
            triton_scaled_mm,
        )
        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        # ⭐ CUTLASS kernel 调用
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)
```

---

## 6. 完整调用链图

### 6.1 FP8 量化的完整调用链

```
Model Forward
    │
    ▼
QKVParallelLinear/MergedColumnParallelLinear/RowParallelLinear.forward()
    │   文件: vllm/model_executor/layers/linear.py
    │
    ▼
self.quant_method.apply(self, input_, bias)
    │   文件: vllm/model_executor/layers/linear.py (行 565, 1405)
    │
    ▼
Fp8LinearMethod.apply(layer, x, bias)
    │   文件: vllm/model_executor/layers/quantization/fp8.py (行 610-687)
    │
    ├─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    ▼                                                                 ▼
[Block Quantization Path]                              [Per-tensor Quantization Path]
    │                                                                 │
    ▼                                                                 ▼
W8A8BlockFp8LinearOp.apply()                          Fp8LinearOp.apply()
    │  文件: vllm/model_executor/layers/                │  文件: vllm/model_executor/layers/
    │        quantization/utils/fp8_utils.py           │        quantization/utils/w8a8_utils.py
    │                                                                 │
    ▼                                                                 ▼
┌─────────────────────────────────┐              ┌─────────────────────────────────┐
│  1. 激活量化 (Quant)              │              │  1. 激活量化 (Quant)              │
│     quant_fp8.apply(input)       │              │     ops.scaled_fp8_quant(input)  │
│     或                           │              │                                 │
│     ops.scaled_fp8_quant(...)    │              │                                 │
├─────────────────────────────────┤              ├─────────────────────────────────┤
│  2. GEMM (含反量化)               │              │  2. GEMM (含反量化)               │
│     cutlass/triton kernel        │              │     cutlass_scaled_mm(...)      │
│                                 │              │     或                           │
│     深度调用:                     │              │     triton_scaled_mm(...)       │
│     - ops.cutlass_scaled_mm()   │              │                                 │
│     - triton kernel             │              │                                 │
│     - DeepGEMM                  │              │                                 │
└─────────────────────────────────┘              └─────────────────────────────────┘
```

### 6.2 关键函数和文件对照表

| 步骤 | 函数名 | 文件路径 | 行号（约） |
|-----|-------|---------|-----------|
| **入口** | `ColumnParallelLinear.forward()` | `vllm/model_executor/layers/linear.py` | 557-575 |
| **入口** | `RowParallelLinear.forward()` | `vllm/model_executor/layers/linear.py` | 1388-1416 |
| **量化方法** | `Fp8LinearMethod.apply()` | `vllm/model_executor/layers/quantization/fp8.py` | 610-687 |
| **量化** | `scaled_fp8_quant()` | `vllm/_custom_ops.py` | 1678-1735 |
| **量化** | `dynamic_scaled_fp8_quant` | `torch.ops._C` (CUDA kernel) | - |
| **量化** | `static_scaled_fp8_quant` | `torch.ops._C` (CUDA kernel) | - |
| **GEMM** | `cutlass_scaled_mm()` | `vllm/_custom_ops.py` | 828-876 |
| **GEMM** | `torch.ops._C.cutlass_scaled_mm` | CUDA kernel (csrc) | - |
| **GEMM** | `triton_scaled_mm()` | `vllm/.../triton_scaled_mm.py` | - |
| **块量化GEMM** | `W8A8BlockFp8LinearOp.apply()` | `vllm/.../fp8_utils.py` | - |

---

## 7. 如何替换自定义 Kernel

### 7.1 方法一：替换量化方法类

创建自定义的 `LinearMethod` 类：

```python
# my_custom_linear_method.py
from vllm.model_executor.layers.linear import LinearMethodBase

class MyCustomFp8LinearMethod(LinearMethodBase):
    """Custom FP8 linear method with custom kernels."""
    
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. 自定义量化
        qinput, scale_a = my_custom_quant(x)
        
        # 2. 自定义 GEMM
        output = my_custom_gemm(qinput, layer.weight, scale_a, layer.weight_scale)
        
        # 3. 添加 bias
        if bias is not None:
            output = output + bias
        
        return output
```

然后在量化配置中注册：

```python
# 修改 vllm/model_executor/layers/quantization/fp8.py
class Fp8Config(QuantizationConfig):
    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            # 返回自定义方法
            return MyCustomFp8LinearMethod(self)
        ...
```

### 7.2 方法二：直接替换底层函数

在 `vllm/_custom_ops.py` 中直接替换函数：

```python
# 替换 scaled_fp8_quant
def scaled_fp8_quant(input, scale=None, ...):
    # 调用自定义 kernel
    return my_custom_fp8_quant(input, scale, ...)

# 替换 cutlass_scaled_mm
def cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None):
    # 调用自定义 GEMM kernel
    return my_custom_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
```

### 7.3 方法三：使用 Monkey Patching

```python
import vllm._custom_ops as ops

# 保存原始函数
original_scaled_fp8_quant = ops.scaled_fp8_quant
original_cutlass_scaled_mm = ops.cutlass_scaled_mm

# 替换为自定义实现
ops.scaled_fp8_quant = my_custom_quant
ops.cutlass_scaled_mm = my_custom_gemm
```

---

## 8. 关键 Import 和依赖

### 8.1 线性层相关 import

```python
# vllm/model_executor/layers/linear.py
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
```

### 8.2 FP8 量化相关 import

```python
# vllm/model_executor/layers/quantization/fp8.py
from vllm import _custom_ops as ops                    # 自定义算子
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp,
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    W8A8BlockFp8LinearOp,
)
```

### 8.3 底层 C++ 绑定

```python
# vllm/_custom_ops.py
import torch
# CUDA kernels are loaded via torch.ops._C
torch.ops._C.cutlass_scaled_mm(...)
torch.ops._C.dynamic_scaled_fp8_quant(...)
torch.ops._C.static_scaled_fp8_quant(...)
```

---

## 9. 总结：替换 GEMM 的检查清单

要替换 vLLM 中的 FP8 quant-gemm-dequant，需要修改以下位置：

### 9.1 量化函数
- **文件**: `vllm/_custom_ops.py`
- **函数**: `scaled_fp8_quant()` (行 1678)
- **底层**: `torch.ops._C.dynamic_scaled_fp8_quant` / `static_scaled_fp8_quant`

### 9.2 GEMM 函数
- **文件**: `vllm/_custom_ops.py`
- **函数**: `cutlass_scaled_mm()` (行 828)
- **底层**: `torch.ops._C.cutlass_scaled_mm`
- **Triton备选**: `vllm/.../triton_scaled_mm.py`

### 9.3 量化方法类
- **文件**: `vllm/model_executor/layers/quantization/fp8.py`
- **类**: `Fp8LinearMethod.apply()` (行 610)
- **工具类**: `Fp8LinearOp` in `w8a8_utils.py`
- **工具类**: `W8A8BlockFp8LinearOp` in `fp8_utils.py`

### 9.4 模型层调用
- **Attention**: `vllm/model_executor/models/qwen2.py` - `Qwen2Attention`
- **MLP**: `vllm/model_executor/models/qwen2.py` - `Qwen2MLP`
- **线性层**: `vllm/model_executor/layers/linear.py` - `forward()` 方法

通过理解这个完整的调用链，你可以选择在适当的层级插入自定义 kernel，实现对 quant-gemm-dequant 流程的替换。
