"""
vLLM utils.py 补丁示例

这个文件展示了如何修改 vllm/model_executor/layers/utils.py 来劫持 GEMM 函数。

实际使用时，你需要将这些修改直接应用到 vllm/model_executor/layers/utils.py 文件中。

=== 方法 1: 直接修改 default_unquantized_gemm ===

找到 vllm/model_executor/layers/utils.py 中的：

```python
def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    return torch.nn.functional.linear(x, weight, bias)
```

替换为：

```python
import os

# 自定义 Kernel 开关
_USE_CUSTOM_GEMM = os.environ.get("VLLM_USE_CUSTOM_GEMM", "0") == "1"
_custom_gemm_fn = None

def _get_custom_gemm():
    global _custom_gemm_fn
    if _custom_gemm_fn is None:
        try:
            from custom_kernels import custom_gemm
            _custom_gemm_fn = custom_gemm
            logger.info("✅ Custom GEMM kernel loaded")
        except ImportError as e:
            logger.warning(f"Custom GEMM not available: {e}")
            _custom_gemm_fn = torch.nn.functional.linear
    return _custom_gemm_fn

def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    if _USE_CUSTOM_GEMM:
        return _get_custom_gemm()(x, weight, bias)
    return torch.nn.functional.linear(x, weight, bias)
```

=== 方法 2: 修改 dispatch_unquantized_gemm ===

这是更干净的方式，添加一个新的分支：

```python
def custom_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from custom_kernels import custom_gemm
    return custom_gemm(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    # 优先检查自定义 GEMM
    if os.environ.get("VLLM_USE_CUSTOM_GEMM", "0") == "1":
        return custom_unquantized_gemm
    elif current_platform.is_rocm():
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm
```

=== 使用方法 ===

1. 修改 vllm/model_executor/layers/utils.py
2. 确保 custom_kernels 在 Python 路径中
3. 运行时设置环境变量:
   
   VLLM_USE_CUSTOM_GEMM=1 vllm bench throughput --model Qwen/Qwen2.5-0.5B

"""

# 这是完整的修改后的 utils.py 末尾部分的示例代码：

import os
from collections.abc import Callable
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# ============================================================================
# Custom GEMM 支持
# ============================================================================

_USE_CUSTOM_GEMM = os.environ.get("VLLM_USE_CUSTOM_GEMM", "0") == "1"
_custom_gemm_module = None

def _load_custom_gemm_module():
    """延迟加载自定义 GEMM 模块"""
    global _custom_gemm_module
    if _custom_gemm_module is None:
        try:
            # 尝试从 custom_kernels 导入
            import custom_kernels
            _custom_gemm_module = custom_kernels
            logger.info(f"✅ Custom kernels loaded: {custom_kernels.get_kernel_status()}")
        except ImportError as e:
            logger.warning(f"⚠️ Custom kernels not available: {e}")
            _custom_gemm_module = False  # 标记为不可用
    return _custom_gemm_module


def custom_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """使用自定义 GEMM kernel"""
    module = _load_custom_gemm_module()
    if module and module is not False:
        return module.custom_gemm(x, weight, bias)
    # Fallback
    return torch.nn.functional.linear(x, weight, bias)


def dispatch_unquantized_gemm_custom() -> Callable[..., torch.Tensor]:
    """
    修改后的 GEMM 派发函数
    
    调用链:
    LinearBase.forward() -> quant_method.apply() -> dispatch_unquantized_gemm()() 
    """
    # 1. 优先检查自定义 GEMM
    if _USE_CUSTOM_GEMM:
        logger.info("Using custom GEMM kernel")
        return custom_unquantized_gemm
    
    # 2. 原有逻辑
    from vllm.platforms import current_platform
    if current_platform.is_rocm():
        from vllm.model_executor.layers.utils import rocm_unquantized_gemm
        return rocm_unquantized_gemm
    elif current_platform.is_cpu():
        from vllm.model_executor.layers.utils import cpu_unquantized_gemm
        return cpu_unquantized_gemm
    else:
        return default_unquantized_gemm_original


def default_unquantized_gemm_original(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
):
    """原始的默认 GEMM 实现"""
    return torch.nn.functional.linear(x, weight, bias)


if __name__ == "__main__":
    print("This is a patch example for vllm/model_executor/layers/utils.py")
    print(f"USE_CUSTOM_GEMM: {_USE_CUSTOM_GEMM}")
