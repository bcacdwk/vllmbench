# SlideSparse 技术深度复盘：驯服 Dynamo 与 CUDA Graph 的冒险

**作者**：expert A
**日期**：2026-01-24  
**目标硬件**：NVIDIA Blackwell (SM120)  

---

## 1. 核心问题：Compiler Stack 的崩塌

项目的起因并非简单的“不支持”，而是深层次的**编译栈失效**。

在 vLLM 现有的架构中，模型推理依赖 `torch.compile` (Dynamo + Inductor) 将 PyTorch 算子编译为高效的 Triton Kernel。当我们将这一流程度部署到 SM120 架构时，遇到了一系列环环相扣的系统性问题：

1.  **Dynamo 捕获失败**：面对全新的硬件指令，PyTorch 的 Inductor 编译器生成的代码与底层硬件特性不匹配，导致 Graph Capture 阶段频繁崩溃。
2.  **算子缺失**：Dynamo 无法自动生成适配 SM120 Tensor Core 的 INT8 GEMM Kernel。
3.  **不得已的逃生舱**：为了绕过编译器的局限性，我们不得不引入 **Custom Kernel** (SlideSparse) 手动接管核心计算。

**真正的挑战由此开始**：引入自定义 Kernel 意味着打破 PyTorch 的闭环生态 —— 我们必须手动解决 Custom Kernel 与 **Dynamo (图捕获)** 和 **CUDA Graph (图执行)** 的兼容性难题。

---

## 2. 解决方案：重构 Wrapper 与 Kernel 的交互协议

为了让 SlideSparse 的 Custom Kernel 能被 System Stack 接纳，我们对 `gemm_wrapper.py` (Python层) 和 `cuda_kernel` (C++层) 进行了大幅度改造。

### 2.1 Python 层改造：欺骗 Dynamo

`torch.compile` 无法追踪黑盒的 `ctypes` 调用。为了让 Dynamo 能够理解并“串联”我们的 Custom Kernel，我们在 `slidesparse/core/gemm_wrapper.py` 中实施了如下策略：

*   **引入 `torch.library` 机制**：
    我们不再直接调用 CUDA 函数，而是将其注册为 PyTorch 的 Custom Op。
    ```python
    # 示意伪代码
    @torch.library.impl(lib, "slidesparse_gemm_fp8", "Meta")
    def _gemm_fp8_meta(input, weight, ...):
        # Fake Implementation
        # 告诉 Dynamo：虽然你看不懂 Kernel 内部，但它会输出这样一个形状的 Tensor
        return torch.empty((M, N), dtype=input.dtype, device=input.device)
    ```
    *   **Meta Implementation (Fake)**：用于编译期的 Shape 推导和 Graph 构建。
    *   **Kernel Implementation (Real)**：用于运行时的真实计算。

这一修改成功欺骗了 Dynamo，使其在构建图时将我们的 Kernel 作为一个“已知节点”保留，而不是因为无法分析而导致 Graph Break。

### 2.2 CUDA Graph 兼容性：内存管理的控制权移交

最棘手的问题在于 vLLM 强依赖 **CUDA Graph**。在 CUDA Graph Capture 过程中，**绝对禁止**任何形式的动态内存分配（如 `cudaMalloc`）。

*   **问题**：通常的高性能 GEMM 库（如 cuBLASLt）都需要临时的 Workspace buffer。如果在 C++ 内部根据输入大小动态 `malloc`，Graph Capture 将立刻失败。
*   **重构方案：Workspace 外部注入**
    我们修改了 C++ 接口，删除了所有内部分配逻辑，改为强行要求调用者传入 Workspace 指针。
    
    *   **Python 端 (`gemm_wrapper.py`)**：
        ```python
        # 在 Python 端利用 PyTorch 的分配器预分配显存
        # PyTorch 的分配器是 Graph-Safe 的
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device="cuda")
        ```
    *   **C++ 端 (`cublaslt_gemm.cu`)**：
        ```cpp
        // 接受外部传入的指针，绝不自己 malloc
        void gemm_forward(..., void* workspace_ptr, size_t workspace_size) {
            // 直接利用 pre-allocated memory
        }
        ```

这一设计打破了封装（Wrapper 需要知道 Kernel 的内存需求），但它是让 Custom Kernel 能够在 CUDA Graph 下存活的唯一路径。

---

## 3. 目前的修改方案与进度

### 3.1 架构图
*   **前端**：`gemm_wrapper.py` (Custom Op 注册 + Workspace 管理)
*   **后端**：`cublaslt_gemm.cu` (无状态纯计算函数)

### 3.2 进度状态
*   ✅ **Dynamo Integration**: 完成 `torch.library` 注册，`torch.compile` 不再报错。
*   ✅ **CUDA Graph Compatibility**: 完成 Workspace 外部化改造，Graph Capture 成功。
*   ✅ **SM120 Compat**: 借助 SlideSparse 后端，成功在不可用的环境下跑通了 INT8 推理。

### 3.3 效果验证
通过解决编译和图执行问题，我们获得的不仅是“能跑”，而且是“跑得快”：
*   **Prefill**: 彻底避开了 Inductor 生成低效 Kernel 的坑，FP8 吞吐提升 **7倍**。
*   **Decode**: 无 Overhead 接入 vLLM 的 Graph Loop，小 Batch 性能提升 **2倍**。

---

## 4. 遗留问题与隐患

尽管我们打通了主流程，但在代码审计中仍发现了一处逻辑上的雷区。

### 4.1 CUDAGraph 下的 TLS (Thread Local Storage) 风险
在 `cublaslt_gemm.cu` 中：
```cpp
static thread_local char g_last_error[1024]; 
```
*   **隐患**：CUDA Graph 在 Capture 时会记录各种状态。如果要捕获的代码涉及到 TLS 变量的初始化或动态操作，行为是未定义的（Undefined Behavior）。
*   **现状**：目前因为我们运行在 Happy Path（无报错），该变量未被写入，因此侥幸过关。
*   **风险**：一旦发生错误导致该变量被写入，可能会导致 Graph Replay 时的内存非法访问或状态污染。

---

## 5. 建议与未来洞察

### 5.1 彻底的“无状态化”改造
为了长治久安，建议彻底移除 C++ 层的任何全局或线程局部状态。
*   **方案**：Kernel 函数应改为返回 `Status` 结构体，或者接受一个 `ErrorBuffer` 指针。让 C++ 代码成为纯粹的、无副作用的计算逻辑。

### 5.2 拥抱 Custom Ops 生态
目前的权宜之计（ctypes + torch.library）虽然有效，但维护成本高。
*   **洞察**：随着 PyTorch 2.0 的普及，未来的高性能计算库应该原生提供 `torch.ops` 接口，而不是让用户自己去写 ctypes wrapper。

### 5.3 自动化 Tuning 闭环
引入了外部 Kernel 后，失去的一个重要特性是 Inductor 的自动调优能力。
*   **建议**：我们需要自己实现一套轻量级的 Autotuning 机制（类似于 Triton 的 Autotuner），在模型加载阶段针对当前硬件搜索最佳的 Algo/Layout ID，并缓存到 `gemm_wrapper` 中。

---

通过这一系列的改造，我们实际上是在 vLLM 的自动化流水线旁搭建了一条“人工干预”的高速通道。虽然牺牲了部分通用性，但在新硬件早期的混乱阶段，这是榨取硬件性能的必经之路。
