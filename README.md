# SlideSparse: Fast and Flexible \(2N-2\):2N Structured Sparsity

<p align="center">
  <em>Bridging Arbitrary Sparsity Ratios to 2:4 Structured Sparsity Hardware Acceleration</em>
</p>

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Contributions](#2-key-contributions)
3. [Theoretical Foundation](#3-theoretical-foundation)
4. [Repository Structure](#4-repository-structure)
5. [Supported Configurations](#5-supported-configurations)
6. [Getting Started](#6-getting-started)
7. [Kernel-Level Benchmarking](#7-kernel-level-benchmarking)
8. [End-to-End Benchmarking](#8-end-to-end-benchmarking)
9. [Weight Conversion Pipeline](#9-weight-conversion-pipeline)
10. [vLLM Integration Architecture](#10-vllm-integration-architecture)
11. [Algorithm Search and Optimization](#11-algorithm-search-and-optimization)
12. [Experimental Results](#12-experimental-results)
13. [Reproducing Results](#13-reproducing-results)
14. [Known Limitations and Edge Cases](#14-known-limitations-and-edge-cases)
15. [Citation](#15-citation)
16. [License](#16-license)

---

## 1. Overview

**SlideSparse** is a novel algorithm that enables models with **arbitrary sparsity ratios** (e.g., 2:6, 2:8, 2:10) to leverage NVIDIA's 2:4 structured sparsity hardware acceleration introduced in the Ampere architecture. By performing an offline transformation on model weights along the hidden dimension (K in A[M,K] * W[N,K]), SlideSparse converts weights that satisfy relaxed sparsity constraints into a format compatible with the cuSPARSELt backend, thereby achieving latency reductions proportional to the underlying sparsity ratio.

In large language model (LLM) inference, **GEMM (General Matrix Multiplication) operations account for approximately 70-80% of total computation time**. Sparse computation represents one of the most promising approaches to accelerate these operations. Our baseline comparison is against dense GEMM using cuBLASLt—the most widely adopted high-performance BLAS library—which serves as the "denominator" for all our speedup calculations. The experimental control group uses the native 2:4 sparse GEMM via cuSPARSELt, which NVIDIA officially claims achieves 2× acceleration.

### Motivation

NVIDIA's Ampere and subsequent GPU architectures provide dedicated hardware support for 2:4 structured sparsity, which theoretically offers a 2× computational speedup for GEMM operations. However, this acceleration is rigidly constrained to 50% sparsity (at least 2 zeros in every 4 consecutive elements). Many practical sparse models exhibit different sparsity patterns (e.g., 33% for 2:6, 25% for 2:8), leaving significant hardware acceleration potential untapped.

SlideSparse addresses this limitation by introducing an **overlapping sliding window transformation** that maps arbitrary Z:L sparsity patterns to the 2:4 format, enabling proportional speedups for any compatible sparsity ratio. The core insight is that models with lower sparsity ratios (e.g., 25% for 2:8) can still benefit from structured sparsity hardware—achieving latency reductions that precisely match their zero element percentage.

### Core Principle

Our sparsity notation uses a **Z:L format**, where Z represents the number of zeros and L represents the window length of consecutive elements. This differs slightly from conventional sparsity definitions—we define sparsity as the ratio of zeros to total elements within the window (Z/L). Notably, the N:L remark is used in our papaer, where N denotes Non-zeros, so 4:6 or 6:8 in N:L remark is equally to 2:6 or 2:8 in this repository.

For a model with Z:L sparsity (Z zeros in every L consecutive elements):

| Original Sparsity | Zero Ratio | Expected Latency Reduction | Expected Speedup | K Expansion |
| ----------------- | ---------- | -------------------------- | ---------------- | ----------- |
| 2:4               | 50%        | 50%                        | 2.00×           | 1.00×      |
| 2:6               | 33%        | 33%                        | 1.50×           | 1.33×      |
| 2:8               | 25%        | 25%                        | 1.33×           | 1.50×      |
| 2:10              | 20%        | 20%                        | 1.25×           | 1.67×      |
| 2:∞ (Dense)      | 0%         | 0%                         | 1.00×           | 2.00×      |

The theoretical speedup directly corresponds to the sparsity ratio: X% sparsity enables X% compute skip, achieving the maximum possible benefit from that sparsity level. While the speedup ratio for lower sparsity (e.g., 1.25× for 2:10) may appear modest, **this represents the theoretical ceiling for that sparsity level**—mathematically, having X% zeros can skip at most X% of computation.

The **2:∞ (Dense) mode** serves as an important experimental validation: it demonstrates the mathematical soundness of SlideSparse by artificially inserting zeros into a fully dense weight matrix and then using sparsity to skip them. The 2× K expansion (doubling K, then halving via 2:4 compression) results in a theoretical 1× speedup (no net change), confirming that the algorithm correctly handles boundary cases where no actual zeros exist in the original weights.

---

## 2. Key Contributions

1. **Arbitrary Sparsity Adaptation**: SlideSparse breaks the rigid 50% sparsity constraint of 2:4 hardware, supporting 2:4, 2:6, 2:8, 2:10, and beyond, as well as finer-grained patterns like 1:2, 1:3, 1:4. This enables a much wider range of sparse models to benefit from hardware acceleration.
2. **Theoretically Optimal Hardware Utilization**: The algorithm guarantees that X% sparsity translates to X% latency reduction, fully exploiting the "zero-skipping" capability of the hardware. This represents the mathematical optimum—having X% zeros allows skipping exactly X% of computation, and SlideSparse achieves this ceiling.
3. **Hardware Compatibility**: SlideSparse requires **no hardware modifications** and directly leverages existing NVIDIA 2:4 sparse tensor cores via the cuSPARSELt library. It works on any GPU from Ampere (sm80) onwards, including consumer, datacenter, and embedded platforms.
4. **End-to-End Integration**: Complete integration with the vLLM v0.13.0 inference framework, supporting both FP8 and INT8 quantization. The implementation employs operator fusion (fused quant slide + sparse GEMM + fused dequant bias) to minimize the overhead introduced by the sliding transformation. The additional quantization and sliding operations are fused into the activation quantization step, nearly eliminating overhead from the transformation.
5. **Comprehensive Validation**: Extensive benchmarking across:
   - **6 GPU platforms** spanning consumer, datacenter, and embedded systems (A100, H100, B200, RTX 4090, RTX 5080, DGX Spark GB10)
   - **5 data precisions** (FP16, BF16, FP8 E4M3, INT8, FP4 E2M1)
   - **5 model architectures** (Llama3.2-1B, Llama3.2-3B, Qwen2.5-7B, Qwen2.5-14B, BitNet1.58-2B)
   - **8 sparsity configurations** (2:4, 2:6, 2:8, 2:10, 2:12, 2:14, 2:16, 2:∞)
   - All results are fully reproducible with provided scripts and Docker images
6. **Production-Ready Deployment**: The entire codebase is packaged into Docker images supporting both x86_64 and aarch64 architectures with CUDA 12.9. Source code is hosted in this repository, enabling direct reproduction without environment configuration challenges.

---

## 3. Theoretical Foundation

### 3.1 Relaxed Structured Sparsity (Z:L Sparsity)

SlideSparse defines a generalized sparsity format **Z:L**, where:

- **L** is the window size (number of consecutive elements)
- **Z** is the minimum number of zeros within each window
- **N = L - Z** is the maximum number of non-zero elements per window

The hardware-supported 2:4 sparsity is a special case where Z=2, L=4. SlideSparse generalizes this to support any Z:L' pattern where L' = L + k×(L-Z) for k = 0, 1, 2, 3, ...

This means the following sparsity patterns are naturally supported:

- **2:4** (k=0): Standard hardware-native 50% sparsity
- **2:6** (k=1): 33% sparsity, 1.5× potential speedup
- **2:8** (k=2): 25% sparsity, 1.33× potential speedup
- **2:10** (k=3): 20% sparsity, 1.25× potential speedup
- And so on...

### 3.2 Overlapping Sliding Window Mechanism

The core transformation operates through overlapping sliding windows that decompose the original weight matrix into multiple sub-windows. Each window of L_source elements (satisfying Z:L_source sparsity) is decomposed into multiple overlapping windows of L_target=4 elements (satisfying 2:4 sparsity).

**Conceptual Example (2:8 → 2:4 transformation):**

Consider an original weight sequence with 2:8 sparsity (2 zeros in every 8 consecutive elements):

```
Original weight sequence (2:8 sparse):  [a₁ 0 a₂ a₃ 0 a₄ a₅ a₆]
                                         ↓ Overlapping window decomposition
Window 1 (positions 0-3):               [a₁  0  a₂  a₃]  → 2:4 compliant
Window 2 (positions 2-5):               [a₂  a₃  0  a₄]  → 2:4 compliant  
Window 3 (positions 4-7):               [0  a₄  a₅  a₆]  → 2:4 compliant

Expanded sequence:  [a₁ 0 a₂ a₃ | a₂ a₃ 0 a₄ | 0 a₄ a₅ a₆]
```

**Key Parameters and Formulas:**

- **Window size** = L_target = 4 (for 2:4 hardware)
- **Stride** = L_target - Z_target = 4 - 2 = 2 (non-zero elements per target window)
- **Number of windows** = (L_source - Z_source) / Stride = (8 - 2) / 2 = 3
- **Expansion ratio** = (num_windows × L_target) / L_source = (3 × 4) / 8 = 1.5

### 3.3 Dimension Expansion and Final Compression

The sliding operation expands the K dimension of weight matrices. Subsequently, cuSPARSELt's 2:4 compression halves this expanded dimension:

| Source Sparsity | Expansion Ratio | Original K | Expanded K' | Compressed K'' (Final) |
| --------------- | --------------- | ---------- | ----------- | ---------------------- |
| 2:4             | 1.00×          | 4096       | 4096        | 2048                   |
| 2:6             | 1.33×          | 4096       | 5460        | 2730                   |
| 2:8             | 1.50×          | 4096       | 6144        | 3072                   |
| 2:10            | 1.67×          | 4096       | 6826        | 3413                   |

**Expected Latency Analysis (2:8 Sparsity Example):**

- Original: 8 elements with 2 zeros, 6 non-zeros
- Required windows: ⌈6/2⌉ = 3 groups of 2:4 sparse operations
- Expanded total length: 3 × 4 = 12 elements
- Expected latency ratio: (6/2 × 4) / 2 / 8 = 75% (compared to dense computation)
- This translates to a 1.33× speedup, matching the theoretical ceiling for 25% sparsity

### 3.4 Greedy Residual Allocation

SlideSparse employs a greedy residual allocation strategy during the sliding transformation that ensures:

1. **Complete coverage**: All zero elements in the original matrix are preserved
2. **Constraint satisfaction**: Non-zero elements are distributed to positions satisfying the target 2:4 sparsity constraints
3. **Minimization**: The expanded total length is minimized to reduce computational overhead

### 3.5 General Formula for Hardware Acceleration

For hardware supporting Z:L sparsity acceleration (L elements with at least Z zeros, at most N=L-Z non-zeros):

| Parameter          | Formula                   | Description                              |
| ------------------ | ------------------------- | ---------------------------------------- |
| Acceleration ratio | L / N                     | e.g., 2:4 → 4/2 = 2×, 3:4 → 4/1 = 4× |
| Supported formats  | Z:L', where L' = L + k×N | k = 0, 1, 2, 3, ...                      |
| Window size        | L (or divisors of L)      | Target hardware window                   |
| Stride             | N = L - Z                 | Non-zeros per window                     |
| Overlap width      | Z                         | Zeros shared between windows             |
| Latency ratio      | (L' - Z) / L'             | Relative to dense baseline               |

---

## 4. Repository Structure

All SlideSparse-specific implementations are contained within the `slidesparse/` directory, with minimal modifications to the vLLM source code. This design philosophy ensures maintainability and allows for easy updates when new vLLM versions are released.

```
slidesparse/
├── __init__.py
├── utils.py                          # Unified utilities for SlideSparse
├── core/                             # Core implementation of SlideSparse LinearMethods
│   ├── __init__.py
│   ├── config.py                     # Environment variable configuration management
│   ├── SlideSparseLinearMethod_FP8.py   # FP8 linear layer implementation
│   ├── SlideSparseLinearMethod_INT8.py  # INT8 linear layer implementation
│   ├── gemm_wrapper.py               # GEMM wrapper with algorithm lookup
│   ├── kernels.py                    # Triton kernel loading and management
│   └── profiler.py                   # Performance profiling utilities
│
├── csrc/                             # CUDA and Triton kernel source code
│   ├── utils.py
│   ├── cublaslt_gemm/                # cuBLASLt dense GEMM implementation
│   │   ├── cublaslt_gemm.cu
│   │   └── build_cublaslt.py
│   ├── cusparselt_gemm/              # cuSPARSELt sparse GEMM implementation
│   │   ├── cusparselt_gemm.cu
│   │   └── build_cusparselt.py
│   ├── quant_only_triton/            # Quantization-only Triton kernel (for cuBLASLt)
│   │   ├── autotune_autogen_quant_only.py
│   │   ├── basic_quant_only_triton.py
│   │   └── run_benchmark.py
│   ├── fused_quant_slide_triton/     # Fused quant+slide Triton kernel (for cuSPARSELt)
│   │   ├── autotune_autogen_quant_slide.py
│   │   ├── basic_quant_slide_triton.py
│   │   ├── run_benchmark.py
│   │   └── benchmark_result/
│   └── fused_dequant_bias_triton/    # Fused dequant+bias Triton kernel (for both)
│       ├── autotune_autogen_dequant_bias.py
│       ├── basic_dequant_bias_triton.py
│       └── run_benchmark.py
│
├── search/                           # Offline algorithm search for GEMM optimization
│   ├── utils.py
│   ├── cuBLASLt_AlgSearch/           # cuBLASLt algorithm ID search
│   │   ├── alg_search.py
│   │   ├── alg_search_cublaslt.cu
│   │   └── alg_search_results/
│   ├── cuBLASLt_LayoutSearch/        # cuBLASLt matrix layout search
│   ├── cuSPARSELt_AlgSearch/         # cuSPARSELt algorithm ID search
│   │   ├── alg_search.py
│   │   ├── alg_search_cusparselt.cu
│   │   └── alg_search_results/
│   └── cuSPARSELt_LayoutSearch/      # cuSPARSELt matrix layout search
│
├── benchmark_kernel/                 # Kernel-level throughput benchmarking
│   ├── __init__.py
│   ├── utils.py
│   ├── benchmark_entry.py            # Unified benchmark entry point
│   ├── prepare_for_kernel_bench.py   # Automated kernel benchmark pipeline
│   ├── extract_kernel_results.py     # Result extraction and analysis
│   ├── cuBLASLt/                     # cuBLASLt kernel benchmarks
│   │   └── alg_search_results/       # Raw benchmark data per GPU
│   ├── cuSPARSELt/                   # cuSPARSELt kernel benchmarks
│   │   └── alg_search_results/       # Raw benchmark data per GPU
│   └── kernel_speedup_results/       # Processed speedup results
│       ├── A100_cc80_py312_cu129_x86_64/
│       ├── H100_cc90_py312_cu129_x86_64/
│       ├── B200_cc100_py312_cu129_x86_64/
│       ├── RTX4090_cc89_py312_cu129_x86_64/
│       ├── RTX5080_cc120_py312_cu129_x86_64/
│       └── GB10_cc121_py312_cu129_aarch64/
│
├── tools/                            # End-to-end benchmarking and utilities
│   ├── utils.py
│   ├── model_download.py             # Model download utility
│   ├── throughput_benchmark.py       # vLLM throughput benchmark script
│   ├── prepare_for_vllm_bench.py     # Automated end-to-end benchmark pipeline
│   ├── prepare_for_bitnet_bench.py   # BitNet-specific benchmark pipeline
│   ├── offline_autotune_algsearch.py # Offline autotuning entry point
│   ├── accuracy_quickbench.py        # Quick accuracy verification
│   ├── extract_end2end_results.py    # Result extraction and analysis
│   ├── throughput_benchmark_results/ # Raw benchmark logs
│   │   ├── prefill/
│   │   └── decode/
│   └── end2end_speedup_results/      # Processed end-to-end results
│       ├── A100/
│       ├── H100/
│       ├── B200/
│       ├── RTX4090/
│       ├── RTX5080/
│       └── GB10/
│
├── weight_convert/                   # Offline weight transformation pipeline
│   ├── README.md
│   ├── utils.py
│   ├── weight_convert_entry.py       # Unified conversion entry point
│   ├── prune.py                      # Magnitude/random pruning
│   ├── slide.py                      # Sliding window transformation
│   ├── compress.py                   # cuSPARSELt 2:4 compression
│   ├── cusparselt_compress.cu        # CUDA compression kernel
│   └── test_correctness/             # Correctness verification tests
│
└── test/                             # Integration tests for vLLM
    ├── utils.py
    ├── run_all_suite.sh
    ├── FP8_vllm/                     # FP8 integration tests
    │   ├── test_01_bridge.py         # Bridge/registration tests
    │   ├── test_02_kernel.py         # Kernel correctness tests
    │   ├── test_03_inference.py      # End-to-end inference tests
    │   └── test_04_throughput.py     # Throughput validation tests
    └── INT8_vllm/                    # INT8 integration tests
```

### 4.1 Core Module (`core/`)

The `core/` directory contains the heart of SlideSparse's implementation:

- **`SlideSparseLinearMethod_FP8.py` and `SlideSparseLinearMethod_INT8.py`**: These implement the `LinearMethodBase` interface from vLLM, modifying the `apply()` method to replace the standard Quant+GEMM+Dequant chain with our custom backend. The key functions exported include `SlideSparseFp8LinearMethod`, `SlideSparseInt8LinearMethod`, `wrap_scheme_fp8()`, and `wrap_scheme_int8()`.
- **`gemm_wrapper.py`**: Wraps both cuBLASLt and cuSPARSELt GEMM implementations and provides the **online algorithm configuration lookup** functionality via the `AlgorithmConfigManager` class. This enables O(1) runtime lookup of pre-computed optimal algorithm configurations.
- **`kernels.py`**: Manages the loading of Triton kernels for quantization and dequantization operations. It handles three kernel types: `quant_only` (for cuBLASLt path), `quant_slide` (for cuSPARSELt path), and `dequant_bias` (shared by both paths).
- **`profiler.py`**: Provides performance profiling utilities to capture and print kernel execution times during end-to-end inference, useful for performance analysis and debugging.

### 4.2 CUDA/Triton Source (`csrc/`)

The `csrc/` directory contains all low-level kernel implementations:

- **Triton Kernels**: Each Triton kernel directory contains:

  - `autotune_autogen_*.py`: Integrates automatic parameter tuning with automatic Triton code generation. These scripts generate model-specific Triton code with if-else branches that enable O(1) complexity lookup of optimal configurations based on actual GEMM dimensions (N, K are known from model parameters; M varies during inference).
  - `basic_*_triton.py`: Fallback implementations without model-specific tuning.
  - `run_benchmark.py`: Benchmarking scripts where the baseline is pure memory copy performance. Our implementations achieve near-theoretical memory bandwidth utilization.
- **CUDA GEMM Libraries**: `cublaslt_gemm/` and `cusparselt_gemm/` contain CUDA source files and build scripts for the GEMM wrapper libraries.

The `fused_quant_slide_triton/` kernel deserves special attention—it has been deeply optimized using double buffering and 1D program allocation strategies to ensure that even on consumer GPUs with smaller L2 caches, there are no cache pressure or overflow issues. Benchmark results in `benchmark_result/` demonstrate that for 2:8 sparsity (1.5× K expansion), the latency ratio is approximately 1.5× as well, meaning the computational overhead is completely masked by memory I/O.

### 4.3 Search and Optimization (`search/`)

The `search/` directory contains offline algorithm search tools, organized into four categories:

- **cuBLASLt_AlgSearch** and **cuSPARSELt_AlgSearch**: Search for optimal algorithm IDs for each (M, N, K) configuration
- **cuBLASLt_LayoutSearch** and **cuSPARSELt_LayoutSearch**: Explore the 8 possible GEMM layout configurations

The algorithm search results are stored in `alg_search_results/` subdirectories, recording algorithm counts, top-3 algorithm configurations, and detailed performance metrics for reproducibility.

### 4.4 Testing (`test/`)

The test suite is organized by data type (FP8 and INT8), with four progressive test stages:

1. **test_01_bridge.py**: Verifies that the SlideSparse bridge file correctly registers with vLLM's quantization framework
2. **test_02_kernel.py**: Tests individual kernel correctness for all three backends (CUTLASS, cuBLASLt, cuSPARSELt)
3. **test_03_inference.py**: Validates end-to-end inference output correctness
4. **test_04_throughput.py**: Confirms expected throughput improvements

### 4.5 vLLM Source Code Modifications

SlideSparse maintains **minimal invasiveness** to the vLLM codebase. Only two files are modified:

1. **`vllm/model_executor/layers/quantization/slidesparse.py`** (New file)

   - Bridge file that imports SlideSparse modules into vLLM's quantization framework
   - Exports configuration functions (`is_slidesparse_enabled()`, `is_cublaslt_enabled()`, `is_cusparselt_enabled()`, `get_sparsity_config()`, etc.) and LinearMethod wrappers
   - All logic resides in the external `slidesparse/` module; this file only handles imports and forwarding
2. **`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`** (Modified)

   - Added SlideSparse wrapping in the `get_scheme()` method
   - When `is_slidesparse_enabled()` returns True and the scheme is `CompressedTensorsW8A8Fp8` or `CompressedTensorsW8A8Int8`, the scheme is wrapped with `wrap_scheme_fp8()` or `wrap_scheme_int8()` respectively
   - This transparently hooks FP8 and INT8 schemes when SlideSparse is enabled via environment variables

### 4.6 Utility Design Philosophy

We have made extensive efforts to unify utilities in the top-level `utils.py` file, avoiding redundant implementations. Each subdirectory has its own `utils.py` that may extend or specialize the top-level utilities for specific needs. The `slidesparse/utils.py` provides:

- `HardwareInfo`: A singleton class that caches all hardware information (GPU name, compute capability, CUDA version, architecture)
- `FileNameBuilder`: Standardized file naming following the pattern `{prefix}_{GPU}_{CC}[_{dtype}]_{PyVer}_{CUDAVer}_{Arch}.{ext}`
- `FileFinder`: File discovery with fuzzy matching support
- `ModuleLoader`: Dynamic module loading for both `.py` and `.so` files

---

## 5. Supported Configurations

### 5.1 GPU Platforms

SlideSparse has been validated across 6 GPU platforms spanning multiple architectures, demonstrating the systematic nature of our implementation:

| GPU                     | Architecture | Compute Capability | Memory         | Platform Type    | Hardware Architecture |
| ----------------------- | ------------ | ------------------ | -------------- | ---------------- | --------------------- |
| NVIDIA A100 80GB PCIe   | Ampere       | sm80               | 80 GB HBM2e    | Cloud/Datacenter | x86_64                |
| NVIDIA H100 80GB PCIe   | Hopper       | sm90               | 80 GB HBM3     | Cloud/Datacenter | x86_64                |
| NVIDIA B200 180GB SXM   | Blackwell    | sm100              | 180 GB HBM3e   | Cloud/Datacenter | x86_64                |
| NVIDIA RTX 4090         | Ada Lovelace | sm89               | 24 GB GDDR6X   | Consumer         | x86_64                |
| NVIDIA RTX 5080         | Blackwell    | sm120              | 16 GB GDDR7    | Consumer         | x86_64                |
| NVIDIA DGX Spark (GB10) | Blackwell    | sm121              | 128 GB Unified | Embedded/Mobile  | aarch64               |

This selection covers:

- **GPU Generations**: Ampere (sm80), Ada Lovelace (sm89), Hopper (sm90), Blackwell (sm100/120/121)
- **Platform Types**: Consumer GPUs, cloud/datacenter GPUs, and embedded/mobile devices
- **Hardware Architectures**: Both x86_64 (AMD64) and aarch64 (ARM64)
- **Memory Types**: HBM2e, HBM3, HBM3e, GDDR6X, GDDR7, and unified memory

The implementation supports both **x86_64** and **aarch64** hardware architectures through separate Docker images, ensuring broad deployment compatibility.

### 5.2 Data Precision Types

Kernel-level benchmarking covers 5 precision types, representing the full spectrum of inference-relevant data formats:

| Precision | Description                             | cuBLASLt | cuSPARSELt | Hardware Requirement  | Primary Use Case                         |
| --------- | --------------------------------------- | -------- | ---------- | --------------------- | ---------------------------------------- |
| FP16      | Half-precision float                    | ✓       | ✓         | Ampere+ (sm80+)       | Most common precision, general inference |
| BF16      | Brain floating-point                    | ✓       | ✓         | Ampere+ (sm80+)       | Training-oriented, better dynamic range  |
| INT8      | 8-bit integer                           | ✓       | ✓         | Ampere+ (sm80+)       | W8A8 quantized inference                 |
| FP8 E4M3  | 8-bit float (4-bit exp, 3-bit mantissa) | ✓       | ✓         | Ada Lovelace+ (sm89+) | Low-precision quantized inference        |
| FP4 E2M1  | 4-bit float (2-bit exp, 1-bit mantissa) | ✓       | ✓         | Blackwell+ (sm100+)   | Future trend, ultra-low precision        |

**Precision Selection Rationale:**

- **FP16**: The most widely deployed precision for LLM inference
- **BF16**: Preferred for training and transfer learning scenarios due to its larger dynamic range
- **INT8**: The standard for W8A8 (weight and activation 8-bit) quantization schemes
- **FP8 E4M3**: Emerging as the go-to precision for quantized inference, balancing accuracy and throughput
- **FP4 E2M1**: Represents the cutting edge of ultra-low precision inference, currently only available on Blackwell GPUs

### 5.3 Sparsity Configurations

| Sparsity     | Zero Ratio | K Expansion | Theoretical Speedup | Supported                    |
| ------------ | ---------- | ----------- | ------------------- | ---------------------------- |
| 2:4          | 50%        | 1.00×      | 2.00×              | ✓                           |
| 2:6          | 33%        | 1.33×      | 1.50×              | ✓                           |
| 2:8          | 25%        | 1.50×      | 1.33×              | ✓                           |
| 2:10         | 20%        | 1.67×      | 1.25×              | ✓                           |
| 2:12         | 17%        | 1.83×      | 1.20×              | ✓ (Extended testing)        |
| 2:14         | 14%        | 2.00×      | 1.17×              | ✓ (Extended testing)        |
| 2:16         | 12.5%      | 2.17×      | 1.14×              | ✓ (Extended testing)        |
| 2:∞ (Dense) | 0%         | 2.00×      | 1.00×              | ✓ (Experimental validation) |

**Note on 2:∞ Dense Mode**: This experimental mode serves as a mathematical validation of the SlideSparse algorithm. It artificially inserts zeros into a fully dense weight matrix and then uses the 2:4 sparsity mechanism to skip them. The 2× K expansion (doubling K through sliding, then halving via 2:4 compression) results in a theoretical 1.00× speedup (no net change), confirming that the algorithm correctly handles the boundary case where no actual zeros exist in the original weights.

**Core Experimental Focus**: Our primary experiments focus on **2:4, 2:6, 2:8, and 2:10** sparsity configurations, as these represent practically achievable sparsity levels that can maintain reasonable model accuracy while providing meaningful speedups.

### 5.4 Model Support

End-to-end testing covers 5 model architectures in both FP8 and INT8 quantization formats:

| Model         | Parameters | Architecture              | FP8 | INT8 | Notes                                                  |
| ------------- | ---------- | ------------------------- | --- | ---- | ------------------------------------------------------ |
| Llama3.2-1B   | 1B         | Llama                     | ✓  | ✓   | Small model, fits on consumer GPUs                     |
| Llama3.2-3B   | 3B         | Llama                     | ✓  | ✓   | Medium model                                           |
| Qwen2.5-7B    | 7B         | Qwen                      | ✓  | ✓   | Large model, requires datacenter/high-end consumer GPU |
| Qwen2.5-14B   | 14B        | Qwen                      | ✓  | ✓   | Extra-large model, datacenter GPU recommended          |
| BitNet1.58-2B | 2B         | BitNet (Ternary 1.58-bit) | ✓  | ✓   | Novel ternary quantization architecture                |

**Model Variant Summary:**

- **10 base models**: 5 model architectures × 2 precision formats (FP8, INT8)
- **40 sparse variants**: 10 base models × 4 sparsity configurations (2:4, 2:6, 2:8, 2:10)
- **Total: 50 model variants** evaluated across all supported GPUs

**BitNet Special Handling**: Due to BitNet's unique ternary (+1/0/-1) quantization scheme, vLLM does not natively support it. We provide a dedicated pipeline script (`prepare_for_bitnet_bench.py`) that handles the necessary model configuration adjustments and weight transformations.

### 5.5 Model Linear Layers

For each model, the following linear layers are processed and benchmarked:

| Linear Layer | Model Location         | Typical Dimensions                  |
| ------------ | ---------------------- | ----------------------------------- |
| Wqkv         | `self_attn.qkv_proj` | `[3×H, H]` or `[(Q+2×KV), H]` |
| Wo           | `self_attn.o_proj`   | `[H, H]`                          |
| W13          | `mlp.gate_up_proj`   | `[2×I, H]`                       |
| W2           | `mlp.down_proj`      | `[H, I]`                          |

Where H is the hidden dimension and I is the intermediate (MLP) dimension. These four linear layer types represent the computational bottleneck in transformer inference.

---

## 6. Getting Started

### 6.1 Environment Setup

SlideSparse is integrated with vLLM v0.13.0 and requires CUDA 12.9+ with cuSPARSELt 0.8.1.

#### 6.1.1 Development Workflow Overview

Our development workflow follows this pattern:

1. **Source Code**: Fork of the public `vllm-project/vllm` repository, using the stable 0.13.0 branch as our local main branch
2. **Base Image**: Public Docker image `vllm/vllm-openai:v0.13.0` containing Ubuntu 22.04, CUDA 12.9, PyTorch 2.9, and Flash-Attention dependencies
3. **Dependency Completion**: Dockerfile builds on the base image to add cuSPARSELt (`cusparselt-cuda-12`) and necessary header files
4. **Uninstall Existing vLLM**: Remove the pre-installed vLLM (`pip uninstall vllm`) to enable development mode
5. **Mount Source**: Use devcontainer to mount the host's GitHub source code to the container's vLLM development directory
6. **Activate**: After container startup, run `pip install -e .` to build vLLM extensions and create symlinks to the source repository

After this setup, all vLLM execution uses the GitHub source `.py` files and precompiled wheels, not the pip-installed vLLM. Subsequent modifications to Python-level Model Executor or Triton Kernels take effect immediately without recompilation (since vLLM code is mounted).

#### 6.1.2 Docker Environment (Recommended)

The development environment is based on the official vLLM Docker image with additional dependencies:

```dockerfile
# Base image
FROM vllm/vllm-openai:v0.13.0

# Install cuSPARSELt
RUN apt-get update && apt-get install -y libcusparselt0 libcusparselt-dev

# Remove vLLM from base image to enable development mode
RUN pip uninstall -y vllm
```

**Development Mode Installation:**

```bash
# Clone the repository
git clone <repository-url>
cd vllmbench

# Install in development mode (creates symlinks to source)
pip install -e .
```

After installation, modifications to Python files in `vllm/` and `slidesparse/` take effect immediately without recompilation.

#### 6.1.3 Architecture Support

We provide Docker images for both hardware architectures (will be released after publication)

- **x86_64 (AMD64)**: For standard servers, workstations, and cloud instances
- **aarch64 (ARM64)**: For ARM-based systems like NVIDIA DGX Spark (GB10)

### 6.2 Environment Variables

SlideSparse behavior is controlled through environment variables:

| Variable                | Values                | Description                                                      |
| ----------------------- | --------------------- | ---------------------------------------------------------------- |
| `DISABLE_SLIDESPARSE` | 0/1                   | Disable SlideSparse entirely, use vLLM native path               |
| `USE_CUBLASLT`        | 0/1                   | Enable cuBLASLt dense GEMM backend (baseline)                    |
| `USE_CUSPARSELT`      | 0/1                   | Enable cuSPARSELt sparse GEMM backend (accelerated)              |
| `SPARSITY`            | 2_4, 2_6, ... , 2_inf | Sparsity configuration (cuSPARSELt only). 2_inf is experimental. |
| `SLIDESPARSE_PROFILE` | 0/1                   | Enable kernel timing diagnostics via profiler.py                 |

**Backend Selection Logic:**

- If `DISABLE_SLIDESPARSE=1`: Use vLLM's native CUTLASS path
- If `USE_CUBLASLT=1`: Use cuBLASLt dense GEMM (our baseline)
- If `USE_CUSPARSELT=1`: Use cuSPARSELt sparse GEMM (our accelerated path)
- Default (no variables set): SlideSparse is enabled with CUTLASS backend provided by vLLM

**Example Usage:**

```bash
# Run with cuSPARSELt backend at 2:8 sparsity
USE_CUSPARSELT=1 SPARSITY=2_8 vllm serve model_path

# Run with cuBLASLt baseline
USE_CUBLASLT=1 vllm serve model_path

# Disable SlideSparse completely (use vLLM native CUTLASS)
DISABLE_SLIDESPARSE=1 vllm serve model_path

# Enable profiling to observe kernel execution times
SLIDESPARSE_PROFILE=1 USE_CUSPARSELT=1 SPARSITY=2_8 vllm serve model_path
```

### 6.3 GEMM Backend Paths

SlideSparse provides three backend paths for FP8 and INT8 linear layers:

**Path 1: CUTLASS (Fallback)**

- Uses vLLM's native `cutlass_scaled_mm` function
- Activated when `DISABLE_SLIDESPARSE=1`
- Note: CUTLASS compilation may have sm version limitations

**Path 2: cuBLASLt (Dense Baseline)**

- Our implemented dense GEMM baseline
- Pipeline: `Triton quant_only` → `cuBLASLt FP8/INT8 GEMM` → `Triton dequant_bias`
- Activated with `USE_CUBLASLT=1`

**Path 3: cuSPARSELt (Sparse Accelerated)**

- Our sparse GEMM acceleration path
- Pipeline: `Triton fused_quant_slide` → `cuSPARSELt 2:4 sparse GEMM` → `Triton dequant_bias`
- Activated with `USE_CUSPARSELT=1` and `SPARSITY=2_L`

### 6.4 Quick Start

**Basic Inference:**

```bash
# Using compressed-tensors quantization (SlideSparse hooks automatically)
vllm serve Qwen2.5-7B-Instruct-FP8-dynamic --quantization compressed-tensors
```

**Throughput Benchmark:**

```bash
cd slidesparse/tools

# Quick benchmark with default settings
python throughput_benchmark.py --model qwen2.5-7b-fp8 --backend cublaslt

# Benchmark with cuSPARSELt sparse backend
python throughput_benchmark.py --model qwen2.5-7b-fp8 --backend cusparselt --sparsity 2_8
```

### 6.5 torch.compile Compatibility

To support vLLM's full graph compilation, SlideSparse registers custom operators via `torch.library`:

- GEMM wrappers provide both real implementations and fake implementations for tracing
- Triton kernels are registered with proper signatures
- Algorithm lookup is made compile-friendly through pre-loading at module import time

**Important Note**: Due to the newness of the DGX Spark GB10 hardware, it is the only platform that requires eager mode execution. This results in approximately 30% performance degradation compared to compiled mode on other platforms.

### 6.6 Algorithm Configuration Caching

The algorithm search system uses a two-tier approach:

1. **Offline Search**: Pre-computed optimal algorithm configurations for each GPU/model/precision combination
2. **Online Lookup**: O(1) runtime lookup of configurations based on (model, N, K, M_range)

**Cache Invalidation**: Algorithm configurations are tied to specific hardware (GPU model, compute capability, driver version) and software (Python version, CUDA version, architecture). When using the Docker image on the exact tested platforms, no additional configuration is needed. If running on different hardware, the system automatically performs algorithm search and caches results for subsequent runs.

---

## 7. Kernel-Level Benchmarking

### 7.1 Overview

Kernel-level benchmarks measure raw GEMM performance isolated from end-to-end inference overhead. This provides critical insights into the theoretical speedup achievable at the computational level before considering system-level factors.

Two benchmark modes are provided:

1. **Square Mode**: Tests M=N=K configurations (where M, N, K all equal the same value) for systematic performance analysis. This mode is useful for understanding hardware characteristics and comparing performance across different problem sizes in a controlled manner.
2. **Model Mode**: Tests actual (N, K) dimensions extracted from target model linear layers. This mode directly reflects the real-world GEMM shapes encountered during inference and enables direct comparison with end-to-end results.

### 7.2 Benchmark Entry Points

**Unified Entry Script (`prepare_for_kernel_bench.py`):**

This script provides a comprehensive automated pipeline with checkpoint recovery, enabling robust execution of long-running benchmark suites:

```bash
cd slidesparse/benchmark_kernel

# Run complete benchmark suite
python prepare_for_kernel_bench.py --task 1,1,1,1,1,1

# Task breakdown:
# Task 1: cuBLASLt Model benchmarks (8 models, 5 precisions)
# Task 2: cuBLASLt Square benchmarks (5 precisions)
# Task 3: cuSPARSELt Model benchmarks - high sparsity (2_4, 2_6, 2_8, 2_10)
# Task 4: cuSPARSELt Square benchmarks - high sparsity
# Task 5: cuSPARSELt Model benchmarks - low sparsity (2_12, 2_14, 2_16, 2_inf)
# Task 6: cuSPARSELt Square benchmarks - low sparsity
```

**Individual Benchmark (`benchmark_entry.py`):**

For targeted testing of specific configurations:

```bash
# Square mode (M=N=K): omit --model or use --model square
python benchmark_entry.py --dtype fp8e4m3 --backend cublaslt --m_list 1024,2048,4096

# Model-based mode: specify --model with a model name, use --Lmax or --sparsity for sparsity config
python benchmark_entry.py --model Qwen2.5-7B --dtype int8 --backend cusparselt --sparsity 2_8
```

### 7.3 M Dimension Configurations

The M dimension (batch size × sequence length for activations) is systematically varied to capture performance across different workload sizes:

| Mode   | M Values                                                | Purpose              |
| ------ | ------------------------------------------------------- | -------------------- |
| Square | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384        | Systematic analysis  |
| Model  | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 | Real-world workloads |

### 7.4 K Dimension Handling for Sparsity

In kernel testing, the K dimension is adjusted based on sparsity to reflect the actual GEMM dimensions after the sliding transformation:

- **cuBLASLt baseline** and **2:4 sparsity**: K remains unchanged (no expansion)
- **2:6 sparsity**: K_slide = K × 1.33
- **2:8 sparsity**: K_slide = K × 1.50
- **2:10 sparsity**: K_slide = K × 1.67
- **2:∞ (dense)**: K_slide = K × 2.00

For example, in M=N=K=1024 Square testing with 2:8 sparsity, the actual GEMM performed is M=1024, N=1024, K_slide=1536.

### 7.5 Benchmark Metrics

Each benchmark records comprehensive performance data:

- **Latency (μs)**: Average execution time per GEMM operation, computed from multiple iterations with proper warmup
- **Throughput (TOPS)**: Tera-operations per second, calculated as `2×M×N×K / latency`
- **Speedup**: Ratio relative to cuBLASLt dense baseline for the same effective computation
- **Algorithm Configuration**: Top-3 algorithm IDs and their relative performance, enabling reproducibility
- **Algorithm Count**: Total number of available algorithms for the given configuration

### 7.6 Model-Specific Analysis

For Model Mode benchmarks, we recognize that during actual inference, a single M value triggers four different (N, K) linear layer pairs simultaneously. Therefore, we aggregate results by:

1. Recording absolute latency for each individual (M, N, K) combination
2. Summing latencies across all four (N, K) pairs for the same M
3. Computing total speedup as the ratio of summed baseline latency to summed sparse latency

This aggregated analysis directly corresponds to expected end-to-end performance improvements.

### 7.7 Result Extraction

```bash
# Extract and analyze kernel benchmark results
python extract_kernel_results.py

# Results are saved to kernel_speedup_results/{GPU_ID}/
```

The extracted results include:

- Per-precision latency tables
- Cross-sparsity speedup comparisons
- Model-aggregated performance summaries

---

## 8. End-to-End Benchmarking

### 8.1 Overview

End-to-end benchmarks measure actual inference throughput (tokens/s) using the methodology of the official `vllm bench throughput` command. The custom `throughput_benchmark.py` script wraps this functionality with additional automation and result collection specific to SlideSparse experiments.

The core goal is to validate that kernel-level speedups translate to real-world inference performance improvements. We test both Prefill (compute-bound) and Decode (memory-bound) phases to understand performance characteristics across different inference workloads.

### 8.2 Benchmark Modes

**Prefill Mode (Compute-Bound):**

Prefill is the initial phase of inference where the model processes the entire input prompt. This phase is **compute-bound**, making it an ideal target for GEMM acceleration.

- **M Control**: `M = max_num_seqs × prompt_length`
- **Decode Minimization**: Setting `output_len=1` ensures minimal decode overhead
- **Iteration Stability**: We configure N=128 complete inference iterations to obtain stable performance measurements
- **M Values**: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 (representing long-context input scenarios)

**Decode Mode (Memory-Bound):**

Decode is the autoregressive generation phase where the model generates one token at a time. This phase is **memory-bound** due to the need to access the KV cache.

- **M Control**: `M = max_num_seqs` (each request generates tokens independently)
- **Prefill Minimization**: Using single-token prompts ensures minimal prefill overhead
- **Iteration Stability**: We configure N=256 decode iterations per request to obtain stable performance
- **M Values**: 64, 128, 256, 512 (representing high-concurrency request scenarios)

### 8.3 Benchmark Entry Points

**Unified Pipeline (`prepare_for_vllm_bench.py`):**

This script orchestrates the complete end-to-end benchmark workflow, including model preparation, offline optimization, and benchmark execution:

```bash
cd slidesparse/tools

# Complete end-to-end benchmark pipeline
python prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1

# Task breakdown:
# Task 1: Model download (model_download.py - downloads 8 base models)
# Task 2: Model conversion (weight_convert_entry.py - generates 2_4, 2_6, 2_8, 2_10 variants)
# Task 3: Offline coarse autotuning (cuBLASLt algorithm search + Triton quant_only tuning)
# Task 4: Offline fine autotuning (Triton dequant + quant_slide + cuSPARSELt algorithm search)
# Task 5: Quick validation benchmark (accuracy_quickbench.py)
# Task 6: Full Prefill benchmark
# Task 7: Full Decode benchmark
```

**Individual Throughput Test:**

```bash
python throughput_benchmark.py \
    --model qwen2.5-7b-fp8 \
    --backend cusparselt \
    --sparsity 2_8 \
    --stage prefill \
    --M 1024,2048,4096
```

**Offline Autotuning (`offline_autotune_algsearch.py`):**

For preparing optimal algorithm configurations on new hardware:

```bash
python offline_autotune_algsearch.py --model Qwen2.5-7B --dtype fp8 --Lmax 8
```

### 8.4 BitNet-Specific Benchmarking

Due to BitNet's unique requirements (1.58-bit ternary weights with +1/0/-1 values), vLLM does not natively support it. A dedicated script handles the complete workflow:

```bash
python prepare_for_bitnet_bench.py --task 1,1,1,1,1

# Task breakdown:
# Task 1: BitNet model download
# Task 2: JSON configuration adjustment and weight transformation
# Task 3: Offline autotuning
# Task 4: Prefill benchmark
# Task 5: Decode benchmark
```

### 8.5 Result Structure

```
throughput_benchmark_results/
├── prefill/
│   ├── {GPU}/
│   │   ├── cutlass/           # CUTLASS baseline
│   │   ├── cublaslt/          # cuBLASLt dense
│   │   └── cusparselt/        # cuSPARSELt sparse
│   │       ├── 2_4/
│   │       ├── 2_6/
│   │       ├── 2_8/
│   │       └── 2_10/
└── decode/
    └── {GPU}/
        └── ...
```

### 8.6 Result Extraction

```bash
python extract_end2end_results.py

# Outputs to end2end_speedup_results/{GPU}/
# - Absolute throughput (tokens/s) for each configuration
# - Speedup relative to cuBLASLt baseline
```

### 8.7 Ensuring Fair Comparison

To maximize the fairness of our comparisons, we implement several optimizations:

1. **Algorithm Search**: Both cuBLASLt and cuSPARSELt algorithms are exhaustively searched to find optimal configurations for each (M, N, K) combination
2. **Layout Optimization**: After testing 8 possible GEMM layout configurations (combinations of NN/NT/TN/TT transpose options and RC/CR/CC/RR major ordering), we standardized on **TN+CC+C** layout which showed consistent performance across problem sizes
3. **Warmup**: Each benchmark includes sufficient warmup iterations to ensure GPU caches and TLBs are populated
4. **Repetition**: Multiple iterations are averaged to reduce measurement variance

**Matrix Layout Convention (TN+CC+C):**

```
D[M,N]_row = D[N,M]_col = W[K,N]_col^T × A[K,M]_col = W[N,K]_row^T × A[M,K]_row
```

- **T/N**: First operand (Weight) is transposed; second operand (Activation) is not transposed
- **C/R**: Both inputs are column-major; output is column-major
- This layout showed stable performance with minimal variation across different problem sizes

---

## 9. Weight Conversion Pipeline

### 9.1 Overview

The weight conversion pipeline transforms dense or coarsely-sparse models into SlideSparse-compatible formats through a three-stage process. This offline transformation is essential for enabling the runtime sparse acceleration.

```
Original Model     →  [Prune]  →  [Slide]   →  [Compress]  →  SlideSparse Model
Shape: [N, K]         [N, K]      [N, K']       [N, K'/2]
```

### 9.2 Pipeline Stages

**Stage 1: Pruning (`prune.py`)** — Shape: `[N, K] → [N, K]`

The pruning stage applies Z:L sparsity constraints to the model weights:

- **Magnitude Mode**: Within each window of L consecutive elements, identifies and zeros out the Z smallest elements by absolute value. This typically preserves model accuracy better than random pruning.
- **Random Mode**: Randomly selects Z positions within each window to zero out. Useful for baseline comparisons and analysis.

Input/Output:

- Input: Dense weight tensor `[N, K]`
- Parameters: Z (zeros per window), L (window size), pruning mode
- Output: Pruned weight tensor `[N, K]` satisfying Z:L sparsity constraint

**Stage 2: Sliding (`slide.py`)** — Shape: `[N, K] → [N, K']` where `K' = K × expand_ratio`

The sliding stage is the core SlideSparse transformation:

1. **Padding**: K must be divisible by `4 × L_source` for cuSPARSELt alignment. The formula is:

   ```
   K_padded = ⌈K / (4 × L)⌉ × (4 × L)
   ```
2. **Parameter Calculation**:

   - Stride = L_target - Z_target = 4 - 2 = 2
   - Number of windows = (L_source - Z_source) / Stride
   - Expansion ratio = (num_windows × L_target) / L_source
3. **Window Extraction**: For each window `i` in the original sequence, extract positions `[i×stride, i×stride+L_target)`
4. **Concatenation**: Concatenate all windows to form the expanded weight

**Stage 3: Compression (`compress.py`)** — Shape: `[N, K'] → 1D`

The compression stage calls cuSPARSELt's compression routines to produce the final hardware-compatible format:

- Input: Slided weight `[N, K']` satisfying 2:4 sparsity
- Output: Compressed weight `1D tensor` (non-zero elements only)

The compression is implemented via `cusparselt_compress.cu` which wraps the cuSPARSELt compression API.

### 9.3 Usage

**Complete Conversion (`weight_convert_entry.py`):**

```bash
cd slidesparse/weight_convert

# Convert single model
python weight_convert_entry.py --model qwen2.5-7b-fp8 --Z 2 --L 8

# Convert with offline compression
python weight_convert_entry.py --model qwen2.5-7b-fp8 --Z 2 --L 8 --compress

# Convert all FP8 models
python weight_convert_entry.py --all --quant fp8 --Z 2 --L 8
```

**Individual Stage Execution:**

```bash
# Pruning only
python prune.py --input /path/to/model --Z 2 --L 8 --mode magnitude

# Sliding only
python slide.py --input /path/to/pruned --Z 2 --L 8

# Compression only
python compress.py --input /path/to/slided
```

### 9.4 Output Format

Converted models are saved in the standard safetensors format, maintaining compatibility with vLLM's model loading infrastructure:

```
{Model}-SlideSparse-{Z}_{L}/
├── model.safetensors           # Converted weights (key names unchanged, shapes modified)
├── config.json                 # Original model config (copied)
├── tokenizer.json              # Tokenizer files (copied)
├── tokenizer_config.json       # Tokenizer config (copied)
└── slidesparse_config.json     # SlideSparse metadata
    ├── sparsity: "2:8"
    ├── expand_ratio: 1.5
    ├── processed_layers: [...]
    ├── original_shapes: {...}
    └── compressed_shapes: {...}
```

**Key Design Decision**: Weight key names remain unchanged (e.g., `model.layers.0.self_attn.q_proj.weight`) but shapes change from `[N, K]` to `[N, K'/2]`. This design enables seamless integration with vLLM's `DefaultModelLoader` without requiring modifications to the loader logic.

### 9.5 Accuracy Considerations

While this work primarily focuses on demonstrating inference acceleration, we acknowledge the accuracy impact of pruning:

- **2:4 sparsity (50%)**: Significant accuracy loss, typically requires fine-tuning
- **2:6 sparsity (33%)**: Moderate accuracy loss, may maintain reasonable quality
- **2:8 sparsity (25%)**: Mild accuracy loss, often preserves most capabilities
- **2:10 sparsity (20%)**: Minimal accuracy loss, close to dense model quality

Our current implementation uses magnitude pruning for demonstration purposes. Based on existing literature, 2:6 sparsity can maintain satisfactory model quality. Future work includes sparsity-aware training and fine-tuning for accuracy recovery.

---

## 10. vLLM Integration Architecture

### 10.1 Integration Design Philosophy

SlideSparse follows a **minimal invasion principle**, modifying only two files in the vLLM source while implementing all functionality in the external `slidesparse/` module. This approach ensures:

- **Maintainability**: Upgrades to new vLLM versions require minimal adaptation
- **Modularity**: SlideSparse can be enabled/disabled without affecting core vLLM functionality
- **Transparency**: The integration is invisible to users who simply use the standard `--quantization compressed-tensors` flag

### 10.2 vLLM Quantization Framework (Two-Layer Architecture)

vLLM employs a two-layer architecture for quantization:

**Layer 1: Config + LinearMethod (Glue Layer)**

- `CompressedTensorsConfig`: Parses model configuration and dispatches to appropriate schemes
- `CompressedTensorsLinearMethod`: Implements `LinearMethodBase` interface, delegates all operations to the underlying Scheme
- This layer handles configuration parsing but performs no computation

**Layer 2: Scheme (Implementation Layer)**

- `CompressedTensorsScheme`: Abstract base class defining the quantization interface
- Concrete implementations (e.g., `CompressedTensorsW8A8Fp8`, `CompressedTensorsW8A8Int8`): Handle actual weight management and computation
- Includes internal `LinearOp` classes that wrap kernel calls

SlideSparse intercepts at the Scheme level, wrapping the original schemes to replace the GEMM path.

### 10.3 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        vLLM Quantization Framework                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CompressedTensorsConfig.get_scheme()                               │
│        │                                                            │
│        ├─► CompressedTensorsW8A8Fp8  ─┬─► wrap_scheme_fp8()         │
│        │                              │        │                    │
│        ├─► CompressedTensorsW8A8Int8 ─┴─► wrap_scheme_int8()        │
│        │                                       │                    │
│        │   ┌───────────────────────────────────┘                    │
│        │   │                                                        │
│        │   ▼                                                        │
│        │  SlideSparse LinearMethod (FP8/INT8)                       │
│        │        │                                                   │
│        │        ├── create_weights()    [Delegate to original]      │
│        │        ├── process_weights()   [+ cuSPARSELt compression]  │
│        │        └── apply_weights()     [Replace GEMM path]         │
│        │                 │                                          │
└────────┼─────────────────┼──────────────────────────────────────────┘
         │                 │
         │                 ▼
         │  ┌─────────────────────────────────────────────────────────┐
         │  │              SlideSparse Kernel Paths                   │
         │  ├─────────────────────────────────────────────────────────┤
         │  │                                                         │
         │  │  Path 1: CUTLASS (Fallback)                             │
         │  │    └── vLLM native cutlass_scaled_mm                    │
         │  │                                                         │
         │  │  Path 2: cuBLASLt (Dense Baseline)                      │
         │  │    ├── Triton quant_only kernel                         │
         │  │    ├── cuBLASLt FP8/INT8 GEMM                           │
         │  │    └── Triton dequant_bias kernel                       │
         │  │                                                         │
         │  │  Path 3: cuSPARSELt (Sparse Accelerated)                │
         │  │    ├── Triton fused_quant_slide kernel                  │
         │  │    ├── cuSPARSELt 2:4 sparse GEMM                       │
         │  │    └── Triton dequant_bias kernel                       │
         │  │                                                         │
         │  └─────────────────────────────────────────────────────────┘
         │
         └─► Original vLLM path (when DISABLE_SLIDESPARSE=1)
```

### 10.4 LinearMethod Implementation

The core integration is through `SlideSparseLinearMethod_FP8.py` and `SlideSparseLinearMethod_INT8.py`, which implement vLLM's `LinearMethodBase` interface:

- **`create_weights()`**: Delegates to the original scheme to create weight parameters. For compressed weights, calculates the correct shape `[N, K']` based on sparsity configuration.
- **`process_weights_after_loading()`**: Extends the original processing with optional cuSPARSELt compression if weights are not pre-compressed.
- **`apply_weights()` (Key Method)**: Replaces the original Quant+GEMM+Dequant chain:

  - For cuBLASLt: `quant_only` → `cuBLASLt GEMM` → `dequant_bias`
  - For cuSPARSELt: `fused_quant_slide` → `cuSPARSELt sparse GEMM` → `dequant_bias`

### 10.5 Kernel Fusion Strategy

To minimize overhead from the sliding transformation, SlideSparse employs operator fusion. The sliding transformation expands the K dimension (e.g., 1.5× for 2:8 sparsity), which naively would introduce corresponding memory bandwidth overhead. Our fusion strategy addresses this:

**Fused Quant + Slide Kernel (`fused_quant_slide_triton/`):**

- Combines FP16/BF16 → FP8/INT8 quantization with sliding window expansion in a single memory pass
- Implements double buffering to hide memory latency
- Uses 1D program allocation to manage L2 cache efficiently, avoiding pressure on consumer GPUs with smaller caches
- **Result**: For 2:8 sparsity (1.5× K expansion), the latency overhead is approximately 1.5× compared to quantization alone—achieving near-theoretical memory bandwidth utilization with computational overhead completely masked by I/O

**Fused Dequant + Bias Kernel (`fused_dequant_bias_triton/`):**

- Combines INT32/FP32 → BF16/FP16 dequantization with bias addition
- Shared between cuBLASLt and cuSPARSELt paths (no sliding needed at output)

### 10.6 torch.compile Compatibility

vLLM uses `torch.compile` for full graph compilation to maximize performance. SlideSparse ensures compatibility through several mechanisms:

**Custom Operator Registration:**

- GEMM wrappers are registered via `torch.library` with namespace `"slidesparse"`
- Each operator provides both a real implementation (actual kernel execution) and a fake implementation (returns correctly-shaped empty tensors for tracing)
- Triton kernels are registered with proper function signatures

**Compile-Friendly Algorithm Lookup:**

- Algorithm configurations are loaded at module import time, not during graph tracing
- Lookup uses pure Python dictionary operations that `torch.compile` can handle
- No filesystem operations or dynamic module loading during the compilation phase

**Platform-Specific Notes:**

- DGX Spark GB10 (sm121) currently requires **eager mode** due to limited torch.compile support for this new architecture
- This results in approximately 30% Decode performance degradation compared to compiled mode on other platforms

### 10.7 Matrix Layout Configuration

After extensive testing of 8 possible GEMM layout configurations (combinations of operand transpose options and memory ordering), SlideSparse standardizes on **TN+CC+C** layout:

```
D[M,N]_row = D[N,M]_col = W[K,N]_col^T × A[K,M]_col
           = W[N,K]_row^T × A[M,K]_row
```

**Layout Notation:**

- **T/N**: Transpose (T) or No-transpose (N) for each operand
- **C/R**: Column-major (C) or Row-major (R) for memory layout
- **TN**: First operand (Weight) is transposed; second operand (Activation) is not transposed
- **CC+C**: Both inputs are column-major; output is column-major

This layout was selected because:

1. It showed consistent performance across all tested problem sizes
2. It aligns with NVIDIA's documentation recommendations
3. Some alternative layouts exhibited performance instability on certain matrix dimensions

### 10.8 Model Loading Integration

SlideSparse leverages vLLM's `DefaultModelLoader` without modification:

1. **Offline**: Conversion scripts produce `.safetensors` files with unchanged key names but modified shapes
2. **Online**: `SlideSparseLinearMethod.create_weights()` defines parameters with matching shapes
3. **Loading**: `DefaultModelLoader` loads tensors into the correctly-shaped parameters
4. **Processing**: `process_weights_after_loading()` performs any necessary post-processing

This approach requires no changes to vLLM's model loading infrastructure.

---

## 11. Algorithm Search and Optimization

### 11.1 Algorithm Search Overview

Both cuBLASLt and cuSPARSELt are closed-source libraries that support multiple algorithm implementations for each GEMM configuration. The optimal algorithm varies significantly based on matrix dimensions (M, N, K), data precision, and hardware characteristics. SlideSparse performs extensive offline search to identify optimal algorithms for each configuration, ensuring maximum performance during inference.

### 11.2 Search Methodology

**Algorithm ID Search Process:**

1. **Enumerate Model Shapes**: Extract all (N, K) pairs from target model linear layers (typically 4 pairs per model)
2. **M Value Sweep**: For each M value in the test range, iterate through all available algorithm IDs
3. **Timing Measurement**: Execute 100+ iterations with proper warmup (10+ iterations) to obtain stable latency measurements
4. **Selection**: Record the top-3 performing algorithms along with their latency for reproducibility
5. **Caching**: Save results to JSON files keyed by (model, precision, sparsity, GPU)

**Layout Search Process:**
We systematically tested 8 layout configurations:

- **NN+RC+C, NN+RC+R**: No transpose for either operand, row-major A, column-major B
- **NT+RR+C, NT+RR+R**: No transpose for A, transpose B
- **TN+CC+C, TN+CC+R**: Transpose A, no transpose for B (selected configuration)
- **TT+CR+C, TT+CR+R**: Transpose both operands

After comprehensive testing, we standardized on **TN+CC+C** due to:

- Consistent performance across all tested problem sizes
- Minimal variation on edge cases
- Alignment with NVIDIA documentation recommendations

### 11.3 Search Scripts

The search infrastructure is organized in the `slidesparse/search/` directory:

```bash
cd slidesparse/search

# cuBLASLt algorithm search
cd cuBLASLt_AlgSearch
python alg_search.py --model Qwen2.5-7B --dtype fp8

# cuSPARSELt algorithm search (use --Lmax for sparsity configuration)
cd cuSPARSELt_AlgSearch
python alg_search.py --model Qwen2.5-7B --dtype fp8 --Lmax 8

# Layout search (for validation/exploration)
cd cuBLASLt_LayoutSearch
python layout_search.py --model Qwen2.5-7B --dtype fp8
```

### 11.4 Online Algorithm Lookup

The `AlgorithmConfigManager` class in `gemm_wrapper.py` provides O(1) runtime lookup:

1. **Startup Loading**: At module import time, loads all precomputed JSON configurations for the current hardware
2. **Runtime Matching**: Given (model, N, K, M), performs dictionary lookup to find the optimal algorithm
3. **M-Range Binning**: Since M varies continuously during inference, we partition M into discrete ranges and select the best algorithm for each range
4. **Fallback Strategy**: When no exact match is found, falls back to heuristic selection (typically algorithm ID 0)

### 11.5 Triton Kernel Autotuning

Beyond GEMM algorithm search, we also autotune the three Triton kernels:

**Autotuning Parameters:**

- Block sizes (BLOCK_M, BLOCK_N, BLOCK_K)
- Number of warps
- Number of pipeline stages

**Autotune Scripts (`autotune_autogen_*.py`):**
These scripts generate model-specific Triton code with embedded if-else branches that select optimal parameters based on runtime dimensions:

- N and K are known at model load time (static for each linear layer)
- M varies during inference and is matched to pre-tuned ranges

### 11.6 Search Results Location

```
search/
├── cuBLASLt_AlgSearch/alg_search_results/
│   └── {GPU}_{Model}_{dtype}.json
└── cuSPARSELt_AlgSearch/alg_search_results/
    └── {GPU}_{Model}_{dtype}_{sparsity}.json
```

Each JSON file records:

- Algorithm count (total available algorithms)
- Top-3 algorithm IDs with their latencies
- Detailed configuration information for reproducibility

---

## 12. Experimental Results

### 12.1 Kernel-Level Performance

Kernel-level benchmarks comprehensively evaluate SlideSparse GEMM performance across **6 GPU platforms** (RTX 4090, H100, B200, RTX 5080, GB10, A100), **5 precisions** (FP8, INT8, FP16, BF16, FP4), and **8 sparsity configurations** (2:4, 2:6, 2:8, 2:10, 2:12, 2:14, 2:16, 2:∞). Full results are available in **Appendix A** (square GEMM benchmarks) and **Appendix B** (model-specific GEMM benchmarks) in PDF format.

**Key Observations from Square GEMM Benchmarks (Appendix A):**

- **FP8 Precision**: At large M dimensions (M≥4096), RTX 4090 achieves near-theoretical speedups: 2:4 reaches 1.87-2.08×, 2:6 reaches 1.40-1.51×, 2:8 reaches 1.27-1.37×, and 2:10 reaches 1.18-1.28×. H100 and B200 show similar trends but with slightly lower peak speedups (H100: 2:4 at 1.53-1.73×; B200: 2:4 at 1.51-1.85×). At small M dimensions (M≤1024), overhead dominates and speedups may fall below 1.0× due to insufficient parallelism to hide the transformation cost.

- **INT8 Precision**: B200 demonstrates exceptional INT8 performance with 2:4 achieving up to 6.47× speedup at M=8192 (significantly exceeding the theoretical 2× due to favorable memory access patterns). RTX 4090 shows moderate INT8 speedups (2:4 at 1.49-2.06× for large M). A100 and H100 maintain consistent but modest speedups across sparsity levels.

- **FP16/BF16 Precision**: RTX 4090 achieves strong FP16/BF16 speedups at large M: 2:4 reaches 1.83-2.01×, 2:6 reaches 1.33-1.51×, 2:8 reaches 1.16-1.33×. H100 shows relatively lower speedups for FP16/BF16 (2:4 at 0.97-1.59×) due to its optimized dense GEMM baseline. B200 and RTX 5080 maintain consistent speedup patterns similar to FP8.

- **Low-Sparsity Configurations (2:12 through 2:∞)**: These configurations generally show speedups below 1.0× or marginal improvements at most M dimensions, as expected since the overhead of the sliding transformation exceeds the benefit from reduced computation. However, at very large M (M≥8192), some configurations still achieve modest speedups (e.g., RTX 4090 2:12 at 1.17-1.22× for FP8).

**Key Observations from Model-Specific GEMM Benchmarks (Appendix B):**

- **Model Size Impact**: Larger models (Qwen2.5-7B, Qwen2.5-14B) consistently show higher speedups than smaller models (Llama3.2-1B) due to larger K dimensions providing better parallelism. For example, on RTX 4090 with FP8 at M=16384: Qwen2.5-14B achieves 2:4 at 2.10× while Llama3.2-1B achieves 2:4 at 2.00×.

- **Architecture Differences**: H100 and B200 show more conservative speedup ratios compared to RTX 4090, likely due to their highly optimized dense cuBLASLt baselines. The RTX 5080 and GB10 (DGX Spark) show intermediate performance characteristics.

- **Sparsity-Speedup Correlation**: Across all models and GPUs, the speedup monotonically decreases as sparsity ratio increases (2:4 > 2:6 > 2:8 > 2:10 > ...), closely following theoretical predictions. At large M, most GPUs achieve 85-100% of the theoretical speedup for high-sparsity configurations (2:4, 2:6, 2:8).

### 12.2 End-to-End Performance

End-to-end benchmarks evaluate complete LLM inference throughput with SlideSparse integrated into vLLM, measuring both **Prefill** (compute-bound) and **Decode** (memory-bound) stages. Results cover **6 GPUs**, **5 models**, and **2 quantization schemes** (FP8, INT8). Full results are available in **Appendix C** (Prefill) and **Appendix D** (Decode) in PDF format.

**Prefill Stage Observations (Appendix C):**

- **FP8 Prefill Performance**: At long context lengths (M≥8192), speedups align well with kernel-level results:
  - RTX 4090: Qwen2.5-7B achieves 2:4 at 1.50-1.55×, 2:6 at 1.25-1.30×, 2:8 at 1.17-1.19× for M=8192-32768
  - H100: Qwen2.5-14B achieves 2:4 at 1.30-1.31×, 2:6 at 1.07-1.08×, 2:8 at 1.00-1.01× for M≥4096
  - B200: Shows consistent 2:4 speedups of 1.19-1.28× across all models at long contexts
  - RTX 5080: Demonstrates excellent efficiency with 2:4 at 1.32-1.52×, 2:6 at 1.14-1.24× for M≥2048
  - GB10 (DGX Spark): Achieves moderate speedups (2:4 at 1.04-1.16×) but lower than datacenter GPUs

- **INT8 Prefill Performance**: A100 shows exceptional INT8 Prefill speedups: Qwen2.5-7B achieves 2:4 at 1.63-1.75×, 2:6 at 1.34-1.41×, 2:8 at 1.26-1.34× for M≥2048. RTX 4090 INT8 closely follows FP8 patterns. H100 and B200 show more modest INT8 speedups, generally matching or slightly below their FP8 results.

- **Short Context Behavior**: At small M (M≤1024), Prefill speedups are often below 1.0× or near 1.0× due to insufficient compute to amortize the transformation overhead. This is expected and consistent with kernel-level observations.

**Decode Stage Observations (Appendix D):**

- **Memory-Bound Characteristics**: Decode is inherently memory-bound with small batch sizes (M=64-512), resulting in more modest speedups compared to Prefill:
  - FP8 Decode on RTX 4090: 2:4 achieves 1.04-1.25× across models, with larger models (Qwen2.5-7B, Qwen2.5-14B) showing better speedups
  - H100 Decode: Relatively consistent 1.00-1.30× for 2:4 across all configurations
  - B200 Decode: Shows stable 1.00-1.16× speedups for 2:4

- **INT8 Decode Performance**: Generally shows slightly better speedups than FP8 for Decode:
  - A100: 2:4 achieves 1.09-1.40×, 2:6 achieves 1.06-1.23×, consistently outperforming FP8
  - RTX 4090: INT8 Decode shows 1.03-1.34× for 2:4, comparable to FP8
  - B200: INT8 Decode maintains 1.05-1.36× for 2:4, similar to FP8 patterns

- **High-Concurrency Benefits**: At M=256-512 (higher batch sizes), Decode speedups approach those of Prefill, demonstrating that memory bandwidth bottleneck is alleviated with increased parallelism.

**Cross-Platform Summary:**

- RTX 4090 provides the best absolute speedup ratios due to its less optimized dense baseline, making it ideal for demonstrating SlideSparse benefits
- H100 and B200 show more conservative but consistent improvements, reflecting their highly tuned dense GEMM implementations
- RTX 5080 achieves excellent efficiency, often matching or exceeding RTX 4090 speedups
- GB10 (DGX Spark) provides moderate speedups suitable for edge/embedded deployment scenarios
- A100 demonstrates strong INT8 performance, making it a good choice for INT8 quantized sparse models

### 12.3 Result Data Location

All benchmark results are organized for transparency and reproducibility:

**Human-Readable PDF Tables (Recommended):**
- **Appendix A - Square GEMM Kernel Benchmarks**: `slidesparse/tools/appendix_tables_pdf/appendix_a_square_{precision}.pdf`
- **Appendix B - Model-Specific GEMM Kernel Benchmarks**: `slidesparse/tools/appendix_tables_pdf/appendix_b_model_kernel_{precision}.pdf`
- **Appendix C - End-to-End Prefill Benchmarks**: `slidesparse/tools/appendix_tables_pdf/appendix_c_prefill_{precision}.pdf`
- **Appendix D - End-to-End Decode Benchmarks**: `slidesparse/tools/appendix_tables_pdf/appendix_d_decode_{precision}.pdf`

**CSV Data Files:**
- **Kernel-level CSV data**: `slidesparse/tools/appendix_tables/appendix_a_*.csv` and `appendix_b_*.csv`
- **End-to-end CSV data**: `slidesparse/tools/appendix_tables/appendix_c_*.csv` and `appendix_d_*.csv`

**Raw Benchmark Outputs:**
- **Kernel results**: `slidesparse/benchmark_kernel/kernel_speedup_results/{GPU_ID}/`
- **End-to-end results**: `slidesparse/tools/end2end_speedup_results/{GPU}/`
- **Raw logs**: `slidesparse/tools/throughput_benchmark_results/`

**Result Generation Scripts:**
- **Generate appendix tables**: `slidesparse/tools/generate_appendix_tables.py`
- **Convert CSV to PDF**: `slidesparse/tools/csv_to_pdf_table.py`

### 12.4 Mathematical Correctness Verification

All GEMM operations are validated with `--verify` flags to ensure numerical consistency between cuSPARSELt and cuBLASLt backends:

- **Precision Guarantee**: The only precision loss comes from the pruning operation itself
- **Transformation Correctness**: Slide and compress operations maintain bit-exact results
- **GEMM Equivalence**: cuSPARSELt produces mathematically identical results to cuBLASLt for the same effective computation

---

## 13. Reproducing Results

### 13.1 Complete Reproduction Pipeline

**Kernel-Level Benchmarks:**

```bash
cd slidesparse/benchmark_kernel

# Run complete kernel benchmark suite
python prepare_for_kernel_bench.py --task 1,1,1,1,1,1

# Extract results
python extract_kernel_results.py
```

**End-to-End Benchmarks:**

```bash
cd slidesparse/tools

# Complete pipeline: download, convert, tune, benchmark
python prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1

# Extract results
python extract_end2end_results.py
```

**BitNet Benchmarks:**

```bash
cd slidesparse/tools
python prepare_for_bitnet_bench.py --task 1,1,1,1,1
```

### 13.2 Checkpoint Recovery

Both pipeline scripts implement checkpoint-based recovery for robustness in long-running experiments:

```bash
# Resume from last checkpoint
python prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1 --resume

# The scripts automatically:
# - Save progress after each sub-task
# - Detect incomplete runs
# - Resume from the last successful checkpoint
```

### 13.3 GPU-Specific Considerations

When running on hardware different from our pre-tested configurations:

1. **Automatic Algorithm Search**: The system detects mismatched GPU/driver signatures and triggers algorithm search
2. **Triton Autotuning**: New hardware will trigger Triton's autotune mechanism to find optimal kernel parameters
3. **Result Caching**: All search results are cached to `build/` directories for subsequent runs
4. **Expected Overhead**: First run on new hardware may take several hours due to algorithm search; subsequent runs use cached results

### 13.4 Docker Image Usage

Using our provided Docker images ensures exact environment reproduction:

```bash
# Pull the appropriate image
docker pull <registry>/slidesparse:cuda129-x86_64  # For x86_64 systems
docker pull <registry>/slidesparse:cuda129-aarch64 # For ARM64 systems

# Run with GPU access
docker run --gpus all -v /path/to/models:/models -it <image>

# Inside container, algorithm configurations are pre-loaded
# No additional setup required
```

**Note**: Due to anonymous review requirements, Docker images and Hugging Face model links are temporarily unavailable. Please use the GitHub source code directly during the review period.

---

## 14. Known Limitations and Edge Cases

### 14.1 Benchmark Coverage Gaps

Several edge cases result in incomplete benchmark data:

| Issue                   | Affected Configuration         | Root Cause                                            |
| ----------------------- | ------------------------------ | ----------------------------------------------------- |
| Triton Index Overflow   | M=65536, Qwen2.5-7B            | INT32 indexing limit exceeded for large linear layers |
| OOM                     | 7B/14B models on RTX 4090/5080 | Insufficient VRAM (24GB/16GB) for large models        |
| FP4 Illegal Address     | Specific M,N,K with FP4        | cuBLASLt/cuSPARSELt API limitations                   |
| FP4 Illegal Instruction | Certain dimensions             | API implementation bugs                               |

### 14.2 Hardware Requirements

| Feature                 | Minimum Requirement          |
| ----------------------- | ---------------------------- |
| 2:4 Sparse Acceleration | Ampere (sm80) or later       |
| FP8 Support             | Ada Lovelace (sm89) or later |
| FP4 Support             | Blackwell (sm100) or later   |
| Full torch.compile      | All platforms except GB10    |

### 14.3 Platform-Specific Notes

**DGX Spark GB10 (sm121)**:

- Requires eager mode execution (torch.compile not fully supported)
- Approximately 30% Decode performance penalty compared to compiled mode
- Unique aarch64 architecture requires separate Docker image

### 14.4 Accuracy Considerations

While this work focuses on inference acceleration rather than accuracy preservation:

- **Pruning Impact**: Magnitude pruning at higher sparsity (2:4, 2:6) causes noticeable accuracy degradation
- **2:8 Threshold**: Slight accuracy decline begins around 2:8 sparsity for most models
- **Future Work**: Sparsity-aware training and fine-tuning for accuracy recovery
- **Literature Reference**: According to existing research, 2:6 sparsity can maintain satisfactory quality with proper training

### 14.5 API Stability

- cuBLASLt and cuSPARSELt are closed-source NVIDIA libraries
- Algorithm behavior may change across driver versions
- We recommend using the exact CUDA 12.9 environment specified

---

## 15. Citation

If you find SlideSparse useful in your research, please consider citing our work.

*[Citation information will be provided upon publication]*

---

## 16. License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work builds upon the vLLM inference framework (v0.13.0) and leverages NVIDIA's cuSPARSELt library (v0.8.1) for structured sparsity acceleration. We acknowledge the contributions of the open-source community in advancing efficient LLM inference.

**Key Dependencies:**

- vLLM: High-throughput LLM inference engine
- cuSPARSELt: NVIDIA's structured sparsity library
- cuBLASLt: NVIDIA's high-performance BLAS library
- Triton: OpenAI's GPU programming language
- PyTorch: Deep learning framework

---

## Appendix: Quick Reference

### Environment Variables Summary

| Variable                  | Purpose                  | Example                |
| ------------------------- | ------------------------ | ---------------------- |
| `DISABLE_SLIDESPARSE=1` | Use vLLM native path     | Debugging              |
| `USE_CUBLASLT=1`        | Dense GEMM baseline      | Performance comparison |
| `USE_CUSPARSELT=1`      | Sparse GEMM acceleration | Production             |
| `SPARSITY=2_8`          | Select sparsity level    | 2_4, 2_6, 2_8, 2_10    |
| `SLIDESPARSE_PROFILE=1` | Enable profiling         | Performance analysis   |

### Key Entry Points

| Purpose               | Script Location                                                                                              |
| --------------------- | ------------------------------------------------------------------------------------------------------------ |
| Kernel benchmarks     | `slidesparse/benchmark_kernel/prepare_for_kernel_bench.py`                                                 |
| End-to-end benchmarks | `slidesparse/tools/prepare_for_vllm_bench.py`                                                              |
| BitNet benchmarks     | `slidesparse/tools/prepare_for_bitnet_bench.py`                                                            |
| Model conversion      | `slidesparse/weight_convert/weight_convert_entry.py`                                                       |
| Offline autotuning    | `slidesparse/tools/offline_autotune_algsearch.py`                                                          |
| Model download        | `slidesparse/tools/model_download.py`                                                                      |
| Result extraction     | `slidesparse/tools/extract_end2end_results.py`, `slidesparse/benchmark_kernel/extract_kernel_results.py` |

### Sparsity Configuration Quick Reference

| Sparsity | Zero Ratio | K Expansion | Theoretical Speedup |
| -------- | ---------- | ----------- | ------------------- |
| 2:4      | 50%        | 1.00×      | 2.00×              |
| 2:6      | 33%        | 1.33×      | 1.50×              |
| 2:8      | 25%        | 1.50×      | 1.33×              |
| 2:10     | 20%        | 1.67×      | 1.25×              |
