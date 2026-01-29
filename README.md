# SlideSparse: Enabling Arbitrary Structured Sparsity for Hardware-Accelerated LLM Inference

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
14. [Citation](#14-citation)
15. [License](#15-license)

---

## 1. Overview

**SlideSparse** is a novel algorithm that enables models with **arbitrary sparsity ratios** (e.g., 2:6, 2:8, 2:10) to leverage NVIDIA's 2:4 structured sparsity hardware acceleration introduced in the Ampere architecture. By performing an offline transformation on model weights along the K dimension, SlideSparse converts weights that satisfy relaxed sparsity constraints into a format compatible with the cuSPARSELt backend, thereby achieving latency reductions proportional to the underlying sparsity ratio.

### Motivation

NVIDIA's Ampere and subsequent GPU architectures provide dedicated hardware support for 2:4 structured sparsity, which theoretically offers a 2× computational speedup for GEMM operations. However, this acceleration is rigidly constrained to exactly 50% sparsity (2 zeros in every 4 consecutive elements). Many practical sparse models exhibit different sparsity patterns (e.g., 33% for 2:6, 25% for 2:8), leaving significant hardware acceleration potential untapped.

SlideSparse addresses this limitation by introducing an **overlapping sliding window transformation** that maps arbitrary Z:L sparsity patterns to the 2:4 format, enabling proportional speedups for any compatible sparsity ratio.

### Core Principle

For a model with Z:L sparsity (Z zeros in every L consecutive elements):

| Original Sparsity | Sparsity Ratio | Expected Latency Reduction | Expected Speedup |
|-------------------|----------------|----------------------------|------------------|
| 2:4 | 50% | 50% | 2.00× |
| 2:6 | 33% | 33% | 1.50× |
| 2:8 | 25% | 25% | 1.33× |
| 2:10 | 20% | 20% | 1.25× |
| 2:∞ (Dense) | 0% | 0% | 1.00× |

The theoretical speedup directly corresponds to the sparsity ratio: X% sparsity enables X% compute skip, achieving the maximum possible benefit from that sparsity level.

---

## 2. Key Contributions

1. **Arbitrary Sparsity Adaptation**: SlideSparse breaks the rigid 50% sparsity constraint of 2:4 hardware, supporting 2:4, 2:6, 2:8, 2:10, and beyond, as well as finer-grained patterns like 1:2, 1:3, 1:4.

2. **Theoretically Optimal Hardware Utilization**: The algorithm ensures that X% sparsity translates to X% latency reduction, fully exploiting the "zero-skipping" capability of the hardware.

3. **Hardware Compatibility**: SlideSparse requires no hardware modifications and directly leverages existing NVIDIA 2:4 sparse tensor cores via the cuSPARSELt library.

4. **End-to-End Integration**: Complete integration with the vLLM inference framework, supporting both FP8 and INT8 quantization, with operator fusion (fused_quant_slide + sparse_GEMM + fused_dequant_bias) to minimize overhead.

5. **Comprehensive Validation**: Extensive benchmarking across 6 GPU platforms, 5 data precisions, 5 model architectures, and 4 sparsity configurations, with all results fully reproducible.

---

## 3. Theoretical Foundation

### 3.1 Relaxed Structured Sparsity (Z:L Sparsity)

SlideSparse defines a generalized sparsity format **Z:L**, where:
- **L** is the window size (number of consecutive elements)
- **Z** is the minimum number of zeros within each window
- **N = L - Z** is the maximum number of non-zero elements per window

The hardware-supported 2:4 sparsity is a special case where Z=2, L=4.

### 3.2 Overlapping Sliding Window Mechanism

The core transformation operates through overlapping sliding windows that decompose the original weight matrix into multiple sub-windows:

```
Original weight sequence (2:8 sparse):  [a₁ 0 a₂ a₃ 0 a₄ a₅ a₆]
                                         ↓ Overlapping window decomposition
Window 1 (positions 0-3):               [a₁  0  a₂  a₃]  → 2:4 compliant
Window 2 (positions 2-5):               [a₂  a₃  0  a₄]  → 2:4 compliant  
Window 3 (positions 4-7):               [0  a₄  a₅  a₆]  → 2:4 compliant

Expanded sequence:  [a₁ 0 a₂ a₃ | a₂ a₃ 0 a₄ | 0 a₄ a₅ a₆]
```

**Key Parameters:**
- Window size = L_target = 4 (for 2:4 hardware)
- Stride = L_target - Z_target = 2
- Number of windows = (L_source - Z_source) / Stride

### 3.3 Dimension Expansion

The sliding operation expands the K dimension of weight matrices:

| Source Sparsity | Expansion Ratio | Original K | Expanded K' |
|-----------------|-----------------|------------|-------------|
| 2:4 | 1.00× | 4096 | 4096 |
| 2:6 | 1.33× | 4096 | 5460 |
| 2:8 | 1.50× | 4096 | 6144 |
| 2:10 | 1.67× | 4096 | 6826 |

After cuSPARSELt 2:4 compression, the K dimension is halved, resulting in final compressed dimensions.

### 3.4 Greedy Residual Allocation

SlideSparse employs a greedy residual allocation strategy that ensures:
1. Complete coverage of all zero elements
2. Non-zero elements are distributed to positions satisfying Z:L constraints
3. Minimization of the expanded total length

---

## 4. Repository Structure

All SlideSparse-specific implementations are contained within the `slidesparse/` directory, with minimal modifications to the vLLM source code.

```
slidesparse/
├── __init__.py
├── utils.py                          # Unified utilities for hardware info, file naming, module loading
│
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
│   └── fused_dequant_bias_triton/    # Fused dequant+bias Triton kernel
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
├── test/                             # Integration tests for vLLM
│   ├── utils.py
│   ├── run_all_suite.sh
│   ├── FP8_vllm/                     # FP8 integration tests
│   │   ├── test_01_bridge.py         # Bridge/registration tests
│   │   ├── test_02_kernel.py         # Kernel correctness tests
│   │   ├── test_03_inference.py      # End-to-end inference tests
│   │   └── test_04_throughput.py     # Throughput validation tests
│   └── INT8_vllm/                    # INT8 integration tests
│       ├── test_01_bridge.py
│       ├── test_02_kernel.py
│       ├── test_03_inference.py
│       └── test_04_throughput.py
│
└── docs/                             # Technical documentation
    ├── framework_overview.md         # vLLM framework overview
    ├── framework_vllmcore.md         # vLLM core architecture
    ├── framework_lineargemm.md       # Linear layer and GEMM details
    ├── framework_slidesparse.md      # SlideSparse implementation guide
    └── fp8_gemm_integration_analysis.md  # FP8 GEMM integration analysis
```

### vLLM Source Code Modifications

SlideSparse maintains **minimal invasiveness** to the vLLM codebase. Only two files are modified:

1. **`vllm/model_executor/layers/quantization/slidesparse.py`** (New file)
   - Bridge file that imports SlideSparse modules into vLLM's quantization framework
   - Exports configuration functions and LinearMethod wrappers

2. **`vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py`** (Modified)
   - Added SlideSparse wrapping in the `get_scheme()` method
   - Transparently hooks FP8 and INT8 schemes when SlideSparse is enabled

---

## 5. Supported Configurations

### 5.1 GPU Platforms

SlideSparse has been validated across 6 GPU platforms spanning multiple architectures:

| GPU | Architecture | Compute Capability | Memory | Platform Type |
|-----|--------------|-------------------|--------|---------------|
| NVIDIA A100 80GB PCIe | Ampere | sm80 | 80 GB HBM2e | Cloud/Datacenter |
| NVIDIA H100 80GB PCIe | Hopper | sm90 | 80 GB HBM3 | Cloud/Datacenter |
| NVIDIA B200 180GB SXM | Blackwell | sm100 | 180 GB HBM3e | Cloud/Datacenter |
| NVIDIA RTX 4090 | Ada Lovelace | sm89 | 24 GB GDDR6X | Consumer |
| NVIDIA RTX 5080 | Blackwell | sm120 | 16 GB GDDR7 | Consumer |
| NVIDIA DGX Spark (GB10) | Blackwell | sm121 | 128 GB | Embedded/Mobile (aarch64) |

The implementation supports both **x86_64** and **aarch64** hardware architectures.

### 5.2 Data Precision Types

Kernel-level benchmarking covers 5 precision types:

| Precision | Use Case | cuBLASLt | cuSPARSELt | Notes |
|-----------|----------|----------|------------|-------|
| FP16 | General inference | ✓ | ✓ | Most common precision |
| BF16 | Training-oriented | ✓ | ✓ | Better dynamic range |
| INT8 | Quantized inference | ✓ | ✓ | W8A8 quantization |
| FP8 E4M3 | Low-precision inference | ✓ | ✓ | Ada Lovelace+ (sm89+) |
| FP4 E2M1 | Future trend | ✓ | ✓ | Blackwell+ (sm100+) |

### 5.3 Sparsity Configurations

| Sparsity | Zero Ratio | K Expansion | Supported |
|----------|------------|-------------|-----------|
| 2:4 | 50% | 1.00× | ✓ |
| 2:6 | 33% | 1.33× | ✓ |
| 2:8 | 25% | 1.50× | ✓ |
| 2:10 | 20% | 1.67× | ✓ |
| 2:∞ (Dense) | 0% | 2.00× | ✓ (Experimental, see note) |

**Note on 2:∞ Dense Mode**: This experimental mode demonstrates the mathematical soundness of SlideSparse by inserting zeros into a fully dense weight matrix and then using sparsity to skip them. The 2× K expansion (doubling K, then halving via 2:4 compression) results in a theoretical 1.00× speedup (no change), validating that the algorithm correctly handles the boundary case where no actual zeros exist in the original weights.

### 5.4 Model Support

End-to-end testing covers 5 model architectures in FP8 and INT8 quantization:

| Model | Parameters | Architecture | FP8 | INT8 |
|-------|------------|--------------|-----|------|
| Llama3.2-1B | 1B | Llama | ✓ | ✓ |
| Llama3.2-3B | 3B | Llama | ✓ | ✓ |
| Qwen2.5-7B | 7B | Qwen | ✓ | ✓ |
| Qwen2.5-14B | 14B | Qwen | ✓ | ✓ |
| BitNet1.58-2B | 2B | BitNet (Ternary) | ✓ | ✓ |

With 4 sparsity configurations (2:4, 2:6, 2:8, 2:10) applied to each model, a total of **50 model variants** (10 base + 40 sparse) are evaluated.

---

## 6. Getting Started

### 6.1 Environment Setup

SlideSparse is integrated with vLLM v0.13.0 and requires CUDA 12.9+ with cuSPARSELt support.

**Docker Environment (Recommended):**

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

### 6.2 Environment Variables

SlideSparse behavior is controlled through environment variables:

| Variable | Values | Description |
|----------|--------|-------------|
| `DISABLE_SLIDESPARSE` | 0/1 | Disable SlideSparse entirely, use vLLM native path |
| `USE_CUBLASLT` | 0/1 | Enable cuBLASLt dense GEMM backend |
| `USE_CUSPARSELT` | 0/1 | Enable cuSPARSELt sparse GEMM backend |
| `SPARSITY` | 2_4, 2_6, 2_8, 2_10, 2_inf | Sparsity configuration (cuSPARSELt only). 2_inf is experimental. |
| `INNER_DTYPE_32` | 0/1 | Use high-precision accumulation (FP8→FP32) |
| `SLIDESPARSE_PROFILE` | 0/1 | Enable kernel timing diagnostics |

**Example Usage:**

```bash
# Run with cuSPARSELt backend at 2:8 sparsity
USE_CUSPARSELT=1 SPARSITY=2_8 vllm serve model_path

# Run with cuBLASLt baseline
USE_CUBLASLT=1 vllm serve model_path

# Disable SlideSparse completely
DISABLE_SLIDESPARSE=1 vllm serve model_path
```

### 6.3 Quick Start

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

---

## 7. Kernel-Level Benchmarking

### 7.1 Overview

Kernel-level benchmarks measure raw GEMM performance isolated from end-to-end inference overhead. Two benchmark modes are provided:

1. **Square Mode**: Tests M=N=K configurations for systematic performance analysis
2. **Model Mode**: Tests actual (N, K) dimensions from target model linear layers

### 7.2 Benchmark Entry Points

**Unified Entry Script:**

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

**Individual Benchmark:**

```bash
python benchmark_entry.py --backend cublaslt --mode square --dtype fp8e4m3 --M 1024,2048,4096
python benchmark_entry.py --backend cusparselt --mode model --sparsity 2_8 --dtype int8
```

### 7.3 M Dimension Configurations

| Mode | M Values |
|------|----------|
| Square | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 |
| Model | 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 |

### 7.4 Benchmark Metrics

Each benchmark records:
- **Latency (μs)**: Average execution time per GEMM operation
- **Throughput (TOPS)**: Tera-operations per second
- **Speedup**: Ratio relative to cuBLASLt dense baseline
- **Algorithm Configuration**: Top-3 algorithm IDs for reproducibility

### 7.5 Result Extraction

```bash
# Extract and analyze kernel benchmark results
python extract_kernel_results.py

# Results are saved to kernel_speedup_results/{GPU_ID}/
```

---

## 8. End-to-End Benchmarking

### 8.1 Overview

End-to-end benchmarks measure actual inference throughput (tokens/s) using the methodology of the official `vllm bench throughput` command. The custom `throughput_benchmark.py` script wraps this functionality with additional automation and result collection for SlideSparse experiments.

### 8.2 Benchmark Modes

**Prefill Mode (Compute-Bound):**
- Controls: `M = max_num_seqs × prompt_length`
- Minimizes Decode overhead by setting `output_len=1`
- M values: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536

**Decode Mode (Memory-Bound):**
- Controls: `M = max_num_seqs`
- Minimizes Prefill overhead with single-token prompts
- M values: 64, 128, 256, 512

### 8.3 Benchmark Entry Points

**Unified Pipeline:**

```bash
cd slidesparse/tools

# Complete end-to-end benchmark pipeline
python prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1

# Task breakdown:
# Task 1: Model download (8 base models)
# Task 2: Model conversion (generate 2_4, 2_6, 2_8, 2_10 variants)
# Task 3: Offline coarse autotuning (cuBLASLt + Triton quant_only)
# Task 4: Offline fine autotuning (Triton dequant + quant_slide + cuSPARSELt)
# Task 5: Quick validation benchmark
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

### 8.4 BitNet-Specific Benchmarking

Due to BitNet's unique requirements (1.58-bit ternary weights), a dedicated script handles its benchmarking:

```bash
python prepare_for_bitnet_bench.py --task 1,1,1,1,1
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
# - Absolute throughput (tokens/s)
# - Speedup relative to cuBLASLt baseline
```

---

## 9. Weight Conversion Pipeline

### 9.1 Overview

The weight conversion pipeline transforms dense or coarsely-sparse models into SlideSparse-compatible formats through a three-stage process:

```
Original Model     →  [Prune]  →  [Slide]   →  [Compress]  →  SlideSparse Model
Shape: [N, K]         [N, K]      [N, K']       [N, K'/2]
```

### 9.2 Pipeline Stages

**Stage 1: Pruning (`prune.py`)** — Shape: `[N, K] → [N, K]`
- Applies Z:L sparsity constraints through magnitude or random pruning
- Supports FP8, INT8, and BF16 (BitNet) input models
- Outputs sparsified weights maintaining original dimensions

**Stage 2: Sliding (`slide.py`)** — Shape: `[N, K] → [N, K']` where `K' = K × expand_ratio`
- Transforms Z:L sparse weights to 2:4 compliant format
- Expands K dimension according to the expansion ratio
- Ensures proper padding for cuSPARSELt alignment (must be divisible by 4L)

**Stage 3: Compression (`compress.py`)** — Shape: `[N, K'] → [N, K'/2]`
- Calls cuSPARSELt compression routines
- Reduces K dimension by half (2:4 compression)
- Generates sparse metadata for hardware decompression

### 9.3 Usage

**Complete Conversion:**

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

Converted models are saved in the standard safetensors format:

```
{Model}-SlideSparse-{Z}_{L}/
├── model.safetensors           # Converted weights
├── config.json                 # Original model config
├── tokenizer.json              # Tokenizer files
└── slidesparse_config.json     # SlideSparse metadata
    ├── sparsity: "2:8"
    ├── expand_ratio: 1.5
    ├── original_shapes: {...}
    └── compressed_shapes: {...}
```

---

## 10. vLLM Integration Architecture

### 10.1 Integration Design Philosophy

SlideSparse follows a **minimal invasion principle**, modifying only two files in the vLLM source while implementing all functionality in the external `slidesparse/` module.

### 10.2 Architecture Overview

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

### 10.3 Kernel Fusion Strategy

To minimize overhead from the sliding transformation, SlideSparse employs operator fusion:

**Fused Quant + Slide Kernel:**
- Combines FP16/BF16 → FP8/INT8 quantization with sliding window expansion
- Implemented as optimized Triton kernel with double buffering
- Achieves near-theoretical memory bandwidth utilization

**Fused Dequant + Bias Kernel:**
- Combines INT32/FP32 → BF16/FP16 dequantization with bias addition
- Shared between cuBLASLt and cuSPARSELt paths

### 10.4 torch.compile Compatibility

To support vLLM's full graph compilation, SlideSparse registers custom operators via `torch.library`:

- GEMM wrappers provide both real and fake implementations
- Triton kernels are registered with proper signatures
- Algorithm lookup is made compile-friendly

**Note:** DGX Spark GB10 requires eager mode due to hardware newness, with approximately 30% Decode performance impact.

### 10.5 Matrix Layout Configuration

After extensive testing of 8 GEMM layout configurations, SlideSparse standardizes on **TN+CC+C** layout:

```
D[M,N]_row = D[N,M]_col = W[K,N]_col^T × A[K,M]_col
           = W[N,K]_row^T × A[M,K]_row
```

Where:
- **T/N**: Transpose/No-transpose for operands
- **C/R**: Column/Row major for memory layout
- First operand (Weight) is transposed
- Second operand (Activation) is not transposed
- Both inputs are column-major, output is column-major (equivalent to row-major with transposed indices)

---

## 11. Algorithm Search and Optimization

### 11.1 Algorithm Search Overview

Both cuBLASLt and cuSPARSELt support multiple algorithm implementations for each GEMM configuration. SlideSparse performs offline search to identify optimal algorithms for each (M, N, K) combination.

### 11.2 Search Methodology

**Search Process:**
1. Enumerate all (N, K) pairs from target model linear layers
2. For each M value in the test range, enumerate available algorithm IDs
3. Execute multiple iterations (100+) with proper warmup
4. Record latency and select top performers

**Layout Search:**
- Tests 8 layout configurations (NN, NT, TN, TT with RC/CR variants)
- Identifies performance-stable layouts
- Validates consistency across problem sizes

### 11.3 Search Scripts

```bash
cd slidesparse/search

# cuBLASLt algorithm search
cd cuBLASLt_AlgSearch
python alg_search.py --model qwen2.5-7b --dtype fp8

# cuSPARSELt algorithm search
cd cuSPARSELt_AlgSearch
python alg_search.py --model qwen2.5-7b --dtype fp8 --sparsity 2_8
```

### 11.4 Online Algorithm Lookup

The `AlgorithmConfigManager` class in `gemm_wrapper.py` provides O(1) runtime lookup:

1. Loads precomputed JSON configurations at startup
2. Matches runtime (model, N, K, M_range) to optimal algorithm
3. Falls back to heuristic selection when no match is found

### 11.5 Search Results Location

```
search/
├── cuBLASLt_AlgSearch/alg_search_results/
│   └── {GPU}_{Model}_{dtype}.json
└── cuSPARSELt_AlgSearch/alg_search_results/
    └── {GPU}_{Model}_{dtype}_{sparsity}.json
```

---

## 12. Experimental Results

### 12.1 Kernel-Level Performance

Kernel benchmarks demonstrate consistent speedups proportional to sparsity ratio:

| Sparsity | Theoretical Speedup | A100 | H100 | B200 |
|----------|---------------------|------|------|------|
| 2:4 | 2.00× | ~1.8-2.0× | ~1.9-2.0× | ~2.0× |
| 2:6 | 1.50× | ~1.4-1.5× | ~1.4-1.5× | ~1.5× |
| 2:8 | 1.33× | ~1.25-1.33× | ~1.3-1.33× | ~1.33× |
| 2:10 | 1.25× | ~1.2-1.25× | ~1.2-1.25× | ~1.25× |

### 12.2 End-to-End Performance

End-to-end throughput improvements vary by model size and inference stage:

**Prefill Stage (Compute-Bound):**
- Larger models show more consistent speedups
- Speedups closely track kernel-level improvements

**Decode Stage (Memory-Bound):**
- Speedups are more modest due to memory bandwidth limitations
- Still demonstrates meaningful improvements for high-concurrency scenarios

### 12.3 Known Limitations

Several edge cases result in incomplete benchmark data:

1. **Triton Index Overflow**: M=65536 with Qwen2.5-7B exceeds INT32 indexing limits
2. **OOM on Consumer GPUs**: 7B/14B models exceed memory on RTX 4090/5080
3. **FP4 API Issues**: Certain (M, N, K) configurations trigger illegal memory access
4. **Hardware Requirements**: FP8 requires Ada Lovelace+; FP4 requires Blackwell+

### 12.4 Mathematical Correctness

All GEMM operations are validated with `--verify` flag to ensure numerical consistency between cuSPARSELt and cuBLASLt backends. The only precision loss comes from the pruning operation itself; all subsequent transformations (slide, compress, GEMM) maintain bit-exact results.

---

## 13. Reproducing Results

### 13.1 Complete Reproduction Pipeline

```bash
# 1. Kernel-level benchmarks
cd slidesparse/benchmark_kernel
python prepare_for_kernel_bench.py --task 1,1,1,1,1,1

# 2. End-to-end benchmarks
cd slidesparse/tools
python prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1

# 3. Extract and analyze results
python extract_kernel_results.py
python extract_end2end_results.py
```

### 13.2 Checkpoint Recovery

Both pipeline scripts support checkpoint-based recovery for long-running experiments:

```bash
# Resume from last checkpoint
python prepare_for_vllm_bench.py --task 1,1,1,1,1,1,1 --resume
```

### 13.3 GPU-Specific Considerations

When running on a different GPU than the pre-computed configurations:

1. Algorithm search results are automatically regenerated
2. Triton autotuning will find optimal parameters for the new hardware
3. Results are cached for subsequent runs

---

## 14. Citation

If you find SlideSparse useful in your research, please consider citing our work.

---

## 15. License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This work builds upon the vLLM inference framework and leverages NVIDIA's cuSPARSELt library for structured sparsity acceleration. We acknowledge the contributions of the open-source community in advancing efficient LLM inference.
