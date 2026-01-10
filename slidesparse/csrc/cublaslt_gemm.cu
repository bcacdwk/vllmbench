// SPDX-License-Identifier: Apache-2.0
/**
 * cuBLASLt GEMM Implementation for SlideSparse
 * 
 * 设计目标：
 * =========
 * 1. 使用 cuBLASLt API 实现纯矩阵乘法（不带 scale/bias 融合）
 * 2. 支持 FP8E4M3 和 INT8 输入
 * 3. 支持 BF16 和 FP32 输出（inner_dtype）
 * 4. Dequant + bias 由后续 Triton kernel 处理
 * 
 * 计算流程：
 * =========
 * D[M,N] = A[M,K] @ W[N,K]^T
 * 
 * cuBLASLt 配置：
 * ==============
 * - 布局：TN + CCC（W 转置，A 不转置，全列主序）
 * - W[N,K] 行主序 → 声明列主序 [K,N] → opA=T → [N,K]
 * - A[M,K] 行主序 → 声明列主序 [K,M] → opB=N → [K,M]
 * - D[N,M] 列主序结果 → 按行主序读 = [M,N]
 * 
 * 支持的数据类型组合：
 * ===================
 * - input_dtype: "fp8e4m3" (FP8E4M3FN) 或 "int8" (INT8)
 * - inner_dtype: "bf16" (BFloat16) 或 "fp32" (Float32)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

#include <mutex>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>

// ============================================================================
// 错误检查宏
// ============================================================================

#define CHECK_CUDA(expr)                                                       \
  do {                                                                         \
    cudaError_t _status = (expr);                                              \
    if (_status != cudaSuccess) {                                              \
      std::ostringstream _oss;                                                 \
      _oss << "[CUDA Error] " << cudaGetErrorString(_status)                   \
           << " (code " << _status << ") at " << __FILE__ << ":" << __LINE__;  \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

#define CHECK_CUBLASLT(expr)                                                   \
  do {                                                                         \
    cublasStatus_t _status = (expr);                                           \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::ostringstream _oss;                                                 \
      _oss << "[cuBLASLt Error] " << cublasLtGetStatusString(_status)          \
           << " (code " << static_cast<int>(_status) << ") at "                \
           << __FILE__ << ":" << __LINE__;                                     \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// ============================================================================
// 全局 cuBLASLt Handle 管理
// ============================================================================
// 使用 mutex 保护初始化，避免多线程竞争（如 TP 场景）
// 句柄创建后可以跨线程安全使用

static cublasLtHandle_t g_cublaslt_handle = nullptr;
static bool g_cublaslt_initialized = false;
static std::mutex g_cublaslt_init_mutex;

/**
 * 获取全局 cuBLASLt 句柄（懒初始化，线程安全）
 */
static cublasLtHandle_t get_cublaslt_handle() {
  // 双检锁模式
  if (!g_cublaslt_initialized) {
    std::lock_guard<std::mutex> lock(g_cublaslt_init_mutex);
    if (!g_cublaslt_initialized) {
      cublasStatus_t status = cublasLtCreate(&g_cublaslt_handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            "Failed to create cuBLASLt handle: " +
            std::string(cublasLtGetStatusString(status)));
      }
      g_cublaslt_initialized = true;

      // 注册程序退出时的清理函数
      std::atexit([]() {
        if (g_cublaslt_initialized && g_cublaslt_handle) {
          cublasLtDestroy(g_cublaslt_handle);
          g_cublaslt_handle = nullptr;
          g_cublaslt_initialized = false;
        }
      });
    }
  }
  return g_cublaslt_handle;
}

// ============================================================================
// Workspace 管理
// ============================================================================
// cuBLASLt 需要 workspace 来存储中间结果
// 预分配一个合理大小的 workspace，避免每次调用都分配

static constexpr size_t WORKSPACE_SIZE = 32 * 1024 * 1024;  // 32 MB

// 使用 thread_local 避免多线程竞争，每个线程有自己的 workspace
// 注意：这在 TP 场景下可能会增加显存使用
static thread_local void* t_workspace = nullptr;
static thread_local size_t t_workspace_size = 0;

static void* get_workspace(size_t required_size, cudaStream_t stream) {
  if (required_size == 0) {
    return nullptr;
  }
  
  // 确保有足够的 workspace
  size_t alloc_size = std::max(required_size, WORKSPACE_SIZE);
  
  if (t_workspace == nullptr || t_workspace_size < alloc_size) {
    // 释放旧的 workspace
    if (t_workspace != nullptr) {
      cudaFree(t_workspace);
      t_workspace = nullptr;
      t_workspace_size = 0;
    }
    
    // 分配新的 workspace
    cudaError_t err = cudaMalloc(&t_workspace, alloc_size);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          "Failed to allocate cuBLASLt workspace: " +
          std::string(cudaGetErrorString(err)));
    }
    t_workspace_size = alloc_size;
  }
  
  return t_workspace;
}

// ============================================================================
// 数据类型转换辅助函数
// ============================================================================

/**
 * 将字符串 input_dtype 转换为 CUDA 数据类型
 */
static cudaDataType_t get_cuda_input_dtype(const std::string& input_dtype) {
  if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
    return CUDA_R_8F_E4M3;
  } else if (input_dtype == "int8") {
    return CUDA_R_8I;
  } else {
    throw std::invalid_argument(
        "Unsupported input_dtype: " + input_dtype +
        ". Supported: fp8e4m3, int8");
  }
}

/**
 * 将字符串 inner_dtype 转换为 CUDA 数据类型
 */
static cudaDataType_t get_cuda_inner_dtype(const std::string& inner_dtype) {
  if (inner_dtype == "bf16") {
    return CUDA_R_16BF;
  } else if (inner_dtype == "fp32") {
    return CUDA_R_32F;
  } else {
    throw std::invalid_argument(
        "Unsupported inner_dtype: " + inner_dtype +
        ". Supported: bf16, fp32");
  }
}

/**
 * 将字符串 inner_dtype 转换为 PyTorch 数据类型
 */
static at::ScalarType get_torch_inner_dtype(const std::string& inner_dtype) {
  if (inner_dtype == "bf16") {
    return at::kBFloat16;
  } else if (inner_dtype == "fp32") {
    return at::kFloat;
  } else {
    throw std::invalid_argument(
        "Unsupported inner_dtype: " + inner_dtype +
        ". Supported: bf16, fp32");
  }
}

/**
 * 根据 input_dtype 获取计算类型
 */
static cublasComputeType_t get_compute_type(const std::string& input_dtype) {
  if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
    return CUBLAS_COMPUTE_32F;
  } else if (input_dtype == "int8") {
    return CUBLAS_COMPUTE_32I;
  } else {
    throw std::invalid_argument(
        "Unsupported input_dtype: " + input_dtype);
  }
}

/**
 * 验证 PyTorch tensor 的数据类型是否与 input_dtype 匹配
 */
static void validate_tensor_dtype(
    const torch::Tensor& tensor,
    const std::string& input_dtype,
    const std::string& tensor_name) {
  
  if (input_dtype == "fp8e4m3" || input_dtype == "fp8") {
    TORCH_CHECK(tensor.scalar_type() == at::kFloat8_e4m3fn,
                tensor_name, " must be FP8E4M3, got ", tensor.scalar_type());
  } else if (input_dtype == "int8") {
    TORCH_CHECK(tensor.scalar_type() == at::kChar,
                tensor_name, " must be INT8, got ", tensor.scalar_type());
  }
}

// ============================================================================
// cuBLASLt GEMM 实现（无 scale/bias）
// ============================================================================

/**
 * cuBLASLt Matrix Multiplication（纯 GEMM，不带 dequant）
 * 
 * 计算：D[M,N] = A[M,K] @ W[N,K]^T
 * 
 * 参数说明：
 * @param W           权重矩阵 [N, K]，FP8/INT8，行主序存储
 * @param A           输入矩阵 [M, K]，FP8/INT8，行主序存储
 * @param input_dtype 输入数据类型字符串："fp8e4m3" 或 "int8"
 * @param inner_dtype 输出数据类型字符串："bf16" 或 "fp32"
 * @return            输出矩阵 [M, N]，BF16/FP32（inner_dtype）
 * 
 * 实现细节：
 * - W 放在 cuBLASLt 的 A 位置（左矩阵），使用 opA=CUBLAS_OP_T
 * - A 放在 cuBLASLt 的 B 位置（右矩阵），使用 opB=CUBLAS_OP_N
 * - 所有矩阵声明为列主序（实际是行主序内存，利用转置等价）
 * - alpha = 1.0, beta = 0.0（纯矩阵乘法）
 */
torch::Tensor cublaslt_mm(
    torch::Tensor W,               // [N, K] FP8/INT8 行主序
    torch::Tensor A,               // [M, K] FP8/INT8 行主序
    const std::string& input_dtype,  // "fp8e4m3" 或 "int8"
    const std::string& inner_dtype)  // "bf16" 或 "fp32"
{
  // ========== 参数提取 ==========
  const int64_t N = W.size(0);
  const int64_t K_w = W.size(1);
  const int64_t M = A.size(0);
  const int64_t K_a = A.size(1);
  
  // ========== 输入验证 ==========
  // 1. 维度检查
  TORCH_CHECK(W.dim() == 2, "W must be 2D, got ", W.dim(), "D");
  TORCH_CHECK(A.dim() == 2, "A must be 2D, got ", A.dim(), "D");
  TORCH_CHECK(K_w == K_a, "K dimension mismatch: W.K=", K_w, " vs A.K=", K_a);
  
  // 2. 数据类型检查
  validate_tensor_dtype(W, input_dtype, "W");
  validate_tensor_dtype(A, input_dtype, "A");
  
  // 3. 设备检查
  TORCH_CHECK(W.is_cuda(), "W must be on CUDA");
  TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
  
  // 4. Contiguous 检查
  TORCH_CHECK(W.is_contiguous(), "W must be contiguous (row-major)");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  
  // 5. 对齐检查（cuBLASLt FP8 要求 16 字节对齐，INT8 要求 4 字节对齐）
  int64_t align_req = (input_dtype == "fp8e4m3" || input_dtype == "fp8") ? 16 : 4;
  if (M % align_req != 0 || N % align_req != 0 || K_w % align_req != 0) {
    // 暂时允许，但打印警告
    TORCH_WARN_ONCE(
        "cuBLASLt: dimensions (M=", M, ", N=", N, ", K=", K_w,
        ") may not be aligned to ", align_req, ". "
        "This may cause performance degradation or errors on some GPUs.");
  }
  
  // ========== 获取数据类型配置 ==========
  cudaDataType_t cuda_input_dtype = get_cuda_input_dtype(input_dtype);
  cudaDataType_t cuda_inner_dtype = get_cuda_inner_dtype(inner_dtype);
  at::ScalarType torch_inner_dtype = get_torch_inner_dtype(inner_dtype);
  cublasComputeType_t compute_type = get_compute_type(input_dtype);
  
  // Scale 类型始终为 FP32
  cudaDataType_t scale_type = CUDA_R_32F;
  
  // ========== 分配输出 ==========
  // 输出 D[M, N]，与输入在同一设备
  auto options = torch::TensorOptions()
                     .dtype(torch_inner_dtype)
                     .device(A.device());
  torch::Tensor D = torch::empty({M, N}, options);
  
  // ========== 获取 cuBLASLt handle 和 stream ==========
  cublasLtHandle_t handle = get_cublaslt_handle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // ========== 创建 MatmulDesc ==========
  cublasLtMatmulDesc_t matmulDesc = nullptr;
  CHECK_CUBLASLT(cublasLtMatmulDescCreate(
      &matmulDesc, compute_type, scale_type));
  
  // 设置转置操作
  // W 在 cuBLASLt 的 A 位置，需要转置（因为我们用行主序声明为列主序）
  // A 在 cuBLASLt 的 B 位置，不转置
  cublasOperation_t opW = CUBLAS_OP_T;
  cublasOperation_t opA = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
  
  // 默认 epilogue（无 bias）
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
  
  // ========== 创建矩阵布局描述符 ==========
  // 关键技巧：行主序矩阵声明为列主序，配合转置操作实现正确计算
  //
  // W[N,K] 行主序存储：
  //   - 物理内存：N 行，每行 K 个元素，stride = K
  //   - 声明为列主序 [K,N]：K 行，N 列，ld = K
  //   - opA=T 后：[N,K]
  //
  // A[M,K] 行主序存储：
  //   - 物理内存：M 行，每行 K 个元素，stride = K
  //   - 声明为列主序 [K,M]：K 行，M 列，ld = K
  //   - opB=N 后：[K,M]
  //
  // 输出 D：
  //   - cuBLASLt 输出 [N,M] 列主序
  //   - 我们创建 [M,N] 行主序 tensor，ld = N
  //   - 声明为列主序 [N,M]：N 行，M 列，ld = N
  //   - 存储到 [M,N] 行主序内存中
  
  cublasLtMatrixLayout_t layoutW = nullptr;
  cublasLtMatrixLayout_t layoutA = nullptr;
  cublasLtMatrixLayout_t layoutD = nullptr;
  
  // W 布局：声明为列主序 [K, N]，ld = K
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutW, cuda_input_dtype, K_w, N, K_w));
  
  // A 布局：声明为列主序 [K, M]，ld = K
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutA, cuda_input_dtype, K_a, M, K_a));
  
  // D 布局：声明为列主序 [N, M]，ld = N
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutD, cuda_inner_dtype, N, M, N));
  
  // 设置所有矩阵为列主序（这是默认值，但显式设置更清晰）
  cublasLtOrder_t order = CUBLASLT_ORDER_COL;
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(
      layoutD, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
  
  // ========== 创建 Matmul Preference（算法选择） ==========
  cublasLtMatmulPreference_t preference = nullptr;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
  
  // 设置最大 workspace 大小
  size_t max_workspace_size = WORKSPACE_SIZE;
  CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &max_workspace_size, sizeof(max_workspace_size)));
  
  // ========== 获取最优算法 ==========
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedAlgoCount = 0;
  
  cublasStatus_t heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      matmulDesc,
      layoutW,
      layoutA,
      layoutD,  // C 和 D 使用相同布局（in-place）
      layoutD,
      preference,
      1,  // 只请求 1 个算法
      &heuristicResult,
      &returnedAlgoCount);
  
  // 如果启发式搜索失败，使用 algo=NULL（让 cuBLASLt 内部选择）
  const cublasLtMatmulAlgo_t* algo = nullptr;
  size_t workspace_size = 0;
  
  if (heuristic_status == CUBLAS_STATUS_SUCCESS && returnedAlgoCount > 0) {
    algo = &heuristicResult.algo;
    workspace_size = heuristicResult.workspaceSize;
  } else {
    workspace_size = WORKSPACE_SIZE;
  }
  
  // ========== 获取 Workspace ==========
  void* workspace = get_workspace(workspace_size, stream);
  
  // ========== 执行 Matmul ==========
  // alpha = 1.0, beta = 0.0（纯矩阵乘法，scale 由后续 kernel 处理）
  float alpha = 1.0f;
  float beta = 0.0f;
  
  CHECK_CUBLASLT(cublasLtMatmul(
      handle,
      matmulDesc,
      &alpha,
      W.data_ptr(),  // A（左矩阵）= W
      layoutW,
      A.data_ptr(),  // B（右矩阵）= A
      layoutA,
      &beta,
      D.data_ptr(),  // C（用于累加，这里 beta=0 所以不使用）
      layoutD,
      D.data_ptr(),  // D（输出）
      layoutD,
      algo,
      workspace,
      workspace_size,
      stream));
  
  // ========== 清理资源 ==========
  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(layoutW);
  cublasLtMatrixLayoutDestroy(layoutA);
  cublasLtMatrixLayoutDestroy(layoutD);
  cublasLtMatmulDescDestroy(matmulDesc);
  
  return D;
}

// ============================================================================
// FP8 专用入口（保持向后兼容）
// ============================================================================

/**
 * cuBLASLt FP8 Matrix Multiplication（纯 GEMM，不带 dequant）
 * 
 * 这是 FP8 专用的简化接口，inner_dtype 默认为 bf16。
 * 
 * 计算：D[M,N] = A[M,K] @ W[N,K]^T
 * 
 * @param W           权重矩阵 [N, K]，FP8E4M3，行主序存储
 * @param A           输入矩阵 [M, K]，FP8E4M3，行主序存储
 * @param inner_dtype 输出数据类型字符串："bf16"（默认）或 "fp32"
 * @return            输出矩阵 [M, N]，BF16/FP32
 */
torch::Tensor cublaslt_fp8_mm(
    torch::Tensor W,                       // [N, K] FP8 行主序
    torch::Tensor A,                       // [M, K] FP8 行主序
    const std::string& inner_dtype = "bf16")  // "bf16" 或 "fp32"
{
  return cublaslt_mm(W, A, "fp8e4m3", inner_dtype);
}

// ============================================================================
// INT8 专用入口
// ============================================================================

/**
 * cuBLASLt INT8 Matrix Multiplication（纯 GEMM，不带 dequant）
 * 
 * 计算：D[M,N] = A[M,K] @ W[N,K]^T
 * 
 * @param W           权重矩阵 [N, K]，INT8，行主序存储
 * @param A           输入矩阵 [M, K]，INT8，行主序存储
 * @param inner_dtype 输出数据类型字符串："bf16"（默认）或 "fp32"
 * @return            输出矩阵 [M, N]，BF16/FP32
 */
torch::Tensor cublaslt_int8_mm(
    torch::Tensor W,                       // [N, K] INT8 行主序
    torch::Tensor A,                       // [M, K] INT8 行主序
    const std::string& inner_dtype = "bf16")  // "bf16" 或 "fp32"
{
  return cublaslt_mm(W, A, "int8", inner_dtype);
}

// ============================================================================
// PyTorch 绑定
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "cuBLASLt GEMM for SlideSparse (without scale/bias fusion)";
  
  // 注意: cublaslt_mm 是内部实现，不对外导出
  // Python 层应直接调用 cublaslt_fp8_mm 或 cublaslt_int8_mm
  
  // FP8 专用入口
  m.def(
      "cublaslt_fp8_mm",
      &cublaslt_fp8_mm,
      "cuBLASLt FP8 Matrix Multiplication (pure GEMM, no dequant)\n"
      "\n"
      "Computes: D[M,N] = A[M,K] @ W[N,K]^T\n"
      "\n"
      "Args:\n"
      "    W: Weight matrix [N, K], FP8E4M3, row-major\n"
      "    A: Input matrix [M, K], FP8E4M3, row-major\n"
      "    inner_dtype: Output data type ('bf16' or 'fp32', default 'bf16')\n"
      "\n"
      "Returns:\n"
      "    Output matrix [M, N], BF16/FP32\n",
      py::arg("W"),
      py::arg("A"),
      py::arg("inner_dtype") = "bf16");
  
  // INT8 专用入口
  m.def(
      "cublaslt_int8_mm",
      &cublaslt_int8_mm,
      "cuBLASLt INT8 Matrix Multiplication (pure GEMM, no dequant)\n"
      "\n"
      "Computes: D[M,N] = A[M,K] @ W[N,K]^T\n"
      "\n"
      "Args:\n"
      "    W: Weight matrix [N, K], INT8, row-major\n"
      "    A: Input matrix [M, K], INT8, row-major\n"
      "    inner_dtype: Output data type ('bf16' or 'fp32', default 'bf16')\n"
      "\n"
      "Returns:\n"
      "    Output matrix [M, N], BF16/FP32\n",
      py::arg("W"),
      py::arg("A"),
      py::arg("inner_dtype") = "bf16");
}
