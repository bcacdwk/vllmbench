// SPDX-License-Identifier: Apache-2.0
/**
 * cuBLASLt FP8 GEMM Implementation for SlideSparse
 * 
 * 设计目标：
 * =========
 * 1. 使用 cuBLASLt API 实现 FP8 GEMM
 * 2. 支持 Outer Vector Scaling（per-token input scale + per-channel weight scale）
 * 3. 支持 Bias 融合在 epilogue 中
 * 4. 输出 BF16 格式
 * 
 * 计算流程：
 * =========
 * D[M,N] = scale_A[M] * scale_B[N] * (A[M,K] @ W[N,K]^T) + bias[N]
 * 
 * cuBLASLt 配置：
 * ==============
 * - 布局：TN + CCC（W 转置，A 不转置，全列主序）
 * - W[N,K] 行主序 → 声明列主序 [K,N] → opA=T → [N,K]
 * - A[M,K] 行主序 → 声明列主序 [K,M] → opB=N → [K,M]
 * - D[N,M] 列主序结果 → 按行主序读 = [M,N]
 * - Scale 模式：CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
 * - Epilogue：CUBLASLT_EPILOGUE_BIAS（可选）
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
// cuBLASLt FP8 Scaled MM 实现
// ============================================================================

/**
 * cuBLASLt FP8 Scaled Matrix Multiplication
 * 
 * 计算：D[M,N] = alpha * scale_A[M] * scale_B[N] * (A[M,K] @ W[N,K]^T) + bias[N]
 * 
 * 参数说明：
 * @param W        权重矩阵 [N, K]，FP8E4M3，行主序存储
 * @param A        输入矩阵 [M, K]，FP8E4M3，行主序存储
 * @param scale_W  权重 scale [N]，FP32，per-channel
 * @param scale_A  输入 scale [M]，FP32，per-token
 * @param bias     偏置向量 [N]，BF16（可选，传空 tensor 表示无 bias）
 * @param out_dtype输出数据类型
 * @return         输出矩阵 [M, N]，BF16/FP16/FP32
 * 
 * 实现细节：
 * - 使用 Outer Vector Scaling：scale_A 和 scale_B 是向量而非标量
 * - W 放在 cuBLASLt 的 A 位置（左矩阵），使用 opA=CUBLAS_OP_T
 * - A 放在 cuBLASLt 的 B 位置（右矩阵），使用 opB=CUBLAS_OP_N
 * - 所有矩阵声明为列主序（实际是行主序内存，利用转置等价）
 */
torch::Tensor cublaslt_scaled_mm(
    torch::Tensor W,       // [N, K] FP8 行主序
    torch::Tensor A,       // [M, K] FP8 行主序
    torch::Tensor scale_W, // [N] FP32
    torch::Tensor scale_A, // [M] FP32
    torch::Tensor bias,    // [N] BF16 或空
    at::ScalarType out_dtype) {
  
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
  TORCH_CHECK(W.scalar_type() == at::kFloat8_e4m3fn,
              "W must be FP8E4M3, got ", W.scalar_type());
  TORCH_CHECK(A.scalar_type() == at::kFloat8_e4m3fn,
              "A must be FP8E4M3, got ", A.scalar_type());
  TORCH_CHECK(scale_W.scalar_type() == at::kFloat,
              "scale_W must be FP32, got ", scale_W.scalar_type());
  TORCH_CHECK(scale_A.scalar_type() == at::kFloat,
              "scale_A must be FP32, got ", scale_A.scalar_type());
  
  // 3. Scale 维度检查
  TORCH_CHECK(scale_W.numel() == N,
              "scale_W size mismatch: expected ", N, ", got ", scale_W.numel());
  TORCH_CHECK(scale_A.numel() == M,
              "scale_A size mismatch: expected ", M, ", got ", scale_A.numel());
  
  // 4. 设备检查
  TORCH_CHECK(W.is_cuda(), "W must be on CUDA");
  TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
  TORCH_CHECK(scale_W.is_cuda(), "scale_W must be on CUDA");
  TORCH_CHECK(scale_A.is_cuda(), "scale_A must be on CUDA");
  
  // 5. Contiguous 检查
  TORCH_CHECK(W.is_contiguous(), "W must be contiguous (row-major)");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous (row-major)");
  TORCH_CHECK(scale_W.is_contiguous(), "scale_W must be contiguous");
  TORCH_CHECK(scale_A.is_contiguous(), "scale_A must be contiguous");
  
  // 6. Bias 检查（如果有）
  bool has_bias = bias.defined() && bias.numel() > 0;
  if (has_bias) {
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D, got ", bias.dim(), "D");
    TORCH_CHECK(bias.numel() == N,
                "bias size mismatch: expected ", N, ", got ", bias.numel());
    TORCH_CHECK(bias.is_cuda(), "bias must be on CUDA");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  }
  
  // 7. 对齐检查（cuBLASLt FP8 要求 16 字节对齐）
  // 对于 FP8，16 字节 = 16 个元素
  // N, K 必须是 16 的倍数（我们可以保证），M 可能不是（需要上层 padding）
  // TODO: 如果 M 不是 16 的倍数，这里应该做 padding 或者报警告
  if (M % 16 != 0) {
    // 暂时允许，但打印警告
    // 某些 cuBLASLt 算法可能仍能工作，但性能可能不是最优
    TORCH_WARN_ONCE(
        "cuBLASLt FP8: M=", M, " is not aligned to 16. "
        "This may cause performance degradation or errors on some GPUs.");
  }
  
  // ========== 确定输出类型 ==========
  cudaDataType_t cuda_out_dtype;
  torch::ScalarType torch_out_dtype;
  
  switch (out_dtype) {
    case at::kBFloat16:
      cuda_out_dtype = CUDA_R_16BF;
      torch_out_dtype = at::kBFloat16;
      break;
    case at::kHalf:
      cuda_out_dtype = CUDA_R_16F;
      torch_out_dtype = at::kHalf;
      break;
    case at::kFloat:
      cuda_out_dtype = CUDA_R_32F;
      torch_out_dtype = at::kFloat;
      break;
    default:
      TORCH_CHECK(false, "Unsupported output dtype: ", out_dtype);
  }
  
  // ========== 分配输出 ==========
  // 输出 D[M, N]，与输入在同一设备
  auto options = torch::TensorOptions()
                     .dtype(torch_out_dtype)
                     .device(A.device());
  torch::Tensor D = torch::empty({M, N}, options);
  
  // ========== 获取 cuBLASLt handle 和 stream ==========
  cublasLtHandle_t handle = get_cublaslt_handle();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // ========== 创建 MatmulDesc ==========
  // 计算类型：FP32（FP8 GEMM 要求）
  // Scale 类型：FP32
  cublasLtMatmulDesc_t matmulDesc = nullptr;
  CHECK_CUBLASLT(cublasLtMatmulDescCreate(
      &matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  
  // 设置转置操作
  // W 在 cuBLASLt 的 A 位置，需要转置（因为我们用行主序声明为列主序）
  // A 在 cuBLASLt 的 B 位置，不转置
  cublasOperation_t opW = CUBLAS_OP_T;
  cublasOperation_t opA = CUBLAS_OP_N;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));
  
  // 设置 Scale 模式为 Outer Vector Scaling
  // 这样 scale_A 和 scale_B 是向量而非标量
  int32_t scale_mode_A = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  int32_t scale_mode_B = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
      &scale_mode_A, sizeof(scale_mode_A)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
      &scale_mode_B, sizeof(scale_mode_B)));
  
  // 设置 Scale 指针
  // 注意：由于 W 在 cuBLASLt 的 A 位置，scale_W 对应 A_SCALE_POINTER
  //       A 在 cuBLASLt 的 B 位置，scale_A 对应 B_SCALE_POINTER
  const void* scale_W_ptr = scale_W.data_ptr();
  const void* scale_A_ptr = scale_A.data_ptr();
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
      &scale_W_ptr, sizeof(scale_W_ptr)));
  CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
      matmulDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
      &scale_A_ptr, sizeof(scale_A_ptr)));
  
  // 设置 Epilogue（带或不带 bias）
  cublasLtEpilogue_t epilogue;
  if (has_bias) {
    epilogue = CUBLASLT_EPILOGUE_BIAS;
    const void* bias_ptr = bias.data_ptr();
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
        &bias_ptr, sizeof(bias_ptr)));
    
    // 设置 bias 数据类型
    cudaDataType_t bias_dtype;
    if (bias.scalar_type() == at::kBFloat16) {
      bias_dtype = CUDA_R_16BF;
    } else if (bias.scalar_type() == at::kHalf) {
      bias_dtype = CUDA_R_16F;
    } else if (bias.scalar_type() == at::kFloat) {
      bias_dtype = CUDA_R_32F;
    } else {
      TORCH_CHECK(false, "Unsupported bias dtype: ", bias.scalar_type());
    }
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(
        matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
        &bias_dtype, sizeof(bias_dtype)));
  } else {
    epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  }
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
      &layoutW, CUDA_R_8F_E4M3, K_w, N, K_w));
  
  // A 布局：声明为列主序 [K, M]，ld = K
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutA, CUDA_R_8F_E4M3, K_a, M, K_a));
  
  // D 布局：声明为列主序 [N, M]，ld = N
  CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(
      &layoutD, cuda_out_dtype, N, M, N));
  
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
  // 使用启发式搜索获取最优算法
  // 如果 algo=NULL，cuBLASLt 会自动选择（但可能不是最优）
  // 这里我们获取启发式推荐的最优算法
  
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
    // 启发式失败，使用默认算法
    // 注意：algo=NULL 时 cuBLASLt 会内部执行启发式搜索
    workspace_size = WORKSPACE_SIZE;
  }
  
  // ========== 获取 Workspace ==========
  void* workspace = get_workspace(workspace_size, stream);
  
  // ========== 执行 Matmul ==========
  // alpha = 1.0, beta = 0.0（scale 已经在 Outer Vector Scaling 中处理）
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
// PyTorch 绑定
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "cuBLASLt FP8 GEMM for SlideSparse";
  
  m.def(
      "cublaslt_scaled_mm",
      &cublaslt_scaled_mm,
      "cuBLASLt FP8 Scaled Matrix Multiplication with Outer Vector Scaling\n"
      "\n"
      "Computes: D[M,N] = scale_A[M] * scale_W[N] * (A[M,K] @ W[N,K]^T) + bias[N]\n"
      "\n"
      "Args:\n"
      "    W: Weight matrix [N, K], FP8E4M3, row-major\n"
      "    A: Input matrix [M, K], FP8E4M3, row-major\n"
      "    scale_W: Weight scale [N], FP32, per-channel\n"
      "    scale_A: Input scale [M], FP32, per-token\n"
      "    bias: Bias vector [N], BF16 (optional, pass empty tensor for no bias)\n"
      "    out_dtype: Output data type (BFloat16, Half, or Float)\n"
      "\n"
      "Returns:\n"
      "    Output matrix [M, N], dtype as specified\n",
      py::arg("W"),
      py::arg("A"),
      py::arg("scale_W"),
      py::arg("scale_A"),
      py::arg("bias"),
      py::arg("out_dtype"));
}
