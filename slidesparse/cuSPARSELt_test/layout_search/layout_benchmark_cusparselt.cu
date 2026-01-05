// cuSPARSELt Layout 性能测试
// 提供 Python API，测试不同 layout 组合的 SpMM 性能
// 支持的数据类型：int8, fp8e4m3

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

// ===== 工具宏 =====
#define CHECK_CUDA_ERR(expr)                                                   \
  do {                                                                         \
    cudaError_t _status = (expr);                                              \
    if (_status != cudaSuccess) {                                              \
      std::ostringstream _oss;                                                 \
      _oss << "CUDA 调用失败: " << cudaGetErrorString(_status)                 \
           << " (code " << _status << ")";                                     \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

#define CHECK_CUSPARSE_ERR(expr)                                               \
  do {                                                                         \
    cusparseStatus_t _status = (expr);                                         \
    if (_status != CUSPARSE_STATUS_SUCCESS) {                                  \
      std::ostringstream _oss;                                                 \
      _oss << "cuSPARSELt 调用失败: " << cusparseLtGetErrorString(_status)     \
           << " (code " << static_cast<int>(_status) << ")";                   \
      throw std::runtime_error(_oss.str());                                    \
    }                                                                          \
  } while (0)

// ===== dtype 相关 =====
static cudaDataType to_cuda_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUDA_R_8I;
  if (dtype == "fp8e4m3") return CUDA_R_8F_E4M3;
  throw std::invalid_argument("不支持的数据类型: " + dtype + "。支持: int8, fp8e4m3");
}

static cudaDataType cuda_out_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUDA_R_32I;
  if (dtype == "fp8e4m3") return CUDA_R_32F;
  throw std::invalid_argument("不支持的数据类型: " + dtype);
}

static cusparseComputeType compute_type_from_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUSPARSE_COMPUTE_32I;
  if (dtype == "fp8e4m3") return CUSPARSE_COMPUTE_32F;
  throw std::invalid_argument("不支持的数据类型: " + dtype);
}

static int dtype_size(const std::string &dtype) {
  if (dtype == "int8" || dtype == "fp8e4m3") return 1;
  return 1;
}

static int out_dtype_size(const std::string &dtype) {
  if (dtype == "int8") return 4;  // int32
  if (dtype == "fp8e4m3") return 4;  // float32
  return 4;
}

// ===== 解析 layout 字符串 =====
static cusparseOperation_t parse_op(const std::string &op) {
  if (op == "T") return CUSPARSE_OPERATION_TRANSPOSE;
  if (op == "N") return CUSPARSE_OPERATION_NON_TRANSPOSE;
  throw std::invalid_argument("无效的操作: " + op + "。支持: T, N");
}

static cusparseOrder_t parse_order(const std::string &order) {
  if (order == "Col") return CUSPARSE_ORDER_COL;
  if (order == "Row") return CUSPARSE_ORDER_ROW;
  throw std::invalid_argument("无效的顺序: " + order + "。支持: Col, Row");
}

// ===== 算法结果 =====
struct AlgResult {
  int alg_id;
  float lat_us;
  float tops;
};

// ===== test_layout 主函数 =====
py::dict test_layout(
    int64_t N, int64_t K, int64_t M,
    const std::string &opW_str,     // "T" or "N"
    const std::string &opA_str,     // "T" or "N"
    const std::string &orderW_str,  // "Col" or "Row"
    const std::string &orderA_str,  // "Col" or "Row"
    const std::string &orderR_str,  // "Col" or "Row" (输出矩阵顺序)
    const std::string &dtype,
    int warmup,
    int repeat
) {
  py::dict result;
  result["supported"] = false;
  result["max_alg_id"] = -1;
  result["top3"] = py::list();

  // 解析 layout 参数
  cusparseOperation_t opW = parse_op(opW_str);
  cusparseOperation_t opA = parse_op(opA_str);
  cusparseOrder_t orderW = parse_order(orderW_str);
  cusparseOrder_t orderA = parse_order(orderA_str);
  cusparseOrder_t orderR = parse_order(orderR_str);

  cudaDataType type_AB = to_cuda_dtype(dtype);
  cudaDataType type_C = cuda_out_dtype(dtype);
  cusparseComputeType comp_type = compute_type_from_dtype(dtype);
  unsigned alignment = 16;

  // 计算矩阵维度
  bool isW_transposed = (opW == CUSPARSE_OPERATION_TRANSPOSE);
  bool isA_transposed = (opA == CUSPARSE_OPERATION_TRANSPOSE);
  bool isW_rowmajor = (orderW == CUSPARSE_ORDER_ROW);
  bool isA_rowmajor = (orderA == CUSPARSE_ORDER_ROW);
  bool isR_rowmajor = (orderR == CUSPARSE_ORDER_ROW);

  // W[N,K]: 不转置时存储为[N,K]，转置时存储为[K,N]
  int64_t num_W_rows = isW_transposed ? K : N;
  int64_t num_W_cols = isW_transposed ? N : K;
  // A[M,K]: 转置时存储为[M,K]，不转置时存储为[K,M]
  int64_t num_A_rows = isA_transposed ? M : K;
  int64_t num_A_cols = isA_transposed ? K : M;
  // R[N,M]
  int64_t num_R_rows = N;
  int64_t num_R_cols = M;

  // Leading dimensions
  int64_t ldw = isW_rowmajor ? num_W_cols : num_W_rows;
  int64_t lda = isA_rowmajor ? num_A_cols : num_A_rows;
  int64_t ldr = isR_rowmajor ? num_R_cols : num_R_rows;

  // 元素数量
  int64_t W_height = isW_rowmajor ? num_W_rows : num_W_cols;
  int64_t A_height = isA_rowmajor ? num_A_rows : num_A_cols;
  int64_t R_height = isR_rowmajor ? num_R_rows : num_R_cols;

  size_t W_elems = static_cast<size_t>(W_height) * ldw;
  size_t A_elems = static_cast<size_t>(A_height) * lda;
  size_t R_elems = static_cast<size_t>(R_height) * ldr;

  size_t W_size = W_elems * dtype_size(dtype);
  size_t A_size = A_elems * dtype_size(dtype);
  size_t R_size = R_elems * out_dtype_size(dtype);

  // 分配设备内存
  void *dW = nullptr, *dA = nullptr, *dR = nullptr;
  int *d_valid = nullptr;
  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  CHECK_CUDA_ERR(cudaMalloc(&dW, W_size));
  CHECK_CUDA_ERR(cudaMalloc(&dA, A_size));
  CHECK_CUDA_ERR(cudaMalloc(&dR, R_size));
  CHECK_CUDA_ERR(cudaMalloc(&d_valid, sizeof(int)));

  // 随机初始化数据
  {
    std::vector<int8_t> hW(W_elems), hA(A_elems);
    for (size_t i = 0; i < W_elems; ++i) hW[i] = static_cast<int8_t>(rand() % 256 - 128);
    for (size_t i = 0; i < A_elems; ++i) hA[i] = static_cast<int8_t>(rand() % 256 - 128);
    CHECK_CUDA_ERR(cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
  }
  CHECK_CUDA_ERR(cudaMemset(dR, 0, R_size));

  // 初始化 cuSPARSELt
  cusparseLtHandle_t handle;
  cusparseStatus_t status = cusparseLtInit(&handle);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  cusparseLtMatDescriptor_t matW, matA, matR;
  cusparseLtMatmulDescriptor_t matmul;

  // 稀疏矩阵 W 描述符
  status = cusparseLtStructuredDescriptorInit(
      &handle, &matW, num_W_rows, num_W_cols, ldw, alignment,
      type_AB, orderW, CUSPARSELT_SPARSITY_50_PERCENT);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 稠密矩阵 A 描述符
  status = cusparseLtDenseDescriptorInit(
      &handle, &matA, num_A_rows, num_A_cols, lda, alignment,
      type_AB, orderA);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 输出矩阵 R 描述符
  status = cusparseLtDenseDescriptorInit(
      &handle, &matR, num_R_rows, num_R_cols, ldr, alignment,
      type_C, orderR);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 矩阵乘法描述符
  status = cusparseLtMatmulDescriptorInit(
      &handle, &matmul, opW, opA, &matW, &matA, &matR, &matR, comp_type);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 设置稀疏矩阵指针
  status = cusparseLtMatmulDescSetAttribute(
      &handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dW, sizeof(dW));
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 剪枝
  status = cusparseLtSpMMAPrune(&handle, &matmul, dW, dW, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 检查剪枝
  status = cusparseLtSpMMAPruneCheck(&handle, &matmul, dW, d_valid, stream);
  if (status != CUSPARSE_STATUS_SUCCESS) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  int is_valid = 0;
  CHECK_CUDA_ERR(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

  if (is_valid != 0) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  // 获取最大算法 ID
  int max_alg_id = -1;
  {
    cusparseLtMatmulAlgSelection_t alg_sel_tmp;
    status = cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel_tmp, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status == CUSPARSE_STATUS_SUCCESS) {
      cusparseLtMatmulAlgGetAttribute(
          &handle, &alg_sel_tmp, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
          &max_alg_id, sizeof(max_alg_id));
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
    }
  }

  if (max_alg_id < 0) {
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
    return result;
  }

  result["max_alg_id"] = max_alg_id;

  // CUDA events for timing
  cudaEvent_t start_event, stop_event;
  CHECK_CUDA_ERR(cudaEventCreate(&start_event));
  CHECK_CUDA_ERR(cudaEventCreate(&stop_event));

  std::vector<AlgResult> alg_results;
  float alpha = 1.0f, beta = 0.0f;

  // 遍历所有算法
  for (int alg_id = 0; alg_id <= max_alg_id; ++alg_id) {
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) continue;

    status = cusparseLtMatmulAlgSetAttribute(
        &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &alg_id, sizeof(alg_id));
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      continue;
    }

    cusparseLtMatmulPlan_t plan;
    status = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      continue;
    }

    // 压缩稀疏矩阵
    size_t compressed_size = 0, compressed_buffer_size = 0;
    CHECK_CUSPARSE_ERR(cusparseLtSpMMACompressedSize(&handle, &plan, &compressed_size, &compressed_buffer_size));

    void *dW_compressed = nullptr, *dW_compressedBuffer = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&dW_compressed, compressed_size));
    if (compressed_buffer_size > 0) {
      CHECK_CUDA_ERR(cudaMalloc(&dW_compressedBuffer, compressed_buffer_size));
    }

    status = cusparseLtSpMMACompress(&handle, &plan, dW, dW_compressed, dW_compressedBuffer, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      cudaFree(dW_compressed);
      if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
      cusparseLtMatmulPlanDestroy(&plan);
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      continue;
    }

    // Workspace
    size_t workspace_size = 0;
    cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
    void *d_workspace = nullptr;
    if (workspace_size > 0) {
      CHECK_CUDA_ERR(cudaMalloc(&d_workspace, workspace_size));
    }

    // Warmup
    for (int w = 0; w < warmup; ++w) {
      status = cusparseLtMatmul(&handle, &plan, &alpha, dW_compressed, dA, &beta, dR, dR, d_workspace, &stream, 1);
      if (status != CUSPARSE_STATUS_SUCCESS) break;
    }
    CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

    if (status != CUSPARSE_STATUS_SUCCESS) {
      if (d_workspace) cudaFree(d_workspace);
      cudaFree(dW_compressed);
      if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
      cusparseLtMatmulPlanDestroy(&plan);
      cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
      continue;
    }

    // Benchmark
    CHECK_CUDA_ERR(cudaEventRecord(start_event, stream));
    for (int r = 0; r < repeat; ++r) {
      cusparseLtMatmul(&handle, &plan, &alpha, dW_compressed, dA, &beta, dR, dR, d_workspace, &stream, 1);
    }
    CHECK_CUDA_ERR(cudaEventRecord(stop_event, stream));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop_event));

    float total_ms = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&total_ms, start_event, stop_event));
    float avg_us = (total_ms * 1000.0f) / repeat;

    // 计算 TOPS
    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    double tops = (flops / (avg_us * 1e-6)) / 1e12;

    alg_results.push_back({alg_id, avg_us, static_cast<float>(tops)});

    // 清理
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(dW_compressed);
    if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
  }

  // 按吞吐量排序
  std::sort(alg_results.begin(), alg_results.end(),
            [](const AlgResult &a, const AlgResult &b) { return a.tops > b.tops; });

  // 构建 top3 结果
  py::list top3_list;
  int fill_count = std::min(3, static_cast<int>(alg_results.size()));
  for (int i = 0; i < fill_count; ++i) {
    py::tuple entry = py::make_tuple(
        alg_results[i].alg_id,
        alg_results[i].lat_us,
        alg_results[i].tops
    );
    top3_list.append(entry);
  }

  result["supported"] = !alg_results.empty();
  result["top3"] = top3_list;

  // 清理
  CHECK_CUDA_ERR(cudaEventDestroy(start_event));
  CHECK_CUDA_ERR(cudaEventDestroy(stop_event));
  cusparseLtMatDescriptorDestroy(&matW);
  cusparseLtMatDescriptorDestroy(&matA);
  cusparseLtMatDescriptorDestroy(&matR);
  cusparseLtDestroy(&handle);
  cudaFree(dW);
  cudaFree(dA);
  cudaFree(dR);
  cudaFree(d_valid);

  return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("test_layout", &test_layout, "测试单个 layout 配置的 SpMM 性能",
        py::arg("N"), py::arg("K"), py::arg("M"),
        py::arg("opW"), py::arg("opA"),
        py::arg("orderW"), py::arg("orderA"), py::arg("orderR"),
        py::arg("dtype"),
        py::arg("warmup") = 10,
        py::arg("repeat") = 50);
}
