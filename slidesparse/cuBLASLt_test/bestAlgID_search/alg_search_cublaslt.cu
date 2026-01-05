// cuBLASLt 算法离线搜索
// 固定的layout: 权重W在左，T/N + C/C GEMM，输出矩阵order固定为 Column 主序
// C[N,M]_col = W[N,K]^T_col * A[K,M]_col
// 支持的数据类型：int8, fp8e4m3
//
// =====================================
// cuBLASLt API 约束 (基于官方文档):
// =====================================
//
// INT8 IMMA 内核要求:
//   - 所有矩阵指针必须 4 字节对齐 (16 字节对齐性能更佳)
//   - Leading dimensions 必须是 4 的倍数
//   - 只支持 TN 格式: A 必须转置, B 不转置
//   - m 和 k 必须是 4 的倍数
//   - scaleType 为 CUDA_R_32I 时, alpha/beta 只能是 0 或 1
//   - computeType: CUBLAS_COMPUTE_32I
//
// FP8 内核要求:
//   - 所有矩阵维度必须满足 16 字节对齐
//   - TN 格式是首选
//   - computeType 必须是 CUBLAS_COMPUTE_32F
//   - scaleType 必须是 CUDA_R_32F
//
// 数据类型支持:
//   - INT8: Input=CUDA_R_8I, Output=CUDA_R_32I, Scale=CUDA_R_32I
//   - FP8:  Input=CUDA_R_8F_E4M3, Output=CUDA_R_32F, Scale=CUDA_R_32F

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

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

#define CHECK_CUBLAS_ERR(expr)                                                 \
  do {                                                                         \
    cublasStatus_t _status = (expr);                                           \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                    \
      std::ostringstream _oss;                                                 \
      _oss << "cuBLASLt 调用失败: " << cublasLtGetStatusString(_status)        \
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

static cublasComputeType_t compute_type_from_dtype(const std::string &dtype) {
  if (dtype == "int8") return CUBLAS_COMPUTE_32I;
  if (dtype == "fp8e4m3") return CUBLAS_COMPUTE_32F;
  throw std::invalid_argument("不支持的数据类型: " + dtype);
}

static cudaDataType scale_type_from_dtype(const std::string &dtype) {
  // INT8 使用 CUDA_R_32I scale type
  // FP8 使用 CUDA_R_32F scale type
  if (dtype == "int8") return CUDA_R_32I;
  if (dtype == "fp8e4m3") return CUDA_R_32F;
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

// ===== 简单量化：将 BF16/FP16 张量量化到 INT8 =====
static std::pair<torch::Tensor, double> quantize_int8(torch::Tensor x) {
  auto abs_max = x.abs().max().item<double>();
  double scale = abs_max > 0 ? 127.0 / abs_max : 1.0;
  auto q = (x * scale).round().clamp(-128, 127).to(torch::kChar);
  return {q, scale};
}

// ===== FP8 转换 =====
// FP8E4M3: 4位指数, 3位尾数, 动态范围小但精度高, 适合权重和激活
static torch::Tensor to_fp8_e4m3(torch::Tensor x) {
  if (!x.is_cuda()) throw std::runtime_error("FP8 转换需要 CUDA 张量");
  try {
    return x.to(torch::kFloat8_e4m3fn);
  } catch (...) {
    throw std::runtime_error("当前 PyTorch 版本不支持 FP8E4M3 类型，请升级到 PyTorch 2.1+");
  }
}

// ===== 搜索结果记录 =====
struct AlgRecord {
  int alg_id{-1};
  float lat_us{0.f};
  float tops{0.f};
  int64_t workspace{0};
  bool valid{false};
  float max_abs_err{0.f};
};

// ===== search_topk =====
// 固定布局: T/N + Col/Col + Col (权重W在左)
// W[N,K]^T_col * A[K,M]_col = C[N,M]_col
py::dict search_topk(torch::Tensor W_bf16, torch::Tensor A_bf16,
                     const std::vector<int64_t> &M_list,
                     const std::string &layout, const std::string &dtype,
                     int warmup, int repeat, bool verify,
                     const std::vector<int64_t> &blacklist_ids,
                     int topk) {
  if (!W_bf16.is_cuda() || !A_bf16.is_cuda()) {
    throw std::invalid_argument("W 与 A 必须在 CUDA 上");
  }
  if (W_bf16.dim() != 2 || A_bf16.dim() != 2) {
    throw std::invalid_argument("输入张量必须是二维 [N,K] / [M,K]");
  }
  if (topk <= 0) topk = 3;

  auto W_fp = W_bf16.contiguous();
  auto A_fp = A_bf16.contiguous();
  int64_t N = W_fp.size(0);
  int64_t K = W_fp.size(1);
  
  // === 维度对齐检查 ===
  // cuBLASLt 要求：维度需满足对齐要求（INT8 需 4 对齐，FP8 需 16 对齐）
  auto check_alignment = [](int64_t val, int64_t align, const char* name) {
    if (val % align != 0) {
      std::ostringstream oss;
      oss << "维度不满足对齐要求: " << name << "=" << val << " 不是 " << align << " 的倍数";
      throw std::invalid_argument(oss.str());
    }
  };
  int64_t align_req = (dtype == "fp8e4m3") ? 16 : 4;
  check_alignment(N, align_req, "N");
  check_alignment(K, align_req, "K");
  for (auto m : M_list) {
    check_alignment(m, align_req, "M");
  }

  // 量化/转换
  torch::Tensor W_q;
  torch::Tensor A_q;
  cudaDataType type_AB = to_cuda_dtype(dtype);
  cudaDataType type_C = cuda_out_dtype(dtype);
  cublasComputeType_t comp_type = compute_type_from_dtype(dtype);
  cudaDataType scale_type = scale_type_from_dtype(dtype);

  if (dtype == "int8") {
    auto qW = quantize_int8(W_fp);
    auto qA = quantize_int8(A_fp);
    W_q = qW.first;
    A_q = qA.first;
  } else if (dtype == "fp8e4m3") {
    W_q = to_fp8_e4m3(W_fp);
    A_q = to_fp8_e4m3(A_fp);
  } else {
    throw std::invalid_argument("不支持的数据类型: " + dtype + "。支持: int8, fp8e4m3");
  }

  // 获取最大 M 用于确定可用算法数
  int64_t max_M = 0;
  for (auto m : M_list) max_M = std::max(max_M, m);

  // === cuBLASLt 初始化 ===
  cublasLtHandle_t handle;
  CHECK_CUBLAS_ERR(cublasLtCreate(&handle));

  cudaStream_t stream = at::cuda::getDefaultCUDAStream();

  // 固定布局: T/N + Col/Col + Col
  // W 逻辑维度 [N,K], opW=T 所以存储为 [K,N] (转置后)
  // A 逻辑维度 [K,M], opA=N 所以存储为 [K,M]
  // C 逻辑维度 [N,M]
  cublasOperation_t opW = CUBLAS_OP_T;
  cublasOperation_t opA = CUBLAS_OP_N;
  cublasLtOrder_t orderW = CUBLASLT_ORDER_COL;
  cublasLtOrder_t orderA = CUBLASLT_ORDER_COL;
  cublasLtOrder_t orderC = CUBLASLT_ORDER_COL;

  // W 存储维度: [K,N] (因为 opW=T)
  int64_t num_W_rows = K;  // 存储的行数
  int64_t num_W_cols = N;  // 存储的列数
  int64_t ldw = num_W_rows; // Col major: leading dim = rows

  // 为结果分配存储
  int64_t numM = static_cast<int64_t>(M_list.size());
  torch::Tensor topk_alg = torch::full({numM, topk}, -1, torch::dtype(torch::kInt32));
  torch::Tensor topk_lat = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));
  torch::Tensor topk_tops = torch::zeros({numM, topk}, torch::dtype(torch::kFloat32));
  torch::Tensor topk_workspace = torch::zeros({numM, topk}, torch::dtype(torch::kInt64));
  torch::Tensor valid_mask = torch::zeros({numM, topk}, torch::dtype(torch::kUInt8));
  torch::Tensor num_valid = torch::zeros({numM}, torch::dtype(torch::kInt32));
  torch::Tensor verify_err = torch::zeros({numM}, torch::dtype(torch::kFloat32));
  std::vector<int> verify_failed_ids;

  // Workspace 分配
  size_t workspace_size = 32 * 1024 * 1024;  // 32 MB
  void *d_workspace = nullptr;
  CHECK_CUDA_ERR(cudaMalloc(&d_workspace, workspace_size));

  // 最大算法数（cuBLASLt 启发式搜索返回的最大数量）
  int max_returned_alg_count = 0;

  for (int64_t m_index = 0; m_index < numM; ++m_index) {
    int64_t M = M_list[m_index];
    
    // A 存储维度: [K,M] (因为 opA=N)
    int64_t num_A_rows = K;
    int64_t num_A_cols = M;
    int64_t lda = num_A_rows; // Col major

    // C 存储维度: [N,M]
    int64_t num_C_rows = N;
    int64_t num_C_cols = M;
    int64_t ldc = num_C_rows; // Col major

    // 分配设备内存
    size_t W_size = static_cast<size_t>(N) * K * dtype_size(dtype);
    size_t A_size = static_cast<size_t>(K) * M * dtype_size(dtype);
    size_t C_size = static_cast<size_t>(N) * M * out_dtype_size(dtype);

    void *dW = nullptr, *dA = nullptr, *dC = nullptr;
    CHECK_CUDA_ERR(cudaMalloc(&dW, W_size));
    CHECK_CUDA_ERR(cudaMalloc(&dA, A_size));
    CHECK_CUDA_ERR(cudaMalloc(&dC, C_size));

    // 拷贝数据到设备
    CHECK_CUDA_ERR(cudaMemcpy(dW, W_q.data_ptr(), W_size, cudaMemcpyHostToDevice));
    auto A_slice = A_q.narrow(0, 0, M).contiguous();
    CHECK_CUDA_ERR(cudaMemcpy(dA, A_slice.data_ptr(), A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemset(dC, 0, C_size));

    // 创建矩阵乘法描述符
    cublasLtMatmulDesc_t matmulDesc = nullptr;
    CHECK_CUBLAS_ERR(cublasLtMatmulDescCreate(&matmulDesc, comp_type, scale_type));
    CHECK_CUBLAS_ERR(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opW, sizeof(opW)));
    CHECK_CUBLAS_ERR(cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opA, sizeof(opA)));

    // 创建矩阵布局描述符
    cublasLtMatrixLayout_t layoutW = nullptr, layoutA = nullptr, layoutC = nullptr;

    // W 矩阵布局
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutCreate(&layoutW, type_AB, num_W_rows, num_W_cols, ldw));
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutSetAttribute(layoutW, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderW, sizeof(orderW)));

    // A 矩阵布局
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutCreate(&layoutA, type_AB, num_A_rows, num_A_cols, lda));
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderA, sizeof(orderA)));

    // C 矩阵布局
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutCreate(&layoutC, type_C, num_C_rows, num_C_cols, ldc));
    CHECK_CUBLAS_ERR(cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderC, sizeof(orderC)));

    // 创建算法偏好
    cublasLtMatmulPreference_t preference = nullptr;
    CHECK_CUBLAS_ERR(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLAS_ERR(cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    // 获取可用算法（启发式搜索）
    const int max_algo_count = 64;  // 请求更多算法
    cublasLtMatmulHeuristicResult_t heuristicResult[max_algo_count];
    int returnedAlgoCount = 0;

    cublasStatus_t heur_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        matmulDesc,
        layoutW,
        layoutA,
        layoutC,
        layoutC,
        preference,
        max_algo_count,
        heuristicResult,
        &returnedAlgoCount);

    if (heur_status != CUBLAS_STATUS_SUCCESS || returnedAlgoCount == 0) {
      // 清理资源并跳过此 M
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(layoutW);
      cublasLtMatrixLayoutDestroy(layoutA);
      cublasLtMatrixLayoutDestroy(layoutC);
      cublasLtMatmulDescDestroy(matmulDesc);
      cudaFree(dW); cudaFree(dA); cudaFree(dC);
      continue;
    }

    if (returnedAlgoCount > max_returned_alg_count) {
      max_returned_alg_count = returnedAlgoCount;
    }

    // CUDA events for timing
    cudaEvent_t start_event, stop_event;
    CHECK_CUDA_ERR(cudaEventCreate(&start_event));
    CHECK_CUDA_ERR(cudaEventCreate(&stop_event));

    std::vector<AlgRecord> records;

    // alpha/beta 根据 dtype 选择
    float alpha_f = 1.0f, beta_f = 0.0f;
    int32_t alpha_int = 1, beta_int = 0;
    const void *alpha_ptr = (dtype == "int8") ? (const void*)&alpha_int : (const void*)&alpha_f;
    const void *beta_ptr = (dtype == "int8") ? (const void*)&beta_int : (const void*)&beta_f;

    // 遍历所有返回的算法
    for (int alg_idx = 0; alg_idx < returnedAlgoCount; ++alg_idx) {
      if (heuristicResult[alg_idx].state != CUBLAS_STATUS_SUCCESS) {
        continue;
      }

      // 检查黑名单
      if (std::find(blacklist_ids.begin(), blacklist_ids.end(), alg_idx) != blacklist_ids.end()) {
        continue;
      }

      const cublasLtMatmulAlgo_t *algo = &heuristicResult[alg_idx].algo;
      size_t ws_size = heuristicResult[alg_idx].workspaceSize;

      // Warmup
      bool warmup_success = true;
      for (int w = 0; w < warmup; ++w) {
        cublasStatus_t st = cublasLtMatmul(
            handle,
            matmulDesc,
            alpha_ptr,
            dW, layoutW,
            dA, layoutA,
            beta_ptr,
            dC, layoutC,
            dC, layoutC,
            algo,
            d_workspace,
            ws_size,
            stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
          warmup_success = false;
          break;
        }
      }
      CHECK_CUDA_ERR(cudaStreamSynchronize(stream));

      if (!warmup_success) {
        continue;
      }

      // Benchmark
      CHECK_CUDA_ERR(cudaEventRecord(start_event, stream));
      for (int r = 0; r < repeat; ++r) {
        cublasLtMatmul(
            handle,
            matmulDesc,
            alpha_ptr,
            dW, layoutW,
            dA, layoutA,
            beta_ptr,
            dC, layoutC,
            dC, layoutC,
            algo,
            d_workspace,
            ws_size,
            stream);
      }
      CHECK_CUDA_ERR(cudaEventRecord(stop_event, stream));
      CHECK_CUDA_ERR(cudaEventSynchronize(stop_event));

      float total_ms = 0.0f;
      CHECK_CUDA_ERR(cudaEventElapsedTime(&total_ms, start_event, stop_event));
      float avg_us = (total_ms * 1000.0f) / repeat;

      // 计算 TOPS
      double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
      double tops = (flops / (avg_us * 1e-6)) / 1e12;

      AlgRecord rec;
      rec.alg_id = alg_idx;
      rec.lat_us = avg_us;
      rec.tops = static_cast<float>(tops);
      rec.workspace = static_cast<int64_t>(ws_size);
      rec.valid = true;

      // 校验（如需）
      if (verify) {
        // 拷贝结果回 CPU
        torch::Tensor C_host;
        if (dtype == "int8") {
          C_host = torch::empty({N, M}, torch::dtype(torch::kInt32));
        } else {
          C_host = torch::empty({N, M}, torch::dtype(torch::kFloat32));
        }
        CHECK_CUDA_ERR(cudaMemcpy(C_host.data_ptr(), dC, C_size, cudaMemcpyDeviceToHost));

        // 使用量化后的数据做 FP32 参考计算
        auto A_slice_v = A_q.narrow(0, 0, M);
        auto A_fp32 = A_slice_v.to(torch::kFloat32);
        auto W_fp32 = W_q.to(torch::kFloat32);
        auto ref = torch::matmul(W_fp32, A_fp32.transpose(0, 1));  // [N, M]

        // 将输出转为 FP32 比较
        torch::Tensor out_fp32 = C_host.to(torch::kFloat32);

        // 计算相对误差
        auto ref_abs = ref.abs().clamp_min(1.0f);
        auto rel_diff = ((out_fp32 - ref) / ref_abs).abs();
        float max_rel_err = rel_diff.max().item<float>();
        rec.max_abs_err = max_rel_err;

        constexpr float tol = 0.05f;
        constexpr float critical_tol = 1.00f;

        if (max_rel_err > critical_tol || std::isnan(max_rel_err)) {
          rec.valid = false;
          verify_failed_ids.push_back(alg_idx);
        } else if (max_rel_err > tol) {
          verify_failed_ids.push_back(alg_idx);
        }
      }

      if (rec.valid) {
        records.push_back(rec);
      }
    }

    CHECK_CUDA_ERR(cudaEventDestroy(start_event));
    CHECK_CUDA_ERR(cudaEventDestroy(stop_event));

    // 排序（按延迟升序，即吞吐量降序）
    std::sort(records.begin(), records.end(),
              [](const AlgRecord &a, const AlgRecord &b) {
                return a.lat_us < b.lat_us;
              });

    // 填充 topk 结果
    int fill = std::min(static_cast<int>(records.size()), topk);
    for (int i = 0; i < fill; ++i) {
      topk_alg.index_put_({m_index, i}, records[i].alg_id);
      topk_lat.index_put_({m_index, i}, records[i].lat_us);
      topk_tops.index_put_({m_index, i}, records[i].tops);
      topk_workspace.index_put_({m_index, i}, records[i].workspace);
      valid_mask.index_put_({m_index, i}, static_cast<uint8_t>(1));
      if (verify && i == 0) {
        verify_err.index_put_({m_index}, records[i].max_abs_err);
      }
    }
    num_valid.index_put_({m_index}, static_cast<int>(records.size()));

    // 清理当前 M 的资源
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layoutW);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    cudaFree(dW);
    cudaFree(dA);
    cudaFree(dC);
  }

  // 清理
  if (d_workspace) cudaFree(d_workspace);
  cublasLtDestroy(handle);

  // 构造返回
  py::dict out;
  out["M_list"] = torch::tensor(M_list, torch::dtype(torch::kInt32));
  out["NK"] = torch::tensor({static_cast<int32_t>(N), static_cast<int32_t>(K)},
                             torch::dtype(torch::kInt32));
  out["topk_alg_id"] = topk_alg;
  out["topk_lat_us"] = topk_lat;
  out["topk_tops"] = topk_tops;
  out["topk_workspace"] = topk_workspace;
  out["valid_mask"] = valid_mask;
  out["max_returned_alg_count"] = max_returned_alg_count;  // cuBLASLt 返回的最大算法数
  out["num_valid_algs_per_M"] = num_valid;
  if (verify) {
    out["verify_max_abs_err"] = verify_err;
    out["verify_failed_algs"] = torch::tensor(verify_failed_ids,
                                              torch::dtype(torch::kInt32));
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("search_topk", &search_topk, "枚举算法并返回 topk (cuBLASLt)", 
        py::arg("W_bf16"),
        py::arg("A_bf16"), 
        py::arg("M_list"), 
        py::arg("layout"),
        py::arg("dtype") = "int8", 
        py::arg("warmup") = 25,
        py::arg("repeat") = 100, 
        py::arg("verify") = false,
        py::arg("blacklist_ids") = std::vector<int64_t>{},
        py::arg("topk") = 3);
}
