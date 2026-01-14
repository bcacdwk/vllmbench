// SPDX-License-Identifier: Apache-2.0
/**
 * @file alg_search_cusparselt.cu
 * @brief cuSPARSELt 算法离线搜索实现 (extern "C" 接口版本)
 *
 * 架构说明:
 * =========
 * 本文件提供 cuSPARSELt 稀疏矩阵乘法的算法搜索功能，包含:
 *   - 2:4 稀疏模式 (prune_24)
 *   - Split-K 支持
 *   - Segment-K 检测 (SM90+)
 *
 * 固定 Layout:
 * - T/N + Col/Col + Col (权重 W 在左, 2:4 稀疏)
 *
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      alg_search_cusparselt.cu -lcusparseLt -lcusparse -lcublas \
 *      -o alg_search_cusparselt.so
 *
 * 主要接口:
 * ---------
 * - cusparselt_search_single_m()      : 搜索单个 (N,K,M) 的最优算法
 * - cusparselt_prune_24()             : 对矩阵进行 2:4 剪枝
 * - cusparselt_supports_segment_k()   : 检测是否支持 segment-k
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

// =============================================================================
// 常量定义
// =============================================================================

// 最大 workspace 大小: 512 MB
static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;

// Split-K 搜索范围
static constexpr int MAX_SPLIT_K = 32;
static constexpr int SPLIT_K_VALUES[] = {1, 2, 4, 8, 16, 32};
static constexpr int NUM_SPLIT_K = 6;

// =============================================================================
// 错误处理
// =============================================================================

thread_local std::string tls_last_error;

static void set_error(const char* msg) {
    tls_last_error = msg;
}

static void set_error(const std::string& msg) {
    tls_last_error = msg;
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "CUDA Error %d: %s at %s:%d", \
                 (int)err, cudaGetErrorString(err), __FILE__, __LINE__); \
        set_error(buf); \
        return -1; \
    } \
} while (0)

#define CHECK_CUSPARSELT(call) do { \
    cusparseStatus_t err = (call); \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "cuSPARSELt Error %d at %s:%d", \
                 (int)err, __FILE__, __LINE__); \
        set_error(buf); \
        return -1; \
    } \
} while (0)

// =============================================================================
// 全局资源
// =============================================================================

static cusparseLtHandle_t g_handle;
static bool g_handle_init = false;
static std::mutex g_handle_mutex;

static int ensure_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_init) {
        cusparseStatus_t err = cusparseLtInit(&g_handle);
        if (err != CUSPARSE_STATUS_SUCCESS) {
            set_error("Failed to initialize cuSPARSELt handle");
            return -1;
        }
        g_handle_init = true;
    }
    return 0;
}

// =============================================================================
// 数据类型辅助
// =============================================================================

struct DtypeInfo {
    cudaDataType_t cuda_type;
    int elem_size;
    int alignment;
    bool is_valid;
};

static DtypeInfo get_dtype_info(const char* dtype) {
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return {CUDA_R_8I, 1, 16, true};
    } else if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "FP8") == 0) {
        return {CUDA_R_8F_E4M3, 1, 16, true};
    }
    return {{}, 0, 0, false};
}

static cudaDataType_t get_out_dtype(const char* outdtype) {
    if (strcmp(outdtype, "bf16") == 0 || strcmp(outdtype, "BF16") == 0) {
        return CUDA_R_16BF;
    } else if (strcmp(outdtype, "fp32") == 0 || strcmp(outdtype, "FP32") == 0) {
        return CUDA_R_32F;
    }
    return CUDA_R_16BF;
}

static cusparseComputeType get_compute_type(const char* dtype) {
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return CUSPARSE_COMPUTE_32I;
    }
    return CUSPARSE_COMPUTE_32F;
}

// =============================================================================
// Segment-K 检测
// =============================================================================

static bool check_segment_k_support() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    // Segment-K 需要 SM90+ (Hopper 及更高)
    return (prop.major >= 9);
}

// =============================================================================
// 2:4 Prune 函数
// =============================================================================

static int prune_24_internal(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    cudaStream_t stream)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype for prune");
        return -1;
    }
    
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    cusparseLtMatDescriptor_t matA;
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rows, cols, cols,  // ld = rows for col-major
        16,  // alignment
        info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    CHECK_CUSPARSELT(cusparseLtSpMMAPrune(
        &g_handle, &matA,
        input, output,
        CUSPARSELT_PRUNE_SPMMA_STRIP,
        stream));
    
    CHECK_CUSPARSELT(cusparseLtMatDescriptorDestroy(&matA));
    
    return 0;
}

// =============================================================================
// 算法搜索结构
// =============================================================================

struct AlgResult {
    int alg_id;
    int split_k;
    float lat_us;
    float tops;
    int64_t workspace;
    float waves_count;
    uint8_t valid;
};

// =============================================================================
// 单个配置测试
// =============================================================================

static int test_single_config(
    cusparseLtHandle_t* handle,
    cusparseLtMatmulDescriptor_t* matmul,
    cusparseLtMatmulAlgSelection_t* alg_sel,
    cusparseLtMatmulPlan_t* plan,
    void* A_ptr, void* B_ptr, void* C_ptr,
    void* workspace, size_t workspace_size,
    int64_t M, int64_t N, int64_t K,
    int alg_id, int split_k,
    int warmup, int repeat,
    cudaStream_t stream,
    AlgResult* result)
{
    result->alg_id = alg_id;
    result->split_k = split_k;
    result->valid = 0;
    
    // 设置算法
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSetAttribute(
        handle, alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg_id, sizeof(alg_id)));
    
    // 设置 split-k
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSetAttribute(
        handle, alg_sel,
        CUSPARSELT_MATMUL_SPLIT_K,
        &split_k, sizeof(split_k)));
    
    // 获取 workspace 大小
    size_t ws_size = 0;
    cusparseStatus_t status = cusparseLtMatmulPlanInit(
        handle, plan, matmul, alg_sel, workspace_size);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return 0;  // 配置不支持
    }
    
    status = cusparseLtMatmulGetWorkspace(handle, plan, &ws_size);
    if (status != CUSPARSE_STATUS_SUCCESS || ws_size > workspace_size) {
        cusparseLtMatmulPlanDestroy(plan);
        return 0;
    }
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 验证可执行
    status = cusparseLtMatmul(
        handle, plan,
        &alpha, A_ptr, B_ptr,
        &beta, C_ptr, C_ptr,
        workspace, &stream, 1);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulPlanDestroy(plan);
        return 0;
    }
    
    cudaStreamSynchronize(stream);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        cusparseLtMatmul(
            handle, plan,
            &alpha, A_ptr, B_ptr,
            &beta, C_ptr, C_ptr,
            workspace, &stream, 1);
    }
    cudaStreamSynchronize(stream);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < repeat; ++i) {
        cusparseLtMatmul(
            handle, plan,
            &alpha, A_ptr, B_ptr,
            &beta, C_ptr, C_ptr,
            workspace, &stream, 1);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    float lat_us = (ms_total * 1000.0f) / repeat;
    // 2:4 稀疏: 实际计算量是 dense 的一半
    float tops = (2.0 * M * N * K * 0.5 / 1e12) / (lat_us / 1e6);
    
    result->lat_us = lat_us;
    result->tops = tops;
    result->workspace = (int64_t)ws_size;
    result->waves_count = 0;  // cuSPARSELt 不提供 waves count
    result->valid = 1;
    
    cusparseLtMatmulPlanDestroy(plan);
    
    return 1;
}

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 搜索单个 (N,K,M) 组合的最优算法
 *
 * @param W_ptr        W 矩阵设备指针 (已压缩的稀疏矩阵)
 * @param A_ptr        A 矩阵设备指针
 * @param C_ptr        C 矩阵设备指针
 * @param N            N 维度
 * @param K            K 维度
 * @param M            M 维度
 * @param dtype        输入数据类型 ("int8" / "fp8e4m3")
 * @param outdtype     输出数据类型 ("bf16" / "fp32")
 * @param warmup       预热次数
 * @param repeat       计时重复次数
 * @param topk         返回前 k 个结果
 * @param search_split_k  是否搜索 split-k
 *
 * 输出参数:
 * @param out_alg_ids      算法 ID 数组 (大小 = topk)
 * @param out_split_k      split-k 值数组 (大小 = topk)
 * @param out_lat_us       延迟数组 (大小 = topk)
 * @param out_tops         吞吐量数组 (大小 = topk)
 * @param out_workspace    workspace 数组 (大小 = topk)
 * @param out_valid        有效标记数组 (大小 = topk)
 * @param out_num_valid    有效结果数量
 * @param out_alg_count    算法总数
 * @param stream           CUDA 流 (可为 nullptr)
 *
 * @return 0 成功, -1 失败
 */
int cusparselt_search_single_m(
    void* W_ptr, void* A_ptr, void* C_ptr,
    int64_t N, int64_t K, int64_t M,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    int topk,
    int search_split_k,
    // 输出
    int* out_alg_ids,
    int* out_split_k,
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    uint8_t* out_valid,
    int* out_num_valid,
    int* out_alg_count,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cudaDataType_t out_type = get_out_dtype(outdtype);
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    // 创建矩阵描述符
    // C = W^T * A  (W: N x K sparse, A: K x M dense, C: N x M)
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    // W (稀疏, structured)
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        K, N, K,  // rows=K, cols=N, ld=K (转置后为 N x K)
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense)
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matB,
        K, M, K,  // rows=K, cols=M, ld=K
        16, info.cuda_type, order));
    
    // C (dense)
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matC,
        N, M, N,  // rows=N, cols=M, ld=N
        16, out_type, order));
    
    // 创建 matmul 描述符
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matA, &matB, &matC, &matC,
        compute_type));
    
    // 算法选择
    cusparseLtMatmulAlgSelection_t alg_sel;
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT));
    
    // 获取算法数量
    int num_algs = 0;
    CHECK_CUSPARSELT(cusparseLtMatmulAlgGetAttribute(
        &g_handle, &alg_sel,
        CUSPARSELT_MATMUL_NUM_ALG_IDS,
        &num_algs, sizeof(num_algs)));
    
    *out_alg_count = num_algs;
    
    // 分配 workspace
    void* workspace = nullptr;
    size_t workspace_size = MAX_WORKSPACE_SIZE;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    
    // 收集所有有效结果
    std::vector<AlgResult> all_results;
    all_results.reserve(num_algs * (search_split_k ? NUM_SPLIT_K : 1));
    
    cusparseLtMatmulPlan_t plan;
    
    for (int alg_id = 0; alg_id < num_algs; ++alg_id) {
        if (search_split_k) {
            for (int sk_idx = 0; sk_idx < NUM_SPLIT_K; ++sk_idx) {
                int split_k = SPLIT_K_VALUES[sk_idx];
                AlgResult res;
                
                int ok = test_single_config(
                    &g_handle, &matmul, &alg_sel, &plan,
                    W_ptr, A_ptr, C_ptr,
                    workspace, workspace_size,
                    M, N, K,
                    alg_id, split_k,
                    warmup, repeat,
                    cu_stream,
                    &res);
                
                if (ok > 0 && res.valid) {
                    all_results.push_back(res);
                }
            }
        } else {
            AlgResult res;
            int ok = test_single_config(
                &g_handle, &matmul, &alg_sel, &plan,
                W_ptr, A_ptr, C_ptr,
                workspace, workspace_size,
                M, N, K,
                alg_id, 1,
                warmup, repeat,
                cu_stream,
                &res);
            
            if (ok > 0 && res.valid) {
                all_results.push_back(res);
            }
        }
    }
    
    // 按 TOPS 排序
    std::sort(all_results.begin(), all_results.end(),
              [](const AlgResult& a, const AlgResult& b) {
                  return a.tops > b.tops;
              });
    
    // 输出 top-k
    int num_valid = std::min((int)all_results.size(), topk);
    *out_num_valid = num_valid;
    
    for (int i = 0; i < topk; ++i) {
        if (i < num_valid) {
            const auto& r = all_results[i];
            out_alg_ids[i] = r.alg_id;
            out_split_k[i] = r.split_k;
            out_lat_us[i] = r.lat_us;
            out_tops[i] = r.tops;
            out_workspace[i] = r.workspace;
            out_valid[i] = 1;
        } else {
            out_alg_ids[i] = -1;
            out_split_k[i] = 0;
            out_lat_us[i] = 0;
            out_tops[i] = 0;
            out_workspace[i] = 0;
            out_valid[i] = 0;
        }
    }
    
    // 清理
    cudaFree(workspace);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatmulDescriptorDestroy(&matmul);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    
    return 0;
}

/**
 * @brief 对矩阵进行 2:4 剪枝
 *
 * @param input   输入矩阵设备指针
 * @param output  输出矩阵设备指针 (可以与 input 相同)
 * @param rows    行数
 * @param cols    列数
 * @param dtype   数据类型
 * @param stream  CUDA 流
 *
 * @return 0 成功, -1 失败
 */
int cusparselt_prune_24(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    return prune_24_internal(input, output, rows, cols, dtype, cu_stream);
}

/**
 * @brief 压缩稀疏矩阵
 *
 * @param input   输入矩阵设备指针 (已剪枝)
 * @param output  输出压缩矩阵设备指针
 * @param rows    行数
 * @param cols    列数
 * @param dtype   数据类型
 * @param stream  CUDA 流
 *
 * @return 压缩后大小 (字节), -1 表示失败
 */
int64_t cusparselt_compress(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype for compress");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    cusparseLtMatDescriptor_t matA;
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rows, cols, rows,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    size_t compressed_size = 0;
    CHECK_CUSPARSELT(cusparseLtSpMMACompressedSize(
        &g_handle, &matA, &compressed_size));
    
    CHECK_CUSPARSELT(cusparseLtSpMMACompress(
        &g_handle, &matA,
        input, output,
        cu_stream));
    
    cusparseLtMatDescriptorDestroy(&matA);
    
    return (int64_t)compressed_size;
}

/**
 * @brief 检查是否支持 segment-k
 */
int cusparselt_supports_segment_k() {
    return check_segment_k_support() ? 1 : 0;
}

/**
 * @brief 检查 cuSPARSELt 是否可用
 */
int cusparselt_alg_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 获取最后一条错误信息
 */
const char* cusparselt_alg_search_get_last_error() {
    return tls_last_error.c_str();
}

/**
 * @brief 获取对齐要求
 */
int cusparselt_alg_search_get_alignment(const char* dtype) {
    return 16;  // cuSPARSELt 通常需要 16 字节对齐
}

/**
 * @brief 获取压缩后的大小
 */
int64_t cusparselt_get_compressed_size(
    int64_t rows, int64_t cols,
    const char* dtype)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) return -1;
    
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    cusparseLtMatDescriptor_t matA;
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rows, cols, rows,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT);
    
    if (status != CUSPARSE_STATUS_SUCCESS) return -1;
    
    size_t compressed_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &matA, &compressed_size);
    
    cusparseLtMatDescriptorDestroy(&matA);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? (int64_t)compressed_size : -1;
}

}  // extern "C"
