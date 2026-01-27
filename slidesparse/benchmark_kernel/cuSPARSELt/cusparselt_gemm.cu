// SPDX-License-Identifier: Apache-2.0
/**
 * cuSPARSELt Sparse GEMM Benchmark Implementation
 * 
 * 支持的数据类型（按用户规格）:
 * ==============================
 * - FP16:    FP16 输入, FP32 计算, FP16 输出
 * - BF16:    BF16 输入, FP32 计算, BF16 输出
 * - INT8:    INT8 输入, INT32 计算, INT8 输出
 * - FP8E4M3: FP8 输入, FP32 计算, FP8 输出
 * - FP4E2M1: FP4 输入, FP32 计算, FP4 输出
 * 
 * 固定 Layout:
 * ============
 * - T/N + Col/Col + Col (权重 W 在左, 2:4 稀疏)
 * - R[N,M]_col = W_compressed[K,N]^T_col @ A[K,M]_col
 * 
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      cusparselt_gemm.cu -lcusparseLt -lcusparse -lcublas -lcuda \
 *      -o cusparselt_gemm.so
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include <cmath>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cusparseLt.h>

// =============================================================================
// 常量定义
// =============================================================================

static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;  // 512 MB

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

// 前向声明 - Segment-K 检测（需要在 ensure_handle 之前声明）
static bool check_segment_k_support();

// 全局禁用标志 (跨调用保持)
static bool g_disable_segment_k = false;
static bool g_disable_split_k_doubling = false;

static int ensure_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_init) {
        cusparseStatus_t err = cusparseLtInit(&g_handle);
        if (err != CUSPARSE_STATUS_SUCCESS) {
            set_error("Failed to initialize cuSPARSELt handle");
            return -1;
        }
        g_handle_init = true;
        
        // 早期检测：如果硬件不支持 segment-k，全局禁用
        // 这是防止后续搜索卡死的关键预拦截
        if (!check_segment_k_support()) {
            g_disable_segment_k = true;
        }
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
    if (strcmp(dtype, "fp16") == 0) {
        return {CUDA_R_16F, 2, 32, true};
    } else if (strcmp(dtype, "bf16") == 0) {
        return {CUDA_R_16BF, 2, 32, true};
    } else if (strcmp(dtype, "int8") == 0) {
        return {CUDA_R_8I, 1, 32, true};
    } else if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) {
        return {CUDA_R_8F_E4M3, 1, 32, true};
    }
#if CUDART_VERSION >= 12050
    else if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) {
        return {CUDA_R_4F_E2M1, 1, 32, true};  // packed, min unit is byte
    }
#endif
    return {{}, 0, 0, false};
}

// 输出类型 - 根据 cuSPARSELt 硬件限制设置
// - FP16/BF16: 输出与输入相同
// - INT8: 输出 BF16 (cuSPARSELt 支持 INT8 → BF16)
// - FP8: 输出 BF16
// - FP4: 输出 BF16
static cudaDataType_t get_out_dtype(const char* dtype) {
    if (strcmp(dtype, "fp16") == 0) return CUDA_R_16F;
    if (strcmp(dtype, "bf16") == 0) return CUDA_R_16BF;
    if (strcmp(dtype, "int8") == 0) return CUDA_R_16BF;  // INT8 → BF16
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return CUDA_R_16BF;
#if CUDART_VERSION >= 12050
    if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) return CUDA_R_16BF;
#endif
    return CUDA_R_16BF;  // 默认 BF16
}

static int get_out_dtype_size(const char* dtype) {
    if (strcmp(dtype, "fp16") == 0) return 2;
    if (strcmp(dtype, "bf16") == 0) return 2;
    if (strcmp(dtype, "int8") == 0) return 2;  // BF16 输出
    if (strcmp(dtype, "fp8e4m3") == 0 || strcmp(dtype, "fp8") == 0) return 2;  // BF16 输出
#if CUDART_VERSION >= 12050
    if (strcmp(dtype, "fp4e2m1") == 0 || strcmp(dtype, "fp4") == 0) return 2;  // BF16 输出
#endif
    return 2;  // 默认 BF16 = 2 bytes
}

static cusparseComputeType get_compute_type(const char* dtype) {
    if (strcmp(dtype, "int8") == 0) {
        return CUSPARSE_COMPUTE_32I;  // INT8 使用 INT32 累加
    }
    return CUSPARSE_COMPUTE_32F;  // FP16/BF16/FP8/FP4 使用 FP32 累加
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
// 算法搜索结构
// =============================================================================

struct AlgRecord {
    int alg_id;
    int split_k;
    float lat_us;
    float tops;
    int64_t workspace;
    bool valid;
    bool from_api_search;
    
    AlgRecord() : alg_id(-1), split_k(1), lat_us(0), tops(0), workspace(0), 
                  valid(false), from_api_search(false) {}
};

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 获取最后一条错误信息
 */
const char* cusparselt_alg_search_get_last_error() {
    return tls_last_error.c_str();
}

/**
 * @brief 检查 cuSPARSELt 是否可用
 */
int cusparselt_alg_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 检查是否支持 segment-k
 */
int cusparselt_supports_segment_k() {
    return check_segment_k_support() ? 1 : 0;
}

/**
 * @brief 获取对齐要求
 */
int cusparselt_alg_search_get_alignment(const char* dtype) {
    return 32;
}

/**
 * @brief 对矩阵进行 2:4 剪枝
 * 
 * @param input   输入矩阵设备指针 [rows x cols, 列主序]
 * @param output  输出矩阵设备指针
 * @param rows    行数 (K)
 * @param cols    列数 (N)
 * @param dtype   数据类型
 * @param stream  CUDA 流
 * @return 0 成功, -1 失败
 */
int cusparselt_prune_24(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype for prune");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    int64_t dummy_M = 32;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    cudaDataType_t out_dtype = get_out_dtype(dtype);  // 获取正确的输出类型
    
    // W (稀疏, structured) - [rows, cols] = [K, N]
    cusparseLtMatDescriptor_t matW;
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matW,
        rows, cols, rows,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense) - [K, M]
    cusparseLtMatDescriptor_t matA;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matA,
        rows, dummy_M, rows,
        16, info.cuda_type, order));
    
    // R (dense) - [N, M] - 使用正确的输出类型
    cusparseLtMatDescriptor_t matR;
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matR,
        cols, dummy_M, cols,
        16, out_dtype, order));  // 使用输出类型而不是输入类型
    
    // 创建 matmul 描述符
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR,
        compute_type));
    
    // 调用 prune
    cusparseStatus_t prune_status = cusparseLtSpMMAPrune(
        &g_handle, &matmul,
        input, output,
        CUSPARSELT_PRUNE_SPMMA_TILE,
        cu_stream);
    
    // 清理
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    if (prune_status != CUSPARSE_STATUS_SUCCESS) {
        char buf[256];
        snprintf(buf, sizeof(buf), "cusparseLtSpMMAPrune failed with status %d", (int)prune_status);
        set_error(buf);
        return -1;
    }
    
    cudaStreamSynchronize(cu_stream);
    
    return 0;
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
    
    int64_t dummy_M = 32;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    cudaDataType_t out_dtype = get_out_dtype(dtype);  // 获取正确的输出类型
    
    // 创建描述符
    cusparseLtMatDescriptor_t matW, matA, matR;
    cusparseStatus_t status;
    
    status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matW, rows, cols, rows, 16,
        info.cuda_type, order, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) return -1;
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matA, rows, dummy_M, rows, 16, info.cuda_type, order);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        return -1;
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matR, cols, dummy_M, cols, 16, out_dtype, order);  // 使用输出类型
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        return -1;
    }
    
    cusparseLtMatmulDescriptor_t matmul;
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR, compute_type);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        return -1;
    }
    
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        return -1;
    }
    
    cusparseLtMatmulPlan_t plan;
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        return -1;
    }
    
    size_t compressed_size = 0, compressed_buffer_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &plan, &compressed_size, &compressed_buffer_size);
    
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? (int64_t)compressed_size : -1;
}

/**
 * @brief 压缩稀疏矩阵
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
    
    int64_t dummy_M = 32;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    cusparseComputeType compute_type = get_compute_type(dtype);
    cudaDataType_t out_dtype = get_out_dtype(dtype);  // 获取正确的输出类型
    
    // 创建描述符
    cusparseLtMatDescriptor_t matW, matA, matR;
    cusparseStatus_t status;
    
    status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matW, rows, cols, rows, 16,
        info.cuda_type, order, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) { set_error("Failed to init matW"); return -1; }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matA, rows, dummy_M, rows, 16, info.cuda_type, order);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        set_error("Failed to init matA");
        return -1;
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matR, cols, dummy_M, cols, 16, out_dtype, order);  // 使用输出类型
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        set_error("Failed to init matR");
        return -1;
    }
    
    cusparseLtMatmulDescriptor_t matmul;
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR, compute_type);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to init matmul");
        return -1;
    }
    
    cusparseLtMatmulAlgSelection_t alg_sel;
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to init alg_sel");
        return -1;
    }
    
    cusparseLtMatmulPlan_t plan;
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to init plan");
        return -1;
    }
    
    // 获取压缩大小
    size_t compressed_size = 0, compressed_buffer_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &plan, &compressed_size, &compressed_buffer_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to get compressed size");
        return -1;
    }
    
    // 分配压缩 buffer
    void* compress_buffer = nullptr;
    if (compressed_buffer_size > 0) {
        cudaError_t alloc_err = cudaMalloc(&compress_buffer, compressed_buffer_size);
        if (alloc_err != cudaSuccess) {
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
            cusparseLtMatDescriptorDestroy(&matW);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matR);
            set_error("Failed to allocate compress buffer");
            return -1;
        }
    }
    
    // 压缩
    status = cusparseLtSpMMACompress(&g_handle, &plan, input, output, compress_buffer, cu_stream);
    
    if (compress_buffer) cudaFree(compress_buffer);
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        set_error("cusparseLtSpMMACompress failed");
        return -1;
    }
    
    cudaStreamSynchronize(cu_stream);
    
    return (int64_t)compressed_size;
}

/**
 * @brief 搜索单个 (N,K,M) 组合的最优算法
 *
 * @param W_pruned_ptr  W 矩阵设备指针 (已剪枝但未压缩, K x N 列主序)
 * @param A_ptr         A 矩阵设备指针 (K x M 列主序)
 * @param R_ptr         R 矩阵设备指针 (N x M 列主序)
 * @param N             N 维度
 * @param K             K 维度
 * @param M             M 维度
 * @param dtype         数据类型
 * @param warmup        预热次数
 * @param repeat        计时重复次数
 * @param topk          返回前 k 个结果
 * @param test_segment_k  是否测试 segment-k (-1)
 * @param do_api_search   是否执行官方 API 搜索对比
 *
 * 输出参数略
 *
 * @return 0 成功, -1 失败
 */
int cusparselt_search_single_m(
    void* W_pruned_ptr, void* A_ptr, void* R_ptr,
    int64_t N, int64_t K, int64_t M,
    const char* dtype,
    int warmup, int repeat,
    int topk,
    int test_segment_k,
    int do_api_search,
    // 输出
    int* out_alg_ids,
    int* out_split_k,
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    uint8_t* out_valid,
    int* out_num_valid,
    int* out_alg_count,
    int* out_config_count,
    int* out_api_alg_id,
    int* out_api_split_k,
    float* out_api_lat_us,
    int* out_api_rank,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype");
        return -1;
    }
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cudaDataType_t out_type = get_out_dtype(dtype);
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    // 初始化输出
    *out_num_valid = 0;
    *out_alg_count = 0;
    *out_config_count = 0;
    *out_api_alg_id = -1;
    *out_api_split_k = 1;
    *out_api_lat_us = 0.0f;
    *out_api_rank = -1;
    
    for (int i = 0; i < topk; ++i) {
        out_alg_ids[i] = -1;
        out_split_k[i] = 0;
        out_lat_us[i] = 0;
        out_tops[i] = 0;
        out_workspace[i] = 0;
        out_valid[i] = 0;
    }
    
    float alpha = 1.0f, beta = 0.0f;
    int32_t alpha_i = 1, beta_i = 0;
    
    // INT8 需要使用 INT32 scale
    bool use_int_scale = (strcmp(dtype, "int8") == 0);
    const void* alpha_ptr = use_int_scale ? (const void*)&alpha_i : (const void*)&alpha;
    const void* beta_ptr = use_int_scale ? (const void*)&beta_i : (const void*)&beta;
    
    // 创建基础矩阵描述符
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
    
    cusparseLtMatDescriptor_t matW, matA, matR;
    
    // W (稀疏, structured) - 存储为 [K, N], 转置后为 [N, K]
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &matW,
        K, N, K,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT));
    
    // A (dense) - [K, M]
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matA,
        K, M, K,
        16, info.cuda_type, order));
    
    // R (dense) - [N, M]
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &matR,
        N, M, N,
        16, out_type, order));
    
    // 创建 matmul 描述符
    cusparseLtMatmulDescriptor_t matmul;
    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &matW, &matA, &matR, &matR,
        compute_type));
    
    // 获取最大算法 ID
    cusparseLtMatmulAlgSelection_t alg_sel_tmp;
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel_tmp, &matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT));
    
    int max_alg_id = 0;
    cusparseStatus_t attr_status = cusparseLtMatmulAlgGetAttribute(
        &g_handle, &alg_sel_tmp,
        CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
        &max_alg_id, sizeof(max_alg_id));
    
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
    
    if (attr_status != CUSPARSE_STATUS_SUCCESS || max_alg_id <= 0) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        set_error("Failed to get max algorithm ID");
        return -1;
    }
    
    *out_alg_count = max_alg_id;
    
    // 共享 workspace
    void* shared_workspace = nullptr;
    size_t current_workspace_size = MAX_WORKSPACE_SIZE;
    CHECK_CUDA(cudaMalloc(&shared_workspace, current_workspace_size));
    
    // 收集所有有效结果
    std::vector<AlgRecord> records;
    int config_count = 0;
    
    // API 搜索结果
    AlgRecord api_search_record;
    api_search_record.valid = false;
    
    // === 官方 API 搜索 (可选) ===
    if (do_api_search) {
        cusparseLtMatmulAlgSelection_t alg_sel_api;
        cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
            &g_handle, &alg_sel_api, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        
        if (sel_st == CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulPlan_t plan_api;
            cusparseStatus_t plan_st = cusparseLtMatmulPlanInit(
                &g_handle, &plan_api, &matmul, &alg_sel_api);
            
            if (plan_st == CUSPARSE_STATUS_SUCCESS) {
                size_t comp_size_api = 0, comp_buf_size_api = 0;
                cusparseLtSpMMACompressedSize(&g_handle, &plan_api, &comp_size_api, &comp_buf_size_api);
                
                void* W_comp_api = nullptr;
                void* comp_buf_api = nullptr;
                cudaMalloc(&W_comp_api, comp_size_api);
                if (comp_buf_size_api > 0) cudaMalloc(&comp_buf_api, comp_buf_size_api);
                
                cusparseStatus_t comp_st = cusparseLtSpMMACompress(
                    &g_handle, &plan_api, W_pruned_ptr, W_comp_api, comp_buf_api, cu_stream);
                
                if (comp_st == CUSPARSE_STATUS_SUCCESS) {
                    size_t ws_api = 0;
                    cusparseLtMatmulGetWorkspace(&g_handle, &plan_api, &ws_api);
                    
                    cusparseStatus_t search_st = cusparseLtMatmulSearch(
                        &g_handle, &plan_api, alpha_ptr, W_comp_api, A_ptr,
                        beta_ptr, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                    
                    if (search_st == CUSPARSE_STATUS_SUCCESS) {
                        int api_alg_id = 0, api_split_k_val = 1;
                        
                        cusparseLtMatmulAlgGetAttribute(
                            &g_handle, &alg_sel_api, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                            &api_alg_id, sizeof(api_alg_id));
                        cusparseLtMatmulAlgGetAttribute(
                            &g_handle, &alg_sel_api, CUSPARSELT_MATMUL_SPLIT_K,
                            &api_split_k_val, sizeof(api_split_k_val));
                        
                        bool success = true;
                        
                        for (int i = 0; i < warmup && success; ++i) {
                            cusparseStatus_t st = cusparseLtMatmul(
                                &g_handle, &plan_api, alpha_ptr, W_comp_api, A_ptr,
                                beta_ptr, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                            if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                        }
                        cudaStreamSynchronize(cu_stream);
                        
                        if (success) {
                            cudaEvent_t start, stop;
                            cudaEventCreate(&start);
                            cudaEventCreate(&stop);
                            cudaEventRecord(start, cu_stream);
                            
                            for (int r = 0; r < repeat && success; ++r) {
                                cusparseStatus_t st = cusparseLtMatmul(
                                    &g_handle, &plan_api, alpha_ptr, W_comp_api, A_ptr,
                                    beta_ptr, R_ptr, R_ptr, shared_workspace, &cu_stream, 1);
                                if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                            }
                            
                            cudaEventRecord(stop, cu_stream);
                            cudaEventSynchronize(stop);
                            float total_ms = 0;
                            cudaEventElapsedTime(&total_ms, start, stop);
                            cudaEventDestroy(start);
                            cudaEventDestroy(stop);
                            
                            if (success) {
                                api_search_record.alg_id = api_alg_id;
                                api_search_record.split_k = api_split_k_val;
                                api_search_record.lat_us = (total_ms * 1000.0f) / (float)repeat;
                                double ops = 2.0 * (double)M * (double)N * (double)K;
                                api_search_record.tops = (float)(ops / (api_search_record.lat_us / 1e6) / 1e12);
                                api_search_record.workspace = (int64_t)ws_api;
                                api_search_record.valid = true;
                                api_search_record.from_api_search = true;
                            }
                        }
                    }
                }
                
                if (W_comp_api) cudaFree(W_comp_api);
                if (comp_buf_api) cudaFree(comp_buf_api);
                cusparseLtMatmulPlanDestroy(&plan_api);
            }
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_api);
        }
    }
    
    if (api_search_record.valid) {
        records.push_back(api_search_record);
    }
    
    // === 双层网格搜索 ===
    for (int alg_id = 0; alg_id < max_alg_id; ++alg_id) {
        // 为当前 alg_id 创建 plan 并压缩权重
        cusparseLtMatmulAlgSelection_t alg_sel_compress;
        cusparseStatus_t sel_st = cusparseLtMatmulAlgSelectionInit(
            &g_handle, &alg_sel_compress, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (sel_st != CUSPARSE_STATUS_SUCCESS) continue;
        
        cusparseStatus_t set_st = cusparseLtMatmulAlgSetAttribute(
            &g_handle, &alg_sel_compress, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &alg_id, sizeof(alg_id));
        if (set_st != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
            continue;
        }
        
        cusparseLtMatmulPlan_t plan_compress;
        cusparseStatus_t plan_st = cusparseLtMatmulPlanInit(
            &g_handle, &plan_compress, &matmul, &alg_sel_compress);
        if (plan_st != CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
            continue;
        }
        
        size_t compressed_size = 0, compressed_buffer_size = 0;
        cusparseLtSpMMACompressedSize(&g_handle, &plan_compress, &compressed_size, &compressed_buffer_size);
        
        void* W_compressed = nullptr;
        void* compress_buffer = nullptr;
        cudaError_t alloc_st = cudaMalloc(&W_compressed, compressed_size);
        if (alloc_st != cudaSuccess) {
            cusparseLtMatmulPlanDestroy(&plan_compress);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
            continue;
        }
        if (compressed_buffer_size > 0) {
            alloc_st = cudaMalloc(&compress_buffer, compressed_buffer_size);
            if (alloc_st != cudaSuccess) {
                cudaFree(W_compressed);
                cusparseLtMatmulPlanDestroy(&plan_compress);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
                continue;
            }
        }
        
        cusparseStatus_t compress_st = cusparseLtSpMMACompress(
            &g_handle, &plan_compress, W_pruned_ptr, W_compressed, compress_buffer, cu_stream);
        
        if (compress_buffer) {
            cudaFree(compress_buffer);
            compress_buffer = nullptr;
        }
        
        cusparseLtMatmulPlanDestroy(&plan_compress);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel_compress);
        
        if (compress_st != CUSPARSE_STATUS_SUCCESS) {
            cudaFree(W_compressed);
            continue;
        }
        
        float best_lat_us_for_doubling = -1.0f;
        
        std::vector<int> split_k_candidates;
        split_k_candidates.push_back(1);
        if (!g_disable_split_k_doubling) {
            for (int sk = 2; sk <= K; sk *= 2) {
                split_k_candidates.push_back(sk);
            }
        }
        // 关键：只有在硬件支持 segment-k（SM90+）且参数请求时才测试
        // 这是防止卡死的关键预拦截
        bool hw_supports_segment_k = check_segment_k_support();
        if (test_segment_k && !g_disable_segment_k && hw_supports_segment_k) {
            split_k_candidates.push_back(-1);
        }
        
        bool stop_doubling = false;
        
        for (int split_k_val : split_k_candidates) {
            if (api_search_record.valid &&
                alg_id == api_search_record.alg_id &&
                split_k_val == api_search_record.split_k) {
                continue;
            }
            
            if (stop_doubling && split_k_val > 1 && split_k_val != -1) {
                continue;
            }
            
            cusparseLtMatmulAlgSelection_t alg_sel;
            cusparseStatus_t sel_status = cusparseLtMatmulAlgSelectionInit(
                &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
            if (sel_status != CUSPARSE_STATUS_SUCCESS) break;
            
            cusparseStatus_t set_status = cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &alg_id, sizeof(alg_id));
            if (set_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
            cusparseStatus_t split_k_status = cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel, CUSPARSELT_MATMUL_SPLIT_K,
                &split_k_val, sizeof(split_k_val));
            if (split_k_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                if (split_k_val > 1) stop_doubling = true;
                continue;
            }
            
            cusparseLtMatmulPlan_t plan;
            cusparseStatus_t plan_status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel);
            if (plan_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                if (split_k_val == -1) {
                    g_disable_segment_k = true;
                } else if (split_k_val > 1) {
                    stop_doubling = true;
                    g_disable_split_k_doubling = true;
                }
                continue;
            }
            
            size_t workspace_size = 0;
            cusparseLtMatmulGetWorkspace(&g_handle, &plan, &workspace_size);
            
            if (workspace_size > current_workspace_size) {
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
            void* workspace = (workspace_size > 0) ? shared_workspace : nullptr;
            bool success = true;
            
            for (int i = 0; i < warmup && success; ++i) {
                cusparseStatus_t st = cusparseLtMatmul(
                    &g_handle, &plan, alpha_ptr, W_compressed, A_ptr,
                    beta_ptr, R_ptr, R_ptr, workspace, &cu_stream, 1);
                if (st != CUSPARSE_STATUS_SUCCESS) success = false;
            }
            cudaStreamSynchronize(cu_stream);
            
            if (!success) {
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                if (split_k_val == -1) {
                    g_disable_segment_k = true;
                } else if (split_k_val > 1) {
                    stop_doubling = true;
                    g_disable_split_k_doubling = true;
                }
                continue;
            }
            
            cudaEvent_t start = nullptr, stop = nullptr;
            float total_ms = 0.0f;
            
            if (success) {
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start, cu_stream);
                
                for (int r = 0; r < repeat && success; ++r) {
                    cusparseStatus_t st = cusparseLtMatmul(
                        &g_handle, &plan, alpha_ptr, W_compressed, A_ptr,
                        beta_ptr, R_ptr, R_ptr, workspace, &cu_stream, 1);
                    if (st != CUSPARSE_STATUS_SUCCESS) success = false;
                }
                
                cudaEventRecord(stop, cu_stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&total_ms, start, stop);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            }
            
            if (success) {
                AlgRecord rec;
                rec.alg_id = alg_id;
                rec.split_k = split_k_val;
                rec.lat_us = (total_ms * 1000.0f) / (float)repeat;
                double ops = 2.0 * (double)M * (double)N * (double)K;
                rec.tops = (float)(ops / (rec.lat_us / 1e6) / 1e12);
                rec.workspace = (int64_t)workspace_size;
                rec.valid = true;
                rec.from_api_search = false;
                
                records.push_back(rec);
                ++config_count;
                
                if (split_k_val >= 1) {
                    if (best_lat_us_for_doubling < 0 || rec.lat_us < best_lat_us_for_doubling) {
                        best_lat_us_for_doubling = rec.lat_us;
                    } else if (rec.lat_us * 1.05f > best_lat_us_for_doubling && split_k_val > 1) {
                        stop_doubling = true;
                    }
                }
            }
            
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        }
        
        cudaFree(W_compressed);
    }
    
    *out_config_count = config_count;
    
    // 按延迟排序
    std::sort(records.begin(), records.end(),
              [](const AlgRecord& a, const AlgRecord& b) {
                  return a.lat_us < b.lat_us;
              });
    
    // 记录 API 搜索结果的排名
    if (api_search_record.valid) {
        for (size_t i = 0; i < records.size(); ++i) {
            if (records[i].from_api_search) {
                *out_api_rank = (int)(i + 1);
                *out_api_alg_id = records[i].alg_id;
                *out_api_split_k = records[i].split_k;
                *out_api_lat_us = records[i].lat_us;
                break;
            }
        }
    }
    
    // 填充输出
    *out_num_valid = (int)records.size();
    
    for (int i = 0; i < topk; ++i) {
        if (i < (int)records.size()) {
            const auto& r = records[i];
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
    cudaFree(shared_workspace);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    
    return 0;
}

}  // extern "C"
