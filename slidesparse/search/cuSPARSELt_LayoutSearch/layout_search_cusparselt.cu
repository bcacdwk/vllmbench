// SPDX-License-Identifier: Apache-2.0
/**
 * @file layout_search_cusparselt.cu
 * @brief cuSPARSELt 布局搜索实现 (extern "C" 接口版本)
 *
 * 架构说明:
 * =========
 * 本文件提供 cuSPARSELt 稀疏矩阵乘法的布局配置搜索功能。
 * 测试 8 种布局组合 (转置 + 存储顺序):
 *   - 转置 : TT, TN, NT, NN
 *   - A/B 排列 : RowCol, ColCol
 *
 * 2:4 稀疏特性:
 * - 稀疏矩阵需要 2:4 剪枝并压缩
 * - 固定最优布局: T/N + Col/Col + Col
 *
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      layout_search_cusparselt.cu -lcusparseLt -lcusparse -lcublas \
 *      -o layout_search_cusparselt.so
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
// 常量与类型定义
// =============================================================================

static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;
static constexpr int NUM_LAYOUTS = 8;

// 布局配置
struct LayoutConfig {
    cusparseOperation_t transA;
    cusparseOperation_t transB;
    cusparseOrder_t orderA;
    cusparseOrder_t orderB;
    const char* name;
};

// 8 种布局配置
static const LayoutConfig LAYOUT_CONFIGS[NUM_LAYOUTS] = {
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, "TT_RowCol"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, "TN_RowCol"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, "NT_RowCol"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, "NN_RowCol"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "TT_ColCol"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "TN_ColCol"},  // 推荐
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "NT_ColCol"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "NN_ColCol"},
};

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
// 布局搜索结果
// =============================================================================

struct LayoutResult {
    int layout_id;
    char layout_name[32];
    float lat_us;
    float tops;
    int64_t workspace;
    uint8_t valid;
};

// =============================================================================
// 单个布局测试
// =============================================================================

static int test_single_layout(
    const LayoutConfig& layout,
    int layout_id,
    void* A_compressed_ptr, void* B_ptr, void* C_ptr,
    int64_t M, int64_t N, int64_t K,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    void* workspace, size_t workspace_size,
    cudaStream_t stream,
    LayoutResult* result)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        return 0;
    }
    
    cudaDataType_t out_type = get_out_dtype(outdtype);
    cusparseComputeType compute_type = get_compute_type(dtype);
    
    result->layout_id = layout_id;
    strncpy(result->layout_name, layout.name, 31);
    result->layout_name[31] = '\0';
    result->valid = 0;
    
    // 创建描述符
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    
    // 根据布局配置设置维度和 leading dimension
    int64_t rowsA, colsA, ldA;
    int64_t rowsB, colsB, ldB;
    
    if (layout.transA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
        rowsA = N; colsA = K;
    } else {
        rowsA = K; colsA = N;
    }
    
    if (layout.transB == CUSPARSE_OPERATION_NON_TRANSPOSE) {
        rowsB = K; colsB = M;
    } else {
        rowsB = M; colsB = K;
    }
    
    if (layout.orderA == CUSPARSE_ORDER_COL) {
        ldA = rowsA;
    } else {
        ldA = colsA;
    }
    
    if (layout.orderB == CUSPARSE_ORDER_COL) {
        ldB = rowsB;
    } else {
        ldB = colsB;
    }
    
    // 创建 structured (sparse) 矩阵描述符
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rowsA, colsA, ldA,
        16, info.cuda_type, layout.orderA,
        CUSPARSELT_SPARSITY_50_PERCENT);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        return 0;  // 此配置不支持
    }
    
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matB,
        rowsB, colsB, ldB,
        16, info.cuda_type, layout.orderB);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matA);
        return 0;
    }
    
    cusparseOrder_t orderC = CUSPARSE_ORDER_COL;
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matC,
        N, M, N,
        16, out_type, orderC);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        return 0;
    }
    
    // 创建 matmul 描述符
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul,
        layout.transA, layout.transB,
        &matA, &matB, &matC, &matC,
        compute_type);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        cusparseLtMatDescriptorDestroy(&matC);
        return 0;
    }
    
    // 算法选择
    status = cusparseLtMatmulAlgSelectionInit(
        &g_handle, &alg_sel, &matmul,
        CUSPARSELT_MATMUL_ALG_DEFAULT);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatmulDescriptorDestroy(&matmul);
        return 0;
    }
    
    // 初始化计划
    status = cusparseLtMatmulPlanInit(&g_handle, &plan, &matmul, &alg_sel, workspace_size);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatmulDescriptorDestroy(&matmul);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        return 0;
    }
    
    size_t ws_size = 0;
    cusparseLtMatmulGetWorkspace(&g_handle, &plan, &ws_size);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 验证可执行
    status = cusparseLtMatmul(
        &g_handle, &plan,
        &alpha, A_compressed_ptr, B_ptr,
        &beta, C_ptr, C_ptr,
        workspace, &stream, 1);
    
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatmulPlanDestroy(&plan);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matB);
        cusparseLtMatDescriptorDestroy(&matC);
        cusparseLtMatmulDescriptorDestroy(&matmul);
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        return 0;
    }
    
    cudaStreamSynchronize(stream);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        cusparseLtMatmul(
            &g_handle, &plan,
            &alpha, A_compressed_ptr, B_ptr,
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
            &g_handle, &plan,
            &alpha, A_compressed_ptr, B_ptr,
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
    // 2:4 稀疏计算量减半
    float tops = (2.0 * M * N * K * 0.5 / 1e12) / (lat_us / 1e6);
    
    result->lat_us = lat_us;
    result->tops = tops;
    result->workspace = (int64_t)ws_size;
    result->valid = 1;
    
    // 清理
    cusparseLtMatmulPlanDestroy(&plan);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matB);
    cusparseLtMatDescriptorDestroy(&matC);
    cusparseLtMatmulDescriptorDestroy(&matmul);
    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
    
    return 1;
}

// =============================================================================
// Prune 和 Compress 辅助函数
// =============================================================================

static int prune_24_for_layout(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    cusparseOrder_t order,
    cudaStream_t stream)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) return -1;
    
    int64_t ld = (order == CUSPARSE_ORDER_COL) ? rows : cols;
    
    cusparseLtMatDescriptor_t matA;
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rows, cols, ld,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT);
    
    if (status != CUSPARSE_STATUS_SUCCESS) return -1;
    
    status = cusparseLtSpMMAPrune(
        &g_handle, &matA,
        input, output,
        CUSPARSELT_PRUNE_SPMMA_STRIP,
        stream);
    
    cusparseLtMatDescriptorDestroy(&matA);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? 0 : -1;
}

static int64_t compress_for_layout(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    cusparseOrder_t order,
    cudaStream_t stream)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) return -1;
    
    int64_t ld = (order == CUSPARSE_ORDER_COL) ? rows : cols;
    
    cusparseLtMatDescriptor_t matA;
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rows, cols, ld,
        16, info.cuda_type, order,
        CUSPARSELT_SPARSITY_50_PERCENT);
    
    if (status != CUSPARSE_STATUS_SUCCESS) return -1;
    
    size_t compressed_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &matA, &compressed_size);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matA);
        return -1;
    }
    
    status = cusparseLtSpMMACompress(
        &g_handle, &matA,
        input, output,
        stream);
    
    cusparseLtMatDescriptorDestroy(&matA);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? (int64_t)compressed_size : -1;
}

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 测试单个 (N,K,M) 的 8 种布局配置
 */
int cusparselt_layout_search_single(
    void* A_compressed_ptr, void* B_ptr, void* C_ptr,
    int64_t M, int64_t N, int64_t K,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    // 输出数组 (大小 = NUM_LAYOUTS = 8)
    int* out_layout_ids,
    char* out_layout_names,  // 8 * 32 = 256 bytes
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    uint8_t* out_valid,
    int* out_num_valid,
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    
    // 分配 workspace
    void* workspace = nullptr;
    size_t workspace_size = MAX_WORKSPACE_SIZE;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    
    int num_valid = 0;
    
    for (int i = 0; i < NUM_LAYOUTS; ++i) {
        LayoutResult result;
        memset(&result, 0, sizeof(result));
        
        test_single_layout(
            LAYOUT_CONFIGS[i], i,
            A_compressed_ptr, B_ptr, C_ptr,
            M, N, K,
            dtype, outdtype,
            warmup, repeat,
            workspace, workspace_size,
            cu_stream,
            &result);
        
        out_layout_ids[i] = result.layout_id;
        memcpy(out_layout_names + i * 32, result.layout_name, 32);
        out_lat_us[i] = result.lat_us;
        out_tops[i] = result.tops;
        out_workspace[i] = result.workspace;
        out_valid[i] = result.valid;
        
        if (result.valid) {
            num_valid++;
        }
    }
    
    *out_num_valid = num_valid;
    
    cudaFree(workspace);
    
    return 0;
}

/**
 * @brief 对矩阵进行 2:4 剪枝 (支持指定 order)
 */
int cusparselt_layout_prune_24(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    int order,  // 0=COL, 1=ROW
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cusparseOrder_t sp_order = (order == 0) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
    
    return prune_24_for_layout(input, output, rows, cols, dtype, sp_order, cu_stream);
}

/**
 * @brief 压缩稀疏矩阵 (支持指定 order)
 */
int64_t cusparselt_layout_compress(
    void* input, void* output,
    int64_t rows, int64_t cols,
    const char* dtype,
    int order,  // 0=COL, 1=ROW
    void* stream)
{
    if (ensure_handle() < 0) return -1;
    
    cudaStream_t cu_stream = stream ? (cudaStream_t)stream : nullptr;
    cusparseOrder_t sp_order = (order == 0) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
    
    return compress_for_layout(input, output, rows, cols, dtype, sp_order, cu_stream);
}

/**
 * @brief 获取压缩后大小
 */
int64_t cusparselt_layout_get_compressed_size(
    int64_t rows, int64_t cols,
    const char* dtype,
    int order)
{
    if (ensure_handle() < 0) return -1;
    
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) return -1;
    
    cusparseOrder_t sp_order = (order == 0) ? CUSPARSE_ORDER_COL : CUSPARSE_ORDER_ROW;
    int64_t ld = (sp_order == CUSPARSE_ORDER_COL) ? rows : cols;
    
    cusparseLtMatDescriptor_t matA;
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matA,
        rows, cols, ld,
        16, info.cuda_type, sp_order,
        CUSPARSELT_SPARSITY_50_PERCENT);
    
    if (status != CUSPARSE_STATUS_SUCCESS) return -1;
    
    size_t compressed_size = 0;
    status = cusparseLtSpMMACompressedSize(&g_handle, &matA, &compressed_size);
    
    cusparseLtMatDescriptorDestroy(&matA);
    
    return (status == CUSPARSE_STATUS_SUCCESS) ? (int64_t)compressed_size : -1;
}

/**
 * @brief 获取布局名称
 */
const char* cusparselt_layout_get_name(int layout_id) {
    if (layout_id < 0 || layout_id >= NUM_LAYOUTS) {
        return "INVALID";
    }
    return LAYOUT_CONFIGS[layout_id].name;
}

/**
 * @brief 获取布局数量
 */
int cusparselt_layout_get_count() {
    return NUM_LAYOUTS;
}

/**
 * @brief 检查 cuSPARSELt 是否可用
 */
int cusparselt_layout_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 获取最后一条错误信息
 */
const char* cusparselt_layout_search_get_last_error() {
    return tls_last_error.c_str();
}

}  // extern "C"
