// SPDX-License-Identifier: Apache-2.0
/**
 * @file layout_search_cublaslt.cu
 * @brief cuBLASLt 布局搜索实现 (extern "C" 接口版本)
 *
 * 架构说明:
 * =========
 * 本文件提供 cuBLASLt 布局配置搜索功能，测试 8 种布局组合：
 *   - 转置 : TT, TN, NT, NN
 *   - A/B 排列 : RowCol, ColCol
 *   (D 输出固定为 ColMajor)
 *
 * 固定最优布局: T/N + Col/Col + Col (权重 W 在左)
 *
 * 编译方法:
 * ---------
 * nvcc -std=c++17 -O3 -Xcompiler -fPIC --shared \
 *      layout_search_cublaslt.cu -lcublasLt -lcublas \
 *      -o layout_search_cublaslt.so
 *
 * 主要接口:
 * ---------
 * - cublaslt_layout_search_single()  : 测试单个 (N,K,M) 的 8 种布局
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
#include <cublasLt.h>
#include <cublas_v2.h>

// =============================================================================
// 常量与类型定义
// =============================================================================

// 最大 workspace 大小: 512 MB
static constexpr size_t MAX_WORKSPACE_SIZE = 512ULL * 1024 * 1024;

// 8 种布局组合
static constexpr int NUM_LAYOUTS = 8;

// 布局组合枚举
struct LayoutConfig {
    cublasOperation_t transA;
    cublasOperation_t transB;
    cublasLtOrder_t orderA;
    cublasLtOrder_t orderB;
    const char* name;
};

// 所有8种布局配置
static const LayoutConfig LAYOUT_CONFIGS[NUM_LAYOUTS] = {
    // transA,               transB,               orderA,                  orderB,                  name
    {CUBLAS_OP_T, CUBLAS_OP_T, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "TT_RowCol"},
    {CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "TN_RowCol"},
    {CUBLAS_OP_N, CUBLAS_OP_T, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "NT_RowCol"},
    {CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_ORDER_ROW, CUBLASLT_ORDER_COL, "NN_RowCol"},
    {CUBLAS_OP_T, CUBLAS_OP_T, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "TT_ColCol"},
    {CUBLAS_OP_T, CUBLAS_OP_N, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "TN_ColCol"},  // 推荐
    {CUBLAS_OP_N, CUBLAS_OP_T, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "NT_ColCol"},
    {CUBLAS_OP_N, CUBLAS_OP_N, CUBLASLT_ORDER_COL, CUBLASLT_ORDER_COL, "NN_ColCol"},
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

#define CHECK_CUBLASLT(call) do { \
    cublasStatus_t err = (call); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        char buf[512]; \
        snprintf(buf, sizeof(buf), "cuBLASLt Error %d at %s:%d", \
                 (int)err, __FILE__, __LINE__); \
        set_error(buf); \
        return -1; \
    } \
} while (0)

// =============================================================================
// 全局资源
// =============================================================================

static cublasLtHandle_t g_lt_handle = nullptr;
static std::mutex g_handle_mutex;

static int ensure_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_lt_handle) {
        cublasStatus_t err = cublasLtCreate(&g_lt_handle);
        if (err != CUBLAS_STATUS_SUCCESS) {
            set_error("Failed to create cuBLASLt handle");
            return -1;
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
    return CUDA_R_16BF;  // 默认
}

static cublasComputeType_t get_compute_type(const char* dtype) {
    if (strcmp(dtype, "int8") == 0 || strcmp(dtype, "INT8") == 0) {
        return CUBLAS_COMPUTE_32I;
    }
    return CUBLAS_COMPUTE_32F;
}

// =============================================================================
// 布局搜索结果结构
// =============================================================================

struct LayoutResult {
    int layout_id;              // 布局索引 [0, 7]
    char layout_name[32];       // 布局名称
    float lat_us;               // 延迟 (微秒)
    float tops;                 // 吞吐量 (TOPS)
    int best_alg_id;            // 最佳算法 ID
    int64_t workspace;          // workspace 大小
    float waves_count;          // wave count
    uint8_t algo_data[64];      // 算法序列化数据
    uint8_t valid;              // 是否有效
};

// =============================================================================
// 单个布局测试
// =============================================================================

static int test_single_layout(
    const LayoutConfig& layout,
    int layout_id,
    void* A_ptr, void* B_ptr, void* C_ptr,
    int64_t M, int64_t N, int64_t K,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    void* workspace, size_t workspace_size,
    cudaStream_t stream,
    LayoutResult* result)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        set_error("Invalid dtype");
        return -1;
    }
    
    cudaDataType_t out_type = get_out_dtype(outdtype);
    cublasComputeType_t compute_type = get_compute_type(dtype);
    
    result->layout_id = layout_id;
    strncpy(result->layout_name, layout.name, 31);
    result->layout_name[31] = '\0';
    result->valid = 0;
    
    // 创建矩阵描述符
    // 对于 C = A^T * B
    // A: [K, M], B: [K, N], C: [M, N] (逻辑维度)
    cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    
    // 根据转置设置维度
    int64_t ldA, ldB, ldC;
    int64_t rowsA, colsA, rowsB, colsB;
    
    if (layout.transA == CUBLAS_OP_N) {
        rowsA = M; colsA = K;
    } else {
        rowsA = K; colsA = M;
    }
    
    if (layout.transB == CUBLAS_OP_N) {
        rowsB = K; colsB = N;
    } else {
        rowsB = N; colsB = K;
    }
    
    // Leading dimension
    if (layout.orderA == CUBLASLT_ORDER_COL) {
        ldA = rowsA;
    } else {
        ldA = colsA;
    }
    
    if (layout.orderB == CUBLASLT_ORDER_COL) {
        ldB = rowsB;
    } else {
        ldB = colsB;
    }
    
    ldC = M;  // C 固定列主序
    
    // 创建描述符
    cublasStatus_t status;
    
    status = cublasLtMatrixLayoutCreate(&layoutA, info.cuda_type, rowsA, colsA, ldA);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return 0;  // 不支持此布局
    }
    
    status = cublasLtMatmulDescCreate(&opDesc, compute_type, CUDA_R_32F);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutA);
        return 0;
    }
    
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &layout.transA, sizeof(layout.transA));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &layout.transB, sizeof(layout.transB));
    
    status = cublasLtMatrixLayoutCreate(&layoutB, info.cuda_type, rowsB, colsB, ldB);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;
    }
    
    status = cublasLtMatrixLayoutCreate(&layoutC, out_type, M, N, ldC);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;
    }
    
    // 设置 order
    cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &layout.orderA, sizeof(layout.orderA));
    cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &layout.orderB, sizeof(layout.orderB));
    cublasLtOrder_t orderC = CUBLASLT_ORDER_COL;
    cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &orderC, sizeof(orderC));
    
    // 创建偏好
    status = cublasLtMatmulPreferenceCreate(&pref);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;
    }
    
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size));
    
    // 获取启发式算法
    cublasLtMatmulHeuristicResult_t heurResult[8];
    int numResults = 0;
    
    status = cublasLtMatmulAlgoGetHeuristic(g_lt_handle, opDesc, layoutA, layoutB, layoutC, layoutC,
                                            pref, 8, heurResult, &numResults);
    
    if (status != CUBLAS_STATUS_SUCCESS || numResults == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;  // 此布局不支持
    }
    
    // 使用第一个算法进行测试
    cublasLtMatmulAlgo_t& algo = heurResult[0].algo;
    size_t ws_size = heurResult[0].workspaceSize;
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 验证算法可用性
    status = cublasLtMatmul(g_lt_handle, opDesc, &alpha,
                           A_ptr, layoutA, B_ptr, layoutB,
                           &beta, C_ptr, layoutC, C_ptr, layoutC,
                           &algo, workspace, ws_size, stream);
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        return 0;
    }
    
    cudaStreamSynchronize(stream);
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        cublasLtMatmul(g_lt_handle, opDesc, &alpha,
                      A_ptr, layoutA, B_ptr, layoutB,
                      &beta, C_ptr, layoutC, C_ptr, layoutC,
                      &algo, workspace, ws_size, stream);
    }
    cudaStreamSynchronize(stream);
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < repeat; ++i) {
        cublasLtMatmul(g_lt_handle, opDesc, &alpha,
                      A_ptr, layoutA, B_ptr, layoutB,
                      &beta, C_ptr, layoutC, C_ptr, layoutC,
                      &algo, workspace, ws_size, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms_total = 0;
    cudaEventElapsedTime(&ms_total, start, stop);
    
    float lat_us = (ms_total * 1000.0f) / repeat;
    float tops = (2.0 * M * N * K / 1e12) / (lat_us / 1e6);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 填充结果
    result->lat_us = lat_us;
    result->tops = tops;
    result->best_alg_id = 0;  // 使用第一个算法
    result->workspace = (int64_t)ws_size;
    result->waves_count = 0;  // 布局搜索不关心 waves
    memcpy(result->algo_data, &algo, sizeof(algo) < 64 ? sizeof(algo) : 64);
    result->valid = 1;
    
    // 清理
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);
    
    return 1;  // 成功
}

// =============================================================================
// 导出函数
// =============================================================================

extern "C" {

/**
 * @brief 测试单个 (N,K,M) 的 8 种布局配置
 *
 * @param A_ptr        A 矩阵设备指针
 * @param B_ptr        B 矩阵设备指针
 * @param C_ptr        C 矩阵设备指针
 * @param M            M 维度
 * @param N            N 维度
 * @param K            K 维度
 * @param dtype        输入数据类型 ("int8" / "fp8e4m3")
 * @param outdtype     输出数据类型 ("bf16" / "fp32")
 * @param warmup       预热次数
 * @param repeat       计时重复次数
 *
 * 输出参数 (每个数组大小为 8):
 * @param out_layout_ids     布局 ID
 * @param out_layout_names   布局名称 (每个 32 字节)
 * @param out_lat_us         延迟 (微秒)
 * @param out_tops           吞吐量 (TOPS)
 * @param out_workspace      workspace 大小
 * @param out_valid          是否有效
 * @param out_num_valid      有效布局数量
 * @param stream             CUDA 流 (可为 nullptr)
 *
 * @return 0 成功, -1 失败
 */
int cublaslt_layout_search_single(
    void* A_ptr, void* B_ptr, void* C_ptr,
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
        
        int ret = test_single_layout(
            LAYOUT_CONFIGS[i], i,
            A_ptr, B_ptr, C_ptr,
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
 * @brief 获取布局名称
 */
const char* cublaslt_layout_get_name(int layout_id) {
    if (layout_id < 0 || layout_id >= NUM_LAYOUTS) {
        return "INVALID";
    }
    return LAYOUT_CONFIGS[layout_id].name;
}

/**
 * @brief 获取布局数量
 */
int cublaslt_layout_get_count() {
    return NUM_LAYOUTS;
}

/**
 * @brief 检查 cuBLASLt 是否可用
 */
int cublaslt_layout_search_is_available() {
    return (ensure_handle() == 0) ? 1 : 0;
}

/**
 * @brief 获取最后一条错误信息
 */
const char* cublaslt_layout_search_get_last_error() {
    return tls_last_error.c_str();
}

}  // extern "C"
