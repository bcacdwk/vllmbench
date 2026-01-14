// SPDX-License-Identifier: Apache-2.0
/**
 * @file layout_search_cusparselt.cu
 * @brief cuSPARSELt 布局搜索实现 (extern "C" 接口版本)
 *
 * 架构说明:
 * =========
 * 本文件提供 cuSPARSELt 稀疏矩阵乘法的布局配置搜索功能。
 * 测试 16 种布局组合 (转置 + 存储顺序 + D输出):
 *   - 转置 : TT, TN, NT, NN (4种)
 *   - A/B 排列 : RowCol, ColCol (2种)
 *   - D 输出 : Col, Row (2种)
 *
 *   - 使用 CUSPARSELT_PRUNE_SPMMA_TILE 剪枝模式
 *   - 使用 CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID 获取算法数量
 *   - 自适应倍增 Split-K 搜索
 *   - 支持 Segment-K (test_segment_k 参数)
 *   - TOPS 计算使用 2*M*N*K (不乘 0.5)
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

// 16 种布局组合 (4 转置 x 2 AB排列 x 2 D输出)
static constexpr int NUM_LAYOUTS = 16;

// 布局配置
struct LayoutConfig {
    cusparseOperation_t transA;
    cusparseOperation_t transB;
    cusparseOrder_t orderA;
    cusparseOrder_t orderB;
    cusparseOrder_t orderC;  // D 输出格式
    const char* name;
};

// 16 种布局配置
static const LayoutConfig LAYOUT_CONFIGS[NUM_LAYOUTS] = {
    // transA,                               transB,                               orderA,             orderB,             orderC,             name
    // D 输出为 ColMajor (前8种)
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "TT_RowCol_DCol"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "TN_RowCol_DCol"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "NT_RowCol_DCol"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "NN_RowCol_DCol"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "TT_ColCol_DCol"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "TN_ColCol_DCol"},  // 推荐
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "NT_ColCol_DCol"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, "NN_ColCol_DCol"},
    // D 输出为 RowMajor (后8种)
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "TT_RowCol_DRow"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "TN_RowCol_DRow"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "NT_RowCol_DRow"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "NN_RowCol_DRow"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "TT_ColCol_DRow"},
    {CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "TN_ColCol_DRow"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,     CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "NT_ColCol_DRow"},
    {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, "NN_ColCol_DRow"},
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

static int get_out_dtype_size(const char* outdtype) {
    if (strcmp(outdtype, "bf16") == 0 || strcmp(outdtype, "BF16") == 0) {
        return 2;
    } else if (strcmp(outdtype, "fp32") == 0 || strcmp(outdtype, "FP32") == 0) {
        return 4;
    }
    return 2;
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
    int best_alg_id;
    int best_split_k;
    int alg_count;
    int config_count;
    uint8_t valid;
};

// =============================================================================
// 单个布局测试 (与旧代码一致的完整搜索)
// =============================================================================

static int test_single_layout(
    const LayoutConfig& layout,
    int layout_id,
    int64_t N, int64_t K, int64_t M,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    int test_segment_k,
    void* shared_workspace, size_t workspace_size,
    cudaStream_t stream,
    LayoutResult* result)
{
    DtypeInfo info = get_dtype_info(dtype);
    if (!info.is_valid) {
        return 0;
    }
    
    cudaDataType_t out_type = get_out_dtype(outdtype);
    cusparseComputeType compute_type = get_compute_type(dtype);
    int out_elem_size = get_out_dtype_size(outdtype);
    
    result->layout_id = layout_id;
    strncpy(result->layout_name, layout.name, 31);
    result->layout_name[31] = '\0';
    result->valid = 0;
    result->best_alg_id = -1;
    result->best_split_k = 1;
    result->alg_count = 0;
    result->config_count = 0;
    
    // 计算矩阵维度
    bool isW_transposed = (layout.transA == CUSPARSE_OPERATION_TRANSPOSE);
    bool isA_transposed = (layout.transB == CUSPARSE_OPERATION_TRANSPOSE);
    bool isW_rowmajor = (layout.orderA == CUSPARSE_ORDER_ROW);
    bool isA_rowmajor = (layout.orderB == CUSPARSE_ORDER_ROW);
    bool isR_rowmajor = (layout.orderC == CUSPARSE_ORDER_ROW);
    
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
    
    size_t W_elems = (size_t)W_height * ldw;
    size_t A_elems = (size_t)A_height * lda;
    size_t R_elems = (size_t)R_height * ldr;
    
    size_t W_size = W_elems * info.elem_size;
    size_t A_size = A_elems * info.elem_size;
    size_t R_size = R_elems * out_elem_size;
    
    // 分配设备内存
    void *dW = nullptr, *dA = nullptr, *dR = nullptr;
    int *d_valid = nullptr;
    
    if (cudaMalloc(&dW, W_size) != cudaSuccess) return 0;
    if (cudaMalloc(&dA, A_size) != cudaSuccess) { cudaFree(dW); return 0; }
    if (cudaMalloc(&dR, R_size) != cudaSuccess) { cudaFree(dW); cudaFree(dA); return 0; }
    if (cudaMalloc(&d_valid, sizeof(int)) != cudaSuccess) { cudaFree(dW); cudaFree(dA); cudaFree(dR); return 0; }
    
    // 随机初始化
    {
        std::vector<int8_t> hW(W_elems), hA(A_elems);
        for (size_t i = 0; i < W_elems; ++i) hW[i] = (int8_t)(rand() % 256 - 128);
        for (size_t i = 0; i < A_elems; ++i) hA[i] = (int8_t)(rand() % 256 - 128);
        cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice);
        cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice);
    }
    cudaMemset(dR, 0, R_size);
    
    // 初始化描述符
    cusparseLtMatDescriptor_t matW, matA, matR;
    cusparseLtMatmulDescriptor_t matmul;
    
    // 稀疏矩阵 W 描述符
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &g_handle, &matW, num_W_rows, num_W_cols, ldw, 16,
        info.cuda_type, layout.orderA, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 稠密矩阵 A 描述符
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matA, num_A_rows, num_A_cols, lda, 16,
        info.cuda_type, layout.orderB);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 输出矩阵 R 描述符
    status = cusparseLtDenseDescriptorInit(
        &g_handle, &matR, num_R_rows, num_R_cols, ldr, 16,
        out_type, layout.orderC);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 矩阵乘法描述符
    status = cusparseLtMatmulDescriptorInit(
        &g_handle, &matmul, layout.transA, layout.transB,
        &matW, &matA, &matR, &matR, compute_type);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 设置稀疏矩阵指针
    status = cusparseLtMatmulDescSetAttribute(
        &g_handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dW, sizeof(dW));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 剪枝 (使用 TILE 模式与旧代码一致)
    status = cusparseLtSpMMAPrune(&g_handle, &matmul, dW, dW, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 检查剪枝
    status = cusparseLtSpMMAPruneCheck(&g_handle, &matmul, dW, d_valid, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    int is_valid = 0;
    cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    if (is_valid != 0) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    // 获取最大算法 ID (使用与旧代码一致的 API)
    int max_alg_id = -1;
    {
        cusparseLtMatmulAlgSelection_t alg_sel_tmp;
        status = cusparseLtMatmulAlgSelectionInit(
            &g_handle, &alg_sel_tmp, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (status == CUSPARSE_STATUS_SUCCESS) {
            cusparseLtMatmulAlgGetAttribute(
                &g_handle, &alg_sel_tmp, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
                &max_alg_id, sizeof(max_alg_id));
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
        }
    }
    
    if (max_alg_id < 0) {
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return 0;
    }
    
    result->alg_count = max_alg_id;
    
    // 创建 CUDA events
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    
    float alpha = 1.0f, beta = 0.0f;
    float best_lat_us = -1.0f;
    float best_tops = 0.0f;
    int64_t best_workspace = 0;
    int best_alg_id = -1;
    int best_split_k = 1;
    int config_count = 0;
    
    // === 双层网格搜索：外层遍历 alg_id，内层自适应调整 split_k_val ===
    for (int alg_id = 0; alg_id < max_alg_id; ++alg_id) {
        float best_lat_us_for_doubling = -1.0f;
        
        // 构建 split_k 候选列表
        std::vector<int> split_k_candidates;
        split_k_candidates.push_back(1);
        for (int sk = 2; sk <= K; sk *= 2) {
            split_k_candidates.push_back(sk);
        }
        if (test_segment_k) {
            split_k_candidates.push_back(-1);
        }
        
        bool stop_doubling = false;
        
        for (int split_k_val : split_k_candidates) {
            if (stop_doubling && split_k_val > 1) continue;
            
            cusparseLtMatmulAlgSelection_t alg_sel;
            status = cusparseLtMatmulAlgSelectionInit(
                &g_handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
            if (status != CUSPARSE_STATUS_SUCCESS) break;
            
            status = cusparseLtMatmulAlgSetAttribute(
                &g_handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                &alg_id, sizeof(alg_id));
            if (status != CUSPARSE_STATUS_SUCCESS) {
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
            cusparseStatus_t plan_status = cusparseLtMatmulPlanInit(
                &g_handle, &plan, &matmul, &alg_sel, workspace_size);
            if (plan_status != CUSPARSE_STATUS_SUCCESS) {
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                if (split_k_val > 1) stop_doubling = true;
                continue;
            }
            
            // 压缩
            size_t compressed_size = 0, compressed_buffer_size = 0;
            cusparseLtSpMMACompressedSize(&g_handle, &plan, &compressed_size, &compressed_buffer_size);
            
            void *dW_compressed = nullptr, *dW_compressedBuffer = nullptr;
            if (cudaMalloc(&dW_compressed, compressed_size) != cudaSuccess) {
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            if (compressed_buffer_size > 0) {
                if (cudaMalloc(&dW_compressedBuffer, compressed_buffer_size) != cudaSuccess) {
                    cudaFree(dW_compressed);
                    cusparseLtMatmulPlanDestroy(&plan);
                    cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                    continue;
                }
            }
            
            status = cusparseLtSpMMACompress(&g_handle, &plan, dW, dW_compressed, dW_compressedBuffer, stream);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                cudaFree(dW_compressed);
                if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
            // Workspace
            size_t ws_size = 0;
            cusparseLtMatmulGetWorkspace(&g_handle, &plan, &ws_size);
            void *d_workspace = (ws_size > 0 && ws_size <= workspace_size) ? shared_workspace : nullptr;
            
            // Warmup
            bool warmup_success = true;
            for (int w = 0; w < warmup; ++w) {
                status = cusparseLtMatmul(&g_handle, &plan, &alpha, dW_compressed, dA,
                                          &beta, dR, dR, d_workspace, &stream, 1);
                if (status != CUSPARSE_STATUS_SUCCESS) { warmup_success = false; break; }
            }
            cudaStreamSynchronize(stream);
            
            if (!warmup_success) {
                cudaFree(dW_compressed);
                if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
                cusparseLtMatmulPlanDestroy(&plan);
                cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
                continue;
            }
            
            // Benchmark
            cudaEventRecord(start_event, stream);
            for (int r = 0; r < repeat; ++r) {
                cusparseLtMatmul(&g_handle, &plan, &alpha, dW_compressed, dA,
                                 &beta, dR, dR, d_workspace, &stream, 1);
            }
            cudaEventRecord(stop_event, stream);
            cudaEventSynchronize(stop_event);
            
            float total_ms = 0.0f;
            cudaEventElapsedTime(&total_ms, start_event, stop_event);
            float avg_us = (total_ms * 1000.0f) / repeat;
            
            // TOPS 计算 (与旧代码一致，不乘 0.5)
            double flops = 2.0 * (double)M * (double)N * (double)K;
            double tops = (flops / (avg_us * 1e-6)) / 1e12;
            
            ++config_count;
            
            // 更新最佳结果
            if (best_alg_id < 0 || tops > best_tops) {
                best_lat_us = avg_us;
                best_tops = (float)tops;
                best_alg_id = alg_id;
                best_split_k = split_k_val;
                best_workspace = (int64_t)ws_size;
            }
            
            // 自适应倍增策略
            if (split_k_val >= 1) {
                if (best_lat_us_for_doubling < 0 || avg_us < best_lat_us_for_doubling) {
                    best_lat_us_for_doubling = avg_us;
                } else if (avg_us * 1.10f > best_lat_us_for_doubling && split_k_val > 1) {
                    stop_doubling = true;
                }
            }
            
            // 清理
            cudaFree(dW_compressed);
            if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
            cusparseLtMatmulPlanDestroy(&plan);
            cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        }  // end split_k loop
    }  // end alg_id loop
    
    // 填充结果
    result->config_count = config_count;
    result->valid = (best_alg_id >= 0) ? 1 : 0;
    if (best_alg_id >= 0) {
        result->lat_us = best_lat_us;
        result->tops = best_tops;
        result->best_alg_id = best_alg_id;
        result->best_split_k = best_split_k;
        result->workspace = best_workspace;
    }
    
    // 清理
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cudaFree(dW);
    cudaFree(dA);
    cudaFree(dR);
    cudaFree(d_valid);
    
    return result->valid ? 1 : 0;
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
    
    // 使用 TILE 模式与旧代码一致
    status = cusparseLtSpMMAPrune(
        &g_handle, &matA,
        input, output,
        CUSPARSELT_PRUNE_SPMMA_TILE,
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
    status = cusparseLtSpMMACompressedSize(&g_handle, &matA, &compressed_size, nullptr);
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
 * @brief 测试单个 (N,K,M) 的 16 种布局配置
 */
int cusparselt_layout_search_single(
    int64_t M, int64_t N, int64_t K,
    const char* dtype, const char* outdtype,
    int warmup, int repeat,
    int test_segment_k,
    // 输出数组 (大小 = NUM_LAYOUTS = 16)
    int* out_layout_ids,
    char* out_layout_names,  // 16 * 32 = 512 bytes
    float* out_lat_us,
    float* out_tops,
    int64_t* out_workspace,
    int* out_best_alg_id,
    int* out_best_split_k,
    int* out_alg_count,
    int* out_config_count,
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
            N, K, M,
            dtype, outdtype,
            warmup, repeat,
            test_segment_k,
            workspace, workspace_size,
            cu_stream,
            &result);
        
        out_layout_ids[i] = result.layout_id;
        memcpy(out_layout_names + i * 32, result.layout_name, 32);
        out_lat_us[i] = result.lat_us;
        out_tops[i] = result.tops;
        out_workspace[i] = result.workspace;
        out_best_alg_id[i] = result.best_alg_id;
        out_best_split_k[i] = result.best_split_k;
        out_alg_count[i] = result.alg_count;
        out_config_count[i] = result.config_count;
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
    status = cusparseLtSpMMACompressedSize(&g_handle, &matA, &compressed_size, nullptr);
    
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
