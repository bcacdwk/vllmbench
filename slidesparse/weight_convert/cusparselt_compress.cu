/**
 * cuSPARSELt 2:4 结构化稀疏权重压缩库
 *
 * 功能：将满足 2:4 稀疏约束的权重压缩为 cuSPARSELt 硬件加速格式
 * 
 * 布局约定（与 GEMM 推理一致）：
 *   D = op(W) × op(A) = W^T × A
 *   - W: 稀疏权重，逻辑形状 [N, K]，opW = TRANSPOSE
 *        描述符存储 [K, N]（转置前形状），列主序
 *   - A: 稠密激活 [K, M]，列主序，opA = NON_TRANSPOSE
 *   - D: 输出 [N, M]，列主序
 *   
 * 即 TN-CC 格式：W^T * A = D (all column-major)
 * 
 * cuSPARSELt 维度约束 (INT8):
 *   - 稀疏矩阵: rows, cols, ld 必须是 32 的倍数
 *   - 稠密矩阵: rows, cols, ld 必须是 16 的倍数
 *
 * 注意：此库仅用于离线权重压缩，不执行实际 GEMM 运算。
 * 压缩后的权重可直接用于 cuSPARSELt SpMM 加速。
 *
 * 编译示例:
 *   nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC \
 *        -shared cusparselt_compress.cu -lcusparseLt -lcusparse -lcuda \
 *        -gencode=arch=compute_90,code=sm_90 -o cusparselt_compress.so
 */

#include <cusparseLt.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {

// =============================================================================
// 调试工具
// =============================================================================

// 调试开关（默认关闭）
static bool debug_enabled() {
    static bool flag = []() {
        const char* env = std::getenv("CUSPARSELT_COMPRESS_DEBUG");
        return env && std::strcmp(env, "1") == 0;
    }();
    return flag;
}

#define DEBUG_LOG(msg)                                                         \
    do {                                                                       \
        if (debug_enabled()) {                                                 \
            std::ostringstream oss;                                            \
            oss << "[cusparselt-compress] " << msg << std::endl;               \
            std::cerr << oss.str();                                            \
        }                                                                      \
    } while (0)

// =============================================================================
// 错误检查宏
// =============================================================================

#define CHECK_CUSPARSE(func)                                                   \
    do {                                                                       \
        cusparseStatus_t status = (func);                                      \
        if (status != CUSPARSE_STATUS_SUCCESS) {                               \
            std::ostringstream oss;                                            \
            oss << "cuSPARSELt error at " << __FILE__ << ":" << __LINE__       \
                << " - " << #func << " returned " << static_cast<int>(status); \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

#define CHECK_CUDA(func)                                                       \
    do {                                                                       \
        cudaError_t status = (func);                                           \
        if (status != cudaSuccess) {                                           \
            std::ostringstream oss;                                            \
            oss << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(status);                        \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

// =============================================================================
// 全局 cuSPARSELt 句柄管理
// =============================================================================

static cusparseLtHandle_t g_handle;
static bool g_handle_initialized = false;
static std::mutex g_handle_mutex;

static cusparseLtHandle_t get_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_initialized) {
        CHECK_CUSPARSE(cusparseLtInit(&g_handle));
        g_handle_initialized = true;
        DEBUG_LOG("cuSPARSELt handle initialized");
        
        // 注册退出清理
        std::atexit([]() {
            if (g_handle_initialized) {
                cusparseLtDestroy(&g_handle);
                g_handle_initialized = false;
                DEBUG_LOG("cuSPARSELt handle destroyed");
            }
        });
    }
    return g_handle;
}

// =============================================================================
// 压缩计划缓存
// =============================================================================

struct PlanKey {
    int M;  // 激活行数（用于构建计划，压缩时固定为 1024）
    int N;  // 权重行数 (out_features)
    int K;  // 权重列数 (in_features)
    
    bool operator<(const PlanKey& other) const {
        return std::tie(M, N, K) < std::tie(other.M, other.N, other.K);
    }
};

struct PlanContext {
    cusparseLtMatDescriptor_t matW;      // 稀疏权重描述符
    cusparseLtMatDescriptor_t matA;      // 稠密激活描述符
    cusparseLtMatDescriptor_t matD;      // 输出描述符
    cusparseLtMatmulDescriptor_t matmul; // 矩阵乘描述符
    cusparseLtMatmulAlgSelection_t alg;  // 算法选择
    cusparseLtMatmulPlan_t plan;         // 执行计划
    bool initialized = false;
};

static std::map<PlanKey, PlanContext> g_plan_cache;
static std::mutex g_plan_mutex;

static void destroy_plan(PlanContext& ctx) {
    if (!ctx.initialized) return;
    cusparseLtMatmulPlanDestroy(&ctx.plan);
    cusparseLtMatmulAlgSelectionDestroy(&ctx.alg);
    cusparseLtMatDescriptorDestroy(&ctx.matW);
    cusparseLtMatDescriptorDestroy(&ctx.matA);
    cusparseLtMatDescriptorDestroy(&ctx.matD);
    ctx.initialized = false;
}

static void cleanup_all_plans() {
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    for (auto& kv : g_plan_cache) {
        destroy_plan(kv.second);
    }
    g_plan_cache.clear();
}

static bool g_cleanup_registered = false;

/**
 * 获取或创建压缩计划
 * 
 * 布局说明（TN-CC 格式）：
 *   D[N,M] = W[N,K]^T * A[K,M]
 *   - 所有矩阵均为列主序 (Column-Major)
 *   - W: opW = TRANSPOSE
 *   - A: opA = NON_TRANSPOSE
 *   
 * Leading dimensions（列主序）：
 *   - ldW = N (W 是 [N,K] 但以列主序存储，ld = 行数)
 *   - ldA = K
 *   - ldD = N
 */
static PlanContext& get_or_create_plan(cusparseLtHandle_t handle, 
                                        int M, int N, int K) {
    PlanKey key{M, N, K};
    
    std::lock_guard<std::mutex> lock(g_plan_mutex);
    
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end()) {
        DEBUG_LOG("Plan cache hit: M=" << M << " N=" << N << " K=" << K);
        return it->second;
    }
    
    // 注册退出清理
    if (!g_cleanup_registered) {
        std::atexit(cleanup_all_plans);
        g_cleanup_registered = true;
    }
    
    DEBUG_LOG("Creating plan: M=" << M << " N=" << N << " K=" << K);
    
    auto [iter, inserted] = g_plan_cache.try_emplace(key);
    PlanContext& ctx = iter->second;
    
    // 布局配置：TN-CC（W转置，A不转置，全列主序）
    const auto order_col = CUSPARSE_ORDER_COL;
    const auto opW = CUSPARSE_OPERATION_TRANSPOSE;      // W^T
    const auto opA = CUSPARSE_OPERATION_NON_TRANSPOSE;  // A
    
    // 当 opW=TRANSPOSE 时，描述符中存储的是转置前的形状
    // W 逻辑形状 [N,K]，使用 TRANSPOSE 操作，描述符存储 [K,N]
    const int num_W_rows = K;  // 转置时：rows = K
    const int num_W_cols = N;  // 转置时：cols = N
    const int num_A_rows = K;  // A[K,M] 不转置
    const int num_A_cols = M;
    const int num_D_rows = N;  // D[N,M]
    const int num_D_cols = M;
    
    // Leading dimensions（列主序：ld = 行数）
    const int ldW = num_W_rows;  // ld = rows for col-major
    const int ldA = num_A_rows;
    const int ldD = num_D_rows;
    const unsigned alignment = 16;
    
    bool matW_ok = false, matA_ok = false, matD_ok = false;
    bool alg_ok = false, plan_ok = false;
    
    try {
        // Step 1: 稀疏权重 W，描述符存储转置前形状 [K,N]
        CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(
            &handle, &ctx.matW,
            num_W_rows, num_W_cols, ldW,  // [K, N], ld=K
            alignment,
            CUDA_R_8I,           // INT8 数据类型
            order_col,           // 列主序
            CUSPARSELT_SPARSITY_50_PERCENT));
        matW_ok = true;
        DEBUG_LOG("  matW (sparse) initialized: [" << num_W_rows << "," << num_W_cols << "] ld=" << ldW);
        
        // Step 2: 稠密激活 A[K,M]，列主序
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &ctx.matA,
            num_A_rows, num_A_cols, ldA,  // [K, M], ld=K
            alignment,
            CUDA_R_8I,
            order_col));
        matA_ok = true;
        DEBUG_LOG("  matA (dense) initialized: [" << num_A_rows << "," << num_A_cols << "] ld=" << ldA);
        
        // Step 3: 输出 D[N,M]，列主序，INT32 累加
        CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(
            &handle, &ctx.matD,
            num_D_rows, num_D_cols, ldD,  // [N, M], ld=N
            alignment,
            CUDA_R_32I,          // 输出类型
            order_col));
        matD_ok = true;
        DEBUG_LOG("  matD (output) initialized: [" << num_D_rows << "," << num_D_cols << "] ld=" << ldD);
        
        // Step 4: 矩阵乘描述符
        // D = W^T * A，其中 W 是稀疏的
        CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(
            &handle, &ctx.matmul,
            opW, opA,            // W^T * A
            &ctx.matW,           // 稀疏矩阵（第一个输入）
            &ctx.matA,           // 稠密矩阵（第二个输入）
            &ctx.matD,           // 输出
            &ctx.matD,           // D 类型参考
            CUSPARSE_COMPUTE_32I));
        DEBUG_LOG("  matmul descriptor initialized");
        
        // Step 5: 算法选择（使用默认算法 ID=0）
        CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(
            &handle, &ctx.alg, &ctx.matmul,
            CUSPARSELT_MATMUL_ALG_DEFAULT));
        
        // 显式设置算法 ID = 0
        int alg_id = 0;
        CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
            &handle, &ctx.alg,
            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &alg_id, sizeof(alg_id)));
        alg_ok = true;
        DEBUG_LOG("  algorithm selection initialized (alg_id=0)");
        
        // Step 6: 初始化计划
        CHECK_CUSPARSE(cusparseLtMatmulPlanInit(
            &handle, &ctx.plan, &ctx.matmul, &ctx.alg));
        plan_ok = true;
        DEBUG_LOG("  plan initialized");
        
        ctx.initialized = true;
        return ctx;
        
    } catch (...) {
        // 清理已创建的资源
        if (plan_ok) cusparseLtMatmulPlanDestroy(&ctx.plan);
        if (alg_ok) cusparseLtMatmulAlgSelectionDestroy(&ctx.alg);
        if (matW_ok) cusparseLtMatDescriptorDestroy(&ctx.matW);
        if (matA_ok) cusparseLtMatDescriptorDestroy(&ctx.matA);
        if (matD_ok) cusparseLtMatDescriptorDestroy(&ctx.matD);
        g_plan_cache.erase(iter);
        throw;
    }
}

}  // namespace

// =============================================================================
// C 导出接口
// =============================================================================

extern "C" {

/**
 * 查询压缩所需的缓冲区大小
 * 
 * @param N             权重行数 (out_features)
 * @param K             权重列数 (in_features)
 * @param compressed_size    [out] 压缩后数据大小（字节）
 * @param temp_buffer_size   [out] 临时缓冲区大小（字节）
 * 
 * 注意：M 固定为 1024，用于构建压缩计划
 */
void cusparselt_get_compress_sizes(
    int N, int K,
    size_t* compressed_size,
    size_t* temp_buffer_size
) {
    if (!compressed_size || !temp_buffer_size) {
        throw std::runtime_error("Output pointers cannot be null");
    }
    
    const int M = 1024;  // 固定 M 用于构建计划
    
    cusparseLtHandle_t handle = get_handle();
    PlanContext& ctx = get_or_create_plan(handle, M, N, K);
    
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
        &handle, &ctx.plan,
        compressed_size,
        temp_buffer_size));
    
    DEBUG_LOG("Query sizes: N=" << N << " K=" << K 
              << " compressed=" << *compressed_size 
              << " temp=" << *temp_buffer_size);
}

/**
 * 压缩 2:4 稀疏权重
 * 
 * @param input_weight       输入权重指针（GPU 内存，INT8，必须满足 2:4 稀疏）
 * @param compressed_weight  输出压缩数据指针（GPU 内存）
 * @param temp_buffer        临时缓冲区指针（GPU 内存，可为 NULL 自动分配）
 * @param N                  权重行数 (out_features)
 * @param K                  权重列数 (in_features)
 * @param stream             CUDA 流（可为 NULL 使用默认流）
 * 
 * 注意：
 *   - 输入权重必须满足 2:4 结构化稀疏约束
 *   - 权重布局为列主序 [N, K]
 */
void cusparselt_compress_weight(
    const int8_t* input_weight,
    void* compressed_weight,
    void* temp_buffer,
    int N, int K,
    cudaStream_t stream
) {
    if (!input_weight || !compressed_weight) {
        throw std::runtime_error("Input/output pointers cannot be null");
    }
    
    const int M = 1024;  // 固定 M 用于构建计划
    
    DEBUG_LOG("Compress: N=" << N << " K=" << K);
    
    cusparseLtHandle_t handle = get_handle();
    PlanContext& ctx = get_or_create_plan(handle, M, N, K);
    
    // 查询所需大小
    size_t compressed_size = 0;
    size_t temp_buffer_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(
        &handle, &ctx.plan,
        &compressed_size,
        &temp_buffer_size));
    
    // 处理临时缓冲区
    void* temp_ptr = temp_buffer;
    bool temp_allocated = false;
    if (temp_buffer_size > 0 && temp_ptr == nullptr) {
        CHECK_CUDA(cudaMalloc(&temp_ptr, temp_buffer_size));
        temp_allocated = true;
        DEBUG_LOG("  Allocated temp buffer: " << temp_buffer_size << " bytes");
    }
    
    // 分配稀疏性验证标志
    int* d_valid = nullptr;
    CHECK_CUDA(cudaMalloc(&d_valid, sizeof(int)));
    
    try {
        // Step 1: 验证 2:4 稀疏性
        CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(
            &handle, &ctx.matmul,
            input_weight, d_valid, stream));
        
        int h_valid = 0;
        CHECK_CUDA(cudaMemcpyAsync(&h_valid, d_valid, sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
        
        if (stream) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        
        CHECK_CUDA(cudaFree(d_valid));
        d_valid = nullptr;
        
        if (h_valid != 0) {
            throw std::runtime_error("Input weight does not satisfy 2:4 sparsity constraint");
        }
        DEBUG_LOG("  2:4 sparsity verified");
        
        // Step 2: 执行压缩
        CHECK_CUSPARSE(cusparseLtSpMMACompress(
            &handle, &ctx.plan,
            input_weight,
            compressed_weight,
            temp_ptr,
            stream));
        
        if (stream) {
            CHECK_CUDA(cudaStreamSynchronize(stream));
        } else {
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        DEBUG_LOG("  Compression completed");
        
    } catch (...) {
        if (d_valid) cudaFree(d_valid);
        if (temp_allocated && temp_ptr) cudaFree(temp_ptr);
        throw;
    }
    
    // 清理
    if (temp_allocated && temp_ptr) {
        CHECK_CUDA(cudaFree(temp_ptr));
    }
}

/**
 * 检查 cuSPARSELt 库是否可用
 * 
 * @return 1 表示可用，0 表示不可用
 */
int cusparselt_is_available() {
    try {
        get_handle();
        return 1;
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
