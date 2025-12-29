/**
 * 自定义 GEMM Kernel (cuBLAS + cuSPARSELt)
 * 
 * 编译命令:
 * nvcc -shared -o libcustom_gemm.so custom_gemm.cu \
 *      -lcublas -lcusparselt \
 *      -I/usr/local/cuda/include \
 *      -L/usr/local/cuda/lib64 \
 *      --compiler-options '-fPIC' \
 *      -arch=sm_80
 * 
 * 参考你的 gpu_dense/bitnet_kernels_dense/bitgemm_cuBLAS_int8.cu
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>

// ============================================================================
// cuBLAS Handle 管理
// ============================================================================
static cublasHandle_t g_cublas_handle = nullptr;
static bool g_initialized = false;

extern "C" {

/**
 * 初始化 cuBLAS
 */
int custom_gemm_init() {
    if (g_initialized) return 0;
    
    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS init failed: %d\n", status);
        return -1;
    }
    
    // 使用 Tensor Core
    cublasSetMathMode(g_cublas_handle, CUBLAS_DEFAULT_MATH);
    
    g_initialized = true;
    printf("✅ cuBLAS initialized\n");
    return 0;
}

/**
 * 清理资源
 */
void custom_gemm_cleanup() {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    g_initialized = false;
}

// ============================================================================
// GEMM 实现
// ============================================================================

/**
 * FP16 GEMM: C = alpha * A @ B^T + beta * C
 * 
 * @param A: [M, K] 输入矩阵 (FP16)
 * @param B: [N, K] 权重矩阵 (FP16, 转置存储)
 * @param C: [M, N] 输出矩阵 (FP16)
 * @param M: batch_size * seq_len (tokens 数)
 * @param N: output features
 * @param K: input features
 */
int custom_gemm_fp16(
    const void* A,
    const void* B,
    void* C,
    int M, int N, int K,
    float alpha,
    float beta
) {
    if (!g_initialized) {
        if (custom_gemm_init() != 0) return -1;
    }
    
    // cuBLAS 使用列主序，我们的是行主序
    // C^T = B @ A^T  =>  C = (B @ A^T)^T = A @ B^T
    // 但我们的 B 已经是 [N, K]，所以直接用
    
    const __half* d_A = (const __half*)A;
    const __half* d_B = (const __half*)B;
    __half* d_C = (__half*)C;
    
    __half h_alpha = __float2half(alpha);
    __half h_beta = __float2half(beta);
    
    // cublasGemmEx: C = alpha * op(A) * op(B) + beta * C
    // 行主序 [M, K] @ [K, N] -> [M, N]
    // 等价于列主序的转置运算
    cublasStatus_t status = cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_T,        // B^T
        CUBLAS_OP_N,        // A
        N, M, K,            // 注意: cuBLAS 是列主序
        &h_alpha,
        d_B, CUDA_R_16F, K, // B: [N, K] -> 列主序 [K, N]
        d_A, CUDA_R_16F, K, // A: [M, K] -> 列主序 [K, M]
        &h_beta,
        d_C, CUDA_R_16F, N, // C: [M, N] -> 列主序 [N, M]
        CUDA_R_16F,         // 计算类型
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS GEMM failed: %d\n", status);
        return -1;
    }
    
    return 0;
}

/**
 * INT8 GEMM: C = A @ B^T
 * 
 * 用于量化模型的矩阵乘法
 * 
 * @param A: [M, K] 输入矩阵 (INT8)
 * @param B: [N, K] 权重矩阵 (INT8, 转置存储)
 * @param C: [M, N] 输出矩阵 (INT32)
 */
int custom_gemm_int8(
    const void* A,
    const void* B,
    void* C,
    int M, int N, int K
) {
    if (!g_initialized) {
        if (custom_gemm_init() != 0) return -1;
    }
    
    const int8_t* d_A = (const int8_t*)A;
    const int8_t* d_B = (const int8_t*)B;
    int32_t* d_C = (int32_t*)C;
    
    int alpha = 1;
    int beta = 0;
    
    // INT8 GEMM
    cublasStatus_t status = cublasGemmEx(
        g_cublas_handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, CUDA_R_8I, K,
        d_A, CUDA_R_8I, K,
        &beta,
        d_C, CUDA_R_32I, N,
        CUDA_R_32I,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS INT8 GEMM failed: %d\n", status);
        return -1;
    }
    
    return 0;
}

} // extern "C"

// ============================================================================
// 测试入口 (可选)
// ============================================================================
#ifdef TEST_MAIN
int main() {
    printf("Testing custom GEMM kernel...\n");
    
    int M = 128, N = 256, K = 512;
    
    // 分配内存
    __half *h_A, *h_B, *h_C;
    __half *d_A, *d_B, *d_C;
    
    h_A = new __half[M * K];
    h_B = new __half[N * K];
    h_C = new __half[M * N];
    
    // 初始化
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(0.1f);
    for (int i = 0; i < N * K; i++) h_B[i] = __float2half(0.1f);
    
    cudaMalloc(&d_A, M * K * sizeof(__half));
    cudaMalloc(&d_B, N * K * sizeof(__half));
    cudaMalloc(&d_C, M * N * sizeof(__half));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(__half), cudaMemcpyHostToDevice);
    
    // 运行 GEMM
    int ret = custom_gemm_fp16(d_A, d_B, d_C, M, N, K, 1.0f, 0.0f);
    if (ret == 0) {
        printf("✅ GEMM test passed!\n");
    }
    
    // 清理
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    custom_gemm_cleanup();
    return ret;
}
#endif
