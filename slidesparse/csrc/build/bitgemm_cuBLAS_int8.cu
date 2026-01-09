/*
 * BitNet CUDA GEMM Implementation - cuBLAS版本
 * 
 * 这个文件实现了BitNet推理的高性能矩阵乘法，专门用于批量计算(M>1)
 * 使用cuBLAS GEMM API进行加速，支持int8×int8计算
 * 
 * 主要特点：
 * 1. 使用cuBLAS的高度优化GEMM实现
 * 2. 支持任意矩阵尺寸，无硬编码限制
 * 3. 自动选择最优算法和内存布局
 * 4. 简化的代码结构，易于维护
 */

#include <cublas_v2.h>    // cuBLAS API
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <mutex>          // 线程安全支持

/*
 * 全局cuBLAS句柄管理
 * 
 * 为了避免每次GEMM调用都创建/销毁cuBLAS句柄的巨大开销，
 * 我们使用全局句柄，在第一次调用时初始化，程序退出时清理
 */
static cublasHandle_t g_cublas_handle = nullptr;
static bool g_handle_initialized = false;
static std::mutex g_handle_mutex;  // 线程安全保护
static std::mutex g_cublas_call_mutex;


static void cublas_init_once(cublasHandle_t h) {
    // 只在创建后调用一次
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST); // 固定为 HOST
    // 如有需要，也可固定 math/atomics：
    // cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    // cublasSetAtomicsMode(h, CUBLAS_ATOMICS_NOT_ALLOWED);
}

// 修改 get_cublas_handle：创建后调用 cublas_init_once
static cublasHandle_t get_cublas_handle() {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    if (!g_handle_initialized) {
        cublasStatus_t status = cublasCreate(&g_cublas_handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "Error: Global cuBLAS handle creation failed: " << status << std::endl;
            throw std::runtime_error("Global cuBLAS handle creation failed");
        }
        cublas_init_once(g_cublas_handle);
        g_handle_initialized = true;
        std::atexit([](){
            if (g_handle_initialized && g_cublas_handle) {
                cublasDestroy(g_cublas_handle);
                g_cublas_handle = nullptr;
                g_handle_initialized = false;
            }
        });
    }
    return g_cublas_handle;
}


/*
 * C接口函数：bitlinear_int8xint8 - cuBLAS IMMA优化版本
 * 
 * 这是BitNet GEMM操作的外部接口，严格按照IMMA内核要求实现
 * 使用TN格式(第一个矩阵转置，第二个不转置)以获得最佳int8×int8性能
 * 
 * 参数说明：
 *   input0: int8输入激活矩阵 A[M, K] - INT8的量化输入数据，行主序存储
 *   input1: int8权重矩阵 W[N, K] - INT8的1/0/-1权重矩阵，行主序存储
 *   output0: int32输出矩阵 R[M, N] - 累加结果，行主序存储
 *   M, N, K: 实际的输入矩阵维度
 *   stream: CUDA流，用于异步执行
 * 
 * IMMA内核要求：
 *   1. 必须使用TN格式: CUBLAS_OP_T, CUBLAS_OP_N
 *   2. leading dimension必须是4的倍数
 * 
 * 计算过程：
 * 输入：
 *   input1 = W[N,K]_row, ldW=K    input0 = A[M,K]_row, ldA=K
 *   
 * 经过cuBLAS列主序读取后，得到：
 *   input1 = W[K,N]_col           input0 = A[K,M]_col
 *   
 * 经过乘法之前的T/N转置：
 *   input1 = W[N,K]_col           input0 = A[K,M]_col
 * 
 * GemmEx计算：
 *   R[N,M]_col = W[N,K]_col × A[K,M]_col   乘法维度 N,M,K
 * 
 * 经过地址行主序读出后，得到：
 *   output0 = R[M,N]_row, ldR=N
 * 
 */
extern "C" void bitlinear_int8_GEMM(int8_t *input0, int8_t *input1,
                                    int32_t *output0, int M, int N, int K,
                                    cudaStream_t stream = 0) {
  
  // === 获取全局cuBLAS句柄(懒加载，只在第一次调用时创建) ===
  cublasHandle_t handle = get_cublas_handle();


  // 串行化这个全局 handle 的使用
  std::lock_guard<std::mutex> call_lock(g_cublas_call_mutex);

  // 无论 stream 是否为 0，都要设置；避免遗留上一次的绑定
  cublasStatus_t st = cublasSetStream(handle, stream);
  if (st != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error: cublasSetStream failed: " << st << std::endl;
    throw std::runtime_error("cublasSetStream failed");
  }



  // === IMMA内核leading dimension要求 ===
  // IMMA内核要求leading dimension必须是4的倍数，但通常原始K值即可
  // 如果遇到性能问题，可以考虑向上对齐
  int ldW = K;  // 权重矩阵W[N,K]_row的leading dimension
  int ldA = K;  // 激活矩阵A[M,K]_row的leading dimension  
  int ldR = N;  // 输出矩阵R[N,M]_col的leading dimension

  // === 设置GEMM标量参数 ===
  const int32_t alpha = 1;  // 乘法标量：C = alpha * A * B + beta * C
  const int32_t beta = 0;   // 累加标量：设为0表示不累加到原C

  // === 执行cuBLAS GEMM - 严格TN格式 ===
  cublasStatus_t status = cublasGemmEx(
      handle,                   // cuBLAS句柄
      CUBLAS_OP_T,              // W矩阵操作：转置
      CUBLAS_OP_N,              // A矩阵操作：不转置
      N,                        // op(W) 转置后的行数
      M,                        // op(A) 的列数
      K,                        // 内积维度
      &alpha,                   // alpha标量参数
      input1,                   // Weight矩阵指针 W[N,K]_row
      CUDA_R_8I,                // Weight矩阵数据类型：int8
      ldW,                      // Weight矩阵的leading dimension = K
      input0,                   // Activation矩阵指针 A[M,K]_row
      CUDA_R_8I,                // Activation矩阵数据类型：int8
      ldA,                      // Activation矩阵的leading dimension = K
      &beta,                    // beta标量参数
      output0,                  // 输出矩阵指针 R[M,N]_row
      CUDA_R_32I,               // 输出矩阵数据类型：int32
      ldR,                      // 输出矩阵的leading dimension = N
      CUBLAS_COMPUTE_32I,       // 计算精度：int32累加 (IMMA内核支持)
      CUBLAS_GEMM_DEFAULT       // 使用默认算法选择
  );

  // === 检查GEMM执行状态 ===
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "Error: cublasGemmEx failed with status: " << status << std::endl;
    std::cerr << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << std::endl;
    throw std::runtime_error("cublasGemmEx execution failed");
  }

  // === 检查CUDA异步错误 ===
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    std::cerr << "CUDA error after cuBLAS operation: " << cudaGetErrorString(cuda_error) << std::endl;
    throw std::runtime_error("CUDA error detected");
  }
}