/*
稀疏矩阵乘法Layout性能测试：遍历不同layout组合，记录最优算法ID
nvcc -o layout_benchmark layout_benchmark.cu -lcusparseLt && ./layout_benchmark

矩阵乘法定义：
  R[N,M] = W[N,K] * A[M,K]^T
  - W: 稀疏权重矩阵 [N,K]（左矩阵）
  - A: 稠密输入矩阵 [M,K]（右矩阵，需转置）
  - R: 输出矩阵 [N,M]
  - M: batch size（输入序列数）
  - N: 输出特征维度（W的行数）
  - K: 输入特征维度（共享维度）

测试的layout组合（4种主要layout）：
  1. T/N + Col/Col (opW=T, opA=N, orderW=Col, orderA=Col)
  2. N/T + Row/Row (opW=N, opA=T, orderW=Row, orderA=Row)
  3. N/N + Row/Col (opW=N, opA=N, orderW=Row, orderA=Col)
  4. T/T + Col/Row (opW=T, opA=T, orderW=Col, orderA=Row)

测试配置：
  - M列表: [16, 256, 2048, 16384]
  - (N,K) pairs: Wqkv(3840,2560), Wo(2560,2560), W13(13824,2560), W2(2560,6912)

对于每种layout组合，分别测试 R=Row 和 R=Col 两种输出布局，
记录 top3 算法ID 和对应的 TOPS。

输出目录结构：
  layout_benchmark_results/
    └── <GPU名称>/           (如 A100, B200)
        ├── n_3840_k_2560.csv
        ├── n_2560_k_2560.csv
        └── ...

CSV格式：按M排序，相同M下按4种layout顺序排列
*/

#include <cuda_runtime_api.h>
#include <cusparseLt.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>

using AB_t      = int8_t;
using C_t       = int;
using COMPUTE_t = int;

template <typename value_t>
struct cuda_type { };

template <>
struct cuda_type<int8_t> {
    static constexpr cudaDataType value = CUDA_R_8I;
};

template <>
struct cuda_type<int> {
    static constexpr cudaDataType value = CUDA_R_32I;
};

template <typename value_t>
struct cusparse_compute_type { };

template <>
struct cusparse_compute_type<int> {
    static constexpr cusparseComputeType value = CUSPARSE_COMPUTE_32I;
};

// (N, K) pairs 配置，对应不同的权重矩阵
struct NKPair {
    int n;
    int k;
    std::string name;
};

// 获取GPU名称（简化版，如 A100, B200）
std::string getGpuName() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string fullName = prop.name;
    
    // 提取GPU型号关键字（如 A100, H100, B200 等）
    // 常见格式: "NVIDIA A100-SXM4-40GB" -> "A100"
    //          "NVIDIA H100 PCIe" -> "H100"
    std::string shortName = fullName;
    
    // 移除 "NVIDIA " 前缀
    size_t nvidia_pos = fullName.find("NVIDIA ");
    if (nvidia_pos != std::string::npos) {
        shortName = fullName.substr(nvidia_pos + 7);
    }
    
    // 提取第一个空格或连字符之前的部分作为GPU型号
    size_t end_pos = shortName.find_first_of(" -");
    if (end_pos != std::string::npos) {
        shortName = shortName.substr(0, end_pos);
    }
    
    // 如果提取失败，使用清理后的完整名称（替换空格和特殊字符）
    if (shortName.empty()) {
        shortName = fullName;
        for (char& c : shortName) {
            if (c == ' ' || c == '-' || c == '/') c = '_';
        }
    }
    
    return shortName;
}

// 创建目录（递归创建）
bool createDirectory(const std::string& path) {
    size_t pos = 0;
    std::string dir;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        dir = path.substr(0, pos);
        mkdir(dir.c_str(), 0755);
    }
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
}

#define CHECK_CUDA(func)                                                        \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        std::printf("CUDA API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n",  \
                   __LINE__, cudaGetErrorString(status), status);               \
        return -1;                                                              \
    }                                                                           \
}

#define CHECK_CUDA_MAIN(func)                                                   \
{                                                                               \
    cudaError_t status = (func);                                                \
    if (status != cudaSuccess) {                                                \
        std::printf("CUDA API 调用失败，位置：第 %d 行，错误信息：%s (错误代码：%d)\n",  \
                   __LINE__, cudaGetErrorString(status), status);               \
        return EXIT_FAILURE;                                                    \
    }                                                                           \
}

// Layout配置结构体
struct LayoutConfig {
    std::string name;           // 配置名称，如 "T/N+Col/Col"
    cusparseOperation_t opW;    // 稀疏矩阵W的操作（转置/不转置）
    cusparseOperation_t opA;    // 稠密矩阵A的操作（转置/不转置）
    cusparseOrder_t orderW;     // 稀疏矩阵W的存储顺序（行/列主序）
    cusparseOrder_t orderA;     // 稠密矩阵A的存储顺序（行/列主序）
    bool isPrimary;             // 是否为主要测试（预期可工作）
};

// 算法结果结构体
struct AlgResult {
    int alg_id;
    double throughput;  // TOPS
    float time_ms;
};

// 比较函数：按吞吐量降序排序
bool compareByThroughput(const AlgResult& a, const AlgResult& b) {
    return a.throughput > b.throughput;
}

// 获取操作名称字符串
const char* getOpName(cusparseOperation_t op) {
    return (op == CUSPARSE_OPERATION_TRANSPOSE) ? "T" : "N";
}

// 获取顺序名称字符串
const char* getOrderName(cusparseOrder_t order) {
    return (order == CUSPARSE_ORDER_COL) ? "Col" : "Row";
}

// 测试单个layout配置的函数
// 返回值：0=成功，-1=CUDA错误，10=操作不支持
// out_max_alg_id：输出最大算法ID
int testLayoutConfig(
    const LayoutConfig& config,
    cusparseOrder_t orderR,  // 输出矩阵R的存储顺序
    int M, int N, int K,
    int num_runs,
    cudaEvent_t& start,
    cudaEvent_t& stop,
    std::vector<AlgResult>& results,  // 输出：所有有效算法的结果
    int& out_max_alg_id  // 输出：最大允许算法ID
) {
    out_max_alg_id = -1;
    results.clear();

    // 矩阵乘法: R = W * A
    // W: 稀疏矩阵，A: 稠密矩阵，R: 输出矩阵
    // 根据操作和顺序确定实际的矩阵维度

    auto opW           = config.opW;
    auto opA           = config.opA;
    auto orderW        = config.orderW;
    auto orderA        = config.orderA;
    auto type_AB       = cuda_type<AB_t>::value;
    auto type_C        = cuda_type<C_t>::value;
    auto compute_type  = cusparse_compute_type<COMPUTE_t>::value;
    unsigned alignment = 16;

    // 根据cuSPARSELt的矩阵乘法定义：
    // C = alpha * op(A) * op(B) + beta * C
    // 这里 matA=W(稀疏), matB=A(稠密), matC=R(输出)
    // 
    // 用户定义的矩阵乘法: R[N,M] = W[N,K] * A[M,K]^T
    //   - W: 稀疏权重矩阵 [N,K]
    //   - A: 稠密输入矩阵 [M,K]，需要转置后参与运算
    //   - R: 输出矩阵 [N,M]
    //
    // op(W) 的逻辑维度: [N, K]（不转置）或 [K, N]（转置）
    // op(A) 的逻辑维度: [K, M]（转置后）或 [M, K]（不转置）

    bool isW_transposed = (opW == CUSPARSE_OPERATION_TRANSPOSE);
    bool isA_transposed = (opA == CUSPARSE_OPERATION_TRANSPOSE);
    bool isW_rowmajor   = (orderW == CUSPARSE_ORDER_ROW);
    bool isA_rowmajor   = (orderA == CUSPARSE_ORDER_ROW);
    bool isR_rowmajor   = (orderR == CUSPARSE_ORDER_ROW);

    // 稀疏矩阵W的存储维度
    // W[N,K]：不转置时存储为[N,K]，转置时存储为[K,N]
    int num_W_rows = isW_transposed ? K : N;
    int num_W_cols = isW_transposed ? N : K;
    // 稠密矩阵A的存储维度
    // A[M,K]：转置时存储为[M,K]（逻辑维度[K,M]），不转置时存储为[K,M]
    int num_A_rows = isA_transposed ? M : K;
    int num_A_cols = isA_transposed ? K : M;
    // 输出矩阵R的维度 [N,M]
    int num_R_rows = N;
    int num_R_cols = M;

    // 计算leading dimension
    int ldw = isW_rowmajor ? num_W_cols : num_W_rows;
    int lda = isA_rowmajor ? num_A_cols : num_A_rows;
    int ldr = isR_rowmajor ? num_R_cols : num_R_rows;

    // 计算存储高度（用于计算元素总数）
    int W_height = isW_rowmajor ? num_W_rows : num_W_cols;
    int A_height = isA_rowmajor ? num_A_rows : num_A_cols;
    int R_height = isR_rowmajor ? num_R_rows : num_R_cols;

    size_t W_elems = static_cast<size_t>(W_height) * ldw;
    size_t A_elems = static_cast<size_t>(A_height) * lda;
    size_t R_elems = static_cast<size_t>(R_height) * ldr;

    size_t W_size = W_elems * sizeof(AB_t);
    size_t A_size = A_elems * sizeof(AB_t);
    size_t R_size = R_elems * sizeof(C_t);

    // 分配并初始化主机内存
    std::vector<AB_t> hW(W_elems);
    std::vector<AB_t> hA(A_elems);
    std::vector<C_t>  hR(R_elems, static_cast<C_t>(0));

    // 初始化稀疏矩阵W
    for (size_t i = 0; i < W_elems; ++i) {
        hW[i] = static_cast<AB_t>(std::rand() % 256 - 128);
    }
    // 初始化稠密矩阵A
    for (size_t i = 0; i < A_elems; ++i) {
        hA[i] = static_cast<AB_t>(std::rand() % 256 - 128);
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    AB_t *dW = nullptr, *dA = nullptr;
    C_t *dR = nullptr, *dD = nullptr;
    int *d_valid = nullptr;

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dW), W_size));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dA), A_size));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&dR), R_size));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_valid), sizeof(int)));

    dD = dR;

    CHECK_CUDA(cudaMemcpy(dW, hW.data(), W_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), A_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dR, hR.data(), R_size, cudaMemcpyHostToDevice));

    cusparseLtHandle_t           handle;
    cusparseLtMatDescriptor_t    matW, matA, matR;
    cusparseLtMatmulDescriptor_t matmul;
    cudaStream_t                 stream = nullptr;

    cusparseStatus_t init_status = cusparseLtInit(&handle);
    if (init_status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("cusparseLtInit 失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(init_status), init_status);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return -1;
    }

    // 初始化稀疏矩阵W描述符
    cusparseStatus_t status = cusparseLtStructuredDescriptorInit(
        &handle, &matW, num_W_rows, num_W_cols, ldw, alignment,
        type_AB, orderW, CUSPARSELT_SPARSITY_50_PERCENT);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("稀疏矩阵W描述符初始化失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return (status == CUSPARSE_STATUS_NOT_SUPPORTED) ? 10 : -1;
    }

    // 初始化稠密矩阵A描述符
    status = cusparseLtDenseDescriptorInit(
        &handle, &matA, num_A_rows, num_A_cols, lda, alignment,
        type_AB, orderA);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("稠密矩阵A描述符初始化失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return (status == CUSPARSE_STATUS_NOT_SUPPORTED) ? 10 : -1;
    }

    // 初始化输出矩阵R描述符
    status = cusparseLtDenseDescriptorInit(
        &handle, &matR, num_R_rows, num_R_cols, ldr, alignment,
        type_C, orderR);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("输出矩阵R描述符初始化失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return (status == CUSPARSE_STATUS_NOT_SUPPORTED) ? 10 : -1;
    }

    // 初始化矩阵乘法描述符
    status = cusparseLtMatmulDescriptorInit(
        &handle, &matmul, opW, opA, &matW, &matA, &matR, &matR, compute_type);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("矩阵乘法描述符初始化失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return (status == CUSPARSE_STATUS_NOT_SUPPORTED) ? 10 : -1;
    }

    // 设置稀疏矩阵指针
    status = cusparseLtMatmulDescSetAttribute(
        &handle, &matmul, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &dW, sizeof(dW));
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("设置稀疏矩阵指针失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return (status == CUSPARSE_STATUS_NOT_SUPPORTED) ? 10 : -1;
    }

    // 对稀疏矩阵进行剪枝
    status = cusparseLtSpMMAPrune(
        &handle, &matmul, dW, dW, CUSPARSELT_PRUNE_SPMMA_TILE, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("稀疏矩阵剪枝失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return (status == CUSPARSE_STATUS_NOT_SUPPORTED) ? 10 : -1;
    }

    // 检查剪枝有效性
    status = cusparseLtSpMMAPruneCheck(&handle, &matmul, dW, d_valid, stream);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        std::printf("剪枝检查失败: %s (错误代码：%d)\n",
                   cusparseLtGetErrorString(status), status);
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return -1;
    }

    int is_valid = 0;
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (is_valid != 0) {
        std::printf("矩阵剪枝验证失败\n");
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return -1;
    }

    // 查询最大算法ID
    int max_alg_id = -1;
    {
        cusparseLtMatmulAlgSelection_t alg_sel_tmp;
        status = cusparseLtMatmulAlgSelectionInit(
            &handle, &alg_sel_tmp, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            std::printf("算法选择初始化失败: %s\n", cusparseLtGetErrorString(status));
            cusparseLtMatDescriptorDestroy(&matW);
            cusparseLtMatDescriptorDestroy(&matA);
            cusparseLtMatDescriptorDestroy(&matR);
            cusparseLtDestroy(&handle);
            cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
            return -1;
        }
        cusparseLtMatmulAlgGetAttribute(
            &handle, &alg_sel_tmp, CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID,
            &max_alg_id, sizeof(max_alg_id));
        cusparseLtMatmulAlgSelectionDestroy(&alg_sel_tmp);
    }

    if (max_alg_id < 0) {
        std::printf("无法获取有效的最大算法ID\n");
        cusparseLtMatDescriptorDestroy(&matW);
        cusparseLtMatDescriptorDestroy(&matA);
        cusparseLtMatDescriptorDestroy(&matR);
        cusparseLtDestroy(&handle);
        cudaFree(dW); cudaFree(dA); cudaFree(dR); cudaFree(d_valid);
        return -1;
    }

    std::cout << "  最大算法ID: " << max_alg_id << std::endl;
    out_max_alg_id = max_alg_id;  // 输出最大算法ID

    // 遍历所有算法ID
    for (int alg_id = 0; alg_id <= max_alg_id; ++alg_id) {
        AB_t *dW_compressed = nullptr;
        void *dW_compressedBuffer = nullptr;
        void *d_workspace = nullptr;
        bool record_valid = true;
        bool selection_created = false;
        bool plan_created = false;

        cusparseLtMatmulAlgSelection_t alg_sel;
        cusparseLtMatmulPlan_t plan;

        auto cleanup = [&]() {
            if (d_workspace) cudaFree(d_workspace);
            if (dW_compressedBuffer) cudaFree(dW_compressedBuffer);
            if (dW_compressed) cudaFree(dW_compressed);
            if (plan_created) cusparseLtMatmulPlanDestroy(&plan);
            if (selection_created) cusparseLtMatmulAlgSelectionDestroy(&alg_sel);
        };

        // 初始化算法选择
        status = cusparseLtMatmulAlgSelectionInit(
            &handle, &alg_sel, &matmul, CUSPARSELT_MATMUL_ALG_DEFAULT);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cleanup();
            continue;
        }
        selection_created = true;

        // 设置算法ID
        status = cusparseLtMatmulAlgSetAttribute(
            &handle, &alg_sel, CUSPARSELT_MATMUL_ALG_CONFIG_ID,
            &alg_id, sizeof(alg_id));
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cleanup();
            continue;
        }

        // 初始化执行计划
        status = cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cleanup();
            continue;
        }
        plan_created = true;

        // 获取压缩尺寸
        size_t compressed_size = 0;
        size_t compressed_buffer_size = 0;
        status = cusparseLtSpMMACompressedSize(
            &handle, &plan, &compressed_size, &compressed_buffer_size);
        if (status != CUSPARSE_STATUS_SUCCESS || compressed_size == 0) {
            cleanup();
            continue;
        }

        // 分配压缩矩阵内存
        if (cudaMalloc(reinterpret_cast<void **>(&dW_compressed), compressed_size) != cudaSuccess) {
            cleanup();
            continue;
        }

        if (compressed_buffer_size > 0) {
            if (cudaMalloc(&dW_compressedBuffer, compressed_buffer_size) != cudaSuccess) {
                cleanup();
                continue;
            }
        }

        // 压缩稀疏矩阵
        status = cusparseLtSpMMACompress(
            &handle, &plan, dW, dW_compressed, dW_compressedBuffer, stream);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cleanup();
            continue;
        }

        // 获取工作空间大小
        size_t workspace_size = 0;
        status = cusparseLtMatmulGetWorkspace(&handle, &plan, &workspace_size);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cleanup();
            continue;
        }

        if (workspace_size > 0) {
            if (cudaMalloc(&d_workspace, workspace_size) != cudaSuccess) {
                cleanup();
                continue;
            }
        }

        // 预热
        cudaMemcpy(dR, hR.data(), R_size, cudaMemcpyHostToDevice);
        status = cusparseLtMatmul(
            &handle, &plan, &alpha, dW_compressed, dA,
            &beta, dR, dD, d_workspace, nullptr, 0);
        if (status != CUSPARSE_STATUS_SUCCESS) {
            cleanup();
            continue;
        }

        // 计时测试
        float total_time = 0.0f;
        for (int run = 0; run < num_runs; ++run) {
            cudaMemcpy(dR, hR.data(), R_size, cudaMemcpyHostToDevice);
            cudaEventRecord(start);

            status = cusparseLtMatmul(
                &handle, &plan, &alpha, dW_compressed, dA,
                &beta, dR, dD, d_workspace, nullptr, 0);
            if (status != CUSPARSE_STATUS_SUCCESS) {
                record_valid = false;
                break;
            }

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float milliseconds = 0.0f;
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_time += milliseconds;
        }

        if (record_valid && total_time > 0.0f) {
            float avg_time = total_time / num_runs;
            double ops = 2.0 * static_cast<double>(M) * N * K;
            double throughput = ops / (avg_time / 1000.0) / 1e12;

            AlgResult res;
            res.alg_id = alg_id;
            res.throughput = throughput;
            res.time_ms = avg_time;
            results.push_back(res);
        }

        cleanup();
    }

    // 清理资源
    cusparseLtMatDescriptorDestroy(&matW);
    cusparseLtMatDescriptorDestroy(&matA);
    cusparseLtMatDescriptorDestroy(&matR);
    cusparseLtDestroy(&handle);
    cudaFree(dW);
    cudaFree(dA);
    cudaFree(dR);
    cudaFree(d_valid);

    // 按吞吐量排序
    std::sort(results.begin(), results.end(), compareByThroughput);

    return 0;
}

int main() {
    std::srand(static_cast<unsigned>(time(nullptr)));

    // ============================================================
    // 配置参数
    // ============================================================
    
    // M值列表（dense矩阵A的行数，即batch size）
	std::vector<int> m_list = {16, 256, 2048, 16384};
    
    const int num_runs = 10;  // 每个配置运行次数

    // (N, K) pairs 配置，对应不同的权重矩阵
    std::vector<NKPair> nk_pairs = {
        {3840, 2560, "Wqkv"},    // Wqkv: (3840, 2560)
        {2560, 2560, "Wo"},      // Wo:   (2560, 2560)
        {13824, 2560, "W13"},    // W13:  (13824, 2560)
        {2560, 6912, "W2"}       // W2:   (2560, 6912)
    };

    // 定义4种主要layout配置（按指定顺序）
    std::vector<LayoutConfig> configs = {
        {"T/N+Col/Col", CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_ORDER_COL, CUSPARSE_ORDER_COL, true},
        {"N/T+Row/Row", CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
         CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_ROW, true},
        {"N/N+Row/Col", CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
         CUSPARSE_ORDER_ROW, CUSPARSE_ORDER_COL, true},
        {"T/T+Col/Row", CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
         CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW, true},
    };

    // ============================================================
    // 获取GPU信息并创建输出目录
    // ============================================================
    std::string gpuName = getGpuName();
    std::cout << "========================================" << std::endl;
    std::cout << "Layout性能测试" << std::endl;
    std::cout << "检测到GPU: " << gpuName << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 创建输出目录: layout_benchmark_results/<GPU名称>/
    std::string baseDir = "layout_benchmark_results";
    std::string outputDir = baseDir + "/" + gpuName;
    
    createDirectory(outputDir);
    std::cout << "输出目录: " << outputDir << std::endl;
    
    // ============================================================
    // 打印测试配置
    // ============================================================
    std::cout << "\n测试配置:" << std::endl;
    std::cout << "  M值列表: [";
    for (size_t i = 0; i < m_list.size(); ++i) {
        std::cout << m_list[i];
        if (i < m_list.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "  (N, K) pairs:" << std::endl;
    for (const auto& nk : nk_pairs) {
        std::cout << "    " << nk.name << ": (" << nk.n << ", " << nk.k << ")" << std::endl;
    }
    
    std::cout << "  Layout配置:" << std::endl;
    for (const auto& cfg : configs) {
        std::cout << "    " << cfg.name << std::endl;
    }
    
    int total_nk = nk_pairs.size();
    int total_m = m_list.size();
    int total_layouts = configs.size();
    std::cout << "  总CSV文件数: " << total_nk << std::endl;
    std::cout << "  每个CSV测试数: " << total_m << " × " << total_layouts << " = " 
              << (total_m * total_layouts) << std::endl;

    cudaEvent_t start, stop;
    CHECK_CUDA_MAIN(cudaEventCreate(&start));
    CHECK_CUDA_MAIN(cudaEventCreate(&stop));

    // ============================================================
    // 外层循环：遍历(N,K) pairs，每个pair生成一个CSV文件
    // ============================================================
    int nk_idx = 0;
    for (const auto& nk : nk_pairs) {
        nk_idx++;
        int N = nk.n;
        int K = nk.k;
        
        std::cout << "\n########################################" << std::endl;
        std::cout << "处理 [" << nk_idx << "/" << total_nk << "]: " 
                  << nk.name << " (N=" << N << ", K=" << K << ")" << std::endl;
        std::cout << "########################################" << std::endl;
        
        // 构造CSV文件路径: n_XXX_k_XXX.csv
        std::string csvFileName = "n_" + std::to_string(N) + "_k_" + std::to_string(K) + ".csv";
        std::string csvPath = outputDir + "/" + csvFileName;
        
        // 打开CSV文件
        std::ofstream csv(csvPath);
        if (!csv.is_open()) {
            std::cerr << "无法创建结果文件 " << csvPath << std::endl;
            continue;
        }
        
        // CSV表头：M在前，然后是Layout
        csv << "M,Layout,N,K,"
            << "R_Row_MaxAlgID,R_Row_Top1_ID,R_Row_Top2_ID,R_Row_Top3_ID,"
            << "R_Row_Top1_TOPS,R_Row_Top2_TOPS,R_Row_Top3_TOPS,"
            << "R_Col_MaxAlgID,R_Col_Top1_ID,R_Col_Top2_ID,R_Col_Top3_ID,"
            << "R_Col_Top1_TOPS,R_Col_Top2_TOPS,R_Col_Top3_TOPS\n";
        
        // ============================================================
        // 内层循环：遍历M值（外层），然后遍历layout配置（内层）
        // 按M排序，相同M下按4种layout顺序排列
        // ============================================================
        for (int M : m_list) {
            std::cout << "\n  ----------------------------------------" << std::endl;
            std::cout << "  测试 M=" << M << ", N=" << N << ", K=" << K << std::endl;
            std::cout << "  ----------------------------------------" << std::endl;
            
            // 遍历4种layout配置
            for (const auto& config : configs) {
                std::cout << "\n    测试Layout: " << config.name << std::endl;
                std::cout << "      opW=" << getOpName(config.opW)
                          << ", opA=" << getOpName(config.opA)
                          << ", orderW=" << getOrderName(config.orderW)
                          << ", orderA=" << getOrderName(config.orderA) << std::endl;

                std::vector<AlgResult> results_row, results_col;
                bool row_valid = false, col_valid = false;
                int max_alg_id_row = -1, max_alg_id_col = -1;

                // 测试R=Row布局
                std::cout << "\n      测试 R=Row 布局..." << std::endl;
                int ret_row = testLayoutConfig(
                    config, CUSPARSE_ORDER_ROW, M, N, K, num_runs, start, stop, results_row, max_alg_id_row);

                if (ret_row == 0 && !results_row.empty()) {
                    row_valid = true;
                    std::cout << "        R=Row 有效，共 " << results_row.size() << " 个算法" << std::endl;
                    int show_count = std::min(3, static_cast<int>(results_row.size()));
                    for (int i = 0; i < show_count; ++i) {
                        std::cout << "          Top" << (i+1) << ": AlgID=" << results_row[i].alg_id
                                  << ", " << results_row[i].throughput << " TOPS" << std::endl;
                    }
                } else if (ret_row == 10) {
                    std::cout << "        R=Row 操作不支持 (错误代码10)，跳过" << std::endl;
                } else {
                    std::cout << "        R=Row 测试失败或无有效算法" << std::endl;
                }

                // 测试R=Col布局
                std::cout << "\n      测试 R=Col 布局..." << std::endl;
                int ret_col = testLayoutConfig(
                    config, CUSPARSE_ORDER_COL, M, N, K, num_runs, start, stop, results_col, max_alg_id_col);

                if (ret_col == 0 && !results_col.empty()) {
                    col_valid = true;
                    std::cout << "        R=Col 有效，共 " << results_col.size() << " 个算法" << std::endl;
                    int show_count = std::min(3, static_cast<int>(results_col.size()));
                    for (int i = 0; i < show_count; ++i) {
                        std::cout << "          Top" << (i+1) << ": AlgID=" << results_col[i].alg_id
                                  << ", " << results_col[i].throughput << " TOPS" << std::endl;
                    }
                } else if (ret_col == 10) {
                    std::cout << "        R=Col 操作不支持 (错误代码10)，跳过" << std::endl;
                } else {
                    std::cout << "        R=Col 测试失败或无有效算法" << std::endl;
                }

                // 写入CSV（即使无效也写入，用-1和0表示）
                csv << M << "," << config.name << "," << N << "," << K << ",";

                // R=Row的MaxAlgID和Top3
                if (row_valid) {
                    csv << max_alg_id_row << ",";
                    for (int i = 0; i < 3; ++i) {
                        if (i < static_cast<int>(results_row.size())) {
                            csv << results_row[i].alg_id;
                        } else {
                            csv << "-1";
                        }
                        csv << ",";
                    }
                    for (int i = 0; i < 3; ++i) {
                        if (i < static_cast<int>(results_row.size())) {
                            csv << results_row[i].throughput;
                        } else {
                            csv << "0";
                        }
                        csv << ",";
                    }
                } else {
                    csv << "-1,-1,-1,-1,0,0,0,";
                }

                // R=Col的MaxAlgID和Top3
                if (col_valid) {
                    csv << max_alg_id_col << ",";
                    for (int i = 0; i < 3; ++i) {
                        if (i < static_cast<int>(results_col.size())) {
                            csv << results_col[i].alg_id;
                        } else {
                            csv << "-1";
                        }
                        csv << ",";
                    }
                    for (int i = 0; i < 3; ++i) {
                        if (i < static_cast<int>(results_col.size())) {
                            csv << results_col[i].throughput;
                        } else {
                            csv << "0";
                        }
                        if (i < 2) csv << ",";
                    }
                } else {
                    csv << "-1,-1,-1,-1,0,0,0";
                }

                csv << "\n";
            }
        }
        
        csv.close();
        std::cout << "\n  CSV文件已写入: " << csvPath << std::endl;
    }

    CHECK_CUDA_MAIN(cudaEventDestroy(start));
    CHECK_CUDA_MAIN(cudaEventDestroy(stop));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA异步错误: " << cudaGetErrorString(err) << std::endl;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "全部测试完成!" << std::endl;
    std::cout << "输出目录: " << outputDir << std::endl;
    std::cout << "生成CSV文件数: " << total_nk << std::endl;
    std::cout << "========================================" << std::endl;

    return EXIT_SUCCESS;
}
