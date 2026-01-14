#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse CSRC 工具库

本模块提供 Triton Autotune 配置等 CSRC 专用功能。

编译相关功能请使用顶层 slidesparse.utils 模块：
    from slidesparse.utils import (
        build_cuda_extension,
        build_cuda_extension_direct,
        get_nvcc_arch_flags,
        CUBLASLT_LDFLAGS,
        CUSPARSELT_LDFLAGS,
    )

主要功能
========
1. Triton Autotune 配置（dequant, quant kernels）

向后兼容
========
为保持向后兼容，以下符号从顶层 utils 重新导出：
- get_nvcc_arch_flags, get_current_arch_flag
- SUPPORTED_ARCHITECTURES
- DEFAULT_CFLAGS, DEFAULT_CUDA_CFLAGS
- should_rebuild, clean_build_artifacts, build_cuda_extension
- CUBLASLT_LDFLAGS, CUSPARSELT_LDFLAGS, get_gemm_ldflags

使用示例
========
>>> from slidesparse.csrc.utils import get_dequant_autotune_configs
>>> configs = get_dequant_autotune_configs()
"""

# =============================================================================
# 向后兼容：从顶层 utils 重新导出
# =============================================================================

from slidesparse.utils import (
    # NVCC 架构标志
    SUPPORTED_ARCHITECTURES,
    get_nvcc_arch_flags,
    get_current_arch_flag,
    # 编译选项
    DEFAULT_CFLAGS,
    DEFAULT_CUDA_CFLAGS,
    # 编译工具
    should_rebuild,
    clean_build_artifacts,
    build_cuda_extension,
    # GEMM 链接库
    CUBLASLT_LDFLAGS,
    CUSPARSELT_LDFLAGS,
    get_backend_ldflags as get_gemm_ldflags,  # 别名兼容
)


# =============================================================================
# Triton Autotune 配置
# =============================================================================

def get_dequant_autotune_configs():
    """
    获取 dequant+bias kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100), SM100(B200), SM120(5080)
    
    因为 M 是灵活可变的batchsize 此处相当于是搜索 R[M,N] = A[M,K] * W[N,K] 中的 [M,N]
    
    Returns:
        triton.Config 对象列表
    """
    import triton
    
    return [
        # =====================================================================
        # Tier 1: Proven Winners (A100 validated)
        # =====================================================================
        # Small M King (M=1~128): 32x32
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        # Medium M King (M=256~8192): 64x32
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=4),
        # Large M King (M=12288+): 128x64
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=4),

        # =====================================================================
        # Tier 2: Basic kernel heuristics
        # =====================================================================
        # Small M, N<=4096: (32, 64, 4)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # Small M, N>4096: (32, 128, 4)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        # Medium M, N<=4096: (64, 64, 4)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        # Medium M, N>4096: (64, 128, 8)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=4),
        # Large M, N>4096: (128, 128, 8)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8, num_stages=4),

        # =====================================================================
        # Tier 3: Read/Write bias exploration
        # =====================================================================
        # Write Heavy (tall blocks): 128x32
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=4),
        # Read Heavy (wide blocks): 64x128 with lower warps
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4, num_stages=4),
        # Balanced high warp: 64x64 w=8
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        # Low warp large block
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4, num_stages=4),

        # =====================================================================
        # Tier 4: H100/Blackwell exploration (SM90/100/120)
        # =====================================================================
        # Super Wide: 256x64
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        # Super Tall: 64x256
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        # Super Square: 128x128 high warp
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=16, num_stages=4),
        # Wide variants for large N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256}, num_warps=8, num_stages=4),
        # Extreme Wide: 256x32
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32}, num_warps=8, num_stages=4),

        # =====================================================================
        # Tier 5: Small M + Large N special cases
        # =====================================================================
        # 32x128 with various warps
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        # 32x64 with high warps
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=4),
        # 32x64 with low warps
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=2, num_stages=3),

        # =====================================================================
        # Tier 6: Tiny M = 16 special cases
        # =====================================================================
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=2, num_stages=3),
        # Very wide for large N
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=4, num_stages=3),
    ]


def get_quant_autotune_configs():
    """
    获取 quant (per-row quantization) kernel 的 Triton autotune 配置
    
    覆盖: SM80(A100), SM89(4090), SM90(H100), SM100(B200), SM120(5080)
    
    因为 M 是灵活可变的batchsize 此处相当于是搜索 R[M,N] = A[M,K] * W[N,K] 中的 [M,K]

    - BLOCK_K 是主要的块大小参数，控制每次循环处理的元素数
    - M 虽然不在 kernel 块参数中，但影响最佳 num_warps/num_stages
    - autotune key = ['M', 'K']
    
    BLOCK_K 选择原则：
    - 必须是 2 的幂次
    - 典型值：512, 1024, 2048, 4096, 8192
    - K=2560 → BLOCK_K=2048/4096 (1-2 次循环)
    - K=6912 → BLOCK_K=4096/8192 (1-2 次循环)
    
    num_warps 选择原则：
    - 小 M（1-64）：1-4 warps
    - 中 M（64-4096）：4-8 warps
    - 大 M（4096+）：8-32 warps
    
    Returns:
        triton.Config 对象列表
    """
    import triton
    
    return [
        # =====================================================================
        # Tier 1: 小 BLOCK_K (适合小 K 或需要低寄存器压力)
        # =====================================================================
        triton.Config({'BLOCK_K': 512}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 512}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 512}, num_warps=4, num_stages=3),
        
        # =====================================================================
        # Tier 2: 中等 BLOCK_K (K <= 2048 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 1024}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 1024}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 1024}, num_warps=8, num_stages=3),
        
        # =====================================================================
        # Tier 3: 大 BLOCK_K (K=2560-4096 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_K': 2048}, num_warps=16, num_stages=3),
        
        # =====================================================================
        # Tier 4: 超大 BLOCK_K (K=4096-8192 的默认选择)
        # =====================================================================
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=32, num_stages=3),
        
        # =====================================================================
        # Tier 5: 极大 BLOCK_K (K > 6000 的选择，如 K=6912)
        # =====================================================================
        triton.Config({'BLOCK_K': 8192}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=3),
        
        # =====================================================================
        # Tier 6: 边界探索 - 小 M 特殊优化
        # =====================================================================
        # 小 M (1-16) 需要低 num_warps
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=3),
        triton.Config({'BLOCK_K': 2048}, num_warps=1, num_stages=4),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=3),
        triton.Config({'BLOCK_K': 4096}, num_warps=1, num_stages=4),
        
        # =====================================================================
        # Tier 7: 边界探索 - 大 M 特殊优化  
        # =====================================================================
        # 大 M (16384+) 需要高 num_warps
        triton.Config({'BLOCK_K': 2048}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_K': 4096}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_K': 8192}, num_warps=32, num_stages=4),
    ]


def get_quant_or_dequant_autotune_configs():
    """
    获取 quant/dequant 通用的 Triton autotune 配置
    
    这是旧版 2D tiled kernel 的配置（BLOCK_M x BLOCK_N/BLOCK_K）
    对于新版 per-row kernel，请使用 get_quant_autotune_configs()
    
    Returns:
        triton.Config 对象列表（等同于 get_dequant_autotune_configs）
    """
    return get_dequant_autotune_configs()



# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # =========================================================================
    # 向后兼容（从顶层 utils 重新导出）
    # =========================================================================
    # NVCC 架构标志
    'SUPPORTED_ARCHITECTURES',
    'get_nvcc_arch_flags',
    'get_current_arch_flag',
    # 编译选项
    'DEFAULT_CFLAGS',
    'DEFAULT_CUDA_CFLAGS',
    # 编译工具
    'should_rebuild',
    'clean_build_artifacts',
    'build_cuda_extension',
    # GEMM 链接库
    'CUBLASLT_LDFLAGS',
    'CUSPARSELT_LDFLAGS',
    'get_gemm_ldflags',
    
    # =========================================================================
    # 本模块特有功能
    # =========================================================================
    # Triton Autotune 配置
    'get_dequant_autotune_configs',
    'get_quant_autotune_configs',
    'get_quant_or_dequant_autotune_configs',
]
