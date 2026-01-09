#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
cuBLASLt FP8 GEMM Extension Setup Script

使用方法：
    cd /root/vllmbench/slidesparse/csrc
    python setup_cublaslt.py install
    
或者开发模式：
    python setup_cublaslt.py develop
"""

import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取 CUDA 相关路径
cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')

# 编译选项
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '-std=c++17',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        # 支持的 GPU 架构
        '-gencode=arch=compute_89,code=sm_89',   # Ada Lovelace (RTX 40xx)
        '-gencode=arch=compute_90,code=sm_90',   # Hopper (H100)
        '-gencode=arch=compute_120,code=sm_120', # Blackwell (RTX 50xx)
    ]
}

# 链接库
libraries = ['cublasLt', 'cublas', 'cuda']

setup(
    name='slidesparse_cublaslt',
    version='0.1.0',
    description='cuBLASLt FP8 GEMM Extension for SlideSparse',
    ext_modules=[
        CUDAExtension(
            name='slidesparse_cublaslt',
            sources=['cublaslt_fp8_gemm.cu'],
            include_dirs=[
                os.path.join(cuda_home, 'include'),
            ],
            library_dirs=[
                os.path.join(cuda_home, 'lib64'),
            ],
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
