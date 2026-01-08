# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt FP8 配置模块

通过继承 CompressedTensorsConfig，实现最小侵入式的 cuBLASLt 集成。
保持与 compressed-tensors 格式的完全兼容，仅替换底层 GEMM kernel。

设计思路:
=========
1. 继续使用 compressed-tensors 量化格式（模型不需要改变）
2. 通过环境变量 VLLM_USE_CUBLASLT=1 启用 cuBLASLt 后端
3. 量化逻辑（quant）完全复用原有代码
4. 仅替换 GEMM+Dequant 部分为 cuBLASLt 实现

当前阶段（Phase 3 初期）:
========================
- 先使用 vllm 官方的 cutlass_scaled_mm 作为 GEMM 后端
- 为后续的 cuBLASLt kernel 替换做好框架准备
- 验证整个替换流程的可行性
"""

import os
from typing import Any, Optional

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# 环境变量控制
# 支持两种环境变量名称，便于兼容
VLLM_USE_CUBLASLT = os.environ.get("VLLM_USE_CUBLASLT", "0").lower() in ("1", "true", "yes")
SLIDESPARSE_USE_CUBLASLT = os.environ.get("SLIDESPARSE_USE_CUBLASLT", "0").lower() in ("1", "true", "yes")

# 任意一个启用即可
_CUBLASLT_ENABLED = VLLM_USE_CUBLASLT or SLIDESPARSE_USE_CUBLASLT

# Debug 模式
SLIDESPARSE_CUBLASLT_DEBUG = os.environ.get("SLIDESPARSE_CUBLASLT_DEBUG", "0").lower() in ("1", "true", "yes")


def is_cublaslt_enabled() -> bool:
    """检查是否启用 cuBLASLt 后端
    
    支持两种环境变量:
    - VLLM_USE_CUBLASLT=1
    - SLIDESPARSE_USE_CUBLASLT=1
    """
    # 每次检查时重新读取环境变量，支持运行时修改
    vllm_env = os.environ.get("VLLM_USE_CUBLASLT", "0").lower() in ("1", "true", "yes")
    slidesparse_env = os.environ.get("SLIDESPARSE_USE_CUBLASLT", "0").lower() in ("1", "true", "yes")
    return vllm_env or slidesparse_env


def get_cublaslt_status() -> str:
    """获取 cuBLASLt 状态信息"""
    if is_cublaslt_enabled():
        return "cuBLASLt backend ENABLED (via VLLM_USE_CUBLASLT=1 or SLIDESPARSE_USE_CUBLASLT=1)"
    else:
        return "cuBLASLt backend DISABLED (set VLLM_USE_CUBLASLT=1 or SLIDESPARSE_USE_CUBLASLT=1 to enable)"
