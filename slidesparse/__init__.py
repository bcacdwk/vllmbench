# SPDX-License-Identifier: Apache-2.0
# SlideSparse: Sparse Acceleration for LLM Inference
"""
SlideSparse 外挂模块 - vLLM 集成

此模块实现 SlideSparse 稀疏加速方法与 vLLM 的集成，包括:
- cuBLASLt Dense 基线 (Phase 3)
- cuSPARSELt Sparse 加速 (Phase 6)

目录结构:
- core/: 核心逻辑 (LinearMethod, Config)
- kernels/: Kernel 实现 (Triton, cuBLASLt, cuSPARSELt)
- offline/: 离线工具链 (prune, slide, compress)
- test/: 测试脚本
"""

__version__ = "0.1.0"
