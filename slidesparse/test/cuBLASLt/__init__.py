#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse cuBLASLt 测试包

测试套件:
- 01_bridge: 模块导入和集成测试
- 02_kernel: GEMM kernel 正确性测试
- 03_inference: 端到端推理测试
- 04_throughput: 吞吐量对比测试

使用方法:
    # 直接运行测试脚本
    python 01_bridge.py
    python 02_kernel.py --cublaslt
    
    # 或用 run_all.sh
    ./run_all.sh --cublaslt
"""

# 注意：这些测试文件设计为直接运行的脚本，不是作为模块导入的
# 所以这里的 __all__ 主要是文档作用，实际很少会 import 它们
