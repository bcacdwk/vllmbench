#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接调试：在 vLLM 推理中精确测量 quant kernel 的 MK 和时间

通过 monkey-patch 方式记录每次 quant 调用的详细信息
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import torch

# 设置环境变量
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["SLIDESPARSE_PROFILE"] = "1"

# 添加项目路径
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# 全局记录
_quant_records = []
_MAX_RECORDS = 1000


def run_debug_inference():
    """运行推理并记录 quant kernel 调用详情"""
    sys.path.insert(0, str(_SCRIPT_DIR.parent))
    from utils import ModelFinder
    
    # 找到模型
    model_path = ModelFinder.find_small_model("FP8")
    if not model_path:
        print("模型未找到")
        return
    
    print(f"模型路径: {model_path}")
    print()
    
    # Monkey-patch quant_only_fp8_kernel 来记录调用
    from slidesparse.core import SlideSparseLinearMethod_FP8
    original_quant_only = SlideSparseLinearMethod_FP8.quant_only_fp8_kernel
    
    def patched_quant_only(input):
        M, K = input.shape
        
        # 只记录前 N 次调用
        if len(_quant_records) < _MAX_RECORDS:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_quant_only(input)
            end.record()
            _quant_records.append(("triton", M, K, start, end))
        else:
            result = original_quant_only(input)
        return result
    
    SlideSparseLinearMethod_FP8.quant_only_fp8_kernel = patched_quant_only
    
    # 同样 patch CUTLASS 的 quant
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    original_forward_cuda = QuantFP8.forward_cuda
    
    def patched_forward_cuda(self, x, scale=None, scale_ub=None):
        M = x.shape[0]
        K = x.shape[1] if x.dim() > 1 else 1
        
        if len(_quant_records) < _MAX_RECORDS:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_forward_cuda(self, x, scale, scale_ub)
            end.record()
            _quant_records.append(("cutlass", M, K, start, end))
        else:
            result = original_forward_cuda(self, x, scale, scale_ub)
        return result
    
    QuantFP8.forward_cuda = patched_forward_cuda
    
    # 运行推理
    from vllm import LLM, SamplingParams
    
    print("=" * 80)
    print("加载模型...")
    print("=" * 80)
    
    llm = LLM(
        model=str(model_path),
        max_model_len=256,
        gpu_memory_utilization=0.8,
        disable_log_stats=True,
        enforce_eager=True,
    )
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,
    )
    
    print()
    print("=" * 80)
    print("运行推理...")
    print("=" * 80)
    
    # 清空记录
    _quant_records.clear()
    
    # 运行
    outputs = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    
    # 分析记录
    print()
    print("=" * 80)
    print(f"收集到 {len(_quant_records)} 次 quant 调用")
    print("=" * 80)
    
    # 按类型和 MK 统计
    stats = defaultdict(lambda: {"count": 0, "times": []})
    
    for backend, M, K, start, end in _quant_records:
        key = (backend, M, K)
        stats[key]["count"] += 1
        stats[key]["times"].append(start.elapsed_time(end))
    
    print()
    print(f"{'Backend':12s} {'M':>8s} {'K':>8s} {'Count':>8s} {'Mean(ms)':>12s} {'Min(ms)':>12s} {'Max(ms)':>12s}")
    print("-" * 80)
    
    for (backend, M, K), data in sorted(stats.items()):
        times = data["times"]
        mean_t = sum(times) / len(times)
        min_t = min(times)
        max_t = max(times)
        print(f"{backend:12s} {M:8d} {K:8d} {data['count']:8d} {mean_t:12.4f} {min_t:12.4f} {max_t:12.4f}")
    
    print()
    
    # 清理
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # 测试 cuBLASLt 路径
    print("#" * 80)
    print("# 测试 cuBLASLt 路径")
    print("#" * 80)
    os.environ["USE_CUBLASLT"] = "1"
    os.environ["USE_CUSPARSELT"] = "0"
    os.environ["DISABLE_SLIDESPARSE"] = "0"
    
    run_debug_inference()
