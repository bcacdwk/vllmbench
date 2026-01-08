#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""
测试 3: vLLM 最简推理验证测试

验证启用 cuBLASLt 后端后，vLLM 能否正常进行推理。
使用本地已下载的 FP8 模型进行测试。

运行方式:
    # 禁用 cuBLASLt (默认，使用原生 cutlass)
    python3 slidesparse/test/test_cublaslt_inference.py
    
    # 启用 cuBLASLt
    VLLM_USE_CUBLASLT=1 python3 slidesparse/test/test_cublaslt_inference.py
    
    # 指定 GPU
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_CUBLASLT=1 python3 slidesparse/test/test_cublaslt_inference.py

环境要求:
    - 已下载 FP8 模型到 checkpoints/ 目录
    - 可以通过 slidesparse/tools/model_download.sh 下载
"""

import sys
import os
import time

# 添加项目根目录到 Python 路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def find_available_model():
    """查找可用的 FP8 模型"""
    checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    
    # 优先级列表：从小到大
    fp8_models = [
        "Qwen2.5-0.5B-FP8",
        "Qwen2.5-1.5B-FP8", 
        "Llama3.2-1B-FP8",
        "Qwen2.5-3B-FP8",
        "Llama3.2-3B-FP8",
        "Qwen2.5-7B-FP8",
    ]
    
    for model_name in fp8_models:
        model_path = os.path.join(checkpoint_dir, model_name)
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            return model_path, model_name
    
    return None, None


def test_inference():
    """测试 vLLM 推理"""
    print("=" * 60)
    print("测试 3: vLLM 最简推理验证测试")
    print("=" * 60)
    
    # 检查环境变量
    use_cublaslt = os.environ.get("VLLM_USE_CUBLASLT", "0").lower() in ("1", "true", "yes")
    print(f"\n环境变量 VLLM_USE_CUBLASLT: {use_cublaslt}")
    
    # 查找可用模型
    print("\n[3.1] 查找可用的 FP8 模型...")
    model_path, model_name = find_available_model()
    
    if model_path is None:
        print("    ✗ 未找到可用的 FP8 模型!")
        print("    请先运行 slidesparse/tools/model_download.sh 下载模型")
        print("    示例: ./slidesparse/tools/model_download.sh --fp8 --qwen")
        return False
    
    print(f"    ✓ 找到模型: {model_name}")
    print(f"    路径: {model_path}")
    
    # 导入 vLLM
    print("\n[3.2] 导入 vLLM...")
    try:
        from vllm import LLM, SamplingParams
        print("    ✓ vLLM 导入成功")
    except ImportError as e:
        print(f"    ✗ 导入失败: {e}")
        return False
    
    # 检查 cuBLASLt 状态
    print("\n[3.3] 检查 cuBLASLt 状态...")
    try:
        from vllm.model_executor.layers.quantization.cublaslt import (
            is_cublaslt_enabled,
            get_cublaslt_status,
        )
        print(f"    {get_cublaslt_status()}")
    except ImportError as e:
        print(f"    ⚠ cuBLASLt 模块导入失败: {e}")
    
    # 加载模型
    print(f"\n[3.4] 加载模型 {model_name}...")
    print("    (这可能需要几分钟...)")
    
    try:
        start_time = time.time()
        llm = LLM(
            model=model_path,
            dtype="auto",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            max_model_len=512,  # 限制长度以加快测试
        )
        load_time = time.time() - start_time
        print(f"    ✓ 模型加载成功 (耗时: {load_time:.2f}s)")
    except Exception as e:
        print(f"    ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 进行推理
    print("\n[3.5] 执行推理测试...")
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,  # 确定性输出
        max_tokens=32,
    )
    
    try:
        start_time = time.time()
        outputs = llm.generate(prompts, sampling_params)
        infer_time = time.time() - start_time
        
        print(f"    ✓ 推理成功 (耗时: {infer_time:.2f}s)")
        
        print("\n    --- 推理结果 ---")
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated = output.outputs[0].text
            print(f"    Prompt {i+1}: {prompt[:50]}...")
            print(f"    Output {i+1}: {generated[:100]}...")
            print()
            
    except Exception as e:
        print(f"    ✗ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 总结
    print("=" * 60)
    backend = "cuBLASLt" if use_cublaslt else "原生 cutlass"
    print(f"✓ 推理测试通过! (后端: {backend})")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_inference()
    sys.exit(0 if success else 1)
