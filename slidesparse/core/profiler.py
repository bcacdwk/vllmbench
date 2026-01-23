# SPDX-License-Identifier: Apache-2.0
"""
SlideSparse 计时诊断模块

通过环境变量 SLIDESPARSE_PROFILE=1 启用。

使用方式:
=========
    with ProfileTimer("operation_name"):
        # 需要计时的代码
        ...

    # 每次 apply 后调用
    profile_step()

注意事项:
=========
- 启用 profiling 会破坏 CUDA Graph，必须使用 eager 模式
- 异步计时机制，批量同步减少 GPU pipeline 阻塞
- torch.compile 追踪时自动禁用（返回 nullcontext）
"""

import os
from collections import defaultdict
from contextlib import nullcontext

import torch


# ============================================================================
# 配置常量
# ============================================================================

_PROFILE_ENABLED = os.environ.get("SLIDESPARSE_PROFILE", "0") == "1"
_PROFILE_PRINT_INTERVAL = 1000  # 每 N 次调用打印一次统计
_PENDING_FLUSH_THRESHOLD = 1000  # 积累多少个 events 后批量同步


# ============================================================================
# 全局状态
# ============================================================================

_profile_data = defaultdict(lambda: {"count": 0, "total_ms": 0.0})
_profile_call_count = 0

# 异步计时：存储 pending 的 CUDA events，延迟同步
_pending_events: list = []  # [(name, start_event, end_event), ...]


# ============================================================================
# ProfileTimer 实现
# ============================================================================

class _ProfileTimerImpl:
    """
    Async CUDA timer using events (no GPU pipeline blocking).
    Events are batched and synchronized lazily for accurate timing.
    """
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled and _PROFILE_ENABLED
        self.start_event = None
        self.end_event = None
        
    def __enter__(self):
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self
    
    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            _pending_events.append((self.name, self.start_event, self.end_event))
            if len(_pending_events) >= _PENDING_FLUSH_THRESHOLD:
                _flush_pending_events()


def ProfileTimer(name: str, enabled: bool = True):
    """ProfileTimer factory - returns nullcontext during torch.compile tracing."""
    if torch.compiler.is_compiling():
        return nullcontext()
    return _ProfileTimerImpl(name, enabled)


# ============================================================================
# 事件处理函数
# ============================================================================

def _flush_pending_events():
    """Synchronize and collect all pending events."""
    global _pending_events
    if not _pending_events:
        return
    
    torch.cuda.synchronize()
    
    for name, start_event, end_event in _pending_events:
        elapsed_ms = start_event.elapsed_time(end_event)
        _profile_data[name]["total_ms"] += elapsed_ms
        _profile_data[name]["count"] += 1
    
    _pending_events = []


def profile_step():
    """Check if stats should be printed (called after each apply)."""
    global _profile_call_count
    if not _PROFILE_ENABLED:
        return
    
    _profile_call_count += 1
    if _profile_call_count % _PROFILE_PRINT_INTERVAL == 0:
        print_profile_stats()


# ============================================================================
# 统计输出函数
# ============================================================================

def print_profile_stats():
    """Print profiling statistics."""
    _flush_pending_events()
    
    if not _profile_data:
        return
    
    print(f"\n{'=' * 80}")
    print(f"SlideSparse Profile Stats (after {_profile_call_count} calls)")
    print(f"{'=' * 80}")
    print(f"{'Operation':<40} {'Count':>10} {'Total(ms)':>12} {'Avg(us)':>10}")
    print(f"{'-' * 80}")
    
    # 按 total_ms 降序排列
    sorted_items = sorted(_profile_data.items(), key=lambda x: -x[1]["total_ms"])
    
    for name, data in sorted_items:
        count = data["count"]
        total_ms = data["total_ms"]
        avg_ms = total_ms / count if count > 0 else 0
        print(f"{name:<40} {count:>10} {total_ms:>12.3f} {avg_ms * 1000:>10.1f}")
    
    print(f"{'=' * 80}\n")


def reset_profile_stats():
    """重置计时统计（会丢弃所有未 flush 的 pending events）"""
    global _profile_call_count, _pending_events
    _profile_data.clear()
    _profile_call_count = 0
    _pending_events = []


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    "ProfileTimer",
    "profile_step",
    "print_profile_stats",
    "reset_profile_stats",
    "_flush_pending_events",
]
