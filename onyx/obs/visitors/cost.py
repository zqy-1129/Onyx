"""cost visitor：本地部署的"成本"是 GPU 时间，不是钱。

刻意不算电费：那需要功耗曲线与电价，属于猜测；GPU-秒是可直接测量、可比较、
可用于"这次评测花了多少卡时"的硬指标。要算钱的人可以基于它自己乘系数。
"""

from __future__ import annotations

from onyx.core.event import TraceEvent
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor


class CostVisitor(BaseVisitor):
    name = "cost"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        return None

    def finalize(self, state: TraceState) -> None:
        wall_ms = state.wall_ms
        usage = state.reconciled
        cost: dict[str, object] = {}
        if wall_ms:
            cost["gpu_seconds"] = round(wall_ms / 1000, 4)
        if usage:
            total = (usage.in_tokens or 0) + (usage.out_tokens or 0)
            cost["tokens_total"] = total
            cost["token_source"] = str(usage.source)
            cost["token_confidence"] = str(usage.confidence)
            if wall_ms and total:
                cost["tokens_per_second"] = round(total / (wall_ms / 1000), 2)
            if usage.thinking_tokens:
                cost["thinking_tokens"] = usage.thinking_tokens
                cost["note"] = "输出 token 含推理（P12：thinking 计入 eval_count）"
        if cost:
            state.extra["cost"] = cost
