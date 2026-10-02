"""gpu visitor：载入状态与显存采样。

`size_vram < size` 意味着部分权重在 CPU 上，吞吐会低一个数量级。
这种请求的结果**不能**与全量载入的结果混算，所以必须显式标记。
"""

from __future__ import annotations

from onyx.core.event import EventType, TraceEvent
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor


class GpuVisitor(BaseVisitor):
    name = "gpu"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        if event.type is not EventType.GPU_SAMPLE:
            return
        payload = event.payload
        state.gpu = {
            **state.gpu,
            **{k: v for k, v in payload.items() if v is not None},
        }
        size = payload.get("size")
        vram = payload.get("size_vram")
        if isinstance(size, int) and isinstance(vram, int) and size > 0 and vram < size:
            state.add_anomaly("OFFLOADED_TO_CPU", {
                "size": size, "size_vram": vram,
                "vram_share": round(vram / size, 4),
                "hint": "吞吐不可与全量载入的结果混算",
            })

    def finalize(self, state: TraceState) -> None:
        ctx = state.gpu.get("context_length")
        if not isinstance(ctx, int) or ctx <= 0 or state.reconciled is None:
            return
        in_tokens = state.reconciled.in_tokens or 0
        ratio = in_tokens / ctx
        state.extra["ctx_util"] = round(ratio, 4)
        if ratio > 1.0:
            state.add_anomaly("CONTEXT_OVERFLOW", {"in_tokens": in_tokens, "context_length": ctx})
        elif ratio >= 0.9:
            state.add_anomaly("CONTEXT_NEAR_LIMIT", {
                "in_tokens": in_tokens, "context_length": ctx, "ctx_util": round(ratio, 4),
            })
