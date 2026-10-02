"""anomaly visitor：跨切面规则。

必须排在最后（见 `default_visitors()` 的顺序契约）：它依赖 token visitor 的对账结果
与 gpu visitor 的上下文占用，才能判断"这个数字可不可信"。
"""

from __future__ import annotations

from onyx.core.event import EventType, TraceEvent
from onyx.obs.anomalies import Severity, severity_of
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor

COLD_LOAD_NS = 50_000_000  # 50ms


class AnomalyVisitor(BaseVisitor):
    name = "anomaly"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        payload = event.payload
        if event.type is EventType.ANOMALY:
            code = str(payload.get("code", ""))
            if code:
                state.add_anomaly(
                    code, payload.get("detail") or {},
                    severity=str(payload.get("severity") or severity_of(code)),
                )
        elif event.type is EventType.MODEL_LOAD:
            load_ns = payload.get("load_duration_ns")
            if payload.get("cold") or (isinstance(load_ns, int) and load_ns > COLD_LOAD_NS):
                state.add_anomaly("COLD_LOAD", {
                    "load_ms": round(load_ns / 1e6, 1) if isinstance(load_ns, int) else None,
                    "hint": "TTFT 与吞吐应归入 cold 分组",
                })

    def finalize(self, state: TraceState) -> None:
        if state.status != "ok":
            state.add_anomaly("PROVIDER_ERROR", {
                "status": state.status, "error": state.error[:300],
            }, severity=Severity.ERROR)
            return  # 请求本身失败了，输出形态类判定没有意义

        if state.text_chars == 0 and state.thinking_chars > 0:
            state.add_anomaly("EMPTY_CONTENT_WITH_THINKING", {
                "thinking_chars": state.thinking_chars,
                "hint": "P5/P12：预算被推理吃光。评测中这不算答错，是没预算答",
            })
        elif state.text_chars == 0 and state.thinking_chars == 0:
            state.add_anomaly("EMPTY_OUTPUT", {"finish_reason": state.finish_reason})

        summary = state.latency_summary()
        if summary.get("prefill_mode") == "warm":
            state.add_anomaly("PREFILL_CACHE_HIT", {
                "prefill_ms_per_token": summary.get("prefill_ms_per_token"),
                "prefill_tps": summary.get("prefill_tps"),
                "hint": "等效吞吐，不可与 cold 合并聚合（P11：差 4.65×）",
            }, severity=Severity.INFO)

        if state.extra.get("attribution_clamped"):
            state.add_anomaly("ATTRIBUTION_CLAMPED", {
                "residual_raw": state.extra.get("attribution_residual"),
                "hint": "分段和超过引擎计数：count_fn 高估或引擎截断",
            })
