"""token visitor：收集各来源计数、做归因、对账并产出采信结果。

这是"测量必须带出处"原则的落地点：采信谁、置信度多高、偏差多大，
全部由 `measurement.reconciler` 决定，visitor 只负责把结果写进状态与异常。
"""

from __future__ import annotations

from typing import Any

from onyx.core.event import EventType, TraceEvent
from onyx.core.types import Confidence, TokenPart, TokenSample, TokenSource
from onyx.llm.measurement.reconciler import reconcile
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor

_USAGE_EVENTS = frozenset({EventType.USAGE_ENGINE, EventType.USAGE_COMPAT, EventType.USAGE_LOCAL})


def _confidence(value: Any) -> Confidence:
    try:
        return Confidence(str(value)) if value else Confidence.HIGH
    except ValueError:
        return Confidence.LOW


def sample_from_payload(event: TraceEvent) -> TokenSample | None:
    """把一个 usage 事件翻成 TokenSample。

    **source 非法时返回 None 而不是回退到某个默认档**：早先这里回退成 heuristic，
    结果一个归因事件把真正的启发式样本覆盖掉，采信直接掉到"无来源"。
    宁可丢弃一条可疑样本，也不要污染另一个档位的数字。
    """
    payload = event.payload
    raw_source = str(payload.get("source") or _default_source(event.type))
    try:
        source = TokenSource(raw_source)
    except ValueError:
        return None
    return TokenSample(
        source=source,
        in_tokens=_int(payload.get("in_tokens")),
        out_tokens=_int(payload.get("out_tokens")),
        thinking_tokens=_int(payload.get("thinking_tokens")),
        cached_tokens=_int(payload.get("cached_tokens")),
        ok=bool(payload.get("ok", True)) and _int(payload.get("in_tokens")) is not None,
        confidence=_confidence(payload.get("confidence")),
        note=str(payload.get("note") or ""),
    )


def _default_source(event_type: EventType) -> TokenSource:
    return {
        EventType.USAGE_ENGINE: TokenSource.ENGINE,
        EventType.USAGE_COMPAT: TokenSource.COMPAT,
    }.get(event_type, TokenSource.HEURISTIC)


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


class TokenVisitor(BaseVisitor):
    name = "token"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        if event.type is EventType.RECONCILED:
            self._apply_reconciled(event, state)
            return
        if event.type is EventType.USAGE_ATTRIBUTION:
            self._apply_attribution(event, state)
            return
        if event.type not in _USAGE_EVENTS:
            return
        sample = sample_from_payload(event)
        if sample is None:
            return
        state.add_usage(sample)
        if event.type is EventType.USAGE_ENGINE:
            latency = event.payload.get("latency_ns") or {}
            if latency:
                state.engine_latency = dict(latency)

    def _apply_attribution(self, event: TraceEvent, state: TraceState) -> None:
        """分段归因是独立事件，不是某个档位的计数——两者出处不同，不许混。"""
        payload = event.payload
        state.parts = [
            TokenPart(
                part=str(raw.get("part", "")), ord=int(raw.get("ord", 0)),
                tokens=int(raw.get("tokens", 0)), bytes=_int(raw.get("bytes")),
            )
            for raw in payload.get("parts") or []
        ]
        attribution = dict(payload.get("attribution") or {})
        attribution["count_source"] = payload.get("count_source")
        state.extra["attribution"] = attribution
        if attribution.get("clamped"):
            state.extra["attribution_clamped"] = True
            state.extra["attribution_residual"] = attribution.get("residual_raw")

    def _apply_reconciled(self, event: TraceEvent, state: TraceState) -> None:
        payload = event.payload
        try:
            source = TokenSource(str(payload.get("chosen_source")))
        except (TypeError, ValueError):
            return
        state.reconciled = state.reconciled or _reconciled_from_payload(payload, source)

    def finalize(self, state: TraceState) -> None:
        if state.reconciled is not None:
            return
        result = reconcile(tuple(state.usage_samples.values()), parts=tuple(state.parts))
        state.reconciled = result.usage
        for code, detail in result.anomalies:
            state.add_anomaly(code, detail)
        for note in result.notes:
            if "compat" in note:
                state.add_anomaly("COMPAT_DIVERGENCE", {"note": note}, severity="info")
        if state.reconciled.source is TokenSource.ENGINE and state.reconciled.in_tokens is None:
            state.add_anomaly("NO_ENGINE_COUNT", {"reason": "引擎计数缺失"})


def _reconciled_from_payload(payload: dict[str, Any], source: TokenSource):
    from onyx.core.types import ReconciledUsage

    return ReconciledUsage(
        source=source,
        confidence=_confidence(payload.get("confidence")),
        in_tokens=_int(payload.get("in_tokens")),
        out_tokens=_int(payload.get("out_tokens")),
        drift_pct=payload.get("drift_pct"),
    )
