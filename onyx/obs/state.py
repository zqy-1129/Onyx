"""TraceState：一条 trace 的事件累加器。

visitor 往这里写，engine 在 TRACE_END 时把它翻成 store 记录。
这样 visitor 之间互不依赖，也不各自持有半份状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from onyx.core.ids import new_trace_id
from onyx.core.types import ReconciledUsage, TokenPart, TokenSample, TokenSource
from onyx.llm.measurement.reconciler import latency_summary
from onyx.obs.anomalies import severity_of
from onyx.store.records import (
    AnomalyRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)


@dataclass(slots=True)
class ToolCallDraft:
    step: int
    name: str = ""
    call_id: str = ""
    args: dict[str, Any] | None = None
    args_raw: str = ""
    parse_status: str = "ok"
    parse_source: str = ""
    result_status: str | None = None
    result_ref: str | None = None
    result_bytes: int | None = None
    started_at: str | None = None
    latency_ms: float | None = None
    tool_id: str | None = None
    tool_def_hash: str | None = None
    executed_by: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TraceState:
    trace_id: str
    started_at: str = ""
    kind: str = "generation"
    purpose: str = "chat"
    provider_id: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    parent_id: str | None = None
    eval_run_id: str | None = None
    case_id: str | None = None
    sample_seq: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    messages_ref: str | None = None
    tools_ref: str | None = None
    rendered_prompt_ref: str | None = None
    raw_request_ref: str | None = None
    raw_response_ref: str | None = None
    output_ref: str | None = None
    first_token_at: str | None = None
    ttft_ms: float | None = None
    finished_at: str | None = None
    wall_ms: float | None = None
    status: str = "ok"
    error: str = ""
    finish_reason: str | None = None
    engine_latency: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    keep_alive: str | None = None
    contract_version: int = 1
    text_chars: int = 0
    thinking_chars: int = 0
    usage_samples: dict[str, TokenSample] = field(default_factory=dict)
    reconciled: ReconciledUsage | None = None
    parts: list[TokenPart] = field(default_factory=list)
    tool_calls: list[ToolCallDraft] = field(default_factory=list)
    anomalies: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    #: 观测组件自身出错时计数，用于 OBSERVER_ERROR
    observer_errors: int = 0

    # ── 写入辅助 ──────────────────────────────────────────────────
    def add_anomaly(self, code: str, detail: dict[str, Any] | None = None, *, severity: str = "") -> None:
        self.anomalies.append((code, severity or severity_of(code), detail or {}))

    def add_usage(self, sample: TokenSample) -> None:
        self.usage_samples[str(sample.source)] = sample

    def latency_summary(self, *, warm_threshold_ms_per_token: float = 0.30) -> dict[str, Any]:
        """复用计量层的冷/热判定，保证看板与库里的口径完全一致。"""
        from onyx.core.types import EngineLatency, Generation

        latency = EngineLatency(
            total_ns=_ns(self.engine_latency.get("total")),
            load_ns=_ns(self.engine_latency.get("load")),
            prompt_eval_ns=_ns(self.engine_latency.get("prompt_eval")),
            eval_ns=_ns(self.engine_latency.get("eval")),
        )
        engine = self.usage_samples.get(str(TokenSource.ENGINE))
        gen = Generation(
            usage=tuple(self.usage_samples.values()), latency=latency,
            ttft_ms=self.ttft_ms, wall_ms=self.wall_ms,
        )
        summary = latency_summary(gen, warm_threshold_ms_per_token=warm_threshold_ms_per_token)
        if engine is not None and engine.in_tokens and summary.get("prefill_ms_per_token"):
            summary["in_tokens"] = engine.in_tokens
        return summary

    # ── 翻成 store 记录 ───────────────────────────────────────────
    def to_trace_record(self) -> TraceRecord:
        return TraceRecord(
            id=self.trace_id, kind=self.kind, purpose=self.purpose, started_at=self.started_at,
            status=self.status, parent_id=self.parent_id, root_id=self.parent_id or self.trace_id,
            eval_run_id=self.eval_run_id, case_id=self.case_id, sample_seq=self.sample_seq,
            provider_id=self.provider_id, model_id=self.model_id, model_name=self.model_name,
            first_token_at=self.first_token_at, finished_at=self.finished_at,
            error=self.error or None, params=self.params, messages_ref=self.messages_ref,
            tools_ref=self.tools_ref, rendered_prompt_ref=self.rendered_prompt_ref,
            output_ref=self.output_ref, raw_request_ref=self.raw_request_ref,
            raw_response_ref=self.raw_response_ref, finish_reason=self.finish_reason,
            engine_latency=self.engine_latency, gpu=self.gpu, keep_alive=self.keep_alive,
            contract_version=self.contract_version,
            extra={**self.extra, "text_chars": self.text_chars, "thinking_chars": self.thinking_chars,
                   "observer_errors": self.observer_errors},
        )

    def to_usage_record(self) -> UsageRecord | None:
        chosen = self.reconciled
        if chosen is None:
            return None
        summary = self.latency_summary()
        return UsageRecord(
            trace_id=self.trace_id, source=chosen.source, confidence=chosen.confidence,
            in_tokens=chosen.in_tokens, out_tokens=chosen.out_tokens,
            thinking_tokens=chosen.thinking_tokens, cached_tokens=chosen.cached_tokens,
            ttft_ms=self.ttft_ms, prefill_tps=_round(summary.get("prefill_tps")),
            decode_tps=_round(summary.get("decode_tps")), wall_ms=self.wall_ms,
            bytes_out=self.text_chars + self.thinking_chars,
            drift_pct=chosen.drift_pct,
            prefill_mode=summary.get("prefill_mode"),
            prefill_ms_per_token=_round(summary.get("prefill_ms_per_token"), 6),
            extra={"alts": [str(s.source) for s in chosen.alts]},
        )

    def to_alt_records(self) -> list[UsageAltRecord]:
        return [
            UsageAltRecord(
                trace_id=self.trace_id, source=str(sample.source), in_tokens=sample.in_tokens,
                out_tokens=sample.out_tokens, thinking_tokens=sample.thinking_tokens,
                cached_tokens=sample.cached_tokens, ok=sample.ok,
                confidence=str(sample.confidence) if sample.confidence else None, note=sample.note,
            )
            for sample in self.usage_samples.values()
        ]

    def to_part_records(self) -> list[TokenPartRecord]:
        return [
            TokenPartRecord(trace_id=self.trace_id, part=p.part, ord=p.ord, tokens=p.tokens, bytes=p.bytes)
            for p in self.parts
        ]

    def to_tool_call_records(self) -> list[ToolCallRecord]:
        return [
            ToolCallRecord(
                id=new_trace_id(), trace_id=self.trace_id, step=draft.step, parse_status=draft.parse_status,
                name=draft.name or None, call_id=draft.call_id or None, args=draft.args,
                args_raw=draft.args_raw or None, parse_source=draft.parse_source or None,
                result_status=draft.result_status, result_ref=draft.result_ref,
                result_bytes=draft.result_bytes, started_at=draft.started_at,
                latency_ms=draft.latency_ms, tool_id=draft.tool_id,
                tool_def_hash=draft.tool_def_hash, executed_by=draft.executed_by, extra=draft.extra,
            )
            for draft in self.tool_calls
        ]

    def to_anomaly_records(self) -> list[AnomalyRecord]:
        from onyx.core.clock import utc_now_iso

        now = self.finished_at or utc_now_iso()
        return [
            AnomalyRecord(id=new_trace_id(), code=code, severity=severity, trace_id=self.trace_id,
                          detail=detail, created_at=now)
            for code, severity, detail in self.anomalies
        ]


def _ns(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and value else None


def _round(value: Any, ndigits: int = 4) -> float | None:
    return round(float(value), ndigits) if isinstance(value, int | float) else None
