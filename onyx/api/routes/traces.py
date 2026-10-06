"""Trace 与 Token Ledger 的只读查询。"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from onyx.api.deps import AppState, get_state
from onyx.api.schemas import (
    AnomalyView,
    LatencyView,
    TokenPart,
    ToolCallView,
    TraceDetail,
    TracePage,
    TraceSummary,
    UsageAlt,
    UsageSummaryView,
)
from onyx.obs.anomalies import SPECS
from onyx.store.records import TraceRecord
from onyx.store.repos.usage_repo import UsageBundle

router = APIRouter(prefix="/api", tags=["traces"])


def _anomaly_view(item: Any) -> AnomalyView:
    spec = SPECS.get(item.code)
    return AnomalyView(
        id=item.id, code=item.code, severity=item.severity,
        meaning=spec.meaning if spec else "", action=spec.action if spec else "",
        detail=item.detail,
    )


def _latency(record: TraceRecord, bundle: UsageBundle) -> LatencyView:
    usage = bundle.usage
    engine = record.engine_latency or {}
    load_ns = engine.get("load")
    return LatencyView(
        ttft_ms=usage.ttft_ms if usage else None,
        wall_ms=usage.wall_ms if usage else None,
        prefill_mode=(usage.prefill_mode or "unknown") if usage else "unknown",
        prefill_ms_per_token=usage.prefill_ms_per_token if usage else None,
        prefill_tps=usage.prefill_tps if usage else None,
        decode_tps=usage.decode_tps if usage else None,
        load_ms=round(load_ns / 1e6, 2) if isinstance(load_ns, int | float) else None,
        cold_load=bool(isinstance(load_ns, int | float) and load_ns > 50_000_000),
    )


def _summary(record: TraceRecord, bundle: UsageBundle, tools: int, anomalies: int) -> TraceSummary:
    usage = bundle.usage
    return TraceSummary(
        id=record.id, purpose=record.purpose, kind=record.kind, model_name=record.model_name,
        provider_id=record.provider_id, started_at=record.started_at, status=record.status,
        finish_reason=record.finish_reason,
        in_tokens=usage.in_tokens if usage else None,
        out_tokens=usage.out_tokens if usage else None,
        source=usage.source if usage else None,
        confidence=usage.confidence if usage else None,
        ttft_ms=usage.ttft_ms if usage else None,
        decode_tps=usage.decode_tps if usage else None,
        prefill_mode=usage.prefill_mode if usage else None,
        tool_calls=tools, anomalies=anomalies,
        eval_run_id=record.eval_run_id, case_id=record.case_id,
    )


@router.get("/traces", response_model=TracePage)
def list_traces(
    limit: int = Query(50, ge=1, le=500),
    cursor: str | None = Query(None, description="上一页最后一条的 id"),
    purpose: str | None = None,
    model: str | None = Query(None, description="model_id 精确匹配"),
    status: str | None = None,
    since: str | None = None,
    state: AppState = Depends(get_state),
) -> TracePage:
    rows = state.traces.list(
        limit=limit, cursor=cursor, purpose=purpose, model_id=model, status=status, since=since
    )
    if not rows:
        return TracePage(items=[], next_cursor=None, total=state.traces.count(since=since))

    ids = tuple(r.id for r in rows)
    placeholders = ",".join("?" * len(ids))
    tool_counts = {
        str(r["trace_id"]): int(r["n"]) for r in state.runtime.db.query(
            f"SELECT trace_id, COUNT(*) AS n FROM tool_call WHERE trace_id IN ({placeholders}) "
            "GROUP BY trace_id", ids
        )
    }
    anomaly_counts = {
        str(r["trace_id"]): int(r["n"]) for r in state.runtime.db.query(
            f"SELECT trace_id, COUNT(*) AS n FROM anomaly WHERE trace_id IN ({placeholders}) "
            "GROUP BY trace_id", ids
        )
    }
    items = [
        _summary(r, state.usage.fetch(r.id), tool_counts.get(r.id, 0), anomaly_counts.get(r.id, 0))
        for r in rows
    ]
    return TracePage(
        items=items,
        next_cursor=rows[-1].id if len(rows) == limit else None,
        total=state.traces.count(since=since),
    )


@router.get("/traces/{trace_id}", response_model=TraceDetail)
def trace_detail(
    trace_id: str,
    include_blobs: bool = Query(True, description="是否内联 messages/output 原文"),
    state: AppState = Depends(get_state),
) -> TraceDetail:
    record = state.traces.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace 不存在: {trace_id}")
    bundle = state.usage.fetch(trace_id)
    calls = state.traces.list_tool_calls(trace_id)
    anomalies = [a for a in state.traces.list_anomalies(limit=500) if a.trace_id == trace_id]

    messages: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    if include_blobs:
        blobs = state.runtime.blobs
        if record.messages_ref and blobs.exists(record.messages_ref):
            messages = blobs.get_json(record.messages_ref)
        if record.output_ref and blobs.exists(record.output_ref):
            output = blobs.get_json(record.output_ref)

    usage = bundle.usage
    return TraceDetail(
        trace=_summary(record, bundle, len(calls), len(anomalies)),
        params=record.params, latency=_latency(record, bundle),
        usage=UsageAlt(
            source=usage.source, in_tokens=usage.in_tokens, out_tokens=usage.out_tokens,
            thinking_tokens=usage.thinking_tokens, cached_tokens=usage.cached_tokens,
            confidence=usage.confidence, ok=True,
            note=f"drift={usage.drift_pct}" if usage.drift_pct is not None else "",
        ) if usage else None,
        alts=[
            UsageAlt(
                source=a.source, in_tokens=a.in_tokens, out_tokens=a.out_tokens,
                thinking_tokens=a.thinking_tokens, cached_tokens=a.cached_tokens,
                ok=a.ok, confidence=a.confidence, note=a.note,
            )
            for a in bundle.alts
        ],
        parts=[TokenPart(part=p.part, ord=p.ord, tokens=p.tokens, bytes=p.bytes) for p in bundle.parts],
        tool_calls=[
            ToolCallView(
                id=c.id, step=c.step, name=c.name, parse_status=c.parse_status,
                parse_source=c.parse_source, args=c.args, args_raw=c.args_raw,
                result_status=c.result_status, latency_ms=c.latency_ms, executed_by=c.executed_by,
            )
            for c in calls
        ],
        anomalies=[_anomaly_view(a) for a in anomalies],
        gpu=record.gpu, engine_latency=record.engine_latency,
        refs={
            "messages": record.messages_ref or "", "output": record.output_ref or "",
            "raw_request": record.raw_request_ref or "",
            "raw_response": record.raw_response_ref or "",
            "rendered_prompt": record.rendered_prompt_ref or "",
        },
        messages=messages, output=output,
        attribution=(record.extra or {}).get("attribution", {}) if record.extra else {},
    )


@router.get("/usage/summary", response_model=UsageSummaryView)
def usage_summary(
    since: str | None = None,
    model: str | None = None,
    bucket_minutes: int = Query(60, ge=1, le=1440),
    state: AppState = Depends(get_state),
) -> UsageSummaryView:
    """Token Ledger 的汇总。**算法在 `report.usage_report.build_overview`**：
    CLI 的 `onyx report usage` 走同一个函数——两处各算 p50 与阈值，
    同一份库就会在看板与终端上给出两个"漂移超阈率"。"""
    from onyx.report.usage_report import build_overview

    try:
        overview = build_overview(state.usage, since=since, model=model,
                                  bucket_minutes=bucket_minutes)
    except ValueError as exc:
        # `?since=7d` 这类写法字典序比所有行都大：不拦就等于告诉前端"这段时间没有请求"。
        raise HTTPException(400, str(exc)) from exc
    return UsageSummaryView(**overview.as_dict())
