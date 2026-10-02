"""Playground、SSE 实时流、以及唯一的写操作（admin）。"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from onyx.api.deps import AppState, get_state
from onyx.api.schemas import (
    AnomalyView,
    ChatRequest,
    ChatResponse,
    LatencyView,
    TokenPart,
    ToolCallView,
    UsageAlt,
)
from onyx.core.types import (
    GenerationRequest,
    GenParams,
    Message,
    Role,
    ToolSpec,
    TraceContext,
    TracePurpose,
)
from onyx.llm.measurement.reconciler import latency_summary
from onyx.obs.anomalies import SPECS

router = APIRouter(prefix="/api", tags=["playground"])

#: Playground 演示工具。真实工具注册表在 S10 接入后从这里迁走。
DEMO_TOOLS: dict[str, ToolSpec] = {
    "weather": ToolSpec(
        name="get_weather", description="查询指定城市当前天气",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    ),
}

#: 等待 GPU 锁的上限。本地单卡是独占资源（DESIGN §8.5）：
#: 与其让两个请求互相踩出无法解释的延迟数字，不如明确排队或拒绝。
GPU_LOCK_TIMEOUT = 120.0


@router.post("/playground/chat", response_model=ChatResponse)
def chat(body: ChatRequest, state: AppState = Depends(get_state)) -> ChatResponse:
    messages: list[Message] = []
    if body.system:
        messages.append(Message(role=Role.SYSTEM, content=body.system))
    messages.append(Message(role=Role.USER, content=body.prompt))
    tools = tuple(DEMO_TOOLS[name] for name in body.tools if name in DEMO_TOOLS)
    unknown = [name for name in body.tools if name not in DEMO_TOOLS]
    if unknown:
        raise HTTPException(status_code=422, detail=f"未知演示工具: {unknown}，可用: {sorted(DEMO_TOOLS)}")

    req = GenerationRequest(
        model=body.model, messages=tuple(messages), tools=tools, stream=body.stream,
        thinking=body.thinking, keep_alive="5m",
        params=GenParams(max_tokens=body.max_tokens, temperature=body.temperature),
        context=TraceContext(
            purpose=TracePurpose.PLAYGROUND,
            extra={"client_key": body.client_key} if body.client_key else {},
        ),
    )
    acquired = state.gpu_lock.acquire(timeout=GPU_LOCK_TIMEOUT)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail=f"GPU 被其他请求占用超过 {GPU_LOCK_TIMEOUT:.0f}s，请稍后重试",
        )
    started = time.monotonic()
    try:
        result = state.runtime.gateway.generate(req, purpose="playground")
    finally:
        state.gpu_lock.release()
    state.runtime.flush()

    usage = result.usage
    latency = latency_summary(result.generation)
    return ChatResponse(
        trace_id=result.trace_id,
        text=result.generation.text,
        thinking=result.generation.thinking,
        finish_reason=str(result.generation.finish_reason),
        tool_calls=[
            ToolCallView(
                id=f"{result.trace_id}-{c.index}", step=c.index + 1, name=c.name,
                parse_status=str(c.parse_status), parse_source=c.parse_source,
                args=c.arguments, args_raw=c.arguments_raw,
            )
            for c in result.generation.tool_calls
        ],
        usage=UsageAlt(
            source=str(usage.source), confidence=str(usage.confidence),
            in_tokens=usage.in_tokens, out_tokens=usage.out_tokens,
            thinking_tokens=usage.thinking_tokens, cached_tokens=usage.cached_tokens, ok=True,
            note=f"drift={usage.drift_pct}" if usage.drift_pct is not None else "",
        ) if usage else None,
        latency=LatencyView(
            ttft_ms=result.generation.ttft_ms,
            wall_ms=round((time.monotonic() - started) * 1000, 2),
            prefill_mode=str(latency.get("prefill_mode") or "unknown"),
            prefill_ms_per_token=latency.get("prefill_ms_per_token"),
            prefill_tps=latency.get("prefill_tps"), decode_tps=latency.get("decode_tps"),
            load_ms=latency.get("load_ms"), cold_load=bool(latency.get("cold_load")),
        ),
        anomalies=[
            AnomalyView(
                id=f"{result.trace_id}-{index}", code=code, severity=severity,
                meaning=SPECS[code].meaning if code in SPECS else "",
                action=SPECS[code].action if code in SPECS else "",
                detail=detail,
            )
            for index, (code, severity, detail) in enumerate(result.anomalies)
        ],
        parts=[TokenPart(part=p.part, ord=p.ord, tokens=p.tokens, bytes=p.bytes)
                for p in (usage.parts if usage else ())],
    )


@router.get("/stream")
async def stream(state: AppState = Depends(get_state)) -> StreamingResponse:
    """实时事件流。前端断线重连即可，事件本身是幂等的展示数据。"""
    sub_id, queue_ = state.broker.subscribe()
    return StreamingResponse(
        state.broker.stream(sub_id, queue_),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tools/demo")
def demo_tools() -> dict[str, Any]:
    return {
        "tools": [
            {"key": key, "name": spec.name, "description": spec.description,
             "parameters": spec.parameters}
            for key, spec in DEMO_TOOLS.items()
        ]
    }


@router.post("/admin/models/unload")
def unload_model(
    name: str = Query(...),
    confirm: int = Query(0, description="必须为 1；卸载会丢 KV 缓存，下次请求变冷启动"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    if confirm != 1:
        raise HTTPException(status_code=400, detail="卸载需要 confirm=1")
    result = state.runtime.provider.unload(name)
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "卸载失败")
    return {"ok": True, "action": result.action, "detail": result.detail}
