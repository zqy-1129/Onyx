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
from onyx.core.errors import GpuLockBusy
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
    # 排队失败要报**谁在占用、预计还要多久**，而不是只说"稍后重试"：
    # 本地 GPU 一次请求可能就是几十秒，让人盲等只会换来一次 Ctrl-C
    try:
        state.gpu_lock.acquire(timeout=GPU_LOCK_TIMEOUT)
    except GpuLockBusy as exc:
        raise HTTPException(status_code=429, detail=exc.message) from None
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


def _admin_provider(state: AppState, action: str):
    """控制面三件事（unload / pull / rm）都要先确认这个通道真的暴露控制面。

    OpenAI 兼容层没有统一的这些端点，`AdminProvider` 也就不实现它 ——
    直接 `provider.unload(...)` 会抛 AttributeError，界面上就变成一条 500。
    报 501 并说清"为什么不做个假的"比那样有用。
    """
    from onyx.core.types import Cap

    provider = state.runtime.provider
    if Cap.ADMIN not in provider.capabilities():
        raise HTTPException(
            status_code=501,
            detail=f"{provider.id}（{provider.kind} 通道）不暴露控制面，{action} 做不了。"
                   "兼容层没有统一的卸载/拉取/删除端点，编一个会让「显存已经让出来了」这种"
                   "判断建立在谎话上；请换 --provider ollama，或直接在引擎侧操作。",
        )
    return provider


@router.post("/admin/models/unload")
def unload_model(
    name: str = Query(...),
    confirm: int = Query(0, description="必须为 1；卸载会丢 KV 缓存，下次请求变冷启动"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    if confirm != 1:
        raise HTTPException(status_code=400, detail="卸载需要 confirm=1")
    provider = _admin_provider(state, "卸载")
    result = provider.unload(name)
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "卸载失败")
    return {"ok": True, "action": result.action, "detail": result.detail}


@router.post("/admin/models/pull")
def pull_model(
    name: str = Query(..., description="模型名，例如 qwen3:8b"),
    confirm: int = Query(0, description="必须为 1；这会把 GB 级权重写进引擎的模型目录"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """拉取一个模型进本地，并把清单同步回来。

    **这是一次长请求**：几 GB 的下载会占住这条 HTTP 几分钟，而中间层（代理/网关）
    通常在这之前就把连接掐了 —— 那种失败看起来像"Onyx 拉取失败"，实际是超时。
    所以界面上这句话要写出来，大下载请走 `onyx models pull`（它还带进度提示）。
    """
    from onyx.runtime import sync_models

    if confirm != 1:
        raise HTTPException(status_code=400, detail="拉取会写入 GB 级权重，需要 confirm=1")
    provider = _admin_provider(state, "拉取")
    result = provider.pull(name)
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "拉取失败")
    # 拉完必须刷清单：模型不落库的话界面上"拉成功了但列表里没有"，会被当成 bug
    synced = sync_models(state.runtime)
    state.runtime.flush()
    return {
        "ok": True, "action": "pull", "name": name,
        "digest": str(result.detail.get("digest") or ""),
        "models_synced": synced,
    }


@router.post("/admin/models/rm")
def remove_model(
    name: str = Query(...),
    confirm: int = Query(0, description="必须为 1；删除权重不可逆"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """删掉本地模型权重。**只删权重**：历史 trace 与分数一行都不动。

    那次测量已经发生了，删权重不会让它变成没发生 —— 这正是"分数永久、证据有限期"里
    "永久"那一半的意义。
    """
    from onyx.runtime import sync_models

    if confirm != 1:
        raise HTTPException(status_code=400, detail="删除权重不可逆，需要 confirm=1")
    provider = _admin_provider(state, "删除")
    result = provider.delete(name)
    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error or "删除失败")
    synced = sync_models(state.runtime)
    state.runtime.flush()
    return {
        "ok": True, "action": "delete", "name": name, "models_synced": synced,
        "note": "权重已释放；历史 trace 与分数保留",
    }
