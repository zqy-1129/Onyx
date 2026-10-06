"""/api/embed 的实现（S33）。

实测形状（Ollama 0.35.1 + qwen3-embedding:0.6b，2026-10-06）：

    请求  {"model": …, "input": ["a", "b"]}
    响应  {"model": …, "embeddings": [[1024 floats], …],
           "total_duration": ns, "load_duration": ns, "prompt_eval_count": 10}

三件事按实测写死了判断，不是照文档猜的：
- **顺序保持**：`embeddings[i]` 对应 `input[i]`（实测单条调用与混在 20 条一批里
  的向量 cos = 1.0，冷启动首调与热调也是 1.0）。
- **批次不改变向量**，所以批量只是把请求数降下来（排队友好），**不宣称更快**：
  warm 状态下 4 条一批 74ms，逐条 4 次约 50ms。
- 只有 `prompt_eval_count` 这一个计数，**没有输出 token**——向量的"输出"不是 token。
  所以 `out_tokens=None` 而不是 0（「未知 ≠ 0」）。
"""

from __future__ import annotations

from typing import Any

from onyx.core.clock import Clock
from onyx.core.event import EventType, make_event
from onyx.core.types import (
    Confidence,
    Embedding,
    EmbedRequest,
    EngineLatency,
    Status,
    TokenSample,
    TokenSource,
)
from onyx.llm.providers.base import EventCB, emit
from onyx.llm.providers.ollama.client import OllamaClient


def embed(
    client: OllamaClient,
    req: EmbedRequest,
    *,
    trace_id: str,
    clock: Clock,
    on_event: EventCB | None = None,
) -> Embedding:
    """一次批量向量化。异常（引擎不可达 / 超时 / 被拒）照常抛给 gateway 记 trace。"""
    payload: dict[str, Any] = {"model": req.model, "input": list(req.inputs)}
    if req.keep_alive:
        payload["keep_alive"] = req.keep_alive
    raw = client.post_json("/api/embed", payload)
    return _parse(raw, req=req, trace_id=trace_id, clock=clock, on_event=on_event)


def _parse(
    raw: dict[str, Any],
    *,
    req: EmbedRequest,
    trace_id: str,
    clock: Clock,
    on_event: EventCB | None,
) -> Embedding:
    vectors = tuple(
        tuple(float(x) for x in row) for row in (raw.get("embeddings") or [])
    )
    latency = EngineLatency(
        total_ns=_int(raw.get("total_duration")),
        load_ns=_int(raw.get("load_duration")),
        prompt_eval_ns=_int(raw.get("prompt_eval_duration")),
    )
    in_tokens = _int(raw.get("prompt_eval_count"))
    usage: tuple[TokenSample, ...] = ()
    if in_tokens is not None:
        usage = (TokenSample(
            source=TokenSource.ENGINE, in_tokens=in_tokens, out_tokens=None,
            confidence=Confidence.HIGH,
        ),)

    # 条数不对是**错误**而不是"少给几条"：少一条会让后面每一条都错位，
    # 而错位的排名看着完全正常——静默错位比失败难发现得多。
    if len(vectors) != req.batch_size:
        result = Embedding(
            vectors=(), model=req.model, status=Status.ERROR,
            error=(
                f"引擎返回 {len(vectors)} 条向量，请求了 {req.batch_size} 条"
                "（对不齐就不判分：错位排名看着完全正常，比报错难发现）"
            ),
            usage=usage, latency=latency,
            extra={"requested": req.batch_size, "returned": len(vectors)},
        )
    else:
        result = Embedding(vectors=vectors, model=req.model, status=Status.OK, usage=usage,
                           latency=latency, extra={"returned": len(vectors)})

    _emit(result, raw, trace_id=trace_id, clock=clock, on_event=on_event)
    return result


def _emit(
    result: Embedding,
    raw: dict[str, Any],
    *,
    trace_id: str,
    clock: Clock,
    on_event: EventCB | None,
) -> None:
    """把引擎自报的计数与耗时发成事件：看板上那个数字必须有出处。"""
    if result.latency and result.latency.load_ns:
        emit(on_event, make_event(EventType.MODEL_LOAD, trace_id, {
            "cold": result.latency.is_cold, "load_duration_ns": result.latency.load_ns,
            "model": result.model,
        }, clock=clock))
    engine = result.usage_from(TokenSource.ENGINE)
    if engine is not None:
        emit(on_event, make_event(EventType.USAGE_ENGINE, trace_id, {
            "in_tokens": engine.in_tokens, "out_tokens": None, "thinking_tokens": None,
            "cached_tokens": None, "ok": engine.ok, "note": "",
            "latency_ns": {
                "total": result.latency.total_ns if result.latency else None,
                "load": result.latency.load_ns if result.latency else None,
                "prompt_eval": result.latency.prompt_eval_ns if result.latency else None,
                "eval": None,
            },
            "raw_keys": sorted(k for k in raw if k.endswith(("_count", "_duration"))),
        }, clock=clock))


def _int(value: Any) -> int | None:
    """引擎没报就是 None。填 0 会造出一个"耗时 0ns / 0 token"的假事实。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
