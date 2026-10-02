"""Ollama 原生 `/api/chat` 的请求构造与响应解析。

选原生通道而非 `/v1` 的理由（DESIGN §7.1）：只有原生通道同时给到
`prompt_eval_count` / `eval_count` / 四段纳秒时序 / `done_reason` / `keep_alive`，
而 `/v1` 明确不支持 `tool_choice`、`n`、`logit_bias`、`user`。
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

from onyx.core.clock import Clock
from onyx.core.errors import CapabilityMissing, SchemaInvalid
from onyx.core.types import (
    Generation,
    GenerationRequest,
    Message,
    Role,
    Status,
    ToolSpec,
)
from onyx.llm.params import to_ollama_options
from onyx.llm.providers.base import EventCB
from onyx.llm.providers.ollama.client import OllamaClient
from onyx.llm.streaming import StreamAssembler, consume_chunks, emit_final_events

MediaResolver = Callable[[str], bytes]


def build_chat_payload(
    req: GenerationRequest, *, stream: bool, media_resolver: MediaResolver | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": parse_messages(req.messages, media_resolver=media_resolver),
        "stream": stream,
    }
    options = to_ollama_options(req.params)
    if options:
        payload["options"] = options
    if req.tools:
        payload["tools"] = parse_tools(req.tools)
    if req.tool_choice:
        # Ollama 原生接口同样不接受 tool_choice；强制调用只能由客户端 loop 实现
        raise CapabilityMissing(
            "Ollama 不支持 tool_choice，请用 tools/loop.py 的客户端循环实现强制调用",
            detail={"tool_choice": req.tool_choice},
        )
    if req.thinking is not None:
        payload["think"] = bool(req.thinking)
    if req.keep_alive is not None:
        payload["keep_alive"] = req.keep_alive
    if req.params.json_schema:
        payload["format"] = req.params.json_schema
    elif raw_format := req.params.extra.get("format"):
        payload["format"] = raw_format
    return payload


def parse_messages(
    messages: tuple[Message, ...] | list[Message], *, media_resolver: MediaResolver | None = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        item: dict[str, Any] = {"role": str(msg.role), "content": msg.content or ""}
        if msg.role is Role.TOOL and msg.name:
            # Ollama 用 tool_name 关联工具结果；缺失会让模型无法对上号
            item["tool_name"] = msg.name
        if msg.tool_calls:
            item["tool_calls"] = [
                {"function": {"name": c.name, "arguments": c.arguments or {}}} for c in msg.tool_calls
            ]
        if msg.thinking:
            item["thinking"] = msg.thinking
        if msg.media_refs:
            if media_resolver is None:
                raise SchemaInvalid(
                    "消息含 media_refs 但未提供 media_resolver，无法序列化为 base64",
                    detail={"refs": list(msg.media_refs)},
                )
            item["images"] = [
                base64.b64encode(media_resolver(ref)).decode("ascii") for ref in msg.media_refs
            ]
        out.append(item)
    return out


def parse_tools(tools: tuple[ToolSpec, ...] | list[ToolSpec]) -> list[dict[str, Any]]:
    """Ollama 接受 OpenAI 的 function 形状，但只取 function 部分。"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def generate(
    client: OllamaClient,
    req: GenerationRequest,
    *,
    trace_id: str,
    clock: Clock,
    on_event: EventCB | None = None,
    media_resolver: MediaResolver | None = None,
) -> Generation:
    """非流式：一次 POST，用同一个 assembler 解析，保证两条路径口径一致。"""
    payload = build_chat_payload(req, stream=False, media_resolver=media_resolver)
    raw = client.post_json("/api/chat", payload)
    assembler = StreamAssembler(style="native", model=req.model)
    assembler.feed(raw)
    gen = assembler.build(model=req.model, status=Status.OK)
    emit_final_events(gen, raw, trace_id=trace_id, clock=clock, on_event=on_event, ttft_ms=None)
    return gen


def generate_streaming(
    client: OllamaClient,
    req: GenerationRequest,
    *,
    trace_id: str,
    clock: Clock,
    on_event: EventCB | None = None,
    media_resolver: MediaResolver | None = None,
) -> Generation:
    """流式：逐行缝合，并在过程中产出增量事件（TTFT / delta / 末包计数）。

    实测要点（PROBES P13）：计数只在 `done=true` 的最后一个事件里，且与非流式一致。
    缝合与事件产出统一走 `streaming.consume_chunks`，MockProvider 复用同一条路径，
    这样用 mock 测出来的观测管道与真实管道等价。
    """
    payload = build_chat_payload(req, stream=True, media_resolver=media_resolver)
    return consume_chunks(
        client.post_ndjson("/api/chat", payload),
        trace_id=trace_id, clock=clock, style="native", model=req.model, on_event=on_event,
    )
