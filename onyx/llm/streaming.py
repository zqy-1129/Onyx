"""流式增量缝合。

为什么单独一层：工具调用的参数在流里是**分片**到达的，thinking 与 content 是**两条流**，
而最终 token 计数只在最后一个事件里。这三件事如果散落在 provider 里，
每接一个新引擎就要重犯一次同样的错。

本模块是纯函数式的状态机，不发网络请求，因此可以离线穷举测试
（分片切分位置、畸形 JSON、截断、空 tool_calls 等）。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

from onyx.core.types import (
    EngineLatency,
    FinishReason,
    Generation,
    ParseStatus,
    Status,
    TokenSample,
    TokenSource,
    ToolCall,
)

#: Ollama done_reason → 归一化 FinishReason
_OLLAMA_DONE_REASON: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
    "eos": FinishReason.EOS,
    "cancelled": FinishReason.CANCELLED,
}
#: OpenAI finish_reason → 归一化
_OPENAI_FINISH_REASON: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


@dataclass(slots=True)
class _ToolAcc:
    index: int
    id: str = ""
    name: str = ""
    args_obj: dict[str, Any] | None = None
    arg_fragments: list[str] = field(default_factory=list)
    raw_fragments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StreamAssembler:
    """把增量 chunk 缝成一个 `Generation`。

    同时支持两种线格式：
    - Ollama 原生：`message.content` / `message.thinking` / `message.tool_calls[].function.arguments`(dict)
    - OpenAI 兼容：`choices[].delta.content`（流式）或 `choices[].message.content`（非流式）
      / `.reasoning_content` / `.tool_calls[].function.arguments`(str 分片)
    """

    style: str = "native"  # native | openai
    model: str = ""
    _content: list[str] = field(default_factory=list)
    _thinking: list[str] = field(default_factory=list)
    _tools: dict[int, _ToolAcc] = field(default_factory=dict)
    _finish: FinishReason = FinishReason.UNKNOWN
    _usage: list[TokenSample] = field(default_factory=list)
    _latency: dict[str, int] = field(default_factory=dict)
    _raw_count: int = 0
    _last_raw: dict[str, Any] = field(default_factory=dict)
    _done: bool = False
    _error: str = ""
    _first_token_seen: bool = False
    _auto_index: int = 0

    # ── 喂入 ──────────────────────────────────────────────────────
    def feed(self, chunk: dict[str, Any]) -> bool:
        """处理一个 chunk；返回 True 表示这是首个内容分片（用于 TTFT）。"""
        self._raw_count += 1
        self._last_raw = chunk
        if self.style == "openai":
            self._feed_openai(chunk)
        else:
            self._feed_native(chunk)
        first = not self._first_token_seen and bool(self._content or self._thinking or self._tools)
        self._first_token_seen = self._first_token_seen or first
        return first

    def _feed_native(self, chunk: dict[str, Any]) -> None:
        message = chunk.get("message") or {}
        if text := message.get("content"):
            self._content.append(str(text))
        if thinking := message.get("thinking"):
            self._thinking.append(str(thinking))
        for raw_call in message.get("tool_calls") or []:
            self._absorb_tool_call(raw_call, style="native")
        if chunk.get("error"):
            self._error = str(chunk["error"])
        if chunk.get("done"):
            self._absorb_native_final(chunk)

    def _absorb_native_final(self, chunk: dict[str, Any]) -> None:
        self._done = True
        reason = str(chunk.get("done_reason") or "")
        self._finish = _OLLAMA_DONE_REASON.get(reason, FinishReason.UNKNOWN)
        for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
            if isinstance(chunk.get(key), int):
                self._latency[key] = int(chunk[key])
        sample = TokenSample(
            source=TokenSource.ENGINE,
            in_tokens=_int_or_none(chunk.get("prompt_eval_count")),
            out_tokens=_int_or_none(chunk.get("eval_count")),
            thinking_tokens=_int_or_none(chunk.get("thinking_eval_count")),
            cached_tokens=_int_or_none(chunk.get("prompt_eval_cached_count")),
            ok=chunk.get("prompt_eval_count") is not None or chunk.get("eval_count") is not None,
            note="" if chunk.get("eval_count") is not None else "引擎未返回计数",
        )
        self._usage.append(sample)

    def _feed_openai(self, chunk: dict[str, Any]) -> None:
        for choice in chunk.get("choices") or []:
            # 流式形状是 `delta`，非流式是 `message`。两种都吃：如果只认 delta，
            # 非流式响应会解析成"空正文"，而它看起来和"模型真的没输出"一模一样
            delta = choice.get("delta") or choice.get("message") or {}
            if text := delta.get("content"):
                self._content.append(str(text))
            thinking = delta.get("reasoning_content") or delta.get("reasoning")
            if thinking:
                self._thinking.append(str(thinking))
            for raw_call in delta.get("tool_calls") or []:
                self._absorb_tool_call(raw_call, style="openai")
            if choice.get("finish_reason"):
                self._finish = _OPENAI_FINISH_REASON.get(
                    str(choice["finish_reason"]), FinishReason.UNKNOWN
                )
        if chunk.get("error"):
            self._error = str(chunk["error"].get("message", chunk["error"]))
        if usage := chunk.get("usage"):
            self._usage.append(
                TokenSample(
                    source=TokenSource.COMPAT,
                    in_tokens=_int_or_none(usage.get("prompt_tokens")),
                    out_tokens=_int_or_none(usage.get("completion_tokens")),
                    cached_tokens=_int_or_none((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
                    ok=usage.get("prompt_tokens") is not None,
                    note="" if usage.get("prompt_tokens") is not None else "需 stream_options.include_usage",
                )
            )
        if chunk.get("done") or any(c.get("finish_reason") for c in chunk.get("choices") or []):
            self._done = True

    def _absorb_tool_call(self, raw: dict[str, Any], *, style: str) -> None:
        # OpenAI 风格用 index 标识分片归属；原生风格的 tool_calls 不带 index，
        # 若一律落到 0 号累加器，**并行工具调用会互相覆盖**（只剩最后一个）。
        raw_index = raw.get("index")
        if raw_index is None:
            index = self._auto_index
            self._auto_index += 1
        else:
            index = int(raw_index)
        acc = self._tools.setdefault(index, _ToolAcc(index=index))
        if raw.get("id"):
            acc.id = str(raw["id"])
        function = raw.get("function") or {}
        if name := function.get("name"):
            acc.name = str(name) if not acc.name else acc.name + str(name)
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            acc.args_obj = arguments
        elif isinstance(arguments, str) and arguments:
            acc.arg_fragments.append(arguments)
        if tool := raw.get("tool"):  # 部分引擎用 tool 而非 function
            acc.name = acc.name or str(tool.get("name", ""))

    # ── 产出 ──────────────────────────────────────────────────────
    @property
    def saw_first_token(self) -> bool:
        return self._first_token_seen

    def build(self, *, model: str = "", status: Status = Status.OK, error: str = "") -> Generation:
        text = "".join(self._content)
        thinking = "".join(self._thinking)
        calls = tuple(self._build_tool_call(acc) for acc in sorted(self._tools.values(), key=lambda a: a.index))
        finish = self._finish
        if finish is FinishReason.UNKNOWN and calls and self._done:
            # 引擎完全没报 done_reason 时才派生；报了 stop 就保留 stop（见 Generation.wants_tool_call）
            finish = FinishReason.TOOL_CALLS
        resolved_status = status
        resolved_error = error or self._error
        if resolved_error and resolved_status is Status.OK:
            resolved_status = Status.ERROR
        return Generation(
            text=text,
            thinking=thinking,
            tool_calls=calls,
            finish_reason=finish,
            status=resolved_status,
            error=resolved_error,
            usage=tuple(self._usage),
            latency=self._build_latency(),
            model=model or self.model,
            extra={"raw_chunks": self._raw_count, "style": self.style, "saw_done": self._done},
        )

    def _build_tool_call(self, acc: _ToolAcc) -> ToolCall:
        raw_joined = "".join(acc.arg_fragments)
        if acc.args_obj is not None:
            return ToolCall(
                index=acc.index, id=acc.id, name=acc.name, arguments=acc.args_obj,
                arguments_raw=raw_joined or json.dumps(acc.args_obj, ensure_ascii=False),
                parse_status=ParseStatus.OK, parse_source="native_head",
            )
        if not raw_joined:
            status = ParseStatus.OK if not acc.name else ParseStatus.MISSING_REQUIRED
            return ToolCall(
                index=acc.index, id=acc.id, name=acc.name, arguments={}, arguments_raw="",
                parse_status=status, parse_source="native_head",
            )
        try:
            parsed = json.loads(raw_joined)
        except json.JSONDecodeError as exc:
            # 截断与格式错必须分开：前者是 max_tokens 不够，后者是模型不会用工具
            truncated = self._finish is FinishReason.LENGTH or _looks_truncated(raw_joined, exc)
            return ToolCall(
                index=acc.index, id=acc.id, name=acc.name, arguments=None, arguments_raw=raw_joined,
                parse_status=ParseStatus.TRUNCATED if truncated else ParseStatus.JSON_ERROR,
                parse_source="native_head",
            )
        if not isinstance(parsed, dict):
            return ToolCall(
                index=acc.index, id=acc.id, name=acc.name, arguments=None, arguments_raw=raw_joined,
                parse_status=ParseStatus.TYPE_MISMATCH, parse_source="native_head",
            )
        return ToolCall(
            index=acc.index, id=acc.id, name=acc.name, arguments=parsed, arguments_raw=raw_joined,
            parse_status=ParseStatus.OK, parse_source="native_head",
        )

    def _build_latency(self) -> EngineLatency | None:
        if not self._latency:
            return None
        return EngineLatency(
            total_ns=self._latency.get("total_duration"),
            load_ns=self._latency.get("load_duration"),
            prompt_eval_ns=self._latency.get("prompt_eval_duration"),
            eval_ns=self._latency.get("eval_duration"),
        )

    @property
    def last_raw(self) -> dict[str, Any]:
        return self._last_raw


def _looks_truncated(raw: str, exc: json.JSONDecodeError) -> bool:
    """JSON 在半路断掉（未闭合括号/引号）视为截断，而不是模型写错了格式。"""
    if exc.msg in {"Unterminated string starting at", "Expecting value", "Expecting ',' delimiter",
                   "Expecting ':' delimiter", "Expecting property name enclosed in double quotes"}:
        return exc.pos >= len(raw) - 1 or raw.count("{") > raw.count("}")
    return False


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


# ── chunk 流 → Generation + 事件 ───────────────────────────────────
# 真实适配器与 MockProvider 共用这一段：mock 只有在"产出同样的事件"时，
# 用它测出来的观测管道才等价于真实管道。
def emit_final_events(
    gen: Generation,
    raw: dict[str, Any],
    *,
    trace_id: str,
    clock: Any,
    on_event: Any = None,
    ttft_ms: float | None = None,
) -> None:
    from onyx.core.event import EventType, make_event
    from onyx.llm.providers.base import emit

    if gen.latency and gen.latency.load_ns:
        emit(on_event, make_event(
            EventType.MODEL_LOAD, trace_id,
            {"cold": gen.latency.is_cold, "load_duration_ns": gen.latency.load_ns, "model": gen.model},
            clock=clock,
        ))
    if ttft_ms is None and gen.latency and gen.latency.prompt_eval_ns:
        # 非流式没有真实 TTFT：用 prompt_eval 时长做代理，并显式标注这是代理值
        ttft_ms = gen.latency.prompt_eval_ns / 1e6
        emit(on_event, make_event(
            EventType.FIRST_TOKEN, trace_id,
            {"ttft_ms": ttft_ms, "proxy": "prompt_eval_duration"}, clock=clock,
        ))
    elif ttft_ms is not None:
        emit(on_event, make_event(EventType.FIRST_TOKEN, trace_id, {"ttft_ms": ttft_ms}, clock=clock))

    engine = gen.usage_from(TokenSource.ENGINE)
    if engine is not None:
        emit(on_event, make_event(
            EventType.USAGE_ENGINE, trace_id,
            {
                "in_tokens": engine.in_tokens, "out_tokens": engine.out_tokens,
                "thinking_tokens": engine.thinking_tokens, "cached_tokens": engine.cached_tokens,
                "ok": engine.ok, "note": engine.note,
                "latency_ns": {
                    "total": gen.latency.total_ns if gen.latency else None,
                    "load": gen.latency.load_ns if gen.latency else None,
                    "prompt_eval": gen.latency.prompt_eval_ns if gen.latency else None,
                    "eval": gen.latency.eval_ns if gen.latency else None,
                },
                "raw_keys": sorted(k for k in raw if k.endswith(("_count", "_duration"))),
            },
            clock=clock,
        ))
    emit(on_event, make_event(
        EventType.GENERATION_END, trace_id,
        {
            "finish_reason": str(gen.finish_reason), "done_reason": raw.get("done_reason"),
            "text_chars": len(gen.text), "thinking_chars": len(gen.thinking),
            "tool_calls": [
                {
                    "step": c.index + 1, "name": c.name, "call_id": c.id, "args": c.arguments,
                    "args_raw": c.arguments_raw, "parse_status": str(c.parse_status),
                    "parse_source": c.parse_source,
                }
                for c in gen.tool_calls
            ],
        },
        clock=clock,
    ))


def consume_chunks(
    chunks: Any,
    *,
    trace_id: str,
    clock: Any,
    style: str = "native",
    model: str = "",
    on_event: Any = None,
) -> Generation:
    """消费 ndjson chunk 流，边缝合边发事件，返回最终 Generation。"""
    from onyx.core.event import EventType, make_event
    from onyx.llm.providers.base import emit

    assembler = StreamAssembler(style=style, model=model)
    start = clock.monotonic_ns()
    ttft_ms: float | None = None
    seq = 0
    last_raw: dict[str, Any] = {}
    for chunk in chunks:
        last_raw = chunk
        if chunk.get("_unparsed"):
            emit(on_event, make_event(
                EventType.ANOMALY, trace_id,
                {"code": "UNPARSED_STREAM_LINE", "severity": "warn",
                 "detail": {"line": str(chunk["_unparsed"])[:500]}},
                clock=clock,
            ))
            continue
        if assembler.feed(chunk) and ttft_ms is None:
            ttft_ms = (clock.monotonic_ns() - start) / 1e6
        message = chunk.get("message") or (chunk.get("choices") or [{}])[0].get("delta") or {}
        if text := message.get("content"):
            emit(on_event, make_event(
                EventType.TEXT_DELTA, trace_id, {"seq": seq, "text": str(text)}, clock=clock))
            seq += 1
        if thinking := message.get("thinking") or message.get("reasoning_content"):
            emit(on_event, make_event(
                EventType.THINKING_DELTA, trace_id, {"seq": seq, "text": str(thinking)}, clock=clock))
            seq += 1
        for idx, raw_call in enumerate(message.get("tool_calls") or []):
            function = raw_call.get("function") or {}
            emit(on_event, make_event(
                EventType.TOOL_CALL_DELTA, trace_id,
                {"idx": idx, "name_fragment": function.get("name", ""),
                 "args_fragment": function.get("arguments")},
                clock=clock,
            ))
    gen = dataclasses.replace(
        assembler.build(model=model, status=Status.OK),
        ttft_ms=ttft_ms,
        wall_ms=(clock.monotonic_ns() - start) / 1e6,
    )
    emit_final_events(gen, last_raw, trace_id=trace_id, clock=clock, on_event=on_event, ttft_ms=ttft_ms)
    return gen
