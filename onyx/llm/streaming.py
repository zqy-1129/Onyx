"""流式增量缝合。

为什么单独一层：工具调用的参数在流里是**分片**到达的，thinking 与 content 是**两条流**，
而最终 token 计数只在最后一个事件里。这三件事如果散落在 provider 里，
每接一个新引擎就要重犯一次同样的错。

本模块是纯函数式的状态机，不发网络请求，因此可以离线穷举测试
（分片切分位置、畸形 JSON、截断、空 tool_calls 等）。
"""

from __future__ import annotations

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
    - OpenAI 兼容：`choices[].delta.content` / `.reasoning_content` / `.tool_calls[].function.arguments`(str 分片)
    """

    style: str = "native"  # native | openai
    model: str = ""
    _content: list[str] = field(default_factory=list)
    _thinking: list[str] = field(default_factory=list)
    _tools: dict[int, _ToolAcc] = field(default_factory=dict)
    _finish: FinishReason = FinishReason.UNKNOWN
    _usage: list[TokenSample] = field(default_factory=list)
    _latency: dict[str, int] = field(default_factory=dict)
    _raw: list[dict[str, Any]] = field(default_factory=list)
    _done: bool = False
    _error: str = ""
    _first_token_seen: bool = False

    # ── 喂入 ──────────────────────────────────────────────────────
    def feed(self, chunk: dict[str, Any]) -> bool:
        """处理一个 chunk；返回 True 表示这是首个内容分片（用于 TTFT）。"""
        self._raw.append(chunk)
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
            delta = choice.get("delta") or {}
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
        index = int(raw.get("index") or 0)
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
            extra={"raw_chunks": len(self._raw), "style": self.style, "saw_done": self._done},
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

    def raw_chunks(self) -> list[dict[str, Any]]:
        return list(self._raw)


def _looks_truncated(raw: str, exc: json.JSONDecodeError) -> bool:
    """JSON 在半路断掉（未闭合括号/引号）视为截断，而不是模型写错了格式。"""
    if exc.msg in {"Unterminated string starting at", "Expecting value", "Expecting ',' delimiter",
                   "Expecting ':' delimiter", "Expecting property name enclosed in double quotes"}:
        return exc.pos >= len(raw) - 1 or raw.count("{") > raw.count("}")
    return False


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
