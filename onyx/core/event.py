"""L3 事件契约 —— gateway 与观测层之间唯一的耦合面。

设计约束（DESIGN §10）：
- 加字段 = 兼容变更；改名/删字段/改语义 = 递增 CONTRACT_VERSION，消费者按 `v` 分派。
- 消费者对**未知事件类型必须忽略而非崩溃**（`from_dict` 落到 `EventType.UNKNOWN`）。
- payload 允许携带表未声明的额外键（前向兼容），但缺必填键立即报错——
  宁可早失败，也不要让"少一个字段"变成看板上的 0。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .clock import SYSTEM_CLOCK, Clock
from .errors import UnknownEventType

CONTRACT_VERSION = 1


class EventType(StrEnum):
    TRACE_START = "trace_start"
    MODEL_LOAD = "model_load"
    FIRST_TOKEN = "first_token"
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    GENERATION_END = "generation_end"
    USAGE_ENGINE = "usage_engine"
    USAGE_COMPAT = "usage_compat"
    USAGE_LOCAL = "usage_local"
    USAGE_ATTRIBUTION = "usage_attribution"
    RECONCILED = "reconciled"
    TOOL_EXEC_START = "tool_exec_start"
    TOOL_EXEC_END = "tool_exec_end"
    GPU_SAMPLE = "gpu_sample"
    ANOMALY = "anomaly"
    TRACE_END = "trace_end"
    UNKNOWN = "unknown"


#: 每种事件的必填 payload 键。表驱动测试会断言 EventType 与此表一一对应（UNKNOWN 除外）。
PAYLOAD_REQUIRED: dict[EventType, frozenset[str]] = {
    EventType.TRACE_START: frozenset({"kind", "purpose", "provider_id", "model"}),
    EventType.MODEL_LOAD: frozenset({"cold"}),
    EventType.FIRST_TOKEN: frozenset({"ttft_ms"}),
    EventType.TEXT_DELTA: frozenset({"seq", "text"}),
    EventType.THINKING_DELTA: frozenset({"seq", "text"}),
    EventType.TOOL_CALL_DELTA: frozenset({"idx"}),
    EventType.GENERATION_END: frozenset({"finish_reason"}),
    EventType.USAGE_ENGINE: frozenset(),  # 计数可能全缺（引擎没报），缺即是信息
    EventType.USAGE_COMPAT: frozenset(),
    EventType.USAGE_LOCAL: frozenset({"source"}),
    EventType.USAGE_ATTRIBUTION: frozenset(),
    EventType.RECONCILED: frozenset({"chosen_source", "confidence"}),
    EventType.TOOL_EXEC_START: frozenset({"name", "step"}),
    EventType.TOOL_EXEC_END: frozenset({"name", "step", "status"}),
    EventType.GPU_SAMPLE: frozenset(),
    EventType.ANOMALY: frozenset({"code", "severity"}),
    EventType.TRACE_END: frozenset({"status", "wall_ms"}),
}

#: 可选键（文档化用途，便于 visitor 作者发现可用字段）
PAYLOAD_OPTIONAL: dict[EventType, frozenset[str]] = {
    EventType.TRACE_START: frozenset({"messages_ref", "tools_ref", "params", "context", "trace_parent"}),
    EventType.MODEL_LOAD: frozenset({"load_duration_ns", "model"}),
    EventType.FIRST_TOKEN: frozenset({"model"}),
    EventType.TEXT_DELTA: frozenset({"model"}),
    EventType.THINKING_DELTA: frozenset({"model"}),
    EventType.TOOL_CALL_DELTA: frozenset({"name_fragment", "args_fragment"}),
    EventType.GENERATION_END: frozenset({"output_ref", "raw_response_ref", "done_reason"}),
    EventType.USAGE_ENGINE: frozenset(
        {"in_tokens", "out_tokens", "thinking_tokens", "cached_tokens", "latency_ns"}
    ),
    EventType.USAGE_COMPAT: frozenset({"in_tokens", "out_tokens", "total_tokens"}),
    EventType.USAGE_LOCAL: frozenset({"in_tokens", "out_tokens", "thinking_tokens", "cached_tokens",
                                      "confidence", "ok", "note"}),
    EventType.USAGE_ATTRIBUTION: frozenset({"parts", "attribution", "count_source"}),
    EventType.RECONCILED: frozenset({"in_tokens", "out_tokens", "drift_pct", "alts"}),
    EventType.TOOL_EXEC_START: frozenset({"args_ref", "tool_id", "executed_by"}),
    EventType.TOOL_EXEC_END: frozenset(
        {"latency_ms", "result_ref", "result_bytes", "error", "executed_by", "mocked"}
    ),
    EventType.GPU_SAMPLE: frozenset({"size_vram", "size", "context_length", "model"}),
    EventType.ANOMALY: frozenset({"detail", "trace_ref"}),
    EventType.TRACE_END: frozenset({"error", "usage_summary"}),
    EventType.UNKNOWN: frozenset(),
}


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """一次事件。`ts_ns` 用单调钟便于算间隔，`wall_iso` 用于落库与展示。"""

    type: EventType
    trace_id: str
    ts_ns: int
    wall_iso: str
    payload: dict[str, Any] = field(default_factory=dict)
    version: int = CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": self.version,
            "type": str(self.type),
            "trace_id": self.trace_id,
            "ts_ns": self.ts_ns,
            "wall_iso": self.wall_iso,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), default=str)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TraceEvent:
        """宽容解析：未知类型 → UNKNOWN 并保留原文（原则 6：不丢数据、不崩溃）。"""
        try:
            etype = EventType(raw.get("type", ""))
        except ValueError:
            etype = EventType.UNKNOWN
        payload = dict(raw.get("payload") or {})
        if etype is EventType.UNKNOWN:
            payload.setdefault("_raw_type", raw.get("type"))
        return cls(
            type=etype,
            trace_id=str(raw.get("trace_id", "")),
            ts_ns=int(raw.get("ts_ns", 0)),
            wall_iso=str(raw.get("wall_iso", "")),
            payload=payload,
            version=int(raw.get("v", CONTRACT_VERSION)),
        )

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)


EventCB = Any  # Callable[[TraceEvent], None]；用 Any 避免在 core 里引入 typing 环


def make_event(
    etype: EventType | str,
    trace_id: str,
    payload: dict[str, Any] | None = None,
    *,
    clock: Clock = SYSTEM_CLOCK,
    strict: bool = True,
) -> TraceEvent:
    """事件工厂。`strict=True` 时校验类型已知且必填键齐备。"""
    resolved = etype if isinstance(etype, EventType) else _coerce_type(etype, strict=strict)
    data = dict(payload or {})
    if strict and resolved is not EventType.UNKNOWN:
        required = PAYLOAD_REQUIRED.get(resolved, frozenset())
        missing = sorted(k for k in required if k not in data)
        if missing:
            raise UnknownEventType(
                f"事件 {resolved} 缺少必填 payload 键: {missing}",
                detail={"event_type": str(resolved), "missing": missing},
            )
    return TraceEvent(
        type=resolved,
        trace_id=trace_id,
        ts_ns=clock.monotonic_ns(),
        wall_iso=clock.wall_iso(),
        payload=data,
    )


def _coerce_type(value: str, *, strict: bool) -> EventType:
    try:
        return EventType(value)
    except ValueError:
        if strict:
            raise UnknownEventType(f"未知事件类型: {value!r}", detail={"event_type": value}) from None
        return EventType.UNKNOWN


def declared_event_types() -> frozenset[EventType]:
    """给表驱动测试与文档生成用。"""
    return frozenset(PAYLOAD_REQUIRED) | frozenset(PAYLOAD_OPTIONAL)
