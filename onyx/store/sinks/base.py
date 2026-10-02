"""Sink 抽象：把"记录什么"与"记到哪"彻底分开。

两类 sink，职责不同、可独立替换：
- `EventSink`  —— 原始事件流（NDJSON 日志、SSE 转发、Langfuse/OTLP 导出都实现它）
- `RecordSink` —— 结构化记录（SQLite、DuckDB、Parquet 导出实现它）

Fanout 对每个下游做**错误隔离**：一个 sink 崩了不许影响请求主链路（DESIGN 原则 2）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from onyx.core.event import TraceEvent
from onyx.store.records import (
    AnomalyRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)

log = logging.getLogger("onyx.store")


@runtime_checkable
class EventSink(Protocol):
    name: str

    def emit(self, event: TraceEvent) -> None: ...
    def flush(self, timeout: float = 1.0) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class RecordSink(Protocol):
    name: str

    def write_trace(self, rec: TraceRecord) -> None: ...
    def finish_trace(self, trace_id: str, **fields: object) -> None: ...
    def write_usage(
        self,
        rec: UsageRecord,
        *,
        alts: Sequence[UsageAltRecord] = (),
        parts: Iterable[TokenPartRecord] = (),
    ) -> None: ...
    def write_tool_call(self, rec: ToolCallRecord) -> None: ...
    def write_anomaly(self, rec: AnomalyRecord) -> None: ...
    def flush(self, timeout: float = 1.0) -> None: ...
    def close(self) -> None: ...


class EventFanout:
    """多路分发原始事件。任何下游异常都被吞掉并计数，主链路继续。"""

    name = "fanout"

    def __init__(self, sinks: Sequence[EventSink] = ()) -> None:
        self.sinks: list[EventSink] = list(sinks)
        self.errors: dict[str, int] = {}

    def add(self, sink: EventSink) -> EventSink:
        self.sinks.append(sink)
        return sink

    def emit(self, event: TraceEvent) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event)
            except Exception as exc:  # noqa: BLE001 - 下游 sink 失败必须被隔离，不许影响主链路
                self.errors[sink.name] = self.errors.get(sink.name, 0) + 1
                log.warning("event sink %s 失败: %s", sink.name, exc)

    def flush(self, timeout: float = 1.0) -> None:
        for sink in self.sinks:
            try:
                sink.flush(timeout)
            except Exception as exc:  # noqa: BLE001 - 下游 sink 失败必须被隔离，不许影响主链路
                self.errors[sink.name] = self.errors.get(sink.name, 0) + 1
                log.warning("event sink %s flush 失败: %s", sink.name, exc)

    def close(self) -> None:
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as exc:  # noqa: BLE001 - 下游 sink 失败必须被隔离，不许影响主链路
                self.errors[sink.name] = self.errors.get(sink.name, 0) + 1
                log.warning("event sink %s close 失败: %s", sink.name, exc)


class RecordFanout:
    """多路分发结构化记录（如同时写 SQLite 与导出 Parquet）。"""

    name = "record-fanout"

    def __init__(self, sinks: Sequence[RecordSink] = ()) -> None:
        self.sinks: list[RecordSink] = list(sinks)
        self.errors: dict[str, int] = {}

    def _dispatch(self, method: str, *args: object, **kwargs: object) -> None:
        for sink in self.sinks:
            try:
                getattr(sink, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - 下游 sink 失败必须被隔离，不许影响主链路
                self.errors[sink.name] = self.errors.get(sink.name, 0) + 1
                log.warning("record sink %s.%s 失败: %s", sink.name, method, exc)

    def write_trace(self, rec: TraceRecord) -> None:
        self._dispatch("write_trace", rec)

    def finish_trace(self, trace_id: str, **fields: object) -> None:
        self._dispatch("finish_trace", trace_id, **fields)

    def write_usage(
        self,
        rec: UsageRecord,
        *,
        alts: Sequence[UsageAltRecord] = (),
        parts: Iterable[TokenPartRecord] = (),
    ) -> None:
        parts = list(parts)
        self._dispatch("write_usage", rec, alts=alts, parts=parts)

    def write_tool_call(self, rec: ToolCallRecord) -> None:
        self._dispatch("write_tool_call", rec)

    def write_anomaly(self, rec: AnomalyRecord) -> None:
        self._dispatch("write_anomaly", rec)

    def flush(self, timeout: float = 1.0) -> None:
        self._dispatch("flush", timeout)

    def close(self) -> None:
        self._dispatch("close")
