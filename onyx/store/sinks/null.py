"""空实现：dry-run、性能基线、以及"关掉观测"的开关。

有一个显式的 null sink 很重要：它让"观测开销有多大"可以被测量
（同一负载跑 null vs sqlite，差值就是观测成本）。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from onyx.core.event import TraceEvent
from onyx.store.records import (
    AnomalyRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)


class NullEventSink:
    name = "null"

    def __init__(self) -> None:
        self.count = 0

    def emit(self, event: TraceEvent) -> None:
        self.count += 1

    def flush(self, timeout: float = 1.0) -> None: ...
    def close(self) -> None: ...


class NullRecordSink:
    name = "null"

    def __init__(self) -> None:
        self.traces = 0
        self.usages = 0
        self.tool_calls = 0
        self.anomalies = 0

    def write_trace(self, rec: TraceRecord) -> None:
        self.traces += 1

    def finish_trace(self, trace_id: str, **fields: object) -> None: ...

    def write_usage(
        self,
        rec: UsageRecord,
        *,
        alts: Sequence[UsageAltRecord] = (),
        parts: Iterable[TokenPartRecord] = (),
    ) -> None:
        self.usages += 1

    def write_tool_call(self, rec: ToolCallRecord) -> None:
        self.tool_calls += 1

    def write_anomaly(self, rec: AnomalyRecord) -> None:
        self.anomalies += 1

    def flush(self, timeout: float = 1.0) -> None: ...
    def close(self) -> None: ...
