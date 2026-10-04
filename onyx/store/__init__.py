"""L0 存储层：SQLite/WAL + 幂等迁移 + repo + 可替换 sink。

只依赖 `onyx.core` 与 stdlib（import-linter 契约强制）。
"""

from __future__ import annotations

from .db import Database, discover_migrations
from .records import (
    AnomalyRecord,
    ModelRecord,
    ProviderRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)
from .repos import ModelRepo, TraceRepo, UsageRepo
from .retention import DiskReport, Outcome, disk_report, history, parse_window, sweep
from .sinks import (
    EventFanout,
    EventSink,
    JsonlEventSink,
    NullEventSink,
    NullRecordSink,
    RecordFanout,
    RecordSink,
    SqliteRecordSink,
)

__all__ = [
    "AnomalyRecord",
    "Database",
    "DiskReport",
    "EventFanout",
    "EventSink",
    "JsonlEventSink",
    "ModelRecord",
    "ModelRepo",
    "NullEventSink",
    "NullRecordSink",
    "Outcome",
    "ProviderRecord",
    "RecordFanout",
    "RecordSink",
    "SqliteRecordSink",
    "TokenPartRecord",
    "ToolCallRecord",
    "TraceRecord",
    "TraceRepo",
    "UsageAltRecord",
    "UsageRecord",
    "UsageRepo",
    "discover_migrations",
    "disk_report",
    "history",
    "parse_window",
    "sweep",
]
