from __future__ import annotations

from .base import EventFanout, EventSink, RecordFanout, RecordSink
from .jsonl import JsonlEventSink
from .null import NullEventSink, NullRecordSink
from .registry import BUILTIN_SINKS, build_event_sink, builtin_sink_names, sink_names
from .sqlite import SqliteRecordSink

__all__ = [
    "BUILTIN_SINKS",
    "EventFanout",
    "EventSink",
    "JsonlEventSink",
    "NullEventSink",
    "NullRecordSink",
    "RecordFanout",
    "RecordSink",
    "SqliteRecordSink",
    "build_event_sink",
    "builtin_sink_names",
    "sink_names",
]
