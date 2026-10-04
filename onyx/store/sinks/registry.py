"""Sink 注册表（扩展点 `onyx.sinks`）。

内置 sink 静态登记，外部导出走 entry points：
    [project.entry-points."onyx.sinks"]
    otlp = "my_pkg.export:OtlpSinkBuilder"

契约：`builder(**options) -> EventSink`。内核传给它已知的环境（`data_dir`），
其余配置由实现自己按行业惯例读（如 OTLP 读 `OTEL_EXPORTER_OTLP_ENDPOINT`）——
不在 onyx 里为每种后端复制一套 flag，那会让"加一个 sink"变成"改一次内核 CLI"。

sink 是**只写**的观测出口，任何 sink 崩了都由 `EventFanout` 隔离（DESIGN 原则 2）。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from onyx.discovery import GROUP_SINKS, discover, failures
from onyx.store.sinks.base import EventSink
from onyx.store.sinks.jsonl import JsonlEventSink
from onyx.store.sinks.null import NullEventSink

Builder = Callable[..., EventSink]


def _builtin_jsonl(**options: Any) -> EventSink:
    path = options.get("path")
    if not path:
        data_dir = options.get("data_dir")
        if not data_dir:
            raise ValueError("sink 'jsonl' 需要 path=<文件>，或 data_dir=<数据目录>")
        path = Path(data_dir) / "events.ndjson"
    return JsonlEventSink(path, buffer_limit=int(options.get("buffer_limit") or 256))


def _builtin_null(**_options: Any) -> EventSink:
    return NullEventSink()


BUILTIN_SINKS: dict[str, Builder] = {
    "jsonl": _builtin_jsonl,
    "null": _builtin_null,
}


def sink_names() -> tuple[str, ...]:
    return tuple(sorted(discover(GROUP_SINKS, BUILTIN_SINKS)))


def build_event_sink(name: str, **options: Any) -> EventSink:
    """按名字构造一个事件 sink。未知名字报错并列出可选项（静默不导出比不导出更糟）。"""
    registry = discover(GROUP_SINKS, BUILTIN_SINKS)
    builder = registry.get(name)
    if builder is None:
        broken = [f.name for f in failures() if f.group == GROUP_SINKS]
        hint = f"；该组有插件加载失败: {broken}（onyx plugins 有详情）" if broken else ""
        raise KeyError(f"未知 sink {name!r}；可选: {sorted(registry)}{hint}")
    return builder(**options)


__all__ = ["BUILTIN_SINKS", "build_event_sink", "sink_names"]
