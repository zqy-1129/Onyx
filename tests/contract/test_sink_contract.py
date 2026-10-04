"""Sink 抽象契约：每个 `EventSink` 实现对同一套断言负责。

这是 S16c 的本体——导出 sink 的价值不在"能发 HTTP"，而在于**它能被换进来**：
只实现 `emit/flush/close`，不改 `core/**`、`gateway.py`、`obs/**`
（`scripts/check_extension_boundary.py` 判定）。

顺带钉住一条本项目用血换来的教训（SSE broker 事故）：
**隔离机制会掩盖故障**。所以这里必须断言"下游 sink 崩了会被 Fanout 计数并可见"，
而不是只断言"崩了不影响主链路"——只测后半句的话，一条静默失效的导出通道
可以绿上几个里程碑。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from onyx.core.event import EventType, TraceEvent, make_event
from onyx.store.sinks import (
    EventFanout,
    EventSink,
    JsonlEventSink,
    NullEventSink,
    build_event_sink,
)
from onyx.store.sinks.otlp import OtlpEventSink

TRACE = "01M4SINKTRACE0000000000001"


def _stream() -> list[TraceEvent]:
    return [
        make_event(EventType.TRACE_START, TRACE,
                   {"kind": "chat", "purpose": "chat", "provider_id": "p", "model": "m"}),
        make_event(EventType.TEXT_DELTA, TRACE, {"seq": 0, "text": "你好"}),
        make_event(EventType.USAGE_ENGINE, TRACE, {"prompt_eval_count": 12}),
        make_event(EventType.TRACE_END, TRACE, {"status": "ok", "wall_ms": 3.5}),
    ]


def _jsonl(tmp_path) -> EventSink:
    return JsonlEventSink(tmp_path / "events.ndjson")


def _null() -> EventSink:
    return NullEventSink()


def _otlp() -> EventSink:
    return OtlpEventSink(
        "http://collector.test:4318",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
    )


#: 被测实现。`prepare` 给需要临时目录的 sink 用
IMPLS: dict[str, Callable[[Any], EventSink]] = {
    "jsonl": _jsonl,
    "null": lambda _tmp: _null(),
    "otlp": lambda _tmp: _otlp(),
    "otlp_via_registry": lambda _tmp: build_event_sink(
        "otlp", endpoint="http://collector.test:4318",
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
    ),
}


def _factories():
    return IMPLS


@pytest.fixture(params=sorted(_factories()), ids=sorted(_factories()))
def sink(request, tmp_path):
    instance = _factories()[request.param](tmp_path)
    yield instance
    instance.close()


def test_sink_satisfies_protocol(sink):
    """协议不符时 Fanout 只会把它当"失败的一个下游"，界面上一切正常。

    SSE broker 事故就是这么发生的：实现了 `publish()` 没实现 `emit()`，
    每个事件都失败，但异常被隔离成一行 warning。
    """
    assert isinstance(sink, EventSink), f"{type(sink).__name__} 不满足 EventSink 协议"
    assert isinstance(sink.name, str) and sink.name


def test_emit_flush_close_never_raise(sink):
    for event in _stream():
        sink.emit(event)
    sink.flush(1.0)
    sink.close()


def test_flush_without_events_is_a_no_op(sink):
    sink.flush(1.0)  # 空缓冲不该发请求、不该开文件、不该抛
    sink.flush(1.0)


@pytest.mark.parametrize("name", sorted(IMPLS))
def test_data_actually_leaves_the_sink(name, tmp_path):
    """"接上了"与"有流量"是两件事：每个实现都要证明自己收到了东西。"""
    instance = IMPLS[name](tmp_path)
    try:
        for event in _stream():
            instance.emit(event)
        instance.flush(1.0)
        if name == "jsonl":
            lines = (tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()
            assert len(lines) == 4, f"4 个事件应写成 4 行，实际 {len(lines)}"
            assert json.loads(lines[0])["type"] == "trace_start"
        elif name == "null":
            assert instance.count == 4  # type: ignore[attr-defined]
        else:
            stats = instance.stats  # type: ignore[attr-defined]
            assert stats["exported"] == 1, f"一条完整 trace 应导出一个根 span：{stats}"
            assert stats["orphans"] == 0, "事件顺序正确时不该有孤儿事件"
    finally:
        instance.close()


def test_close_flushes_what_is_pending(tmp_path):
    """只 emit 不 flush 就退出时，数据必须仍然出去（否则进程一停导出就静音）。"""
    posted: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(request.content)
        return httpx.Response(200, json={})

    sink = OtlpEventSink("http://collector.test:4318", transport=httpx.MockTransport(handler))
    for event in _stream():
        sink.emit(event)
    assert not posted, "emit 阶段不该发消息：主链路不为外部服务付网络延迟"
    sink.close()
    assert posted, "close 必须把攒着的 span 发出去"
    assert sink.exported >= 1


def test_one_broken_sink_cannot_take_down_the_healthy_one(tmp_path):
    """Fanout 的错误隔离 + **故障必须可见**（两条一起才算守住教训）。"""

    class Broken:
        name = "broken"

        def emit(self, event) -> None:
            raise RuntimeError("collector 挂了")

        def flush(self, timeout: float = 1.0) -> None: ...

        def close(self) -> None: ...

    healthy = _jsonl(tmp_path)
    fanout = EventFanout([Broken(), healthy])
    for event in _stream():
        fanout.emit(event)
    fanout.flush(1.0)

    assert len((tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines()) == 4
    assert fanout.errors.get("broken") == 4, (
        "坏 sink 必须被计数：只隔离不报告 = 把故障伪装成正常运行"
    )

    fanout.close()


def test_sink_failure_after_close_is_isolated(tmp_path):
    """close 之后 sink 允许失败，但失败必须被 Fanout 记账，且不许弄丢已写出的数据。

    otlp 的 `emit()` 只入队（主链路不为外部服务付网络延迟），所以故障在 `flush()`
    才现形——这正是"隔离点与故障点不在同一处"的形态，断言必须覆盖 flush。
    """
    healthy = _jsonl(tmp_path)
    for event in _stream()[:2]:
        healthy.emit(event)
    healthy.flush(1.0)
    written = len((tmp_path / "events.ndjson").read_text(encoding="utf-8").splitlines())

    closed = _otlp()
    closed.close()
    fanout = EventFanout([closed, healthy])
    for event in _stream():
        fanout.emit(event)
    fanout.flush(1.0)

    assert written == 2, "另一个 sink 的失败不许影响已落盘的数据"
    assert fanout.errors.get(closed.name), "关闭后仍被使用的 sink 要被计数，不是静默"
    fanout.close()


def test_external_sink_satisfies_same_contract(entry_points_env):
    """插件 sink 与内建 sink 走同一套断言——抽象不是给内置开的后门。"""
    entry_points_env.install("onyx.sinks", "mem=test_sink_contract:_memory_sink_builder")
    instance = build_event_sink("mem")
    assert isinstance(instance, EventSink)
    for event in _stream():
        instance.emit(event)
    instance.flush(1.0)
    assert len(instance.events) == 4  # type: ignore[attr-defined]
    instance.close()


class _MemorySink:
    name = "memory"

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []
        self.flushed = 0
        self.closed = 0

    def emit(self, event) -> None:
        self.events.append(event)

    def flush(self, timeout: float = 1.0) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


def _memory_sink_builder(**_options: Any) -> _MemorySink:
    return _MemorySink()
