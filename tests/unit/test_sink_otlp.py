"""OTLP 导出 sink 的映射细节。

契约测试（`tests/contract/test_sink_contract.py`）管"能不能被换进来"，
这里管"导出去的东西是不是真的对"：OTLP 的 JSON 编码有几处**看起来能发、
实际会被 collector 丢掉**的坑，都要有断言。

全部用 `httpx.MockTransport`，零真实网络。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from onyx.core.event import EventType, make_event
from onyx.store.sinks import EventFanout
from onyx.store.sinks.otlp import OtlpEventSink, _attr, _epoch_ns, _sha_hex

TID = "01M4OTLPTRACE00000000000001"
WALL = "2026-10-04T12:00:00.500000+00:00"


def _value(any_value: dict):
    """把 OTLP 的 AnyValue 解回 Python 值。

    这个解包本身也是断言：形状不认识就直接失败，
    而不是让测试"用某种特定写法通过"、collector 那边却看不懂。
    """
    if "stringValue" in any_value:
        return any_value["stringValue"]
    if "intValue" in any_value:
        return int(any_value["intValue"])
    if "doubleValue" in any_value:
        return any_value["doubleValue"]
    if "boolValue" in any_value:
        return any_value["boolValue"]
    if "arrayValue" in any_value:
        return [_value(item) for item in any_value["arrayValue"].get("values", [])]
    raise AssertionError(f"未知的 AnyValue 形状: {any_value}")


def _ok_sink(**kw) -> tuple[OtlpEventSink, list[dict]]:
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={})

    sink = OtlpEventSink(kw.pop("endpoint", "http://collector.test:4318"),
                         transport=httpx.MockTransport(handler), **kw)
    sink._posted = posted  # type: ignore[attr-defined]
    return sink, posted


def _one_trace(sink: OtlpEventSink, *, tool: bool = False, status: str = "ok") -> None:
    sink.emit(make_event(EventType.TRACE_START, TID,
                         {"kind": "chat", "purpose": "chat", "provider_id": "p1",
                          "model": "qwen3.5:9b"}))
    sink.emit(make_event(EventType.TEXT_DELTA, TID, {"seq": 0, "text": "hi"}))
    sink.emit(make_event(EventType.TEXT_DELTA, TID, {"seq": 1, "text": " there"}))
    sink.emit(make_event(EventType.RECONCILED, TID,
                         {"chosen_source": "engine", "confidence": "high"}))
    if tool:
        sink.emit(make_event(EventType.TOOL_EXEC_START, TID,
                             {"name": "get_weather", "step": 1, "args": {"city": "北京"}}))
        sink.emit(make_event(EventType.TOOL_EXEC_END, TID,
                             {"name": "get_weather", "step": 1, "status": "ok",
                              "latency_ms": 12.0}))
    sink.emit(make_event(EventType.TRACE_END, TID, {"status": status, "wall_ms": 25.0}))


def _spans(payload: dict) -> list[dict]:
    return payload["resourceSpans"][0]["scopeSpans"][0]["spans"]


# ── 构造 ──────────────────────────────────────────────────────────
def test_missing_endpoint_is_a_hard_error_not_silence(monkeypatch):
    """`--sink otlp` 而没配端点：必须报错，而不是"接上了但其实什么都没发"。"""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with pytest.raises(ValueError, match="OTEL_EXPORTER_OTLP_ENDPOINT"):
        OtlpEventSink()


def test_endpoint_comes_from_the_standard_env(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://col:4318")
    sink = OtlpEventSink(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    assert sink.endpoint == "http://col:4318/v1/traces"
    sink.close()


def test_headers_parsed_from_env_comma_list(monkeypatch):
    """OTLP 规范里 `OTEL_EXPORTER_OTLP_HEADERS` 是 `k=v,k2=v2`。"""
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer x, x-scope=a")
    sink, _ = _ok_sink()
    assert sink._headers["authorization"] == "Bearer x"
    assert sink._headers["x-scope"] == "a"
    sink.close()


def test_content_type_is_json_and_declared_in_payload():
    """我们导出的是 OTLP/**JSON** 编码。"""
    sink, posted = _ok_sink()
    _one_trace(sink)
    sink.flush()
    assert "children" not in _spans(posted[0])[0], "children 是我们内部的中间形状，不能外发"
    resource_attrs = {a["key"]: a["value"] for a in posted[0]["resourceSpans"][0]["resource"]["attributes"]}
    assert resource_attrs["service.name"]["stringValue"] == "onyx"
    assert resource_attrs["onyx.encoding"]["stringValue"] == "json", \
        "接收端必须能看出自己面对的是 JSON 而不是 protobuf"
    sink.close()


# ── id 派生 ───────────────────────────────────────────────────────
def test_ids_have_otlp_widths_and_are_deterministic():
    sink, posted = _ok_sink()
    _one_trace(sink)
    sink.flush()
    span = _spans(posted[0])[0]
    assert len(span["traceId"]) == 32, "OTLP traceId 必须是 16 字节 hex"
    assert len(span["spanId"]) == 16, "spanId 必须是 8 字节 hex"
    attrs = {a["key"]: a["value"] for a in span["attributes"]}
    assert _value(attrs["onyx.trace_id"]) == TID, "原始 id 必须留在属性里，否则回不到 onyx"
    assert "intValue" in attrs["onyx.contract_version"], "int 值要按 intValue 导出"

    second, posted2 = _ok_sink()
    _one_trace(second)
    second.flush()
    assert _spans(posted2[0])[0]["traceId"] == span["traceId"], "同一 trace 必须同一 traceId"
    assert _sha_hex(TID, 16) != _sha_hex(TID, 8), "traceId 与 spanId 宽度不同，不能撞车"
    sink.close()
    second.close()


def test_unparsable_wall_time_does_not_become_1970_silently():
    """单调钟不能拿去导出；解析失败时宁可留 0 并让原始事实仍可从 onyx 查。"""
    assert _epoch_ns("not-a-date") == 0
    ns = _epoch_ns(WALL)
    assert 1_700_000_000 * 10**9 < ns < 2_000_000_000 * 10**9, "必须是 2026 年附近的 epoch ns"
    # 半开区间检查：解析出来的时间与原串一致
    assert datetime.fromtimestamp(ns / 10**9, UTC).isoformat() == WALL


# ── 属性编码 ──────────────────────────────────────────────────────
def test_int64_attributes_are_strings_in_json_encoding():
    """OTLP JSON 规定 int64 用字符串表示。发数字会被多数 collector 拒收。"""
    assert _attr("n", 12) == {"key": "n", "value": {"intValue": "12"}}
    assert _attr("f", 1.5)["value"]["doubleValue"] == 1.5
    assert _attr("b", True)["value"]["boolValue"] is True
    assert _attr("s", "x")["value"]["stringValue"] == "x"
    assert _attr("none", None) is None, "没有的值不导出，而不是导成空字符串"
    arr = _attr("list", ["a", "b"])
    assert [v["stringValue"] for v in arr["value"]["arrayValue"]["values"]] == ["a", "b"]


def test_reconciled_and_folded_counters_are_exported():
    sink, posted = _ok_sink()
    _one_trace(sink)
    sink.flush()
    span = _spans(posted[0])[0]
    attrs = {a["key"]: _value(a["value"]) for a in span["attributes"]}
    assert attrs["onyx.usage.source"] == "engine"
    assert attrs["onyx.usage.confidence"] == "high"
    # 折叠计数是个 map，而 OTLP 没有 map 值类型 ⇒ 按 JSON 字符串导出。
    # 若导出 `str(dict)`（Python repr，单引号），collector 侧 loads 会失败，
    # 而失败发生在对面——我们什么都看不见，只以为"导出成功了"
    events = json.loads(attrs["onyx.events"])
    assert events["text_delta"] == 2, f"两次增量应折成计数 2：{events}"
    assert attrs["onyx.wall_ms"] == 25.0
    sink.close()


# ── 工具子 span ───────────────────────────────────────────────────
def test_tool_execution_becomes_a_child_span_without_arg_values():
    sink, posted = _ok_sink()
    _one_trace(sink, tool=True)
    sink.flush()
    spans = _spans(posted[0])
    root = next(s for s in spans if not s.get("parentSpanId"))
    tool = next(s for s in spans if s["name"] == "tool get_weather")
    assert tool["parentSpanId"] == root["spanId"]
    assert tool["status"]["code"] == 1, "ok ⇒ STATUS_CODE_OK"
    attrs = {a["key"]: _value(a["value"]) for a in tool["attributes"]}
    assert "onyx.tool.args" not in attrs, "默认不外发参数值（里面常有地址、身份、内部 ID）"
    assert attrs["onyx.tool.arg_keys"] == ["city"]
    assert attrs["onyx.tool.arg_count"] == 1, "int64 在 JSON 编码里是字符串，解回来必须是 1"
    sink.close()


def test_include_args_is_opt_in():
    sink, posted = _ok_sink(include_args=True)
    _one_trace(sink, tool=True)
    sink.flush()
    tool = next(s for s in _spans(posted[0]) if s["name"] == "tool get_weather")
    attrs = {a["key"]: _value(a["value"]) for a in tool["attributes"]}
    assert json.loads(attrs["onyx.tool.args"]) == {"city": "北京"}
    sink.close()


def test_unpaired_tool_end_still_exported():
    """只等到 END 也要导出：漏一个 span 会被读成"这次没执行工具"。"""
    sink, posted = _ok_sink()
    sink.emit(make_event(EventType.TRACE_START, TID,
                         {"kind": "chat", "purpose": "chat", "provider_id": "p", "model": "m"}))
    sink.emit(make_event(EventType.TOOL_EXEC_END, TID,
                         {"name": "orphan_tool", "step": 3, "status": "timeout"}))
    sink.emit(make_event(EventType.TRACE_END, TID, {"status": "ok", "wall_ms": 1.0}))
    sink.flush()
    tool = next(s for s in _spans(posted[0]) if s["name"] == "tool orphan_tool")
    assert tool["status"]["code"] == 2, "非 ok 状态必须映射成 STATUS_CODE_ERROR"
    assert tool["startTimeUnixNano"] == tool["endTimeUnixNano"] or int(
        tool["endTimeUnixNano"]) >= int(tool["startTimeUnixNano"])
    sink.close()


def test_error_trace_maps_to_error_status():
    sink, posted = _ok_sink()
    _one_trace(sink, status="error")
    sink.flush()
    root = next(s for s in _spans(posted[0]) if not s.get("parentSpanId"))
    assert root["status"]["code"] == 2
    sink.close()


# ── 批处理与故障 ──────────────────────────────────────────────────
def test_emit_does_not_touch_the_network():
    sink, posted = _ok_sink()
    for _ in range(5):
        _one_trace(sink)
    assert posted == []
    sink.flush()
    assert len(posted) == 1
    sink.close()


def test_batching_splits_into_multiple_posts():
    sink, posted = _ok_sink(batch_spans=1)
    for _ in range(3):
        _one_trace(sink)
    sink.flush()
    # flush 的语义是"把现在完成的都发出去"，所以是 3 次 POST（每次 1 个 span）而不是一次
    assert len(posted) == 3, f"每批 1 个 span，3 条 trace 应发 3 次：{len(posted)}"
    assert sink.stats["pending"] == 0
    assert sink.exported == 3
    sink.flush()
    assert len(posted) == 3, "空队列不该再发一次"
    sink.close()


def test_collector_rejection_raises_and_returns_the_batch():
    """失败必须冒到 Fanout（可见），同时把 span **原样退回队列**。

    退回的是数据本身，不是"记一笔欠账"：编码是纯函数，重试不需要伪造数据；
    只记账的重试第二次就没东西可发了，那等于把故障伪装成"已经尽力"。
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="queue full")

    sink = OtlpEventSink("http://collector.test", transport=httpx.MockTransport(handler))
    _one_trace(sink)
    with pytest.raises(RuntimeError, match="OTLP 导出失败"):
        sink.flush()
    assert sink.failures == 1
    assert sink.exported == 0, "发失败了一律不算已导出"
    assert sink.stats["pending"] == 1, "span 必须原样退回队列"
    assert sink.requeued == 1

    fanout = EventFanout([sink])
    _one_trace(sink)
    fanout.flush()  # Fanout 吞掉异常并计数——隔离不许掩盖故障
    assert sink.failures == 2, "sink 自己记的是累计失败次数"
    assert fanout.errors.get("otlp") == 1, "Fanout 每次 flush 记一次，不是每个事件记一次"

    # 端点恢复后，退回的数据仍然发得出去（这才证明"退回"不是记账）
    pending = len(sink._ready)
    assert pending == 2, "两次失败的 span 都该还压在队列里"
    ok = OtlpEventSink("http://collector.test", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})))
    ok._ready = list(sink._ready)
    ok.flush()
    assert ok.exported == pending
    ok.close()
    sink.close()


def test_queue_overflow_drops_oldest_but_counts_them():
    """观测系统宁可丢样本也不拖垮主链路，但**丢多少必须可见**。

    上限在入队时就生效（不是等 flush）：否则一次长评测会把一整夜的 span 堆在内存里，
    现象是"看板进程慢慢涨死"，而这恰恰是它要防的事。
    """
    sink, posted = _ok_sink(batch_spans=10, queue_cap=2)
    for _ in range(6):
        _one_trace(sink)
    assert sink.dropped == 4, f"6 条 trace、上限 2 ⇒ 应丢最老的 4 条：{sink.dropped}"
    assert sink.stats["pending"] == 2
    sink.flush()
    assert len(posted) == 1 and len(_spans(posted[0])) == 2
    sink.close()


def test_close_never_raises_even_when_the_collector_is_down():
    """`close()` 站在 `finally` 里。它抛出会盖掉真正让进程出错的那条异常。

    数据没丢（SQLite 才是权威存储），欠着的 span 仍然能在 `stats` 里看到——
    "关不掉"从来不该变成"看不见问题"。
    """
    sink = OtlpEventSink("http://collector.test", transport=httpx.MockTransport(
        lambda r: httpx.Response(503, text="down")))
    _one_trace(sink)
    sink.close()  # 不抛
    assert sink.failures == 1
    assert sink.stats["pending"] == 1


def test_engine_counts_land_on_the_root_span():
    """服务器报的计数要能被下游看到；没报的就**不导**，不导 0、也不导"采信来源"。

    `RECONCILED` 在事件契约里存在但 gateway 目前不发（采信是 observer 内部算完落库的），
    所以导出里只有各来源的原始报告——这比编一个"采信=engine"诚实。
    """
    sink, posted = _ok_sink()
    sink.emit(make_event(EventType.TRACE_START, TID,
                         {"kind": "chat", "purpose": "chat", "provider_id": "p", "model": "m"}))
    sink.emit(make_event(EventType.USAGE_ENGINE, TID,
                         {"in_tokens": 22, "out_tokens": 7, "thinking_tokens": None,
                          "cached_tokens": 4, "ok": True, "note": ""}))
    sink.emit(make_event(EventType.TRACE_END, TID, {"status": "ok", "wall_ms": 1.0}))
    sink.flush()
    span = _spans(posted[0])[0]
    attrs = {a["key"]: _value(a["value"]) for a in span["attributes"]}
    assert attrs["onyx.usage.engine.in_tokens"] == 22
    assert attrs["onyx.usage.engine.out_tokens"] == 7
    assert attrs["onyx.usage.engine.cached_tokens"] == 4
    assert "onyx.usage.engine.thinking_tokens" not in attrs, "没报的值不导出，而不是导成 0/null"
    assert "onyx.usage.source" not in attrs
    sink.close()


def test_events_without_a_trace_are_counted_not_dropped_silently():
    """孤儿事件增长说明装配或顺序出了问题；没有计数就只能靠人猜。"""
    sink, posted = _ok_sink()
    sink.emit(make_event(EventType.TEXT_DELTA, "01M4NOTRACE00000000000000",
                         {"seq": 0, "text": "x"}))
    sink.flush()
    assert sink.orphans == 1
    assert posted == []
    assert sink.stats["orphans"] == 1
    sink.close()
