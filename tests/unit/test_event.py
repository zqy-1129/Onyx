from __future__ import annotations

import pytest

from onyx.core.clock import FakeClock
from onyx.core.errors import UnknownEventType
from onyx.core.event import (
    CONTRACT_VERSION,
    PAYLOAD_OPTIONAL,
    PAYLOAD_REQUIRED,
    EventType,
    TraceEvent,
    declared_event_types,
    make_event,
)


def test_every_event_type_is_documented():
    """表驱动：新增事件类型必须同时声明 payload 契约，否则 visitor 作者无从下手。"""
    declared = {e for e in EventType if e is not EventType.UNKNOWN}
    assert declared == declared_event_types() - {EventType.UNKNOWN}
    assert declared == set(PAYLOAD_REQUIRED)
    assert set(PAYLOAD_OPTIONAL) - {EventType.UNKNOWN} <= declared


def test_required_keys_enforced():
    with pytest.raises(UnknownEventType) as ei:
        make_event(EventType.TRACE_END, "T1", {"status": "ok"})
    assert "wall_ms" in str(ei.value)
    assert ei.value.detail["missing"] == ["wall_ms"]


def test_unknown_type_strict_vs_lenient():
    with pytest.raises(UnknownEventType):
        make_event("totally_new_event", "T1", strict=True)
    ev = make_event("totally_new_event", "T1", {"x": 1}, strict=False)
    assert ev.type is EventType.UNKNOWN


def test_extra_payload_keys_allowed_for_forward_compat():
    """引擎将来多报一个字段时，事件层不许炸（原则 6）。"""
    ev = make_event(EventType.USAGE_ENGINE, "T1", {"in_tokens": 10, "brand_new_field": 42})
    assert ev.get("brand_new_field") == 42


def test_roundtrip_dict():
    clock = FakeClock(wall="2026-01-01T00:00:00.000000+00:00")
    ev = make_event(EventType.FIRST_TOKEN, "T1", {"ttft_ms": 142.5}, clock=clock)
    back = TraceEvent.from_dict(ev.to_dict())
    assert back == ev
    assert back.version == CONTRACT_VERSION


def test_from_dict_survives_unknown_type():
    raw = {"v": 1, "type": "future_event", "trace_id": "T1", "ts_ns": 5, "wall_iso": "x", "payload": {"a": 1}}
    ev = TraceEvent.from_dict(raw)
    assert ev.type is EventType.UNKNOWN
    assert ev.payload["_raw_type"] == "future_event", "未知事件的原始类型必须保留"
    assert ev.payload["a"] == 1


def test_json_serialisable():
    ev = make_event(EventType.ANOMALY, "T1", {"code": "TOKEN_DRIFT", "severity": "warn", "detail": {"d": 1}})
    assert '"TOKEN_DRIFT"' in ev.to_json()


def test_timestamps_come_from_injected_clock():
    clock = FakeClock(wall="2030-05-05T05:05:05.000000+00:00", mono_ns=1000, step_ns=7)
    ev = make_event(EventType.TEXT_DELTA, "T1", {"seq": 0, "text": "a"}, clock=clock)
    assert ev.wall_iso.startswith("2030-05-05")
    assert ev.ts_ns == 1007
