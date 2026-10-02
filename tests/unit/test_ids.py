from __future__ import annotations

import pytest

from onyx.core.ids import id_timestamp_ms, is_valid_id, new_trace_id, reset_state


@pytest.fixture(autouse=True)
def _clean_state():
    reset_state()
    yield
    reset_state()


def test_id_shape_and_alphabet():
    tid = new_trace_id()
    assert len(tid) == 26
    assert is_valid_id(tid)
    assert not is_valid_id("short")
    assert not is_valid_id("01J" + "!" * 23)


def test_ids_strictly_increasing_and_unique():
    ids = [new_trace_id() for _ in range(1000)]
    assert ids == sorted(ids), "id 必须按字典序即时间序，游标分页依赖这一点"
    assert len(set(ids)) == 1000


def test_same_millisecond_still_orders():
    """同一毫秒内连续生成也必须严格递增（靠内部计数器）。"""
    ids = [new_trace_id(now_ms=1_700_000_000_000) for _ in range(500)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 500


def test_clock_going_backwards_does_not_break_order():
    a = new_trace_id(now_ms=1_700_000_000_000)
    b = new_trace_id(now_ms=1_600_000_000_000)  # 时钟回拨
    assert b > a, "时钟回拨时必须钳制到上次时间，保证单调"


def test_timestamp_roundtrip():
    ms = 1_700_000_123_456
    assert id_timestamp_ms(new_trace_id(now_ms=ms)) == ms
