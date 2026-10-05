"""S27 存储层：`alert_trigger` 的读写与异常窗口查询。

这张表只回答"某天某条规则到底通知没通知"，所以测试全部围绕**投递结果的可查性**：
测试行不许混进真实历史、失败要留下一行、cooldown 的判据（上次真的投过）必须来自库而不是进程。
"""

from __future__ import annotations

import pytest

from onyx.store.db import Database
from onyx.store.records import AlertTriggerRecord, AnomalyRecord
from onyx.store.repos import AlertRepo, TraceRepo


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "t.sqlite") as database:
        yield database


@pytest.fixture
def alerts(db):
    return AlertRepo(db)


def _trigger(i: str, *, code: str = "CONTEXT_OVERFLOW", channel: str = "file",
             status: str = "sent", created_at: str = "2026-10-05T03:00:00+00:00",
             is_test: bool = False, **kw) -> AlertTriggerRecord:
    base = dict(
        id=i, created_at=created_at, code=code, severity="error",
        rule={"min_count": 2, "window_s": 300, "source": "file"},
        n_in_window=3, window_s=300, first_anomaly_id="a1", last_anomaly_id="a3",
        trace_ids=("t1", "t2"), channel=channel, status=status,
        detail="写入 1 行", is_test=is_test,
    )
    base.update(kw)
    return AlertTriggerRecord(**base)


def test_migration_creates_the_trigger_table(db):
    assert "alert_trigger" in db.table_names()
    assert db.version() == 7


def test_round_trip_keeps_rule_and_sample_traces(alerts):
    """规则快照与样本 trace 必须原样回来：事后解释"为什么触发"靠的就是这两样。"""
    alerts.insert_trigger(_trigger("a"))
    rows = alerts.list_triggers()
    assert len(rows) == 1
    got = rows[0]
    assert got.rule == {"min_count": 2, "window_s": 300, "source": "file"}
    assert got.trace_ids == ("t1", "t2")
    assert got.is_test is False and got.n_in_window == 3


def test_filters_are_independent(alerts):
    for rec in (
        _trigger("w1", channel="webhook", status="failed", detail="超时"),
        _trigger("f1", channel="file"),
        _trigger("t1", code="PROVIDER_ERROR"),
    ):
        alerts.insert_trigger(rec)
    assert {r.id for r in alerts.list_triggers(channel="webhook")} == {"w1"}
    assert {r.id for r in alerts.list_triggers(status="failed")} == {"w1"}
    assert {r.id for r in alerts.list_triggers(code="PROVIDER_ERROR")} == {"t1"}
    assert len(alerts.list_triggers()) == 3


def test_since_bound_is_inclusive(alerts):
    alerts.insert_trigger(_trigger("old", created_at="2026-10-04T23:00:00+00:00"))
    alerts.insert_trigger(_trigger("new", created_at="2026-10-05T00:00:00+00:00"))
    assert {r.id for r in alerts.list_triggers(since="2026-10-05T00:00:00+00:00")} == {"new"}


def test_a_failed_delivery_still_has_a_row(alerts):
    """渠道失败恰恰是最需要被查到的一行——没有它，"我没收到"就永远只是人的一句抱怨。"""
    alerts.insert_trigger(_trigger("w", channel="webhook", status="failed",
                                   detail="ConnectError: all attempts failed"))
    got = alerts.list_triggers()[0]
    assert (got.status, got.channel) == ("failed", "webhook")
    assert "ConnectError" in got.detail
    assert alerts.counts_by_status() == {"failed:webhook": 1}


def test_test_rows_stay_out_of_the_real_history(alerts):
    """`alerts test` 造的行不能污染"上次真的通知是什么时候"。"""
    alerts.insert_trigger(_trigger("probe", channel="test", is_test=True,
                                   created_at="2026-10-05T05:00:00+00:00"))
    assert alerts.list_triggers(include_test=False) == []
    assert alerts.last_real_trigger_at("CONTEXT_OVERFLOW") is None
    assert alerts.counts_by_status() == {}

    alerts.insert_trigger(_trigger("real", created_at="2026-10-05T01:00:00+00:00"))
    alerts.insert_trigger(_trigger("probe2", channel="test", is_test=True,
                                   created_at="2026-10-05T06:00:00+00:00"))
    # 真实那行时间更早，但它是唯一"真的投过"的记录
    assert alerts.last_real_trigger_at("CONTEXT_OVERFLOW") == "2026-10-05T01:00:00+00:00"


def test_last_real_trigger_reads_per_code(alerts):
    alerts.insert_trigger(_trigger("a", code="CONTEXT_OVERFLOW",
                                   created_at="2026-10-05T01:00:00+00:00"))
    alerts.insert_trigger(_trigger("b", code="ORPHAN_TOOL_CALL",
                                   created_at="2026-10-05T02:00:00+00:00"))
    assert alerts.last_real_trigger_at("TOOL_LOOP") is None
    assert alerts.last_real_trigger_at("ORPHAN_TOOL_CALL") == "2026-10-05T02:00:00+00:00"


def test_reinserting_the_same_id_updates_the_delivery_result(alerts):
    """同一次命中的重试走同一行：历史里不该出现两条"同一个命中"。"""
    alerts.insert_trigger(_trigger("x", status="failed", detail="第 1 次超时"))
    alerts.insert_trigger(_trigger("x", status="sent", detail="第 2 次成功"))
    rows = alerts.list_triggers()
    assert len(rows) == 1 and rows[0].status == "sent"


def test_ids_sort_newest_first(alerts):
    for i, ident in enumerate(("01ABC", "01ABD", "01ABB")):
        alerts.insert_trigger(_trigger(ident, created_at=f"2026-10-05T0{i}:00:00+00:00"))
    assert [r.id for r in alerts.list_triggers(limit=2)] == ["01ABD", "01ABC"]


# ── 异常窗口查询（告警判定的读侧）──────────────────────────────────
def _anomaly(i: str, code: str, at: str, trace_id: str | None = None) -> AnomalyRecord:
    return AnomalyRecord(id=i, code=code, severity="error", trace_id=trace_id or f"tr-{i}",
                         detail={"n": 1}, created_at=at)


@pytest.fixture
def traces(db):
    return TraceRepo(db)


def test_window_scan_filters_by_since_and_codes(traces):
    traces.insert_anomaly(_anomaly("a1", "CONTEXT_OVERFLOW", "2026-10-05T02:00:00+00:00"))
    traces.insert_anomaly(_anomaly("a2", "TOOL_LOOP", "2026-10-05T02:30:00+00:00"))
    traces.insert_anomaly(_anomaly("a3", "CONTEXT_OVERFLOW", "2026-10-05T03:10:00+00:00"))

    got = traces.anomalies_in_window(since="2026-10-05T02:00:00+00:00",
                                    codes=["CONTEXT_OVERFLOW"])
    assert [r.id for r in got] == ["a1", "a3"], "窗口是闭区间，且不该把别的码混进来"

    everything = traces.anomalies_in_window(since="2026-10-05T02:00:00+00:00")
    assert [r.id for r in everything] == ["a1", "a2", "a3"], "按时间升序（判定要按发生顺序数）"


def test_window_scan_limit_is_a_real_cap(traces):
    for i in range(5):
        traces.insert_anomaly(_anomaly(f"a{i}", "PROVIDER_ERROR", f"2026-10-05T0{i}:00:00+00:00"))
    got = traces.anomalies_in_window(since="2026-10-05T00:00:00+00:00", limit=3)
    assert len(got) == 3, "limit 是真的上限；调用方据此把计数标成下限"


def test_window_scan_carries_the_severity_and_trace(traces):
    traces.insert_anomaly(_anomaly("a1", "ORPHAN_TOOL_CALL", "2026-10-05T02:00:00+00:00",
                                   trace_id="tr-special"))
    got = traces.anomalies_in_window(since="2026-10-05T00:00:00+00:00")
    assert got[0].severity == "error" and got[0].trace_id == "tr-special"
    assert got[0].detail == {"n": 1}
