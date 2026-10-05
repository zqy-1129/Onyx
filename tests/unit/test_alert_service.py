"""告警服务（S27）：轮询 → 判定 → 逐渠道投递 → 落库留痕。

测的是"通知下落"这件事能不能被回答：投了什么、谁失败了、被 cooldown 挡住的为什么没写行、
测试行会不会污染真实历史。渠道坏掉不许带崩后台线程，也不许拖累别的渠道。
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta

import pytest

from onyx.obs.alerts.channels import Alert, ChannelResult, FileChannel
from onyx.obs.alerts.rules import AlertRule
from onyx.obs.alerts.service import AlertService
from onyx.store.db import Database
from onyx.store.records import AlertTriggerRecord, AnomalyRecord
from onyx.store.repos import AlertRepo, TraceRepo

NOW = "2026-10-05T03:00:00+00:00"


def _iso(seconds_from_now: float) -> str:
    return (datetime.fromisoformat(NOW) + timedelta(seconds=seconds_from_now)).isoformat()


class _Failing:
    name = "webhook"

    def __init__(self) -> None:
        self.sent = 0

    def deliver(self, alert: Alert) -> ChannelResult:
        self.sent += 1
        return ChannelResult(False, "ConnectError: 连接被拒绝")


class _Raising:
    name = "boom"

    def deliver(self, alert: Alert) -> ChannelResult:
        raise RuntimeError("渠道自己炸了")


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "a.sqlite")
    path = tmp_path / "alerts" / "alerts.jsonl"
    svc = AlertService(db, rule=AlertRule(), channels=[FileChannel(path)])
    yield svc, db, TraceRepo(db), AlertRepo(db), path
    svc.stop()
    db.close()


def _anomaly(i: str, code: str = "CONTEXT_OVERFLOW", ago_s: float = -20,
             severity: str = "error") -> AnomalyRecord:
    return AnomalyRecord(id=i, code=code, severity=severity, trace_id=f"tr-{i}",
                         created_at=_iso(ago_s))


def _lines(path):
    """按天分片之后，"写了什么"要看那个目录里的所有 jsonl 而不是单个文件名。"""
    files = sorted(path.parent.glob(f"{path.stem}*.jsonl"))
    return [json.loads(line) for f in files for line in
            f.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_a_hit_delivers_and_leaves_one_row_per_channel(env):
    svc, _, traces, _alerts, path = env
    traces.insert_anomaly(_anomaly("a1"))
    rows = svc.tick(now=NOW)
    assert [r.status for r in rows] == ["sent"]
    assert rows[0].channel == "file" and rows[0].n_in_window == 1
    written = _lines(path)
    assert len(written) == 1 and written[0]["code"] == "CONTEXT_OVERFLOW"
    assert written[0]["n_is_lower_bound"] is False
    assert written[0]["trace_ids"] == ["tr-a1"], "通知要能指回现场"
    assert written[0]["rule"]["window_s"] == 300, "落库的是命中当时的判据快照"


def test_second_tick_inside_cooldown_writes_nothing(env):
    """抑制不写行：这张表记"发过什么"，不是"本可以发但规则挡了"。"""
    svc, _, traces, _alerts, path = env
    traces.insert_anomaly(_anomaly("a1"))
    assert len(svc.tick(now=NOW)) == 1
    assert svc.tick(now=_iso(5)) == [], "cooldown 内不该重复投"
    assert len(_lines(path)) == 1


def test_cooldown_expiry_delivers_again(env):
    """上次投递已经在 cooldown 之外 ⇒ 该再通知一次（一条一直坏的异常要能被反复提醒）。

    窗口比 cooldown 长才有这个场景：异常还在窗口里，但冷却期已经过了。
    """
    svc, _, traces, _, path = env
    svc.rule = AlertRule(cooldown_s=60)
    traces.insert_anomaly(_anomaly("a1"))
    assert len(svc.tick(now=NOW)) == 1
    assert svc.tick(now=_iso(30)) == [], "还在冷却里"
    assert len(svc.tick(now=_iso(61))) == 1, "冷却过了就该再发一次"
    assert len(_lines(path)) == 2


def test_a_failing_channel_still_records_and_does_not_block_others(tmp_path):
    db = Database(tmp_path / "b.sqlite")
    good = FileChannel(tmp_path / "alerts.jsonl")
    bad = _Failing()
    svc = AlertService(db, rule=AlertRule(), channels=[bad, good])
    try:
        TraceRepo(db).insert_anomaly(_anomaly("a1"))
        rows = svc.tick(now=NOW)
        assert {r.channel: r.status for r in rows} == {"webhook": "failed", "file": "sent"}
        assert any(r.detail.startswith("ConnectError") for r in rows)
        assert _lines(tmp_path / "alerts.jsonl"), "一个出口坏了不能拖累另一个"
    finally:
        db.close()


def test_a_channel_that_raises_is_recorded_as_failed(tmp_path):
    """渠道抛异常不能冒到轮询线程：那会让"没有通知"与"通知系统崩了"再也分不开。"""
    db = Database(tmp_path / "c.sqlite")
    svc = AlertService(db, rule=AlertRule(), channels=[_Raising()])
    try:
        TraceRepo(db).insert_anomaly(_anomaly("a1"))
        rows = svc.tick(now=NOW)
        assert rows[0].status == "failed" and "渠道自身抛错" in rows[0].detail
    finally:
        db.close()


def test_warn_level_anomalies_never_reach_a_channel(env):
    svc, _, traces, _, path = env
    traces.insert_anomaly(_anomaly("w1", code="TOKEN_DRIFT", severity="warn"))
    assert svc.tick(now=NOW) == []
    assert _lines(path) == []


def test_min_count_batches_the_window(env):
    """阈值是"几次才算事"：窗口内没到次数就不该发，而不是发一条"目前 1 次"。"""
    svc, _, traces, _, _path = env
    svc.rule = AlertRule(min_count=3, cooldown_s=0)
    traces.insert_anomaly(_anomaly("a1"))
    traces.insert_anomaly(_anomaly("a2", ago_s=-30))
    assert svc.tick(now=NOW) == []
    traces.insert_anomaly(_anomaly("a3", ago_s=-10))
    rows = svc.tick(now=NOW)
    assert len(rows) == 1 and rows[0].n_in_window == 3


def test_truncated_scan_is_marked_as_a_lower_bound(tmp_path):
    db = Database(tmp_path / "d.sqlite")
    svc = AlertService(db, rule=AlertRule(cooldown_s=0), scan_limit=2,
                       channels=[FileChannel(tmp_path / "alerts.jsonl")])
    try:
        traces = TraceRepo(db)
        for i in range(5):
            traces.insert_anomaly(_anomaly(f"a{i}", ago_s=-(10 + i)))
        rows = svc.tick(now=NOW)
        assert len(rows) == 1 and rows[0].n_in_window == 2, "limit 是真的上限"
        assert _lines(tmp_path / "alerts.jsonl")[0]["n_is_lower_bound"] is True
    finally:
        db.close()


def test_undated_rows_do_not_break_the_tick(env, caplog):
    svc, _, traces, _, _ = env
    traces.insert_anomaly(AnomalyRecord(id="x", code="CONTEXT_OVERFLOW", severity="error",
                                        created_at="上周三"))
    traces.insert_anomaly(_anomaly("a1"))
    rows = svc.tick(now=NOW)
    assert [r.n_in_window for r in rows] == [1], "坏时间戳的行不参与计数"


def test_send_test_marks_the_row_and_leaves_cooldown_alone(env):
    """`alerts test` 造的行不许污染"上次真的通知是什么时候"。"""
    svc, _, _, alerts, path = env
    rows = svc.send_test()
    assert len(rows) == 1 and rows[0].is_test is True
    assert _lines(path)[0]["is_test"] is True
    assert "测试" in _lines(path)[0]["message"]
    assert alerts.last_real_trigger_at("CONTEXT_OVERFLOW") is None

    TraceRepo(svc.traces.db).insert_anomaly(_anomaly("a1"))
    assert len(svc.tick(now=NOW)) == 1, "测试行不构成 cooldown，真实命中照样要发"


def test_send_test_can_target_one_channel(tmp_path):
    db = Database(tmp_path / "e.sqlite")
    bad = _Failing()
    svc = AlertService(db, rule=AlertRule(), channels=[bad, FileChannel(tmp_path / "f.jsonl")])
    try:
        rows = svc.send_test(channel_filter="file")
        assert [r.channel for r in rows] == ["file"]
        assert bad.sent == 0, "点名 file 就不该去打 webhook 的门"
    finally:
        db.close()


def test_status_reports_what_a_page_would_need(tmp_path):
    db = Database(tmp_path / "g.sqlite")
    svc = AlertService(db, rule=AlertRule(enabled=False), channels=[])
    try:
        assert svc.status() == {
            "enabled": False, "channels": [], "poll_s": svc.poll_s,
            "ticks": 0, "last_error": "", "thread_alive": False,
        }
    finally:
        db.close()


class _BrokenTick(AlertService):
    """让轮询本体炸，而不是让渠道炸：渠道的失败在 tick 内部就被收成 failed 行了。"""

    def __init__(self, *a, **kw) -> None:
        super().__init__(*a, **kw)
        self.booms = 0

    def tick(self, *, now=None):
        self.booms += 1
        raise RuntimeError("轮询自己炸了")


def test_thread_survives_its_own_failure_and_keeps_polling(tmp_path):
    """后台线程崩过一次就要被看见（last_error），但仍然继续下一轮——
    否则"没有通知"与"通知系统悄悄停摆"再也分不开。"""
    db = Database(tmp_path / "h.sqlite")
    svc = _BrokenTick(db, rule=AlertRule(), poll_s=0.01,
                      channels=[FileChannel(tmp_path / "alerts.jsonl")])
    svc.start()
    assert svc.status()["thread_alive"] is True
    deadline = time.monotonic() + 5.0
    while svc.booms < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert svc.last_error.startswith("RuntimeError"), "线程内的失败必须留下名字"
    assert svc.booms >= 2, "一轮出错之后还要继续轮询"
    svc.stop(timeout=2.0)
    assert svc.status()["thread_alive"] is False
    db.close()


def test_thread_starts_and_stops(tmp_path):
    db = Database(tmp_path / "h2.sqlite")
    svc = AlertService(db, rule=AlertRule(), poll_s=0.01,
                       channels=[FileChannel(tmp_path / "alerts.jsonl")])
    TraceRepo(db).insert_anomaly(_anomaly("a1", ago_s=-1))
    svc.start()
    deadline = time.monotonic() + 5.0
    while svc.ticks == 0 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert svc.ticks > 0
    svc.stop(timeout=2.0)
    assert svc.status()["thread_alive"] is False
    db.close()


def test_start_is_a_noop_without_channels_or_when_disabled(tmp_path):
    db = Database(tmp_path / "i.sqlite")
    no_channel = AlertService(db, rule=AlertRule(), channels=[])
    off = AlertService(db, rule=AlertRule(enabled=False), channels=[FileChannel("x.jsonl")])
    try:
        no_channel.start()
        off.start()
        assert no_channel.status()["thread_alive"] is False
        assert off.status()["thread_alive"] is False
    finally:
        db.close()


def test_trigger_row_round_trips_the_rule_snapshot(env):
    svc, _db, traces, alerts, _ = env
    svc.rule = AlertRule(codes=("CONTEXT_OVERFLOW",), source="file")
    traces.insert_anomaly(_anomaly("a1"))
    svc.tick(now=NOW)
    got = alerts.list_triggers()[0]
    assert isinstance(got, AlertTriggerRecord)
    assert got.rule["codes"] == ["CONTEXT_OVERFLOW"] and got.rule["source"] == "file"


# ── 出口自己坏掉的那几条路 ────────────────────────────────────────
def test_file_channel_reports_a_write_failure_instead_of_raising(tmp_path, monkeypatch):
    """磁盘满/目录被占是本地出口唯一的失败模式，它必须变成一个 failed 结果而不是一场崩。"""
    target = tmp_path / "alerts" / "alerts.jsonl"

    def _boom(self, *a, **kw):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("pathlib.Path.open", _boom)
    result = FileChannel(target).deliver(Alert(
        code="TOOL_LOOP", severity="error", message="m", n=1, window_s=60,
        triggered_at=NOW, rule={},
    ))
    assert result.ok is False and "OSError" in result.detail


def test_file_channel_falls_back_to_today_when_the_timestamp_is_broken(tmp_path):
    """一条坏时间戳不该让通知写不出去；落到"今天"那个文件，消息本身仍然完整。"""
    target = tmp_path / "alerts" / "alerts.jsonl"
    result = FileChannel(target).deliver(Alert(
        code="TOOL_LOOP", severity="error", message="m", n=1, window_s=60,
        triggered_at="上周三", rule={},
    ))
    assert result.ok is True
    written = list((tmp_path / "alerts").glob("alerts-*.jsonl"))
    assert len(written) == 1 and written[0].name != target.name


def test_build_channels_installs_nothing_when_disabled(tmp_path):
    """`enabled = false` 就是真的不装出口，而不是"装了但什么都不发"。"""
    from dataclasses import replace

    from onyx.config import load_config
    from onyx.obs.alerts.service import build_channels

    cfg_path = tmp_path / "onyx.toml"
    cfg_path.write_text("[alerts]\nfile = 'alerts/a.jsonl'\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert [c.name for c in build_channels(cfg, tmp_path)] == ["file"]

    off = replace(cfg, alerts=replace(cfg.alerts, enabled=False))
    assert build_channels(off, tmp_path) == [], "关掉之后连文件都不该写"
    assert not (tmp_path / "alerts").exists()


def test_tick_with_an_unreadable_now_does_not_crash(env, caplog):
    """now 解析不出来时窗口边界退回"现在"，判定照常跑——坏的是那行时间，不是整个通知链。"""
    svc, _, traces, _, _ = env
    traces.insert_anomaly(_anomaly("a1"))
    rows = svc.tick(now="不是时间戳")
    assert isinstance(rows, list)
