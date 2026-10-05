"""告警判定（S27）：窗口、阈值、cooldown 三条规则全部在纯函数里，所以能穷举。

这里刻意不测"真的等了 5 秒"，也不测发消息——那是 service 与 channels 的事。
"""

from __future__ import annotations

import pytest

from onyx.obs.alerts.channels import Alert, FileChannel
from onyx.obs.alerts.rules import AlertRule, Row, evaluate

NOW = "2026-10-05T03:00:00+00:00"


def _row(i: str, code: str = "CONTEXT_OVERFLOW", severity: str = "error",
         ago_s: int = 10, trace_id: str | None = None) -> Row:
    from datetime import datetime, timedelta

    created = (datetime.fromisoformat(NOW) - timedelta(seconds=ago_s)).isoformat()
    return Row(id=i, code=code, severity=severity, created_at=created,
               trace_id=trace_id or f"tr-{i}")


def _eval(rows, rule: AlertRule | None = None, **kw):
    return evaluate(rows, rule=rule or AlertRule(), now=NOW,
                    last_triggered_at=kw.pop("last", {}), **kw)


# ── 筛选：级别与码是两条独立的闸 ──────────────────────────────────
def test_default_rule_only_eyes_error_level():
    """warn 级在本地是常态（低置信计数、缓存命中），推给人会训练出"忽略通知"的习惯。"""
    hits = _eval([_row("a", severity="warn"), _row("b", severity="error")]).hits
    assert [h.code for h in hits] == ["CONTEXT_OVERFLOW"]


def test_code_whitelist_and_exclusion():
    rows = [_row("a", code="CONTEXT_OVERFLOW"), _row("b", code="TOOL_LOOP"),
            _row("c", code="PROVIDER_ERROR")]
    only = _eval(rows, rule=AlertRule(codes=("TOOL_LOOP",))).hits
    assert [h.code for h in only] == ["TOOL_LOOP"]

    minus = _eval(rows, rule=AlertRule(exclude_codes=("TOOL_LOOP",))).hits
    assert {h.code for h in minus} == {"CONTEXT_OVERFLOW", "PROVIDER_ERROR"}


def test_severity_gate_still_applies_inside_a_code_whitelist():
    rule = AlertRule(codes=("NO_ENGINE_COUNT",), severities=("warn",))
    assert rule.matches("NO_ENGINE_COUNT", "error") is False
    assert rule.matches("NO_ENGINE_COUNT", "warn") is True


def test_disabled_rule_produces_nothing():
    assert _eval([_row("a")], rule=AlertRule(enabled=False)).hits == ()


# ── 窗口 ─────────────────────────────────────────────────────────
def test_window_is_inclusive_at_the_edge_and_excludes_older():
    rule = AlertRule(window_s=60, min_count=1)
    rows = [_row("in", ago_s=60), _row("out", ago_s=61), _row("future", ago_s=-5)]
    assert _eval(rows, rule=rule).hits[0].n == 1, "出窗的与来自未来的都不算"


def test_groups_are_counted_per_code():
    rows = [_row("a1"), _row("a2"), _row("b1", code="TOOL_LOOP")]
    hits = {h.code: h.n for h in _eval(rows, rule=AlertRule(min_count=1)).hits}
    assert hits == {"CONTEXT_OVERFLOW": 2, "TOOL_LOOP": 1}


def test_min_count_is_a_gate_not_a_display():
    rows = [_row("a1")]
    assert _eval(rows, rule=AlertRule(min_count=3)).hits == ()
    assert _eval([*rows, _row("a2"), _row("a3")], rule=AlertRule(min_count=3)).hits[0].n == 3


# ── cooldown ─────────────────────────────────────────────────────
def test_cooldown_suppresses_and_says_why():
    hits = _eval([_row("a")], last={"CONTEXT_OVERFLOW": "2026-10-05T02:55:00+00:00"}).hits
    assert len(hits) == 1 and hits[0].suppressed != ""
    assert "cooldown" in hits[0].suppressed and "300s" in hits[0].suppressed


def test_cooldown_is_per_code():
    rows = [_row("a", code="CONTEXT_OVERFLOW"), _row("b", code="TOOL_LOOP")]
    hits = _eval(rows, last={"CONTEXT_OVERFLOW": "2026-10-05T02:59:00+00:00"}).hits
    by = {h.code: h.suppressed for h in hits}
    assert by["CONTEXT_OVERFLOW"] != "" and by["TOOL_LOOP"] == ""


def test_cooldown_expiry_re_notifies():
    """上次投递已经在 cooldown 之外 ⇒ 该再通知一次（一条一直坏的异常要能被反复提醒）。"""
    hits = _eval([_row("a")], last={"CONTEXT_OVERFLOW": "2026-10-05T01:00:00+00:00"}).hits
    assert hits[0].suppressed == ""


def test_unparseable_last_trigger_time_is_not_treated_as_suppressed():
    hits = _eval([_row("a")], last={"CONTEXT_OVERFLOW": "不是时间"}).hits
    assert hits[0].suppressed == "", "解析不出来就当没投过：宁多发一条，不要静默不发"


# ── 计数可信度 ───────────────────────────────────────────────────
def test_truncated_scan_reports_a_lower_bound():
    hit = _eval([_row("a")], truncated=True).hits[0]
    assert hit.count_text() == "1+" and hit.truncated is True


def test_bad_timestamps_are_counted_not_dropped():
    """库里时间戳坏掉的行不能算进窗口，但必须报出来——"有一批异常的时间是坏的"值得知道。"""
    rows = [_row("a"), Row(id="x", code="CONTEXT_OVERFLOW", severity="error",
                           created_at="昨天吧")]
    result = _eval(rows)
    assert result.undated == 1
    assert [h.n for h in result.hits] == [1]


# ── 规则自身的形状 ───────────────────────────────────────────────
@pytest.mark.parametrize("kw", [
    {"window_s": 0}, {"window_s": -1}, {"min_count": 0}, {"cooldown_s": -5},
])
def test_impossible_rules_are_rejected_at_construction(kw):
    # 配了但永远命中不了的规则比没配更危险：它看起来在工作
    with pytest.raises(ValueError):
        AlertRule(**kw)


def test_rule_snapshot_carries_its_source():
    snap = AlertRule(codes=("A",), source="file").as_dict()
    assert snap["source"] == "file" and snap["codes"] == ["A"]
    assert snap["min_count"] == 1, "快照里要有阈值本身，事后才解释得了当时为什么触发"


def test_hit_summary_is_one_line_and_names_the_window():
    hit = _eval([_row("a"), _row("b")], rule=AlertRule(window_s=120)).hits[0]
    assert hit.summary() == "CONTEXT_OVERFLOW 在 120s 窗口里出现 2 次"


# ── 消息文案与 SPECS 同源 ────────────────────────────────────────
def test_message_carries_the_spec_action_not_a_rewritten_one():
    """通知必须带"下一步做什么"，且文案只能来自 obs/anomalies.py。"""
    from onyx.obs.alerts.channels import default_alert_path
    from onyx.obs.alerts.service import message_for
    from onyx.obs.anomalies import SPECS

    hit = _eval([_row("a", code="CONTEXT_OVERFLOW", ago_s=5)]).hits[0]
    text = message_for(hit)
    assert SPECS["CONTEXT_OVERFLOW"].action in text
    assert SPECS["CONTEXT_OVERFLOW"].meaning in text
    assert "1 次 / 300s" in text

    # 未登记的码不许编一句解释，只说它没登记
    ghost = _eval([_row("g", code="NOT_A_CODE")], rule=AlertRule(codes=("NOT_A_CODE",))).hits[0]
    assert "未登记的异常码 NOT_A_CODE" in message_for(ghost)

    # 文件出口的落盘按天分片（这里只验证路径推导，写入本身在 service 测试里）
    assert default_alert_path("D:/x/.data").name == "alerts.jsonl"
    assert FileChannel(default_alert_path("D:/x/.data"))._target(NOW).name == "alerts-20261005.jsonl"
    assert Alert(code="A", severity="error", message="m", n=1, window_s=60,
                 triggered_at=NOW, rule={}).as_dict()["n"] == 1
