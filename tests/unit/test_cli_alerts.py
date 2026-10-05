"""S27 验收：`onyx alerts ls` / `onyx alerts test`。

这两条命令存在的唯一理由是回答"我没收到通知"。四种原因（没命中 / 被 cooldown 挡了 /
渠道失败 / 这个进程没装配出口）必须在输出里分得开，否则人会去做一件没用的事：
重启服务，然后祈祷。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from onyx.cli import app
from onyx.store.db import Database
from onyx.store.records import AlertTriggerRecord
from onyx.store.repos import AlertRepo

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ONYX_CONFIG", raising=False)
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


def _trigger(i: str, **kw) -> AlertTriggerRecord:
    base = dict(
        id=i, created_at="2026-10-05T03:00:00+00:00", code="CONTEXT_OVERFLOW",
        severity="error", rule={"min_count": 1, "window_s": 300, "source": "default"},
        n_in_window=2, window_s=300, first_anomaly_id="a1", last_anomaly_id="a2",
        trace_ids=("t1",), channel="file", status="sent", detail="写入 1 行",
    )
    base.update(kw)
    return AlertTriggerRecord(**base)


# ── ls ────────────────────────────────────────────────────────────
def test_ls_prints_the_effective_rule_even_when_empty(isolated_data_dir):
    """空历史必须能区分"没出事"与"根本没在判"。判据那一行就是干这件事的。"""
    result = _run("alerts", "ls")
    assert result.exit_code == 0, result.output
    assert "没有触发记录" in result.output
    assert "cooldown" in result.output and "error" in result.output
    assert "出处 default" in result.output


def test_ls_lists_rows_newest_first_and_hides_test_rows(isolated_data_dir):
    db = Database(isolated_data_dir / "onyx.sqlite")
    repo = AlertRepo(db)
    repo.insert_trigger(_trigger("r1"))
    repo.insert_trigger(_trigger("r2", channel="webhook", status="failed",
                                detail="ConnectError: 拒绝连接"))
    repo.insert_trigger(_trigger("r3", is_test=True))
    db.close()

    out = _run("alerts", "ls").output
    assert "✓" in out and "✗" in out
    assert "webhook" in out and "file" in out
    assert "ConnectError" in out, "失败原因要直接可见，否则人只会看到「没通知」"
    assert "r3" not in out  # 测试行默认不混进来
    assert "告警触发（3）" in _run("alerts", "ls", "--include-test").output


def test_ls_filters_by_code_and_status(isolated_data_dir):
    db = Database(isolated_data_dir / "onyx.sqlite")
    repo = AlertRepo(db)
    repo.insert_trigger(_trigger("r1"))
    repo.insert_trigger(_trigger("r2", code="TOOL_LOOP"))
    db.close()
    out = _run("alerts", "ls", "--code", "TOOL_LOOP").output
    assert "TOOL_LOOP" in out and "CONTEXT_OVERFLOW" not in out
    assert "没有触发记录" in _run("alerts", "ls", "--status", "failed").output


def test_ls_shows_the_file_provenance_when_a_config_file_sets_it(isolated_data_dir, monkeypatch):
    cfg = isolated_data_dir / "onyx.toml"
    cfg.write_text("[alerts]\nmin_count = 4\nwindow_s = 90\n", encoding="utf-8")
    monkeypatch.setenv("ONYX_CONFIG", str(cfg))
    out = _run("alerts", "ls").output
    assert "满 4 次触发" in out and "90s" in out and "出处 file" in out


# ── test ──────────────────────────────────────────────────────────
def test_alerts_test_writes_to_the_file_channel_and_marks_it(isolated_data_dir):
    result = _run("alerts", "test")
    assert result.exit_code == 0, result.output
    assert "✓ file:" in result.output
    assert "不影响真实 cooldown" in result.output

    files = list((isolated_data_dir / "alerts").glob("*.jsonl"))
    assert files, "出口说写入了，盘上就得真有那一行"
    payload = json.loads(files[0].read_text(encoding="utf-8").splitlines()[-1])
    assert payload["is_test"] is True
    assert payload["code"] == "CONTEXT_OVERFLOW"
    assert "建议" in payload["message"], "通知要带下一步做什么，不然人还得再查一次"

    db = Database(isolated_data_dir / "onyx.sqlite")
    try:
        rows = AlertRepo(db).list_triggers()
        assert [r.is_test for r in rows] == [True]
        assert AlertRepo(db).last_real_trigger_at("CONTEXT_OVERFLOW") is None
    finally:
        db.close()


def test_alerts_test_can_name_a_channel_that_is_not_configured(isolated_data_dir):
    """点名一个没装配的出口要说"没有这个出口、可选项有哪些"，而不是默默不发。"""
    result = _run("alerts", "test", "--channel", "webhook")
    assert result.exit_code == 2
    assert "webhook" in result.output and "file" in result.output


def test_alerts_test_refuses_when_alerts_are_disabled(isolated_data_dir, monkeypatch):
    cfg = isolated_data_dir / "onyx.toml"
    cfg.write_text("[alerts]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setenv("ONYX_CONFIG", str(cfg))
    result = _run("alerts", "test")
    assert result.exit_code == 2
    assert "enabled = false" in result.output
    assert not (isolated_data_dir / "alerts").exists(), "关掉就是真的不发，而不是发了但不显示"


def test_alerts_test_honours_a_custom_file_path(isolated_data_dir, monkeypatch):
    target = isolated_data_dir / "notify" / "my-alerts.jsonl"
    cfg = isolated_data_dir / "onyx.toml"
    cfg.write_text(f'[alerts]\nfile = "{target.as_posix()}"\n', encoding="utf-8")
    monkeypatch.setenv("ONYX_CONFIG", str(cfg))
    assert _run("alerts", "test").exit_code == 0
    produced = list(target.parent.glob("*.jsonl"))
    assert produced and produced[0].name.startswith("my-alerts-"), "按天分片跟着自定义名走"
