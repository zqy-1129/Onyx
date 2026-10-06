"""S38 的两个新入口：`onyx token explain` 与 `onyx report usage`。

CLI 测试的价值不在"跑通了"，在于**退出码与措辞**：这两条命令是给人排障用的，
"不闭合 ⇒ 非 0"、"没有 trace ⇒ 说没跑过而不是 0"必须是可断言的行为。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from onyx.cli import app
from onyx.report.usage_report import CSV_COLUMNS
from onyx.store.db import Database
from onyx.store.records import TokenPartRecord, TraceRecord, UsageRecord
from onyx.store.repos import TraceRepo, UsageRepo

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


def _db(tmp_path) -> Database:
    return Database(tmp_path / "onyx.sqlite")


def _seed(tmp_path, *, trace_id: str = "T1", with_parts: bool = True, closing: bool = True):
    db = _db(tmp_path)
    TraceRepo(db).upsert(TraceRecord(
        id=trace_id, kind="generation", purpose="chat",
        started_at="2026-10-05T10:00:00+00:00", status="ok",
        provider_id="ollama-local", model_id="ollama-local/qwen3.5:9b", model_name="qwen3.5:9b",
    ))
    UsageRepo(db).upsert(UsageRecord(
        trace_id=trace_id, source="engine", confidence="high",
        in_tokens=1000, out_tokens=50,
    ))
    if with_parts:
        messages = 800 if closing else 1200
        UsageRepo(db).replace_parts(trace_id, [
            TokenPartRecord(trace_id=trace_id, part="system", tokens=100),
            TokenPartRecord(trace_id=trace_id, part="messages", ord=1, tokens=messages),
            TokenPartRecord(trace_id=trace_id, part="template_ctl", tokens=100),
            TokenPartRecord(trace_id=trace_id, part="output", tokens=50),
        ])
    db.close()


# ── token explain ────────────────────────────────────────────────
def test_explain_rejects_an_unknown_trace_with_a_way_forward(tmp_path):
    result = _run("token", "explain", "nope")
    assert result.exit_code == 2
    assert "onyx traces ls" in result.output, "报错要给出下一步，而不是只说没有"


def test_explain_reports_closure_and_exits_one_when_it_does_not_add_up(tmp_path):
    _seed(tmp_path, closing=False)
    result = _run("token", "explain", "T1")
    assert result.exit_code == 1, result.output
    assert "不闭合" in result.output, "人读的那份要直接给判定"
    assert "阶梯" in result.output and "未实现" in result.output
    # 数值断言走 --json：rich 会按终端宽度折行，在人读的输出里抠数字测的是渲染器不是逻辑
    payload = json.loads(_run("token", "explain", "T1", "--json").output)
    assert payload["closure"] == {"checked": True, "closed": False, "sum": 1400,
                                 "reported": 1000, "delta": 400,
                                 "note": "分段求和比采信值多 400 tok"}
    assert payload["clean"] is False


def test_explain_is_clean_when_parts_add_up(tmp_path):
    _seed(tmp_path, closing=True)
    result = _run("token", "explain", "T1")
    assert result.exit_code == 0, result.output
    assert "✓ 闭合" in result.output


def test_explain_marks_missing_attribution_as_undetermined_not_closed(tmp_path):
    """没有分段归因 ⇒ 「未判定」。写成"闭合"就是把没测说成通过。"""
    _seed(tmp_path, with_parts=False)
    result = _run("token", "explain", "T1")
    assert result.exit_code == 0, result.output
    assert "未判定" in result.output and "✓ 闭合" not in result.output


def test_explain_json_carries_the_same_thresholds_the_reconciler_uses(tmp_path):
    _seed(tmp_path, closing=True)
    payload = json.loads(_run("token", "explain", "T1", "--json").output)
    assert payload["clean"] is True and payload["chosen"]["source"] == "engine"
    assert payload["thresholds"]["drift_pct"] == 0.10
    assert {row["source"] for row in payload["tiers"]} >= {"engine", "fitted", "compat", "heuristic"}


# ── report usage ─────────────────────────────────────────────────
def test_report_usage_on_an_empty_db_says_no_traces(tmp_path):
    _db(tmp_path).close()   # 建库（跑迁移），但不塞数据
    result = _run("report", "usage")
    assert result.exit_code == 0, result.output
    assert "没有 trace" in result.output, "空范围要说「没跑过」，不是打印一串 0"


def test_report_usage_rejects_an_unknown_format(tmp_path):
    _db(tmp_path).close()
    result = _run("report", "usage", "--fmt", "pdf")
    assert result.exit_code == 2
    assert "table/csv/markdown/json" in result.output


def test_report_usage_csv_to_file_keeps_the_pinned_header(tmp_path):
    _seed(tmp_path, closing=True)
    target = tmp_path / "usage.csv"
    result = _run("report", "usage", "--fmt", "csv", "--out", str(target))
    assert result.exit_code == 0, result.output
    assert "已写出" in result.output
    assert target.read_text(encoding="utf-8").splitlines()[0] == ",".join(CSV_COLUMNS)


def test_report_usage_json_matches_the_seeded_totals(tmp_path):
    _seed(tmp_path, closing=True)
    payload = json.loads(_run("report", "usage", "--fmt", "json").output)
    assert payload["traces"] == 1 and payload["in_tokens"] == 1000
    assert payload["drift"]["n"] == 0, "没有第二路计数就没有漂移样本，不是 0% 漂移"


def test_report_usage_refuses_a_relative_since_instead_of_printing_an_empty_table(tmp_path):
    """真机踩过：`--since 7d`  exit 0 + 一张空表，读起来像"这周没跑过"。

    空表本身有一条专门的输出（`test_report_usage_on_an_empty_db...`），所以这里要测的是
    **写错格式必须响**：退出码 2、给出能用的写法。
    """
    _seed(tmp_path, closing=True)
    result = _run("report", "usage", "--since", "7d")
    assert result.exit_code == 2, result.output
    assert "7d" in result.output and "没有 trace" not in result.output

    ok = _run("report", "usage", "--since", "2026-10-01", "--fmt", "json")
    assert ok.exit_code == 0, ok.output
    assert json.loads(ok.output)["since"] == "2026-10-01T00:00:00+00:00"
