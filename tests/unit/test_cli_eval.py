"""S13 验收：eval CLI。

这些测试同时是"自测命令"的可执行版本——IMPLEMENTATION.md 里写的
`onyx eval import/run/show/ls` 必须真的能跑出预期输出。

重点盯两条容易被忽略的口径：
1. 未知显示「—」，**绝不显示 0**（UI_DESIGN R2）。曾经出现过
   macro_f1=None 时列表页悄悄改显示 pass_hat_k 的 0.000，看起来像"模型得了 0 分"，
   而真相是"一条可判定样本都没有"——两者的修法完全相反。
2. 分数必须带指标名与口径，否则一个孤零零的数字无法解释。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from onyx.cli import _fmt, _headline_score, app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


MOCK_RUN = ["eval", "run", "--task", "intent_classification", "--model", "mock/echo",
            "--provider", "mock", "--limit", "8", "--seed", "42", "--quiet"]


def _import_builtin():
    result = _run("eval", "import", "--builtin", "intent_zh")
    assert result.exit_code == 0, result.output
    return result


# ── 显示口径 ──────────────────────────────────────────────────────
def test_fmt_shows_unknown_as_a_dash_never_as_zero():
    assert _fmt(None) == "—"
    assert _fmt(0) == "0.000", "0 是一个真实的值，必须显示成 0"
    assert _fmt(0.9913) == "0.991"
    assert _fmt(True) == "是" and _fmt(False) == "否"


def test_headline_score_names_the_metric_it_shows():
    text = _headline_score({"macro_f1": 0.812, "macro_f1_ci": {"low": 0.77, "high": 0.85},
                            "low_confidence": False})
    assert text == "macro_f1 0.812 [0.770–0.850]"


def test_headline_score_shows_a_dash_when_the_primary_metric_is_unknown():
    """这条是回归测试：曾经悄悄换成 pass_hat_k 的 0.000，把"未知"显示成"0 分"。"""
    aggregate = {"macro_f1": None, "accuracy": None, "pass_hat_k": 0.0, "n_total": 12}
    text = _headline_score(aggregate)
    assert text.startswith("macro_f1 —"), text
    assert "0.000" not in text


def test_headline_score_flags_low_sample_size():
    text = _headline_score({"macro_f1": 0.9, "low_confidence": True})
    assert "⚠低样本" in text


def test_headline_score_handles_skipped_and_empty():
    assert _headline_score({}) == "—"
    assert _headline_score({"skip": {"reason": "x"}}) == "skipped"


def test_headline_score_falls_back_to_a_named_alternative():
    """任务没有 macro_f1 时才轮到下一个指标，而且同样要带名字。"""
    assert _headline_score({"accuracy": 0.5}).startswith("accuracy 0.500")
    assert _headline_score({"pass_hat_k": 0.25}).startswith("pass_hat_k 0.250")


# ── import / ls ───────────────────────────────────────────────────
def test_import_builtin_reports_provenance_and_subsets():
    result = _import_builtin()
    assert "intent_zh-v1" in result.output
    assert "236 条" in result.output
    assert "builtin:intent_zh" in result.output, "来源必须报出来"
    assert "seed=20261003" in result.output, "revision 必须报出来，否则换版本后分数不可比"
    assert "hard" in result.output, "子集要能单独跑，所以得让人知道有哪些"


def test_import_requires_a_source():
    result = _run("eval", "import")
    assert result.exit_code == 2
    assert "--builtin" in result.output


def test_import_from_a_jsonl_file(tmp_path):
    path = tmp_path / "mine.jsonl"
    path.write_text(
        '{"input": {"instruction": "查余额"}, "expect": {"label": "查余额"}}\n'
        '{"input": {"instruction": "转钱给张伟"}, "expect": {"label": "转账"}}\n',
        encoding="utf-8",
    )
    result = _run("eval", "import", str(path), "--id", "mine-v1", "--upstream", "手工整理",
                  "--license", "internal")
    assert result.exit_code == 0, result.output
    assert "mine-v1" in result.output and "2 条" in result.output
    assert "手工整理" in result.output


def test_import_reports_the_offending_line_of_a_broken_file(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": {"instruction": "a"}}\n{坏了}\n', encoding="utf-8")
    result = _run("eval", "import", str(path))
    assert result.exit_code == 1
    assert "第 2 行" in result.output


def test_ls_lists_datasets_and_tasks():
    _import_builtin()
    result = _run("eval", "ls")
    assert result.exit_code == 0, result.output
    assert "intent_zh-v1" in result.output
    assert "intent_classification" in result.output


# ── run ───────────────────────────────────────────────────────────
def test_run_offline_completes_and_reports_a_run_id():
    _import_builtin()
    result = _run(*MOCK_RUN)
    assert result.exit_code == 0, result.output
    assert "intent_classification · mock/echo · n=8" in result.output
    assert "run_id" in result.output and "onyx eval show" in result.output
    # mock provider 的默认回复不是标签，所以全是越界；关键是**报成越界而不是 0 分**
    assert "out_of_label" in result.output


def test_run_separates_the_content_and_format_dimensions():
    """DESIGN §9.4：两个维度必须分开打印，混成一个正确率会把格式问题读成能力问题。"""
    _import_builtin()
    result = _run(*MOCK_RUN)
    assert "[内容]" in result.output and "[格式]" in result.output
    assert "format_valid" in result.output and "out_of_label" in result.output
    assert "gen-based" in result.output, "打分口径必须交代，否则会被拿去和 leaderboard 比"


def test_run_json_output_is_machine_readable():
    _import_builtin()
    result = _run(*MOCK_RUN, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "done"
    assert payload["n_cases"] == 8 and payload["n_done"] == 8
    assert payload["aggregate"]["verdicts"]["out_of_label"] == 8
    assert payload["aggregate"]["scoring"] == "gen-based"
    assert payload["cost"]["requests"] == 8
    assert payload["run_id"]


def test_run_reports_cost_including_unmeasured_requests():
    _import_builtin()
    payload = json.loads(_run(*MOCK_RUN, "--json").output)
    cost = payload["cost"]
    assert cost["requests"] == 8
    assert cost["in_tokens"] > 0 and cost["out_tokens"] > 0
    assert cost["in_tokens_unknown"] == 0


def test_run_rejects_an_unknown_task():
    result = _run("eval", "run", "--task", "nope", "--model", "m", "--provider", "mock")
    assert result.exit_code == 2
    assert "未知任务" in result.output


def test_run_rejects_an_unknown_dataset():
    _import_builtin()
    result = _run(*MOCK_RUN, "--dataset", "nope")
    assert result.exit_code == 2
    assert "未知数据集" in result.output


def test_hard_subset_can_be_run_on_its_own():
    """难例单独跑才有意义：混在 200 条简单样本里会被稀释成一个好看的平均分。"""
    _import_builtin()
    payload = json.loads(_run(*MOCK_RUN, "--split", "hard", "--json").output)
    assert payload["status"] == "done"
    assert payload["n_cases"] > 0


def test_k_sampling_multiplies_the_request_count():
    _import_builtin()
    payload = json.loads(_run(*MOCK_RUN, "--k", "2", "--json").output)
    assert payload["cost"]["requests"] == 16, "8 条 × 2 次采样"
    assert payload["aggregate"]["k"] == 2
    assert "pass_hat_k" in payload["aggregate"]


# ── show ──────────────────────────────────────────────────────────
def test_show_lists_grades_with_trace_ids():
    """DoD：任一分数都能跳到 trace。表格里必须直接看得见 trace_id。"""
    _import_builtin()
    run_id = json.loads(_run(*MOCK_RUN, "--json").output)["run_id"]
    result = _run("eval", "show", run_id, "--limit", "3")
    assert result.exit_code == 0, result.output
    assert "out_of_label" in result.output
    assert "trace" in result.output
    # trace 列必须是真实 id 而不是「—」，否则"分数能点进 trace"就只是文档里的说法
    table = result.output.split("trace")[-1].split("错误样例")[0]
    body_rows = [line for line in table.splitlines() if "izh-" in line]
    assert body_rows, "表格里没有任何 grade 行"
    for row in body_rows:
        assert "01M" in row, f"trace 列是空的: {row}"


def test_show_json_includes_params_snapshot_and_costs():
    _import_builtin()
    run_id = json.loads(_run(*MOCK_RUN, "--json").output)["run_id"]
    payload = json.loads(_run("eval", "show", run_id, "--json").output)
    run = payload["run"]
    assert run["id"] == run_id
    # 参数快照必须落库：否则换了 temperature 之后的分数差异无法解释
    assert run["params_snapshot"]["temperature"] == 0.0
    assert run["params_snapshot"]["max_tokens"] == 32
    assert run["app_version"]
    assert run["aggregate"]["scoring"] == "gen-based"
    assert len(payload["grades"]) == 8
    assert all(g["trace_id"] for g in payload["grades"])


def test_show_filters_by_verdict():
    _import_builtin()
    run_id = json.loads(_run(*MOCK_RUN, "--json").output)["run_id"]
    payload = json.loads(_run("eval", "show", run_id, "--verdict", "out_of_label", "--json").output)
    assert len(payload["grades"]) == 8
    empty = json.loads(_run("eval", "show", run_id, "--verdict", "correct", "--json").output)
    assert empty["grades"] == []


def test_show_rejects_an_unknown_run():
    result = _run("eval", "show", "does-not-exist")
    assert result.exit_code == 2
    assert "找不到 run" in result.output
