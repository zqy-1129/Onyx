"""S15 验收：模型 × 任务矩阵与导出。

报告的全部价值是"脱离看板也能读"，所以这里盯三件事：
1. 选格规则：每组合只取**最新一次 done**，未完成的运行不许进矩阵；
2. 「—」与 0 的区分在所有格式里都成立（UI_DESIGN R2）；
3. 模型名/任务名是外部输入，HTML 里必须转义——报告是会被别人打开的文件。
"""

from __future__ import annotations

import pytest

from onyx.eval.compare import Comparison, PairedCase, compare_runs
from onyx.eval.task import HEADLINE_METRICS, headline_of
from onyx.report.eval_report import (
    MIN_AXES_FOR_RADAR,
    build_matrix,
    render_csv,
    render_html,
    render_markdown,
)
from onyx.store.records import RunRecord


def _run(run_id: str, model: str, task: str, *, status="done", started="2026-10-03T00:00:00+00:00",
         aggregate=None, dataset_id="intent_zh-v1", revision="seed=1", cost=None) -> RunRecord:
    return RunRecord(
        id=run_id, task_id=task, model_id=model, started_at=started, status=status,
        n_cases=10, n_done=10, aggregate=aggregate or {"macro_f1": 0.8, "low_confidence": False},
        cost=cost or {"requests": 10, "in_tokens": 100, "out_tokens": 20, "wall_ms": 1500.0},
        dataset_id=dataset_id, dataset_revision=revision,
    )


# ── 选格规则 ──────────────────────────────────────────────────────
def test_matrix_keeps_the_latest_done_run_per_model_and_task():
    runs = [
        _run("r1", "a", "t1", aggregate={"macro_f1": 0.5}, started="2026-10-01T00:00:00+00:00"),
        _run("r2", "a", "t1", aggregate={"macro_f1": 0.9}, started="2026-10-03T00:00:00+00:00"),
    ]
    matrix = build_matrix(runs)
    cell = matrix.cell("a", "t1")
    assert cell.run_id == "r2" and cell.value == pytest.approx(0.9), "取最新，不取最好"


def test_incomplete_runs_are_excluded():
    """running / cancelled 的分数没有可比性，放进网格会看起来一样可信。"""
    matrix = build_matrix([
        _run("done", "a", "t1"), _run("half", "b", "t1", status="cancelled"),
        _run("live", "c", "t1", status="running"),
    ])
    assert matrix.models == ("a",)
    assert matrix.cell("b", "t1") is None and matrix.cell("c", "t1") is None


def test_mixed_datasets_warn_instead_of_blending():
    """跨数据集的矩阵格子不可比，而表格本身看不出来。"""
    matrix = build_matrix([
        _run("r1", "a", "t1", dataset_id="ds-v1", revision="r1"),
        _run("r2", "b", "t2", dataset_id="ds-v2", revision="r2"),
    ])
    assert len(matrix.provenance) == 2
    assert any("混了 2 份数据" in w for w in matrix.warnings)


def test_single_dataset_matrix_has_no_warnings():
    matrix = build_matrix([_run("r1", "a", "t1"), _run("r2", "b", "t1")])
    assert matrix.warnings == ()


def test_undecidable_cell_is_reported_as_unknown_not_zero():
    """`macro_f1=None` 是"没有可判定样本"，矩阵里必须是「—」并且进警告。"""
    matrix = build_matrix([_run("r1", "a", "t1", aggregate={"macro_f1": None,
                                                             "low_confidence": True})])
    cell = matrix.cell("a", "t1")
    assert cell.value is None
    assert "—" in render_markdown(matrix) and "0.000" not in render_markdown(matrix)
    assert any("没有可判定样本" in w for w in matrix.warnings)
    assert "<span class=\"ci\">—</span>" in render_html(matrix)


# ── 主分数口径 ────────────────────────────────────────────────────
def test_thin_coverage_cell_shows_its_denominator_and_is_warned():
    """主分数只覆盖少数样本时，那一格必须自己说出来。

    真机踩到的形态：qwen 输出守格式，gpt-oss 输出不守格式 ⇒
    后者的 `macro_f1` 在 8/236 个可判定样本上算出 1.000。
    矩阵里单看这个数会得出"这个模型更强"，而它的 `format_valid_rate` 只有 3.4%。
    """
    aggregate = {"macro_f1": 1.0, "macro_f1_ci": {"low": 1.0, "high": 1.0, "n": 8},
                 "n_judged": 8, "n_total": 236, "format_valid_rate": 0.034,
                 "low_confidence": True}
    matrix = build_matrix([_run("r1", "gpt-oss:20b", "intent_classification",
                                aggregate=aggregate)])
    cell = matrix.cell("gpt-oss:20b", "intent_classification")
    assert cell.coverage == pytest.approx(8 / 236)
    assert matrix.thin_cells == (cell,)
    assert any("只覆盖不到一半样本" in w and "8/236" in w for w in matrix.warnings)
    text = render_markdown(matrix)
    assert "可判定 8/236" in text, "分数旁边就要看见分母"

    html = render_html(matrix)
    assert "可判定 8/236" in html
    assert "coverage" in render_csv(matrix).splitlines()[0]


def test_full_coverage_cell_does_not_babble_about_coverage():
    """236/236 时再写"可判定 236/236"是噪声，会把真正需要点名的格子淹掉。"""
    matrix = build_matrix([_run("r1", "a", "t1", aggregate={
        "macro_f1": 0.99, "macro_f1_ci": {"low": 0.97, "high": 1.0, "n": 236},
        "n_judged": 236, "n_total": 236})])
    assert "可判定 236/236" not in render_markdown(matrix)
    assert matrix.thin_cells == ()


def test_tasks_without_a_judged_count_are_left_alone():
    """没有 `n_judged` 键的任务不猜比例——宁可不知道，也不编一个分母。"""
    matrix = build_matrix([_run("r1", "a", "tool_selection",
                                aggregate={"must_call_acc": 0.64})])
    assert matrix.cell("a", "tool_selection").coverage is None
    assert matrix.thin_cells == ()


def test_headline_priority_is_shared_and_does_not_skip_an_unknown():
    """headline_of 取"第一个出现的"，不是"第一个非空的"。

    跳过 None 去选下一个候选，会把"这个任务的主分数算不出来"显示成
    "另一个指标得了分"——两个结论的修法完全相反。
    """
    assert headline_of({"macro_f1": None, "accuracy": 0.9}) == ("macro_f1", None)
    assert headline_of({"must_call_acc": 0.6}) == ("must_call_acc", 0.6)
    assert headline_of({}) is None
    assert HEADLINE_METRICS[0] == "macro_f1"


def test_tool_task_cell_uses_its_own_headline_metric():
    matrix = build_matrix([_run("r1", "a", "tool_selection",
                                aggregate={"must_call_acc": 0.639, "low_confidence": True})])
    cell = matrix.cell("a", "tool_selection")
    assert cell.metric == "must_call_acc" and cell.low_confidence is True
    assert "must_call_acc 0.639" in render_markdown(matrix)


# ── 导出格式 ──────────────────────────────────────────────────────
def test_csv_has_one_row_per_cell_and_blank_for_unknown():
    matrix = build_matrix([_run("r1", "a", "t1"),
                           _run("r2", "b", "t1", aggregate={"macro_f1": None})])
    rows = render_csv(matrix).strip().splitlines()
    assert rows[0].startswith("model,task,metric,value,ci_low,ci_high")
    assert len(rows) == 3
    import csv as _csv
    parsed = list(_csv.DictReader(rows))
    unknown = next(row for row in parsed if row["model"] == "b")
    assert unknown["value"] == "" and unknown["ci_low"] == "" and unknown["ci_high"] == "", \
        "未知必须是空字段，不能是 0——表格软件会把空与 0 画成两种柱子的高度"
    assert unknown["n"] == "10", "n 仍然要报，否则看不出是「没考到」还是「考了 0 分」"
    known = next(row for row in parsed if row["model"] == "a")
    assert known["value"] == "0.800000"
    assert "intent_zh-v1" in rows[1], "来历要跟着每一行走，否则 csv 出了门就没人知道来源"


def test_markdown_carries_ci_and_the_cost_of_each_run():
    matrix = build_matrix([_run(
        "r1", "a", "t1",
        aggregate={"macro_f1": 0.812, "macro_f1_ci": {"low": 0.77, "high": 0.85, "n": 40},
                   "low_confidence": False},
    )])
    text = render_markdown(matrix)
    assert "0.812 [0.770–0.850]" in text
    assert "每次运行的成本" in text and "100" in text
    assert "每格是该模型在该任务上**最新一次 done 运行**" in text or "最新一次 done 运行" in text


def test_html_escapes_names_because_reports_are_opened_by_other_people():
    """模型名来自外部（Ollama 的 tag），报告是会被别人双击打开的 html。"""
    matrix = build_matrix([_run("r1", "<script>alert(1)</script>", "t1")])
    html = render_html(matrix)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_radar_is_skipped_when_there_are_too_few_axes():
    """两个任务的"雷达"是一条线段，形状不表达任何东西。"""
    two = build_matrix([_run("r1", "a", "t1"), _run("r2", "a", "t2")])
    assert MIN_AXES_FOR_RADAR == 3
    assert "<svg" not in render_html(two)

    three = build_matrix([_run("r1", "a", f"t{i}") for i in range(3)])
    html = render_html(three)
    assert "<svg" in html and html.count("<polygon") >= 5, "4 条网格 + 每条模型序列一个多边形"


# ── 对比结论进报告 ────────────────────────────────────────────────
def _paired(tmp_path, deltas):
    """直接在库里造两个 run + grade，然后走真实的 compare_runs。"""
    from onyx.store.db import Database
    from onyx.store.records import CaseRecord, DatasetRecord, GradeRecord, TaskRecord
    from onyx.store.repos import EvalRepo

    db = Database(tmp_path / "t.sqlite")
    repo = EvalRepo(db)
    repo.upsert_dataset(DatasetRecord(id="ds", imported_at="2026-10-03", revision="r1"))
    repo.upsert_task(TaskRecord(id="intent_classification", name="t", dataset_id="ds",
                                metrics=["macro_f1"]))
    for run_id, model in (("run-a", "A"), ("run-b", "B")):
        repo.insert_run(RunRecord(id=run_id, task_id="intent_classification", model_id=model,
                                  started_at="2026-10-03T00:00:00+00:00", status="done",
                                  dataset_id="ds", dataset_revision="r1",
                                  config={"k": 1, "split": "default"},
                                  params_snapshot={"temperature": 0.0}))
    for index, (a, b) in enumerate(deltas):
        case_id = f"c{index}"
        repo.upsert_cases([CaseRecord(id=case_id, dataset_id="ds", ord=index,
                                      input={"instruction": case_id}, expect={})])
        for run_id, score in (("run-a", a), ("run-b", b)):
            repo.upsert_grade(GradeRecord(id=f"{run_id}-{case_id}", eval_run_id=run_id,
                                          case_id=case_id, seq=0, score=score,
                                          verdict="correct" if score else "wrong",
                                          passed=bool(score),
                                          trace_id=f"tr-{run_id}-{case_id}",
                                          graded_at="2026-10-03"))
    result = compare_runs(repo, "run-a", "run-b")
    db.close()
    return result


def test_comparison_appears_in_markdown_and_html(tmp_path):
    comparison = _paired(tmp_path, [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    matrix = build_matrix([comparison.base, comparison.target])
    text = render_markdown(matrix, [comparison])
    assert "配对对比" in text and "劣化 2" in text and "改善 1" in text
    assert "`c0`" in text, "劣化清单要能直接当工单用"

    html = render_html(matrix, [comparison])
    assert "配对对比" in html and "run-a" in html
    assert "劣化 <span class=\"bad\">2</span>" in html


def test_comparison_dict_form_renders_too(tmp_path):
    """CLI 与 API 传的是 as_dict()，报告层两种都要吃得下。"""
    comparison = _paired(tmp_path, [(1.0, 0.0), (0.0, 1.0)])
    assert isinstance(comparison, Comparison)
    payload = comparison.as_dict()
    text = render_markdown(build_matrix([comparison.base, comparison.target]), [payload])
    assert "配对样本" in text
    assert "PairedCase" not in text


def test_markdown_does_not_claim_an_interval_when_there_is_none(tmp_path):
    comparison = _paired(tmp_path, [(1.0, 0.0)])
    text = render_markdown(build_matrix([comparison.base, comparison.target]), [comparison])
    assert "给不出区间" in text, "n=1 时要明说没有区间"
    assert "95% CI [" not in text, "不许编一个区间出来"


def test_paired_case_kind_and_instruction_survive_to_the_report(tmp_path):
    comparison = _paired(tmp_path, [(1.0, 0.0), (0.0, 0.0)])
    item = comparison.paired[0]
    assert isinstance(item, PairedCase)
    assert item.instruction == "c0", "对比页要能看出是哪道题，题干得从 grade 里带出来"
