"""指令遵循任务在 mock 引擎上真跑一轮（S31 第三条同源）。

契约测试管的是"指标声明与产出同源"，这里管**跑起来之后**：
- 分数能下钻到真实 trace；
- 三个口径在真数据上仍然各说各话（不是都被四舍五入成同一个数）；
- `by_kind` 每种约束都带着自己的分母；
- 一个"什么都不写"的模型必须得 **0 分**而不是「没考到」，
  也不能因为 `max_chars` 对空输出判通过而拿到分——这只能在真跑时看得见。
"""

from __future__ import annotations

import json

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.eval.datasets.loader import load_builtin
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.task import Verdict
from onyx.eval.tasks.instruction_following import InstructionFollowing
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "mock/ins"


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _task() -> InstructionFollowing:
    return InstructionFollowing(load_builtin("instructions_zh"), model=MODEL)


def _runner(env, texts: list[str], task: InstructionFollowing) -> EvalRunner:
    _, _, observer, blobs = env
    scripts = [MockScript(text=text, done_reason="stop", in_tokens=140, out_tokens=30)
               for text in texts]
    provider = MockProvider(scripts={MODEL: scripts}, models=(MODEL,))
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())
    return EvalRunner(gateway, EvalRepo(env[0]), task, dataset=task.dataset, clock=FakeClock())


def _answers(task, n: int, *, break_every: int = 0) -> list[str]:
    """按 `load()` 的顺序造剧本：默认交参考回答（必然全满足），每 `break_every` 条故意漏一条。"""
    texts: list[str] = []
    for index, case in enumerate(task.load(limit=n)):
        reference = str(case.meta.get("reference") or "")
        if break_every and index % break_every == 0 and reference:
            texts.append(f"- {reference}\n- 我还多写了一行，好让列表符号与字数都出问题：" + "若" * 60)
        else:
            texts.append(reference)
    return texts


def test_a_real_run_keeps_the_three_denominators_apart(env):
    _, sink, _, _ = env
    task, n = _task(), 12
    report = _runner(env, _answers(task, n, break_every=4), task).run(
        RunConfig(model=MODEL, seed=42, limit=n)
    )
    sink.flush(5.0)

    assert report.status == "done" and report.ok
    assert report.n_cases == n and report.n_done == n and report.n_error == 0

    aggregate = report.aggregate
    assert aggregate["n_total"] == aggregate["n_attributable"] == n
    assert aggregate["n_judged"] == n
    assert aggregate["verdicts"][Verdict.CORRECT.value] == 9
    # 故意破坏的那 3 条不是 WRONG：它们仍然抄对了内容，只是排版与字数没听话。
    # PARTIAL 与 WRONG 分开计是这任务的核心（前者改形态约束，后者才是没照做）
    assert aggregate["verdicts"][Verdict.PARTIAL.value] == 3
    assert Verdict.WRONG.value not in aggregate["verdicts"]
    # 三个口径必须互不相等，否则其中一个已经被悄悄合并掉了
    assert aggregate["score"] != aggregate["micro_rate"]
    assert aggregate["all_satisfied_rate"] < aggregate["score"]
    assert aggregate["all_satisfied_rate"] == pytest.approx(9 / 12)
    assert aggregate["constraint_total"] == sum(
        len(case.expect["constraints"]) for case in task.load(limit=n)
    )
    assert aggregate["constraint_satisfied"] < aggregate["constraint_total"]
    assert aggregate["score_ci"]["n"] == aggregate["n_judged"] == n
    assert aggregate["score_ci"]["point"] == pytest.approx(aggregate["score"])
    assert aggregate["low_confidence"] is True
    assert aggregate["empty_outputs"] == 0 and aggregate["refusal_rate"] == 0.0
    assert aggregate["scoring"] == "gen-based"


def test_by_kind_reports_a_denominator_for_every_tested_constraint(env):
    _, sink, _, _ = env
    task, n = _task(), 10
    report = _runner(env, _answers(task, n), task).run(RunConfig(model=MODEL, limit=n))
    sink.flush(5.0)

    by_kind = report.aggregate["by_kind"]
    tested = {str(item["kind"]) for item in (
        check for case in task.load(limit=n)
        for check in case.expect["constraints"]
    )}
    assert set(by_kind) == tested, "by_kind 的键必须正好是这次真考到的约束类型"
    for kind, entry in by_kind.items():
        assert entry["n"] > 0, f"{kind} 没有分母"
        assert entry["satisfied"] <= entry["n"]
        assert entry["rate"] == pytest.approx(entry["satisfied"] / entry["n"])
    # 全交参考回答时每条约束都该满分——这也是"考卷有解"的另一种证法
    assert all(entry["rate"] == 1.0 for entry in by_kind.values()), by_kind


def test_every_grade_points_at_a_real_trace(env):
    _, sink, _, _ = env
    task, n = _task(), 6
    report = _runner(env, _answers(task, n), task).run(RunConfig(model=MODEL, limit=n))
    sink.flush(5.0)

    repo = TraceRepo(env[0])
    assert len(report.grades) == n
    for grade in report.grades:
        assert grade.trace_id, f"{grade.case_id} 没有 trace_id"
        assert repo.get(grade.trace_id) is not None
        assert grade.metrics["checks"], "分数要能解释：每条约束的判定都得留下来"
    assert len({g.trace_id for g in report.grades}) == n


def test_a_silent_model_scores_zero_not_unknown(env):
    """什么都不写的模型必须得 0 分。

    逐条判的话 `max_chars` 与 `forbids` 会对空输出判"通过"，
    于是最省token的答案能拿到一半分——那正是这任务要抓住的形态。
    """
    _, sink, _, _ = env
    task, n = _task(), 8
    report = _runner(env, [""] * n, task).run(RunConfig(model=MODEL, limit=n))
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["n_judged"] == n, "没写不等于没考到"
    assert aggregate["score"] == 0.0 and aggregate["micro_rate"] == 0.0
    assert aggregate["all_satisfied_rate"] == 0.0
    assert aggregate["constraint_satisfied"] == 0
    assert aggregate["empty_outputs"] == n
    assert aggregate["invalid_format_rate"] == 1.0
    assert aggregate["verdicts"][Verdict.INVALID_FORMAT.value] == n


def test_the_exam_provenance_lands_with_the_run(env):
    _, sink, _, _ = env
    task = _task()
    report = _runner(env, _answers(task, 4), task).run(RunConfig(model=MODEL, limit=4))
    sink.flush(5.0)

    assert report.dataset_id == "instructions_zh-v1"
    assert report.dataset_revision == "seed=20261005"
    record = EvalRepo(env[0]).get_dataset(report.dataset_id)
    assert record is not None and record.license == "generated-in-repo"
    assert record.splits["hard"] == 3
    stored = EvalRepo(env[0]).get_run(report.run_id)
    assert stored is not None and stored.task_id == "instruction_following"
    assert json.dumps(stored.aggregate, ensure_ascii=False)
