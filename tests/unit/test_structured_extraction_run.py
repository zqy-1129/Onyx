"""结构化抽取任务在 mock 引擎上真跑一轮（S30 第三条同源）。

契约测试管的是"指标声明与产出同源"，这里管的是**跑起来之后**：
- 每条分数都能点进一条真实 trace（点不进去的分数不可验证，DESIGN §15）
- 分母之间必须自洽：`n_judged ≤ n_attributable ≤ n_total`，CI 的 n 等于 n_judged
- 负样本单独一条通路：一个"全都输出空对象"的模型在主分数上必须是「没考到」，
  不是满分——这条只能在真跑一遍时看得见
- 考卷的来历（revision）随 run 落库，否则两次分数不可比这件事发现不了
"""

from __future__ import annotations

import json

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.eval.datasets.loader import load_builtin
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.task import Verdict
from onyx.eval.tasks.structured_extraction import StructuredExtraction
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "mock/sie"


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _task() -> StructuredExtraction:
    return StructuredExtraction(load_builtin("structured_ie"), model=MODEL)


def _runner(env, texts: list[str], task: StructuredExtraction):
    _, _, observer, blobs = env
    scripts = [MockScript(text=text, done_reason="stop", in_tokens=120, out_tokens=24)
               for text in texts]
    provider = MockProvider(scripts={MODEL: scripts}, models=(MODEL,))
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())
    return EvalRunner(gateway, EvalRepo(env[0]), task, dataset=task.dataset,
                      clock=FakeClock())


def _answers(task: StructuredExtraction, n: int, *, split: str = "default",
              break_every: int = 0) -> list[str]:
    """按 `load()` 的顺序造剧本：默认全答对，`break_every` 每隔几条故意答错。

    剧本必须与 run 用同一个 split：MockProvider 按调用次序推进，
    错位会让"哪条答对了"变成两个序列的巧合，测不出任何东西。
    """
    texts: list[str] = []
    for index, case in enumerate(task.load(split=split, limit=n)):
        fields = dict(case.expect.get("fields") or {})
        if break_every and index % break_every == 0 and fields:
            first = sorted(fields)[0]
            fields[first] = "故意答错" if isinstance(fields[first], str) else 0.0
        texts.append(json.dumps(fields, ensure_ascii=False))
    return texts


def test_a_real_run_scores_every_case_and_keeps_the_denominators_consistent(env):
    _, sink, _, _ = env
    task, n = _task(), 12
    report = _runner(env, _answers(task, n, split="template", break_every=3), task).run(
        RunConfig(model=MODEL, seed=42, limit=n, split="template")
    )
    sink.flush(5.0)

    assert report.status == "done" and report.ok
    assert report.n_cases == n and report.n_done == n and report.n_error == 0

    aggregate = report.aggregate
    assert aggregate["n_total"] == aggregate["n_attributable"] == n
    assert aggregate["n_judged"] == n, "template 子集每条都有字段可抽，分母不该缩水"
    assert aggregate["verdicts"][Verdict.PARTIAL.value] == 4, "0/3/6/9 四条被故意答错"
    assert 0.0 < aggregate["score"] < 1.0, "全对里掺了故意答错的题，主分数必须掉下来"
    assert aggregate["score"] == pytest.approx(8 / 12)
    assert aggregate["score_ci"]["n"] == aggregate["n_judged"]
    assert aggregate["score_ci"]["point"] == pytest.approx(aggregate["score"])
    assert aggregate["json_valid_rate"] == 1.0 and aggregate["invalid_format_rate"] == 0.0
    assert aggregate["low_confidence"] is True, "12 条样本必须标低置信"
    assert aggregate["scoring"] == "gen-based"
    # 故意答错的仍算"结构合规"：它们丢的是内容分，不是格式分（§9.4）
    assert aggregate["schema_valid_rate"] == 1.0
    assert aggregate["exact_object_rate"] == pytest.approx(aggregate["score"])
    # 每个字段的分母都要看得见：只有 person 全对不足以解释总分
    assert all(entry["n"] > 0 for entry in aggregate["per_field"].values())


def test_every_grade_points_at_a_real_trace(env):
    """DoD：分数能下钻到真实 trace。"""
    _, sink, _, _ = env
    task, n = _task(), 6
    report = _runner(env, _answers(task, n, split="template"), task).run(
        RunConfig(model=MODEL, limit=n, split="template")
    )
    sink.flush(5.0)

    repo = TraceRepo(env[0])
    assert len(report.grades) == n
    for grade in report.grades:
        assert grade.trace_id, f"{grade.case_id} 没有 trace_id"
        trace = repo.get(grade.trace_id)
        assert trace is not None, f"{grade.trace_id} 在库里不存在"
        assert trace.eval_run_id == report.run_id
    assert len({g.trace_id for g in report.grades}) == n, "每条分数指向不同的 trace"


def test_a_model_that_extracts_nothing_gets_no_headline_score(env):
    """负样本不进主分数：全都输出 `{}` 时主分数是「没考到」，而不是 1.0。

    这条只能在真跑一遍时看得见——单测里很容易写成"分母是负样本条数"，
    于是"什么都不抽"被读成满分，而它恰恰是抽取任务里最贵的失败。
    """
    _, sink, _, _ = env
    task = _task()
    negatives = list(task.load(split="none"))
    assert negatives
    report = _runner(env, ["{}"] * len(negatives), task).run(
        RunConfig(model=MODEL, split="none")
    )
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["n_total"] == len(negatives)
    assert aggregate["n_judged"] == 0
    assert aggregate["score"] is None and aggregate["field_em"] is None
    assert aggregate["none_correct_rate"] == 1.0
    assert aggregate["exact_object_rate"] == 1.0, "端到端口径里它们确实是全对的"
    assert aggregate["hallucinated_fields"] == 0


def test_hallucinated_fields_are_visible_on_a_negative_run(env):
    _, sink, _, _ = env
    task = _task()
    negatives = list(task.load(split="none"))
    texts = ['{"person": ""}'] + ["{}"] * (len(negatives) - 1)
    report = _runner(env, texts, task).run(RunConfig(model=MODEL, split="none"))
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["none_correct_rate"] == pytest.approx((len(negatives) - 1) / len(negatives))
    assert aggregate["hallucinated_fields"] == 1
    assert aggregate["verdicts"].get(Verdict.WRONG.value) == 1


def test_the_exam_provenance_lands_with_the_run(env):
    """分数与考卷必须连得上：revision 不落地，换数据这件事在两次分数之间发现不了。"""
    _, sink, _, _ = env
    task = _task()
    report = _runner(env, _answers(task, 4, split="hard"), task).run(
        RunConfig(model=MODEL, limit=4, split="hard")
    )
    sink.flush(5.0)

    record = EvalRepo(env[0]).get_dataset(report.dataset_id)
    assert report.dataset_id == "structured_ie-v1"
    assert report.dataset_revision == "seed=20261003", (
        "跑完的 report 不带考卷来历，进程内调用方就只能去库里再查一遍——"
        "而字段声明的就是「每个结果都要能回答这是哪份数据考出来的」"
    )
    assert record is not None
    assert record.id == "structured_ie-v1"
    assert record.revision == "seed=20261003"
    assert record.license == "generated-in-repo"
    assert record.splits["none"] == 5 and record.splits["hard"] == 4
    stored = EvalRepo(env[0]).get_run(report.run_id)
    assert stored is not None and stored.task_id == "structured_extraction"
    assert json.dumps(stored.aggregate, ensure_ascii=False)
