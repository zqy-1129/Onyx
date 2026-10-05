"""长上下文任务在 mock 引擎上真跑（S32 第三条同源）。

契约测试管声明与产出同源，这里管跑起来之后的四件事：
- 每条分数都能下钻到真实 trace（长上下文的 trace 特别值钱：它带着 in_tokens 与窗口）；
- 窗口是**请求里带出去的**，并且 `window_tokens` 与 `ctx_util` 在聚合里自洽；
- 引擎回报的 in_tokens **低于正文的汉字下限**时走 skip，不进分母也不进分子——
  一个"16k 全错"的 run 与一个"16k 没测成"的 run 必须在数字上长得不一样
  （真机上这就是那条判据的由来：`--num-ctx 4096` 时引擎回报 2050 tok，比窗口还小，
   只看"超没超窗口"就永远不会跳，于是 0 分挂在了配置头上）；
- mock 引擎回报的数按汉字数折算（`_reported`）：写死一个小数字会让健康样本全被判成截断，
  那不是任务对了，是装置错了。
"""

from __future__ import annotations

import json

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.eval.datasets.loader import load_builtin
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.task import Verdict
from onyx.eval.tasks.long_context import LongContext
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "mock/lc"


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _task(**kw) -> LongContext:
    return LongContext(load_builtin("longctx_zh"), model=MODEL, **kw)


def _runner(env, scripts: list[MockScript], task: LongContext) -> EvalRunner:
    _, _, observer, blobs = env
    provider = MockProvider(scripts={MODEL: scripts}, models=(MODEL,))
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())
    return EvalRunner(gateway, EvalRepo(env[0]), task, dataset=task.dataset, clock=FakeClock())


#: 真机实测比例（qwen3.5:9b，2026-10-05：16k 档 24,754 汉字 ⇒ 引擎回报 16,755 tok）
TOK_PER_HANZI = 0.68


def _reported(case) -> int:
    """mock 引擎该回报的 in_tokens：按正文汉字数折算，而不是随手写一个 1500。

    刻意让它大于任务的"汉字下限"（0.5 tok/字）：低于下限会被判成正文被裁 ⇒ skip。
    写死一个假数字会让每条健康样本都走进截断分支，那才是真的没测到东西。
    """
    return int(int(case.meta["hanzi"]) * TOK_PER_HANZI)


def _perfect(task: LongContext, n: int, *, in_tokens: int | None = None) -> list[MockScript]:
    return [
        MockScript(text=json.dumps({item["id"]: item["value"] for item in case.meta["needles"]},
                                   ensure_ascii=False),
                   done_reason="stop", in_tokens=in_tokens or _reported(case), out_tokens=18)
        for case in task.load(limit=n)
    ]


def test_a_real_run_carries_the_window_and_the_denominators(env):
    _, sink, _, _ = env
    task, n = _task(num_ctx=20480), 6
    cases = list(task.load(limit=n))
    report = _runner(env, _perfect(task, n), task).run(RunConfig(model=MODEL, seed=42, limit=n))
    sink.flush(5.0)

    assert report.status == "done" and report.ok
    assert report.n_cases == n and report.n_done == n and report.n_error == 0

    aggregate = report.aggregate
    expected = [_reported(case) for case in cases]
    assert aggregate["n_total"] == aggregate["n_judged"] == n
    assert aggregate["score"] == 1.0 and aggregate["needle_rate"] == 1.0
    assert aggregate["needle_total"] == 3 * n
    assert aggregate["window_tokens"] == 20480
    assert aggregate["reported_in_tokens"] == n, "每条都要拿到引擎回报的 in_tokens"
    assert aggregate["mean_in_tokens"] == round(sum(expected) / len(expected))
    assert aggregate["max_ctx_util"] == pytest.approx(max(expected) / 20480, abs=1e-3)
    assert aggregate["n_truncated"] == 0
    assert aggregate["verdicts"] == {Verdict.CORRECT.value: n}
    assert aggregate["low_confidence"] is True


def test_a_window_too_small_to_hold_the_document_is_reported_as_not_measured(env):
    """窗口装不下正文 ⇒ 分数必须是「没测到」而不是 0 分。

    这条是 S32 的立足点，而它的**触发方式**是真机教我的：`--num-ctx 4096` 跑 16k 档时
    Ollama 把正文裁到 in_tokens=2050（小于窗口），"≥窗口"那条判据永远不会响，
    于是三条被判成 partial、score 0.000 —— 让人以为该换模型，其实该调窗口。
    这里用 1,500 tok 重现那个形状：引擎给的数低于正文的汉字下限 ⇒ 一定被切过。
    """
    _, sink, _, _ = env
    task, n = _task(num_ctx=1024), 6
    report = _runner(env, _perfect(task, n, in_tokens=1500), task).run(
        RunConfig(model=MODEL, limit=n)
    )
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["n_total"] == n and aggregate["n_attributable"] == 0
    assert aggregate["n_judged"] == 0 and aggregate["n_truncated"] == n
    assert aggregate["score"] is None and aggregate["needle_rate"] is None
    assert aggregate["by_position"] == {} and aggregate["by_bucket"] == {}
    assert aggregate["verdicts"][Verdict.SKIPPED.value] == n
    # 窗口占用率仍把这些算进来：它就是"该调 --num-ctx"的证据
    assert aggregate["max_ctx_util"] == pytest.approx(1500 / 1024, abs=1e-3)
    assert report.n_skipped == n
    assert all("num-ctx" in (grade.error or "") for grade in report.grades)
    # 每条grade都要能自证"我是被切的，不是不会"：下限数字必须留在指标里
    for grade in report.grades:
        assert grade.metrics["truncated"] is True
        assert grade.metrics["min_prompt_tokens"] > grade.metrics["in_tokens"]


def test_every_grade_points_at_a_real_trace_with_the_window(env):
    _, sink, _, _ = env
    task, n = _task(), 4
    cases = list(task.load(limit=n))
    report = _runner(env, _perfect(task, n), task).run(RunConfig(model=MODEL, limit=n))
    sink.flush(5.0)

    repo = TraceRepo(env[0])
    assert len(report.grades) == n
    by_id = {case.id: case for case in cases}
    for grade in report.grades:
        trace = repo.get(grade.trace_id)
        assert trace is not None, f"{grade.case_id} 的分数点不进真实请求"
        assert trace.eval_run_id == report.run_id
        # 长上下文里 trace 的独特价值：窗口占用率与 in_tokens 都能从分数走到证据
        assert grade.metrics["in_tokens"] == _reported(by_id[grade.case_id])
        assert grade.metrics["bucket"] in {"4k", "8k", "16k"}


def test_by_position_survives_a_mixed_run(env):
    """一半只答末部、一半全答 ⇒ 主分数 0.5、逐埋点 12/18、last 1.0 / first 0.5。

    这四个数一起才说清"掉的是哪一段没读到"；任何单个数都不行。
    """
    _, sink, _, _ = env
    task = _task()
    cases = list(task.load(limit=6))
    scripts = [
        MockScript(text=json.dumps({item["id"]: item["value"] for item in case.meta["needles"]}
                                   if index % 2 else {"q3": case.meta["needles"][2]["value"]},
                                   ensure_ascii=False),
                   done_reason="stop", in_tokens=_reported(case), out_tokens=18)
        for index, case in enumerate(cases)
    ]
    report = _runner(env, scripts, task).run(RunConfig(model=MODEL, limit=6))
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["score"] == pytest.approx(0.5)
    assert aggregate["needle_rate"] == pytest.approx(12 / 18)
    by_position = aggregate["by_position"]
    assert {slot: entry["n"] for slot, entry in by_position.items()} == {
        "first": 6, "middle": 6, "last": 6
    }
    assert by_position["last"]["rate"] == 1.0
    assert by_position["first"]["rate"] == 0.5
    assert by_position["middle"]["rate"] == 0.5


def test_position_asymmetry_is_visible_while_the_total_score_is_not(env):
    """一条"只找得到开头与结尾"的模型：`needle_rate 0.667` 听着还行，塌陷只在分桶里看得见。

    这就是这任务存在的理由——逐埋点命中率 0.667 与 first/middle/last 的 1.0 / 0.0 / 1.0
    说的是两个不同的故事，而只有后者能指导你改文档排布。
    """
    _, sink, _, _ = env
    task = _task()
    cases = list(task.load(limit=3))
    scripts = [
        MockScript(text=json.dumps({item["id"]: item["value"] for item in case.meta["needles"]
                                    if item["position"] != "middle"}, ensure_ascii=False),
                   done_reason="stop", in_tokens=_reported(case), out_tokens=18)
        for case in cases
    ]
    report = _runner(env, scripts, task).run(RunConfig(model=MODEL, limit=3))
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["needle_rate"] == pytest.approx(2 / 3)
    assert aggregate["score"] == 0.0, "没有一条题全找到 ⇒ 主分数如实为零"
    assert aggregate["by_position"] == {
        "first": {"matched": 3, "confused": 0, "n": 3, "rate": 1.0},
        "middle": {"matched": 0, "confused": 0, "n": 3, "rate": 0.0},
        "last": {"matched": 3, "confused": 0, "n": 3, "rate": 1.0},
    }


def test_answering_the_real_distractors_reads_as_confusion(env):
    """把真数据集里的干扰值答进去：0 分，但 9 个埋点全是"认错实体"。

    这一条同时证明两件事：干扰项真的进了正文（不只是写在 meta 里），
    以及"掉分"与"认错实体"在数字上长得不一样——后者才是加干扰项的目的。
    """
    _, sink, _, _ = env
    task = _task()
    cases = list(task.load(limit=3))
    scripts = [
        MockScript(text=json.dumps(
            {item["id"]: item["distractor_value"] for item in case.meta["needles"]},
            ensure_ascii=False), done_reason="stop", in_tokens=_reported(case), out_tokens=18)
        for case in cases
    ]
    report = _runner(env, scripts, task).run(RunConfig(model=MODEL, limit=3))
    sink.flush(5.0)

    aggregate = report.aggregate
    assert aggregate["score"] == 0.0 and aggregate["needle_matched"] == 0
    assert aggregate["needle_confused"] == 9 == aggregate["needle_total"]
    assert aggregate["confusion_rate"] == 1.0
    assert all(entry["confused"] == 3 for entry in aggregate["by_position"].values())


def test_provenance_and_json_round_trip(env):
    _, sink, _, _ = env
    task = _task()
    report = _runner(env, _perfect(task, 3), task).run(RunConfig(model=MODEL, limit=3))
    sink.flush(5.0)

    assert report.dataset_id == "longctx_zh-v1"
    assert report.dataset_revision == "seed=20261005+budgets=6000,12000,24000+distractors=yes"
    record = EvalRepo(env[0]).get_dataset(report.dataset_id)
    assert record is not None and record.splits["16k"] == 3
    stored = EvalRepo(env[0]).get_run(report.run_id)
    assert stored is not None and stored.task_id == "long_context"
    assert json.dumps(stored.aggregate, ensure_ascii=False)
