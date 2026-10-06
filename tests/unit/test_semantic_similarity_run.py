"""S33 的第三条同源：`semantic_similarity` 在 mock 引擎上真跑一轮。

契约测试管"声明的指标 == 产出的指标"，这里管跑起来之后：
每个分数都能点进一条 **kind=embed** 的真实 trace、一条 case 恰好一个请求、
能力不满足时整场 skip，以及 k>1 时 `pass^k == pass@k`（向量检索本该确定性，
分叉了就是装置引入的非确定性，那要查通路不是查模型）。
"""

from __future__ import annotations

import json
import math

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.types import Cap
from onyx.eval.datasets.loader import load_builtin
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.task import Verdict
from onyx.eval.tasks.semantic_similarity import SemanticSimilarity
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "mock/embed"
#: 帧间角度 0.4 弧度；偏移都压在 0.15 以内 ⇒ 任何"别的帧的句子"离 query 都比 gold 远
SPACING = 0.4


def _unit(angle: float) -> tuple[float, float]:
    return (math.cos(angle), math.sin(angle))


def _table(cases, *, anti_wins: bool) -> dict[str, tuple[float, float]]:
    """按 frame 造一张全局 文本→向量 表。

    同一个句子在别的题里当"无关项"出现时用的是**同一个向量**（真实引擎就是这样，
    向量不依赖上下文），所以这张表必须自洽。
    """
    table: dict[str, tuple[float, float]] = {}
    for index, case in enumerate(cases):
        base = SPACING * index
        gold_text = case["input"]["candidates"][int(case["expect"]["gold"]) - 1]
        anti_text = case["input"]["candidates"][int(case["expect"]["antonym"]) - 1]
        table[str(case["input"]["query"])] = _unit(base)
        gold_offset, anti_offset = (0.12, 0.05) if anti_wins else (0.05, 0.15)
        table[gold_text] = _unit(base + gold_offset)
        table[anti_text] = _unit(base + anti_offset)
        for text in case["input"]["candidates"]:
            table.setdefault(str(text), _unit(base + 0.3))
    return table


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _run(env, *, anti_wins: bool = False, k: int = 1, limit: int | None = None, caps=None):
    _, _, observer, blobs = env
    dataset = load_builtin("embeddings_zh")
    cases = list(dataset.cases)[: (limit or len(dataset.cases))]
    provider = MockProvider(
        scripts={MODEL: []}, models=(MODEL,), embeddings=_table(cases, anti_wins=anti_wins)
    )
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())
    task = SemanticSimilarity(dataset, model=MODEL, k=3)
    runner = EvalRunner(gateway, EvalRepo(env[0]), task, dataset=dataset, clock=FakeClock(),
                        caps=caps if caps is not None else frozenset({Cap.EMBED}))
    return runner.run(RunConfig(model=MODEL, seed=42, k=k, limit=limit)), provider


def test_a_clean_run_scores_one_and_asks_once_per_case(env):
    report, provider = _run(env, limit=6)
    assert report.status == "done" and report.ok
    assert report.n_cases == 6 and report.n_done == 6 and report.n_error == 0

    aggregate = report.aggregate
    assert aggregate["score"] == 1.0 == aggregate["recall_at_1"]
    assert aggregate["anti_first_rate"] == 0.0 and aggregate["anti_above_gold_rate"] == 0.0
    assert aggregate["mrr"] == 1.0 and aggregate["recall_at_k"] == 1.0
    assert aggregate["requests"] == 6
    assert aggregate["inputs_total"] == 30 and aggregate["top_k"] == 3 and aggregate["k"] == 1
    assert aggregate["dimension"] == 2
    assert len(provider.embed_calls) == 6, "一条 case 必须恰好一个向量请求"
    assert {call.batch_size for call in provider.embed_calls} == {5}


def test_an_antonym_first_run_scores_zero_but_stays_inside_the_top_k(env):
    """反义句抢第 1：主分数为零，`anti_first_rate` 满格——这才是要看的那一位。"""
    report, _provider = _run(env, anti_wins=True, limit=6)
    aggregate = report.aggregate
    assert aggregate["score"] == 0.0 and aggregate["recall_at_1"] == 0.0
    assert aggregate["anti_first_rate"] == 1.0 and aggregate["anti_above_gold_rate"] == 1.0
    assert aggregate["recall_at_k"] == 1.0, "同义句仍在前 3：掉的是名次不是命中"
    assert aggregate["mrr"] == pytest.approx(0.5)
    assert {str(g.verdict) for g in report.grades} == {Verdict.PARTIAL.value}
    assert {(int(g.metrics["rank"]), int(g.metrics["rank_antonym"])) for g in report.grades} == {(2, 1)}


def test_every_grade_points_at_a_real_embed_trace(env):
    _, sink, _, _ = env
    report, _ = _run(env, limit=4)
    sink.flush(5.0)
    repo = TraceRepo(env[0])
    for grade in report.grades:
        trace = repo.get(grade.trace_id)
        assert trace is not None, f"{grade.case_id} 的分数点不进真实请求"
        assert str(trace.kind) == "embed"
        assert trace.eval_run_id == report.run_id and trace.case_id == grade.case_id


def test_unknown_token_counts_are_counted_not_invented(env):
    """mock 不报 token 数 ⇒ 成本里记"未知"，而不是把 0 当成实测值。"""
    report, _ = _run(env, limit=3)
    assert report.cost["requests"] == 3
    assert report.cost["in_tokens"] == 0
    assert report.cost["in_tokens_unknown"] == 3
    assert report.aggregate["reported_in_tokens"] == 0
    assert report.aggregate["mean_in_tokens"] is None


def test_missing_embed_capability_skips_the_whole_run(env):
    report, provider = _run(env, caps=frozenset({Cap.CHAT}))
    assert report.status == "skipped" and provider.embed_calls == []
    assert "embed" in (report.skip_reason or "").lower()


def test_repeated_sampling_agrees_with_itself(env):
    """k=2 时 pass^k 必须等于 pass@k：向量检索是确定性的，分叉说明通路引入了抖动。"""
    report, provider = _run(env, k=2, limit=3)
    assert len(provider.embed_calls) == 6, "每条题被采样两次"
    aggregate = report.aggregate
    assert aggregate["pass_hat_k"] == aggregate["pass_at_k"] == aggregate["recall_at_1"] == 1.0


def test_provenance_and_json_round_trip(env):
    report, _ = _run(env, limit=2)
    assert report.dataset_id == "embeddings_zh-v1"
    assert report.dataset_revision == "seed=20261006+frames=12+pool=4"
    stored = EvalRepo(env[0]).get_run(report.run_id)
    assert stored is not None and stored.task_id == "semantic_similarity"
    assert json.dumps(stored.aggregate, ensure_ascii=False)


def test_topic_splits_select_their_own_cases():
    """按话题跑子集要真能跑：某话题只 2 条时它的分数必须带着那个分母出门。"""
    task = SemanticSimilarity(load_builtin("embeddings_zh"), model=MODEL)
    topics = sorted({str(case.meta["topic"]) for case in task.load()})
    assert len(topics) == 6
    for topic in topics:
        subset = list(task.load(split=topic))
        assert subset and {str(c.meta["topic"]) for c in subset} == {topic}
