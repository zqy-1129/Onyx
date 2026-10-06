"""runner 按**请求类型**分派：`build()` 交出 `EmbedRequest` 就走 `gateway.embed`。

这条断言守的是"抽象装不装得下第二个任务族"（S33）：

- 分依据是 `isinstance(request, EmbedRequest)`，不是任务 id。
  写成 `if self.task.id == …` 的话，每加一类非生成任务都要改内核；
- 向量调用没有输出 token，所以成本判据必须按调用种类分开——
  沿用"进出都要有数"会把每条 embed 记成 `in_tokens_unknown`，
  新通路在自己的成本报表里就变成"什么都没测到"；
- 分数仍然必须能点进一条真实 trace（这条与生成任务同权重）。
"""

from __future__ import annotations

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.event import EventType, make_event
from onyx.core.types import (
    Cap,
    Embedding,
    EmbedRequest,
    Generation,
    GenerationRequest,
    Status,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.task import Case, Grade, Verdict
from onyx.llm.gateway import Gateway
from onyx.llm.providers.base import emit
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "stub/embed"


class _EmbedOnlyProvider:
    """只实现向量化：generate 一被调用就红，这样"走错喉咙"是测得出来的。"""

    id = "stub-embed"

    def __init__(self, *, in_tokens: int | None = 17) -> None:
        self.in_tokens = in_tokens
        self.embed_calls: list[EmbedRequest] = []

    def capabilities(self) -> frozenset[Cap]:
        return frozenset({Cap.EMBED})

    def list_models(self) -> list:
        return []

    def show_model(self, name: str):
        raise KeyError(name)

    def running(self) -> list:
        return []

    def generate(self, req: GenerationRequest, **kw) -> Generation:
        raise AssertionError("向量化任务不许走 generate（那会建立第二条调用路径）")

    def embed(self, req: EmbedRequest, *, trace_id: str = "", on_event=None) -> Embedding:
        self.embed_calls.append(req)
        if self.in_tokens is not None:
            emit(on_event, make_event(EventType.USAGE_ENGINE, trace_id, {
                "in_tokens": self.in_tokens, "out_tokens": None, "thinking_tokens": None,
                "cached_tokens": None, "ok": True, "note": "",
                "latency_ns": {"total": None, "load": None, "prompt_eval": None, "eval": None},
                "raw_keys": [],
            }))
        # 让"甲"总是最近邻：向量与文本的对应关系由这里显式钉住，可断言
        vectors = tuple((1.0, 0.0) if t == "甲" else (0.0, 1.0) for t in req.inputs)
        return Embedding(vectors=vectors, model=req.model, status=Status.OK)


class _RankTask:
    """最小可用的排序任务：query 与候选池打成一个 EmbedRequest，gold 排第几就是分数。"""

    id = "rank_stub"
    name = "排序桩任务"
    requires = frozenset({Cap.EMBED})
    metric_names = ("n_cases", "recall_at_1")

    def __init__(self, dataset: Dataset, *, model: str) -> None:
        self.dataset = dataset
        self.model = model

    def load(self, *, split: str = "default", limit: int | None = None):
        for raw in self.dataset.select(split=split or "default", limit=limit):
            yield Case(
                id=str(raw["id"]), input=dict(raw["input"]), expect=dict(raw["expect"]),
                dataset_id=self.dataset.id, ord=int(raw.get("ord") or 0),
                kind=str(raw.get("kind") or "rank"),
            )

    def build(self, case) -> EmbedRequest:
        texts = [case.input["query"], *case.input["candidates"]]
        return EmbedRequest(model=self.model, inputs=tuple(texts))

    def grade(self, case, sample) -> Grade:
        assert isinstance(sample, Embedding), "runner 送来的样本类型必须与请求类型一致"
        if sample.status is not Status.OK:
            return Grade(case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                         error=sample.error or "engine 未返回向量")
        query, *pool = sample.vectors

        def cos(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        sims = sorted((cos(query, v), i) for i, v in enumerate(pool, start=1))
        ranked = [i for _s, i in reversed(sims)]
        gold_hit = ranked[0] == int(case.expect["gold"])
        return Grade(
            case_id=case.id, score=1.0 if gold_hit else 0.0,
            verdict=Verdict.CORRECT if gold_hit else Verdict.WRONG, passed=gold_hit,
            metrics={"rank": ranked.index(int(case.expect["gold"])) + 1, "recall_at_1": gold_hit},
        )

    def aggregate(self, grades, *, seed: int = 0) -> dict:
        judged = [g for g in grades if g.attributable]
        hits = sum(1 for g in judged if g.verdict is Verdict.CORRECT)
        return {"n_cases": len(grades),
                "recall_at_1": (hits / len(judged)) if judged else None}


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _dataset() -> Dataset:
    cases = [
        {"id": "r-0", "ord": 0, "kind": "rank",
         "input": {"query": "甲", "candidates": ["甲", "乙"]},
         "expect": {"gold": 1}, "tags": [], "meta": {}},
        {"id": "r-1", "ord": 1, "kind": "rank",
         "input": {"query": "乙", "candidates": ["乙", "甲"]},
         "expect": {"gold": 1}, "tags": [], "meta": {}},
    ]
    return Dataset(id="rankstub", cases=tuple(cases), upstream="test", revision="r1")


def _runner(env, provider, *, caps=None):
    _, _, observer, blobs = env
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())
    data = _dataset()
    task = _RankTask(data, model=MODEL)
    return EvalRunner(gateway, EvalRepo(env[0]), task, dataset=data, clock=FakeClock(),
                      caps=caps if caps is not None else frozenset({Cap.EMBED}))


def test_embed_request_goes_through_the_embed_entry(env):
    db, sink, _, _ = env
    provider = _EmbedOnlyProvider()
    report = _runner(env, provider).run(RunConfig(model=MODEL))
    sink.flush(5.0)

    assert report.status == "done" and report.ok
    assert report.n_done == 2, "provider.generate 被调用会直接抛 AssertionError"
    assert len(provider.embed_calls) == 2
    assert provider.embed_calls[0].inputs == ("甲", "甲", "乙"), "query 与候选打成一批"
    assert report.aggregate["recall_at_1"] == 1.0

    #: 分数还是必须能点进真实 trace，而且这条 trace 是 embed 而不是 generation
    repo = TraceRepo(db)
    for grade in report.grades:
        trace = repo.get(grade.trace_id)
        assert trace is not None and str(trace.kind) == "embed"
        assert trace.eval_run_id == report.run_id and trace.case_id == grade.case_id


def test_cost_does_not_demand_output_tokens_from_an_embed_call(env):
    """向量没有输出 token：成本口径要按调用种类分开，否则新通路会被记成"全没测到"。"""
    _, sink, _, _ = env
    report = _runner(env, _EmbedOnlyProvider(in_tokens=17)).run(RunConfig(model=MODEL))
    sink.flush(5.0)
    assert report.cost["requests"] == 2
    assert report.cost["in_tokens"] == 34 and report.cost["in_tokens_unknown"] == 0
    #: 不谎报：embed 的 out_tokens 既不是 17 也不是"未知"，它就是这个调用没有输出
    assert report.cost["out_tokens"] == 0

    unknown = _runner(env, _EmbedOnlyProvider(in_tokens=None)).run(RunConfig(model=MODEL))
    assert unknown.cost["in_tokens_unknown"] == 2 and unknown.cost["in_tokens"] == 0


def test_capability_gap_skips_the_whole_run_with_a_reason(env):
    """没有 EMBED 能力就整场 skip 并写明原因——不许改成"用 chat 拼一个假向量"。"""
    provider = _EmbedOnlyProvider()
    report = _runner(env, provider, caps=frozenset({Cap.CHAT})).run(RunConfig(model=MODEL))
    assert provider.embed_calls == [], "能力不满足时一条都不该发出去"
    assert report.status == "skipped"
    assert "embed" in (report.skip_reason or "").lower() or "EMBED" in (report.skip_reason or "")


def test_engine_failure_on_one_case_does_not_kill_the_run(env):
    """单条 embed 抛错只判那一条 ERROR：已经花掉的 GPU 时间不能连带丢掉别的样本。"""

    class _Boom(_EmbedOnlyProvider):
        def embed(self, req, *, trace_id="", on_event=None):
            if req.inputs[0] == "甲":
                raise RuntimeError("engine down")
            return super().embed(req, trace_id=trace_id, on_event=on_event)

    _, sink, _, _ = env
    report = _runner(env, _Boom()).run(RunConfig(model=MODEL))
    sink.flush(5.0)
    verdicts = [str(g.verdict) for g in report.grades]
    assert verdicts.count("error") == 1 and len(verdicts) == 2
    assert report.n_error == 1 and report.status == "done"
    survived = next(g for g in report.grades if str(g.verdict) != "error")
    assert survived.score == 1.0, "另一条照常判分：一条引擎故障不许污染别的样本"
    assert next(g for g in report.grades if str(g.verdict) == "error").trace_id == "",         "没跑成就没有 trace，不许留一个指点不存在 trace 的 id"
    #: 出错那条也要留下 stage=embed，否则"哪一步崩的"只能靠猜
    assert next(g for g in report.grades if str(g.verdict) == "error").extra["stage"] == "embed"
