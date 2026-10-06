"""`semantic_similarity` 的判分与聚合（S33）。

要紧的是**四种形状必须长成四种不同的样子**：同义句抢第 1、反义句抢第 1、
无关句抢第 1、引擎没给够向量。把它们混成一个"0 分"，
下一步就会被拿去做错的决定——尤其"反义抢第 1"，那是要查模型而不是查通路。

三条口径同时看：`score`（= recall@1，最严）、`recall_at_k`（宽一档）、`mrr`（位置加权），
再加 `anti_first_rate` / `anti_above_gold_rate`——后者才是 embedding 模型的真短板。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from onyx.core.types import (
    Confidence,
    Embedding,
    Status,
    TokenSample,
    TokenSource,
)
from onyx.eval.datasets.loader import Dataset, load_builtin
from onyx.eval.task import Verdict
from onyx.eval.tasks.semantic_similarity import SemanticSimilarity

MODEL = "mock/seme"
POOL = 4


def _cases() -> list[dict[str, Any]]:
    """两条题：池子 4 项，gold/anti 位置不同（免得"总在第 2 位"这种巧合被当成判据）。"""
    return [
        {
            "id": "em-a", "ord": 0, "kind": "embed",
            "input": {"query": "评审会推迟到下午三点",
                      "candidates": ["评审会延后至下午三点", "主泵需要每周巡检",
                                     "评审会没有推迟到下午三点", "冷却塔达到合格标准"]},
            "expect": {"gold": 1, "antonym": 3},
            "tags": ["会议"], "meta": {"topic": "会议", "frame": "mtg-delay"},
        },
        {
            "id": "em-b", "ord": 1, "kind": "embed",
            "input": {"query": "这张单据通过财务复核",
                      "candidates": ["差旅额度超出上限", "这张单据获批财务复核",
                                     "这批图纸已送交档案室", "这张单据未通过财务复核"]},
            "expect": {"gold": 2, "antonym": 4},
            "tags": ["报销"], "meta": {"topic": "报销", "frame": "fin-approve"},
        },
    ]


def _task(**kw: Any) -> SemanticSimilarity:
    dataset = Dataset(id="seme-tiny-v1", cases=tuple(_cases()), upstream="test", revision="r1")
    return SemanticSimilarity(dataset, model=MODEL, **kw)


def _vectors(order: list[int], *, n: int = POOL) -> tuple[tuple[float, float], ...]:
    """query 固定在角度 0，候选按 `order`（池内 1 基序号 → 名次）排开角度。"""
    #: `order[i]` 是"第 i+1 名"对应的池内序号（1 基），所以直接给每个序号一个递减的角度
    by_index = {idx: 0.2 * rank for rank, idx in enumerate(order, start=1)}
    return ((1.0, 0.0), *(
        (math.cos(by_index[i]), math.sin(by_index[i])) for i in range(1, n + 1)
    ))


def _sample(task: SemanticSimilarity, case_id: str, order: list[int], **kw: Any) -> Embedding:
    case = next(c for c in task.load() if c.id == case_id)
    inputs = task.build(case).inputs
    assert len(inputs) == len(order) + 1, "query + 候选池"
    in_tokens = kw.pop("in_tokens", 24)
    usage = () if in_tokens is None else (TokenSample(
        source=TokenSource.ENGINE, in_tokens=in_tokens, out_tokens=None,
        confidence=Confidence.HIGH),)
    return Embedding(vectors=_vectors(order), model=MODEL, status=kw.pop("status", Status.OK),
                     usage=usage, **kw)


# ── 形态表 ─────────────────────────────────────────────────────────
def test_paraphrase_first_is_correct_and_scores_one():
    task = _task()
    grade = task.grade(next(task.load()), _sample(task, "em-a", [1, 4, 3, 2]))
    assert grade.verdict is Verdict.CORRECT and grade.passed is True and grade.score == 1.0
    assert grade.metrics["rank"] == 1 and grade.metrics["rank_antonym"] == 3
    assert grade.metrics["sim_gap"] > 0
    #: 界面要能一眼看完"期望/预测"（S30 的教训：dict 直接渲染成 [object Object]）
    assert grade.metrics["expected"] == "评审会延后至下午三点"
    assert grade.metrics["predicted"] == "评审会延后至下午三点"


def test_antonym_first_is_zero_but_still_inside_the_top_k():
    """反义句抢第 1：主分数为 0，但它是 PARTIAL 而不是 WRONG——同义句还在前 k 里。

    这两种失败的修法不同：PARTIAL 说明"读到了但排不对"，WRONG 说明"根本没找到"。
    """
    task = _task()
    grade = task.grade(next(task.load()), _sample(task, "em-a", [3, 1, 4, 2]))
    assert grade.verdict is Verdict.PARTIAL and grade.passed is False and grade.score == 0.0
    assert grade.metrics["rank"] == 2 and grade.metrics["rank_antonym"] == 1
    assert grade.metrics["sim_gap"] < 0
    aggregate = task.aggregate([grade])
    assert aggregate["score"] == 0.0 and aggregate["recall_at_k"] == 1.0
    assert aggregate["anti_first_rate"] == 1.0 and aggregate["anti_above_gold_rate"] == 1.0
    assert aggregate["mrr"] == pytest.approx(0.5)


def test_paraphrase_beyond_k_is_wrong_not_partial():
    task = _task(k=2)
    grade = task.grade(next(task.load()), _sample(task, "em-a", [2, 3, 4, 1]))
    assert grade.metrics["rank"] == 4
    assert grade.verdict is Verdict.WRONG and grade.passed is False
    assert task.aggregate([grade])["recall_at_k"] == 0.0


def test_unrelated_first_is_a_plain_miss_without_antonym_blame():
    """无关句抢第 1 而反义句排在同义句之后 ⇒ 掉分不许记账到"反义抢占"头上。

    这两个数是分开归因的：`anti_first_rate` 说的是模型被否定词骗了，
    这一条说的是它压根没找到同义句——修法完全不同。
    """
    task = _task()
    grade = task.grade(next(task.load()), _sample(task, "em-a", [2, 4, 1, 3]))
    assert grade.metrics["rank"] == 3 and grade.metrics["rank_antonym"] == 4
    aggregate = task.aggregate([grade])
    assert aggregate["anti_first_rate"] == 0.0 and aggregate["anti_above_gold_rate"] == 0.0
    assert aggregate["sim_gap"] > 0, "同义句比反义句近，gap 必须是正的"


def test_ties_are_broken_by_position_so_the_rank_is_reproducible():
    """并列时按池内序号小者优先：不写死规则的话，同一份数据两次能跑出不同名次。"""
    task = _task()
    case = next(task.load())
    #: gold(序号 1) 与无关项(序号 2) 完全同角度 ⇒ 名次由序号决定
    tied = ((1.0, 0.0), (0.9, 0.1), (0.9, 0.1), (0.0, 1.0), (-0.2, 0.9))
    assert task.grade(case, Embedding(vectors=tied, model=MODEL)).metrics["rank"] == 1
    #: 把无关项抬到严格更高 ⇒ 不再并列，gold 退到第 2。规则可复现，不是巧合
    ahead = ((1.0, 0.0), (0.9, 0.1), (0.95, 0.05), (0.0, 1.0), (-0.2, 0.9))
    assert task.grade(case, Embedding(vectors=ahead, model=MODEL)).metrics["rank"] == 2


def test_engine_failure_is_error_and_leaves_the_denominator():
    task = _task()
    grade = task.grade(next(task.load()), Embedding(
        model=MODEL, status=Status.ERROR, error="连接被重置"))
    assert grade.verdict is Verdict.ERROR and not grade.attributable
    aggregate = task.aggregate([grade])
    assert aggregate["n_total"] == 1 and aggregate["n_attributable"] == 0
    assert aggregate["score"] is None and aggregate["mrr"] is None


def test_short_vector_count_is_an_error_naming_both_numbers():
    """少一条向量 ⇒ 排名不可信。这里报错而不是猜，因为错位的排名看着完全正常。"""
    task = _task()
    case = next(task.load())
    broken = Embedding(vectors=_vectors([1, 2, 3], n=3), model=MODEL, status=Status.OK)
    grade = task.grade(case, broken)
    assert grade.verdict is Verdict.ERROR
    assert "4" in grade.error and "5" in grade.error
    assert not grade.attributable


# ── 聚合的口径 ─────────────────────────────────────────────────────
def test_score_is_recall_at_1_and_the_ci_follows_it():
    task = _task()
    grades = [
        task.grade(next(task.load()), _sample(task, "em-a", [1, 3, 4, 2])),
        task.grade(next(task.load()), _sample(task, "em-a", [3, 1, 4, 2])),
    ]
    aggregate = task.aggregate(grades)
    assert aggregate["score"] == aggregate["recall_at_1"] == 0.5
    assert aggregate["score_ci"]["point"] == pytest.approx(aggregate["score"])
    assert aggregate["score_ci"]["n"] == aggregate["n_judged"] == 2


def test_by_topic_carries_its_own_denominator():
    task = _task()
    cases = list(task.load())
    grades = [
        task.grade(cases[0], _sample(task, "em-a", [1, 3, 4, 2])),
        task.grade(cases[1], _sample(task, "em-b", [2, 1, 4, 3])),
    ]
    by_topic = task.aggregate(grades)["by_topic"]
    assert {topic: entry["cases"] for topic, entry in by_topic.items()} == {"会议": 1, "报销": 1}
    for entry in by_topic.values():
        assert set(entry) == {"cases", "recall_at_1", "mrr", "anti_above_rate"}


def test_token_accounting_only_counts_engine_numbers():
    task = _task()
    case = next(iter(task.load()))
    graded = task.grade(case, _sample(task, "em-a", [1, 3, 4, 2], in_tokens=24))
    missing = task.grade(case, _sample(task, "em-a", [1, 3, 4, 2], in_tokens=None))
    aggregate = task.aggregate([graded, missing])
    assert aggregate["reported_in_tokens"] == 1, "只有引擎回报的那条能进成本"
    assert aggregate["mean_in_tokens"] == 24
    assert aggregate["requests"] == 2 and aggregate["inputs_total"] == 10
    assert aggregate["dimension"] == 2


def test_declared_metrics_match_empty_and_real_aggregates():
    task = _task()
    grades = [task.grade(next(iter(task.load())), _sample(task, "em-a", [1, 3, 4, 2]))]
    assert set(task.aggregate([])) == set(task.metric_names)
    assert set(task.aggregate(grades)) == set(task.metric_names)


def test_scoring_is_labelled_rank_based_not_gen_based():
    """口径名字要说真话：这个分数不是"生成对不对"，是"排第几"。"""
    task = _task()
    assert task.aggregate([])["scoring"] == "rank-based"


def test_real_dataset_builds_one_request_per_case():
    """一条 case 一个请求：池子大小决定批次形状，query 固定在第 0 位（向量按序对齐）。"""
    task = SemanticSimilarity(load_builtin("embeddings_zh"), model=MODEL)
    cases = list(task.load())
    assert len(cases) == 12
    for case in cases:
        request = task.build(case)
        assert request.inputs[0] == str(case.expect["query"])
        assert list(request.inputs[1:]) == list(case.expect["candidates"])
        assert request.batch_size == len(case.meta["texts"]) + 1
