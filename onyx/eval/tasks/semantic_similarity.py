"""中文语义检索任务（M11 / S33）：`Cap.EMBED` 的第一个消费者。

一句话：**同义句在候选池里排第几，以及反义句有没有抢到第 1**。

四个与生成任务不同的判断，全部来自 2026-10-06 的真机实测（qwen3-embedding:0.6b）：

1. **判据是排序，不是相似度阈值**。同义对 cos 实测 0.749–0.909、反义对 0.649–0.796，
   **区间重叠**——"离得近"完全可能是"意思相反"。任何"≥0.8 算相似"都是在拍脑袋，
   换个模型那个阈值就得重拍，而分数看起来还是同一个口径。
2. **反义项与 query 几乎同词面**（"评审会推迟到下午三点" vs "评审会没有推迟到下午三点"），
   所以靠词面重叠的模型会把它排第一。`anti_first_rate` / `anti_above_gold_rate`
   单独成指标：这一位比 recall@1 更能分出模型间高下。
3. **一条 case 一个请求**：query 与候选打成一批。实测批次不改变向量
   （单条与混在 20 条一批里 cos = 1.0），所以批量只是把请求数降下来，不是"更快"。
4. **向量没有输出 token**，成本侧要按调用种类分开（runner 里做）；这里只统计引擎给的 in_tokens。

`pass_hat_k` 与 `pass_at_k` 在这个任务上多了一层用途：向量检索本该是确定性的，
如果 k>1 时两者分叉，说明**引擎或通路引入了非确定性**（实测同一批重复调用有 ≤3e-4 的抖动），
那是要查装置而不是查模型。
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

from onyx.core.types import (
    Cap,
    Embedding,
    EmbedRequest,
    Status,
    TokenSource,
    TraceContext,
    TracePurpose,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.metrics import (
    LOW_CONFIDENCE_N,
    mean_ci,
    pass_at_k,
    pass_hat_k,
    rate,
    stability_gap,
)
from onyx.eval.task import Case, Grade, Verdict

#: 与"同义句排第 1"分开的第二档：gold 进前 k 即算命中，衡量"读到了但没排第一"的那部分
DEFAULT_K = 3

class SemanticSimilarity:
    """`EvalTask` 协议的实现（第一个 `build()` 不返回 GenerationRequest 的任务）。"""

    id = "semantic_similarity"
    name = "中文语义检索（同义/反义）"
    requires = frozenset({Cap.EMBED})
    metric_names = (
        "n_total", "n_attributable", "n_judged", "verdicts",
        "score", "score_ci", "recall_at_1", "recall_at_k", "top_k", "k", "mrr",
        "anti_first_rate", "anti_above_gold_rate",
        "mean_gold_sim", "mean_antonym_sim", "mean_unrelated_sim", "sim_gap",
        "by_topic", "dimension", "inputs_total", "requests",
        "mean_in_tokens", "reported_in_tokens",
        "pass_hat_k", "pass_at_k", "stability_gap", "low_confidence", "scoring",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        k: int = DEFAULT_K,
        split: str = "default",
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.k = max(1, int(k))
        self.split = split

    # ── 契约实现 ──────────────────────────────────────────────────
    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]:
        for raw in self.dataset.select(split=split or self.split, limit=limit):
            expect = dict(raw.get("expect") or {})
            yield Case(
                id=str(raw["id"]), input=dict(raw.get("input") or {}),
                expect={
                    "query": str(raw.get("input", {}).get("query") or ""),
                    "candidates": [str(t) for t in raw.get("input", {}).get("candidates") or ()],
                    "gold": int(expect.get("gold") or 0),
                    "antonym": int(expect.get("antonym") or 0),
                },
                dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "embed"),
                meta=dict(raw.get("meta") or {}), tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> EmbedRequest:
        """query 打头，候选跟后：一条 case 一个请求，池大小决定批次形状。"""
        return EmbedRequest(
            model=self.model,
            inputs=(str(case.expect["query"]), *[str(t) for t in case.expect["candidates"]]),
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Embedding) -> Grade:
        query = str(case.expect["query"])
        pool = [str(t) for t in case.expect["candidates"]]
        gold, anti = int(case.expect["gold"]), int(case.expect["antonym"])
        base: dict[str, Any] = {
            "pool_size": len(pool), "query": query, "topic": str(case.meta.get("topic") or ""),
            "in_tokens": _engine_in_tokens(sample),
            "dimension": sample.dimension,
            "expected": pool[gold - 1] if 0 < gold <= len(pool) else "",
        }
        if sample.status is not Status.OK:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}", metrics=base,
            )
        expected_n = 1 + len(pool)
        if len(sample.vectors) != expected_n:
            #: 条数不对 ⇒ 排名全部不可信。这里不猜，直接判 ERROR
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=(f"拿到 {len(sample.vectors)} 条向量，需要 {expected_n} 条"
                       "（query + 候选池），排名不可信"),
                metrics={**base, "returned_vectors": len(sample.vectors)},
            )

        query_vector, *pool_vectors = sample.vectors
        sims = [round(_cosine(query_vector, vector), 6) for vector in pool_vectors]
        #: 排名口径：相似度**降序**，同分时按池内序号小者优先（并列不是罕见情况，
        #: 而是一个会导致"同一份数据两次跑出不同名次"的口子，所以必须写死规则）
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], i))
        rank = order.index(gold - 1) + 1
        rank_anti = order.index(anti - 1) + 1
        unrelated = [s for i, s in enumerate(sims) if (i + 1) not in (gold, anti)]
        hit = rank == 1
        return Grade(
            case_id=case.id, score=1.0 if hit else 0.0,
            verdict=(Verdict.CORRECT if hit else Verdict.PARTIAL if rank <= self.k
                     else Verdict.WRONG),
            passed=hit,
            metrics={
                **base, "predicted": pool[order[0]],
                "sims": sims, "gold": gold, "antonym": anti,
                "rank": rank, "rank_antonym": rank_anti,
                "gold_sim": sims[gold - 1], "antonym_sim": sims[anti - 1],
                "unrelated_sims": unrelated,
                "sim_gap": round(sims[gold - 1] - sims[anti - 1], 6),
                "mrr": round(1.0 / rank, 6),
            },
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        attributable = [g for g in grades if g.attributable]
        judged = [g for g in attributable if g.metrics.get("rank") is not None]
        counts: dict[str, int] = {}
        for grade in grades:
            counts[str(grade.verdict)] = counts.get(str(grade.verdict), 0) + 1

        ranks = [int(g.metrics["rank"]) for g in judged]
        flags = [1.0 if rank == 1 else 0.0 for rank in ranks]
        anti_first = sum(1 for g in judged if int(g.metrics["rank_antonym"]) == 1)
        tokens = [int(g.metrics["in_tokens"]) for g in attributable if g.metrics.get("in_tokens")]
        stability_k, stability_hat, stability_at, gap = _stability(judged)

        def _mean(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 6) if values else None

        return {
            "n_total": len(grades),
            "n_attributable": len(attributable),
            "n_judged": len(judged),
            "verdicts": counts,
            #: 主分数就是 recall@1（同义句抢到第 1 的题占比）；
            #: 两个名字都留着是因为"这个模型语义检索 1.000"这句话必须能说出是哪个分母
            "score": rate(int(sum(flags)), len(judged)),
            "score_ci": mean_ci(flags, seed=seed).as_dict(),
            "recall_at_1": rate(int(sum(flags)), len(judged)),
            "recall_at_k": rate(sum(1 for r in ranks if r <= self.k), len(judged)),
            #: `top_k` 是本任务的判据档位；`k` 是**采样次数**（与其他任务同义，别混用）
            "top_k": self.k,
            "k": stability_k,
            "mrr": _mean([float(g.metrics["mrr"]) for g in judged]),
            #: 反义句抢走第 1 位：这才是 embedding 模型的真短板（词面几乎全同）
            "anti_first_rate": rate(anti_first, len(judged)),
            "anti_above_gold_rate": rate(
                sum(1 for g in judged
                    if int(g.metrics["rank_antonym"]) < int(g.metrics["rank"])), len(judged)),
            "mean_gold_sim": _mean([float(g.metrics["gold_sim"]) for g in judged]),
            "mean_antonym_sim": _mean([float(g.metrics["antonym_sim"]) for g in judged]),
            "mean_unrelated_sim": _mean([
                round(sum(g.metrics["unrelated_sims"]) / len(g.metrics["unrelated_sims"]), 6)
                for g in judged if g.metrics["unrelated_sims"]
            ]),
            "sim_gap": _mean([float(g.metrics["sim_gap"]) for g in judged]),
            "by_topic": _by_topic(judged),
            "dimension": judged[0].metrics.get("dimension") if judged else None,
            "inputs_total": sum(int(g.metrics["pool_size"]) + 1 for g in judged),
            "requests": len(judged),
            "mean_in_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            "reported_in_tokens": len(tokens),
            "pass_hat_k": stability_hat,
            "pass_at_k": stability_at,
            "stability_gap": gap,
            "low_confidence": len({g.case_id for g in attributable}) < LOW_CONFIDENCE_N,
            "scoring": "rank-based",
        }


def _by_topic(judged: Sequence[Grade]) -> dict[str, dict[str, Any]]:
    """按话题聚合，每格带分母：某个话题只有 2 条时它的"1.000"没有意义。"""
    out: dict[str, dict[str, Any]] = {}
    for grade in judged:
        topic = str(grade.metrics.get("topic") or "unknown")
        rank = int(grade.metrics["rank"])
        entry = out.setdefault(topic, {"cases": 0, "recall_at_1": 0, "mrr_sum": 0.0,
                                       "anti_above": 0})
        entry["cases"] += 1
        entry["recall_at_1"] += 1 if rank == 1 else 0
        entry["mrr_sum"] += float(grade.metrics["mrr"])
        entry["anti_above"] += 1 if int(grade.metrics["rank_antonym"]) < rank else 0
    return {
        topic: {
            "cases": entry["cases"],
            "recall_at_1": rate(entry["recall_at_1"], entry["cases"]),
            "mrr": round(entry["mrr_sum"] / entry["cases"], 6) if entry["cases"] else None,
            "anti_above_rate": rate(entry["anti_above"], entry["cases"]),
        }
        for topic, entry in sorted(out.items())
    }


def _stability(judged: Sequence[Grade]) -> tuple[int, Any, Any, Any]:
    by_case: dict[str, list[bool]] = {}
    for grade in judged:
        by_case.setdefault(grade.case_id, []).append(bool(grade.passed))
    groups = list(by_case.values())
    return (
        max((len(v) for v in groups), default=1),
        pass_hat_k(groups) if groups else None,
        pass_at_k(groups) if groups else None,
        stability_gap(groups) if groups else None,
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if not norm_a or not norm_b:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


def _engine_in_tokens(sample: Embedding) -> int | None:
    """只取引擎自己回报的输入 token 数。没回报就是 None，不拿估算冒充。"""
    for usage in sample.usage or ():
        if not usage.ok or usage.source is not TokenSource.ENGINE:
            continue
        if usage.in_tokens is not None:
            return int(usage.in_tokens)
    return None
