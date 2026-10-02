"""意图识别任务（DESIGN §9.2 的第一个内置任务）。

这个任务的价值不在于"测分类准不准"，而在于它把 DESIGN §9.4 那条硬约束落地了：
API-only 拿不到受约束的 logprob，只能生成式打分，于是模型会因为**输出格式不听话**
额外掉分。所以这里刻意维护两个正交的维度：

- `verdict`  内容对不对（转账 / 查余额 判得准不准）
- `Grade.invalid_format`  格式听不听话（有没有只输出一个裸标签）

两者混成一个"正确率"就会把格式问题误读成能力问题——而修法完全相反
（前者要换模型或改任务难度，后者只要改提示词或停止词）。

由此得到四种可区分的输出形态，本地小模型上都会高频出现：

| 模型输出 | verdict | invalid_format | 说明 |
|---|---|---|---|
| `转账` | correct | False | 干净且正确 |
| `查余额` | wrong | False | 格式对了，内容错——真的分类错误 |
| `退款` | out_of_label | True | **标签集之外的幻觉**，不是"选错" |
| `这个意图是转账。` | correct | True | 内容对但没遵守格式；仍可判，但格式合法率要扣 |
| `不是转账，是查余额` | invalid_format | True | 出现两个候选 ⇒ **不可判定**，绝不猜 |
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

from onyx.core.types import (
    Cap,
    Generation,
    GenerationRequest,
    GenParams,
    Message,
    Role,
    Status,
    TraceContext,
    TracePurpose,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.graders.normalize import match_label
from onyx.eval.metrics import (
    pass_at_k,
    pass_hat_k,
    rate,
    stability_gap,
    summarize_pairs,
    top_confusions,
)
from onyx.eval.task import Case, Grade, Verdict

SYSTEM_PROMPT = (
    "你是一个意图分类器。把用户的话归入下面列出的意图之一，"
    "**只输出意图名称本身**，不要输出解释、标点、引号或任何其它文字。\n"
    "可选意图：{labels}\n"
    "如果用户的话不属于以上任何意图，输出「{fallback}」。"
)

#: 拒答的常见形态。单独统计是因为"过度拒答"是一种真实缺陷（DESIGN §9.2 refusal_safety）
REFUSAL_MARKERS = (
    "抱歉", "对不起", "无法回答", "不能回答", "作为ai", "作为一个ai",
    "我无法", "不予回答", "不便回答",
)

#: 分类任务不需要大预算；给小一点还能顺带暴露"输出被截断"这类问题。
#: 但不能太小：P12 实测 thinking 模型会把预算吃光，正文变成空串
DEFAULT_MAX_TOKENS = 32


class IntentClassification:
    id = "intent_classification"
    name = "中文意图识别"
    #: 纯文本分类不需要 tools / structured_output；声明为空集，
    #: 这样任何 provider 都能跑，`check_capabilities` 永远不会 skip
    requires: frozenset[Cap] = frozenset()
    metric_names = (
        "macro_f1", "macro_f1_ci", "accuracy", "balanced_accuracy",
        "format_valid_rate", "out_of_label_rate", "invalid_format_rate", "refusal_rate",
        "per_class_f1", "top_confusions", "pass_hat_k", "pass_at_k", "stability_gap",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        labels: Sequence[str] = ("转账", "查余额", "投诉", "其他"),
        fallback_label: str = "其他",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        split: str = "default",
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.labels = tuple(labels)
        self.fallback_label = fallback_label
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.split = split
        self.system_prompt = system_prompt
        if fallback_label not in self.labels:
            # 兜底标签不在标签集里，模型输出它就永远算"越界"，分数会莫名偏低
            raise ValueError(
                f"fallback_label {fallback_label!r} 必须在 labels {list(self.labels)} 里"
            )

    # ── 契约实现 ──────────────────────────────────────────────────
    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]:
        for raw in self.dataset.select(split=split or self.split, limit=limit):
            yield Case(
                id=str(raw["id"]), input=dict(raw.get("input") or {}),
                expect=dict(raw.get("expect") or {}), dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "single"),
                meta=dict(raw.get("meta") or {}), tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> GenerationRequest:
        instruction = str(case.input.get("instruction") or "")
        return GenerationRequest(
            model=self.model,
            messages=(
                Message(role=Role.SYSTEM, content=self._system_text()),
                Message(role=Role.USER, content=instruction),
            ),
            params=GenParams(max_tokens=self.max_tokens, temperature=self.temperature),
            # thinking 必须显式关掉：P12 实测 thinking token 计入 eval_count，
            # 而且会把小预算吃光，正文变成空串——那会被误判成"格式非法"
            thinking=False,
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Generation) -> Grade:
        expected = str(case.expect.get("label") or "")
        if not expected:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                invalid_format=True, error="样本缺少 expect.label，无法评分",
            )
        if sample.status is not Status.OK:
            # 引擎失败是环境问题，不该算进模型能力分母
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}",
                metrics={"expected": expected},
            )

        text = sample.text or ""
        matched = match_label(text, self.labels)

        if not text.strip():
            # 空正文要区分"被 thinking 吃光"与"真的什么都没输出"，修法完全不同
            ate_by_thinking = bool(sample.thinking.strip())
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True,
                error=("正文为空但产出了推理内容：预算被 thinking 吃光（P12），"
                       "提高 max_tokens 或确认 thinking=False"
                       if ate_by_thinking else "正文为空"),
                metrics={"expected": expected, "match_status": matched.status,
                         "thinking_chars": len(sample.thinking)},
            )

        if matched.status == "ambiguous":
            # 文本里同时出现多个候选标签 ⇒ 不可判定。挑一个会让分数凭空变高
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True,
                error=f"输出里同时出现了多个候选标签 {list(matched.found)}，不可判定",
                metrics={"expected": expected, "match_status": "ambiguous",
                         "found": list(matched.found), "text": text[:200]},
            )

        if matched.status == "not_found":
            if _is_refusal(text):
                return Grade(
                    case_id=case.id, score=0.0, verdict=Verdict.REFUSED, passed=False,
                    invalid_format=True,
                    error="模型拒答", metrics={"expected": expected, "text": text[:200]},
                )
            # 标签集之外的输出是**幻觉**，不是"选错了"：修法在标签集与提示词，不在模型能力
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.OUT_OF_LABEL, passed=False,
                invalid_format=True, out_of_set=True,
                error=f"输出 {text[:80]!r} 不在标签集 {list(self.labels)} 内",
                metrics={"expected": expected, "match_status": "not_found", "text": text[:200]},
            )

        # 到这里内容可判定。格式是否"干净"单独记，不影响 verdict
        clean = matched.status in {"exact", "normalized"}
        correct = matched.label == expected
        return Grade(
            case_id=case.id, score=1.0 if correct else 0.0,
            verdict=Verdict.CORRECT if correct else Verdict.WRONG,
            passed=correct, invalid_format=not clean,
            metrics={
                "expected": expected, "predicted": matched.label,
                "match_status": matched.status, "text": text[:200],
            },
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        attributable = [g for g in grades if g.attributable]
        # 只有"内容可判定"的样本才进混淆矩阵：ambiguous / empty / out_of_label
        # 没有预测标签，塞进矩阵就等于凭空造一个类别
        pairs = [
            (str(g.metrics.get("expected") or ""), str(g.metrics.get("predicted") or ""))
            for g in attributable
            if g.metrics.get("predicted") is not None
        ]
        summary = summarize_pairs(pairs, seed=seed) if pairs else summarize_pairs([])

        by_case: dict[str, list[bool]] = {}
        for grade in attributable:
            if grade.passed is not None:
                by_case.setdefault(grade.case_id, []).append(grade.passed)
        samples = list(by_case.values())

        n = len(grades)
        counts = _verdict_counts(grades)
        return {
            "n_total": n,
            "n_attributable": len(attributable),
            "n_judged": len(pairs),
            "verdicts": counts,
            # 内容维度
            "macro_f1": summary["macro_f1"],
            "macro_f1_ci": summary["macro_f1_ci"],
            "accuracy": summary["accuracy"],
            "balanced_accuracy": summary["balanced_accuracy"],
            "per_class_f1": {
                label: item.f1 for label, item in summary["per_class"].items()
            },
            "top_confusions": [
                {"expected": a, "actual": b, "count": c} for a, b, c in top_confusions(pairs)
            ],
            # 格式维度（与内容正交，DESIGN §9.4）
            "format_valid_rate": rate(
                sum(1 for g in attributable if not g.invalid_format), len(attributable)
            ),
            "invalid_format_rate": rate(
                sum(1 for g in attributable if g.invalid_format), len(attributable)
            ),
            "out_of_label_rate": rate(counts.get(Verdict.OUT_OF_LABEL.value, 0), n),
            "refusal_rate": rate(counts.get(Verdict.REFUSED.value, 0), n),
            # 稳定性（k>1 时才有意义）
            "k": max((len(v) for v in samples), default=1),
            "pass_hat_k": pass_hat_k(samples) if samples else None,
            "pass_at_k": pass_at_k(samples) if samples else None,
            "stability_gap": stability_gap(samples) if samples else None,
            "low_confidence": summary["low_confidence"],
            "confusion": summary["confusion"],
            "labels": summary["labels"],
            "scoring": "gen-based",  # API-only 的硬约束，必须显示在看板上（§9.4）
        }

    # ── 内部 ──────────────────────────────────────────────────────
    def _system_text(self) -> str:
        return self.system_prompt.format(
            labels="、".join(self.labels), fallback=self.fallback_label
        )


def _verdict_counts(grades: Sequence[Grade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for grade in grades:
        counts[str(grade.verdict)] = counts.get(str(grade.verdict), 0) + 1
    return counts


def _is_refusal(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def make_case_pairs(grades: Sequence[Grade]) -> list[tuple[str, str]]:
    """导出 (期望, 预测) 对，供外部重新聚合（例如只看 hard 子集）。"""
    return [
        (str(g.metrics.get("expected") or ""), str(g.metrics.get("predicted") or ""))
        for g in grades
        if g.attributable and g.metrics.get("predicted") is not None
    ]


def with_trace_id(grade: Grade, trace_id: str) -> Grade:
    """runner 拿到 gateway 返回的 trace_id 后补进 grade。

    分成两步是因为 `task.grade` 只看到 `Generation`，看不到 trace_id——
    而 trace_id 属于 gateway。任务不该知道 gateway 的存在，否则就违反了
    "评测不建立第二条调用路径"里那条边界。
    """
    return replace(grade, trace_id=trace_id)
