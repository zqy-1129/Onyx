"""指令遵循任务（M11 / S31）。

一条样本 = 一句把要求写明的中文指令 + 一组**可机械检查**的约束。
分数是"满足了几条 / 要求几条"，但同时报三个口径，因为它们会分叉：

| 口径 | 问的问题 | 只报它会不会骗人 |
|---|---|---|
| `score` | 每条样本的平均遵循度（样本等权） | 会：一道 5 条约束的题与一道 2 条的题等权 |
| `micro_rate` | 所有约束里满足了多少（约束等权） | 会：约束多的题主导整个数 |
| `all_satisfied_rate` | 有多少题是**全部**照做的 | 不会，但它是三者里最严的那个 |

`by_kind` 每种约束各带分母：某类约束只考了 3 条时，它的"满足率"说明不了任何事。

**防空手套分（这任务的形状与 S30 的负样本同源）**：**空正文不给任何约束记分**。
逐条判的话 `max_chars` 与 `forbids` 会对一句废话都判"通过"，
于是"什么都不写"能拿到 2/6 ≈ 0.33 的分，而它一条指令都没遵循。
所以空正文直接记 0 分、全部约束记为未满足，并由 `empty_outputs` 与 `format_valid_rate` 说明为什么。
**拒答不一样**：那是真有产出，就按产出逐条实测（一句"抱歉，我无法完成"确实只满足 1/5 条），
只在 verdict 上单独标出来 —— 把实测清零是"为了故事好看"而改测量。
反过来，**全部约束都满足的回答永远不会被启发式改判成拒答**，哪怕它开头写了"抱歉"。
`requires` 只要 `CHAT`：约束全是文本形态，不需要受理解码也不需要工具。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
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
from onyx.eval.graders.constraints import Check, check_all
from onyx.eval.metrics import (
    LOW_CONFIDENCE_N,
    mean_ci,
    pass_at_k,
    pass_hat_k,
    rate,
    stability_gap,
)
from onyx.eval.task import Case, Grade, Verdict

#: 一句中性系统提示：不能与任何约束冲突（S30 的教训——提示词与判据必须同源）。
SYSTEM_PROMPT = (
    "按用户给出的要求直接输出内容本身。\n"
    "不要解释你做了什么，不要复述要求，不要加前后缀。"
)

#: 拒答的形态清单。**只用自我声明式的短语**，不收 bare "无法/不能/不清楚"：
#: 生成类任务的正常答案里会出现这些词（"无法离线使用"就是一句合格的隐私说明），
#: 把它们当拒答会把一条真的遵循了指令的回答清零。
#: 与 intent / S30 各写一份是刻意的：三段面向中文措辞的启发式清单，阈值不该被迫共用。
REFUSAL_MARKERS = ("抱歉", "对不起", "我无法", "不能回答", "无法回答", "作为ai", "不予回答")

#: 拒答通常是短的。长答案里冒出一个"抱歉"是内容而不是拒绝，所以加一道长度闸门
REFUSAL_MAX_VISIBLE = 60

#: 空正文的逐条原因。写"不计满足"而不是"违反"：我们没有观察到任何产出
_NO_OUTPUT = "没有可判定的产出，不计满足"


def _looks_like_refusal(text: str) -> bool:
    """拒答判据同时要求**短**：长答案里冒出一个"抱歉"通常是内容而不是拒绝。"""
    if len(_visible(text)) > 60:
        return False
    lowered = text.casefold()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


class InstructionFollowing:
    """`EvalTask` 协议的实现。数据来自 `instructions_zh` 生成器。"""

    id = "instruction_following"
    name = "中文指令遵循"
    requires = frozenset({Cap.CHAT})
    metric_names = (
        "n_total", "n_attributable", "n_judged", "verdicts",
        "score", "score_ci", "micro_rate", "all_satisfied_rate",
        "constraint_total", "constraint_satisfied", "mean_constraints",
        "by_kind", "empty_outputs",
        "format_valid_rate", "invalid_format_rate", "refusal_rate",
        "k", "pass_hat_k", "pass_at_k", "stability_gap",
        "low_confidence", "scoring",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        max_tokens: int = 220,
        temperature: float = 0.0,
        split: str = "default",
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.split = split

    # ── 契约实现 ──────────────────────────────────────────────────
    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]:
        for raw in self.dataset.select(split=split or self.split, limit=limit):
            expect = dict(raw.get("expect") or {})
            yield Case(
                id=str(raw["id"]), input=dict(raw.get("input") or {}),
                expect={"constraints": [dict(c) for c in expect.get("constraints") or ()]},
                dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "instruction"),
                meta=dict(raw.get("meta") or {}), tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> GenerationRequest:
        return GenerationRequest(
            model=self.model,
            messages=(
                Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                Message(role=Role.USER, content=str(case.input.get("text") or "")),
            ),
            params=GenParams(max_tokens=self.max_tokens, temperature=self.temperature),
            thinking=False,
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Generation) -> Grade:
        constraints = [dict(c) for c in case.expect.get("constraints") or ()]
        text = sample.text or ""

        if sample.status is not Status.OK:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}",
                metrics={"constraint_total": len(constraints)},
            )
        if not text.strip():
            # 空正文：一条都不记满足（见模块 docstring 的防空手套分）
            return self._no_output(case, constraints, error=_empty_reason(sample))

        checks = check_all(constraints, text)
        satisfied = sum(1 for item in checks if item.ok)
        total = len(checks)
        score = (satisfied / total) if total else 0.0
        all_ok = bool(total) and satisfied == total
        # 顺序很重要：**全部约束都满足了就不许再被启发式改写成拒答**。
        # 一句"抱歉，七天内原路退回，不超过 14 字"这种怪答案确实照做了所有要求，
        # 把它判成拒答是把测量改成故事。
        refused = not all_ok and _looks_like_refusal(text)
        return Grade(
            case_id=case.id, score=score,
            verdict=(Verdict.REFUSED if refused else Verdict.CORRECT if all_ok
                     else Verdict.PARTIAL if satisfied else Verdict.WRONG),
            passed=all_ok, invalid_format=_wrapped_in_fence(text),
            error="模型拒答" if refused else "",
            metrics={
                "constraint_total": total, "constraint_satisfied": satisfied,
                "checks": [item.as_dict() for item in checks],
                "violated": [item.detail for item in checks if item.violated],
                "refused": refused,
                "text": text[:400],
            },
        )

    def _no_output(self, case: Case, constraints: list[dict[str, Any]], *, error: str) -> Grade:
        """没有任何正文时，把全部约束记为未满足，而不是逐条放过。

        逐条判的话 `max_chars` 与 `forbids` 会对一句废话都判"通过"，
        于是"什么都不写"能拿到 2/6 ≈ 0.33 的分——那正是本任务最该抓住的形态。
        """
        checks = [Check(str(c.get("kind")), False, _NO_OUTPUT) for c in constraints]
        return Grade(
            case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
            invalid_format=True, error=error,
            metrics={
                "constraint_total": len(checks), "constraint_satisfied": 0,
                "checks": [item.as_dict() for item in checks],
                "violated": [item.detail for item in checks],
                "empty": True,
            },
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        attributable = [g for g in grades if g.attributable]
        judged = [g for g in attributable if int(g.metrics.get("constraint_total") or 0) > 0]
        counts: dict[str, int] = {}
        for grade in grades:
            counts[str(grade.verdict)] = counts.get(str(grade.verdict), 0) + 1

        per_case = [(int(g.metrics.get("constraint_satisfied") or 0),
                     int(g.metrics.get("constraint_total") or 0)) for g in judged]
        satisfied = sum(ok for ok, _total in per_case)
        declared = sum(total for _ok, total in per_case)
        scores = [ok / total for ok, total in per_case if total]

        by_kind: dict[str, dict[str, Any]] = {}
        for kind in sorted({str(item.get("kind")) for g in judged
                            for item in g.metrics.get("checks") or ()}):
            items = [item for g in judged for item in g.metrics.get("checks") or ()
                     if str(item.get("kind")) == kind]
            ok = sum(1 for item in items if item.get("ok"))
            by_kind[kind] = {"satisfied": ok, "n": len(items), "rate": rate(ok, len(items))}

        samples: dict[str, list[bool]] = {}
        for grade in attributable:
            if grade.passed is not None:
                samples.setdefault(grade.case_id, []).append(bool(grade.passed))
        groups = list(samples.values())

        return {
            "n_total": len(grades),
            "n_attributable": len(attributable),
            "n_judged": len(judged),
            "verdicts": counts,
            # 主分数 = 每条样本满足率的平均（样本等权）
            "score": (sum(scores) / len(scores)) if scores else None,
            # CI 与主分数同一个统计量、同一个分母（G5 的 CI 同源判据）
            "score_ci": mean_ci(scores, seed=seed).as_dict(),
            "micro_rate": rate(satisfied, declared),
            "all_satisfied_rate": rate(
                sum(1 for g in judged if g.verdict is Verdict.CORRECT), len(judged)),
            "constraint_total": declared,
            "constraint_satisfied": satisfied,
            "mean_constraints": (declared / len(judged)) if judged else None,
            "by_kind": by_kind,
            "empty_outputs": sum(1 for g in attributable if g.metrics.get("empty")),
            "format_valid_rate": rate(
                sum(1 for g in attributable if not g.invalid_format), len(attributable)),
            "invalid_format_rate": rate(
                sum(1 for g in attributable if g.invalid_format), len(attributable)),
            "refusal_rate": rate(counts.get(Verdict.REFUSED.value, 0), len(grades)),
            "k": max((len(v) for v in groups), default=1),
            "pass_hat_k": pass_hat_k(groups) if groups else None,
            "pass_at_k": pass_at_k(groups) if groups else None,
            "stability_gap": stability_gap(groups) if groups else None,
            # 低样本按 case 数判，不按 grade 条数：k>1 时后者会虚高
            "low_confidence": len(samples) < LOW_CONFIDENCE_N,
            "scoring": "gen-based",
        }


def _empty_reason(sample: Generation) -> str:
    """空正文要分清"被 thinking 吃光"与"真的什么都没输出"（P12），修法完全不同。"""
    if sample.thinking.strip():
        return ("正文为空但产出了推理内容：预算被 thinking 吃光（P12），"
                "提高 max_tokens 或确认 thinking=False")
    return "正文为空"


def _wrapped_in_fence(text: str) -> bool:
    """整段被代码围栏包住：形态上没听话，但内容仍可判，所以只记格式维度。"""
    stripped = text.strip()
    return stripped.startswith("```") and stripped.endswith("```")


def _visible(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def _looks_like_refusal(text: str) -> bool:
    """拒答判据同时要求**短**：长答案里冒出一个"抱歉"通常是内容而不是拒绝。

    清单刻意只收自我声明式短语，不收 bare「无法」「不能」——生成类任务的合格答案里
    会出现这些词（"无法离线使用"就是一句合格的隐私说明），把它们当拒答会把一条
    真的遵循了指令的回答清零。与 intent / S30 各写一份也是刻意的。
    """
    if not 0 < _visible(text) <= REFUSAL_MAX_VISIBLE:
        return False
    lowered = text.casefold()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
