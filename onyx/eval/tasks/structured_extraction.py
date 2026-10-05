"""结构化抽取任务（M11 / S30）。

它测的是**没有受理解码器时，模型能不能自己吐出合规 JSON**。
判定拆成三层正交，缺一层就会把不同的病读成同一种：

| 层 | 问的问题 | 坏了该怎么修 |
|---|---|---|
| `json_valid_rate` | 吐的是不是一个 JSON 对象 | 提示词、停止词、max_tokens |
| `schema_valid_rate` | 字段集合与类型对不对（少抽/多抽/类型错） | 字段约束、description、schema 严格度 |
| `field_em` / `score` | 抽出来的值准不准 | 这才是模型能力问题 |

混成一个"JSON 正确率"会把"提示词没写清"读成"模型不行"，而这两件事的修法完全相反
（DESIGN §9.4）。所以三层各自成指标，`Grade.invalid_format` 也照常落在每条样本上。

**两个"正确率"分母不同，是刻意的**：`score` 只在"结构合规且有字段可抽"的样本里算，
`exact_object_rate` 在所有可归因样本里算（格式坏、结构坏、负样本造字段都算不合规）。
两者的差就是格式与结构层造成的损失——只看前者会低估"能不能直接接下游"。

**`requires` 刻意不含 `Cap.STRUCTURED_OUTPUT`**：那是"引擎支持受理解码"的能力位。
要求它就等于把要测的东西当成前提——分数恒等于 1，而探针里 `structured_output` 大多还是
`?`（未实测）。这个任务存在的理由就是把那一格从"未知"变成"有数"。

**负样本（`kind=none`）是这任务的一半价值**：句子里没有可抽取信息时，正确输出是 `{}`。
没有这类样本，模型把"今天天气不错"编成 `{"person": ""}` 也得满分——
而凭空造字段恰恰是抽取任务里最贵的失败（它会往下游流进数据库）。
负样本也**不进主分数**：它们走 `none_correct_rate`，混进均值等于奖励"什么都不抽"。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date, timedelta
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
from onyx.eval.datasets.builtin.structured_ie import (
    ANCHOR,
    FIELD_TYPES,
    VALUE_VOCAB,
    schema_for,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.graders.json_schema import check_schema, clean_json_object, field_em, parse_json
from onyx.eval.metrics import (
    LOW_CONFIDENCE_N,
    mean_ci,
    pass_at_k,
    pass_hat_k,
    rate,
    stability_gap,
)
from onyx.eval.task import Case, Grade, Verdict

SYSTEM_PROMPT = (
    "从用户给的中文句子里抽取结构化信息。\n"
    "只输出一个 JSON 对象，不要输出解释、前后缀或代码围栏。\n"
    "只允许这些字段：{fields}。\n"
    "必须包含的字段：{required}。\n"
    "{values}"
    "句子里没有可抽取的信息时，输出 {{}}（一个空对象），不要编造字段。"
)

#: 每个字段的取值口径。**必须写进提示词**：真机跑第一次时发现，原 prompt 只说
#: "字段值必须来自原句，不要改写"，而期望值却是 ISO 日期与纯数值——
#: 于是 `date` 的字段级 EM 是 **0.000（15 条全错）**：模型照原句抄了「3 月 4 号」，
#: 我们却要求 `2026-03-04`。那不是模型不会抽，是考卷没说清答题格式，
#: 而分数看起来完全像是能力问题（DESIGN §9.4 防的就是这种误读）。
#: 归一化本身仍是可判定的要求：中文数字换算与相对日期换算都留着，因为它们是真的能力项。
def _value_rules() -> str:
    vocabulary = "".join(
        f"「{name}」的取值只能是：{'、'.join(values)}。\n"
        for name, values in VALUE_VOCAB.items()
    )
    anchor = date.fromisoformat(ANCHOR)
    tomorrow = (anchor + timedelta(days=1)).isoformat()
    return (
        "人名、机构、地点、事件照原句抄写，不要改写、不要加标点。\n"
        + vocabulary
        + f"日期输出 YYYY-MM-DD；句中的相对说法按锚定日 {ANCHOR} 换算，"
        f"例如该日的「明天」是 {tomorrow}。\n"
        "金额只输出数值，不带单位、引号与千分位；中文数字要换算成阿拉伯数字，"
        "例如「三千二」是 3200。\n"
    )


VALUE_RULES = _value_rules()

#: 拒答的本地形态。与 intent 任务各写一份是刻意的：这是一段面向中文措辞的启发式清单，
#: 抽成公共常量会让两个任务被迫共用同一份判定阈值
REFUSAL_MARKERS = ("抱歉", "无法", "不能", "不清楚", "没有相关信息", "作为ai")


class StructuredExtraction:
    """`EvalTask` 协议的实现。数据来自 `structured_ie` 生成器。"""

    id = "structured_extraction"
    name = "中文结构化抽取"
    #: 只需要会聊天。见模块 docstring：受理解码是被测对象，不是前提
    requires = frozenset({Cap.CHAT})
    metric_names = (
        "n_total", "n_attributable", "n_judged", "verdicts",
        "score", "score_ci", "field_em", "field_em_ci", "exact_object_rate",
        "json_valid_rate", "schema_valid_rate", "schema_verified",
        "format_valid_rate", "invalid_format_rate", "off_vocabulary_rate", "refusal_rate",
        "none_total", "none_correct_rate", "hallucinated_fields",
        "per_field", "k", "pass_hat_k", "pass_at_k", "stability_gap",
        "low_confidence", "scoring",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        max_tokens: int = 160,
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
                expect={"fields": dict(expect.get("fields") or {}),
                        "keys": tuple(expect.get("keys") or ())},
                dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "single"),
                meta=dict(raw.get("meta") or {}), tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> GenerationRequest:
        keys = tuple(case.expect.get("keys") or ())
        return GenerationRequest(
            model=self.model,
            messages=(
                Message(role=Role.SYSTEM, content=self._system_text(keys)),
                Message(role=Role.USER, content=str(case.input.get("text") or "")),
            ),
            params=GenParams(max_tokens=self.max_tokens, temperature=self.temperature),
            # P12：thinking 计入 eval_count 且会吃光预算，正文变空会被误判成"格式非法"
            thinking=False,
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Generation) -> Grade:
        expected = dict(case.expect.get("fields") or {})
        keys = tuple(case.expect.get("keys") or ())
        text = sample.text or ""

        if sample.status is not Status.OK:
            # 引擎失败是环境问题，不进模型能力分母
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}",
                metrics={"expected_keys": list(keys)},
            )
        if not text.strip():
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True,
                error=("正文为空但产出了推理内容：预算被 thinking 吃光（P12），"
                       "提高 max_tokens 或确认 thinking=False"
                       if sample.thinking.strip() else "正文为空"),
                metrics={"expected_keys": list(keys), "json_valid": False,
                         "schema_valid": False, "thinking_chars": len(sample.thinking)},
            )

        check = parse_json(text, strict_object=True)
        clean = clean_json_object(text)
        if not check.parsed or check.as_object is None:
            if _is_refusal(text):
                return Grade(
                    case_id=case.id, score=0.0, verdict=Verdict.REFUSED, passed=False,
                    invalid_format=True, error="模型拒答",
                    metrics={"expected_keys": list(keys), "json_valid": False,
                             "schema_valid": False, "text": text[:200]},
                )
            # 解析不出来就到此为止：再往下每一层都无意义，
            # 硬算会把"格式没听话"记成"字段抽错"，修的方向就全错了
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True, error=check.error[:200],
                metrics={"expected_keys": list(keys), "json_valid": False,
                         "schema_valid": False, "text": text[:200]},
            )

        actual = dict(check.as_object)
        if not keys:
            return self._grade_negative(case, actual, clean=clean, text=text)

        schema = check_schema(actual, schema_for(keys))
        if not schema.valid:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.WRONG, passed=False,
                invalid_format=not clean,
                # 结构不对不是"答错内容"：修法在字段约束与提示词，不在换模型
                error=("；".join(schema.errors[:3]) or "schema 不合规")[:200],
                metrics={
                    "expected_keys": list(keys), "expected": expected, "predicted": actual,
                    "json_valid": True, "schema_valid": False,
                    "schema_verified": schema.verified,
                    "missing": list(schema.missing),
                    "unexpected": sorted(name for name in actual if name not in expected),
                    "text": text[:200],
                },
            )

        compared = field_em(expected, actual)
        matched, total = int(compared["matched"]), int(compared["total"])
        all_ok = matched == total and not compared["unexpected"]
        # 值落在词表之外是**幻觉**，与"抽了另一个在表里的值"是两种病：
        # 前者改词表说明与"不要改写"，后者才是分不清类别（与 out_of_label 同一条理由）
        off = sorted(
            name for name, allowed in VALUE_VOCAB.items()
            if name in expected and actual.get(name) not in allowed
        )
        return Grade(
            case_id=case.id, score=float(compared["score"] or 0.0),
            verdict=(Verdict.CORRECT if all_ok
                     else Verdict.PARTIAL if matched else Verdict.WRONG),
            passed=all_ok, invalid_format=not clean, out_of_set=bool(off),
            metrics={
                "expected_keys": list(keys), "expected": expected, "predicted": actual,
                "json_valid": True, "schema_valid": True,
                "schema_verified": schema.verified,
                "field_matched": matched, "field_total": total,
                "per_field_ok": compared["per_field"],
                "missing": compared["missing"], "unexpected": compared["unexpected"],
                "off_vocabulary": off,
                "text": text[:200],
            },
        )

    def _grade_negative(self, case: Case, actual: dict[str, Any], *, clean: bool,
                        text: str) -> Grade:
        """负样本：期望是空对象。多抽一个字段就是幻觉，分值直接归零。"""
        invented = sorted(actual)
        empty = not invented
        return Grade(
            case_id=case.id, score=1.0 if empty else 0.0,
            verdict=Verdict.CORRECT if empty else Verdict.WRONG,
            passed=empty, invalid_format=not clean,
            error="" if empty else f"句中没有可抽取信息，却造出了 {invented}",
            metrics={
                "expected_keys": [], "expected": {}, "predicted": actual,
                "json_valid": True, "schema_valid": empty,
                "schema_verified": True, "hallucinated": invented, "field_total": 0,
                "text": text[:200],
            },
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        attributable = [g for g in grades if g.attributable]
        n = len(grades)
        counts: dict[str, int] = {}
        for grade in grades:
            counts[str(grade.verdict)] = counts.get(str(grade.verdict), 0) + 1

        # 主分数的分母：只数"确实有字段可抽"且结构合规的样本
        judged = [g for g in attributable if int(g.metrics.get("field_total") or 0) > 0]
        negatives = [g for g in attributable if g.metrics.get("expected_keys") == []]
        em_values = [float(g.score) for g in judged]
        exact_flags = [1.0 if g.verdict is Verdict.CORRECT else 0.0 for g in judged]
        strict_ok = sum(1 for g in attributable if g.passed is True)
        none_ok = sum(1 for g in negatives if g.verdict is Verdict.CORRECT)
        hallucinated = sum(len(g.metrics.get("hallucinated") or []) for g in attributable)

        per_field: dict[str, dict[str, Any]] = {}
        for name in FIELD_TYPES:
            hits = [
                (g.metrics.get("per_field_ok") or {}).get(name)
                for g in judged if name in (g.metrics.get("expected") or {})
            ]
            usable = [h for h in hits if h is not None]
            per_field[name] = {
                "em": rate(sum(1 for h in usable if h), len(usable)) if usable else None,
                "n": len(usable),
            }

        checked = [g for g in attributable if "schema_verified" in g.metrics]
        by_case: dict[str, list[bool]] = {}
        for grade in attributable:
            if grade.passed is not None:
                by_case.setdefault(grade.case_id, []).append(bool(grade.passed))
        samples = list(by_case.values())

        return {
            "n_total": n,
            "n_attributable": len(attributable),
            # CI 的分母与主分数同源：两者都用 n_judged，不是一边算 grade 条数一边算 case 条数
            "n_judged": len(judged),
            "verdicts": counts,
            # 主分数 = 结构合规样本里"每个字段都对且没多抽"的比例
            "score": rate(int(sum(exact_flags)), len(judged)),
            # CI 的点估计必须等于 score：同一个 judged、同一个统计量，
            # 否则界面上"分数"与"区间"来自两个分母（G5 的 CI 同源判据）
            "score_ci": mean_ci(exact_flags, seed=seed).as_dict(),
            # 字段级均值：比 score 宽松，用来看"错是错在一个字段还是全错"
            "field_em": (sum(em_values) / len(em_values)) if em_values else None,
            "field_em_ci": mean_ci(em_values, seed=seed).as_dict(),
            # 端到端严格口径：分母含负样本、且格式坏/结构坏一律算不合规。
            # 它与 score 的差就是"提示词/解码层"造成的损失，那个数才是能不能直接接下游的依据
            "exact_object_rate": rate(strict_ok, len(attributable)),
            "json_valid_rate": rate(
                sum(1 for g in attributable if g.metrics.get("json_valid")), len(attributable)),
            "schema_valid_rate": rate(
                sum(1 for g in attributable if g.metrics.get("schema_valid")), len(attributable)),
            # jsonschema 缺席时退化成"只查 required 与顶层类型"，
            # 那只能报"没发现问题"，不能报"合规"——这一位就是用来戳穿这种混淆的。
            # 一条都没走到 schema 检查时是 None（未知）：格式全坏的 run
            # 不许把"没校验过"显示成"校验过且通过"
            "schema_verified": (
                all(g.metrics["schema_verified"] for g in checked) if checked else None
            ),
            "format_valid_rate": rate(
                sum(1 for g in attributable if not g.invalid_format), len(attributable)),
            "invalid_format_rate": rate(
                sum(1 for g in attributable if g.invalid_format), len(attributable)),
            # 值不在词表里（"退款申请"而不是"退款"）：这是造词，与"选错了类别"分开计
            "off_vocabulary_rate": rate(
                sum(1 for g in attributable if g.out_of_set), len(attributable)),
            "refusal_rate": rate(counts.get(Verdict.REFUSED.value, 0), n),
            "none_total": len(negatives),
            "none_correct_rate": rate(none_ok, len(negatives)),
            "hallucinated_fields": hallucinated,
            "per_field": per_field,
            "k": max((len(v) for v in samples), default=1),
            "pass_hat_k": pass_hat_k(samples) if samples else None,
            "pass_at_k": pass_at_k(samples) if samples else None,
            "stability_gap": stability_gap(samples) if samples else None,
            # 低样本按 case 数判，不按 grade 条数：k=4 时后者会虚高 4 倍
            "low_confidence": len(by_case) < LOW_CONFIDENCE_N,
            "scoring": "gen-based",
        }

    # ── 内部 ──────────────────────────────────────────────────────
    def _system_text(self, keys: Sequence[str]) -> str:
        fields = "、".join(f"{name}:{FIELD_TYPES[name]}" for name in FIELD_TYPES)
        required = "、".join(sorted(keys)) if keys else "（无，输出空对象即可）"
        return SYSTEM_PROMPT.format(fields=fields, required=required, values=VALUE_RULES)


def _is_refusal(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
