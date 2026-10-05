"""工具选择评测（DESIGN §9.2）。

这个任务只看**模型发起了什么调用**，不执行工具——执行是 `tools fire` 的事。
分开的好处：分数低时可以立刻判定是"不会选"还是"选了但跑不通"，
而这两者的修法完全相反。

判定分七种，因为七种的修法各不相同：

| verdict | 含义 | 该改什么 |
|---|---|---|
| `correct` | 该调的调了、不该调的没调，参数也对 | — |
| `no_call` | 该调工具却直接编了答案 | 提示词 / 工具 description 的"什么时候该用" |
| `wrong_tool` | 调了，但选错工具 | 工具之间的描述区分度 |
| `bad_args` | 工具对，参数错 | 参数 description / required / enum 说明 |
| `hallucinated_tool` | 调了工具集里**不存在**的名字 | 提示词限定可用工具；检查是否漏给了工具 |
| `invalid_format` | 参数 JSON 截断或非法 | max_tokens / 停止词 / 模板（P20） |
| `false_call`（计入 metrics） | 不该调却调了 | 工具 description 太诱人；提示词加"不需要就别调" |

`no_call_needed` 子集的误调率**单独统计**，绝不与"没调对"合并：
一个从不乱调工具的模型和一个不会调工具的模型，在合并后的分数上完全一样，
但它们相反。
"""

from __future__ import annotations

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
    ToolCall,
    ToolSpec,
    TraceContext,
    TracePurpose,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.graders.args_match import ArgMatch, FieldMatch, match_args, summarize
from onyx.eval.graders.set_match import set_match
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
    "你可以调用提供的工具来完成任务。规则：\n"
    "1. 只有在确实需要外部信息或精确计算时才调用工具；能直接回答的就直接回答。\n"
    "2. 一次可以并行发起多个调用，但每个调用都必须是必要的。\n"
    "3. 只能使用给定列表里的工具，参数必须严格符合其 schema。\n"
    "4. 不要编造工具名，也不要猜测参数值。"
)

#: 参数比对默认不按严格档：模型多给一个可选字段通常无害
DEFAULT_EXACT_ARGS = False


class ToolSelection:
    id = "tool_selection"
    name = "工具选择与参数"
    #: 需要原生工具调用能力。不支持时**必须 skip 并写明原因**，
    #: 不做隐式降级——用提示词模拟出来的分数无法与原生支持比较，却看不出区别
    requires: frozenset[Cap] = frozenset({Cap.TOOLS})
    #: 与 `aggregate()` 的键完全一致，理由见 intent_classification 的同名注释
    metric_names = (
        "must_call_acc", "must_call_acc_ci", "no_call_rate", "false_call_rate",
        "wrong_tool_rate", "hallucinated_tool_rate",
        "hit_at_1", "set_precision", "set_recall", "set_f1",
        "args_exact_rate", "args_subset_rate", "args_field_rate", "args_relaxed_share",
        "args_match_kinds", "by_kind", "parse_fail_rate",
        "pass_hat_k", "pass_at_k", "stability_gap", "k",
        "n_total", "n_attributable", "n_must_call", "n_no_call_needed",
        "n_bootstrap_units", "verdicts", "low_confidence", "scoring",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        split: str = "default",
        max_tokens: int = 512,
        temperature: float = 0.0,
        exact_args: bool = DEFAULT_EXACT_ARGS,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.split = split
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.exact_args = exact_args
        self.system_prompt = system_prompt

    # ── 契约实现 ──────────────────────────────────────────────────
    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]:
        for raw in self.dataset.select(split=split or self.split, limit=limit):
            yield Case(
                id=str(raw["id"]), input=dict(raw.get("input") or {}),
                expect=dict(raw.get("expect") or {}), dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "single"),
                tools=tuple(_to_spec(item) for item in (raw.get("tools") or ())),
                meta=dict(raw.get("meta") or {}), tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> GenerationRequest:
        return GenerationRequest(
            model=self.model,
            messages=(
                Message(role=Role.SYSTEM, content=self.system_prompt),
                Message(role=Role.USER, content=str(case.input.get("instruction") or "")),
            ),
            params=GenParams(max_tokens=self.max_tokens, temperature=self.temperature),
            tools=case.tools,
            thinking=False,
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Generation) -> Grade:
        if sample.status is not Status.OK:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}",
                metrics={"kind": case.kind},
            )

        expected = [dict(item) for item in (case.expect.get("calls") or [])]
        must_call = bool(case.expect.get("must_call", expected))
        available = {spec.name for spec in case.tools}
        actual = list(sample.tool_calls)
        actual_names = [call.name for call in actual]

        base: dict[str, Any] = {
            "kind": case.kind, "must_call": must_call,
            "expected": [_brief(item) for item in expected],
            "actual": [_brief(call) for call in actual],
            "available_tools": sorted(available),
        }

        # 幻觉工具名优先判：调了一个不存在的工具，后面的比对都没有意义
        hallucinated = [name for name in actual_names if name not in available]
        if hallucinated:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.HALLUCINATED_TOOL, passed=False,
                out_of_set=True, metrics={**base, "hallucinated": hallucinated},
                error=f"调用了工具集里不存在的名字: {hallucinated}",
            )

        # 参数解析失败也优先判：这不是"选错"，是"输出没成形"（P20：模板与停止词）
        unparsed = [call for call in actual if str(call.parse_status) != "ok"]
        if unparsed:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True,
                metrics={**base, "parse_status": [str(c.parse_status) for c in unparsed],
                         "args_raw": [c.arguments_raw[:200] for c in unparsed]},
                error=f"{len(unparsed)} 个调用的参数没能解析"
                      f"（{[str(c.parse_status) for c in unparsed]}）；原文已保留",
            )

        if not must_call:
            # no_call_needed：不调是对的，调了是误调。误调单独计一个指标
            if not actual:
                return Grade(case_id=case.id, score=1.0, verdict=Verdict.CORRECT, passed=True,
                             metrics={**base, "false_call": False})
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.WRONG, passed=False,
                metrics={**base, "false_call": True},
                error=f"不该调用工具，却调了 {actual_names}",
            )

        if not actual:
            # 该调却不调 —— 与"调错了"必须分开：前者是覆盖问题，后者是区分度问题
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.NO_CALL, passed=False,
                metrics={**base, "no_call": True},
                error="需要调用工具，模型却直接给了正文",
                extra={"text_chars": len(sample.text)},
            )

        selection = set_match([item["name"] for item in expected], actual_names)
        matched = _pair_arguments(expected, actual)
        arg_matches = [
            match_args(want, got, schema=_schema_of(case, name), exact=self.exact_args)
            for name, want, got in matched
        ]
        args_ok = all(item.ok for item in arg_matches) and len(matched) == len(expected)

        if not selection.exact:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.WRONG_TOOL, passed=False,
                metrics={**base, "missing": list(selection.missing),
                         "unexpected": list(selection.unexpected),
                         **_set_scores(selection)},
                error=f"工具集合不匹配：缺 {list(selection.missing)} 多 {list(selection.unexpected)}",
            )

        if not args_ok:
            return Grade(
                case_id=case.id, score=_partial_score(arg_matches), verdict=Verdict.BAD_ARGS,
                passed=False,
                # 选择维度的信息也要带上：并行调用少发一次时集合是相等的，
                # 只有 missing/unexpected/set_f1 能说明"选对了但没调够"
                metrics={**base, "hit_at_1": _hit_at_1(expected, actual),
                         **_set_scores(selection), "missing": list(selection.missing),
                         "unexpected": list(selection.unexpected),
                         "paired": len(matched), "expected_calls": len(expected),
                         "args": [_arg_report(item) for item in arg_matches]},
                error=_describe_arg_failures(
                    arg_matches, unpaired=len(expected) - len(matched)
                ),
            )

        return Grade(
            case_id=case.id, score=1.0, verdict=Verdict.CORRECT, passed=True,
            metrics={**base, "hit_at_1": _hit_at_1(expected, actual),
                     **_set_scores(selection),
                     "args": [_arg_report(item) for item in arg_matches]},
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        attributable = [g for g in grades if g.attributable]
        n = len(grades)

        must = [g for g in attributable if g.metrics.get("must_call")]
        optional = [g for g in attributable if not g.metrics.get("must_call")]
        by_kind: dict[str, dict[str, Any]] = {}
        kind_cases: dict[str, set[str]] = {}
        for grade in attributable:
            kind = str(grade.metrics.get("kind") or "unknown")
            bucket = by_kind.setdefault(kind, {"n": 0, "correct": 0})
            bucket["n"] += 1
            bucket["correct"] += int(grade.verdict is Verdict.CORRECT)
            # 每个 kind 同时给 sample 数和 case 数：只给 n=138 的话，
            # 没人知道那是 46 个 case × 3 次采样，会把它当成 138 个独立观测
            kind_cases.setdefault(kind, set()).add(grade.case_id)

        arg_reports = [
            _rebuild_match(item) for grade in attributable
            for item in (grade.metrics.get("args") or [])
        ]
        arg_summary = summarize([m for m in arg_reports if m is not None])

        by_case: dict[str, list[bool]] = {}
        for grade in attributable:
            if grade.passed is not None:
                by_case.setdefault(grade.case_id, []).append(grade.passed)
        samples = list(by_case.values())

        # 比率的折算单位是 **case**：同一个 case 的 k 次采样不是 k 个独立观测
        # （temperature=0 下它们几乎是同一个答案）。按 sample 算 CI 会让 n 虚高 k 倍，
        # 区间窄得像"模型很确定"，其实只是同一件事被数了三遍。
        must_scores = _per_case(must, lambda g: float(g.verdict is Verdict.CORRECT))
        return {
            "n_total": n,
            "n_attributable": len(attributable),
            "n_must_call": len(must),
            "n_no_call_needed": len(optional),
            "verdicts": _verdict_counts(grades),
            # 选择维度
            "must_call_acc": _mean_of(must_scores),
            "must_call_acc_ci": mean_ci(must_scores, seed=seed) if must_scores else mean_ci([]),
            "no_call_rate": rate(
                sum(1 for g in must if g.verdict is Verdict.NO_CALL), len(must)
            ),
            "false_call_rate": rate(
                sum(1 for g in optional if g.metrics.get("false_call")), len(optional)
            ),
            "wrong_tool_rate": rate(
                sum(1 for g in must if g.verdict is Verdict.WRONG_TOOL), len(must)
            ),
            # 下面两个率的分母同样是 attributable：引擎失败的那条不是一次
            # "有机会幻觉/有机会解析失败"的样本，算进去会低估这两个率
            "hit_at_1": _mean_per_case(attributable, "hit_at_1"),
            "set_precision": _mean_per_case(attributable, "set_precision"),
            "set_recall": _mean_per_case(attributable, "set_recall"),
            "set_f1": _mean_per_case(attributable, "set_f1"),
            "hallucinated_tool_rate": rate(
                sum(1 for g in attributable if g.verdict is Verdict.HALLUCINATED_TOOL),
                len(attributable),
            ),
            "parse_fail_rate": rate(
                sum(1 for g in attributable if g.verdict is Verdict.INVALID_FORMAT),
                len(attributable),
            ),
            # 参数维度（与选择维度分开：选对工具但填错参数，修的是 schema 不是描述）
            "args_exact_rate": arg_summary["exact_rate"],
            "args_subset_rate": arg_summary["subset_rate"],
            "args_field_rate": arg_summary["field_rate"],
            "args_relaxed_share": arg_summary["relaxed_share"],
            "args_match_kinds": arg_summary["kinds"],
            # 稳定性
            "k": max((len(v) for v in samples), default=1),
            "pass_hat_k": pass_hat_k(samples) if samples else None,
            "pass_at_k": pass_at_k(samples) if samples else None,
            "stability_gap": stability_gap(samples) if samples else None,
            "by_kind": {
                kind: {**bucket, "cases": len(kind_cases[kind]),
                       "acc": rate(bucket["correct"], bucket["n"])}
                for kind, bucket in sorted(by_kind.items())
            },
            "n_bootstrap_units": len(samples),
            # 单位是 **case** 不是 sample：bootstrap_ci 重采样的是 case，
            # 所以 k=3 时 291 条 grade 也只相当于 97 个样本。
            # 早先这里写的是 `len(attributable) < 100`，于是 97 个 case 报成"样本充足"，
            # 而 CI 自己带的 n=97 说相反的话——两个数打架时该以重采样单位为准
            "low_confidence": len(samples) < LOW_CONFIDENCE_N,
            "scoring": "gen-based",
        }


# ── 辅助 ──────────────────────────────────────────────────────────
def _to_spec(raw: Any) -> ToolSpec:
    if isinstance(raw, ToolSpec):
        return raw
    if isinstance(raw, dict):
        return ToolSpec.from_openai_tool(raw)
    raise TypeError(f"无法把 {type(raw).__name__} 转成 ToolSpec")


def _schema_of(case: Case, name: str) -> dict[str, Any]:
    spec = next((s for s in case.tools if s.name == name), None)
    return dict(spec.parameters or {}) if spec else {}


def _per_case(grades: Sequence[Grade], key_fn) -> list[float]:
    """把每条 grade 折成"每个 case 一个值"（该 case 内各次采样的均值）。

    bootstrap 的重采样单位必须是 case：同一 case 的 k 次采样不是 k 个独立观测。
    """
    buckets: dict[str, list[float]] = {}
    for grade in grades:
        value = key_fn(grade)
        if value is None:
            continue
        buckets.setdefault(grade.case_id, []).append(float(value))
    return [sum(items) / len(items) for items in buckets.values()]


def _mean_of(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_per_case(grades: Sequence[Grade], key: str) -> float | None:
    """某个逐 sample 指标的 case 级均值。

    跳过"这一维没判出来"的 grade（`key_fn` 返回 None），而不是当成 0：
    当成 0 会让 WRONG_TOOL 那些根本没走到集合比对的样本把均值拉低，
    于是分数里混进了"没机会得分"而不是"得了 0 分"。
    """
    values = _per_case(
        grades, lambda g: (float(g.metrics[key]) if g.metrics.get(key) is not None else None)
    )
    return _mean_of(values)


def _set_scores(selection: Any) -> dict[str, Any]:
    """选择集合的三个分数。三个都要写：只报 F1 就看不出是"多调"还是"漏调"，
    而 precision 低（发了期望之外的调用）要改工具描述的边界与提示词，
    recall 低（少发了期望的调用）要改 description 的"什么时候该用"——修法相反。"""
    return {"set_precision": selection.precision, "set_recall": selection.recall,
            "set_f1": selection.f1}


def _brief(item: Any) -> dict[str, Any]:
    """写进 metrics 的精简形态：只留名字与参数，避免把整份 schema 抄进每条 grade。"""
    if isinstance(item, ToolCall):
        return {"name": item.name, "arguments": item.arguments,
                "parse_status": str(item.parse_status)}
    return {"name": item.get("name"), "arguments": item.get("arguments")}


def _pair_arguments(
    expected: Sequence[dict[str, Any]], actual: Sequence[ToolCall]
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """把期望调用与实际调用按 (名字, 出现次序) 配对。

    按次序而不是按"第一个同名的"配：`get_weather(北京)` 与 `get_weather(上海)`
    是两个不同的调用，配错了会得出"参数全对"的假结论。
    """
    remaining = list(actual)
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for want in expected:
        name = str(want.get("name") or "")
        index = next(
            (i for i, call in enumerate(remaining)
             if call.name == name and call.arguments is not None),
            None,
        )
        if index is None:
            continue
        call = remaining.pop(index)
        pairs.append((name, dict(want.get("arguments") or {}), dict(call.arguments or {})))
    return pairs


def _hit_at_1(expected: Sequence[dict[str, Any]], actual: Sequence[ToolCall]) -> bool | None:
    """第一个调用是否就是期望的第一个。`None` 表示无从判定（有一边是空的）。"""
    if not expected or not actual:
        return None
    return actual[0].name == str(expected[0].get("name") or "")


def _arg_report(item: ArgMatch) -> dict[str, Any]:
    return {
        "ok": item.ok, "subset_ok": item.subset_ok, "score": item.score,
        "exact": item.exact,
        "fields": [
            {"field": f.field, "ok": f.ok, "kind": f.kind, "detail": f.detail,
             "expected": f.expected, "actual": f.actual}
            for f in item.fields
        ],
    }


def _rebuild_match(report: dict[str, Any]) -> ArgMatch | None:
    """从落库的 metrics 里还原 ArgMatch，好让 aggregate 能重算汇总。

    必须能还原：否则聚合只能读运行时内存里的那一份，续跑时就丢了一半数据。
    """
    if not isinstance(report, dict) or "fields" not in report:
        return None
    return ArgMatch(
        fields=tuple(
            FieldMatch(
                field=str(item.get("field") or ""), ok=bool(item.get("ok")),
                kind=str(item.get("kind") or ""), expected=item.get("expected"),
                actual=item.get("actual"), detail=str(item.get("detail") or ""),
            )
            for item in report["fields"]
        ),
        exact=bool(report.get("exact")),
    )


def _partial_score(matches: Sequence[ArgMatch]) -> float:
    scores = [item.score for item in matches if item.score is not None]
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def _describe_arg_failures(
    matches: Sequence[ArgMatch], *, unpaired: int = 0
) -> str:
    parts: list[str] = []
    if unpaired:
        # 并行调用少发一次时，配上的那些参数可能全对——
        # 只报字段级失败就会得到一句没有信息量的"参数不匹配"
        parts.append(f"期望的调用里有 {unpaired} 次没被发起")
    for item in matches:
        for field_match in item.fields:
            if field_match.ok:
                continue
            parts.append(
                f"{field_match.field}[{field_match.kind}]"
                + (f": {field_match.detail}" if field_match.detail else "")
            )
    return "；".join(parts[:5]) or "参数不匹配"


def _verdict_counts(grades: Sequence[Grade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for grade in grades:
        counts[str(grade.verdict)] = counts.get(str(grade.verdict), 0) + 1
    return counts
