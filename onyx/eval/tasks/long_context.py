"""长上下文检索任务（M11 / S32）。

一句话：**测的是"在长文里找到那一句"，不是"模型能不能读很多字"**。
每条样本埋 3 个可精确匹配的事实，位置固定在 first / middle / last，
要求只输出一个 JSON 对象（复用 S30/S31 已有的解析与字段比对，不新写判据）。

四处与短文本任务不同的判断：

1. **窗口由任务钉住**（默认 20,480 token，可 `--num-ctx` 覆盖）。
   2026-10-05 实测 qwen3.5:9b：`/api/tags` 的 card `context_length` = 262,144，
   而 `/api/ps` 报的**实际载入** = 131,072；传 `options.num_ctx=20480` 后 `/api/ps` 立刻回报
   20480（显存 9.32G → 5.73G）⇒ Ollama 确实按请求采纳窗口，"窗口多大"不必猜。
   看 card 会高估一个数量级，而 `obs/visitors/gpu.py` 用的正是实际载入那一份。
   一次运行只用一个窗口值：逐条改 `num_ctx` 会让引擎反复重载，把长上下文测试变成重载速度测试。
2. **越界不记分**：grade 读**引擎回报**的 in_tokens，≥ 窗口就判 `SKIPPED` 并写明原因。
   这是「未知 ≠ 0 分」在长上下文上的形态——被静默截断的检索失败如果算成"模型不会"，
   分数会指导人去换模型，而该改的是窗口配置。
   引擎没回报 in_tokens 时**不做越界判断**（拿启发式估算冒充引擎数字，
   等于凭空造出一个"被截断"的结论），只在 `reported_in_tokens` 里露出这个缺口。
3. **位置与档位各是一个分桶**。总分 0.78 既看不出"中部塌陷"（lost in the middle），
   也看不出"过了 8k 就开始掉"，而只有这两条能指导你改文档排布或换窗口。
4. **答错的埋点要分清"认错实体"还是"没读到"**。第一版数据没有干扰项，真机跑出来 9/9 全对
   （qwen3.5:9b，16k 档 in≈16.7k tok）——那不是模型强，是"扫到任意一个数字"就能得分。
   加了同句式干扰项之后，`needle_confused` / `confusion_rate` 把"两个候选值挑错了"这一段
   单独量出来：它和 `needle_matched` 掉的那部分是两种病，前者要改题面排布，后者才是读不到。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

from onyx.core.types import (
    Cap,
    FinishReason,
    Generation,
    GenerationRequest,
    GenParams,
    Message,
    Role,
    Status,
    TokenSource,
    TraceContext,
    TracePurpose,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.graders.json_schema import clean_json_object, field_em, parse_json
from onyx.eval.metrics import (
    LOW_CONFIDENCE_N,
    mean_ci,
    pass_at_k,
    pass_hat_k,
    rate,
    stability_gap,
)
from onyx.eval.task import Case, Grade, Verdict

#: 默认窗口：16k 档 + 问题块 + 答案余量（实测 0.6–0.7 token/汉字 ⇒ 24,700 字约 15–17k token）
DEFAULT_NUM_CTX = 20480

#: 答案就是几个数值/编号。给多了它会开始解释自己，那反而会把形态判成不合规
DEFAULT_MAX_TOKENS = 64

#: 拒答清单。与本仓库另外两个任务各写一份是刻意的：三段面向中文措辞的启发式，阈值不该被迫共用
REFUSAL_MARKERS = ("抱歉", "对不起", "我无法", "不知道", "没有提到", "文中未", "作为ai")


class LongContext:
    """`EvalTask` 协议的实现。数据来自 `longctx_zh` 生成器。"""

    id = "long_context"
    name = "中文长上下文检索"
    requires = frozenset({Cap.CHAT})
    metric_names = (
        "n_total", "n_attributable", "n_judged", "n_truncated", "verdicts",
        "score", "score_ci", "all_correct_rate",
        "needle_total", "needle_matched", "needle_rate",
        "needle_confused", "confusion_rate",
        "by_position", "by_bucket",
        "json_valid_rate", "format_valid_rate", "invalid_format_rate", "refusal_rate",
        "window_tokens", "max_ctx_util", "mean_in_tokens", "reported_in_tokens",
        "k", "pass_hat_k", "pass_at_k", "stability_gap", "low_confidence", "scoring",
    )

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        num_ctx: int = DEFAULT_NUM_CTX,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        split: str = "default",
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.num_ctx = int(num_ctx)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.split = split

    # ── 契约实现 ──────────────────────────────────────────────────
    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]:
        for raw in self.dataset.select(split=split or self.split, limit=limit):
            expect = dict(raw.get("expect") or {})
            yield Case(
                id=str(raw["id"]), input=dict(raw.get("input") or {}),
                expect={
                    "answers": dict(expect.get("answers") or {}),
                    "keys": [str(key) for key in expect.get("keys") or ()],
                    "positions": dict(expect.get("positions") or {}),
                },
                dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "longctx"),
                meta=dict(raw.get("meta") or {}), tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> GenerationRequest:
        return GenerationRequest(
            model=self.model,
            messages=(Message(role=Role.USER, content=str(case.input.get("text") or "")),),
            # 窗口显式钉住：这一条决定"16k 全错"是能力问题还是被截断
            params=GenParams(
                max_tokens=self.max_tokens, temperature=self.temperature, num_ctx=self.num_ctx,
            ),
            thinking=False,
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Generation) -> Grade:
        answers = dict(case.expect.get("answers") or {})
        keys = list(case.expect.get("keys") or ())
        positions = dict(case.expect.get("positions") or {})
        text = sample.text or ""
        in_tokens = _engine_in_tokens(sample)
        base: dict[str, Any] = {
            "expected_keys": keys, "positions": positions, "in_tokens": in_tokens,
            "bucket": str(case.meta.get("bucket") or "unknown"),
            "ctx_util": round(in_tokens / self.num_ctx, 4) if in_tokens else None,
        }

        if sample.status is not Status.OK:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}", metrics=base,
            )
        if in_tokens is not None and in_tokens >= self.num_ctx:
            # 引擎把开头切掉了：这时判"错"是让模型替配置背锅
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.SKIPPED, passed=None,
                error=(
                    f"输入 {in_tokens} tok ≥ 窗口 {self.num_ctx} tok，被切掉的是**开头**，"
                    "检索结果不可信 ⇒ 提高 --num-ctx 或改跑更小的档位（记 skip，不记 0 分）"
                ),
                metrics={**base, "truncated": True, "needle_total": len(keys),
                         "needle_matched": 0, "per_needle_ok": {}},
            )
        if not text.strip():
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True,
                error=("正文为空但产出了推理内容：预算被 thinking 吃光（P12）"
                       if sample.thinking.strip() else "正文为空"),
                metrics={**base, "json_valid": False, "needle_total": len(keys),
                         "needle_matched": 0, "per_needle_ok": {}},
            )

        check = parse_json(text, strict_object=True)
        if not check.parsed or check.as_object is None:
            refused = _is_refusal(text)
            # 输出被预算切断时"不是 JSON"是症状而不是病：不说清就会有人去改提示词
            clipped = sample.finish_reason is FinishReason.LENGTH
            reason = "模型拒答" if refused else (check.error or "输出不是 JSON 对象")[:160]
            if clipped and not refused:
                reason += "；输出被 max_tokens 截断，先提高生成预算再看这一档"
            return Grade(
                case_id=case.id, score=0.0,
                verdict=Verdict.REFUSED if refused else Verdict.INVALID_FORMAT,
                passed=False, invalid_format=True,
                error=reason[:220],
                metrics={**base, "json_valid": False, "needle_total": len(keys),
                         "needle_matched": 0, "per_needle_ok": {}, "clipped": clipped,
                         "text": text[:200]},
            )

        actual = dict(check.as_object)
        compared = field_em(answers, actual)
        matched, total = int(compared["matched"]), int(compared["total"])
        confused = _confused_keys(actual, dict(compared["per_field"]), _distractor_values(case))
        all_ok = matched == total and not compared["unexpected"]
        clipped = sample.finish_reason is FinishReason.LENGTH
        return Grade(
            case_id=case.id, score=float(compared["score"] or 0.0),
            verdict=(Verdict.CORRECT if all_ok else Verdict.PARTIAL if matched
                     else Verdict.WRONG),
            passed=all_ok, invalid_format=not clean_json_object(text),
            # 输出被 max_tokens 切掉是预算问题，与"没读到"是两种病：不说清就会去改模型
            error="答案 JSON 被 max_tokens 截断，先提高生成预算再看分数" if clipped else "",
            metrics={
                **base, "json_valid": True, "expected": answers, "predicted": actual,
                "needle_total": total, "needle_matched": matched,
                "per_needle_ok": dict(compared["per_field"]),
                # 答成干扰值的键：认错实体与没读到要分开算，否则掉分原因会被归错
                "needle_confused": sum(1 for hit in confused.values() if hit),
                "per_needle_confused": confused,
                # 能判混淆的漏掉数（= 带干扰项的那些错键）：confusion_rate 的分母。
                # 数据没带干扰项时它是 0，占比就写「—」而不是 0——"没证据"不等于"没混淆"
                "confusable_misses": len(confused),
                "missing": compared["missing"], "unexpected": compared["unexpected"],
                "clipped": clipped, "text": text[:200],
            },
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        attributable = [g for g in grades if g.attributable]
        judged = [g for g in attributable if int(g.metrics.get("needle_total") or 0) > 0]
        counts: dict[str, int] = {}
        for grade in grades:
            counts[str(grade.verdict)] = counts.get(str(grade.verdict), 0) + 1

        flags = [1.0 if g.verdict is Verdict.CORRECT else 0.0 for g in judged]
        matched = sum(int(g.metrics.get("needle_matched") or 0) for g in judged)
        declared = sum(int(g.metrics.get("needle_total") or 0) for g in judged)
        confused = sum(int(g.metrics.get("needle_confused") or 0) for g in judged)
        confusable = sum(int(g.metrics.get("confusable_misses") or 0) for g in judged)
        reported = [g for g in attributable if g.metrics.get("in_tokens")]
        tokens = [int(g.metrics["in_tokens"]) for g in reported]
        # 窗口占用率把 skip 的那些也算进来：这个数的用途是"该不该调 --num-ctx"，
        # 而最接近越界的那条恰恰是被记成 skip 的那条——只统计 judged 就正好把它藏掉
        util = [float(g.metrics["ctx_util"]) for g in grades if g.metrics.get("ctx_util")]
        stability_k, stability_hat, stability_at, gap = _stability(grades)

        return {
            "n_total": len(grades),
            "n_attributable": len(attributable),
            "n_judged": len(judged),
            # 有多少条被窗口挡在外面：没有这一位，"16k 全错"就分不清是塌陷还是没测
            "n_truncated": sum(1 for g in grades if g.metrics.get("truncated")),
            "verdicts": counts,
            # 主分数：一条题里 3 个埋点全找到才算对（partial 的分数在下面三个数里）
            "score": rate(int(sum(flags)), len(judged)),
            "score_ci": mean_ci(flags, seed=seed).as_dict(),
            "all_correct_rate": rate(int(sum(flags)), len(judged)),
            "needle_total": declared,
            "needle_matched": matched,
            # 逐埋点命中率：与 score 的差就是"三个只找到一个"的那部分题
            "needle_rate": rate(matched, declared),
            # 掉的那部分里有多少是"答成了干扰值"。分母只用**带干扰项的错键**
            # （`confusable_misses`）：数据没带干扰项时这一位是「—」，不是 0
            "needle_confused": confused,
            "confusion_rate": rate(confused, confusable),
            "by_position": _by_position(judged),
            "by_bucket": _by_bucket(judged),
            "json_valid_rate": rate(
                sum(1 for g in attributable if g.metrics.get("json_valid")), len(attributable)),
            "format_valid_rate": rate(
                sum(1 for g in attributable if not g.invalid_format), len(attributable)),
            "invalid_format_rate": rate(
                sum(1 for g in attributable if g.invalid_format), len(attributable)),
            "refusal_rate": rate(counts.get(Verdict.REFUSED.value, 0), len(grades)),
            "window_tokens": self.num_ctx,
            "max_ctx_util": round(max(util), 4) if util else None,
            "mean_in_tokens": round(sum(tokens) / len(tokens)) if tokens else None,
            # 引擎回报了才谈得上"越界没越界"；这一位露出"没回报所以没判"的缺口
            "reported_in_tokens": len(reported),
            "k": stability_k,
            "pass_hat_k": stability_hat,
            "pass_at_k": stability_at,
            "stability_gap": gap,
            "low_confidence": len({g.case_id for g in attributable}) < LOW_CONFIDENCE_N,
            "scoring": "gen-based",
        }


def _by_position(judged: Sequence[Grade]) -> dict[str, dict[str, Any]]:
    """按埋点位置聚合，每格带分母。

    总分 0.78 说不出任何一句话；`first 0.5 / middle 0.5 / last 1.0` 直接指向文档怎么排。
    **没答的那一题也算进它所属位置的分母**：只统计"回答了的键"会让漏答悄悄缩小分母，
    于是中部塌陷看起来像中部没考。
    """
    out: dict[str, dict[str, Any]] = {}
    for grade in judged:
        positions = grade.metrics.get("positions") or {}
        per_ok = grade.metrics.get("per_needle_ok") or {}
        per_confused = grade.metrics.get("per_needle_confused") or {}
        for key in grade.metrics.get("expected_keys") or ():
            slot = str(positions.get(str(key)) or "unknown")
            entry = out.setdefault(slot, {"matched": 0, "confused": 0, "n": 0})
            entry["n"] += 1
            entry["matched"] += 1 if per_ok.get(str(key)) else 0
            entry["confused"] += 1 if per_confused.get(str(key)) else 0
    for entry in out.values():
        entry["rate"] = rate(entry["matched"], entry["n"])
    return {slot: out[slot] for slot in ("first", "middle", "last", *sorted(out)) if slot in out}


def _by_bucket(judged: Sequence[Grade]) -> dict[str, dict[str, Any]]:
    """按档位聚合"全对率"与"逐埋点命中率"。两个数一起看才知道掉在哪一档。"""
    out: dict[str, dict[str, Any]] = {}
    for grade in judged:
        bucket = str(grade.metrics.get("bucket") or "unknown")
        entry = out.setdefault(bucket, {"cases": 0, "all_correct": 0, "matched": 0, "needles": 0})
        entry["cases"] += 1
        entry["all_correct"] += 1 if grade.verdict is Verdict.CORRECT else 0
        entry["matched"] += int(grade.metrics.get("needle_matched") or 0)
        entry["needles"] += int(grade.metrics.get("needle_total") or 0)
    for entry in out.values():
        entry["all_correct_rate"] = rate(entry["all_correct"], entry["cases"])
        entry["needle_rate"] = rate(entry["matched"], entry["needles"])
    return dict(sorted(out.items()))


def _stability(grades: Sequence[Grade]) -> tuple[int, Any, Any, Any]:
    by_case: dict[str, list[bool]] = {}
    for grade in grades:
        if grade.attributable and grade.passed is not None:
            by_case.setdefault(grade.case_id, []).append(bool(grade.passed))
    groups = list(by_case.values())
    return (
        max((len(v) for v in groups), default=1),
        pass_hat_k(groups) if groups else None,
        pass_at_k(groups) if groups else None,
        stability_gap(groups) if groups else None,
    )


def _distractor_values(case: Case) -> dict[str, Any]:
    """每题的干扰值（来自数据集 meta）。外部数据没带就是空表——不猜。"""
    out: dict[str, Any] = {}
    for item in case.meta.get("needles") or ():
        if isinstance(item, dict) and "distractor_value" in item:
            out[str(item.get("id"))] = item["distractor_value"]
    return out


def _confused_keys(
    actual: dict[str, Any], per_ok: dict[str, Any], distractors: dict[str, Any]
) -> dict[str, bool]:
    """答错的埋点里，哪些答的正好是同句式干扰值。

    只返回**判得了**的键（数据带了该题的干扰值），没带干扰项的漏答不进这个表——
    它的缺失就是 `confusion_rate` 的分母缺口，界面上该显示「—」。
    比对复用 `field_em`（而不是自己写 ==）：干扰值与模型输出的归一口径必须和判分时一致，
    否则会出现"判错为错、却认不出它抄的是干扰项"这种自相矛盾的格子。
    """
    out: dict[str, bool] = {}
    for raw_key, ok in per_ok.items():
        key = str(raw_key)
        if ok or key not in distractors:
            continue
        probe = field_em({key: distractors[key]}, {key: actual.get(key)})
        out[key] = int(probe["matched"]) == 1
    return out


def _engine_in_tokens(sample: Generation) -> int | None:
    """只取**引擎自己回报**的输入 token 数，没有就返回 None。

    拿启发式估算冒充引擎数字，会凭空造出"被截断"的结论——
    而"越界没越界"这件事必须只有真实出处才能决定要不要记 skip。
    """
    for usage in sample.usage or ():
        if not usage.ok or usage.source is not TokenSource.ENGINE:
            continue
        if usage.in_tokens is not None:
            return int(usage.in_tokens)
    return None


def _is_refusal(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in REFUSAL_MARKERS)
