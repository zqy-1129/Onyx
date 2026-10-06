"""单条 trace 的 token 采信解释（S38）。

`onyx traces show` 已经渲染了"采信 + 各来源对账 + 分段归因"三张表——那是**事实源**。
这一模块只回答它没答的四问，且全部用现成常量与口径判定，不引入第二套阶梯：

1. 为什么采信落在这一档（哪几档没数、哪几档样本不足、哪几档本版本没实现）；
2. 换成别的档差多少（各来源与采信值的绝对/相对差）；
3. **分段闭合吗**（Σ非 output 分段 == 采信 in_tokens，P9 口径）。`measurement/parts.py` 的
   口径是「template_ctl = 引擎计数 − Σ分段（残差）」，所以**未 clamp 时闭合是构造出来的恒等式**；
   不闭合只有一种成因——分段用的计数器比引擎高估，残差为负被 clamp 成 0。
   这句话 S38 写错过一次（把它当成"采集缺陷"），S39 按真机取证改成条件句；
4. 要升到更可信的一档，还差什么（给命令，不给鼓励）。

分段归因的**档位**不在 `usage` 行上，而在 `trace.extra["attribution"]`
（`count_source` / `clamped` / `residual_raw`），由 gateway 的 `USAGE_ATTRIBUTION` 事件带来。
调用方必须把它传进来，否则这里只能报"求和不等"，说不出**为什么**不等。

`COMPAT` 的位置是"有数但永不参与采信"（P14 实测 `/v1` 与原生通道对同一份 prompt 计数不同），
所以它在表里必须显式标成交叉验证，而不是让人以为"两个来源随便挑一个"。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.core.types import SOURCE_PRIORITY, TokenSource
from onyx.llm.measurement.fidelity import FITTED_MIN_SAMPLES
from onyx.llm.measurement.reconciler import DEFAULT_DRIFT_MIN_TOKENS, DEFAULT_DRIFT_THRESHOLD

#: 阶梯上存在但本版本没有实现的档（PROBES P9 实测：3 个模型里 2 个没有 chat template）。
#: 写在这里而不是让人去翻 DESIGN 的表——"装了 tokens extra 也不生效"这种事必须当场看得见。
UNIMPLEMENTED: dict[TokenSource, str] = {
    TokenSource.HF_TOKENIZER: "本版本未实现（T1 只有插拔位）",
    TokenSource.GGUF_VOCAB: "本版本未实现（T2 未做，见 PROBES P9）",
}

OUTPUT_PART = "output"


@dataclass(frozen=True, slots=True)
class TierRow:
    """阶梯上的一档在这一条 trace 里的实际状态。"""

    source: str
    status: str            # 采信 / 有数（未采信）/ 交叉验证 / 样本不足 / 没报数 / 未实现
    in_tokens: int | None = None
    out_tokens: int | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "status": self.status, "in_tokens": self.in_tokens,
                "out_tokens": self.out_tokens, "note": self.note}


@dataclass(frozen=True, slots=True)
class TokenExplanation:
    trace_id: str
    model: str
    provider_id: str
    chosen: str
    confidence: str
    drift_pct: float | None
    tiers: tuple[TierRow, ...]
    deltas: tuple[dict[str, Any], ...]
    closure: dict[str, Any]
    attribution: dict[str, Any] = field(default_factory=dict)
    advice: tuple[str, ...] = field(default_factory=tuple)

    @property
    def clean(self) -> bool:
        """退出码的依据：漂移不超阈，且分段闭合**没有被证伪**。

        "没做分段归因"是未知（`closed is None`），未知不算脏——把"没测"报成"有问题"
        会让人去查一条没坏的东西；但它会在输出里显式写成「未判定」，不混进「闭合」。
        """
        drift_ok = self.drift_pct is None or self.drift_pct <= DEFAULT_DRIFT_THRESHOLD
        return drift_ok and self.closure.get("closed") is not False

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id, "model": self.model, "provider_id": self.provider_id,
            "chosen": {"source": self.chosen, "confidence": self.confidence,
                       "drift_pct": self.drift_pct},
            "thresholds": {"drift_pct": DEFAULT_DRIFT_THRESHOLD,
                           "drift_min_tokens": DEFAULT_DRIFT_MIN_TOKENS,
                           "fitted_min_samples": FITTED_MIN_SAMPLES},
            "tiers": [row.as_dict() for row in self.tiers],
            "deltas": list(self.deltas),
            "closure": dict(self.closure),
            "attribution": dict(self.attribution),
            "advice": list(self.advice),
            "clean": self.clean,
        }


def explain(
    bundle: Any, *, fitted_ratio: float | None = None, fitted_n: int = 0,
    model: str = "", provider_id: str = "", attribution: dict[str, Any] | None = None,
) -> TokenExplanation:
    """把 `UsageRepo.fetch()` 回来的 bundle 翻成一份"为什么是这个数"。

    `fitted_*` 由调用方从模型档案传入（这一层不碰库）：标定状态属于模型，不属于这条 trace，
    但它决定了 fitted 档到底能不能用，所以必须在这里体现，而不是让人去别处查。
    `attribution` 同理，来自 `trace.extra["attribution"]`——分段是按哪一档计数器数的、
    有没有被 clamp，只有当时记下来才知道。
    """
    usage = getattr(bundle, "usage", None)
    alts = list(getattr(bundle, "alts", ()) or ())
    parts = list(getattr(bundle, "parts", ()) or ())
    trace_id = str(getattr(usage, "trace_id", "") or getattr(bundle, "trace_id", "") or "")
    chosen_source = str(getattr(usage, "source", "") or "")
    confidence = str(getattr(usage, "confidence", "") or "")
    drift_pct = getattr(usage, "drift_pct", None)

    by_source = {str(alt.source): alt for alt in alts}
    engine_in = getattr(usage, "in_tokens", None)

    tiers: list[TierRow] = []
    for source in SOURCE_PRIORITY:
        key = str(source)
        alt = by_source.get(key)
        if chosen_source == key:
            tiers.append(TierRow(key, "采信",
                                 getattr(usage, "in_tokens", None),
                                 getattr(usage, "out_tokens", None),
                                 f"置信度 {confidence}"))
            continue
        if key in UNIMPLEMENTED:
            tiers.append(TierRow(key, "未实现", note=UNIMPLEMENTED[key]))
        elif alt is not None and not alt.ok:
            tiers.append(TierRow(key, "没报数", note=str(alt.note or "该来源本次不可用")[:120]))
        elif key == str(TokenSource.FITTED) and alt is None:
            missing = max(0, FITTED_MIN_SAMPLES - int(fitted_n or 0))
            tiers.append(TierRow(key, "样本不足" if missing else "没报数",
                                 note=(f"标定样本 {int(fitted_n or 0)}/{FITTED_MIN_SAMPLES}"
                                       + (f"，还差 {missing} 条" if missing else ""))))
        elif alt is not None:
            tiers.append(TierRow(key, "交叉验证" if key == str(TokenSource.COMPAT) else "有数（未采信）",
                                 alt.in_tokens, alt.out_tokens,
                                 "P14：/v1 与原生通道计数不同，永不参与采信"
                                 if key == str(TokenSource.COMPAT) else ""))
        else:
            tiers.append(TierRow(key, "没报数", note="这一档在本次请求里没有产出样本"))

    deltas = _deltas(chosen_source, engine_in, getattr(usage, "out_tokens", None), alts)
    attrib = _attribution(attribution)
    closure = _closure(parts, engine_in, attrib)
    advice = _advice(chosen_source, confidence, drift_pct, tiers, closure, fitted_n, model, attrib)
    return TokenExplanation(
        trace_id=trace_id, model=model, provider_id=provider_id,
        chosen=chosen_source or "—", confidence=confidence or "—", drift_pct=drift_pct,
        tiers=tuple(tiers), deltas=deltas, closure=closure, attribution=attrib, advice=advice,
    )


def _attribution(raw: dict[str, Any] | None) -> dict[str, Any]:
    """归因档位只认三件事：用的哪一档计数器、有没有被 clamp、当时的残差是多少。

    没记就是没记（`recorded=False`）——S39 之前的行与 mock 通路都可能没有，
    写成一个假档位比留空更坏。
    """
    if not raw:
        return {"recorded": False, "count_source": None, "clamped": None, "residual_raw": None}
    return {
        "recorded": True,
        "count_source": str(raw["count_source"]) if raw.get("count_source") else None,
        "clamped": bool(raw["clamped"]) if "clamped" in raw else None,
        "residual_raw": raw.get("residual_raw"),
    }


def _deltas(chosen: str, in_tokens: int | None, out_tokens: int | None,
            alts: Sequence[Any]) -> tuple[dict[str, Any], ...]:
    if in_tokens is None:
        return ()
    out: list[dict[str, Any]] = []
    for alt in alts:
        key = str(alt.source)
        if key == chosen or not alt.ok or alt.in_tokens is None:
            continue
        out.append({
            "source": key, "in_tokens": alt.in_tokens,
            "in_delta": alt.in_tokens - in_tokens,
            "in_pct": (alt.in_tokens - in_tokens) / in_tokens * 100.0 if in_tokens else None,
            "out_delta": (alt.out_tokens - out_tokens)
            if alt.out_tokens is not None and out_tokens is not None else None,
        })
    return tuple(out)


def _closure(parts: Sequence[Any], in_tokens: int | None, attrib: dict[str, Any]) -> dict[str, Any]:
    """Σ非 output 分段 == 采信 in_tokens 吗（P9 决定的归因口径）。

    恒等式是这么来的：`template_ctl = 引擎计数 − Σ分段`，**残差非负时闭合由构造保证**。
    所以"不闭合"只有一种成因——分段计数器比引擎高估，残差为负被 clamp 成 0（`parts.py` 明写
    "原样记录，不掩盖"）。把这件事说成"采集缺陷"会把人指向错的方向，所以这里带上档位与残差。
    """
    base = {
        "count_source": attrib.get("count_source"), "clamped": attrib.get("clamped"),
        "residual_raw": attrib.get("residual_raw"),
    }
    if not parts:
        return {**base, "checked": False, "closed": None,
                "note": "这条 trace 没有分段归因（要 fitted / tokenizer 档才有）⇒ 无从判断闭合"}
    total = sum(int(p.tokens or 0) for p in parts if str(p.part) != OUTPUT_PART)
    if in_tokens is None:
        return {**base, "checked": False, "closed": None, "sum": total,
                "note": "没有可比的采信计数 ⇒ 闭合判不了（不等于「闭合」）"}
    closed = total == in_tokens
    if closed:
        note = "闭合" + ("" if attrib.get("recorded") else "（这条没记归因档位，闭合只是求和相等）")
    elif attrib.get("clamped"):
        note = (f"不闭合：分段计数器高估 ⇒ 残差 {attrib.get('residual_raw')} 被 clamp 成 0"
                f"（这一条的分段是按 {attrib.get('count_source') or '?'} 数的）")
    else:
        note = (f"分段求和比采信值{'多' if total > in_tokens else '少'} {abs(total - in_tokens)} tok"
                + ("；而归因记录说没被 clamp ⇒ 分段与采信总数不自洽，要有第三方改过其中一边"
                   if attrib.get("recorded") else "；这条没记归因档位，说不清原因"))
    return {**base, "checked": True, "closed": closed, "sum": total, "reported": in_tokens,
            "delta": total - in_tokens, "note": note}


def _advice(chosen: str, confidence: str, drift_pct: float | None, tiers: list[TierRow],
            closure: dict[str, Any], fitted_n: int, model: str,
            attrib: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    if chosen == str(TokenSource.HEURISTIC):
        out.append("采信落在 heuristic/low：这条 trace 的 token 数是估的，不能与引擎计数的 trace 混着算成本")
    missing = max(0, FITTED_MIN_SAMPLES - int(fitted_n or 0))
    if chosen in {str(TokenSource.HEURISTIC), str(TokenSource.FITTED)} and missing:
        target = model or "<模型名>"
        out.append(f"要升到 fitted：还差 {missing} 条标定样本 → onyx calibrate --model {target}")
    if drift_pct is not None and drift_pct > DEFAULT_DRIFT_THRESHOLD:
        out.append(f"漂移 {drift_pct:.1%} 超过阈值 {DEFAULT_DRIFT_THRESHOLD:.0%}：口径分裂比数字不准更危险，"
                   "先核对是哪个来源在说谎（看上面的差值列）")
    out.extend(_closure_advice(closure, fitted_n, model))
    unimplemented = [row.source for row in tiers if row.status == "未实现"]
    if unimplemented:
        out.append(f"阶梯上这些档本版本没实现：{', '.join(unimplemented)}（装了 tokens extra 也不生效）")
    return tuple(out)


def _closure_advice(closure: dict[str, Any], fitted_n: int, model: str) -> tuple[str, ...]:
    """不闭合要说的是**根因 + 一条命令**，不是"采集侧有缺陷"。

    真机取证（S39）：未标定 ⇒ 分段按启发式数 ⇒ 比引擎高估三成 ⇒ 残差为负被 clamp，
    这类不闭合在 `onyx calibrate` 之后当场消失。标定过了还被 clamp 就是另一件事——
    那时候该查的是引擎有没有裁正文，而不是再标一次。

    每个分支都必须返回**元组**：调用方是 `out.extend(...)`，返回裸字符串会被按字符展开，
    终端上就成了"一句话一个字"的竖排（本机第一次跑就撞到了）。
    """
    if not (closure.get("checked") and closure.get("closed") is False):
        return ()
    target = model or "<模型名>"
    if not closure.get("clamped"):
        if not closure.get("count_source") and closure.get("residual_raw") is None:
            return ("这条没记归因档位（S39 之前的老行），所以只能报求差、说不出原因；"
                    f"新跑的请求会带上 count_source，可对照 {target} 的下一条 trace",)
        return ("求和与采信总数不等，而归因记录说没被 clamp ⇒ 两边不是同一次计算的结果，"
                "这是记录层面的不自洽，值得单独查",)
    if closure.get("count_source") == str(TokenSource.HEURISTIC):
        lead = ("这个模型没标定（fitted 档是死的）⇒ 分段按启发式数，"
                if fitted_n < FITTED_MIN_SAMPLES else "分段按启发式数，")
        head = (f"{lead}比引擎高估 ⇒ 残差 {closure.get('residual_raw')} 被 clamp 成 0。"
                f"分段现在只能比各段的相对占比，不能当「哪部分花了多少 token」。")
        if fitted_n < FITTED_MIN_SAMPLES:
            return (head + f"先标定：onyx calibrate --model {target} --n {FITTED_MIN_SAMPLES}",)
        # 模型档案是**现在**的状态，这条 trace 用的是当时的档位——两者不一致时必须说清，
        # 否则会让人去查一条已经修好的东西（"让你再跑一次 calibrate"就是典型的错建议）。
        return (head + f"这个模型现在已经标定（样本 {fitted_n} 条，门槛 {FITTED_MIN_SAMPLES}）⇒ "
                       f"这一发是标定前跑的，重跑一次就会闭合；历史行不回填（那等于伪造当时的测量）。",)
    return (f"已标定（分段按 {closure.get('count_source')} 数）还被 clamp ⇒ 高估来自标定本身，"
            "或者引擎在这一发上裁过正文（对照 `ctx_util` 与正文的 token 下限，见 PROBES P9/P11）",)
