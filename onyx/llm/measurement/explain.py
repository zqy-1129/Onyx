"""单条 trace 的 token 采信解释（S38）。

`onyx traces show` 已经渲染了"采信 + 各来源对账 + 分段归因"三张表——那是**事实源**。
这一模块只回答它没答的四问，且全部用现成常量与口径判定，不引入第二套阶梯：

1. 为什么采信落在这一档（哪几档没数、哪几档样本不足、哪几档本版本没实现）；
2. 换成别的档差多少（各来源与采信值的绝对/相对差）；
3. **分段闭合吗**（Σ非 output 分段 == 采信 in_tokens，P9 口径）——不闭合是采集侧的缺陷，
   不该靠人眼从表里加出来；
4. 要升到更可信的一档，还差什么（给命令，不给鼓励）。

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
            "advice": list(self.advice),
            "clean": self.clean,
        }


def explain(
    bundle: Any, *, fitted_ratio: float | None = None, fitted_n: int = 0,
    model: str = "", provider_id: str = "",
) -> TokenExplanation:
    """把 `UsageRepo.fetch()` 回来的 bundle 翻成一份"为什么是这个数"。

    `fitted_*` 由调用方从模型档案传入（这一层不碰库）：标定状态属于模型，不属于这条 trace，
    但它决定了 fitted 档到底能不能用，所以必须在这里体现，而不是让人去别处查。
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
    closure = _closure(parts, engine_in)
    advice = _advice(chosen_source, confidence, drift_pct, tiers, closure, fitted_n, model)
    return TokenExplanation(
        trace_id=trace_id, model=model, provider_id=provider_id,
        chosen=chosen_source or "—", confidence=confidence or "—", drift_pct=drift_pct,
        tiers=tuple(tiers), deltas=deltas, closure=closure, advice=advice,
    )


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


def _closure(parts: Sequence[Any], in_tokens: int | None) -> dict[str, Any]:
    """Σ非 output 分段 == 采信 in_tokens 吗（P9 决定的归因口径）。"""
    if not parts:
        return {"checked": False, "closed": None,
                "note": "这条 trace 没有分段归因（要 fitted / tokenizer 档才有）⇒ 无从判断闭合"}
    total = sum(int(p.tokens or 0) for p in parts if str(p.part) != OUTPUT_PART)
    if in_tokens is None:
        return {"checked": False, "closed": None, "sum": total,
                "note": "没有可比的采信计数 ⇒ 闭合判不了（不等于「闭合」）"}
    return {
        "checked": True, "closed": total == in_tokens, "sum": total, "reported": in_tokens,
        "delta": total - in_tokens,
        "note": "闭合" if total == in_tokens else
                f"分段求和比采信值{'多' if total > in_tokens else '少'} {abs(total - in_tokens)} tok",
    }


def _advice(chosen: str, confidence: str, drift_pct: float | None, tiers: list[TierRow],
            closure: dict[str, Any], fitted_n: int, model: str) -> tuple[str, ...]:
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
    if closure.get("checked") and not closure.get("closed"):
        out.append("分段不闭合是采集侧的缺陷：这条 trace 的「哪部分花了多少 token」不能拿去做归因对比")
    unimplemented = [row.source for row in tiers if row.status == "未实现"]
    if unimplemented:
        out.append(f"阶梯上这些档本版本没实现：{', '.join(unimplemented)}（装了 tokens extra 也不生效）")
    return tuple(out)
