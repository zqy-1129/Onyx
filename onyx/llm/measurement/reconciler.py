"""多来源对账：采信谁、置信度多高、偏差多大、要不要报异常。

采信顺序见 `core.types.SOURCE_PRIORITY`。三条硬规则：
1. `compat` 永不参与采信（P14 实测与原生不一致），只作为交叉验证记录；
2. 没有引擎计数时**降级但不隐瞒**：confidence 随之下降，并产出异常码；
3. drift 超阈值必须报 `TOKEN_DRIFT`——口径分裂比数字不准更危险。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.core.types import (
    SOURCE_PRIORITY,
    Confidence,
    Generation,
    ReconciledUsage,
    TokenSample,
    TokenSource,
)

DEFAULT_DRIFT_THRESHOLD = 0.10
#: 短 prompt 上百分比全是噪声（16 vs 19 就是 16%），会淹掉真正的口径分裂。
#: 只有绝对差也超过这个门槛才报警 —— 否则告警疲劳会让真问题被忽略。
DEFAULT_DRIFT_MIN_TOKENS = 24

_CONFIDENCE_BY_SOURCE: dict[TokenSource, Confidence] = {
    TokenSource.ENGINE: Confidence.HIGH,
    TokenSource.HF_TOKENIZER: Confidence.HIGH,
    TokenSource.GGUF_VOCAB: Confidence.HIGH,
    TokenSource.FITTED: Confidence.MEDIUM,
    TokenSource.HEURISTIC: Confidence.LOW,
    TokenSource.COMPAT: Confidence.LOW,
}

#: 可作为"独立复算"参照的来源（用来算 drift）。engine 是采信首选，不能自己校自己。
_ATTRIBUTION_SOURCES: tuple[TokenSource, ...] = (
    TokenSource.HF_TOKENIZER,
    TokenSource.GGUF_VOCAB,
    TokenSource.FITTED,
    TokenSource.HEURISTIC,
)


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    usage: ReconciledUsage
    anomalies: tuple[tuple[str, dict[str, Any]], ...] = ()
    chosen: TokenSample | None = None
    notes: list[str] = field(default_factory=list)


def reconcile(
    samples: Sequence[TokenSample],
    *,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    drift_min_tokens: int = DEFAULT_DRIFT_MIN_TOKENS,
    parts: Sequence[Any] = (),
) -> ReconcileResult:
    ok_samples = [s for s in samples if s.ok and s.in_tokens is not None]
    by_source = {s.source: s for s in ok_samples}
    anomalies: list[tuple[str, dict[str, Any]]] = []
    notes: list[str] = []

    engine = by_source.get(TokenSource.ENGINE)
    if engine is None and TokenSource.COMPAT not in by_source:
        # 兼容层通道（vLLM / LM Studio / Ollama `/v1`）报的计数也是"引擎侧报告"：
        # 明明有服务器的数字却喊"没有引擎计数"，会把注意力引向一个不存在的问题
        anomalies.append(("NO_ENGINE_COUNT", {"available": sorted(str(k) for k in by_source)}))

    chosen = next((by_source[s] for s in SOURCE_PRIORITY if s in by_source), None)
    if chosen is None:
        # 一个可用来源都没有：宁可返回空usage + 异常，也不要填 0
        anomalies.append(("NO_USAGE_SOURCE", {"samples": len(samples)}))
        return ReconcileResult(
            usage=ReconciledUsage(
                source=TokenSource.HEURISTIC, confidence=Confidence.LOW, alts=tuple(samples),
                parts=tuple(parts),
            ),
            anomalies=tuple(anomalies),
            notes=["无任何可用计数来源"],
        )

    confidence = _CONFIDENCE_BY_SOURCE.get(chosen.source, Confidence.LOW)
    if chosen.confidence and _rank(chosen.confidence) > _rank(confidence):
        confidence = chosen.confidence  # Counter 可以主动降级（例如 fitted 样本不足）

    drift_info = _drift(chosen, by_source)
    drift = drift_info[0] if drift_info else None
    abs_diff = drift_info[1] if drift_info else 0
    if drift is not None and drift > drift_threshold and abs_diff >= drift_min_tokens:
        anomalies.append(("TOKEN_DRIFT", {
            "chosen_source": str(chosen.source), "chosen_in": chosen.in_tokens,
            "drift_pct": round(drift, 4), "abs_diff": abs_diff, "threshold": drift_threshold,
            "min_tokens": drift_min_tokens,
        }))
    if confidence is Confidence.LOW:
        anomalies.append(("LOW_CONFIDENCE_USAGE", {"source": str(chosen.source)}))

    cross = by_source.get(TokenSource.COMPAT)
    if cross is not None and chosen.source is not TokenSource.COMPAT \
            and chosen.in_tokens and cross.in_tokens:
        delta = cross.in_tokens - chosen.in_tokens
        # 任何偏差都记录：P14 实测原生 22 vs 兼容层 20（−9%），
        # 若套用 drift 阈值(10%) 就会漏掉这条真实存在的口径分裂。
        if delta:
            notes.append(
                f"compat 层输入计数偏差 {delta:+d}"
                f"（{delta / chosen.in_tokens:+.1%}；P14：两通道模板不同，"
                "有原生计数时 compat 只作交叉验证）"
            )
    elif cross is not None and chosen.source is TokenSource.COMPAT:
        notes.append("该通道只有兼容层计数（openai-compat 系），按 LOW 置信采信；"
                     "本地模板级归因不可用，分段残差会偏大")

    return ReconcileResult(
        usage=ReconciledUsage(
            source=chosen.source,
            confidence=confidence,
            in_tokens=chosen.in_tokens,
            out_tokens=chosen.out_tokens,
            thinking_tokens=chosen.thinking_tokens,
            cached_tokens=chosen.cached_tokens,
            drift_pct=None if drift is None else round(drift, 4),
            alts=tuple(samples),
            parts=tuple(parts),
        ),
        anomalies=tuple(anomalies),
        chosen=chosen,
        notes=notes,
    )


def _drift(chosen: TokenSample, by_source: dict[TokenSource, TokenSample]) -> tuple[float, int] | None:
    """采信值与最优独立复算值的 (相对偏差, 绝对差)。没有独立来源就返回 None（不猜）。"""
    for source in _ATTRIBUTION_SOURCES:
        other = by_source.get(source)
        if other is None or other is chosen or not other.in_tokens:
            continue
        if not chosen.in_tokens:
            return None
        abs_diff = abs(chosen.in_tokens - other.in_tokens)
        return abs_diff / chosen.in_tokens, abs_diff
    return None


def _rank(confidence: Confidence) -> int:
    return {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}[confidence]


def prefill_mode(
    in_tokens: int | None,
    prompt_eval_ms: float | None,
    *,
    warm_threshold_ms_per_token: float = 0.30,
) -> str:
    """判定这次 prefill 是冷算还是命中了 KV 缓存（PROBES P11）。

    实测：644 token 的 prompt，冷 0.597 ms/token、热 0.129 ms/token，差 4.65×。
    把两者混进同一个吞吐 P50 会得到一个既不代表冷启动也不代表稳态的数字，
    而且**看起来完全合理**——所以吞吐必须按这个标记分列聚合。

    阈值与硬件/模型相关，应由 `onyx calibrate` 用冷热中位数取中点标定；
    这里给的默认值 0.30 来自上述实测（0.129 与 0.597 之间）。
    """
    if not in_tokens or not prompt_eval_ms or prompt_eval_ms <= 0:
        return "unknown"
    ms_per_token = prompt_eval_ms / in_tokens
    if ms_per_token <= warm_threshold_ms_per_token:
        return "warm"
    return "cold"


def latency_summary(gen: Generation, *, warm_threshold_ms_per_token: float = 0.30) -> dict[str, Any]:
    """看板延迟卡的数据源：冷/热分列是硬要求，不提供合并视图。"""
    engine = gen.usage_from(TokenSource.ENGINE)
    in_tokens = engine.in_tokens if engine else None
    prompt_ms = gen.latency.ms("prompt_eval") if gen.latency else None
    mode = prefill_mode(in_tokens, prompt_ms, warm_threshold_ms_per_token=warm_threshold_ms_per_token)
    return {
        "ttft_ms": gen.ttft_ms,
        "wall_ms": gen.wall_ms,
        "prefill_mode": mode,
        "prefill_ms_per_token": round(prompt_ms / in_tokens, 4) if in_tokens and prompt_ms else None,
        "prefill_tps": gen.prefill_tps,
        "decode_tps": gen.decode_tps,
        "cold_load": gen.latency.is_cold if gen.latency else False,
        "load_ms": gen.latency.ms("load") if gen.latency else None,
    }
