"""模型标定：用真实 (字符数, 引擎 token 数) 样本拟合 tokens/char 比（T3 fitted 档）。

为什么需要它：P9 实测发现 3 个模型里 2 个**没有可用的 chat template**，
无法离线精确复算；但引擎自己报的 `prompt_eval_count` 是真值。
于是"用真值反推该模型的字符/token 比"就成了性价比最高的兜底——
一次标定（几十条样本）之后，估计误差能压到个位数百分比，远好于 chars/4。

纯最小二乘（过原点），不引入 numpy/scipy。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """tokens ≈ ratio × chars。`r2` 越接近 1 说明"按字符比例"这个假设越成立。"""

    ratio: float
    n: int
    r2: float
    max_rel_error: float
    #: 冷/热 prefill 的 ms/token 中位数，用于 prefill_mode 判定阈值（PROBES P11）
    cold_ms_per_token: float | None = None
    warm_ms_per_token: float | None = None

    @property
    def usable(self) -> bool:
        """样本太少或拟合太差时不许拿去当 fitted 档用（宁可退回 heuristic + low）。"""
        return self.n >= 30 and self.r2 >= 0.90


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    chars: int
    tokens: int
    prompt_eval_ms: float | None = None
    cold: bool | None = None


def fit_ratio(samples: Sequence[CalibrationSample]) -> CalibrationResult:
    usable = [s for s in samples if s.chars > 0 and s.tokens > 0]
    if not usable:
        return CalibrationResult(ratio=0.0, n=0, r2=0.0, max_rel_error=1.0)

    # 过原点最小二乘：ratio = Σ(x·y) / Σ(x²)
    sum_xy = sum(s.chars * s.tokens for s in usable)
    sum_xx = sum(s.chars * s.chars for s in usable)
    ratio = sum_xy / sum_xx if sum_xx else 0.0

    mean_y = sum(s.tokens for s in usable) / len(usable)
    ss_tot = sum((s.tokens - mean_y) ** 2 for s in usable)
    ss_res = sum((s.tokens - ratio * s.chars) ** 2 for s in usable)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 1.0

    max_rel = max(
        (abs(s.tokens - ratio * s.chars) / s.tokens for s in usable if s.tokens), default=1.0
    )
    return CalibrationResult(
        ratio=round(ratio, 6),
        n=len(usable),
        r2=round(r2, 4),
        max_rel_error=round(max_rel, 4),
        cold_ms_per_token=_median_ms_per_token(usable, cold=True),
        warm_ms_per_token=_median_ms_per_token(usable, cold=False),
    )


def _median_ms_per_token(samples: Sequence[CalibrationSample], *, cold: bool) -> float | None:
    values = sorted(
        s.prompt_eval_ms / s.tokens
        for s in samples
        if s.cold is cold and s.prompt_eval_ms and s.tokens
    )
    if not values:
        return None
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return round(median, 6)


def estimate_with_ratio(text_chars: int, ratio: float) -> int:
    return round(text_chars * ratio)
