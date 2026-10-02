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
    """tokens ≈ ratio × chars + intercept。`r2` 越接近 1 说明这个线性假设越成立。

    **截距不是可选项**：真机实测一条 12 token 消息的请求，引擎报 19 token，
    差的 7 个是模板控制符（角色标记 + 生成提示符）。用"过原点"拟合会把这部分
    固定开销摊进比值里，导致长 prompt 系统性高估、短 prompt 系统性低估。
    带截距拟合同时给出两个有用的量：`ratio`（该模型的真实 tokens/char）
    与 `intercept`（每请求固定的模板开销）。
    """

    ratio: float
    n: int
    r2: float
    max_rel_error: float
    intercept: float = 0.0
    #: 双特征拟合（fit_by_script）时填充：中文与其他字符各自的 token 密度
    cjk_ratio: float | None = None
    other_ratio: float | None = None
    #: 冷/热 prefill 的 ms/token 中位数，用于 prefill_mode 判定阈值（PROBES P11）
    cold_ms_per_token: float | None = None
    warm_ms_per_token: float | None = None

    @property
    def usable(self) -> bool:
        """样本太少、拟合太差、或最坏样本偏差过大 ⇒ 不许当 fitted 档用。

        真机教训：R²=0.94 但最大相对误差 66% 的标定是**有害的**——它看起来可信，
        却会让短请求的估计差一倍。所以 max_rel_error 必须进门槛，不能只看 R²。
        """
        return self.n >= 30 and self.r2 >= 0.90 and self.max_rel_error <= 0.20

    def predict(self, chars: int) -> int:
        return max(0, round(chars * self.ratio + self.intercept))

    def predict_split(self, cjk_chars: int, other_chars: int) -> int:
        if self.cjk_ratio is None or self.other_ratio is None:
            return self.predict(cjk_chars + other_chars)
        return max(0, round(
            self.intercept + self.cjk_ratio * cjk_chars + self.other_ratio * other_chars
        ))


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    chars: int
    tokens: int
    prompt_eval_ms: float | None = None
    cold: bool | None = None
    text: str = ""
    cjk_chars: int | None = None
    other_chars: int | None = None

    def split(self) -> tuple[int, int]:
        """返回 (中文字符数, 其他字符数)。"""
        if self.cjk_chars is not None and self.other_chars is not None:
            return self.cjk_chars, self.other_chars
        if self.text:
            from onyx.llm.measurement.heuristic import split_cjk

            return split_cjk(self.text)
        return 0, self.chars


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


def fit_linear(samples: Sequence[CalibrationSample]) -> CalibrationResult:
    """带截距的最小二乘：tokens ≈ ratio × chars + intercept。

    这是 `onyx calibrate` 实际使用的拟合方式（理由见 CalibrationResult 文档）。
    纯 stdlib 实现，不引入 numpy。
    """
    usable = [s for s in samples if s.chars > 0 and s.tokens > 0]
    n = len(usable)
    if n == 0:
        return CalibrationResult(ratio=0.0, n=0, r2=0.0, max_rel_error=1.0)
    if n == 1:
        only = usable[0]
        return CalibrationResult(
            ratio=only.tokens / only.chars, n=1, r2=1.0, max_rel_error=0.0, intercept=0.0,
            cold_ms_per_token=_median_ms_per_token(usable, cold=True),
            warm_ms_per_token=_median_ms_per_token(usable, cold=False),
        )

    mean_x = sum(s.chars for s in usable) / n
    mean_y = sum(s.tokens for s in usable) / n
    sxx = sum((s.chars - mean_x) ** 2 for s in usable)
    sxy = sum((s.chars - mean_x) * (s.tokens - mean_y) for s in usable)
    ratio = sxy / sxx if sxx else 0.0
    intercept = mean_y - ratio * mean_x

    ss_tot = sum((s.tokens - mean_y) ** 2 for s in usable)
    ss_res = sum((s.tokens - (ratio * s.chars + intercept)) ** 2 for s in usable)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 1.0
    max_rel = max(
        (abs(s.tokens - (ratio * s.chars + intercept)) / s.tokens for s in usable), default=1.0
    )
    return CalibrationResult(
        ratio=round(ratio, 6), n=n, r2=round(r2, 4), max_rel_error=round(max_rel, 4),
        intercept=round(intercept, 4),
        cold_ms_per_token=_median_ms_per_token(usable, cold=True),
        warm_ms_per_token=_median_ms_per_token(usable, cold=False),
    )


def fit_by_script(samples: Sequence[CalibrationSample]) -> CalibrationResult:
    """双特征拟合：tokens ≈ intercept + cjk_ratio×中文字符 + other_ratio×其他字符。

    **为什么单比值是错的**：真机标定实测，用混合中英文语料按长度截断采样时，
    短样本几乎纯中文（≈0.7 token/字）、长样本含大量英文（≈0.22 token/字），
    拟合出的"比值"反映的是**采样方式**而不是模型属性——R² 高达 0.94，
    最大相对误差却有 66%。中文与拉丁字符的 token 密度差 3 倍以上，必须分开拟合。

    纯 stdlib：解 3×3 正规方程（高斯消元）。语料里没有中文时矩阵奇异，
    自动降级到 `fit_linear`——宁可给一个粗糙但诚实的比值，也不要伪造双特征结果。
    """
    usable = [s for s in samples if s.chars > 0 and s.tokens > 0]
    n = len(usable)
    if n < 3:
        return CalibrationResult(ratio=0.0, n=n, r2=0.0, max_rel_error=1.0)

    rows = [(1.0, float(s.split()[0]), float(s.split()[1]), float(s.tokens)) for s in usable]
    if not any(r[1] for r in rows):
        return fit_linear(usable)  # 语料无中文，双特征退化
    coef = _solve_normal_equations(rows, features=3)
    if coef is None:
        return fit_linear(usable)
    intercept, cjk_ratio, other_ratio = coef

    def predict(row: tuple[float, float, float, float]) -> float:
        return intercept + cjk_ratio * row[1] + other_ratio * row[2]

    mean_y = sum(r[3] for r in rows) / n
    ss_tot = sum((r[3] - mean_y) ** 2 for r in rows)
    ss_res = sum((r[3] - predict(r)) ** 2 for r in rows)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 1.0
    max_rel = max(abs(r[3] - predict(r)) / r[3] for r in rows)
    return CalibrationResult(
        ratio=round(other_ratio, 6), n=n, r2=round(r2, 4), max_rel_error=round(max_rel, 4),
        intercept=round(intercept, 4),
        cjk_ratio=round(cjk_ratio, 6), other_ratio=round(other_ratio, 6),
        cold_ms_per_token=_median_ms_per_token(usable, cold=True),
        warm_ms_per_token=_median_ms_per_token(usable, cold=False),
    )


def _solve_normal_equations(
    rows: Sequence[tuple[float, ...]], *, features: int
) -> tuple[float, ...] | None:
    """最小二乘：解 (XᵀX)β = Xᵀy。矩阵奇异时返回 None 让调用方降级。"""
    xtx = [[0.0] * features for _ in range(features)]
    xty = [0.0] * features
    for row in rows:
        x, y = row[:features], row[features]
        for i in range(features):
            xty[i] += x[i] * y
            for j in range(features):
                xtx[i][j] += x[i] * x[j]
    augmented = [xtx[i] + [xty[i]] for i in range(features)]
    for col in range(features):
        pivot = max(range(col, features), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-9:
            return None
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        for row in range(features):
            if row == col:
                continue
            factor = augmented[row][col] / augmented[col][col]
            if factor:
                for k in range(col, features + 1):
                    augmented[row][k] -= factor * augmented[col][k]
    return tuple(augmented[i][features] / augmented[i][i] for i in range(features))


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
