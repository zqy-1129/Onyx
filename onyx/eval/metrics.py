"""评测指标：全部纯函数，全部可用手算表格验证。

三条纪律：
1. **零除返回 None 而不是 0**。没有正例时 F1 是"未定义"，不是"0 分"；
   填 0 会让一个根本没被考到的类把 macro 平均拖下去，看起来像模型能力差。
2. **CI 必须真的在算**。bootstrap 重采样单位是 case，统计量每次重算——
   返回常数区间的实现比不返回区间更危险。
3. 样本少时区间就该宽。`n < LOW_CONFIDENCE_N` 时调用方必须标注低置信（DESIGN §9.3）。
"""

from __future__ import annotations

import enum
import math
import random
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

#: 低于这个样本量，CI 只能说明"测过了"，不足以支撑决策；UI 必须标注（DESIGN §9.3）
LOW_CONFIDENCE_N = 100


@dataclass(frozen=True, slots=True)
class PRF1:
    precision: float | None
    recall: float | None
    f1: float | None
    support: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def is_defined(self) -> bool:
        return self.f1 is not None


def prf1(tp: int, fp: int, fn: int, *, support: int | None = None) -> PRF1:
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1: float | None = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return PRF1(
        precision=precision, recall=recall, f1=f1,
        support=(tp + fn) if support is None else support,
        tp=tp, fp=fp, fn=fn,
    )


def accuracy(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    """pairs = [(期望, 实际)]。空集返回 None（没考过，不是 0 分）。"""
    if not pairs:
        return None
    return sum(1 for expected, actual in pairs if expected == actual) / len(pairs)


def confusion(pairs: Sequence[tuple[Any, Any]]) -> dict[Any, dict[Any, int]]:
    """混淆矩阵 `matrix[期望][实际] = 次数`。

    行列的键集是**出现过的全部标签**，不是只出现过的期望标签：
    模型幻觉出一个训练集里没有的标签时，它必须能在矩阵里被看见，
    而不是被静默丢掉——那正是"越界标签率"要抓的东西。
    """
    matrix: dict[Any, dict[Any, int]] = {}
    labels = {expected for expected, _ in pairs} | {actual for _, actual in pairs}
    for label in labels:
        matrix[label] = dict.fromkeys(labels, 0)
    for expected, actual in pairs:
        matrix[expected][actual] += 1
    return matrix


def per_class_prf1(pairs: Sequence[tuple[Any, Any]]) -> dict[Any, PRF1]:
    """逐类 one-vs-rest 的 P/R/F1。"""
    matrix = confusion(pairs)
    labels = sorted(matrix, key=str)
    out: dict[Any, PRF1] = {}
    for label in labels:
        tp = matrix[label][label]
        fn = sum(count for other, count in matrix[label].items() if other != label)
        fp = sum(matrix[other][label] for other in labels if other != label)
        out[label] = prf1(tp, fp, fn)
    return out


def macro(
    scores: Iterable[PRF1 | float | None], *, key: str = "f1"
) -> float | None:
    """宏平均：**只对定义得出来的类求平均**，并如实报告参与平均的类数。

    把 F1=None 当成 0 参与平均是最常见的错误——一个 support=0 的类
    会把 macro_f1 凭空压低，而模型根本没被考过那个类。
    """
    values: list[float] = []
    for item in scores:
        value = getattr(item, key, None) if isinstance(item, PRF1) else item
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def macro_f1(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    return macro(per_class_prf1(pairs).values())


def balanced_accuracy(pairs: Sequence[tuple[Any, Any]]) -> float | None:
    """各类召回率的平均。类别极不均衡时比 accuracy 诚实得多。"""
    per_class = per_class_prf1(pairs)
    recalls = [item.recall for item in per_class.values() if item.recall is not None]
    return sum(recalls) / len(recalls) if recalls else None


def rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def hit_at_k(expected: Any, ranked: Sequence[Any], k: int = 1) -> bool:
    """期望项是否出现在前 k 个里。`ranked` 按模型给出的顺序。"""
    return expected in list(ranked)[: max(0, k)]


def pass_hat_k(samples_per_case: Sequence[Sequence[bool]]) -> float | None:
    """pass^k：k 次采样**全对**的 case 占比。衡量"稳定可用"。

    与 pass@k 的差就是稳定性缺口：0.61 / 0.78 意味着 17pp 的情况下
    模型能答对但不能每次都答对——可用，但不可靠。
    """
    usable = [samples for samples in samples_per_case if samples]
    if not usable:
        return None
    return sum(1 for samples in usable if all(samples)) / len(usable)


def pass_at_k(samples_per_case: Sequence[Sequence[bool]]) -> float | None:
    """pass@k：k 次采样**至少一次对**的 case 占比。衡量"能力上限"。"""
    usable = [samples for samples in samples_per_case if samples]
    if not usable:
        return None
    return sum(1 for samples in usable if any(samples)) / len(usable)


def stability_gap(samples_per_case: Sequence[Sequence[bool]]) -> float | None:
    gap_at = pass_at_k(samples_per_case)
    gap_hat = pass_hat_k(samples_per_case)
    if gap_at is None or gap_hat is None:
        return None
    return gap_at - gap_hat


@dataclass(frozen=True, slots=True)
class CI:
    low: float | None
    high: float | None
    point: float | None
    iterations: int = 0
    n: int = 0
    method: str = "bootstrap"

    @property
    def low_confidence(self) -> bool:
        return self.n < LOW_CONFIDENCE_N

    def as_dict(self) -> dict[str, Any]:
        return {
            "low": self.low, "high": self.high, "point": self.point,
            "iterations": self.iterations, "n": self.n, "method": self.method,
            "low_confidence": self.low_confidence,
        }

    def format(self, digits: int = 3) -> str:
        if self.point is None:
            return "—"
        if self.low is None or self.high is None:
            return f"{self.point:.{digits}f}"
        flag = " ⚠低样本" if self.low_confidence else ""
        return (
            f"{self.point:.{digits}f} [95% CI {self.low:.{digits}f}–{self.high:.{digits}f}]"
            f" (n={self.n}){flag}"
        )


def jsonable(value: Any) -> Any:
    """把聚合结果转成可直接 JSON 序列化的形状。

    **必须在落库与 `--json` 输出之前调用**。`CI` 这类 dataclass 走
    `json.dumps(default=str)` 会变成 `"CI(low=0.97, ...)"` 这样一个字符串——
    写进去不报错，读出来既不能取下界也不能取上界，置信区间就静默消失了。
    """
    if isinstance(value, CI):
        return value.as_dict()
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, set | frozenset):
        return [jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, enum.Enum):
        return str(value.value)
    return value


def bootstrap_ci(
    n: int,
    statistic: Callable[[list[int]], float | None],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> CI:
    """百分位 bootstrap。`statistic(indices)` 每次都用重采样出的下标**重算**。

    重采样单位是 case，不是分数：把已经算好的分数直接重排求均值，
    对 macro_f1 这类非线性统计量会得到错误（而且看起来合理）的区间。

    `seed` 默认固定，所以同一份数据两次跑出的区间一致——
    区间自己会抖动的话，就没法用它做回归判断了。

    **零宽区间不等于确定**：20 条全对时，每次重采样仍然全对，区间就是 [1.0, 1.0]。
    这是退化的 bootstrap，不是"置信度 100%"。所以 `low_confidence` 必须跟着一起看，
    UI 上也不许把零宽区间渲染成"没有不确定性"。
    """
    point = statistic(list(range(n))) if n else None
    if n <= 1 or iterations <= 0:
        # n<=1 时重采样永远得到同一个样本，区间宽度为 0，那是假的确定性
        return CI(low=None, high=None, point=point, iterations=0, n=n)

    rng = random.Random(seed)
    values: list[float] = []
    for _ in range(iterations):
        sample = [rng.randrange(n) for _ in range(n)]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if not values:
        return CI(low=None, high=None, point=point, iterations=iterations, n=n)
    values.sort()
    low = _percentile(values, alpha / 2)
    high = _percentile(values, 1 - alpha / 2)
    return CI(low=low, high=high, point=point, iterations=iterations, n=n)


def mean_ci(
    values: Sequence[float | None], *, iterations: int = 2000, seed: int = 0
) -> CI:
    """均值的 bootstrap CI。None 项被排除，且 `n` 报的是**参与计算的条数**。"""
    usable = [float(v) for v in values if v is not None]

    def statistic(indices: list[int]) -> float | None:
        return sum(usable[i] for i in indices) / len(indices) if indices else None

    return bootstrap_ci(len(usable), statistic, iterations=iterations, seed=seed)


def macro_f1_ci(
    pairs: Sequence[tuple[Any, Any]], *, iterations: int = 2000, seed: int = 0
) -> CI:
    """macro_f1 的 CI：每个 bootstrap 样本里**重算混淆矩阵与逐类 F1**。"""

    def statistic(indices: list[int]) -> float | None:
        return macro_f1([pairs[i] for i in indices])

    return bootstrap_ci(len(pairs), statistic, iterations=iterations, seed=seed)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数。q ∈ [0,1]。"""
    if not sorted_values:
        raise ValueError("空序列没有分位数")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def summarize_pairs(
    pairs: Sequence[tuple[Any, Any]], *, iterations: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """分类任务的一站式汇总（intent_classification 直接用）。"""
    per_class = per_class_prf1(pairs)
    return {
        "n": len(pairs),
        "accuracy": accuracy(pairs),
        "macro_f1": macro_f1(pairs),
        "macro_f1_ci": macro_f1_ci(pairs, iterations=iterations, seed=seed),
        "balanced_accuracy": balanced_accuracy(pairs),
        "per_class": {str(label): item for label, item in per_class.items()},
        "confusion": {str(a): {str(b): c for b, c in row.items()}
                      for a, row in confusion(pairs).items()},
        "labels": sorted({str(expected) for expected, _ in pairs}),
        "low_confidence": len(pairs) < LOW_CONFIDENCE_N,
    }


def top_confusions(
    pairs: Sequence[tuple[Any, Any]], *, limit: int = 3
) -> list[tuple[Any, Any, int]]:
    """最容易混的标签对，按次数降序。

    只看 macro_f1 不知道该改什么；"混淆集中在 投诉↔其他"才是可行动的结论。
    """
    counts: Counter[tuple[Any, Any]] = Counter()
    for expected, actual in pairs:
        if expected != actual:
            counts[(expected, actual)] += 1
    return [(a, b, n) for (a, b), n in counts.most_common(limit)]
