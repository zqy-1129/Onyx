"""分位数：全项目只有这一份定义。

为什么单独成一个模块而不是各算各的：`eval/metrics.py` 的 bootstrap 置信区间、
`llm/measurement/calibrate.py` 的冷热中位数、以及 `onyx perf` 的延迟/吞吐聚合，
量的都是"同一批数落在哪里"。三处各写一遍线性插值的话，换一种插值口径就会让
**同一个数据集的 P95 在两个页面上不一样**，而那种不一致没人能排查——
两边都能自证"我按定义算的"。

口径选择：**线性插值**（与 `eval/metrics._percentile` 原本的算法逐字一致）。
它是"最近秩"之外的另一种常见定义，取哪种不是对错，是约定；一旦约定了就必须只有一处。
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """线性插值分位数。`q ∈ [0,1]`，输入**必须已升序**（调用方负责排序，代价说明见异常）。"""
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


def median(values: Sequence[float]) -> float:
    return percentile(sorted(values), 0.5)


def spread(values: Sequence[float]) -> dict[str, float | int]:
    """一组数的落点摘要：n / 中位 / p95 / 最小 / 最大。

    空序列**抛异常而不是返回 0**：0 会被读成"这一格的延迟是 0ms"，
    而事实是"这一格一条都没测到"——那是两个相反的说法（本项目反复踩的那条线）。
    要表达"没测到"请调用 `spread_or_none()`。
    """
    if not values:
        raise ValueError("空序列没有落点：这里要的是「没测到」，不是 0")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "median": percentile(ordered, 0.5),
        "p95": percentile(ordered, 0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def spread_or_none(values: Sequence[float]) -> dict[str, float | int] | None:
    """没测到就是 `None`，落库/渲染时是「—」。"""
    return spread(values) if values else None
