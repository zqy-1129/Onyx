"""集合匹配 grader：并行工具调用、多标签分类都用它。

集合比对必须给 P/R/F1 三个数，不能只给"全对/不全对"：
漏调一个和多调一个的修法完全不同（前者是覆盖不足，后者是误调），
而"完全匹配率"把这两种错混成同一个 0。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from onyx.eval.graders.normalize import normalize_text


@dataclass(frozen=True, slots=True)
class SetMatch:
    expected: frozenset[str]
    actual: frozenset[str]
    matched: frozenset[str] = frozenset()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    #: 期望集合之外多出来的项。工具场景下这就是"幻觉工具名"，要单独统计
    hallucinated: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def precision(self) -> float | None:
        return len(self.matched) / len(self.actual) if self.actual else None

    @property
    def recall(self) -> float | None:
        return len(self.matched) / len(self.expected) if self.expected else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    @property
    def exact(self) -> bool:
        """完全匹配。空集 == 空集 也算匹配（`no_call_needed` 子集靠这条判定）。"""
        return self.expected == self.actual


def _keys(items: Iterable[Any], *, casefold: bool) -> dict[str, Any]:
    """归一化键 → 原值。归一化后重复的键保留第一个，并在 extra 里能看出来。"""
    out: dict[str, Any] = {}
    for item in items or ():
        key = normalize_text(_stringify(item), casefold=casefold)
        out.setdefault(key, item)
    return out


def _stringify(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # 工具调用常见的形状：{"name": ..., "arguments": {...}}
        name = item.get("name") or item.get("tool") or ""
        return str(name)
    return str(item)


def set_match(
    expected: Iterable[Any],
    actual: Iterable[Any],
    *,
    casefold: bool = True,
) -> SetMatch:
    expected_map = _keys(expected, casefold=casefold)
    actual_map = _keys(actual, casefold=casefold)
    matched = frozenset(expected_map) & frozenset(actual_map)
    missing = tuple(sorted(set(expected_map) - matched))
    unexpected = tuple(sorted(set(actual_map) - matched))
    return SetMatch(
        expected=frozenset(expected_map),
        actual=frozenset(actual_map),
        matched=matched,
        missing=missing,
        unexpected=unexpected,
        hallucinated=unexpected,
        extra={
            "expected_raw": [expected_map[k] for k in sorted(expected_map)],
            "actual_raw": [actual_map[k] for k in sorted(actual_map)],
        },
    )
