"""模糊匹配 grader。

只用于**自由文本**字段（摘要、解释、地址）。分类标签、数值、枚举一律不该走这里——
用模糊匹配判标签，等于把"差一点对"算成对，而分类任务里差一点就是错。

后端有两个：`rapidfuzz`（`bench` extra）与 stdlib 的 `difflib`。
两者算法不同（RapidFuzz 的 ratio 是 Indel 相似度，SequenceMatcher 是基于最长匹配块），
**所以结果里必须报用的是哪个**：混用会让两次运行的分数不可比，
而这种不可比看起来就像模型变好或变差了。
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from onyx.eval.graders.normalize import normalize_text

STDLIB_BACKEND = "difflib"
RAPIDFUZZ_BACKEND = "rapidfuzz"

#: 低于这个相似度基本可以认为答的不是同一个东西
DEFAULT_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class FuzzResult:
    ratio: float
    passed: bool
    backend: str
    threshold: float
    expected: str = ""
    actual: str = ""


def _rapidfuzz():
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return None
    return fuzz


def ratio(expected: Any, actual: Any, *, casefold: bool = True) -> tuple[float, str]:
    """返回 (相似度, 后端名)。相似度 ∈ [0,1]。"""
    left = normalize_text(expected if expected is not None else "", casefold=casefold)
    right = normalize_text(actual if actual is not None else "", casefold=casefold)
    if not left and not right:
        return 1.0, STDLIB_BACKEND
    if not left or not right:
        return 0.0, STDLIB_BACKEND
    fuzz = _rapidfuzz()
    if fuzz is not None:
        return fuzz.ratio(left, right) / 100.0, RAPIDFUZZ_BACKEND
    return SequenceMatcher(None, left, right).ratio(), STDLIB_BACKEND


def fuzzy_match(
    expected: Any,
    actual: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    casefold: bool = True,
) -> FuzzResult:
    score, backend = ratio(expected, actual, casefold=casefold)
    return FuzzResult(
        ratio=round(score, 6), passed=score >= threshold, backend=backend, threshold=threshold,
        expected=normalize_text(expected or "", casefold=casefold)[:200],
        actual=normalize_text(actual or "", casefold=casefold)[:200],
    )


def contains_all(actual: Any, needles: list[str], *, casefold: bool = True) -> dict[str, Any]:
    """关键词全覆盖检查（instruction_following 的"必须包含"约束用它）。

    返回逐关键词结果，而不是一个布尔：**缺了哪个**才是可行动的结论。
    """
    haystack = normalize_text(actual if actual is not None else "", casefold=casefold)
    per_needle = {
        needle: normalize_text(needle, casefold=casefold) in haystack for needle in needles
    }
    missing = sorted(name for name, ok in per_needle.items() if not ok)
    return {
        "per_needle": per_needle,
        "missing": missing,
        "passed": not missing,
        "score": (len(per_needle) - len(missing)) / len(per_needle) if per_needle else None,
    }


def contains_none(actual: Any, banned: list[str], *, casefold: bool = True) -> dict[str, Any]:
    """禁用词检查。命中的词必须列出来。"""
    haystack = normalize_text(actual if actual is not None else "", casefold=casefold)
    hits = sorted(
        word for word in banned if normalize_text(word, casefold=casefold) in haystack
    )
    return {
        "hits": hits,
        "passed": not hits,
        "score": 1.0 if not hits else 0.0,
    }
