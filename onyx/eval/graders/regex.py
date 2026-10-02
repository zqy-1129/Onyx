"""正则抽取 grader：从自由文本里把答案捞出来。

用于 GSM8K 式的"答案在最后一句"、以及"模型加了解释文字"的情况。
两条纪律：
1. **抽不出来返回 None，不是空串**。空串会与"期望值是空串"混淆，
   于是"没抽到答案"被判成"答对了"。
2. 不做"猜一个最像的"兜底。抽取失败是一种明确的结果，
   它应该体现为 `invalid_format`，而不是被悄悄修好。

正则编译失败、组号越界都不抛异常：评测跑到一半因为一个模式崩掉，
比这条样本判错严重得多——但失败会记进 `RegexHit.error`，不许静默。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from onyx.eval.graders.normalize import normalize_number, strip_code_fence


@dataclass(frozen=True, slots=True)
class RegexHit:
    value: str | None
    group: int = 1
    pattern: str = ""
    matched: bool = False
    #: 命中次数。命中多次往往意味着模式写得太松，必须能被看出来
    occurrences: int = 0
    #: 模式本身有问题（编译失败/组号越界）。与"没匹配上"是两回事
    error: str = ""

    @property
    def number(self) -> float | None:
        return normalize_number(self.value) if self.value is not None else None


def find(
    pattern: str,
    text: Any,
    *,
    group: int = 1,
    flags: int = 0,
    multiline: bool = True,
    strip_fence: bool = True,
) -> RegexHit:
    """按 pattern 抽第 `group` 组，取**最后一次**命中。

    取最后一次是因为答案通常在结尾，而开头往往是模型在复述题目。
    """
    raw = "" if text is None else str(text)
    haystack = strip_code_fence(raw) if strip_fence else raw
    try:
        compiled = re.compile(pattern, flags | (re.MULTILINE if multiline else 0))
    except re.error as exc:
        return RegexHit(None, group, pattern, error=f"正则编译失败: {exc}")

    matches = list(compiled.finditer(haystack))
    if not matches:
        return RegexHit(None, group, pattern, matched=False, occurrences=0)
    try:
        value = matches[-1].group(group)
    except (IndexError, re.error) as exc:
        return RegexHit(
            None, group, pattern, occurrences=len(matches), error=f"捕获组 {group} 不可用: {exc}"
        )
    return RegexHit(
        value=value, group=group, pattern=pattern,
        matched=value is not None, occurrences=len(matches),
    )


def first_of(patterns: Sequence[str], text: Any, *, group: int = 1) -> RegexHit:
    """按顺序试多个模式，第一个抽到值的胜出。

    用于兼容模型的不同表达习惯（`答案：X` / `答案是 X` / `**X**`）。
    顺序即优先级，所以把最严格的模式放前面。
    """
    errors: list[str] = []
    for pattern in patterns:
        hit = find(pattern, text, group=group)
        if hit.matched:
            return hit
        if hit.error:
            errors.append(hit.error)
    return RegexHit(
        None, group, "|".join(patterns), matched=False, occurrences=0, error="; ".join(errors)
    )
