"""文本归一化与标签抽取。

本地小模型极少老老实实只吐一个标签，常见形态是：
`「转账」`、`intent: 转账`、`转账（因为用户提到了汇款）`、`TRANSFER`、```json 包起来```。
所以"答案对不对"必须先归一化再比，否则测的是模型的排版习惯而不是能力。

但归一化有一条底线：**文本里同时出现多个候选标签时必须报 ambiguous，
不能挑第一个**。挑第一个等于把"模型说'不是转账，是查余额'"判成"转账"正确，
分数会凭空变高，而且高得看不出来。
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

_FENCE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```$", re.DOTALL)
_WHITESPACE = re.compile(r"\s+")
#: 包裹标签的常见标点与引号（中英全角都要）
_WRAPPERS = "「」『』【】《》\"'“”‘’()（）[]<>:：,，.。!！?？;； \t\r\n"


def strip_code_fence(text: str) -> str:
    """去掉 ```…``` 包裹。模型很爱加，而它会毁掉 JSON 解析与精确匹配。"""
    stripped = text.strip()
    match = _FENCE.match(stripped)
    return match.group(1).strip() if match else stripped


def collapse_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def normalize_text(text: str, *, casefold: bool = True) -> str:
    """通用归一化：NFKC（全角→半角）→ 去代码围栏 → 折叠空白 → 可选小写。"""
    value = unicodedata.normalize("NFKC", str(text))
    value = strip_code_fence(value)
    value = collapse_whitespace(value)
    return value.casefold() if casefold else value


def trim_wrappers(text: str) -> str:
    """剥掉标签外层的引号与标点：`「转账」` → `转账`。"""
    return str(text).strip(_WRAPPERS)


@dataclass(frozen=True, slots=True)
class LabelMatch:
    """一次标签抽取的结果。

    `status` 说明**是怎么匹配上的**，这决定了它能被信任到什么程度：
    - `exact` 原样相等 → 完全可信
    - `normalized` 归一化后相等 → 可信（只是排版差异）
    - `unique_substring` 文本里只出现了一个候选标签 → 可用，但模型加了解释文字
    - `ambiguous` 出现了多个候选 → **不可判定**，绝不能猜
    - `not_found` 一个都没出现 → 越界或答非所问
    - `empty` 输出为空
    """

    label: str | None
    status: str
    found: tuple[str, ...] = ()
    text: str = ""

    @property
    def usable(self) -> bool:
        return self.label is not None and self.status in {"exact", "normalized", "unique_substring"}

    @property
    def ambiguous(self) -> bool:
        return self.status == "ambiguous"


def match_label(text: Any, allowed: Sequence[str], *, casefold: bool = True) -> LabelMatch:
    """从模型输出里抽出它想表达的标签。

    匹配优先级：原样 → 归一化 → 唯一子串。任何一级命中就停，
    所以"越干净的输出走越前面的分支"，`status` 也就自然反映了输出质量。
    """
    raw = "" if text is None else str(text)
    if not raw.strip():
        return LabelMatch(None, "empty", text=raw)

    for label in allowed:
        if raw == label or trim_wrappers(raw) == label:
            return LabelMatch(label, "exact", (label,), raw)

    normalized = normalize_text(raw, casefold=casefold)
    allowed_normalized = {
        label: normalize_text(label, casefold=casefold) for label in allowed
    }
    for label, value in allowed_normalized.items():
        if normalized == value or trim_wrappers(normalized) == value:
            return LabelMatch(label, "normalized", (label,), raw)

    # 附加解释文字：数一数文本里出现了几个候选标签
    hits = tuple(label for label, value in allowed_normalized.items() if value and value in normalized)
    if len(hits) == 1:
        return LabelMatch(hits[0], "unique_substring", hits, raw)
    if len(hits) > 1:
        # 不可判定。挑第一个会让"不是 A，是 B"被判成 A 正确，分数凭空变高
        return LabelMatch(None, "ambiguous", hits, raw)
    return LabelMatch(None, "not_found", (), raw)


def normalize_number(value: Any) -> float | None:
    """把 '1,234.5' / '￥12' / ' 3 ' 之类归一成 float。归一不了返回 None（不是 0）。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"[,\s￥$€£%]", "", value.strip())
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
