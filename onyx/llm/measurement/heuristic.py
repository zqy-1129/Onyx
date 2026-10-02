"""T5 启发式估计：永远可用的兜底档。

`chars/4` 对中文严重低估（DESIGN R8），所以按字符类别分开算：
- CJK 逐字计，系数可配（不同 tokenizer 差异很大：Qwen2 大词表 ≈0.7 token/字，
  cl100k 系 ≈1.5，小词表可到 2+）
- 其余按 chars/latin_chars_per_token

**这一档必须标 `confidence=low`**，并且应当被 `calibrate.fit_ratio` 的实测值取代。
"""

from __future__ import annotations

import re

#: CJK 统一表意文字 + 扩展A + 兼容表意 + 中日韩标点 + 全角形式
CJK_RE = re.compile(
    "[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]"
)

DEFAULT_CJK_TOKENS_PER_CHAR = 1.0
DEFAULT_LATIN_CHARS_PER_TOKEN = 4.0


def split_cjk(text: str) -> tuple[int, int]:
    """返回 (CJK 字符数, 其他字符数)。"""
    cjk = len(CJK_RE.findall(text))
    return cjk, len(text) - cjk


def estimate_tokens(
    text: str,
    *,
    cjk_tokens_per_char: float = DEFAULT_CJK_TOKENS_PER_CHAR,
    latin_chars_per_token: float = DEFAULT_LATIN_CHARS_PER_TOKEN,
) -> int:
    if not text:
        return 0
    cjk, other = split_cjk(text)
    tokens = cjk * cjk_tokens_per_char + (other / latin_chars_per_token if other else 0.0)
    return round(tokens)


def text_tokens(text: str) -> int:
    return estimate_tokens(text)
