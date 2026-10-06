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


#: 一个汉字**至少**值多少 token。这是下限而不是估计值：本机 qwen 系实测约 0.68 tok/汉字
#: （PROBES / S32 量过），取 0.5 是往保守方向留余量——判据只会"少测一条"，不会"把没测说成测过"。
MIN_TOKENS_PER_HANZI = 0.5


def min_prompt_tokens(text: str, *, tokens_per_hanzi: float = MIN_TOKENS_PER_HANZI) -> int:
    """正文的 token **下限**（只由汉字数推出）。0 表示推不出来 ⇒ 不该做任何截断判断。

    用途是"引擎回报的数字低于这个下限 ⇒ 它一定切过正文"。S32 那条负控制就是这么抓到的：
    16.8k tok 的正文被 Ollama 裁到 `in_tokens=2050`，比 `num_ctx` 还小，
    于是"`in_tokens ≥ num_ctx` 才算被切"这条判据永远不响，被切的样本反而得了 0 分。
    吞吐基线同样需要它：一个"8k 档"的格子如果被裁到 2k，数字看着完全合理，量的却是别的长度。

    **这不是"用估算冒充测量"**：估算在这里的用途是把样本踢出分母（记 skip / 不记分），
    而不是给任何东西打分或当聚合值。
    """
    cjk, _other = split_cjk(text)
    return int(cjk * tokens_per_hanzi)
