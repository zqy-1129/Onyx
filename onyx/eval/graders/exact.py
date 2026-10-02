"""精确匹配 grader。

"精确"必须先归一化再比：模型输出 `「转账」` 与 `转账` 是同一个答案，
判它错就是在测排版而不是测能力。但归一化只处理**无语义差异**的形态
（代码围栏、全半角、空白、大小写、外层引号），不做任何语义放宽。
"""

from __future__ import annotations

from typing import Any

from onyx.eval.graders.normalize import normalize_number, normalize_text, trim_wrappers


def exact(
    expected: Any,
    actual: Any,
    *,
    casefold: bool = True,
    trim: bool = True,
) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        # bool 是 int 的子类，必须先分开，否则 True == 1 会判对
        return expected is actual or (
            isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
        )
    left, right = _scalar(expected), _scalar(actual)
    left = normalize_text(left, casefold=casefold)
    right = normalize_text(right, casefold=casefold)
    if trim:
        left, right = trim_wrappers(left), trim_wrappers(right)
    return left == right


def numeric(expected: Any, actual: Any, *, tolerance: float = 1e-9, rel: float = 0.0) -> bool:
    """数值比对：绝对容差 + 相对容差。

    归一化失败（不是数字）返回 False，而不是抛异常——
    评测里"模型输出了非数字"是一种正常的失败形态，必须被计成错，不能中断整轮。
    """
    left, right = normalize_number(expected), normalize_number(actual)
    if left is None or right is None:
        return False
    if left == right:
        return True
    return abs(left - right) <= tolerance or abs(left - right) <= rel * max(abs(left), abs(right))


def _scalar(value: Any) -> str:
    return value if isinstance(value, str) else str(value)
