"""echo：确定性、幂等的参照工具，契约测试用它验证"两次调用结果一致"。"""

from __future__ import annotations

from typing import Any

MAX_CHARS = 4000


def echo(text: str = "", times: int = 1) -> dict[str, Any]:
    if not isinstance(text, str):
        text = str(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    times = max(1, min(int(times), 10))
    return {"echo": text, "times": times, "chars": len(text)}


def slow_echo(text: str = "", delay_ms: int = 50) -> dict[str, Any]:
    """故意慢的参照实现：用来验证 deadline 真的会触发 ToolTimeout。

    没有它，"超时"这条契约断言就只能靠 mock 假装——而假装出来的超时
    验证不了 `run_with_deadline` 的实际行为。
    """
    import time

    delay_ms = max(0, min(int(delay_ms), 5000))
    time.sleep(delay_ms / 1000)
    return {"echo": text, "delay_ms": delay_ms}


def boom(reason: str = "契约测试触发的实现崩溃") -> dict[str, Any]:
    """故意崩的参照实现：用来验证 `error` 这一档真的存在。

    没有它，测试就无法把"实现崩了"与"参数不合法"区分开——而这两者的
    归因完全相反（前者是工具的缺陷，后者是模型的输出问题）。
    """
    raise RuntimeError(reason)
