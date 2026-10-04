"""Onyx 扩展点样板：外部 provider 插件。

装好这个包（`uv run --with-editable plugins_example/example_provider onyx plugins`）后，
`echo` 就会出现在 `onyx plugins` 的 `onyx.providers` 行里，并能直接
`onyx chat --provider echo` —— **而内核一行没改**：
`onyx/core/**`、`onyx/llm/gateway.py`、`onyx/obs/**` 都不需要动
（这条由 `scripts/check_extension_boundary.py` 断言，不靠人 review）。

它是"只实现 `LlmProvider` 协议"的活体证明：确定性地回显最后一条用户消息，
不报 usage —— 于是 token 走本地复算档位，看板显示 heuristic + 低置信度。
「未知不等于 0」在插件上同样成立，这就是那条设计原则的验收。
"""

from .provider import EchoProvider

__all__ = ["EchoProvider"]
