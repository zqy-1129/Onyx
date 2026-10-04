"""Provider 注册与发现（扩展点 `onyx.providers`）。

内置实现静态注册，外部插件通过 entry point 注册：
    [project.entry-points."onyx.providers"]
    my_engine = "my_pkg.provider:MyProvider"
发现语义（坏插件隔离、失败可见、同名覆盖）由 `onyx.discovery` 统一实现，
六个扩展点共用同一套，不会出现"provider 组容忍坏插件、task 组不容忍"这种分裂。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from onyx.discovery import GROUP_PROVIDERS
from onyx.discovery import discover as discover_entries
from onyx.llm.providers.base import LlmProvider


def _builtin_ollama(**kwargs: Any) -> LlmProvider:
    from onyx.llm.providers.ollama import OllamaProvider

    return OllamaProvider(**kwargs)


def _builtin_mock(**kwargs: Any) -> LlmProvider:
    from onyx.llm.providers.mock import MockProvider

    return MockProvider(**kwargs)


def _builtin_openai_compat(**kwargs: Any) -> LlmProvider:
    from onyx.llm.providers.openai_compat import OpenAICompatProvider

    return OpenAICompatProvider(**kwargs)


BUILTIN: dict[str, Callable[..., LlmProvider]] = {
    "ollama": _builtin_ollama,
    "mock": _builtin_mock,
    "openai-compat": _builtin_openai_compat,
}


def discover() -> dict[str, Callable[..., LlmProvider]]:
    """内置 + 插件。同名时插件覆盖内置（便于就地替换实现做实验）。"""
    return discover_entries(GROUP_PROVIDERS, BUILTIN)


def available_kinds() -> list[str]:
    return sorted(discover())


def build_provider(kind: str, **kwargs: Any) -> LlmProvider:
    registry = discover()
    if kind not in registry:
        raise KeyError(f"未知 provider kind: {kind!r}，可用: {sorted(registry)}")
    return registry[kind](**kwargs)
