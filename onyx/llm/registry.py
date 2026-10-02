"""Provider 注册与发现（扩展点 `onyx.providers`）。

内置实现静态注册，外部插件通过 entry point 注册：
    [project.entry-points."onyx.providers"]
    my_engine = "my_pkg.provider:MyProvider"
发现失败（插件坏了）只记警告并跳过——一个坏插件不该让整个看板起不来。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from importlib.metadata import EntryPoints, entry_points
from typing import Any

from onyx.llm.providers.base import LlmProvider

log = logging.getLogger("onyx.llm.registry")

ENTRY_POINT_GROUP = "onyx.providers"


def _builtin_ollama(**kwargs: Any) -> LlmProvider:
    from onyx.llm.providers.ollama import OllamaProvider

    return OllamaProvider(**kwargs)


def _builtin_mock(**kwargs: Any) -> LlmProvider:
    from onyx.llm.providers.mock import MockProvider

    return MockProvider(**kwargs)


BUILTIN: dict[str, Callable[..., LlmProvider]] = {
    "ollama": _builtin_ollama,
    "mock": _builtin_mock,
}


def _plugin_entries() -> EntryPoints:
    try:
        return entry_points(group=ENTRY_POINT_GROUP)
    except Exception as exc:  # noqa: BLE001 - 老版本 importlib.metadata 签名不同，退化即可
        log.warning("entry point 发现失败: %s", exc)
        return []  # type: ignore[return-value]


def discover() -> dict[str, Callable[..., LlmProvider]]:
    """内置 + 插件。同名时插件覆盖内置（便于就地替换实现做实验）。"""
    found: dict[str, Callable[..., LlmProvider]] = dict(BUILTIN)
    for entry in _plugin_entries():
        try:
            found[entry.name] = entry.load()
        except Exception as exc:  # noqa: BLE001 - 坏插件隔离
            log.warning("插件 provider %s 加载失败: %s", entry.name, exc)
    return found


def available_kinds() -> list[str]:
    return sorted(discover())


def build_provider(kind: str, **kwargs: Any) -> LlmProvider:
    registry = discover()
    if kind not in registry:
        raise KeyError(f"未知 provider kind: {kind!r}，可用: {sorted(registry)}")
    return registry[kind](**kwargs)
