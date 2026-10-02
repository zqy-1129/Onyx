"""L0 领域层：类型、id、时钟、事件契约、错误、内容存储。

本包**禁止**依赖任何第三方库（由 import-linter 的 `core uses stdlib only` 契约强制）。
它是整个系统可移植性的锚点：替换 FastAPI / httpx / 存储引擎都不应触及这里。
"""

from __future__ import annotations

CONTRACT_VERSION = 1

__all__ = [
    "CONTRACT_VERSION",
    "BlobStore",
    "Cap",
    "EngineLatency",
    "EventType",
    "FinishReason",
    "GenParams",
    "Generation",
    "GenerationRequest",
    "LoadedModel",
    "Message",
    "ModelCard",
    "ModelDetail",
    "OnyxError",
    "ProviderInfo",
    "Role",
    "TokenSample",
    "ToolCall",
    "ToolSpec",
    "TraceContext",
    "TraceEvent",
    "new_trace_id",
]


def __getattr__(name: str) -> object:  # pragma: no cover - 惰性转发，避免循环导入
    from . import clock, content, errors, event, ids, types

    for module in (types, event, ids, clock, errors, content):
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
