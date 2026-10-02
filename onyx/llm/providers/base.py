"""Provider 契约。

任何引擎（Ollama / vLLM / llama.cpp / LM Studio / 云端）都只实现这一个协议。
`tests/contract/test_provider_contract.py` 对所有实现跑同一套断言——
这就是"可替换"的实际含义：不是文档里写着可替换，而是有一组测试逼着它可替换。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Protocol, runtime_checkable

from onyx.core.event import TraceEvent
from onyx.core.types import (
    AdminResult,
    Cap,
    Generation,
    GenerationRequest,
    LoadedModel,
    ModelCard,
    ModelDetail,
    ProviderInfo,
)

EventCB = Callable[[TraceEvent], None]


@runtime_checkable
class LlmProvider(Protocol):
    id: str

    def info(self) -> ProviderInfo: ...
    def capabilities(self) -> frozenset[Cap]: ...
    def list_models(self) -> list[ModelCard]: ...
    def show_model(self, name: str) -> ModelDetail: ...
    def running(self) -> list[LoadedModel]: ...
    def generate(
        self, req: GenerationRequest, *, on_event: EventCB | None = None
    ) -> Generation: ...


class StreamingProvider(Protocol):
    """可选能力：支持增量流。不支持时 gateway 退化为一次性返回。"""

    def generate_stream(
        self, req: GenerationRequest, *, on_event: EventCB | None = None
    ) -> Iterator[TraceEvent]: ...


class AdminProvider(Protocol):
    """可选能力：运行时治理（加载/卸载/拉取/删除）。"""

    def pull(self, name: str, *, on_event: EventCB | None = None) -> AdminResult: ...
    def unload(self, name: str) -> AdminResult: ...
    def delete(self, name: str) -> AdminResult: ...


def emit(on_event: EventCB | None, event: TraceEvent) -> None:
    """事件回调的容错包装：观测回调抛错不许打断生成。"""
    if on_event is None:
        return
    try:
        on_event(event)
    except Exception:
        import logging

        logging.getLogger("onyx.llm").warning("事件回调失败: type=%s", event.type, exc_info=True)


def unsupported_detail(cap: Cap, provider_id: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"cap": str(cap), "provider_id": provider_id, **(extra or {})}
