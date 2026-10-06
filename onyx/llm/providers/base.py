"""Provider 契约。

任何引擎（Ollama / vLLM / llama.cpp / LM Studio / 云端）都只实现这一个协议。
`tests/contract/test_provider_contract.py` 对所有实现跑同一套断言——
这就是"可替换"的实际含义：不是文档里写着可替换，而是有一组测试逼着它可替换。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from onyx.core.event import TraceEvent
from onyx.core.types import (
    AdminResult,
    Cap,
    Embedding,
    EmbedRequest,
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
        self,
        req: GenerationRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Generation: ...
    # `trace_id` 必须在协议里：gateway 一律用关键字传它（事件要挂到正确的那条 trace）。
    # 协议里没写、实现里却有 = 照着协议写的外部 provider 一定崩在 TypeError 上。
    # `tests/contract/test_provider_contract.py` 用签名断言把这条钉住。


class AdminProvider(Protocol):
    """可选能力：运行时治理（加载/卸载/拉取/删除）。"""

    def pull(self, name: str, *, on_event: EventCB | None = None) -> AdminResult: ...
    def unload(self, name: str) -> AdminResult: ...
    def delete(self, name: str) -> AdminResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """可选能力：把一批文本向量化。**刻意不并入 `LlmProvider`**。

    并进去就是逼每个 provider 假装支持：`openai_compat` 的 `/v1/embeddings` 本机没实测过，
    外部插件更未必有——而协议里每个方法都得有契约测试兜着（S16a 的教训：
    内置实现与内核一起过拟合，只有照协议写的外部实现会把抽象泄漏顶出来）。
    所以这里是"有就声明、没有就明确报 unsupported"，gateway 不做隐式降级。
    """

    def embed(
        self,
        req: EmbedRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Embedding: ...


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
