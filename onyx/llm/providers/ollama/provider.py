"""OllamaProvider：把原生 API 与控制面组合成 `LlmProvider` 实现。"""

from __future__ import annotations

from typing import Any

from onyx.core.clock import SYSTEM_CLOCK, Clock
from onyx.core.types import (
    AdminResult,
    ApiStyle,
    Cap,
    Generation,
    GenerationRequest,
    LoadedModel,
    ModelCard,
    ModelDetail,
    ProviderInfo,
    ProviderKind,
)
from onyx.llm.providers.base import EventCB
from onyx.llm.providers.ollama import lifecycle, native
from onyx.llm.providers.ollama.client import OllamaClient

#: 原生通道基线能力。模型级能力（vision/thinking）由 `model_caps()` 从 /api/tags 推导。
#: 刻意**不含** tool_choice / n_sampling / logprobs：官方文档明确列为不支持（DESIGN §7.1）。
NATIVE_BASE_CAPS: frozenset[Cap] = frozenset({
    Cap.CHAT, Cap.TOOLS, Cap.STRUCTURED_OUTPUT, Cap.STREAM_USAGE, Cap.ADMIN,
})

_OLLAMA_CAP_TO_CAP: dict[str, Cap] = {
    "completion": Cap.CHAT,
    "tools": Cap.TOOLS,
    "thinking": Cap.THINKING,
    "vision": Cap.VISION,
    "embedding": Cap.EMBED,
}


def model_caps(card: ModelCard) -> frozenset[Cap]:
    """把引擎自报的 capabilities 翻成我们的能力位。未知能力保留在 card.capabilities 里不丢。"""
    return frozenset(_OLLAMA_CAP_TO_CAP[c] for c in card.capabilities if c in _OLLAMA_CAP_TO_CAP)


class OllamaProvider:
    kind = ProviderKind.OLLAMA

    def __init__(
        self,
        id: str = "ollama-local",
        base_url: str = "http://127.0.0.1:11434",
        *,
        api_style: ApiStyle = ApiStyle.NATIVE,
        clock: Clock = SYSTEM_CLOCK,
        media_resolver: native.MediaResolver | None = None,
        client: OllamaClient | None = None,
        timeout: Any = None,
    ) -> None:
        self.id = id
        self.base_url = base_url.rstrip("/")
        self.api_style = api_style
        self.clock = clock
        self.media_resolver = media_resolver
        self.client = client or OllamaClient(self.base_url, **({"timeout": timeout} if timeout else {}))

    # ── 元信息 ────────────────────────────────────────────────────
    def info(self) -> ProviderInfo:
        try:
            ver = lifecycle.version(self.client)
            reachable = True
        except Exception:  # noqa: BLE001 - 体检路径：不可达本身就是合法结果
            ver, reachable = "", False
        return ProviderInfo(
            id=self.id, kind=self.kind, base_url=self.base_url, api_style=self.api_style,
            version=ver, reachable=reachable, caps=self.capabilities(),
        )

    def capabilities(self) -> frozenset[Cap]:
        return NATIVE_BASE_CAPS

    # ── 控制面 ────────────────────────────────────────────────────
    def list_models(self) -> list[ModelCard]:
        return lifecycle.list_models(self.client, self.id)

    def show_model(self, name: str) -> ModelDetail:
        return lifecycle.show_model(self.client, name)

    def running(self) -> list[LoadedModel]:
        return lifecycle.running_models(self.client)

    def pull(self, name: str, *, on_event: EventCB | None = None) -> AdminResult:
        ok, error, last = lifecycle.pull_outcome(lifecycle.pull_model(self.client, name))
        return AdminResult(
            ok=ok, action="pull", error=error,
            detail={"name": name, "digest": last.get("digest", ""),
                    "status": str(last.get("status") or ""),
                    # 收尾形状一起留着：判成失败时第一个要问的就是"引擎最后说了什么"
                    "last": {k: v for k, v in last.items() if k != "_unparsed"}},
        )

    def unload(self, name: str) -> AdminResult:
        return lifecycle.unload_model(self.client, name)

    def delete(self, name: str) -> AdminResult:
        return lifecycle.delete_model(self.client, name)

    # ── 数据面 ────────────────────────────────────────────────────
    def generate(
        self,
        req: GenerationRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Generation:
        """一次生成。`trace_id` 由 gateway 分配；直接调用时可留空（仅事件缺少关联）。"""
        if req.stream:
            return native.generate_streaming(
                self.client, req, trace_id=trace_id, clock=self.clock,
                on_event=on_event, media_resolver=self.media_resolver,
            )
        return native.generate(
            self.client, req, trace_id=trace_id, clock=self.clock,
            on_event=on_event, media_resolver=self.media_resolver,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
