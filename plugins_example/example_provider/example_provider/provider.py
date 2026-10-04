"""EchoProvider：一个**只实现 `LlmProvider`** 的外部引擎。

存在的意义不是"有用"，而是"可证伪"：如果接入它需要改内核，说明抽象泄漏。
所以它刻意：

- 不实现 `AdminProvider`（没有 pull/unload）→ 控制面必须优雅地显示"不支持"，
  而不是崩；
- **不报 usage** → 看板的 token 必须走本地复算档位并标低置信度，
  而不是显示一个 0；
- 不声明 TOOLS/THINKING 等能力位 → 评测按能力位 skip 并写明原因（DESIGN §9.1），
  工具任务在这个引擎上必须"整任务跳过且留下记录"，而不是 0 分。
"""

from __future__ import annotations

from typing import Any

from onyx.core.types import (
    ApiStyle,
    Cap,
    FinishReason,
    Generation,
    GenerationRequest,
    LoadedModel,
    ModelCard,
    ModelDetail,
    ProviderInfo,
    ProviderKind,
    Status,
)
from onyx.llm.providers.base import EventCB

#: 只有 CHAT：能力位是评测与探针的分支依据（不是 kind，不是名字匹配）
ECHO_CAPS: frozenset[Cap] = frozenset({Cap.CHAT})


class EchoProvider:
    """确定性回显"引擎"。`generate` 返回最后一条用户消息的回显，零网络、零随机。"""

    kind = ProviderKind.MOCK

    def __init__(
        self,
        id: str = "echo",
        base_url: str = "echo://",
        *,
        models: tuple[str, ...] | list[str] = ("echo/static",),
        **_extra: Any,
    ) -> None:
        self.id = id
        self.base_url = base_url
        self.models = tuple(models)
        #: 供扩展点测试断言"请求确实经过了 provider"，与内置 provider 同形
        self.calls: list[GenerationRequest] = []

    # ── 元信息 ────────────────────────────────────────────────────
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id, kind=self.kind, base_url=self.base_url,
            api_style=ApiStyle.NATIVE, version="0.1.0-example",
            reachable=True, caps=self.capabilities(),
            extra={"plugin": "onyx-example-provider"},
        )

    def capabilities(self) -> frozenset[Cap]:
        return ECHO_CAPS

    def list_models(self) -> list[ModelCard]:
        return [
            ModelCard(
                provider_id=self.id, name=name, capabilities=("completion",),
                context_length=2048,
            )
            for name in self.models
        ]

    def show_model(self, name: str) -> ModelDetail:
        # 未知模型必须报错：返回一个编出来的 detail 会让"这个模型有什么模板"
        # 变成无法证伪的问题
        if name not in self.models:
            raise LookupError(f"echo provider 没有模型 {name!r}；有: {list(self.models)}")
        return ModelDetail(
            name=name, template="{{ .Prompt }}", capabilities=("completion",),
            extra={"example_plugin": True},
        )

    def running(self) -> list[LoadedModel]:
        """没有常驻概念 → 空列表。返回 None 会让 Fleet 页把"没载入"和"不知道"混起来。"""
        return []

    # ── 数据面 ────────────────────────────────────────────────────
    def generate(
        self,
        req: GenerationRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Generation:
        # `trace_id` 与 `on_event` 都必须是关键字参数且**接受但不使用**：
        # gateway 一律按 `generate(req, trace_id=..., on_event=...)` 调用。
        self.calls.append(req)
        prompt = req.messages[-1].content if req.messages else ""
        if req.tools and Cap.TOOLS not in self.capabilities():
            # 声明了工具却没有能力：把事实原样返回（不假装调了工具），
            # 由观测层记成"未产出工具调用"，评测层按能力位 skip
            return Generation(
                text=f"echo(no-tools): {prompt}", model=req.model,
                status=Status.OK, finish_reason=FinishReason.STOP,
                extra={"trace_id": trace_id, "tools_requested": len(req.tools)},
            )
        return Generation(
            text=f"echo: {prompt}", model=req.model,
            status=Status.OK, finish_reason=FinishReason.STOP,
            extra={"trace_id": trace_id},
        )
