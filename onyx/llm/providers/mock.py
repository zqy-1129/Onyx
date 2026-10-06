"""MockProvider：脚本化的假引擎。

它**不是**用来替代真实验证的（S3 的纪律：不做 mock-only 开发），而是用来
穷举真实引擎难以稳定复现的边界：引擎不报计数、截断的工具 JSON、幻觉工具名、
冷启动、正文为空只有 thinking、provider 抛错。

关键设计：mock 产出的是 **Ollama 原生形状的 chunk**，并走与真实适配器**同一条**
`consume_chunks` 路径。否则用 mock 测出来的观测管道与真实管道就不是同一个东西。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.core.clock import SYSTEM_CLOCK, Clock
from onyx.core.types import (
    AdminResult,
    ApiStyle,
    Cap,
    Embedding,
    EmbedRequest,
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
from onyx.llm.streaming import StreamAssembler, consume_chunks, emit_final_events

MOCK_CAPS: frozenset[Cap] = frozenset({
    Cap.CHAT, Cap.TOOLS, Cap.THINKING, Cap.STRUCTURED_OUTPUT, Cap.STREAM_USAGE, Cap.ADMIN,
    Cap.EMBED,
})

#: 假向量的维度。16 维够测排名，又不至于让断言里塞一串浮点
EMBED_DIM = 16


def mock_vector(text: str, *, dim: int = EMBED_DIM) -> tuple[float, ...]:
    """由文本 sha256 造一个确定性的单位向量：同文本恒等、不同文本彼此近乎正交。

    刻意**不模拟语义**——那不是 mock 的职责，而让测试依赖"假出来的相似"会更糟：
    它全绿的时候什么都没说，它红了的时候你不知道是判据错还是模型错。
    需要"同义项必须排第 1"这种可控结果时，用 `MockProvider(embeddings={文本: 向量})` 显式给。
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 127.5 - 1.0 for i in range(dim)]
    norm = sum(v * v for v in raw) ** 0.5
    return tuple(v / norm for v in raw) if norm else tuple([0.0] * dim)


@dataclass(frozen=True, slots=True)
class MockScript:
    """一次生成的剧本。所有字段都对应真实引擎里观测到的某种形态。"""

    text: str = "好的，已完成。"
    thinking: str = ""
    #: 每项形如 {"name": "get_weather", "arguments": {...}} 或
    #: {"name": "w", "arguments_fragment": '{"city": "北'}（模拟截断）
    tool_calls: tuple[dict[str, Any], ...] = ()
    in_tokens: int | None = 120
    out_tokens: int | None = 24
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    report_usage: bool = True
    load_ns: int = 1_000_000
    prompt_eval_ns: int = 90_000_000
    eval_ns: int = 700_000_000
    total_ns: int = 800_000_000
    done_reason: str = "stop"
    chunks: int = 3
    unparsed_line: str = ""
    error: str = ""
    raise_exc: Exception | None = None
    extra: dict[str, Any] = field(default_factory=dict)


DEFAULT_SCRIPT = MockScript()


def script_chunks(script: MockScript) -> list[dict[str, Any]]:
    """把剧本展开成 Ollama 原生 ndjson 事件序列。"""
    out: list[dict[str, Any]] = []

    def content_chunk(text: str) -> dict[str, Any]:
        return {"model": "mock", "created_at": "2026-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": text}, "done": False}

    if script.thinking:
        for piece in _split(script.thinking, script.chunks):
            out.append({"model": "mock", "created_at": "2026-01-01T00:00:00Z",
                        "message": {"role": "assistant", "thinking": piece}, "done": False})
    for piece in _split(script.text, script.chunks):
        out.append(content_chunk(piece))
    if script.tool_calls:
        calls = []
        for call in script.tool_calls:
            function: dict[str, Any] = {"name": call.get("name", "")}
            if "arguments" in call:
                function["arguments"] = call["arguments"]
            elif "arguments_fragment" in call:
                function["arguments"] = call["arguments_fragment"]
            calls.append({"function": function})
        out.append({"model": "mock", "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "tool_calls": calls}, "done": False})
    if script.unparsed_line:
        out.append({"_unparsed": script.unparsed_line})

    final: dict[str, Any] = {
        "model": "mock", "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": True, "done_reason": script.done_reason,
        "total_duration": script.total_ns, "load_duration": script.load_ns,
        "prompt_eval_duration": script.prompt_eval_ns, "eval_duration": script.eval_ns,
    }
    if script.report_usage:
        final["prompt_eval_count"] = script.in_tokens
        final["eval_count"] = script.out_tokens
        if script.thinking_tokens is not None:
            final["thinking_eval_count"] = script.thinking_tokens
        if script.cached_tokens is not None:
            final["prompt_eval_cached_count"] = script.cached_tokens
    if script.error:
        final["error"] = script.error
    out.append(final)
    return out


def _split(text: str, pieces: int) -> list[str]:
    if not text:
        return []
    pieces = max(1, pieces)
    size = max(1, len(text) // pieces)
    return [text[i : i + size] for i in range(0, len(text), size)] or [text]


class MockProvider:
    kind = ProviderKind.MOCK

    def __init__(
        self,
        id: str = "mock",
        base_url: str = "mock://",
        scripts: dict[str, MockScript | Sequence[MockScript]] | None = None,
        *,
        default: MockScript = DEFAULT_SCRIPT,
        clock: Clock = SYSTEM_CLOCK,
        models: tuple[str, ...] = ("mock/echo",),
        embeddings: dict[str, Sequence[float]] | None = None,
    ) -> None:
        self.id = id
        self.base_url = base_url  # 仅为与真实 provider 的构造签名兼容，不发起任何请求
        self.scripts = dict(scripts or {})
        self.default = default
        self.clock = clock
        self.models = models
        self.calls: list[GenerationRequest] = []
        #: 显式给定的向量（文本 → 向量）：排名判据的测试要能指定"谁排第几"
        self.embeddings = {k: tuple(float(x) for x in v) for k, v in (embeddings or {}).items()}
        self.embed_calls: list[EmbedRequest] = []
        #: 序列脚本的消费游标：model → 已消费条数
        self._cursors: dict[str, int] = {}

    # ── 元信息 ────────────────────────────────────────────────────
    def info(self) -> ProviderInfo:
        return ProviderInfo(
            id=self.id, kind=self.kind, base_url=self.base_url, api_style=ApiStyle.NATIVE,
            version="0.0.0-mock", reachable=True, caps=self.capabilities(),
        )

    def capabilities(self) -> frozenset[Cap]:
        return MOCK_CAPS

    def list_models(self) -> list[ModelCard]:
        return [
            ModelCard(provider_id=self.id, name=name, capabilities=("completion", "tools"),
                      context_length=4096, bytes=1, digest="mock")
            for name in self.models
        ]

    def show_model(self, name: str) -> ModelDetail:
        return ModelDetail(name=name, template="{{ .Prompt }}", capabilities=("completion", "tools"))

    def running(self) -> list[LoadedModel]:
        return [LoadedModel(name=self.models[0], size=1, size_vram=1, context_length=4096)]

    def script_for(self, req: GenerationRequest) -> MockScript:
        """取本次调用该用的脚本。

        值可以是单个脚本，也可以是**脚本序列**：多步工具循环的测试必须让同一个模型
        在连续几次调用里返回不同结果（第 1 步要工具、第 2 步给答案），
        否则只能靠"每步换一个模型名"来绕，那样测的就不是真实形态了。
        序列耗尽后停在最后一条——多要一次和真要一次得到同样的响应，便于断言循环已收敛。
        """
        entry = self.scripts.get(req.model, self.default)
        if isinstance(entry, MockScript):
            return entry
        items = list(entry)
        if not items:
            return self.default
        cursor = self._cursors.get(req.model, 0)
        self._cursors[req.model] = cursor + 1
        return items[min(cursor, len(items) - 1)]

    # ── 数据面 ────────────────────────────────────────────────────
    def generate(
        self,
        req: GenerationRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Generation:
        self.calls.append(req)
        script = self.script_for(req)
        if script.raise_exc is not None:
            raise script.raise_exc
        chunks = script_chunks(script)
        if req.stream:
            return consume_chunks(
                iter(chunks), trace_id=trace_id, clock=self.clock,
                style="native", model=req.model, on_event=on_event,
            )
        assembler = StreamAssembler(style="native", model=req.model)
        for chunk in chunks:
            assembler.feed(chunk)
        gen = assembler.build(model=req.model, status=Status.ERROR if script.error else Status.OK)
        emit_final_events(
            gen, assembler.last_raw, trace_id=trace_id, clock=self.clock, on_event=on_event, ttft_ms=None
        )
        return gen

    def embed(
        self,
        req: EmbedRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Embedding:
        """确定性假向量。**不报任何 token 数字**——mock 没跑真模型，编一个计数
        就是让下游以为"引擎报了 12 tok"，而那正是本项目最忌的假事实来源。
        """
        self.embed_calls.append(req)
        vectors = tuple(
            self.embeddings.get(text) or mock_vector(text) for text in req.inputs
        )
        dims = {len(v) for v in vectors}
        if len(dims) > 1:
            # 真实引擎一批里维度恒定；给了不一致的向量就是配置错了，
            # 而"不一致"往下游走会变成余弦计算抛错或静默截断——先在这里报死
            return Embedding(
                model=req.model, status=Status.ERROR,
                error=f"mock 的向量维度不一致：{sorted(dims)}（一批里必须同维）",
            )
        return Embedding(vectors=vectors, model=req.model, status=Status.OK)

    # ── 控制面 ────────────────────────────────────────────────────
    def pull(self, name: str, *, on_event: EventCB | None = None) -> AdminResult:
        return AdminResult(ok=True, action="pull", detail={"name": name})

    def unload(self, name: str) -> AdminResult:
        return AdminResult(ok=True, action="unload", detail={"name": name})

    def delete(self, name: str) -> AdminResult:
        return AdminResult(ok=True, action="delete", detail={"name": name})
