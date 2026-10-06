"""L1 gateway —— 单一咽喉点（DESIGN 原则 1）。

所有模型调用必经此处。它做四件事，且只做这四件事：
1. 归一化请求并把原始证据（消息、工具定义、输出）落到内容寻址存储；
2. 调 provider 生成，把 provider 产出的事件转发给观测；
3. 跑计量阶梯 + 分段归因，产出各来源样本；
4. 产出 TRACE_START / USAGE_LOCAL / GPU_SAMPLE / TRACE_END，由观测引擎折叠成记录。

gateway **不写数据库、不做异常判定**——那是 obs 层的职责。这样"加一个指标"永远不需要改这里。
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from onyx.core.clock import SYSTEM_CLOCK, Clock
from onyx.core.content import BlobStore
from onyx.core.errors import OnyxError
from onyx.core.event import CONTRACT_VERSION, EventType, TraceEvent, make_event
from onyx.core.ids import new_trace_id
from onyx.core.types import (
    Cap,
    Embedding,
    EmbedRequest,
    Generation,
    GenerationRequest,
    ReconciledUsage,
    Status,
    TokenSample,
    TokenSource,
    TraceContext,
    TraceKind,
    TracePurpose,
)
from onyx.llm.measurement.fidelity import Counter, CounterContext, default_counters, text_counter
from onyx.llm.measurement.parts import attribute
from onyx.llm.params import dropped_by_openai
from onyx.llm.providers.base import EventCB, LlmProvider
from onyx.obs.anomalies import severity_of
from onyx.obs.engine import ObserverEngine
from onyx.obs.state import TraceState


class TraceEmitter:
    """绑定到单条 trace 的事件发射口。

    这是 gateway 之外唯一能把事件写进某条 trace 的合法途径——工具循环用它在
    **请求该工具的那一步**里记录 `TOOL_EXEC_START/END`。

    时机是硬约束：必须在该 trace 的 `TRACE_END` 之前用完。observer 在 TRACE_END
    就会 pop 状态并落库，之后再发事件只会得到一条 ORPHAN 或被静默丢弃。
    所以它只能由 `generate(before_trace_end=...)` 拿到，不能长期持有。
    """

    __slots__ = ("_gateway", "_trace_id")

    def __init__(self, gateway: Gateway, trace_id: str) -> None:
        self._gateway = gateway
        self._trace_id = trace_id

    @property
    def trace_id(self) -> str:
        return self._trace_id

    @property
    def blobs(self) -> BlobStore:
        """原始证据存储。工具结果/参数应落 blob，事件里只放 `sha256:` 引用——
        否则一个大响应会把事件流和看板一起撑爆（UI_DESIGN R7：必须能下钻到原文）。"""
        return self._gateway.blobs

    def emit(self, event_type: EventType | str, payload: dict[str, Any] | None = None) -> None:
        self._gateway._emit(
            make_event(event_type, self._trace_id, payload or {}, clock=self._gateway.clock)
        )

    def anomaly(self, code: str, detail: dict[str, Any] | None = None, *, severity: str = "") -> None:
        """注入一条异常码。severity 留空则由码表决定（不许各处自拟级别）。"""
        self.emit(EventType.ANOMALY, {
            "code": code, "severity": severity or severity_of(code), "detail": detail or {},
        })


@dataclass(frozen=True, slots=True)
class GatewayResult:
    trace_id: str

    generation: Generation
    usage: ReconciledUsage | None
    anomalies: tuple[tuple[str, str, dict[str, Any]], ...]
    latency: dict[str, Any]
    record_refs: dict[str, str]

    @property
    def ok(self) -> bool:
        return self.generation.status.value == "ok"


@dataclass(frozen=True, slots=True)
class EmbedGatewayResult:
    """向量化的结果。**向量本身不落库**（一次 1024 维 float 数组比它服务的文本长得多，
    而没有任何消费者读它——存了只会让 `.data` 膨胀）。要留的是出处：输入、计数、耗时。
    """

    trace_id: str
    embedding: Embedding
    usage: ReconciledUsage | None
    anomalies: tuple[tuple[str, str, dict[str, Any]], ...]
    record_refs: dict[str, str]

    @property
    def ok(self) -> bool:
        return self.embedding.status is Status.OK


class Gateway:
    def __init__(
        self,
        provider: LlmProvider,
        *,
        observer: ObserverEngine,
        blobs: BlobStore,
        counters: tuple[Counter, ...] | None = None,
        counter_ctx: CounterContext | None = None,
        counter_ctx_factory: Callable[[str], CounterContext] | None = None,
        clock: Clock = SYSTEM_CLOCK,
        event_sink: EventCB | None = None,
        sample_gpu: bool = False,
        model_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.observer = observer
        self.blobs = blobs
        self.counters: tuple[Counter, ...] = counters if counters is not None else default_counters()
        self.counter_ctx = counter_ctx or CounterContext()
        #: 按模型取标定参数（每个模型的 tokens/char 不同）。缺省则用全局 counter_ctx。
        self.counter_ctx_factory = counter_ctx_factory
        self.clock = clock
        self.event_sink = event_sink
        self.sample_gpu = sample_gpu
        self.model_id = model_id

    def _ctx_for(self, model: str) -> CounterContext:
        if self.counter_ctx_factory is None:
            return self.counter_ctx
        try:
            return self.counter_ctx_factory(model) or self.counter_ctx
        except Exception:  # noqa: BLE001 - 取标定失败退回全局配置，不许让请求失败
            return self.counter_ctx

    # ── 主入口 ────────────────────────────────────────────────────
    def generate(
        self,
        req: GenerationRequest,
        *,
        purpose: TracePurpose | str | None = None,
        trace_id: str | None = None,
        context: TraceContext | None = None,
        before_trace_end: Callable[[Generation, TraceEmitter], None] | None = None,
    ) -> GatewayResult:
        """跑一次生成。

        `before_trace_end` 是给工具循环用的钩子：在 TRACE_END 之前拿到本条 trace 的
        发射口，好把工具执行记在**发起该调用的那一步**里。钩子抛异常时仍然会关 trace
        （否则观测库里会留下一条永远不结束的 trace），异常照常向上传播。
        """
        tid = trace_id or new_trace_id()
        ctx = context or req.context
        if purpose is not None:
            ctx = dataclasses.replace(ctx, purpose=TracePurpose(purpose))
        start_ns = self.clock.monotonic_ns()

        messages_ref = self.blobs.put_json([_message_dict(m) for m in req.messages])
        tools_ref = (
            self.blobs.put_json([t.as_openai_tool() for t in req.tools]) if req.tools else None
        )
        params = self._params_snapshot(req)

        self._emit(make_event(EventType.TRACE_START, tid, {
            "kind": str(ctx.kind or TraceKind.GENERATION),
            "purpose": ctx.purpose_label,
            "provider_id": getattr(self.provider, "id", ""),
            "model": req.model,
            "model_id": self.model_id,
            "params": params,
            "messages_ref": messages_ref,
            "tools_ref": tools_ref,
            "keep_alive": req.keep_alive,
            "context": _context_dict(ctx),
            "contract_version": CONTRACT_VERSION,
        }, clock=self.clock))

        try:
            gen = self.provider.generate(req, trace_id=tid, on_event=self._forward)
        except OnyxError as exc:
            self._fail(tid, start_ns, exc)
            raise
        except Exception as exc:
            self._fail(tid, start_ns, exc)
            raise

        wall_ms = (self.clock.monotonic_ns() - start_ns) / 1e6
        output_ref = self.blobs.put_json({
            "text": gen.text, "thinking": gen.thinking, "finish_reason": str(gen.finish_reason),
            "tool_calls": [_tool_call_dict(c) for c in gen.tool_calls],
            "status": str(gen.status), "error": gen.error,
        })
        self._measure(req, gen, tid)
        if self.sample_gpu:
            self._emit_gpu_sample(tid, req.model)

        # 钩子里跑的是真实工具执行，它崩了也必须先关 trace：留下一条永不结束的
        # trace 会污染所有按 trace 聚合的统计。所以先记下异常，发完 TRACE_END 再抛。
        hook_error: Exception | None = None
        if before_trace_end is not None:
            try:
                before_trace_end(gen, TraceEmitter(self, tid))
            except Exception as exc:  # noqa: BLE001 - 见上，延迟到 TRACE_END 之后再抛
                hook_error = exc

        state = self._emit(make_event(EventType.TRACE_END, tid, {
            "status": str(gen.status), "wall_ms": wall_ms, "error": gen.error,
            "finish_reason": str(gen.finish_reason), "output_ref": output_ref,
            "text_chars": len(gen.text), "thinking_chars": len(gen.thinking),
        }, clock=self.clock))

        if hook_error is not None:
            raise hook_error

        return GatewayResult(
            trace_id=tid,
            generation=dataclasses.replace(gen, wall_ms=wall_ms),
            usage=state.reconciled if state else None,
            anomalies=tuple(state.anomalies) if state else (),
            latency=state.latency_summary() if state else {},
            record_refs={"messages": messages_ref, "tools": tools_ref or "", "output": output_ref},
        )

    # ── 向量化入口 ────────────────────────────────────────────────
    def embed(
        self,
        req: EmbedRequest,
        *,
        purpose: TracePurpose | str | None = None,
        trace_id: str | None = None,
    ) -> EmbedGatewayResult:
        """一次批量向量化。**没有 `EmbeddingProvider` 就是明确的 unsupported**。

        这里刻意不做任何隐式降级（"那就用 chat 生成文本再解析成向量"之类）：
        那条路会产出一个看着像向量的东西，而它测的是模型会不会输出 JSON，不是表示质量。
        能力缺失要在这里留一条错误 trace，而不是在下游变成一个分数。
        """
        from onyx.llm.providers.base import EmbeddingProvider

        tid = trace_id or new_trace_id()
        ctx = req.context
        if purpose is not None:
            ctx = dataclasses.replace(ctx, purpose=TracePurpose(purpose))
        start_ns = self.clock.monotonic_ns()
        inputs_ref = self.blobs.put_json({"model": req.model, "inputs": list(req.inputs)})

        self._emit(make_event(EventType.TRACE_START, tid, {
            # kind 由**入口**决定，不读 `ctx.kind`：TraceContext 的默认值就是 generation，
            # 于是"默认值"与"有人显式要求记成 generation"根本分不开。
            # 一条向量化调用被记成 generation，会让所有按 kind 分的统计说谎（S33 实测踩到）
            "kind": str(TraceKind.EMBED),
            "purpose": ctx.purpose_label,
            "provider_id": getattr(self.provider, "id", ""),
            "model": req.model,
            "model_id": self.model_id,
            "params": {"batch_size": req.batch_size},
            "messages_ref": inputs_ref,
            "tools_ref": None,
            "keep_alive": req.keep_alive,
            "context": _context_dict(ctx),
            "contract_version": CONTRACT_VERSION,
        }, clock=self.clock))

        if not isinstance(self.provider, EmbeddingProvider):
            provider_id = getattr(self.provider, "id", type(self.provider).__name__)
            detail = {"cap": str(Cap.EMBED), "provider_id": provider_id,
                      "model": req.model, "batch_size": req.batch_size}
            self._emit(make_event(EventType.ANOMALY, tid, {
                "code": "EMBED_UNSUPPORTED", "severity": severity_of("EMBED_UNSUPPORTED"),
                "detail": detail,
            }, clock=self.clock))
            failed = Embedding(
                model=req.model, status=Status.ERROR,
                error=f"provider {provider_id} 不提供向量化（Cap.EMBED 未实现）",
                extra=detail,
            )
            state = self._emit(make_event(EventType.TRACE_END, tid, {
                "status": str(Status.ERROR),
                "wall_ms": (self.clock.monotonic_ns() - start_ns) / 1e6,
                "error": failed.error,
            }, clock=self.clock))
            return EmbedGatewayResult(
                trace_id=tid, embedding=failed, usage=state.reconciled if state else None,
                anomalies=tuple(state.anomalies) if state else (),
                record_refs={"messages": inputs_ref},
            )

        try:
            result = self.provider.embed(req, trace_id=tid, on_event=self._forward)
        except OnyxError as exc:
            self._fail(tid, start_ns, exc)
            raise
        except Exception as exc:
            self._fail(tid, start_ns, exc)
            raise

        wall_ms = (self.clock.monotonic_ns() - start_ns) / 1e6
        embedding = dataclasses.replace(result, wall_ms=wall_ms)
        if self.sample_gpu:
            self._emit_gpu_sample(tid, req.model)
        state = self._emit(make_event(EventType.TRACE_END, tid, {
            "status": str(embedding.status), "wall_ms": wall_ms, "error": embedding.error,
            "vectors": len(embedding.vectors), "dimension": embedding.dimension,
            "batch_size": req.batch_size,
        }, clock=self.clock))
        return EmbedGatewayResult(
            trace_id=tid, embedding=embedding, usage=state.reconciled if state else None,
            anomalies=tuple(state.anomalies) if state else (),
            record_refs={"messages": inputs_ref},
        )

    # ── 内部 ──────────────────────────────────────────────────────
    def _measure(self, req: GenerationRequest, gen: Generation, tid: str) -> list[TokenSample]:
        ctx = self._ctx_for(req.model)
        samples: list[TokenSample] = []
        for counter in self.counters:
            try:
                sample = counter.count(req, gen, ctx)
            except Exception as exc:  # noqa: BLE001 - 计量档位失败只降级，不许打断请求
                sample = TokenSample(
                    source=counter.name, ok=False, note=f"{type(exc).__name__}: {exc}"[:200]
                )
            if sample is None:
                continue
            samples.append(sample)
            if sample.source is TokenSource.ENGINE:
                continue  # 引擎计数已由 provider 以 USAGE_ENGINE 事件发出，不重复
            self._emit_usage_local(tid, sample)

        engine = gen.usage_from(TokenSource.ENGINE)
        count_fn = text_counter(ctx)
        if count_fn is not None:
            parts, report = attribute(
                req, count_fn=count_fn, engine_in=engine.in_tokens if engine else None,
                has_template=bool(ctx.chat_template), gen_text=gen.text + gen.thinking,
            )
            if parts:
                self._emit(make_event(EventType.USAGE_ATTRIBUTION, tid, {
                    "count_source": _attribution_source(ctx),
                    "parts": [
                        {"part": p.part, "ord": p.ord, "tokens": p.tokens, "bytes": p.bytes}
                        for p in parts
                    ],
                    "attribution": {
                        "input_segments_tokens": report.input_segments_tokens,
                        "template_ctl_tokens": report.template_ctl_tokens,
                        "residual_raw": report.residual_raw,
                        "clamped": report.clamped,
                        "has_template": report.has_template,
                    },
                }, clock=self.clock))
        return samples

    def _emit_usage_local(self, tid: str, sample: TokenSample) -> None:
        self._emit(make_event(EventType.USAGE_LOCAL, tid, {
            "source": str(sample.source), "in_tokens": sample.in_tokens,
            "out_tokens": sample.out_tokens, "thinking_tokens": sample.thinking_tokens,
            "cached_tokens": sample.cached_tokens, "ok": sample.ok,
            "confidence": str(sample.confidence), "note": sample.note,
        }, clock=self.clock))

    def _emit_gpu_sample(self, tid: str, model: str) -> None:
        running = getattr(self.provider, "running", None)
        if running is None:
            return
        try:
            loaded = next((m for m in running() if m.name == model or m.model == model), None)
        except Exception:  # noqa: BLE001 - 采样失败不能影响请求
            return
        if loaded is None:
            return
        self._emit(make_event(EventType.GPU_SAMPLE, tid, {
            "model": model, "size": loaded.size, "size_vram": loaded.size_vram,
            "context_length": loaded.context_length, "expires_at": loaded.expires_at,
        }, clock=self.clock))

    def _params_snapshot(self, req: GenerationRequest) -> dict[str, Any]:
        snapshot: dict[str, Any] = dict(req.params.as_dict())
        snapshot["stream"] = req.stream
        snapshot["thinking"] = req.thinking
        if req.tools:
            snapshot["tool_names"] = list(req.tool_names)
        if req.tool_choice:
            snapshot["tool_choice"] = req.tool_choice
        dropped = dropped_by_openai(req.params)
        if dropped:
            # 记下来：将来若改走 /v1 通道，这些参数会静默失效（P14）
            snapshot["_would_drop_on_openai_channel"] = dropped
        return snapshot

    def _fail(self, tid: str, start_ns: int, exc: Exception) -> None:
        status = "timeout" if type(exc).__name__ == "RequestTimeout" else "error"
        self._emit(make_event(EventType.TRACE_END, tid, {
            "status": status,
            "wall_ms": (self.clock.monotonic_ns() - start_ns) / 1e6,
            "error": f"{type(exc).__name__}: {exc}"[:1000],
            "finish_reason": "error",
        }, clock=self.clock))

    def _emit(self, event: TraceEvent) -> TraceState | None:
        state = self.observer.handle(event)
        if self.event_sink is not None:
            # 事件订阅方失败不许影响请求主链路
            with contextlib.suppress(Exception):
                self.event_sink(event)
        return state

    def _forward(self, event: TraceEvent) -> None:
        """provider 产出的事件同样进观测与订阅方。"""
        self._emit(event)


def _attribution_source(ctx: CounterContext) -> str:
    """归因用的是哪一档计数函数——UI 上必须与总量计数的出处分开标注。"""
    if getattr(ctx, "tokenizer", None) is not None:
        return str(TokenSource.GGUF_VOCAB)
    if ctx.fitted_ratio and ctx.fitted_n >= 30:
        return str(TokenSource.FITTED)
    return str(TokenSource.HEURISTIC)


def _message_dict(message: Any) -> dict[str, Any]:
    return {
        "role": str(message.role), "content": message.content, "name": message.name,
        "tool_call_id": message.tool_call_id, "thinking": message.thinking,
        "media_refs": list(message.media_refs),
        "tool_calls": [_tool_call_dict(c) for c in message.tool_calls],
    }


def _tool_call_dict(call: Any) -> dict[str, Any]:
    return {
        "index": call.index, "id": call.id, "name": call.name, "arguments": call.arguments,
        "arguments_raw": call.arguments_raw, "parse_status": str(call.parse_status),
        "parse_source": call.parse_source,
    }


def _context_dict(ctx: TraceContext) -> dict[str, Any]:
    return {
        "eval_run_id": ctx.eval_run_id, "case_id": ctx.case_id, "sample_seq": ctx.sample_seq,
        "parent_trace_id": ctx.parent_trace_id, "root_trace_id": ctx.root_trace_id,
        "extra": ctx.extra,
    }
