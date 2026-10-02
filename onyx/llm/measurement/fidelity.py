"""保真阶梯：每个档位是一个独立的 Counter 实现。

新增一档（例如从 GGUF vocab 自建 BPE 的 T2）只需实现 `Counter` 并加进
`default_counters()`，reconciler 与看板都不用改——这是"可扩展"的具体形状。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

from onyx.core.types import (
    Confidence,
    Generation,
    GenerationRequest,
    TokenSample,
    TokenSource,
)
from onyx.llm.measurement.heuristic import estimate_tokens


@dataclass(frozen=True, slots=True)
class CounterContext:
    """档位所需的模型侧信息。全部可选——缺什么就退到哪一档。"""

    fitted_ratio: float | None = None
    fitted_n: int = 0
    tokenizer: Any | None = None
    chat_template: str = ""
    cjk_tokens_per_char: float = 1.0
    per_message_tokens: int = 4  # ChatML 类模板每条消息的角色标记开销
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Counter(Protocol):
    name: ClassVar[TokenSource]
    #: 数字越小越可信；与 core.types.SOURCE_PRIORITY 一致
    priority: ClassVar[int]
    default_confidence: ClassVar[Confidence]

    def count(
        self, req: GenerationRequest, gen: Generation, ctx: CounterContext
    ) -> TokenSample | None: ...


def visible_input_chars(req: GenerationRequest) -> int:
    """请求里"看得见"的字符数：消息正文 + 工具定义的 JSON 序列化。

    刻意不含模板控制符——那部分由 `parts.attribute` 以 `template_ctl` 残差显式暴露，
    而不是悄悄塞进估计值里（否则估计值就无法解释了）。
    """
    total = sum(len(m.content or "") + len(m.thinking or "") for m in req.messages)
    if req.tools:
        total += len(json.dumps([t.as_openai_tool() for t in req.tools], ensure_ascii=False))
    return total


def visible_output_chars(gen: Generation) -> int:
    return len(gen.text or "") + len(gen.thinking or "")


class EngineCounter:
    """T0：引擎自报。P10/P12 实测确认它数的是整个 prompt，且含 thinking。"""

    name = TokenSource.ENGINE
    priority = 0
    default_confidence = Confidence.HIGH

    def count(self, req: GenerationRequest, gen: Generation, ctx: CounterContext) -> TokenSample | None:
        return gen.usage_from(TokenSource.ENGINE)


class CompatCounter:
    """T4：OpenAI 兼容层。P14 实测与原生差 −2/+16 ⇒ **只记录、永不采信**。"""

    name = TokenSource.COMPAT
    priority = 90
    default_confidence = Confidence.LOW

    def count(self, req: GenerationRequest, gen: Generation, ctx: CounterContext) -> TokenSample | None:
        return gen.usage_from(TokenSource.COMPAT)


class FittedCounter:
    """T3：用该模型标定出的 tokens/char 比。需要先跑过 `onyx calibrate`。"""

    name = TokenSource.FITTED
    priority = 30
    default_confidence = Confidence.MEDIUM

    def count(self, req: GenerationRequest, gen: Generation, ctx: CounterContext) -> TokenSample | None:
        if not ctx.fitted_ratio or ctx.fitted_n < 30:
            return TokenSample(
                source=self.name, ok=False, confidence=Confidence.LOW,
                note=f"未标定或样本不足(n={ctx.fitted_n}，需≥30)",
            )
        in_chars = visible_input_chars(req)
        overhead = ctx.per_message_tokens * len(req.messages)
        return TokenSample(
            source=self.name,
            in_tokens=round(in_chars * ctx.fitted_ratio) + overhead,
            out_tokens=round(visible_output_chars(gen) * ctx.fitted_ratio),
            ok=True,
            confidence=self.default_confidence,
            note=f"ratio={ctx.fitted_ratio} (n={ctx.fitted_n})",
        )


class HeuristicCounter:
    """T5：字符类别加权估计。永远可用，永远标 low。"""

    name = TokenSource.HEURISTIC
    priority = 40
    default_confidence = Confidence.LOW

    def count(self, req: GenerationRequest, gen: Generation, ctx: CounterContext) -> TokenSample | None:
        kw = {"cjk_tokens_per_char": ctx.cjk_tokens_per_char}
        in_tokens = sum(
            estimate_tokens(m.content or "", **kw) + estimate_tokens(m.thinking or "", **kw)
            for m in req.messages
        )
        in_tokens += ctx.per_message_tokens * len(req.messages)
        if req.tools:
            in_tokens += estimate_tokens(
                json.dumps([t.as_openai_tool() for t in req.tools], ensure_ascii=False), **kw
            )
        out_tokens = estimate_tokens(gen.text or "", **kw) + estimate_tokens(gen.thinking or "", **kw)
        return TokenSample(
            source=self.name, in_tokens=in_tokens, out_tokens=out_tokens, ok=True,
            confidence=self.default_confidence, note="字符类别加权估计，未经标定",
        )


def default_counters() -> tuple[Counter, ...]:
    """默认阶梯。T1(hf_tokenizer)/T2(gguf_vocab) 在具备条件时由调用方插入。"""
    return (EngineCounter(), FittedCounter(), HeuristicCounter(), CompatCounter())
