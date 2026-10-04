"""保真阶梯：每个档位是一个独立的 Counter 实现。

新增一档（例如从 GGUF vocab 自建 BPE 的 T2）只需实现 `Counter` 并加进
`default_counters()`，reconciler 与看板都不用改——这是"可扩展"的具体形状。
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
    #: 每请求固定的模板开销（带截距拟合得到，见 calibrate.fit_linear）
    fitted_intercept: float = 0.0
    #: 双特征标定的两个密度（见 calibrate.fit_by_script）；缺省则退回单比值
    fitted_cjk_ratio: float | None = None
    fitted_other_ratio: float | None = None
    tokenizer: Any | None = None
    chat_template: str = ""
    cjk_tokens_per_char: float = 1.0
    per_message_tokens: int = 4  # ChatML 类模板每条消息的角色标记开销（未标定时的粗略值）
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


def visible_input_split(req: GenerationRequest) -> tuple[int, int]:
    """输入的中文字符数与其他字符数——双特征标定要用（中英 token 密度差 3 倍以上）。"""
    from onyx.llm.measurement.heuristic import split_cjk

    text = "".join((m.content or "") + (m.thinking or "") for m in req.messages)
    if req.tools:
        text += json.dumps([t.as_openai_tool() for t in req.tools], ensure_ascii=False)
    return split_cjk(text)


class EngineCounter:
    """T0：引擎自报。P10/P12 实测确认它数的是整个 prompt，且含 thinking。"""

    name = TokenSource.ENGINE
    priority = 0
    default_confidence = Confidence.HIGH

    def count(self, req: GenerationRequest, gen: Generation, ctx: CounterContext) -> TokenSample | None:
        return gen.usage_from(TokenSource.ENGINE)


class CompatCounter:
    """T4：OpenAI 兼容层计数。

    P14 实测与原生差 −2/+16 ⇒ **有原生计数时只作交叉验证**（ENGINE 优先级更高，
    reconciler 永远先挑它）；但 `openai-compat` 这类通道只有这个数字，它是服务器对
    自己实际消耗的报告，比本地估计可信，所以排在 heuristic 之前、带 LOW 置信。
    """

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
        out_text = (gen.text or "") + (gen.thinking or "")
        if ctx.fitted_cjk_ratio is not None and ctx.fitted_other_ratio is not None:
            from onyx.llm.measurement.heuristic import split_cjk

            cjk, other = visible_input_split(req)
            out_cjk, out_other = split_cjk(out_text)
            in_tokens = round(
                ctx.fitted_intercept + ctx.fitted_cjk_ratio * cjk + ctx.fitted_other_ratio * other
            )
            out_tokens = round(ctx.fitted_cjk_ratio * out_cjk + ctx.fitted_other_ratio * out_other)
            note = (f"cjk={ctx.fitted_cjk_ratio} other={ctx.fitted_other_ratio} "
                    f"intercept={ctx.fitted_intercept} (n={ctx.fitted_n})")
        else:
            # 单比值退化路径：只在语料没有中文、双特征拟合不可用时才会走到这里
            in_tokens = round(visible_input_chars(req) * ctx.fitted_ratio + ctx.fitted_intercept)
            out_tokens = round(len(out_text) * ctx.fitted_ratio)
            note = f"ratio={ctx.fitted_ratio} intercept={ctx.fitted_intercept} (n={ctx.fitted_n})"
        return TokenSample(
            source=self.name, in_tokens=in_tokens, out_tokens=out_tokens, ok=True,
            confidence=self.default_confidence, note=note,
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


def text_counter(ctx: CounterContext) -> Callable[[str], int] | None:
    """返回"文本 → token 数"的函数，供分段归因使用。

    精度取决于该模型当前具备哪一档能力：
    - 有 tokenizer（T1/T2）⇒ 用它，归因可信度高
    - 有双特征标定（T3）  ⇒ 按中文/其他两种密度分别折算，medium
    - 只有单比值          ⇒ 按比值折算（退化路径）
    - 都没有              ⇒ 启发式，low

    **分段计数一律不含截距**：模板固定开销是"每请求一次"的量，
    摊到每个分段会重复计算。它应当作为 `template_ctl` 残差出现，
    这样残差才有物理意义（= 模板控制符成本），而不是变成误差的垃圾桶。
    """
    tokenizer = getattr(ctx, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        return lambda text: len(tokenizer.encode(text))
    if ctx.fitted_cjk_ratio is not None and ctx.fitted_other_ratio is not None:
        from onyx.llm.measurement.heuristic import split_cjk

        cjk_ratio, other_ratio = ctx.fitted_cjk_ratio, ctx.fitted_other_ratio

        def count_split(text: str) -> int:
            cjk, other = split_cjk(text)
            return round(cjk_ratio * cjk + other_ratio * other)

        return count_split
    if ctx.fitted_ratio and ctx.fitted_n >= 30:
        ratio = ctx.fitted_ratio
        return lambda text: round(len(text) * ratio)
    cjk = ctx.cjk_tokens_per_char
    return lambda text: estimate_tokens(text, cjk_tokens_per_char=cjk)
