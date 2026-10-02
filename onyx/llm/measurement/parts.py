"""分段归因：把"总共 1842 token"拆成 system / 工具定义 / 每条消息 / 模板控制符。

P9 的实测结论决定了这里的做法：**多数模型拿不到可用的 chat template**，
所以不追求"渲染出引擎真正喂进去的字符串"，而是
    Σ(各分段真实计数) + template_ctl 残差 = 引擎报的 prompt_eval_count
残差就是模板控制符（角色标记、生成提示符、工具 schema 包装）的成本——
它是**测出来的**，不是估出来的。有真模板的模型（如 gpt-oss:20b）残差应趋近 0，
这本身就是一个校验信号。

工具定义的开销（DESIGN §6.2 的核心指标）就在 `part='tool_defs'` 这一行。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from onyx.core.types import GenerationRequest, Message, Role, TokenPart

CountFn = Callable[[str], int]


@dataclass(frozen=True, slots=True)
class Segment:
    part: str
    ord: int
    text: str


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """归因的元信息：让"这个数字怎么来的"始终可查。"""

    input_segments_tokens: int
    template_ctl_tokens: int
    residual_raw: int | None
    clamped: bool
    engine_in: int | None
    has_template: bool

    @property
    def complete(self) -> bool:
        return self.engine_in is not None and not self.clamped


def input_segments(req: GenerationRequest) -> list[Segment]:
    """输入侧分段。工具定义单独成段——它是每次请求都要付的固定成本。"""
    segments: list[Segment] = []
    order = 0
    system_text = "\n".join(m.content for m in req.messages if m.role is Role.SYSTEM and m.content)
    if system_text:
        segments.append(Segment(part="system", ord=order, text=system_text))
        order += 1
    if req.tools:
        segments.append(Segment(
            part="tool_defs", ord=order,
            text=json.dumps([t.as_openai_tool() for t in req.tools], ensure_ascii=False),
        ))
        order += 1
    for index, message in enumerate(req.messages):
        if message.role is Role.SYSTEM:
            continue
        text = _message_text(message)
        if not text:
            continue
        segments.append(Segment(part=f"msg:{index}", ord=order, text=text))
        order += 1
    return segments


def _message_text(message: Message) -> str:
    parts = [message.content or ""]
    if message.thinking:
        parts.append(message.thinking)
    for call in message.tool_calls:
        parts.append(json.dumps(
            {"name": call.name, "arguments": call.arguments or {}}, ensure_ascii=False
        ))
    if message.media_refs:
        # 图片 token 由引擎（mmproj）决定，文本口径估不准（DESIGN R6）
        parts.append(f"<image x{len(message.media_refs)}>")
    return "\n".join(p for p in parts if p)


def attribute(
    req: GenerationRequest,
    *,
    count_fn: CountFn,
    engine_in: int | None = None,
    has_template: bool = False,
    gen_text: str = "",
) -> tuple[tuple[TokenPart, ...], AttributionReport]:
    """产出分段计数。`count_fn` 决定精度档位（GGUF BPE / HF / fitted / heuristic）。"""
    parts: list[TokenPart] = []
    total = 0
    for segment in input_segments(req):
        tokens = count_fn(segment.text)
        total += tokens
        parts.append(TokenPart(
            part=segment.part, ord=segment.ord, tokens=tokens,
            bytes=len(segment.text.encode("utf-8")),
        ))
    if gen_text:
        parts.append(TokenPart(
            part="output", ord=len(parts), tokens=count_fn(gen_text),
            bytes=len(gen_text.encode("utf-8")),
        ))

    residual_raw = None
    template_ctl = 0
    clamped = False
    if engine_in is not None:
        residual_raw = engine_in - total
        if residual_raw < 0:
            # 分段和超过引擎计数 ⇒ count_fn 高估，或引擎做了截断。原样记录，不掩盖。
            clamped = True
            template_ctl = 0
        else:
            template_ctl = residual_raw
        parts.append(TokenPart(part="template_ctl", ord=len(parts), tokens=template_ctl, bytes=None))

    return tuple(parts), AttributionReport(
        input_segments_tokens=total,
        template_ctl_tokens=template_ctl,
        residual_raw=residual_raw,
        clamped=clamped,
        engine_in=engine_in,
        has_template=has_template,
    )


def part_totals(parts: Any) -> dict[str, int]:
    """给看板用：part → tokens（同名 part 求和）。"""
    out: dict[str, int] = {}
    for part in parts:
        out[part.part] = out.get(part.part, 0) + part.tokens
    return out
