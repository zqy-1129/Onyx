"""具体探针实现。

每个探针回答一个**必须先知道答案才能正确编码**的问题（对应 docs/PROBES.md 的 U 编号）。
结论由证据推导，推导不出来就 `unknown=True` —— 宁可显示「—」也不猜。
"""

from __future__ import annotations

import json
import re
from typing import Any

from onyx.core.types import (
    GenerationRequest,
    GenParams,
    Message,
    ParseStatus,
    ProbeFinding,
    Role,
    TokenSource,
    ToolSpec,
)
from onyx.probe.runner import ProbeContext, probe

#: 用于缓存实验的长 prompt（确定性构造，保证每次字节一致）
_PARA = (
    "在本地部署大模型时，输入 token 的计数口径决定了成本与上下文占用是否可信。"
    "Local inference requires an exact accounting of prompt tokens, because the KV cache "
    "reuses previously evaluated prefixes and the engine may report only the uncached suffix. "
)
LONG_PROMPT = "请只回复「收到」两个字。\n\n" + (_PARA * 12)

_XML_TOOL_RE = re.compile(r"<\s*(tool_call|function_call|toolcall)\b", re.IGNORECASE)
_JSON_TOOL_RE = re.compile(r'\{\s*"(name|function|tool)"\s*:', re.IGNORECASE)


def _finding(
    ctx: ProbeContext, name: str, verdict: str, evidence: dict[str, Any], *, unknown: bool = False
) -> ProbeFinding:
    return ProbeFinding(
        probe=name,
        subject=ctx.model,
        verdict=verdict,
        unknown=unknown,
        evidence=evidence,
        provider_version=ctx.provider_version,
    )


def _engine_in(gen: Any) -> int | None:
    sample = gen.usage_from(TokenSource.ENGINE)
    return sample.in_tokens if sample else None


def _engine_out(gen: Any) -> int | None:
    sample = gen.usage_from(TokenSource.ENGINE)
    return sample.out_tokens if sample else None


# ── U1 · 缓存命中时 prompt_eval_count 数的是什么 ────────────────────
@probe("cache")
def check_cache_semantics(ctx: ProbeContext) -> ProbeFinding:
    """同一长 prompt 连发 3 次（keep_alive 保持载入），看计数与 prompt_eval 时长如何变化。

    - 计数三次相同 + 时长显著下降 ⇒ 数的是**整个 prompt**，缓存只影响耗时
    - 计数逐次下降           ⇒ 数的是**未缓存后缀**，输入 token 汇总会系统性偏低
    """
    runs = [
        ctx.ask(LONG_PROMPT, max_tokens=8, tag=f"cache-{i}", thinking=False)
        for i in range(3)
    ]
    counts = [_engine_in(r) for r in runs]
    durations = [r.latency.ms("prompt_eval") if r.latency else None for r in runs]
    known = [c for c in counts if c is not None]
    known_d = [d for d in durations if d]

    if len(known) < 3:
        return _finding(ctx, "cache", "insufficient_data", {"counts": counts}, unknown=True)

    same_counts = len(set(known)) == 1
    speedup = (known_d[0] / known_d[-1]) if len(known_d) >= 2 and known_d[-1] else None
    if same_counts:
        verdict = "counts_full_prompt_cache_affects_duration_only"
    elif known[-1] < known[0]:
        verdict = "counts_uncached_suffix_only_INPUT_TOKENS_UNDERCOUNTED"
    else:
        verdict = "unstable"
    return _finding(
        ctx, "cache", verdict,
        {"counts": counts, "prompt_eval_ms": [round(d, 1) if d else None for d in durations],
         "speedup_x": round(speedup, 2) if speedup else None},
        unknown=verdict == "unstable",
    )


# ── U2 · thinking token 是否计入 eval_count ────────────────────────
@probe("think")
def check_thinking_accounting(ctx: ProbeContext) -> ProbeFinding:
    """同一问题分别用 think=true / think=false 跑，比较 eval_count 与文本量。

    若 think=true 时 out 明显更大而正文差不多（甚至为空），说明推理 token **计入了** eval_count
    —— 那么"输出 token"这个指标其实是"输出+推理"，成本口径必须写明。
    """
    prompt = "9.11 和 9.9 哪个大？只回答数字。"
    on = ctx.ask(prompt, max_tokens=256, thinking=True, tag="think-on")
    off = ctx.ask(prompt, max_tokens=256, thinking=False, tag="think-off")
    on_out, off_out = _engine_out(on), _engine_out(off)
    if on_out is None or off_out is None:
        return _finding(ctx, "think", "no_engine_counts", {"on": on_out, "off": off_out}, unknown=True)

    ratio = on_out / off_out if off_out else None
    if ratio and ratio >= 1.3:
        verdict = "thinking_included_in_eval_count"
    elif ratio and ratio <= 1.1 and on.thinking:
        verdict = "thinking_excluded_from_eval_count"
    elif not on.thinking:
        verdict = "thinking_not_produced_by_this_model"
    else:
        verdict = "inconclusive"
    return _finding(
        ctx, "think", verdict,
        {"out_think_on": on_out, "out_think_off": off_out, "ratio": round(ratio, 2) if ratio else None,
         "thinking_chars_on": len(on.thinking), "text_chars_on": len(on.text),
         "text_chars_off": len(off.text), "thinking_field_present": bool(on.thinking)},
        unknown=verdict == "inconclusive",
    )


# ── U4 · 流式与非流式的计数是否一致 ────────────────────────────────
@probe("stream_usage")
def check_stream_usage(ctx: ProbeContext) -> ProbeFinding:
    """同一 prompt 走流式与非流式，比较 in/out 计数。

    同时验证：流式的计数是否只在最后一个 ndjson 事件里（assembler 的 saw_done 标记）。
    """
    prompt = "用一句话说明什么是 KV 缓存。"
    plain = ctx.ask(prompt, max_tokens=64, thinking=False, tag="stream-off")
    streamed = ctx.ask(prompt, max_tokens=64, thinking=False, stream=True, tag="stream-on")

    p = (_engine_in(plain), _engine_out(plain))
    st = (_engine_in(streamed), _engine_out(streamed))
    saw_done = bool(streamed.extra.get("saw_done"))
    if None in p or None in st:
        return _finding(ctx, "stream_usage", "missing_counts", {"plain": p, "stream": st}, unknown=True)
    verdict = "identical" if p == st else "divergent"
    if not saw_done:
        verdict = "stream_final_event_missing_counts"
    return _finding(
        ctx, "stream_usage", verdict,
        {"plain_in_out": p, "stream_in_out": st, "saw_done_event": saw_done,
         "ttft_ms": round(streamed.ttft_ms, 1) if streamed.ttft_ms else None,
         "non_stream_ttft": plain.ttft_ms},
        unknown=verdict == "divergent",
    )


# ── U5 · /v1 兼容层的 usage 与原生是否一致 ─────────────────────────
@probe("compat_parity")
def check_compat_usage_parity(ctx: ProbeContext) -> ProbeFinding:
    """同一请求分别走原生 `/api/chat` 与 `/v1/chat/completions`，比较 token 计数。

    这决定 T4（compat 档）到底有没有交叉验证价值：一致 ⇒ 可作校验；
    不一致 ⇒ 必须永远只记录不采信，并把偏差本身当成异常信号。
    """
    prompt = "解释一下什么是量化（quantization），一句话。"
    native = ctx.ask(prompt, max_tokens=64, thinking=False, tag="compat-native")
    payload = {
        "model": ctx.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        raw = ctx.post("/v1/chat/completions", payload)
    except Exception as exc:  # noqa: BLE001 - 兼容层可能整体不可用，这也是结论
        return _finding(
            ctx, "compat_parity", f"unavailable:{type(exc).__name__}",
            {"error": str(exc)[:200]}, unknown=True,
        )
    usage = (raw or {}).get("usage") or {}
    compat_in, compat_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    native_in, native_out = _engine_in(native), _engine_out(native)

    if compat_in is None:
        return _finding(
            ctx, "compat_parity", "compat_usage_absent",
            {"native_in": native_in, "compat_usage": usage}, unknown=True,
        )
    delta_in = compat_in - (native_in or 0)
    delta_out = (compat_out or 0) - (native_out or 0)
    verdict = "parity" if delta_in == 0 else f"divergent_input_delta_{delta_in:+d}"
    return _finding(
        ctx, "compat_parity", verdict,
        {"native_in": native_in, "compat_in": compat_in, "delta_in": delta_in,
         "native_out": native_out, "compat_out": compat_out, "delta_out": delta_out},
        unknown=False,
    )


# ── U6 · format:json_schema 是否真的强制生效 ───────────────────────
@probe("structured")
def check_structured_output(ctx: ProbeContext) -> ProbeFinding:
    """给一个带 required 与 enum 的 schema，看输出是否合法。

    若引擎不支持却**静默降级**，结构化抽取评测会把"格式运气"当成模型能力。
    """
    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "temperature": {"type": "number"},
        },
        "required": ["city", "unit", "temperature"],
    }
    gen = ctx.ask(
        "北京现在 21 摄氏度，请按要求输出。",
        max_tokens=128, thinking=False, json_schema=schema, tag="structured",
    )
    text = gen.text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return _finding(
            ctx, "structured", "not_enforced_invalid_json",
            {"text_head": text[:160], "error": str(exc)[:120], "finish": str(gen.finish_reason)},
        )
    missing = [k for k in schema["required"] if k not in parsed]
    bad_enum = parsed.get("unit") not in schema["properties"]["unit"]["enum"]
    if missing or bad_enum:
        return _finding(
            ctx, "structured", "partially_enforced",
            {"missing": missing, "bad_enum": bad_enum, "parsed": parsed},
        )
    return _finding(ctx, "structured", "enforced", {"parsed": parsed})


# ── 工具调用走哪种序列化 ───────────────────────────────────────────
@probe("tool_format")
def check_tool_format(ctx: ProbeContext) -> ProbeFinding:
    """判定工具调用是走原生 tool_calls 头，还是被塞进正文（XML / JSON / 自定义模板）。

    这决定 `tools/loop.py` 该用哪种解析器，也决定畸形调用长什么样。
    """
    tool = ToolSpec(
        name="get_weather",
        description="查询指定城市当前天气",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    )
    req = GenerationRequest(
        model=ctx.model,
        messages=(Message(role=Role.USER, content="北京现在天气怎么样？"),),
        tools=(tool,),
        params=GenParams(temperature=0.0, max_tokens=256),
        thinking=False,
        keep_alive="5m",
    )
    gen = ctx.provider.generate(req, trace_id="probe:tool_format")
    ctx.record("tool_format", gen)

    if gen.tool_calls:
        call = gen.tool_calls[0]
        verdict = "native_head" if call.parse_status is ParseStatus.OK else f"native_head_{call.parse_status}"
        return _finding(
            ctx, "tool_format", verdict,
            {"name": call.name, "args": call.arguments, "args_raw": call.arguments_raw[:120],
             "finish": str(gen.finish_reason), "wants_tool_call": gen.wants_tool_call,
             "done_reason_is_stop": str(gen.finish_reason) == "stop"},
        )
    text = gen.text or ""
    if _XML_TOOL_RE.search(text):
        verdict = "xml_in_content"
    elif _JSON_TOOL_RE.search(text):
        verdict = "json_in_content"
    elif not text.strip():
        verdict = "no_output"
    else:
        verdict = "plain_text_no_tool_call"
    return _finding(
        ctx, "tool_format", verdict,
        {"text_head": text[:200], "finish": str(gen.finish_reason)},
        unknown=verdict in {"no_output", "plain_text_no_tool_call"},
    )
