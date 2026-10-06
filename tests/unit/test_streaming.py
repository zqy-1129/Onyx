"""流式缝合的纯逻辑测试：不发网络请求，穷举分片与畸形输入。"""

from __future__ import annotations

import json

import pytest

from onyx.core.types import FinishReason, ParseStatus, TokenSource
from onyx.llm.streaming import StreamAssembler


def _native_chunk(content: str = "", thinking: str = "", tool_calls: list | None = None) -> dict:
    message: dict = {"role": "assistant"}
    if content:
        message["content"] = content
    if thinking:
        message["thinking"] = thinking
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"model": "m", "created_at": "x", "message": message, "done": False}


def _native_final(**kw) -> dict:
    base = {
        "model": "m", "created_at": "x", "message": {"role": "assistant", "content": ""},
        "done": True, "done_reason": "stop",
        "total_duration": 3_000_000_000, "load_duration": 900_000_000,
        "prompt_eval_duration": 500_000_000, "eval_duration": 1_600_000_000,
        "prompt_eval_count": 1842, "eval_count": 213,
    }
    base.update(kw)
    return base


def test_native_final_only_counts_and_latency():
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk("你好"))
    asm.feed(_native_final())
    gen = asm.build(model="m")
    assert gen.text == "你好"
    engine = gen.usage_from(TokenSource.ENGINE)
    assert engine.in_tokens == 1842 and engine.out_tokens == 213
    assert gen.latency.eval_ns == 1_600_000_000
    assert gen.latency.is_cold, "load_duration=900ms 必须判为冷启动"
    assert gen.finish_reason is FinishReason.STOP
    assert gen.decode_tps == pytest.approx(213 / 1.6, rel=1e-6)
    assert gen.prefill_tps == pytest.approx(1842 / 0.5, rel=1e-6)


def test_splitting_chunks_anywhere_gives_same_result():
    """分片切在哪都不该改变结果——这是流式实现最容易出错的地方。"""
    pieces = ["你", "好，", "世界", "！"]
    whole = StreamAssembler(style="native")
    whole.feed(_native_chunk("".join(pieces)))
    whole.feed(_native_final())

    for split in range(1, len(pieces)):
        asm = StreamAssembler(style="native")
        for piece in pieces[:split]:
            asm.feed(_native_chunk(piece))
        asm.feed(_native_chunk("".join(pieces[split:])))
        asm.feed(_native_final())
        assert asm.build().text == whole.build().text


def test_thinking_never_leaks_into_content():
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk(thinking="让我想想"))
    asm.feed(_native_chunk(content="答案是 42"))
    asm.feed(_native_final())
    gen = asm.build()
    assert gen.text == "答案是 42"
    assert gen.thinking == "让我想想"


def test_native_tool_call_with_dict_args():
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk(tool_calls=[{"function": {"name": "weather_now", "arguments": {"city": "北京"}}}]))
    asm.feed(_native_final(done_reason="tool_calls"))
    gen = asm.build()
    assert gen.finish_reason is FinishReason.TOOL_CALLS
    call = gen.tool_calls[0]
    assert call.name == "weather_now" and call.arguments == {"city": "北京"}
    assert call.parse_status is ParseStatus.OK


def test_tool_call_when_done_reason_is_stop():
    """实测常见：有 tool_calls 但 done_reason=stop。保留引擎事实，另给派生信号。"""
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk(tool_calls=[{"function": {"name": "t", "arguments": {}}}]))
    asm.feed(_native_final(done_reason="stop"))
    gen = asm.build()
    assert gen.finish_reason is FinishReason.STOP, "不许改写引擎报的 done_reason"
    assert gen.wants_tool_call is True, "工具循环必须仍能识别出要执行工具"


def test_wants_tool_call_when_reason_is_tool_calls():
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk(tool_calls=[{"function": {"name": "t", "arguments": {}}}]))
    asm.feed(_native_final(done_reason="tool_calls"))
    gen = asm.build()
    assert gen.finish_reason is FinishReason.TOOL_CALLS and gen.wants_tool_call


def test_openai_fragmented_tool_args_are_reassembled():
    asm = StreamAssembler(style="openai")
    fragments = ['{"ci', 'ty": "北', '京", "da', 'ys": 3}']
    for i, frag in enumerate(fragments):
        asm.feed({"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1" if i == 0 else None,
             "function": {"name": "weather" if i == 0 else None, "arguments": frag}},
        ]}}]})
    asm.feed({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
              "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    gen = asm.build()
    call = gen.tool_calls[0]
    assert call.name == "weather" and call.id == "call_1"
    assert call.arguments == {"city": "北京", "days": 3}
    assert call.parse_status is ParseStatus.OK
    compat = gen.usage_from(TokenSource.COMPAT)
    assert compat.in_tokens == 100 and compat.out_tokens == 20


def test_truncated_tool_json_is_distinguished_from_malformed():
    """截断（max_tokens 不够）与格式错（模型不会用）必须分开——修法完全不同。"""
    truncated = StreamAssembler(style="openai")
    truncated.feed({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "w", "arguments": '{"city": "北'}}]}}]})
    truncated.feed({"choices": [{"delta": {}, "finish_reason": "length"}]})
    call = truncated.build().tool_calls[0]
    assert call.parse_status is ParseStatus.TRUNCATED
    assert call.arguments_raw == '{"city": "北', "原文必须保住"
    assert call.arguments is None

    malformed = StreamAssembler(style="openai")
    malformed.feed({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "w", "arguments": "{city: 北京}"}}]}}]})
    malformed.feed({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    assert malformed.build().tool_calls[0].parse_status is ParseStatus.JSON_ERROR


def test_non_dict_tool_args_flagged():
    asm = StreamAssembler(style="openai")
    asm.feed({"choices": [{"delta": {"tool_calls": [
        {"index": 0, "function": {"name": "w", "arguments": "[1,2,3]"}}]}}]})
    asm.feed({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    assert asm.build().tool_calls[0].parse_status is ParseStatus.TYPE_MISMATCH


def test_missing_engine_counts_are_marked_not_zeroed():
    """引擎没报计数 → ok=False + note，绝不填 0（原则 4）。"""
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk("hi"))
    asm.feed({"done": True, "done_reason": "stop", "message": {}})
    engine = asm.build().usage_from(TokenSource.ENGINE)
    assert engine.ok is False and engine.in_tokens is None
    assert "未返回" in engine.note


def test_openai_usage_absent_without_include_usage():
    asm = StreamAssembler(style="openai")
    asm.feed({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
    gen = asm.build()
    assert gen.usage_from(TokenSource.COMPAT) is None
    assert gen.text == "hi"


def test_native_parallel_tool_calls_do_not_overwrite_each_other():
    """回归：原生 tool_calls 不带 index，曾全部落到 0 号累加器只剩最后一个。"""
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk(tool_calls=[
        {"function": {"name": "get_weather", "arguments": {"city": "北京"}}},
        {"function": {"name": "get_weather", "arguments": {"city": "上海"}}},
        {"function": {"name": "calc", "arguments": {"expr": "1+1"}}},
    ]))
    asm.feed(_native_final(done_reason="tool_calls"))
    calls = asm.build().tool_calls
    assert len(calls) == 3, f"并行调用被吞了: {[c.name for c in calls]}"
    assert [c.index for c in calls] == [0, 1, 2]
    assert {c.arguments["city"] for c in calls[:2]} == {"北京", "上海"}


def test_openai_fragments_still_merge_by_index():
    """OpenAI 风格带 index，同一 index 的分片必须合并而不是各成一个调用。"""
    asm = StreamAssembler(style="openai")
    for frag in ('{"ci', 'ty": "北京"}'):
        asm.feed({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "w", "arguments": frag}}]}}]})
    asm.feed({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
    calls = asm.build().tool_calls
    assert len(calls) == 1
    assert calls[0].arguments == {"city": "北京"}


def test_raw_chunk_count_recorded():
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk("a"))
    asm.feed(_native_final())
    assert asm.build().extra["raw_chunks"] == 2


def test_error_chunk_sets_status():
    asm = StreamAssembler(style="native")
    asm.feed({"error": "model 'x' not found"})
    gen = asm.build()
    assert gen.error == "model 'x' not found"
    assert gen.status.value == "error"


def test_reasoning_content_openai_style():
    asm = StreamAssembler(style="openai")
    asm.feed({"choices": [{"delta": {"reasoning_content": "思考中"}}]})
    asm.feed({"choices": [{"delta": {"content": "答案"}}]})
    asm.feed({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    gen = asm.build()
    assert gen.thinking == "思考中" and gen.text == "答案"


def test_ndjson_line_parsing_is_tolerant():
    """未解析的行由 client 标成 _unparsed，assembler 不该崩。"""
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk("a"))
    asm.feed(_native_final())
    gen = asm.build()
    assert json.loads(json.dumps({"n": gen.extra["raw_chunks"]}))["n"] == 2


def test_first_token_event_fires_at_the_moment_not_at_the_end():
    """S37：`trace.first_token_at` 落的是这个事件的墙钟时刻。

    等流结束补发的话，时刻会等于"结束"，那个字段就从"第一个字什么时候到"
    变成"我们什么时候想起要记"——两者在观测里长得一模一样，而后者没有意义。
    同时只许发一条：补发 + 即时发会让前端把首字渲染两次。
    """
    from onyx.core.clock import FakeClock
    from onyx.llm.streaming import consume_chunks

    events: list = []
    gen = consume_chunks(
        [_native_chunk("第"), _native_chunk("二段"), _native_final()],
        trace_id="T1", clock=FakeClock(), style="native", model="m",
        on_event=events.append,
    )
    kinds = [str(e.type) for e in events]
    assert kinds.count("first_token") == 1, f"首字事件重复：{kinds}"
    assert gen.ttft_ms is not None
    assert kinds.index("first_token") < max(i for i, k in enumerate(kinds) if k == "text_delta"), \
        "首字事件必须落在文本增量之间，而不是全部结束之后"


def test_non_streaming_proxy_ttft_is_labeled_a_proxy():
    """非流式没有"首字"这个量：事件仍发（带着 proxy），值不能冒充测量。"""
    from onyx.core.clock import FakeClock
    from onyx.llm.streaming import emit_final_events

    events: list = []
    asm = StreamAssembler(style="native")
    asm.feed(_native_chunk("答案"))
    asm.feed(_native_final())
    emit_final_events(asm.build(model="m"), _native_final(), trace_id="T2",
                      clock=FakeClock(), on_event=events.append, ttft_ms=None)
    first = [e for e in events if str(e.type) == "first_token"]
    assert len(first) == 1 and first[0].payload["proxy"] == "prompt_eval_duration"

