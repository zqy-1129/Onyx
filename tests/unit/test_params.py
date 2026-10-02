"""参数映射与请求构造测试（纯逻辑，无网络）。"""

from __future__ import annotations

import pytest

from onyx.core.errors import CapabilityMissing, SchemaInvalid
from onyx.core.types import GenerationRequest, GenParams, Message, Role, ToolCall, ToolSpec
from onyx.llm.params import (
    OLLAMA_OPENAI_UNSUPPORTED,
    ollama_options_dropped_by_openai,
    to_ollama_options,
    to_openai_params,
)
from onyx.llm.providers.ollama.native import build_chat_payload, parse_messages, parse_tools

STOP_WORD = "</" + "tool_response>"


def test_ollama_options_drops_unset_and_maps_names():
    params = GenParams(temperature=0.2, max_tokens=128, stop=(STOP_WORD,), num_ctx=8192)
    options = to_ollama_options(params)
    assert options == {
        "temperature": 0.2,
        "num_predict": 128,
        "stop": [STOP_WORD],
        "num_ctx": 8192,
    }, "max_tokens 必须映射成 Ollama 的 num_predict；未设置的参数不许出现"


def test_engine_specific_options_pass_through():
    options = to_ollama_options(GenParams(temperature=0.1, extra={"min_p": 0.05, "typical_p": 0.7}))
    assert options["min_p"] == 0.05 and options["typical_p"] == 0.7


def test_seed_zero_is_kept():
    assert to_ollama_options(GenParams(seed=0)) == {"seed": 0}, "seed=0 是有效值，不能被当成未设置"


def test_openai_mapping_records_dropped_params():
    out = to_openai_params(GenParams(temperature=0.3, top_k=40, repeat_penalty=1.1,
                                     extra={"min_p": 0.1}))
    assert out["temperature"] == 0.3
    assert "top_k" not in out and "repeat_penalty" not in out and "min_p" not in out
    assert set(out["_dropped_params"]) == {"top_k", "repeat_penalty", "min_p"}


def test_unsupported_openai_params_are_documented():
    assert {"tool_choice", "logit_bias", "user", "n"} == OLLAMA_OPENAI_UNSUPPORTED


def test_dropped_by_openai_helper():
    assert ollama_options_dropped_by_openai(GenParams(top_k=40, num_ctx=4096)) == ["top_k", "num_ctx"]
    assert ollama_options_dropped_by_openai(GenParams(temperature=0.5)) == []


def test_payload_minimal():
    req = GenerationRequest.of("qwen3.5:9b", "你好")
    payload = build_chat_payload(req, stream=False)
    assert payload == {
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": False,
    }


def test_payload_full():
    req = GenerationRequest(
        model="qwen3.5:9b",
        messages=(Message(role=Role.SYSTEM, content="s"), Message(role=Role.USER, content="u")),
        params=GenParams(temperature=0.0, seed=42, json_schema={"type": "object"}),
        tools=(ToolSpec(name="t", description="d", parameters={"type": "object", "properties": {}}),),
        thinking=False,
        keep_alive="5m",
        stream=True,
    )
    payload = build_chat_payload(req, stream=True)
    assert payload["think"] is False
    assert payload["keep_alive"] == "5m"
    assert payload["format"] == {"type": "object"}
    assert payload["options"] == {"temperature": 0.0, "seed": 42}
    assert payload["tools"][0]["function"]["name"] == "t"
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]


def test_tool_choice_is_refused_not_silently_dropped():
    """静默丢弃 tool_choice 会让"强制调用"评测悄悄变成"自由调用"，分数直接失真。"""
    req = GenerationRequest(model="m", messages=(), tool_choice="required")
    with pytest.raises(CapabilityMissing, match="tool_choice"):
        build_chat_payload(req, stream=False)


def test_tool_message_carries_tool_name():
    messages = (
        Message(role=Role.ASSISTANT, tool_calls=(ToolCall(name="weather_now", arguments={"city": "北京"}),)),
        Message(role=Role.TOOL, name="weather_now", content='{"temp": 21}'),
    )
    parsed = parse_messages(messages)
    assert parsed[0]["tool_calls"][0]["function"]["arguments"] == {"city": "北京"}
    assert parsed[1]["role"] == "tool" and parsed[1]["tool_name"] == "weather_now"


def test_media_refs_require_resolver():
    req = GenerationRequest(
        model="m",
        messages=(Message(role=Role.USER, content="看图", media_refs=("sha256:" + "a" * 64,)),),
    )
    with pytest.raises(SchemaInvalid, match="media_resolver"):
        build_chat_payload(req, stream=False)

    payload = build_chat_payload(req, stream=False, media_resolver=lambda ref: b"png-bytes")
    assert payload["messages"][0]["images"] == ["cG5nLWJ5dGVz"]


def test_tools_always_have_a_parameters_schema():
    parsed = parse_tools((ToolSpec(name="ping", description="p"),))
    assert parsed[0]["function"]["parameters"] == {"type": "object", "properties": {}}
