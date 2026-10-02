from __future__ import annotations

from onyx.core.types import (
    SOURCE_PRIORITY,
    Confidence,
    EngineLatency,
    FinishReason,
    Generation,
    GenerationRequest,
    GenParams,
    LoadedModel,
    Message,
    ModelDetail,
    Role,
    Status,
    TokenSample,
    TokenSource,
    ToolCall,
    ToolSpec,
    TraceContext,
    TraceKind,
    TracePurpose,
)


def test_genparams_drops_unset():
    """None 表示"不传"，绝不能变成 temperature=0 之类的静默默认值。"""
    assert GenParams().as_dict() == {}
    p = GenParams(temperature=0.2, max_tokens=128)
    assert p.as_dict() == {"temperature": 0.2, "max_tokens": 128}


def test_genparams_merge_routes_unknown_to_extra():
    p = GenParams(temperature=0.7).merge(top_k=40, min_p=0.05)
    assert p.temperature == 0.7 and p.top_k == 40
    assert p.extra == {"min_p": 0.05}, "引擎特有参数必须原样透传而不是被丢弃"


def test_toolspec_openai_roundtrip():
    spec = ToolSpec(
        name="weather_now",
        description="查询某城市当前天气",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    back = ToolSpec.from_openai_tool(spec.as_openai_tool())
    assert back == spec


def test_toolspec_empty_parameters_gets_valid_schema():
    tool = ToolSpec(name="ping").as_openai_tool()
    assert tool["function"]["parameters"] == {"type": "object", "properties": {}}


def test_message_helpers():
    assert Message(role=Role.USER).is_empty
    assert not Message(role=Role.USER, content="hi").is_empty
    msg = Message(role=Role.ASSISTANT, tool_calls=(ToolCall(name="x", arguments={"a": 1}),))
    assert not msg.is_empty and msg.tool_calls[0].name == "x"


def test_request_helpers_are_immutable():
    req = GenerationRequest.of("qwen3:8b", "hi", params=GenParams(temperature=0.1))
    req2 = req.with_params(top_k=20)
    assert req.params.top_k is None and req2.params.top_k == 20
    assert req2.messages == req.messages
    assert req.tool_names == ()


def test_trace_context_purpose_label():
    assert TraceContext().purpose_label == "chat"
    ctx = TraceContext(
        kind=TraceKind.GENERATION,
        purpose=TracePurpose.EVAL,
        eval_run_id="tool_selection",
        case_id="c-1",
        sample_seq=2,
    )
    assert ctx.purpose_label == "eval:tool_selection"
    assert ctx.case_id == "c-1" and ctx.sample_seq == 2


def test_engine_latency_cold_detection():
    assert EngineLatency(load_ns=900_000_000).is_cold, "900ms 载入必须判为冷启动"
    assert not EngineLatency(load_ns=1_000_000).is_cold
    assert not EngineLatency().is_cold
    assert EngineLatency(eval_ns=1_000_000_000).ms("eval") == 1000.0
    assert EngineLatency().ms("load") is None


def test_generation_derived_metrics_absent_when_data_missing():
    """没有引擎计数就算不出 TPS —— 返回 None 而不是 0（原则 4）。"""
    assert Generation().decode_tps is None
    assert Generation().prefill_tps is None
    gen = Generation(
        usage=(TokenSample(source=TokenSource.ENGINE, in_tokens=1000, out_tokens=100),),
        latency=EngineLatency(prompt_eval_ns=500_000_000, eval_ns=2_000_000_000),
    )
    assert gen.prefill_tps == 2000.0
    assert gen.decode_tps == 50.0
    assert gen.usage_from(TokenSource.ENGINE).in_tokens == 1000
    assert gen.usage_from(TokenSource.COMPAT) is None


def test_source_priority_excludes_compat():
    assert SOURCE_PRIORITY[0] is TokenSource.ENGINE
    assert TokenSource.COMPAT not in SOURCE_PRIORITY, "兼容层只作交叉验证，不参与采信"


def test_loaded_model_offload_detection():
    assert LoadedModel(name="a", size=100, size_vram=60).offloaded
    assert LoadedModel(name="a", size=100, size_vram=100).offloaded is False
    assert LoadedModel(name="a", size=0).offloaded is False
    assert LoadedModel(name="a", size=200, size_vram=50).vram_share == 0.25


def test_model_detail_tokenizer_family():
    d = ModelDetail(name="qwen3:8b", model_info={"tokenizer.ggml.model": "qwen2"})
    assert d.tokenizer_family == "qwen2"
    assert ModelDetail(name="x").tokenizer_family == ""


def test_status_and_finish_reason_are_strings():
    """枚举值即持久化值：落库/进 JSON 必须是稳定字符串。"""
    assert Status.OK == "ok" and FinishReason.TOOL_CALLS == "tool_calls"
    assert Confidence.LOW == "low"
