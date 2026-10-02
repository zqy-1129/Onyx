"""S3 验收：对着**真实运行的 Ollama** 验证适配器。

方案纪律（IMPLEMENTATION S3）：不做 mock-only 开发。mock 会把没验证过的假设固化进代码，
而 token 语义、缓存行为、工具格式这些东西只有真实引擎能回答。

跑法：`uv run pytest -m live -q -s`
可用 ONYX_TEST_MODEL 指定模型（默认 qwen3.5:9b，6.6GB，16GB 显存可全量载入）。
"""

from __future__ import annotations

import os

import pytest

from onyx.core.event import EventType
from onyx.core.types import Cap, GenerationRequest, GenParams, Message, Role, TokenSource, ToolSpec
from onyx.llm.providers.ollama import OllamaProvider

pytestmark = pytest.mark.live

MODEL = os.environ.get("ONYX_TEST_MODEL", "qwen3.5:9b")
BASE_URL = os.environ.get("ONYX_OLLAMA_URL", "http://127.0.0.1:11434")


@pytest.fixture(scope="module")
def provider():
    p = OllamaProvider(base_url=BASE_URL)
    if not p.client.is_reachable():
        pytest.skip(f"Ollama 不可达: {BASE_URL}")
    names = [m.name for m in p.list_models()]
    if MODEL not in names:
        pytest.skip(f"模型 {MODEL} 未安装，现有: {names}")
    yield p
    p.close()


@pytest.fixture(scope="module")
def warm(provider):
    """预热一次，把冷启动载入与后续测量分开（DESIGN §6.4）。"""
    provider.generate(GenerationRequest.of(MODEL, "hi", params=GenParams(max_tokens=8)))
    return provider


def test_version_and_base_capabilities(provider):
    info = provider.info()
    assert info.reachable and info.version, "必须能读到引擎版本"
    caps = provider.capabilities()
    assert Cap.CHAT in caps and Cap.TOOLS in caps and Cap.ADMIN in caps
    assert Cap.TOOL_CHOICE not in caps, "Ollama 不支持 tool_choice，能力位必须如实反映"
    assert Cap.N_SAMPLING not in caps
    print(f"\n[provider] ollama v{info.version} caps={sorted(str(c) for c in caps)}")


def test_list_models_carries_capabilities_and_context(provider):
    """实测发现：/api/tags 直接带 capabilities 与 details.context_length（官方文档未列出）。"""
    cards = {m.name: m for m in provider.list_models()}
    card = cards[MODEL]
    assert card.bytes > 0 and card.digest
    assert card.capabilities, "capabilities 缺失会让能力位推断退化成猜测"
    assert "tools" in card.capabilities
    assert card.context_length and card.context_length > 0
    assert card.quantization and card.parameter_size
    print(f"\n[tags] {card.name} {card.parameter_size} {card.quantization} "
          f"{card.bytes / 1e9:.1f}GB caps={card.capabilities} ctx={card.context_length}")


def test_show_model_exposes_template_and_gguf_metadata(provider):
    """/api/show 的 model_info 是本地 token 复算的原料，必须真的拿到。"""
    detail = provider.show_model(MODEL)
    assert detail.template, "chat template 缺失 ⇒ 无法做 prompt 归因"
    assert detail.model_info, "model_info 缺失 ⇒ 无法离线复算 token"
    family = detail.tokenizer_family
    assert family, f"tokenizer.ggml.model 缺失，实际键: {list(detail.model_info)[:8]}"
    tokenizer_keys = [k for k in detail.model_info if k.startswith("tokenizer.")]
    print(f"\n[show] tokenizer_family={family} tokenizer_keys={len(tokenizer_keys)} "
          f"capabilities={detail.capabilities}")
    print(f"[show] template_head={detail.template[:80]!r}")


def test_generate_returns_engine_counts_and_latency(warm):
    """常规生成路径。

    必须显式 `thinking=False`：实测该模型默认开推理，512 token 预算会被 thinking
    吃光导致正文为空（见 test_thinking_can_consume_the_entire_budget）。
    """
    events = []
    gen = warm.generate(
        GenerationRequest.of(
            MODEL, "用一句话解释什么是 token", params=GenParams(max_tokens=256), thinking=False
        ),
        trace_id="live-1",
        on_event=events.append,
    )
    engine = gen.usage_from(TokenSource.ENGINE)
    assert engine is not None and engine.ok, "引擎必须报计数"
    assert engine.in_tokens > 0 and engine.out_tokens > 0
    assert gen.latency and gen.latency.eval_ns > 0
    assert gen.text.strip(), f"正文为空但 thinking={len(gen.thinking)} 字符（预算被推理吃光？）"
    assert not gen.thinking, "thinking=False 必须真的关掉推理"
    types = [e.type for e in events]
    assert EventType.USAGE_ENGINE in types and EventType.GENERATION_END in types
    print(f"\n[native] in={engine.in_tokens} out={engine.out_tokens} text_chars={len(gen.text)}")
    print(f"[native] decode_tps={gen.decode_tps:.1f} prefill_tps={gen.prefill_tps:.1f} "
          f"cold={gen.latency.is_cold} finish={gen.finish_reason}")
    print(f"[native] latency total={gen.latency.ms('total')}ms load={gen.latency.ms('load')}ms "
          f"prompt_eval={gen.latency.ms('prompt_eval')}ms eval={gen.latency.ms('eval')}ms")


def test_thinking_can_consume_the_entire_budget(warm):
    """实测坑：小预算 + 推理模型 ⇒ 正文空、finish_reason=length。

    这对评测是致命的——模型不是"答错"，是"没预算答"。看板必须能区分这两种失败，
    所以这个行为被固化成测试，将来 S5 的 anomaly visitor 要据此报 EMPTY_CONTENT_WITH_THINKING。
    """
    gen = warm.generate(
        GenerationRequest.of(MODEL, "用一句话解释什么是 token", params=GenParams(max_tokens=64))
    )
    engine = gen.usage_from(TokenSource.ENGINE)
    print(f"\n[budget] out={engine.out_tokens} text={len(gen.text)} thinking={len(gen.thinking)} "
          f"finish={gen.finish_reason}")
    if not gen.text.strip():
        assert gen.thinking.strip(), "正文空时 thinking 也空 ⇒ 另有问题"
        assert gen.finish_reason.value == "length", "预算耗尽必须如实报 length，不许伪装成 stop"
        assert engine.out_tokens >= 60, "输出计数应逼近 max_tokens 上限"


def test_stream_and_non_stream_agree_on_input_tokens(warm):
    """口径一致性：同一 prompt 走流式与非流式，输入 token 必须完全相同。"""
    prompt = "请数到五，只输出数字。"
    params = GenParams(temperature=0.0, seed=42, max_tokens=256)
    plain = warm.generate(GenerationRequest.of(MODEL, prompt, params=params, thinking=False))
    streamed = warm.generate(
        GenerationRequest.of(MODEL, prompt, params=params, stream=True, thinking=False)
    )

    a = plain.usage_from(TokenSource.ENGINE)
    b = streamed.usage_from(TokenSource.ENGINE)
    assert a.in_tokens == b.in_tokens, f"输入计数不一致: {a.in_tokens} vs {b.in_tokens}"
    assert a.out_tokens == b.out_tokens, (
        f"temperature=0+seed 固定时输出计数应一致: {a.out_tokens} vs {b.out_tokens}"
    )
    assert streamed.text == plain.text
    assert streamed.ttft_ms is not None and streamed.ttft_ms > 0
    assert plain.ttft_ms is None, "非流式没有真实 TTFT，不许编造"
    print(f"\n[stream] in={b.in_tokens} out={b.out_tokens} ttft={streamed.ttft_ms:.1f}ms "
          f"decode_tps={streamed.decode_tps:.1f}")


def test_loaded_model_visible_in_ps(warm):
    loaded = {m.name: m for m in warm.running()}
    assert MODEL in loaded, f"生成之后模型应处于载入态，实际: {list(loaded)}"
    m = loaded[MODEL]
    assert m.size > 0 and m.size_vram > 0
    assert m.expires_at, "expires_at 缺失 ⇒ keep-alive 倒计时无法显示"
    print(f"\n[ps] {m.name} size={m.size / 1e9:.2f}GB vram={m.size_vram / 1e9:.2f}GB "
          f"ctx={m.context_length} offloaded={m.offloaded} expires_at={m.expires_at}")


def test_native_tool_calling(warm):
    tool = ToolSpec(
        name="get_weather",
        description="查询指定城市当前天气",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    )
    req = GenerationRequest(
        model=MODEL,
        messages=(Message(role=Role.USER, content="北京现在天气怎么样？"),),
        tools=(tool,),
        params=GenParams(temperature=0.0, max_tokens=256),
    )
    gen = warm.generate(req, trace_id="live-tool")
    assert gen.tool_calls, f"模型未发起工具调用；原文={gen.text[:200]!r}"
    call = gen.tool_calls[0]
    assert call.name == "get_weather"
    assert call.parse_status.value == "ok", f"参数解析失败: {call.parse_status} raw={call.arguments_raw!r}"
    assert call.arguments.get("city"), f"缺少必填参数 city: {call.arguments}"
    assert gen.wants_tool_call, "工具循环的决策信号必须为真"
    assert gen.finish_reason.value in {"stop", "tool_calls"}, (
        f"实测 Ollama 会在返回 tool_calls 时报 done_reason=stop，故引擎事实原样保留、"
        f"由 wants_tool_call 派生决策；实际={gen.finish_reason}"
    )
    print(f"\n[tool] name={call.name} args={call.arguments} parse={call.parse_status.value} "
          f"finish={gen.finish_reason} wants_tool_call={gen.wants_tool_call}")


def test_thinking_can_be_disabled(warm):
    """think 开关必须真的生效，否则推理 token 口径无法解释。"""
    req = GenerationRequest.of(MODEL, "9.11 和 9.9 哪个大？", params=GenParams(max_tokens=256))
    with_think = warm.generate(GenerationRequest(
        model=MODEL, messages=req.messages, params=req.params, thinking=True
    ))
    without = warm.generate(GenerationRequest(
        model=MODEL, messages=req.messages, params=req.params, thinking=False
    ))
    print(f"\n[think] on: thinking_chars={len(with_think.thinking)} text_chars={len(with_think.text)} "
          f"out={with_think.usage_from(TokenSource.ENGINE).out_tokens}")
    print(f"[think] off: thinking_chars={len(without.thinking)} text_chars={len(without.text)} "
          f"out={without.usage_from(TokenSource.ENGINE).out_tokens}")
    assert not without.thinking, "think=false 时不应产生 thinking 文本"


def test_unload_releases_model(warm):
    result = warm.unload(MODEL)
    assert result.ok, result.error
    assert MODEL not in {m.name for m in warm.running()}, "keep_alive=0 之后模型应已卸载"
    print(f"\n[unload] ok={result.ok} 已卸载 {MODEL}")
