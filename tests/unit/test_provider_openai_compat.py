"""S16b：OpenAI 兼容通道（vLLM / LM Studio / Xinference / Ollama `/v1`）。

这个 provider 的意义有两层，测试也分两层：
1. **抽象验收**：只实现 `LlmProvider`（没有控制面）能不能撑起看板与评测；
2. **通道差异必须被承认**：兼容层缺纳秒时序、缺模板、计数出处是 `compat` 而不是
   `engine`，参数在 `/v1` 上没有等价物。这些差异如果被抹平，
   两次"同一个模型"的分数就再也无法解释为什么不一样。

全部测试用 `httpx.MockTransport`，零真实网络（import-linter 也只允许 llm/tools.http 碰网络）。
"""

from __future__ import annotations

import json

import httpx
import pytest

from onyx.core.errors import CapabilityMissing, ProviderRejected, ProviderUnreachable
from onyx.core.types import (
    Cap,
    GenerationRequest,
    GenParams,
    Message,
    Role,
    TokenSource,
    ToolSpec,
)
from onyx.llm.providers.openai_compat import CompatClient, OpenAICompatProvider

BASE = "http://compat.test/v1"


def _provider(handler, *, api_key: str | None = None, **kw) -> OpenAICompatProvider:
    """构造一个带 MockTransport 的 provider。

    `api_key` 必须交给 client 而不是 provider：provider 收到现成 client 时
    不该再去管密钥，否则 `client=` 与 `api_key=` 同时给出会静默丢掉密钥。
    """
    return OpenAICompatProvider(
        id="compat-test", base_url=BASE,
        client=CompatClient(BASE, api_key=api_key,
                            transport=httpx.MockTransport(handler)), **kw
    )


def _completion(handler_payload=None, **kw):
    """非流式补全的默认响应；参数可覆盖以便测不同形态。"""
    body = handler_payload or {
        "id": "c1", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "你好"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m", "created": 1700000000}]})
        return httpx.Response(200, json=body)

    return _provider(handler, **kw)


def _capturing(body, sink: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "m"}]})
        sink["payload"] = json.loads(request.read())
        sink["headers"] = dict(request.headers)
        sink["path"] = request.url.path
        return httpx.Response(200, json=body)

    return handler


OK_BODY = {
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "好"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 1},
}


# ── 请求体：没设的参数不许出现 ──────────────────────────────────────
def test_unset_params_are_absent_not_defaulted():
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    provider.generate(GenerationRequest.of("m", "hi"))
    payload = sink["payload"]
    assert "temperature" not in payload and "max_tokens" not in payload, \
        "填默认值等于冻结一个假设，之后无法解释「这次到底是按什么参数跑的」"
    assert payload["messages"][0] == {"role": "user", "content": "hi"}
    assert payload["stream"] is False


def test_streaming_must_ask_for_usage():
    """R2：不显式打开 `stream_options.include_usage` 时 usage 恒为 0，
    而 0 会被读成"模型没产 token"——那是把配置错误记成模型行为。"""
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    provider.generate(GenerationRequest.of("m", "hi", stream=True))
    assert sink["payload"]["stream_options"] == {"include_usage": True}


def test_params_map_and_no_equivalent_is_not_faked():
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    req = GenerationRequest.of(
        "m", "hi",
        params=GenParams(temperature=0.3, top_k=40, repeat_penalty=1.1, max_tokens=32),
    )
    provider.generate(req)
    payload = sink["payload"]
    assert payload["temperature"] == 0.3 and payload["max_tokens"] == 32
    # `/v1` 没有 top_k / repeat_penalty：不许翻译成别的键，也不许静默塞进 extra_body
    assert "top_k" not in payload and "repeat_penalty" not in payload
    assert "_dropped_params" not in payload, "内部记账键绝不能发给引擎"


def test_thinking_request_is_refused_when_the_channel_cannot_express_it():
    """P23 实测：Ollama `/v1` 既不认 `think` 也不认 `chat_template_kwargs.thinking`。

    "要求关 thinking 却被静默忽略"会让两次分数不可比（thinking 吃光预算 ⇒ 正文空 ⇒
    被判 invalid_format），所以这里宁可拒绝，也不假装设置过了。
    """
    provider = _completion()
    with pytest.raises(CapabilityMissing) as exc:
        provider.generate(GenerationRequest.of("m", "hi", thinking=False))
    assert "thinking" in str(exc.value) and "thinking_via" in str(exc.value)


def test_thinking_is_translated_when_declared_by_the_operator():
    """声明了该服务器的开关系在哪里，就按声明翻译（vLLM 一类支持模板参数）。"""
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink), thinking_via="chat_template_kwargs.thinking")
    provider.generate(GenerationRequest.of("m", "hi", thinking=False))
    assert sink["payload"]["chat_template_kwargs"] == {"thinking": False}


def test_no_thinking_request_sends_no_thinking_keys():
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink), thinking_via="chat_template_kwargs.thinking")
    provider.generate(GenerationRequest.of("m", "hi"))
    assert "chat_template_kwargs" not in sink["payload"], "没要求就不要替操作者决定"


def test_req_extra_is_the_documented_escape_hatch():
    """vLLM 的 `chat_template_kwargs` 之类专有参数由调用方显式带入。

    provider 不猜服务器型号：一旦按 base_url/名字分支，就会出现"换了个端口行为就变"
    这种查不出来的问题。
    """
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    provider.generate(GenerationRequest.of(
        "m", "hi", extra={"chat_template_kwargs": {"enable_thinking": False}},
    ))
    assert sink["payload"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_tools_use_the_openai_function_shape():
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    provider.generate(GenerationRequest.of(
        "m", "hi", tools=(ToolSpec(name="get_weather", description="查天气",
                                  parameters={"type": "object", "properties": {}}),),
    ))
    tool = sink["payload"]["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "get_weather"
    assert tool["function"]["parameters"] == {"type": "object", "properties": {}}


@pytest.mark.parametrize("choice,expected", [
    ("auto", "auto"), ("required", "required"), ("none", "none"),
    ("get_weather", {"type": "function", "function": {"name": "get_weather"}}),
])
def test_tool_choice_passes_through(choice, expected):
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    provider.generate(GenerationRequest.of(
        "m", "hi", tool_choice=choice,
        tools=(ToolSpec(name="get_weather", description="查天气"),),
    ))
    assert sink["payload"]["tool_choice"] == expected


def test_tool_choice_naming_an_absent_tool_fails_loudly():
    """悄悄降级成 auto 会产出一个"看起来强制调用了"的分数。"""
    provider = _provider(_capturing(OK_BODY, {}))
    with pytest.raises(CapabilityMissing) as exc:
        provider.generate(GenerationRequest.of(
            "m", "hi", tool_choice="send_email",
            tools=(ToolSpec(name="get_weather", description="查天气"),),
        ))
    assert "send_email" in str(exc.value)


def test_json_schema_becomes_response_format():
    sink: dict = {}
    provider = _provider(_capturing(OK_BODY, sink))
    provider.generate(GenerationRequest.of(
        "m", "hi", params=GenParams(json_schema={"type": "object", "properties": {}}),
    ))
    assert sink["payload"]["response_format"]["type"] == "json_schema"


# ── 响应解析：两条路径必须同一个口径 ────────────────────────────────
def test_non_stream_message_shape_is_parsed():
    """兼容层的非流式响应是 `choices[].message`，流式才是 `delta`。

    只认 delta 的话，正文会解析成空串——而空正文与"模型真的没输出"在评测里
    会长成同一个 `invalid_format`。
    """
    gen = _completion().generate(GenerationRequest.of("m", "hi"))
    assert gen.text == "你好"
    assert str(gen.finish_reason) == "stop"
    sample = gen.usage_from(TokenSource.COMPAT)
    assert sample is not None and sample.in_tokens == 11 and sample.out_tokens == 2
    assert gen.usage_from(TokenSource.ENGINE) is None, \
        "出处必须记 compat：把它当 engine 会掩盖两条通道计数口径的差异"


def test_stream_parses_sse_and_final_usage():
    lines = [
        {"choices": [{"index": 0, "delta": {"content": "你"}}]},
        {"choices": [{"index": 0, "delta": {"content": "好"}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in lines) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})

    gen = _provider(handler).generate(GenerationRequest.of("m", "hi", stream=True))
    assert gen.text == "你好"
    assert gen.usage_from(TokenSource.COMPAT).out_tokens == 3


def test_tool_call_fragments_are_reassembled():
    body = {
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {
                "name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}],
        }, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 9},
    }
    gen = _completion(body).generate(GenerationRequest.of("m", "hi"))
    call = gen.tool_calls[0]
    assert call.name == "get_weather" and call.arguments == {"city": "北京"}
    assert str(gen.finish_reason) == "tool_calls" and gen.wants_tool_call


def test_reasoning_content_lands_in_thinking():
    body = {
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "答案是 3",
            "reasoning_content": "先数一遍……"}, "finish_reason": "stop"}],
    }
    gen = _completion(body).generate(GenerationRequest.of("m", "hi"))
    assert gen.thinking.startswith("先数一遍") and gen.text == "答案是 3"


def test_no_engine_timing_means_unknown_throughput():
    """兼容通道没有纳秒级分段时序：`latency` 为空 ⇒ 吞吐显示「—」。

    这里如果"算"出一个 TPS，它是拿客户端墙钟猜的，会和引擎口径冲突。
    """
    gen = _completion().generate(GenerationRequest.of("m", "hi"))
    assert gen.latency is None


def test_unparsable_error_body_still_raises_typed_error():
    """错误体不是 JSON 时也必须归一成 `ProviderRejected`，并把原文留着。

    "引擎回了我们看不懂的东西"本身就是需要被观测的事实。
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="boom not json")

    with pytest.raises(ProviderRejected) as exc:
        _provider(handler).generate(GenerationRequest.of("m", "hi"))
    assert exc.value.detail["status"] == 400
    assert "boom not json" in exc.value.detail["body"]


def test_unreachable_is_normalized():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(ProviderUnreachable):
        _provider(boom).generate(GenerationRequest.of("m", "hi"))


# ── 控制面缺席与能力位 ─────────────────────────────────────────────
def test_admin_capability_is_not_claimed_and_running_is_empty():
    """没有统一的控制面就不要声明 ADMIN。

    声明了，`onyx eval run` 就会以为"能腾显存"从而不再 unload 别的模型，
    结果是两个模型同时驻留、触发 offload，分数被污染却看起来正常。
    """
    provider = _completion()
    assert provider.running() == []
    assert Cap.ADMIN not in provider.capabilities()
    assert not hasattr(provider, "unload"), "不要提供假的 unload"


def test_declared_caps_are_validated_at_construction():
    """拼错能力位会被静默丢弃的话，评测就在"没人支持的能力"上跑出正常分数。"""
    with pytest.raises(ValueError, match="未知能力位"):
        _provider(lambda r: httpx.Response(200, json={} ), caps=("chat", "tool_chioce"))
    assert Cap.TOOLS in _provider(
        lambda r: httpx.Response(200, json={}), caps=("chat", "tools")
    ).capabilities()


def test_default_caps_are_minimal():
    """默认只声明 CHAT：其它能力必须由操作者显式声明或探针实测确认。"""
    provider = _completion()
    assert provider.capabilities() == frozenset({Cap.CHAT})


def test_show_model_reports_template_unknown_instead_of_inventing_one():
    """没有 /api/show：模板未知就必须标出来。

    本地归因的 `template_ctl` 一档依赖模板；编一个模板会算出一个像模像样的数字，
    而它其实是猜的。
    """
    detail = _completion().show_model("m")
    assert detail.name == "m" and detail.template == ""
    assert detail.extra["template_source"] == "unknown"


def test_show_model_unknown_name_lists_what_exists():
    with pytest.raises(LookupError) as exc:
        _completion().show_model("nope")
    assert "m" in str(exc.value)


def test_keep_alive_is_refused_not_ignored():
    """keep_alive 是 Ollama 原生概念。忽略它会让"模型常驻"这个假设静默失效，
    然后下一个模型加载失败时没人知道原因。"""
    provider = _completion()
    with pytest.raises(CapabilityMissing):
        provider.generate(GenerationRequest.of("m", "hi", keep_alive="30m"))


def test_images_are_refused_not_dropped():
    provider = _completion()
    with pytest.raises(CapabilityMissing) as exc:
        provider.generate(GenerationRequest.of("m", "hi").with_messages([
            Message(role=Role.USER, content="看图", media_refs=("sha256:abc",))
        ]))
    assert "图片" in str(exc.value)


def test_tool_result_without_id_is_refused():
    provider = _completion()
    with pytest.raises(CapabilityMissing):
        provider.generate(GenerationRequest.of("m", "hi").with_messages([
            Message(role=Role.TOOL, content="晴", name="get_weather")
        ]))


def test_api_key_header_present_only_when_given(monkeypatch):
    sink: dict = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _provider(_capturing(OK_BODY, sink), api_key="sk-test").generate(
        GenerationRequest.of("m", "hi")
    )
    assert sink["headers"]["authorization"] == "Bearer sk-test"

    sink2: dict = {}
    _provider(_capturing(OK_BODY, sink2)).generate(GenerationRequest.of("m", "hi"))
    assert "authorization" not in {k.lower() for k in sink2["headers"]}, \
        "没给密钥就不要发一个空的 Bearer"


def test_api_key_falls_back_to_the_standard_env(monkeypatch):
    sink: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    _provider(_capturing(OK_BODY, sink)).generate(GenerationRequest.of("m", "hi"))
    assert sink["headers"]["authorization"] == "Bearer sk-from-env"


def test_provider_level_api_key_reaches_the_client(monkeypatch):
    """生产路径（CLI 只给 `--provider openai-compat`，没有现成 client）也必须带上密钥。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    sink: dict = {}
    provider = OpenAICompatProvider(
        id="p", base_url=BASE, api_key="sk-direct",
        transport=httpx.MockTransport(_capturing(OK_BODY, sink)),
    )
    provider.generate(GenerationRequest.of("m", "hi"))
    assert sink["headers"]["authorization"] == "Bearer sk-direct"


def test_info_reports_reachability_without_a_version_claim():
    provider = _completion()
    info = provider.info()
    assert info.reachable is True and info.version == "", "拿不到版本就留空，不编"
    assert str(info.api_style) == "openai"
