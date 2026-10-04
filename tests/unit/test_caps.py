"""能力位三态推断测试。

核心断言：`✗ 不支持` 与 `? 未实测` 必须是**不同的状态**，且各自给出不同的处置建议。
把两者合成一个，评测就会要么错杀模型、要么拿没验证过的能力去跑出无法解释的分数。
"""

from __future__ import annotations

from onyx.core.types import ApiStyle, Cap, ProviderKind
from onyx.llm.caps import ALL_CAPS, CapReport, infer_caps

QWEN_ENGINE_CAPS = ("completion", "vision", "tools", "thinking")


def _ollama(**kw):
    return infer_caps(
        engine_capabilities=QWEN_ENGINE_CAPS, api_style=ApiStyle.NATIVE,
        provider_kind=ProviderKind.OLLAMA, **kw,
    )


def test_engine_capabilities_map_to_confirmed():
    report = _ollama()
    for cap in (Cap.CHAT, Cap.TOOLS, Cap.THINKING, Cap.VISION):
        assert report.state(cap) == "confirmed", f"{cap} 应确认支持"
        assert report.supports(cap)
        assert report.symbol(cap) == "✓"


def test_unlisted_engine_capability_is_missing():
    report = _ollama()
    assert report.state(Cap.EMBED) == "missing"
    assert report.symbol(Cap.EMBED) == "✗"
    assert "未列出" in report.reasons[str(Cap.EMBED)]


def test_ollama_known_missing_has_documented_reason():
    report = _ollama()
    for cap in (Cap.TOOL_CHOICE, Cap.N_SAMPLING, Cap.LOGPROBS):
        assert report.state(cap) == "missing"
        assert "官方文档" in report.reasons[str(cap)]


def test_unprobed_capabilities_are_unknown_not_missing():
    """结构化输出与流式计数需要探针才能判定，没跑过就是 ?，不是 ✗。"""
    report = _ollama()
    for cap in (Cap.STRUCTURED_OUTPUT, Cap.STREAM_USAGE):
        assert report.state(cap) == "unknown"
        assert report.symbol(cap) == "?"
        assert "onyx probe run" in report.skip_reason(cap)


def test_probe_findings_override_engine_claims():
    report = _ollama(probe_findings={
        "structured": "enforced",
        "stream_usage": "identical",
        "tool_format": "native_head",
    })
    assert report.state(Cap.STRUCTURED_OUTPUT) == "confirmed"
    assert report.state(Cap.STREAM_USAGE) == "confirmed"
    assert "探针" in report.reasons[str(Cap.STRUCTURED_OUTPUT)]


def test_probe_can_downgrade_an_engine_claim():
    """引擎自称支持 tools，但实测发起不了调用 ⇒ 降级为未实测，而不是继续相信自报。"""
    report = _ollama(probe_findings={"tool_format": "plain_text_no_tool_call"})
    assert report.state(Cap.TOOLS) == "unknown"
    assert not report.supports(Cap.TOOLS)


def test_thinking_not_produced_downgrades_to_unknown():
    report = _ollama(probe_findings={"think": "thinking_not_produced_by_this_model"})
    assert report.state(Cap.THINKING) == "unknown"


def test_negative_structured_probe_is_missing():
    report = _ollama(probe_findings={"structured": "not_enforced_invalid_json"})
    assert report.state(Cap.STRUCTURED_OUTPUT) == "missing"
    assert "不支持" in report.skip_reason(Cap.STRUCTURED_OUTPUT)


def test_every_cap_lands_in_exactly_one_state():
    """即使探针覆盖了引擎自报，也不许出现同一个 cap 处于两态。"""
    for findings in (
        {},
        {"tool_format": "plain_text_no_tool_call"},
        {"think": "thinking_not_produced_by_this_model", "structured": "enforced"},
        {"stream_usage": "divergent", "tool_format": "native_head"},
    ):
        report = _ollama(probe_findings=findings)
        assert report.confirmed | report.missing | report.unknown == ALL_CAPS
        assert not (report.confirmed & report.missing), findings
        assert not (report.confirmed & report.unknown), findings
        assert not (report.missing & report.unknown), findings
        assert set(report.reasons) == {str(c) for c in ALL_CAPS}


def test_skip_reason_empty_only_when_confirmed():
    report = _ollama()
    assert report.skip_reason(Cap.TOOLS) == ""
    assert report.skip_reason(Cap.TOOL_CHOICE)
    assert report.skip_reason(Cap.STRUCTURED_OUTPUT)


def test_openai_compat_channel_supports_tool_choice():
    report = infer_caps(
        engine_capabilities=QWEN_ENGINE_CAPS, api_style=ApiStyle.OPENAI,
        provider_kind=ProviderKind.OPENAI_COMPAT,
    )
    assert report.state(Cap.TOOL_CHOICE) == "confirmed"
    assert report.state(Cap.ADMIN) == "unknown", "该通道没实现控制面就不该声称支持"


def test_absent_capability_list_is_unknown_not_missing():
    """`/v1/models` 不汇报 per-model capabilities ⇒ 空清单的含义是「没上报」。

    读成 missing 会给出 ✗，而 ✗ 的处置是「评测直接 skip」；? 的处置才是「先跑探针」。
    把不报告写成不支持，等于让看板替服务器撒谎——这条是在真机 `/v1` 上
    跑第二个 provider 时暴露的。
    """
    report = infer_caps(
        engine_capabilities=(), api_style=ApiStyle.OPENAI,
        provider_kind=ProviderKind.OPENAI_COMPAT,
    )
    assert report.state(Cap.CHAT) == "unknown"
    assert report.state(Cap.TOOLS) == "unknown"
    assert Cap.CHAT not in report.missing
    assert "不汇报" in report.reasons[str(Cap.CHAT)]

    # 反向保证：真的上报了且不含某项，才允许判 missing
    reported = infer_caps(engine_capabilities=["completion"], api_style=ApiStyle.NATIVE)
    assert reported.state(Cap.CHAT) == "confirmed"
    assert reported.state(Cap.TOOLS) == "missing"


def test_dict_roundtrip():
    report = _ollama(probe_findings={"structured": "enforced"})
    restored = CapReport.from_dict(report.as_dict())
    assert restored.confirmed == report.confirmed
    assert restored.missing == report.missing
    assert restored.unknown == report.unknown
    assert restored.reasons == report.reasons


def test_from_dict_tolerates_garbage():
    empty = CapReport.from_dict(None)
    assert empty.confirmed == frozenset() and empty.state(Cap.TOOLS) == "unknown"
    bogus = CapReport.from_dict({"confirmed": ["not_a_cap"], "reasons": {"x": 1}})
    assert bogus.confirmed == frozenset()
    assert bogus.reasons == {"x": "1"}
