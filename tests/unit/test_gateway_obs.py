"""S5 验收：gateway → 事件 → 观测 → 落库的完整管道。

用 MockProvider 驱动，因为它走的是与真实适配器**同一条** `consume_chunks` 路径，
所以这里测的管道与真机管道等价；真机数值另由 tests/integration 覆盖。
"""

from __future__ import annotations

import pytest

from onyx.core.content import FileBlobStore
from onyx.core.errors import ProviderUnreachable
from onyx.core.event import EventType, TraceEvent
from onyx.core.types import (
    Confidence,
    GenerationRequest,
    TokenSource,
    ToolSpec,
    TraceContext,
    TracePurpose,
)
from onyx.llm.gateway import Gateway
from onyx.llm.measurement.fidelity import CounterContext
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.llm.registry import available_kinds, build_provider
from onyx.obs.engine import ObserverEngine
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor, default_visitors
from onyx.store.db import Database
from onyx.store.repos import TraceRepo, UsageRepo
from onyx.store.sinks import SqliteRecordSink

WEATHER = ToolSpec(
    name="get_weather", description="查询城市天气",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    observer = ObserverEngine(record_sink=sink)
    blobs = FileBlobStore(tmp_path / "blobs")
    yield db, sink, observer, blobs
    sink.close()
    db.close()


def _gateway(env, provider, **kw) -> Gateway:
    _, _, observer, blobs = env
    return Gateway(provider, observer=observer, blobs=blobs, **kw)


def _flush(env) -> None:
    env[1].flush(5.0)


def _req(model="mock/echo", **kw) -> GenerationRequest:
    return GenerationRequest.of(model, "北京天气怎么样？", **kw)


def _codes(result) -> set[str]:
    return {code for code, _, _ in result.anomalies}


# ── 正常路径 ───────────────────────────────────────────────────────
def test_happy_path_lands_a_complete_trace(env):
    db, _, _, blobs = env
    provider = MockProvider(scripts={"mock/echo": MockScript(
        text="北京晴，21 度。", in_tokens=280, out_tokens=26,
        prompt_eval_ns=164_860_000, eval_ns=596_706_000, load_ns=1_662_200,
    )})
    result = _gateway(env, provider).generate(_req(), purpose=TracePurpose.CHAT)
    _flush(env)

    traces = TraceRepo(db)
    record = traces.get(result.trace_id)
    assert record is not None and record.status == "ok"
    assert record.purpose == "chat" and record.model_name == "mock/echo"
    assert record.finish_reason == "stop"
    assert record.messages_ref and record.output_ref
    assert blobs.get_json(record.output_ref)["text"] == "北京晴，21 度。"
    assert blobs.get_json(record.messages_ref)[0]["content"] == "北京天气怎么样？"

    usage = UsageRepo(db).fetch(result.trace_id)
    assert usage.usage.source == str(TokenSource.ENGINE)
    assert usage.usage.confidence == str(Confidence.HIGH)
    assert usage.usage.in_tokens == 280 and usage.usage.out_tokens == 26
    assert usage.usage.decode_tps == pytest.approx(26 / 0.596706, rel=1e-3)
    assert usage.usage.prefill_mode == "cold", "0.589 ms/token 应判为冷（阈值 0.30）"
    assert {a.source for a in usage.alts} >= {"engine", "heuristic"}
    assert usage.part_tokens("tool_defs") == 0


def test_tool_definition_cost_is_attributed(env):
    """DESIGN §6.2 的核心指标：工具定义吃掉多少上下文。"""
    db, *_ = env
    provider = MockProvider(scripts={"mock/echo": MockScript(
        text="", tool_calls=({"name": "get_weather", "arguments": {"city": "北京"}},),
        in_tokens=280, out_tokens=26, done_reason="stop",
    )})
    result = _gateway(env, provider).generate(_req(tools=(WEATHER,)))
    _flush(env)

    usage = UsageRepo(db).fetch(result.trace_id)
    assert usage.part_tokens("tool_defs") > 20, "工具 schema 必须单独成段"
    assert usage.part_tokens("template_ctl") > 0, "残差 = 模板控制符成本"
    total_parts = usage.part_tokens("system") + usage.part_tokens("tool_defs") \
        + usage.part_tokens("msg:0") + usage.part_tokens("template_ctl")
    assert total_parts == 280, f"Σ分段 + 残差必须等于引擎计数，实际 {total_parts}"

    calls = TraceRepo(db).list_tool_calls(result.trace_id)
    assert len(calls) == 1
    assert calls[0].name == "get_weather" and calls[0].args == {"city": "北京"}
    assert calls[0].parse_status == "ok"


def test_eval_context_is_traced_through(env):
    """评测样本必须能反查——这是"分数可下钻"的前提。"""
    db, *_ = env
    provider = MockProvider()
    ctx = TraceContext(purpose=TracePurpose.EVAL, eval_run_id="tool_selection",
                       case_id="c-7", sample_seq=2)
    result = _gateway(env, provider).generate(_req(), context=ctx)
    _flush(env)
    record = TraceRepo(db).get(result.trace_id)
    assert record.purpose == "eval:tool_selection"
    assert record.case_id == "c-7" and record.sample_seq == 2
    assert record.root_id == result.trace_id


# ── 计量降级 ───────────────────────────────────────────────────────
def test_missing_engine_counts_degrades_visibly(env):
    db, *_ = env
    provider = MockProvider(scripts={"mock/echo": MockScript(report_usage=False)})
    result = _gateway(env, provider).generate(_req())
    _flush(env)

    codes = _codes(result)
    assert "NO_ENGINE_COUNT" in codes
    assert "LOW_CONFIDENCE_USAGE" in codes
    usage = UsageRepo(db).fetch(result.trace_id).usage
    assert usage.source == str(TokenSource.HEURISTIC)
    assert usage.confidence == str(Confidence.LOW), "降级必须显式标注，不能假装精确"
    assert usage.in_tokens and usage.in_tokens > 0


def test_fitted_counter_takes_over_when_calibrated(env):
    provider = MockProvider(scripts={"mock/echo": MockScript(report_usage=False)})
    ctx = CounterContext(fitted_ratio=0.62, fitted_n=210)
    result = _gateway(env, provider, counter_ctx=ctx).generate(_req())
    assert result.usage.source is TokenSource.FITTED
    assert result.usage.confidence is Confidence.MEDIUM


def test_drift_between_sources_is_flagged(env):
    """引擎说 280、启发式说 60 ⇒ 口径分裂，必须报警而不是取一个平均。"""
    provider = MockProvider(scripts={"mock/echo": MockScript(in_tokens=280, out_tokens=26)})
    ctx = CounterContext(fitted_ratio=0.05, fitted_n=200)
    result = _gateway(env, provider, counter_ctx=ctx).generate(_req())
    assert "TOKEN_DRIFT" in _codes(result)
    assert result.usage.drift_pct and result.usage.drift_pct > 0.1


# ── 输出形态异常 ───────────────────────────────────────────────────
def test_thinking_only_output_is_flagged(env):
    """P5/P12：预算被推理吃光，正文为空。评测里这不算答错。"""
    provider = MockProvider(scripts={"mock/echo": MockScript(
        text="", thinking="让我想想……" * 10, out_tokens=64, done_reason="length",
    )})
    result = _gateway(env, provider).generate(_req())
    assert "EMPTY_CONTENT_WITH_THINKING" in _codes(result)


def test_empty_output_is_flagged(env):
    provider = MockProvider(scripts={"mock/echo": MockScript(text="", thinking="")})
    result = _gateway(env, provider).generate(_req())
    assert "EMPTY_OUTPUT" in _codes(result)


def test_cold_load_and_warm_prefill_are_separate_signals(env):
    cold = MockScript(load_ns=900_000_000, prompt_eval_ns=384_500_000, in_tokens=644, out_tokens=8)
    warm = MockScript(load_ns=1_000_000, prompt_eval_ns=83_400_000, in_tokens=644, out_tokens=8)
    provider = MockProvider(scripts={"cold": cold, "warm": warm})
    gateway = _gateway(env, provider)
    cold_result = gateway.generate(_req(model="cold"))
    warm_result = gateway.generate(_req(model="warm"))
    assert "COLD_LOAD" in _codes(cold_result)
    assert cold_result.latency["prefill_mode"] == "cold"
    assert warm_result.latency["prefill_mode"] == "warm"
    assert "PREFILL_CACHE_HIT" in _codes(warm_result)
    assert warm_result.latency["prefill_tps"] > cold_result.latency["prefill_tps"] * 4


# ── 工具调用异常 ───────────────────────────────────────────────────
def test_truncated_tool_json_keeps_raw_and_flags(env):
    db, *_ = env
    provider = MockProvider(scripts={"mock/echo": MockScript(
        text="", tool_calls=({"name": "get_weather", "arguments_fragment": '{"city": "北'},),
        done_reason="length", in_tokens=280, out_tokens=64,
    )})
    result = _gateway(env, provider).generate(_req(tools=(WEATHER,)))
    _flush(env)
    assert "TRUNCATED_TOOL_JSON" in _codes(result)
    call = TraceRepo(db).list_tool_calls(result.trace_id)[0]
    assert call.parse_status == "truncated"
    assert call.args_raw == '{"city": "北', "原文必须落库，这是唯一的诊断证据"


def test_orphan_tool_call_when_reason_without_calls(env):
    provider = MockProvider(scripts={"mock/echo": MockScript(text="", done_reason="tool_calls")})
    result = _gateway(env, provider).generate(_req(tools=(WEATHER,)))
    assert "ORPHAN_TOOL_CALL" in _codes(result)


def test_tool_loop_detected(env):
    same = {"name": "get_weather", "arguments": {"city": "北京"}}
    provider = MockProvider(scripts={"mock/echo": MockScript(text="", tool_calls=(same, same, same))})
    result = _gateway(env, provider).generate(_req(tools=(WEATHER,)))
    assert "TOOL_LOOP" in _codes(result)


# ── 失败路径 ───────────────────────────────────────────────────────
def test_provider_error_is_recorded_then_reraised(env):
    db, *_ = env
    boom = ProviderUnreachable("无法连接 Ollama", base_url="http://127.0.0.1:11434")
    provider = MockProvider(scripts={"mock/echo": MockScript(raise_exc=boom)})
    with pytest.raises(ProviderUnreachable):
        _gateway(env, provider).generate(_req())
    _flush(env)
    records = TraceRepo(db).list(limit=10)
    assert len(records) == 1
    assert records[0].status == "error"
    assert "ProviderUnreachable" in records[0].error


def test_error_response_from_engine_is_not_ok(env):
    provider = MockProvider(scripts={"mock/echo": MockScript(error="model not found", text="")})
    result = _gateway(env, provider).generate(_req())
    assert result.ok is False
    assert "PROVIDER_ERROR" in _codes(result)


# ── 隔离性 ─────────────────────────────────────────────────────────
class BrokenVisitor(BaseVisitor):
    name = "broken"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        raise RuntimeError("visitor 炸了")

    def finalize(self, state: TraceState) -> None:
        raise RuntimeError("finalize 也炸了")


def test_broken_visitor_never_breaks_the_request(env):
    db, sink, _, blobs = env
    observer = ObserverEngine(record_sink=sink, visitors=(BrokenVisitor(), *default_visitors()))
    provider = MockProvider(scripts={"mock/echo": MockScript(in_tokens=100, out_tokens=10)})
    gateway = Gateway(provider, observer=observer, blobs=blobs)
    result = gateway.generate(_req())
    _flush(env)

    assert result.generation.text, "请求本身必须成功"
    assert "OBSERVER_ERROR" in _codes(result)
    assert TraceRepo(db).get(result.trace_id) is not None
    assert UsageRepo(db).fetch(result.trace_id).usage.in_tokens == 100, "其余 visitor 必须照常工作"
    assert observer.stats()["observer_errors"]["broken"] >= 2


def test_event_sink_failure_does_not_break_request(env):
    seen: list[EventType] = []

    def flaky(event: TraceEvent) -> None:
        seen.append(event.type)
        if event.type is EventType.USAGE_LOCAL:
            raise RuntimeError("订阅方炸了")

    provider = MockProvider()
    result = _gateway(env, provider, event_sink=flaky).generate(_req())
    assert result.ok and EventType.TRACE_END in seen


def test_state_eviction_is_bounded(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    sink = SqliteRecordSink(db, idle_wait=0.005)
    observer = ObserverEngine(record_sink=sink, max_states=3)
    from onyx.core.event import make_event

    for i in range(6):
        observer.handle(make_event(EventType.TRACE_START, f"T{i}", {
            "kind": "generation", "purpose": "chat", "provider_id": "mock", "model": "m",
        }))
    _flush((db, sink, observer, None))
    assert observer.stats()["pending_states"] <= 3
    assert observer.stats()["evicted"] >= 3
    sink.close()
    db.close()


# ── GPU 采样 ───────────────────────────────────────────────────────
def test_gpu_sample_detects_cpu_offload(env):
    class OffloadingMock(MockProvider):
        def running(self):
            from onyx.core.types import LoadedModel

            return [LoadedModel(name="mock/echo", size=17_700_000_000, size_vram=15_000_000_000,
                                context_length=4096)]

    provider = OffloadingMock(scripts={"mock/echo": MockScript(in_tokens=3900, out_tokens=10)})
    result = _gateway(env, provider, sample_gpu=True).generate(_req())
    codes = _codes(result)
    assert "OFFLOADED_TO_CPU" in codes
    assert "CONTEXT_NEAR_LIMIT" in codes, "3900/4096 = 95%，必须预警"


# ── 注册表 ─────────────────────────────────────────────────────────
def test_unknown_usage_source_is_ignored_not_aliased():
    """回归：source 非法时曾回退成 heuristic，把真正的启发式样本覆盖掉，
    结果采信直接掉到「无来源」。宁可丢弃可疑样本也不许污染别的档位。"""
    from onyx.core.event import make_event
    from onyx.obs.state import TraceState
    from onyx.obs.visitors.token import TokenVisitor

    state = TraceState(trace_id="t1")
    visitor = TokenVisitor()
    visitor.on(make_event(EventType.USAGE_LOCAL, "t1",
                          {"source": "heuristic", "in_tokens": 12, "out_tokens": 7}), state)
    assert state.usage_samples["heuristic"].in_tokens == 12

    visitor.on(make_event(EventType.USAGE_LOCAL, "t1",
                          {"source": "made_up_source", "in_tokens": 999}), state)
    assert "made_up_source" not in state.usage_samples
    assert state.usage_samples["heuristic"].in_tokens == 12, "非法来源不许覆盖已有档位"
    assert len(state.usage_samples) == 1


def test_attribution_event_does_not_masquerade_as_a_usage_sample():
    from onyx.core.event import make_event
    from onyx.obs.state import TraceState
    from onyx.obs.visitors.token import TokenVisitor

    state = TraceState(trace_id="t2")
    visitor = TokenVisitor()
    visitor.on(make_event(EventType.USAGE_LOCAL, "t2", {"source": "heuristic", "in_tokens": 12}), state)
    visitor.on(make_event(EventType.USAGE_ATTRIBUTION, "t2", {
        "count_source": "heuristic",
        "parts": [{"part": "tool_defs", "ord": 0, "tokens": 260, "bytes": 400},
                  {"part": "template_ctl", "ord": 1, "tokens": 20, "bytes": None}],
        "attribution": {"input_segments_tokens": 280, "template_ctl_tokens": 20,
                        "residual_raw": 20, "clamped": False, "has_template": False},
    }), state)
    assert len(state.usage_samples) == 1, "归因不是计数来源，不许进 usage_samples"
    assert sum(p.tokens for p in state.parts if p.part == "tool_defs") == 260
    assert state.extra["attribution"]["count_source"] == "heuristic"
    assert not state.extra.get("attribution_clamped")


def test_registry_builds_builtin_and_mock():
    kinds = available_kinds()
    assert {"ollama", "mock"} <= set(kinds)
    provider = build_provider("mock")
    assert provider.info().reachable
    with pytest.raises(KeyError, match="未知 provider"):
        build_provider("nope")
