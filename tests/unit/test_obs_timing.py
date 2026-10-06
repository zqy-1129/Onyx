"""S37：TTFT 终于有写入方。

`FIRST_TOKEN` 事件从项目第一天起就在发（`core/event.py` 的载荷契约里写着），
provider 也算得出首字时刻，而观测层没有任何 visitor 接它 ⇒ `usage.ttft_ms` 与
`trace.first_token_at` 恒为 NULL，看板那一栏永远是「—」，还被文档解释成"通道不给分段时序"。

这里三段都测：事件→状态、状态→落库、以及**摘掉注册就会红**（否则"接上了"只是一句声明）。
"""

from __future__ import annotations

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.event import EventType, make_event
from onyx.core.types import GenerationRequest, TracePurpose
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.obs.state import TraceState
from onyx.obs.visitors import default_visitors
from onyx.obs.visitors.timing import TimingVisitor
from onyx.store.db import Database
from onyx.store.repos import TraceRepo, UsageRepo
from onyx.store.sinks import SqliteRecordSink


def _event(payload: dict, *, wall: str = "2026-10-06T00:00:00.140000+00:00"):
    clock = FakeClock(wall=wall)
    # strict=False：这一组测试专门要喂"缺键 / 空值 / 类型不对"的载荷。
    # 契约层要求 FIRST_TOKEN 必带 `ttft_ms`，但 visitor 不能假设上游一定守约——
    # 上游哪天漏了键，正确的表现是这一格留空，而不是观测层抛异常拖垮请求。
    return make_event(EventType.FIRST_TOKEN, "trace-1", payload, clock=clock, strict=False)


def _state() -> TraceState:
    return TraceState(trace_id="trace-1", purpose="chat", started_at="2026-10-06T00:00:00+00:00")


# ── 事件 → 状态 ───────────────────────────────────────────────────
def test_measured_first_token_sets_both_the_duration_and_the_moment():
    state = _state()
    TimingVisitor().on(_event({"ttft_ms": 218.4}), state)
    assert state.ttft_ms == pytest.approx(218.4)
    assert state.first_token_at == "2026-10-06T00:00:00.140000+00:00"
    assert state.extra["ttft_source"] == "measured"


def test_proxy_value_is_not_stored_as_a_measurement():
    """非流式请求的 TTFT 是 `prompt_eval_duration` 代理值——那种请求没有"首字"这个量。

    存进去会让人把 prefill 时长读成交付延迟；这里留空并记下出处，
    于是「—」是"知道没有"，不是"坏掉了"。
    """
    state = _state()
    TimingVisitor().on(_event({"ttft_ms": 90.0, "proxy": "prompt_eval_duration"}), state)
    assert state.ttft_ms is None and state.first_token_at is None
    assert state.extra["ttft_source"] == "proxy:prompt_eval_duration"


@pytest.mark.parametrize("payload", [
    {},                              # 没有这个键
    {"ttft_ms": None},               # 显式空
    {"ttft_ms": -3},                 # 不可能是负的
    {"ttft_ms": "很快"},              # 类型不对也不能换算
    {"ttft_ms": True},               # bool 是 int 的子类——不能因此写成 1.0ms
])
def test_a_missing_or_implausible_value_stays_missing(payload):
    """把"看不清"存成一个数，比留空更坏：它会被平均、被排名、被拿去做回归对比。"""
    state = _state()
    TimingVisitor().on(_event(payload), state)
    assert state.ttft_ms is None
    assert state.extra["ttft_source"] == "absent"


def test_only_the_first_token_counts():
    state = _state()
    TimingVisitor().on(_event({"ttft_ms": 120.0}, wall="2026-10-06T00:00:00.120000+00:00"), state)
    TimingVisitor().on(_event({"ttft_ms": 900.0}, wall="2026-10-06T00:00:09.000000+00:00"), state)
    assert state.ttft_ms == pytest.approx(120.0), "重放或后续首包不得覆盖第一次"
    assert state.first_token_at.endswith(".120000+00:00")


def test_other_events_are_ignored():
    """`seq`/`text` 是 TEXT_DELTA 的必填键；这里带上它们，只为了测"这个 visitor 不管别的事件"。"""
    state = _state()
    TimingVisitor().on(make_event(EventType.TEXT_DELTA, "trace-1",
                                  {"seq": 1, "text": "晴", "ttft_ms": 5}, clock=FakeClock()), state)
    assert state.ttft_ms is None and "ttft_source" not in state.extra


# ── 状态 → 落库（经完整管道）──────────────────────────────────────
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


def test_streaming_request_lands_ttft_in_the_database(env):
    """这条是整步的验收：经 gateway → 观测 → sink → 库，延迟列必须有数。"""
    gateway = _gateway(env, MockProvider(scripts={"北京天气怎么样？": MockScript(text="晴")} ))
    result = gateway.generate(GenerationRequest.of("mock/echo", "北京天气怎么样？", stream=True),
                              purpose=TracePurpose.CHAT)
    env[1].flush(5.0)

    trace = TraceRepo(env[0]).get(result.trace_id)
    usage = UsageRepo(env[0]).fetch(result.trace_id).usage
    assert trace is not None and usage is not None
    assert usage.ttft_ms is not None and usage.ttft_ms >= 0, "TTFT 又没人接了"
    assert trace.first_token_at is not None, "时刻与时长是同一个事件的两个表达，缺一就是没接好"
    assert trace.first_token_at >= trace.started_at
    assert usage.extra.get("ttft_source") == "measured"
    # latency 视图与落库值必须同源（两处各算就会分叉）
    assert result.latency["ttft_ms"] == pytest.approx(usage.ttft_ms)


def test_non_streaming_request_keeps_ttft_empty_with_a_reason(env):
    gateway = _gateway(env, MockProvider(scripts={"北京天气怎么样？": MockScript(text="晴")}))
    result = gateway.generate(GenerationRequest.of("mock/echo", "北京天气怎么样？", stream=False),
                              purpose=TracePurpose.CHAT)
    env[1].flush(5.0)
    usage = UsageRepo(env[0]).fetch(result.trace_id).usage
    trace = TraceRepo(env[0]).get(result.trace_id)
    assert usage.ttft_ms is None and trace.first_token_at is None
    assert usage.extra["ttft_source"].startswith("proxy:"), "空着要能说清为什么空"


def test_removing_the_registration_breaks_the_pipeline(env):
    """注入缺陷自检：摘掉 `timing` 的注册，落库的 TTFT 必须变回空。

    这条测试守的是"接上了"这个事实本身——没有它，"我们消费了 FIRST_TOKEN"
    可以和三年前一样继续是一句没有写入方的空话。
    """
    assert "timing" in {v.name for v in default_visitors()}, "注册点被摘掉了（先修再测）"

    stripped = tuple(v for v in default_visitors() if v.name != "timing")
    db, sink, _, blobs = env
    gateway = Gateway(MockProvider(scripts={"北京天气怎么样？": MockScript(text="晴")}),
                      observer=ObserverEngine(record_sink=sink, visitors=stripped), blobs=blobs)
    result = gateway.generate(GenerationRequest.of("mock/echo", "北京天气怎么样？", stream=True),
                              purpose=TracePurpose.CHAT)
    sink.flush(5.0)
    usage = UsageRepo(db).fetch(result.trace_id).usage
    trace = TraceRepo(db).get(result.trace_id)
    assert usage.ttft_ms is None and trace.first_token_at is None, \
        "摘掉注册却仍有数 ⇒ 说明还有第二条写入路径，那这条测试的存在意义就要重新审"
