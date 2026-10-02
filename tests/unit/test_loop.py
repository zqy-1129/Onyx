"""S12 验收：客户端工具循环的边界形态。

本地小模型上**异常形态才是常态**，所以这里每条边界一个用例：
max_steps 截断 · token 预算 · 墙钟预算 · TOOL_LOOP 熔断 · 幻觉工具名 ·
`finish_reason=tool_calls` 却无 tool_calls · 工具失败后能继续 · 孤儿调用补齐 ·
截断 JSON 进 args_raw。

用 MockProvider 驱动：它走的是与真实适配器**同一条** `consume_chunks` 路径，
所以这里测的管道与真机管道等价（真机数值另由 tests/integration 覆盖）。
"""

from __future__ import annotations

import json

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.errors import ProviderUnreachable
from onyx.core.types import (
    GenerationRequest,
    Message,
    Role,
    TraceContext,
    TracePurpose,
)
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import TraceRepo
from onyx.store.sinks import SqliteRecordSink
from onyx.tools.builtin.defs import CALCULATOR, ECHO
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.executors import MockReplayExecutor
from onyx.tools.loop import CallOutcome, LoopBudget, StopReason, ToolLoop
from onyx.tools.sandbox import PERMISSIVE_POLICY
from onyx.tools.spec import ToolDef

MODEL = "mock/tool"
TOOLS = (ECHO, CALCULATOR)
SPECS = tuple(d.to_spec() for d in TOOLS)


def _call(name: str, **args) -> dict:
    return {"name": name, "arguments": dict(args)}


#: 第 1 步要工具，第 2 步给最终答案
TWO_STEP = [
    MockScript(text="", tool_calls=(_call("echo", text="hi"),), done_reason="tool_calls",
               in_tokens=100, out_tokens=10),
    MockScript(text="已经回显完毕。", done_reason="stop", in_tokens=140, out_tokens=8),
]


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    observer = ObserverEngine(record_sink=sink)
    blobs = FileBlobStore(tmp_path / "blobs")
    yield db, sink, observer, blobs
    sink.close()
    db.close()


def _gateway(env, provider, clock=None) -> Gateway:
    _, _, observer, blobs = env
    clock = clock or FakeClock()
    return Gateway(provider, observer=observer, blobs=blobs, clock=clock)


def _loop(env, scripts, *, budget=None, tools=TOOLS, ctx=None, clock=None,
          executor_factory=None, stop_words=()):
    fake = clock or FakeClock()
    budget = budget or LoopBudget()
    provider = MockProvider(scripts={MODEL: scripts}, models=(MODEL,))
    gateway = _gateway(env, provider, fake)
    loop = ToolLoop(
        gateway, tools, budget=budget,
        ctx=ctx or ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY),
        stop_words=stop_words, executor_factory=executor_factory, clock=fake,
    )
    return loop, provider


def _req(prompt: str = "把 hi 回显一下") -> GenerationRequest:
    return GenerationRequest.of(MODEL, prompt, tools=SPECS,
                                context=TraceContext(purpose=TracePurpose.CHAT))


def _tool_messages(result) -> list[Message]:
    return [m for m in result.messages if m.role is Role.TOOL]


# ── 正常路径 ──────────────────────────────────────────────────────
def test_two_step_loop_reaches_a_final_answer(env):
    loop, provider = _loop(env, TWO_STEP)
    result = loop.run(_req())

    assert result.stop_reason is StopReason.FINAL
    assert result.text == "已经回显完毕。"
    assert len(result.steps) == 2
    assert result.tool_calls_total == 1 and result.executed == 1
    assert len(provider.calls) == 2

    step1, step2 = result.steps
    assert step1.wanted_tool_call and not step2.wanted_tool_call
    assert step1.calls[0].verdict == "executed"
    assert step1.calls[0].result.ok
    assert step1.calls[0].result.output == {"echo": "hi", "times": 1, "chars": 2}


def test_token_totals_are_the_sum_across_steps(env):
    """总和是各步之和——那才是真实付出的成本，不是最后一步的 prompt 长度。"""
    loop, _ = _loop(env, TWO_STEP)
    result = loop.run(_req())
    assert result.in_tokens == 240 and result.out_tokens == 18
    assert (result.steps[0].in_tokens, result.steps[1].in_tokens) == (100, 140)


def test_mixed_fidelity_across_steps_is_reported_not_hidden(env):
    """第 2 步引擎没报计数，保真阶梯退回 heuristic 档。

    总和仍然给（否则看板什么都显示不了），但**必须同时给出出处与置信度**：
    把 engine 档和 heuristic 档加成一个数却不说明其中一半是估的，比不报更糟。
    """
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="hi"),), done_reason="tool_calls",
                   in_tokens=100, out_tokens=10),
        MockScript(text="完成", report_usage=False),
    ]
    loop, _ = _loop(env, scripts)
    result = loop.run(_req())
    assert result.stop_reason is StopReason.FINAL

    assert result.steps[0].usage_source == "engine"
    assert result.steps[0].usage_confidence == "high"
    assert result.steps[1].usage_source != "engine", "引擎没报计数就不该记成 engine 档"

    assert result.usage_source == "mixed"
    assert result.usage_confidence == "low", "总和的置信度不可能高于最弱的那一项"
    assert result.in_tokens is not None, "有数字就该给，但必须带上 mixed/low 的标注"


def test_every_step_is_linked_to_one_root(env):
    """多步循环在库里必须能按 root_id 聚合出来，否则看板上是 N 条互不相干的 trace。"""
    db, sink, _, _ = env
    loop, _ = _loop(env, TWO_STEP)
    result = loop.run(_req())
    sink.flush(5.0)

    repo = TraceRepo(db)
    assert len(result.steps) == 2
    for step in result.steps:
        record = repo.get(step.trace_id)
        assert record is not None, f"步骤 {step.step} 的 trace 没落库"
        # root_id 是分组键，指向一条**不存在**的 trace；parent_id 是外键，必须留空，
        # 否则整条记录会因为 FOREIGN KEY constraint failed 写不进去
        assert record.root_id == result.root_trace_id
        assert record.parent_id is None
    assert result.steps[0].trace_id != result.steps[1].trace_id


def test_tool_execution_lands_in_the_step_that_requested_it(env):
    """TOOL_EXEC_* 必须落在发起调用的那一步的 trace 里。

    落在别处的话，看板上"这一步花了多久"就不含工具耗时，
    而工具往往才是那一步最慢的部分。
    """
    db, sink, _, _ = env
    loop, _ = _loop(env, TWO_STEP)
    result = loop.run(_req())
    sink.flush(5.0)

    calls = TraceRepo(db).list_tool_calls(result.steps[0].trace_id)
    assert len(calls) == 1
    record = calls[0]
    assert record.name == "echo" and record.step == 1
    assert record.result_status == "ok"
    assert record.executed_by == "client"
    assert record.latency_ms is not None
    assert record.result_ref.startswith("sha256:"), "工具原始结果必须落 blob 并留引用"
    assert record.tool_def_hash == ECHO.hash
    # 第 2 步没有工具调用
    assert TraceRepo(db).list_tool_calls(result.steps[1].trace_id) == []


# ── 上下文不变式 ──────────────────────────────────────────────────
def assert_tool_message_invariant(result) -> None:
    """带 N 个 tool_calls 的 assistant 消息，后面必须紧跟恰好 N 条 role=tool 消息。

    少一条，之后每次请求的上下文都永久错位，而引擎通常不报错——只是开始答非所问。
    """
    messages = list(result.messages)
    for index, message in enumerate(messages):
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            continue
        following = []
        for later in messages[index + 1:]:
            if later.role is Role.TOOL:
                following.append(later)
            else:
                break
        assert len(following) == len(message.tool_calls), (
            f"第 {index} 条 assistant 有 {len(message.tool_calls)} 个 tool_calls，"
            f"后面只跟了 {len(following)} 条 tool 消息"
        )
        # 悬空的 tool 消息（没有对应调用）同样是错位
        expected_names = [c.name for c in message.tool_calls]
        assert [m.name for m in following] == expected_names


def test_invariant_holds_on_the_happy_path(env):
    loop, _ = _loop(env, TWO_STEP)
    assert_tool_message_invariant(loop.run(_req()))


def test_invariant_holds_when_a_tool_is_unknown(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("get_weather", city="北京"),),
                   done_reason="tool_calls"),
        MockScript(text="查不到天气。", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts)
    result = loop.run(_req())
    assert_tool_message_invariant(result)

    outcome = result.steps[0].calls[0]
    assert outcome.verdict == "unknown_tool" and outcome.result is None
    assert "UNKNOWN_TOOL" in result.codes()
    # 补进去的占位消息要让模型看得见"这个工具不存在"，否则它会一直重试
    tool_message = _tool_messages(result)[0]
    assert tool_message.name == "get_weather"
    assert json.loads(tool_message.content)["error_kind"] == "unknown_tool"


def test_invariant_holds_when_the_loop_breaks_mid_turn(env):
    """熔断发生在**一轮里的第二个调用**上：剩下的调用仍要补占位。"""
    scripts = [
        MockScript(text="", tool_calls=(
            _call("echo", text="a"), _call("echo", text="a"), _call("echo", text="a"),
        ), done_reason="tool_calls"),
        MockScript(text="收尾", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts, budget=LoopBudget(repeat_threshold=1))
    result = loop.run(_req())

    assert_tool_message_invariant(result)
    assert result.stop_reason is StopReason.TOOL_LOOP
    verdicts = [c.verdict for c in result.steps[0].calls]
    assert verdicts == ["executed", "loop_break", "loop_break"]
    assert result.steps[0].calls[2].backfilled is True
    assert "TOOL_LOOP" in result.codes()
    # 熔断后不再请求模型：第 2 个剧本没被消费
    assert len(result.steps) == 1


# ── 预算 ──────────────────────────────────────────────────────────
ALWAYS_CALLS = [
    MockScript(text="", tool_calls=(_call("echo", text="第 1 次"),), done_reason="tool_calls",
               in_tokens=100, out_tokens=10),
    MockScript(text="", tool_calls=(_call("echo", text="第 2 次"),), done_reason="tool_calls",
               in_tokens=120, out_tokens=10),
    MockScript(text="", tool_calls=(_call("echo", text="第 3 次"),), done_reason="tool_calls",
               in_tokens=140, out_tokens=10),
    MockScript(text="", tool_calls=(_call("echo", text="第 4 次"),), done_reason="tool_calls",
               in_tokens=160, out_tokens=10),
]


def test_max_steps_truncates_and_reports_budget(env):
    loop, _ = _loop(env, ALWAYS_CALLS, budget=LoopBudget(max_steps=3))
    result = loop.run(_req())
    assert result.stop_reason is StopReason.MAX_STEPS
    assert len(result.steps) == 3
    assert "BUDGET_EXCEEDED" in result.codes()
    detail = next(d for c, _, d in result.anomalies if c == "BUDGET_EXCEEDED")
    assert detail["reason"] == "max_steps" and detail["max_steps"] == 3
    assert_tool_message_invariant(result)


def test_token_budget_stops_the_loop(env):
    loop, _ = _loop(env, ALWAYS_CALLS, budget=LoopBudget(max_steps=9, max_total_tokens=250))
    result = loop.run(_req())
    # 100+10 → 120+10 累计 240 < 250，第 3 步前累计已达 240，第 3 步后 390
    assert result.stop_reason is StopReason.TOKEN_BUDGET
    assert len(result.steps) == 3
    detail = next(d for c, _, d in result.anomalies if c == "BUDGET_EXCEEDED")
    assert detail["reason"] == "tokens" and detail["limit"] == 250


def test_wall_budget_stops_the_loop(env):
    clock = FakeClock(step_ns=40_000_000)  # 每次读钟前进 40ms
    loop, _ = _loop(env, ALWAYS_CALLS, budget=LoopBudget(max_steps=9, max_wall_ms=100), clock=clock)
    result = loop.run(_req())
    assert result.stop_reason is StopReason.WALL_BUDGET
    detail = next(d for c, _, d in result.anomalies if c == "BUDGET_EXCEEDED")
    assert detail["reason"] == "wall_ms" and detail["limit_ms"] == 100
    assert detail["elapsed_ms"] > 100


def test_budget_anomalies_are_loop_level_not_step_level(env):
    """预算在两步之间耗尽，那时已经没有打开的 trace 可落——必须显式区分两种来源。"""
    loop, _ = _loop(env, ALWAYS_CALLS, budget=LoopBudget(max_steps=2))
    result = loop.run(_req())
    assert result.anomalies, "循环级异常不能丢"
    for step in result.steps:
        assert not any(code == "BUDGET_EXCEEDED" for code, _, _ in step.anomalies)
    # all_anomalies 是看板与评测该读的唯一口径
    assert "BUDGET_EXCEEDED" in {code for code, _, _ in result.all_anomalies()}


# ── 熔断 ──────────────────────────────────────────────────────────
def test_repeated_identical_call_trips_the_breaker(env):
    scripts = [MockScript(text="", tool_calls=(_call("echo", text="同一个"),),
                          done_reason="tool_calls")] * 4
    loop, _ = _loop(env, scripts, budget=LoopBudget(max_steps=8, repeat_threshold=2))
    result = loop.run(_req())
    assert result.stop_reason is StopReason.TOOL_LOOP
    assert len(result.steps) == 3, "第 3 次同一调用触发熔断"
    assert "TOOL_LOOP" in result.codes()
    anomaly = next(d for s in result.steps for c, _, d in s.anomalies if c == "TOOL_LOOP")
    assert anomaly["tool"] == "echo" and anomaly["count"] == 3


def test_same_tool_with_different_args_is_not_a_loop(env):
    """熔断看的是 (工具, 参数) 指纹：只按工具名熔断会误杀正常的多步任务。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("calculator", expr="1+1"),), done_reason="tool_calls"),
        MockScript(text="", tool_calls=(_call("calculator", expr="2+2"),), done_reason="tool_calls"),
        MockScript(text="算完了", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts, budget=LoopBudget(repeat_threshold=1))
    result = loop.run(_req())
    assert result.stop_reason is StopReason.FINAL
    assert "TOOL_LOOP" not in result.codes()
    assert result.executed == 2


def test_truncated_json_is_not_confused_with_a_repeat(env):
    """两次都截断在**同一位置**才算重复——原文参与指纹，不是拿 None 当相等。"""
    fragment = '{"text": "很长的内容被 max_tok'
    scripts = [
        MockScript(text="", tool_calls=({"name": "echo", "arguments_fragment": fragment},),
                   done_reason="tool_calls"),
        MockScript(text="", tool_calls=({"name": "echo", "arguments_fragment": fragment},),
                   done_reason="tool_calls"),
        MockScript(text="放弃", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts, budget=LoopBudget(max_steps=5, repeat_threshold=1))
    result = loop.run(_req())
    first = result.steps[0].calls[0]
    assert first.verdict == "parse_error" and first.args is None
    assert first.args_raw.startswith('{"text"'), "截断的原文必须保留，否则无从诊断"
    assert result.steps[1].calls[0].verdict == "loop_break"
    assert_tool_message_invariant(result)


# ── 孤儿调用 ──────────────────────────────────────────────────────
def test_tool_calls_finish_reason_with_no_calls_is_an_orphan(env):
    """本地模型高频形态：done_reason=tool_calls 但 tool_calls 为空。

    不能当成"最终回答"（正文是空的），也不能无限等下去。
    """
    scripts = [MockScript(text="", done_reason="tool_calls"), MockScript(text="不会走到")]
    loop, provider = _loop(env, scripts)
    result = loop.run(_req())

    assert result.stop_reason is StopReason.ORPHAN_TOOL_CALL
    assert "ORPHAN_TOOL_CALL" in result.codes()
    assert result.tool_calls_total == 0
    assert len(provider.calls) == 1, "孤儿调用不该继续请求模型"
    assert not _tool_messages(result), "没有调用就不该凭空造 tool 消息"
    # 异常落在**那一步的 trace** 里（钩子跑在 TRACE_END 之前）
    assert any(c == "ORPHAN_TOOL_CALL" for c, _, _ in result.steps[0].anomalies)


# ── 工具失败后能继续 ──────────────────────────────────────────────
def test_loop_continues_after_a_tool_failure(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("calculator", expr="1/0"),), done_reason="tool_calls"),
        MockScript(text="除零了，换个算式。", done_reason="stop"),
    ]
    loop, provider = _loop(env, scripts)
    result = loop.run(_req())

    assert result.stop_reason is StopReason.FINAL and len(provider.calls) == 2
    outcome = result.steps[0].calls[0]
    assert outcome.verdict == "executed" and not outcome.result.ok
    assert outcome.result.error_kind == "arg_error"
    # 模型必须**看到**失败原因，否则它没法改口
    payload = json.loads(_tool_messages(result)[0].content)
    assert payload["ok"] is False and "除零" in payload["error"]
    second_request = provider.calls[1]
    assert any(m.role is Role.TOOL and "除零" in m.content for m in second_request.messages)


def test_arg_error_is_reported_as_a_tool_side_status_not_a_crash(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("echo"),), done_reason="tool_calls"),
        MockScript(text="好", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts)
    result = loop.run(_req())
    outcome = result.steps[0].calls[0]
    assert outcome.verdict == "executed"
    assert outcome.result.error_kind == "arg_error"
    # visitor 只对 error/timeout 报 TOOL_ERROR；arg_error 是模型的问题，不该算工具坏
    assert "TOOL_ERROR" not in result.codes()


def test_tool_error_status_reaches_the_visitor(env):
    """实现崩了必须被记成 TOOL_ERROR，这样"工具坏了"与"模型用错了"才分得开。"""
    crashy = ToolDef(
        name="crashy", description="实现自己会崩的工具，用来验证 TOOL_ERROR 归因",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        impl_ref="onyx.tools.builtin.echo:boom",
    )
    scripts = [
        MockScript(text="", tool_calls=({"name": "crashy", "arguments": {}},),
                   done_reason="tool_calls"),
        MockScript(text="失败", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts, tools=(crashy,))
    result = loop.run(_req())
    assert result.steps[0].calls[0].result.error_kind == "error"
    assert "TOOL_ERROR" in result.codes()


# ── mock 策略：零真实执行 ─────────────────────────────────────────
def test_fixture_policy_executes_nothing_real(env, monkeypatch):
    import importlib

    calls: list[dict] = []

    def canary(text: str = "", times: int = 1):
        calls.append({"text": text})
        return {"echo": text}

    monkeypatch.setattr(importlib.import_module("onyx.tools.builtin.echo"), "echo", canary)

    ctx = ToolCtx(mock_policy=MockPolicy.FIXTURE, policy=PERMISSIVE_POLICY,
                  fixtures={"echo": {"echo": "stub", "times": 1, "chars": 4}})
    loop, _ = _loop(env, TWO_STEP, ctx=ctx)
    result = loop.run(_req())

    assert calls == [], "mock 策略下真实实现被调用了"
    outcome = result.steps[0].calls[0]
    assert outcome.verdict == "executed" and outcome.result.mocked
    assert outcome.result.output == {"echo": "stub", "times": 1, "chars": 4}


def test_mock_executor_can_be_plugged_in_as_the_factory(env):
    """执行器可替换在循环这一层同样成立：换工厂不用改循环代码。"""
    loop, _ = _loop(env, TWO_STEP, executor_factory=lambda d: MockReplayExecutor(d, {"echo": {"stub": 1}}))
    result = loop.run(_req())
    assert result.steps[0].calls[0].result.mocked
    assert result.steps[0].calls[0].result.output == {"stub": 1}


def test_python_fn_executor_is_the_default(env):
    loop, _ = _loop(env, TWO_STEP)
    result = loop.run(_req())
    assert result.steps[0].calls[0].result.mocked is False


# ── 停止词与请求构造 ──────────────────────────────────────────────
def test_stop_words_are_merged_into_every_request(env):
    """文本模板类模型需要 </tool_response> 之类的停止词；由模型的 tool_format 探针结果驱动，
    不在循环里硬编码——原生 tool_calls 头的模型加了反而会截断正文。"""
    loop, provider = _loop(env, TWO_STEP, stop_words=("</tool_response>", "<|end|>"))
    loop.run(_req())
    assert len(provider.calls) == 2
    for request in provider.calls:
        assert request.params.stop == ("</tool_response>", "<|end|>")
    # 原有参数与工具定义不能被覆盖掉
    assert provider.calls[0].tool_names == ("echo", "calculator")


def test_loop_does_not_mutate_the_incoming_request(env):
    loop, provider = _loop(env, TWO_STEP)
    request = _req()
    loop.run(request)
    assert len(request.messages) == 1, "调用方的请求对象必须保持不可变"
    assert len(provider.calls[0].messages) == 1
    assert len(provider.calls[1].messages) == 3  # user + assistant(tool_calls) + tool


def test_purpose_and_context_are_forwarded(env):
    db, sink, _, _ = env
    loop, _ = _loop(env, TWO_STEP)
    result = loop.run(
        _req(), purpose=TracePurpose.EVAL,
        context=TraceContext(purpose=TracePurpose.CHAT, eval_run_id="run-1", case_id="case-7"),
    )
    sink.flush(5.0)
    record = TraceRepo(db).get(result.steps[0].trace_id)
    # purpose_label 对评测会带上 run id，这样按 purpose 聚合时不同评测不会混在一起
    assert record.purpose == "eval:run-1"
    assert record.eval_run_id == "run-1" and record.case_id == "case-7"


# ── 引擎故障 ──────────────────────────────────────────────────────
def test_provider_error_stops_the_loop_with_a_reason(env):
    scripts = [MockScript(raise_exc=ProviderUnreachable("连不上 Ollama", base_url="mock://"))]
    loop, _ = _loop(env, scripts)
    result = loop.run(_req())
    assert result.stop_reason is StopReason.ERROR
    assert "ProviderUnreachable" in result.error
    assert "PROVIDER_ERROR" in {c for c, _, _ in result.anomalies}
    assert result.steps == ()


def test_loop_survives_a_tool_that_times_out(env):
    slow = ToolDef(
        name="slow", description="故意慢的工具，验证超时不会把循环一起带走",
        parameters={"type": "object", "properties": {"text": {"type": "string", "description": "文本"}},
                    "required": ["text"]},
        impl_ref="onyx.tools.builtin.echo:slow_echo",
    )
    scripts = [
        MockScript(text="", tool_calls=(_call("slow", text="x", delay_ms=300),),
                   done_reason="tool_calls"),
        MockScript(text="超时了", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts, tools=(slow,),
                    ctx=ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY,
                                deadline_ms=5))
    result = loop.run(_req())
    outcome = result.steps[0].calls[0]
    assert outcome.result.error_kind == "timeout"
    assert result.stop_reason is StopReason.FINAL
    assert "TOOL_ERROR" in result.codes()
    assert_tool_message_invariant(result)


def test_call_outcome_ordinal_matches_the_visitor_step(env):
    """ordinal 必须等于 GENERATION_END 里的 step（= ToolCall.index + 1），
    否则 visitor 配不上对，每次执行都会被记成 ORPHAN_TOOL_CALL。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="a"), _call("calculator", expr="1+1")),
                   done_reason="tool_calls"),
        MockScript(text="完成", done_reason="stop"),
    ]
    loop, _ = _loop(env, scripts)
    result = loop.run(_req())
    calls = result.steps[0].calls
    assert [c.ordinal for c in calls] == [1, 2]
    assert [c.name for c in calls] == ["echo", "calculator"]
    assert all(c.verdict == "executed" for c in calls)
    assert "ORPHAN_TOOL_CALL" not in result.codes()


def test_parallel_calls_in_one_turn_are_all_executed(env):
    loop, _ = _loop(env, [
        MockScript(text="", tool_calls=(_call("echo", text="a"), _call("echo", text="b")),
                   done_reason="tool_calls"),
        MockScript(text="两次都完成了", done_reason="stop"),
    ])
    result = loop.run(_req())
    assert result.executed == 2
    assert [json.loads(m.content)["output"]["echo"] for m in _tool_messages(result)] == ["a", "b"]
    assert_tool_message_invariant(result)


def test_no_tools_registered_still_produces_a_final_answer(env):
    """没有工具时循环就是一次普通生成，不该报错也不该空转。"""
    loop, provider = _loop(env, [MockScript(text="直接回答", done_reason="stop")], tools=())
    result = loop.run(GenerationRequest.of(MODEL, "你好"))
    assert result.stop_reason is StopReason.FINAL
    assert result.text == "直接回答"
    assert len(provider.calls) == 1
    assert loop.tool_names == ()


def test_call_outcome_to_tool_message_carries_the_name_for_ollama(env):
    """Ollama 用 tool_name 关联工具结果；缺失会让模型对不上号（native.py 里明确处理）。"""
    outcome = CallOutcome(name="echo", call_id="c1", ordinal=1, args={"text": "a"},
                          args_raw="", parse_status="ok", verdict="executed", result=None)
    message = outcome.to_tool_message()
    assert message.role is Role.TOOL and message.name == "echo"
    assert message.tool_call_id == "c1"
    assert json.loads(message.content)["error_kind"] == "executed"
