"""S12 验收：模型侧 fire-and-verify。

DoD 的核心是"能明确区分三种失败"：NO_CALL / WRONG_TOOL / BAD_ARGS。
但只区分这三种还不够——TOOL_FAILED 必须从"模型不会用工具"里分出来，
否则工具坏了会被记成模型能力差，于是你去调提示词，方向完全错。
"""

from __future__ import annotations

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.errors import ProviderUnreachable
from onyx.core.types import GenerationRequest, TraceContext, TracePurpose
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.sinks import SqliteRecordSink
from onyx.tools.builtin.defs import CALCULATOR, ECHO
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.loop import LoopBudget, ToolLoop
from onyx.tools.sandbox import PERMISSIVE_POLICY
from onyx.tools.spec import ToolDef
from onyx.tools.verify import (
    Expectation,
    Verdict,
    VerifyResult,
    summarize,
    verify_case,
    verify_many,
)

MODEL = "mock/tool"
TOOLS = (ECHO, CALCULATOR)
SPECS = tuple(d.to_spec() for d in TOOLS)

CRASHY = ToolDef(
    name="crashy",
    description="实现自己会崩的工具，用来验证 TOOL_FAILED 与 BAD_ARGS 分得开",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    impl_ref="onyx.tools.builtin.echo:boom",
)


def _call(name: str, **args) -> dict:
    return {"name": name, "arguments": dict(args)}


def _req(instruction: str) -> GenerationRequest:
    return GenerationRequest.of(
        MODEL, instruction, tools=SPECS,
        context=TraceContext(purpose=TracePurpose.EVAL, eval_run_id="run-1"),
    )


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    observer = ObserverEngine(record_sink=sink)
    yield db, sink, observer, FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _factory(env, scripts, *, tools=TOOLS, budget=None, ctx=None):
    """每条用例都要一个全新的世界。

    不只是 loop：MockProvider 的**剧本游标**也是状态，共用一个 provider 的话，
    第二条用例会从第一条剩下的剧本继续，于是拿到 NO_CALL——看起来像模型的错，
    其实是测试装置没隔离。真实评测里换的是 loop，模型是同一个；
    但 mock 的剧本队列必须跟着 loop 一起重置，否则测的不是循环。
    """
    _, _, observer, blobs = env

    def build() -> ToolLoop:
        clock = FakeClock()
        provider = MockProvider(scripts={MODEL: scripts}, models=(MODEL,))
        gateway = Gateway(provider, observer=observer, blobs=blobs, clock=clock)
        return ToolLoop(
            gateway, tools, budget=budget or LoopBudget(),
            ctx=ctx or ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY),
            clock=clock,
        )

    return build


FINAL = MockScript(text="已完成。", done_reason="stop")


# ── 六种判定互不相同 ──────────────────────────────────────────────
def test_pass_when_the_tool_and_args_match(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="hi", times=1),),
                   done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi")
    )
    assert result.verdict is Verdict.PASS
    assert result.ok
    assert result.actual_args == {"text": "hi", "times": 1}
    assert result.mocked is False
    assert result.steps == 2 and result.stop_reason == "final"


def test_extra_arguments_pass_by_default_but_fail_under_exact(env):
    """模型多给一个字段通常比少给一个更有用，所以默认只要求期望的字段都在且值对。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="hi", times=3),),
                   done_reason="tool_calls"),
        FINAL,
    ]
    expectation = Expectation("回显 hi", "echo", {"text": "hi"})
    loose = verify_case(_factory(env, scripts)(), expectation, _req("回显 hi"))
    strict = verify_case(_factory(env, scripts)(), Expectation("回显 hi", "echo", {"text": "hi"},
                                                              exact=True), _req("回显 hi"))
    assert loose.verdict is Verdict.PASS
    assert strict.verdict is Verdict.BAD_ARGS
    assert strict.args_diff["subset_ok"] is True, "严格档失败时仍应说明子集是匹配的"
    assert "exact=True" in strict.detail


def test_no_call_when_the_model_answers_directly(env):
    scripts = [MockScript(text="北京今天晴，21 度。", done_reason="stop")]
    result = verify_case(
        _factory(env, scripts)(),
        Expectation("北京天气", "echo", {"text": "北京"}), _req("北京天气怎么样"),
    )
    assert result.verdict is Verdict.NO_CALL
    assert result.called == ()
    assert "没有发起任何工具调用" in result.detail


def test_orphan_tool_call_is_still_no_call_but_says_why(env):
    """"声称要调工具却没给出调用"与"根本没打算调"修法不同，detail 必须区分。"""
    scripts = [MockScript(text="", done_reason="tool_calls")]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显", "echo", {"text": "x"}), _req("回显 x")
    )
    assert result.verdict is Verdict.NO_CALL
    assert "ORPHAN_TOOL_CALL" in result.codes
    assert "finish_reason=tool_calls" in result.detail


def test_wrong_tool_is_distinguished_from_bad_args(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("calculator", expr="1+1"),), done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi")
    )
    assert result.verdict is Verdict.WRONG_TOOL
    assert result.called == ("calculator",)
    assert "期望 echo" in result.detail and "calculator" in result.detail


def test_bad_args_reports_which_fields_are_wrong(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", times=2),), done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi")
    )
    # echo 的 text 有默认值，所以缺字段不会让工具报错——但用例判定的就是"参数对不对"
    assert result.verdict is Verdict.BAD_ARGS
    assert result.args_diff["missing"] == ["text"]
    assert "缺字段" in result.detail


def test_bad_args_reports_wrong_values(env):
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="bye"),), done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi")
    )
    assert result.verdict is Verdict.BAD_ARGS
    assert result.args_diff["mismatched"] == {"text": {"expected": "hi", "actual": "bye"}}
    assert "值不对" in result.detail


def test_unparsable_args_are_bad_args_with_the_raw_text_kept(env):
    """截断的 JSON 与"填错字段"修法不同：前者加 max_tokens，后者改参数描述。"""
    scripts = [
        MockScript(text="", tool_calls=({"name": "echo", "arguments_fragment": '{"text": "很长'},),
                   done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显", "echo", {"text": "很长"}), _req("回显")
    )
    assert result.verdict is Verdict.BAD_ARGS
    assert result.actual_args is None
    assert result.args_diff["args_raw"].startswith('{"text"')
    assert "parse_status" in result.args_diff


def test_tool_failed_is_not_blamed_on_the_model(env):
    """调用完全正确，是实现自己崩了。这一条是 verify 存在的理由。"""
    scripts = [
        MockScript(text="", tool_calls=({"name": "crashy", "arguments": {}},),
                   done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts, tools=(CRASHY,))(),
        Expectation("触发崩溃", "crashy", {}), _req("触发崩溃"),
    )
    assert result.verdict is Verdict.TOOL_FAILED
    assert "onyx tools contract" in result.detail
    assert "RuntimeError" in result.detail


def test_arg_error_is_blamed_on_the_model_not_the_tool(env):
    """同样是执行失败，arg_error 归模型、error 归工具——两者修法相反。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("calculator", expr="__import__('os')"),),
                   done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("算一下", "calculator", {"expr": "1+1"}), _req("算一下")
    )
    assert result.verdict is Verdict.BAD_ARGS
    assert result.args_diff["mismatched"]["expr"]["actual"] == "__import__('os')"


def test_loop_broken_outranks_a_correct_first_call(env):
    """第一次调用参数是对的，但模型随后开始打转 —— 这一轮没有收敛，不能判 PASS。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="同一个"),), done_reason="tool_calls"),
    ]
    result = verify_case(
        _factory(env, scripts, budget=LoopBudget(max_steps=8, repeat_threshold=1))(),
        Expectation("回显", "echo", {"text": "同一个"}), _req("回显"),
    )
    assert result.verdict is Verdict.LOOP_BROKEN
    assert "TOOL_LOOP" in result.codes
    assert "熔断" in result.detail


def test_max_steps_without_a_loop_is_not_a_pass(env):
    """步数耗尽说明任务没收敛；即使每步参数都对也不能算通过。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text=f"第 {i} 次"),),
                   done_reason="tool_calls")
        for i in range(5)
    ]
    result = verify_case(
        _factory(env, scripts, budget=LoopBudget(max_steps=2))(),
        Expectation("回显", "echo", {"text": "第 0 次"}), _req("回显"),
    )
    assert result.verdict is Verdict.PASS, "步数耗尽但期望的调用确实发生了：判 PASS，stop_reason 另记"
    assert result.stop_reason == "max_steps"


def test_provider_error_is_its_own_verdict(env):
    scripts = [MockScript(raise_exc=ProviderUnreachable("连不上", base_url="mock://"))]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显", "echo", {"text": "x"}), _req("回显")
    )
    assert result.verdict is Verdict.ERROR
    assert "ProviderUnreachable" in result.detail


# ── mock 策略下的 PASS 必须标注 ───────────────────────────────────
def test_pass_under_fixture_policy_is_labelled_mocked(env):
    """用桩跑出来的 PASS 和真跑出来的 PASS 不是一回事，报告里必须看得出。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="hi"),), done_reason="tool_calls"),
        FINAL,
    ]
    ctx = ToolCtx(mock_policy=MockPolicy.FIXTURE, policy=PERMISSIVE_POLICY,
                  fixtures={"echo": {"echo": "hi", "times": 1, "chars": 2}})
    result = verify_case(
        _factory(env, scripts, ctx=ctx)(),
        Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi"),
    )
    assert result.verdict is Verdict.PASS
    assert result.mocked is True
    assert "mocked" in result.detail


def test_skipped_execution_does_not_count_as_a_tool_failure(env):
    """deny 策略下工具被跳过是**配置**，不是工具坏了。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="hi"),), done_reason="tool_calls"),
        FINAL,
    ]
    ctx = ToolCtx(mock_policy=MockPolicy.DENY, policy=PERMISSIVE_POLICY)
    result = verify_case(
        _factory(env, scripts, ctx=ctx)(),
        Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi"),
    )
    assert result.verdict is Verdict.PASS
    assert result.mocked is True


# ── 汇总口径 ──────────────────────────────────────────────────────
def _verdict(verdict: Verdict) -> VerifyResult:
    return VerifyResult(verdict=verdict, expectation=Expectation("i", "t", {}))


def test_pass_rate_excludes_tool_and_environment_failures():
    """分母只算"能判定模型对错"的用例。

    把 TOOL_FAILED 与 ERROR 算进模型得分，等于让模型替坏掉的工具和挂掉的引擎背锅。
    """
    results = [
        _verdict(Verdict.PASS), _verdict(Verdict.PASS), _verdict(Verdict.NO_CALL),
        _verdict(Verdict.WRONG_TOOL), _verdict(Verdict.BAD_ARGS),
        _verdict(Verdict.TOOL_FAILED), _verdict(Verdict.ERROR),
    ]
    report = summarize(results)
    assert report["total"] == 7
    assert report["passed"] == 2
    assert report["attributable"] == 5
    assert report["pass_rate"] == 0.4
    assert report["counts"]["tool_failed"] == 1 and report["counts"]["error"] == 1


def test_pass_rate_is_none_when_nothing_is_attributable():
    """全是环境故障时给 None，不给 0——0 会被读成"模型一个都没对"。"""
    assert summarize([_verdict(Verdict.ERROR)])["pass_rate"] is None
    assert summarize([])["pass_rate"] is None


def test_mocked_passes_are_counted_separately():
    results = [
        VerifyResult(verdict=Verdict.PASS, expectation=Expectation("i", "t", {}), mocked=True),
        VerifyResult(verdict=Verdict.PASS, expectation=Expectation("i", "t", {}), mocked=False),
        VerifyResult(verdict=Verdict.PASS, expectation=Expectation("i", "t", {}), mocked=None),
    ]
    assert summarize(results)["mocked_passes"] == 1


# ── 用例来源与批量执行 ────────────────────────────────────────────
def test_expectation_from_example_reads_the_tool_definitions():
    """定义里的 examples 本来就是为 fire-and-verify 准备的（审计规则 NO_EXAMPLE 催的就是它）。

    两处共用同一份数据，才不会各写一套然后悄悄漂移。
    """
    expectation = Expectation.from_example(ECHO.examples[0], case_id="echo-1")
    assert expectation.instruction == "把 hello 原样回显一次"
    assert expectation.tool == "echo"
    assert expectation.arguments == {"text": "hello", "times": 1}
    assert expectation.case_id == "echo-1"

    calc = Expectation.from_example(CALCULATOR.examples[0])
    assert calc.tool == "calculator" and calc.arguments == {"expr": "(12+8)*3"}


def test_expectation_from_example_tolerates_a_nameless_example():
    expectation = Expectation.from_example({"instruction": "做点什么", "expect": {}})
    assert expectation.tool == "" and expectation.arguments == {}


def test_verify_many_isolates_state_between_cases(env):
    """熔断计数是 loop 的状态：跨用例复用会让第 2 条用例被第 1 条的调用记录连累。"""
    looping = [MockScript(text="", tool_calls=(_call("echo", text="同一个"),),
                          done_reason="tool_calls")]
    clean = [
        MockScript(text="", tool_calls=(_call("echo", text="同一个"),), done_reason="tool_calls"),
        FINAL,
    ]
    # 第 1 条会熔断；第 2 条用同样的 (工具, 参数)，若状态没隔离就会跟着熔断
    factory_a = _factory(env, looping, budget=LoopBudget(max_steps=5, repeat_threshold=1))
    first = verify_case(factory_a(), Expectation("回显", "echo", {"text": "同一个"}), _req("回显"))
    assert first.verdict is Verdict.LOOP_BROKEN

    factory_b = _factory(env, clean, budget=LoopBudget(max_steps=5, repeat_threshold=1))
    second = verify_case(factory_b(), Expectation("回显", "echo", {"text": "同一个"}), _req("回显"))
    assert second.verdict is Verdict.PASS

    # verify_many 每条都新建 loop，所以上面两条放在一个批次里也互不影响
    batch = verify_many(factory_b, [
        (Expectation("回显", "echo", {"text": "同一个"}), _req("回显")),
        (Expectation("回显", "echo", {"text": "同一个"}), _req("回显")),
    ])
    assert [item.verdict for item in batch] == [Verdict.PASS, Verdict.PASS]


def test_verdict_enum_values_are_persistence_safe():
    """这些字符串会落库并进看板配色，改字面量等于改契约。"""
    assert [str(v) for v in Verdict] == [
        "pass", "no_call", "wrong_tool", "bad_args", "tool_failed", "loop_broken", "error",
    ]


# ── 诊断证据 ──────────────────────────────────────────────────────
def assert_tool_message_invariant(messages) -> None:
    from onyx.core.types import Message, Role

    assert all(isinstance(m, Message) for m in messages)
    for index, message in enumerate(messages):
        if message.role is not Role.ASSISTANT or not message.tool_calls:
            continue
        following = []
        for later in messages[index + 1:]:
            if later.role is Role.TOOL:
                following.append(later)
            else:
                break
        assert len(following) == len(message.tool_calls)


def test_verify_result_carries_the_message_sequence(env):
    """真机上"上下文错位"只能靠看这条序列诊断，而它在 loop 返回后就没了。"""
    scripts = [
        MockScript(text="", tool_calls=(_call("echo", text="hi"),), done_reason="tool_calls"),
        FINAL,
    ]
    result = verify_case(
        _factory(env, scripts)(), Expectation("回显 hi", "echo", {"text": "hi"}), _req("回显 hi")
    )
    assert result.verdict is Verdict.PASS
    assert_tool_message_invariant(result.messages)
    roles = [str(m.role) for m in result.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    # 第 2 步真正发给模型的序列与报告里的一致，否则报告就是另一套东西
    assert result.messages[2].name == "echo"


def test_message_sequence_is_present_even_when_the_run_fails(env):
    """失败时这条序列更有价值——NO_CALL 也要能看出模型究竟收到了什么。"""
    scripts = [MockScript(text="北京今天晴。", done_reason="stop")]
    result = verify_case(
        _factory(env, scripts)(), Expectation("天气", "echo", {"text": "x"}), _req("北京天气")
    )
    assert result.verdict is Verdict.NO_CALL
    assert [str(m.role) for m in result.messages] == ["user", "assistant"]
    assert_tool_message_invariant(result.messages)


def test_loop_broken_result_still_backfills_placeholders(env):
    """熔断那一轮的每个调用都要有对应的 tool 消息，否则下一次请求就错位了。"""
    scripts = [MockScript(text="", tool_calls=(_call("echo", text="同一个"),),
                          done_reason="tool_calls")]
    result = verify_case(
        _factory(env, scripts, budget=LoopBudget(max_steps=5, repeat_threshold=1))(),
        Expectation("回显", "echo", {"text": "同一个"}), _req("回显"),
    )
    assert result.verdict is Verdict.LOOP_BROKEN
    assert_tool_message_invariant(result.messages)
    tool_messages = [m for m in result.messages if str(m.role) == "tool"]
    # 第 1 步真跑了，第 2 步同一调用触发熔断 → 两步各一条 tool 消息，一条都不能少
    assert result.steps == 2
    assert len(tool_messages) == 2, "每一步的每个调用都要有对应的 tool 消息（含熔断后的占位）"
    assert "loop_break" in tool_messages[-1].content
    assert tool_messages[-1].extra["verdict"] == "loop_break"
