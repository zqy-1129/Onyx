"""S12 出口判据：**真实模型**上的 fire-and-verify。

离线测试能保证循环的形状正确，但保证不了两件只有真机才暴露的事：
1. 本地小模型到底会不会调工具、调成什么样（幻觉工具名 / 截断 JSON 的真实频率）；
2. 工具结果补回上下文后，引擎接不接受这个消息序列。

所以这里的断言刻意**不要求 PASS**：判定是什么不重要，重要的是判定必须是
六种之一且带得出证据，绝不许是 ERROR 或崩溃。要求真机必然 PASS 的测试，
在换模型之后就会变成一个没人信的常绿项。
"""

from __future__ import annotations

import os

import pytest

from onyx.core.content import FileBlobStore
from onyx.core.types import (
    GenerationRequest,
    GenParams,
    Role,
    TraceContext,
    TracePurpose,
)
from onyx.llm.gateway import Gateway
from onyx.llm.providers.ollama import OllamaProvider
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import TraceRepo
from onyx.store.sinks import SqliteRecordSink
from onyx.tools.builtin.defs import CALCULATOR, ECHO
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.loop import LoopBudget, StopReason, ToolLoop
from onyx.tools.sandbox import PERMISSIVE_POLICY
from onyx.tools.verify import Expectation, Verdict, verify_case

pytestmark = pytest.mark.live

MODEL = os.environ.get("ONYX_TEST_MODEL", "qwen3.5:9b")
BASE_URL = os.environ.get("ONYX_OLLAMA_URL", "http://127.0.0.1:11434")
TOOLS = (ECHO, CALCULATOR)
SPECS = tuple(d.to_spec() for d in TOOLS)

#: 判定必须落在这几种里。ERROR 意味着引擎或 Onyx 自己出了问题，不是模型的表现
ACCEPTABLE = {
    Verdict.PASS, Verdict.NO_CALL, Verdict.WRONG_TOOL,
    Verdict.BAD_ARGS, Verdict.LOOP_BROKEN, Verdict.TOOL_FAILED,
}


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=8, idle_wait=0.01)
    provider = OllamaProvider(base_url=BASE_URL)
    gateway = Gateway(provider, observer=ObserverEngine(record_sink=sink),
                      blobs=FileBlobStore(tmp_path / "blobs"))
    yield db, sink, gateway, provider
    sink.close()
    db.close()
    provider.close()


def _loop(env, *, mock_policy=MockPolicy.FIXTURE, fixtures=None, budget=None) -> ToolLoop:
    _, _, gateway, _ = env
    return ToolLoop(
        gateway, TOOLS,
        budget=budget or LoopBudget(max_steps=3),
        ctx=ToolCtx(
            mock_policy=mock_policy, policy=PERMISSIVE_POLICY,
            fixtures=fixtures or {"echo": {"echo": "stub", "times": 1, "chars": 4}},
        ),
    )


def _req(instruction: str) -> GenerationRequest:
    return GenerationRequest.of(
        MODEL, instruction,
        params=GenParams(max_tokens=512, temperature=0.0),
        thinking=False, tools=SPECS,
        context=TraceContext(purpose=TracePurpose.TOOL_TEST),
    )


def assert_tool_message_invariant(result) -> None:
    """带 N 个 tool_calls 的 assistant 消息，后面必须紧跟恰好 N 条 role=tool 消息。

    这条在真机上尤其重要：错位之后引擎通常不报错，只是开始答非所问。
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


@pytest.mark.parametrize("instruction,expectation", [
    ("把 hello 原样回显一次", Expectation("回显", "echo", {"text": "hello"})),
    ("帮我算 (12+8)*3 等于多少", Expectation("算术", "calculator", {"expr": "(12+8)*3"})),
])
def test_real_model_produces_an_attributable_verdict(env, instruction, expectation):
    result = verify_case(_loop(env), expectation, _req(instruction))

    assert result.verdict in ACCEPTABLE, (
        f"{MODEL} 上得到 {result.verdict}：{result.detail}"
    )
    assert result.stop_reason != str(StopReason.ERROR), result.detail
    assert_tool_message_invariant(result)

    # 每个判定都必须带得出证据，否则看板上就是一句无法行动的"失败"
    assert result.detail
    if result.verdict is Verdict.WRONG_TOOL:
        assert result.called, "WRONG_TOOL 必须列出实际调了什么"
    if result.verdict is Verdict.BAD_ARGS:
        assert result.args_diff or result.actual_args is None


def test_three_failure_kinds_are_distinguishable_on_a_real_model(env):
    """DoD 的核心：NO_CALL / WRONG_TOOL / BAD_ARGS 必须能分开。

    这三条指令是**故意**设计成三种不同失败的：
    - 不需要工具的问题 → 期望 NO_CALL
    - 需要工具但只给一个不相干的工具 → 期望 WRONG_TOOL
    - 需要工具且工具对，但指令没给出参数值 → 期望 BAD_ARGS 或 PASS（模型会猜）
    真机不一定如预期（那正是有价值的观测），所以断言的是"三种判定互不相同的
    能力存在"，而不是"某个模型必然落在某一格"。
    """
    loop = _loop(env)
    no_tool = verify_case(
        loop, Expectation("闲聊", "echo", {"text": "x"}),
        _req("用一句话解释什么是递归，不要调用任何工具。"),
    )
    wrong = verify_case(
        loop, Expectation("天气", "get_weather", {"city": "北京"}),
        _req("北京现在天气怎么样？"),
    )
    ambiguous = verify_case(
        loop, Expectation("回显", "echo", {"text": "某个具体值"}),
        _req("帮我回显一句话。"),
    )
    for item in (no_tool, wrong, ambiguous):
        assert item.verdict in ACCEPTABLE, item.detail
        assert_tool_message_invariant(item)

    # get_weather 不在本次工具集里，所以这一条**不可能**判 PASS：
    # 若它 PASS 了，说明判定逻辑把"调了别的工具"当成了成功
    assert wrong.verdict is not Verdict.PASS, wrong.detail
    # 三条指令的判定各自独立可解释（真机落在哪一格是观测结果，不预设）
    for item in (no_tool, wrong, ambiguous):
        assert item.detail and item.stop_reason


def test_fixture_policy_never_executes_a_real_tool(env):
    """评测默认档：零真实副作用。这条断言是"评测可复现"的地基。"""
    result = verify_case(
        _loop(env, mock_policy=MockPolicy.FIXTURE),
        Expectation("回显", "echo", {"text": "hello"}),
        _req("把 hello 原样回显一次"),
    )
    assert result.verdict in ACCEPTABLE
    if result.verdict is Verdict.PASS:
        assert result.mocked is True, "PASS 却不是桩执行——真实工具被调用了"


def test_deny_policy_skips_everything(env):
    result = verify_case(
        _loop(env, mock_policy=MockPolicy.DENY, fixtures=None),
        Expectation("回显", "echo", {"text": "hello"}),
        _req("把 hello 原样回显一次"),
    )
    assert result.verdict in ACCEPTABLE
    assert result.verdict is not Verdict.TOOL_FAILED, "纯观测档不该被记成工具坏了"


def test_live_execution_really_runs_the_tool(env):
    """`--mock live` 档：echo 是只读且幂等的，所以真跑是安全的。

    这条同时验证真实执行的结果能被补回上下文并被引擎接受——
    若消息序列不合法，引擎会在这一步报 400。
    """
    result = verify_case(
        _loop(env, mock_policy=MockPolicy.LIVE),
        Expectation("回显", "echo", {"text": "hello"}),
        _req("把 hello 原样回显一次"),
    )
    assert result.verdict in ACCEPTABLE, result.detail
    if result.verdict is Verdict.PASS:
        assert result.mocked is False
        assert_tool_message_invariant(result)


def test_every_step_is_persisted_and_linked(env):
    db, sink, _, _ = env
    result = verify_case(
        _loop(env), Expectation("回显", "echo", {"text": "hello"}),
        _req("把 hello 原样回显一次"),
    )
    sink.flush(10.0)
    repo = TraceRepo(db)
    assert result.steps >= 1
    # 每一步都是独立的 trace 行，靠 root_id 聚合（root 本身没有 trace 行）
    rows = [r for r in repo.list(limit=50) if r.root_id and r.model_name == MODEL]
    assert len(rows) == result.steps, f"落库 {len(rows)} 条，循环跑了 {result.steps} 步"
    assert len({r.root_id for r in rows}) == 1, "同一次 fire 的各步必须挂在同一个 root 上"
    assert all(row.purpose == "tool_test" for row in rows)


def test_tool_call_records_carry_the_execution_evidence(env):
    """看板要能下钻到工具执行的原始结果（UI_DESIGN R7）。"""
    db, sink, _, _ = env
    result = verify_case(
        _loop(env, mock_policy=MockPolicy.LIVE),
        Expectation("回显", "echo", {"text": "hello"}),
        _req("把 hello 原样回显一次"),
    )
    sink.flush(10.0)
    if result.verdict is not Verdict.PASS:
        pytest.skip(f"{MODEL} 这次没有正确调用 echo（{result.verdict}），无从验证落库字段")

    repo = TraceRepo(db)
    traces = [r for r in repo.list(limit=50) if r.root_id]
    calls = [c for t in traces for c in repo.list_tool_calls(t.id)]
    assert calls, "PASS 却没有工具调用记录落库"
    record = next(c for c in calls if c.name == "echo")
    assert record.result_status == "ok"
    assert record.executed_by == "client"
    assert record.latency_ms is not None
    assert record.tool_def_hash == ECHO.hash
    assert record.result_ref.startswith("sha256:")
    # repo 已经把 args_json 解码成 dict 了，不要再 loads 一遍
    assert record.args == {"text": "hello"} or record.args_raw
