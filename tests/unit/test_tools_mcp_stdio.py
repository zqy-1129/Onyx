"""MCP over stdio：真子进程、真管道。

放在 `tests/unit` 而不是 `tests/integration` 是有意的：后者会去拿机器级 GPU 锁
（那里跑的测试都依赖 Ollama，基准数字必须独占 GPU 才有效），而本文件**不需要 GPU**，
只是需要一个真的操作系统管道。它离线、hermetic、不联网。

为什么不能只靠假传输：伪造的 `read()` 测的是我们对协议的理解，
测不到帧本身——换行分隔、文本缓冲、以及最重要的 **stderr 管道满时死锁**。
那个故障的现象是"卡住"而不是"报错"，只在真管道上才现形。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from onyx.core.errors import ToolTimeout
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.executors.mcp import McpExecutor
from onyx.tools.mcp import McpPool, ServerSpec, tooldefs_from_mcp
from onyx.tools.sandbox import SandboxPolicy
from onyx.tools.spec import SideEffect

SERVER = Path(__file__).resolve().parent.parent / "fixtures" / "mcp_demo_server.py"

DEMO = ServerSpec(
    name="demo", command=(sys.executable, str(SERVER)),
    env=("PATH",), timeout_s=10.0,
)


@pytest.fixture
def pool():
    instance = McpPool({"demo": DEMO})
    yield instance
    instance.close()


def _live_ctx(**kw) -> ToolCtx:
    policy = SandboxPolicy(
        allowed_side_effects=frozenset({SideEffect.READ, SideEffect.WRITE}),
        require_approval=frozenset(),
        allowed_impl_prefixes=("mcp:",),
    )
    return ToolCtx(policy=policy, mock_policy=MockPolicy.LIVE, **kw)


def test_handshake_and_discovery_over_a_real_pipe(pool):
    defs = tooldefs_from_mcp(pool, "demo")
    by_tool = {d.impl_ref: d for d in defs}
    assert len(defs) == 5
    assert by_tool["mcp:demo:weather"].side_effect is SideEffect.READ
    # server 没标注的那个工具必须被当成有副作用——默认当成只读就是审计错误
    assert by_tool["mcp:demo:send_email"].side_effect is SideEffect.WRITE
    assert by_tool["mcp:demo:weather"].name == "demo__weather"


def test_real_call_returns_text_and_reuses_one_process(pool):
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    executor = McpExecutor(defs["weather"], pool=pool)
    first = executor.call("demo__weather", {"city": "北京"}, _live_ctx())
    second = executor.call("demo__weather", {"city": "上海"}, _live_ctx())
    assert first.ok and "北京：晴" in first.output
    assert second.ok and "上海：晴" in second.output
    # 两次调用一个进程：握手有几百毫秒开销，每次重开会让"工具开销"变成主要成本
    assert pool.stats()["started"] == ["demo"]
    assert pool.stats()["calls"]["demo"] == 2


def test_stdout_log_lines_do_not_become_answers(pool):
    """server 在 initialize 之后往 stdout 打了一行非 JSON 文本。

    把它当成"空回答"的话，正文会是空串——而空正文与"模型真的没输出"在评测里
    长成同一个 `invalid_format`。
    """
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    result = McpExecutor(defs["weather"], pool=pool).call(
        "demo__weather", {"city": "广州"}, _live_ctx())
    assert result.ok and result.output == "广州：晴，21 度"


def test_chatty_stderr_does_not_deadlock(pool):
    """2000 行 stderr。客户端不持续排空就会在管道满时死锁，现象是卡住而不是报错。"""
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    started = time.monotonic()
    result = McpExecutor(defs["noisy"], pool=pool).call("demo__noisy", {}, _live_ctx())
    assert result.ok and result.output == "灌完了"
    assert time.monotonic() - started < 8.0, "noisy 应该几秒内返回；卡住说明 stderr 没被排空"


def test_tool_side_error_is_not_reported_as_success(pool):
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    result = McpExecutor(defs["boom"], pool=pool).call("demo__boom", {}, _live_ctx())
    assert not result.ok
    assert result.error_kind == "error", "server 侧失败归 error，不许记成 arg_error"
    assert "550" in result.error


def test_deadline_exceeded_returns_timeout_and_the_process_can_be_reaped(pool):
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    result = McpExecutor(defs["slow"], pool=pool).call(
        "demo__slow", {}, _live_ctx(deadline_ms=300))
    assert not result.ok and result.error_kind == "timeout"

    # 关掉之后子进程必须真的没了：僵尸 server 会攒成句柄泄漏
    transport = pool._sessions["demo"].transport
    process = transport.process
    pool.close()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.05)
    assert process.poll() is not None, "close() 没把子进程收掉"


def test_session_raises_tool_timeout_directly(pool):
    """直接看协议层：超时是 `ToolTimeout`，不是笼统的 RuntimeError。"""
    session = pool.session("demo")
    with pytest.raises(ToolTimeout):
        session.call_tool("slow", {}, timeout=0.3)
