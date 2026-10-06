"""MCP over stdio：真子进程、真管道。

放在 `tests/unit` 而不是 `tests/integration` 是有意的：后者会去拿机器级 GPU 锁
（那里跑的测试都依赖 Ollama，基准数字必须独占 GPU 才有效），而本文件**不需要 GPU**，
只是需要一个真的操作系统管道。它离线、hermetic、不联网。

为什么不能只靠假传输：伪造的 `read()` 测的是我们对协议的理解，
测不到帧本身——换行分隔、文本缓冲、以及最重要的 **stderr 管道满时死锁**。
那个故障的现象是"卡住"而不是"报错"，只在真管道上才现形。
同一份 server 也被 `onyx tools contract` 的 `mcp_stdio` 列使用（S35）。
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from onyx.core.errors import ToolTimeout
from onyx.tools import reference_mcp_server
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.executors.mcp import McpExecutor
from onyx.tools.mcp import McpPool, ServerSpec, tooldefs_from_mcp
from onyx.tools.sandbox import SandboxPolicy
from onyx.tools.spec import SideEffect

#: server 在包里而不在 tests/fixtures：CLI 的契约矩阵也要能起它，而矩阵不经过 conftest。
#: 只留一份——两份会漂成"测试绿、矩阵红"，那时没人说得出哪一份是假的。
SERVER = Path(reference_mcp_server.__file__).resolve()

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


def test_deadline_exceeded_returns_timeout_and_quarantines_the_connection(pool):
    """超时的那条连接必须当场作废，且作废发生在**返回给调用方之前**。

    顺序很关键：如果隔离是靠被放弃的工作线程去做，主线程早已把"超时"返回了，
    下一笔调用就可能**复用**这条正要被关掉的连接——真机撞到的就是这个，
    报的是"下一次调用读到 closed file"，而罪在上一笔已经超时返回的调用。
    """
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    stale = pool.session("demo")
    process = stale.transport.process

    result = McpExecutor(defs["slow"], pool=pool).call(
        "demo__slow", {}, _live_ctx(deadline_ms=300))
    assert not result.ok and result.error_kind == "timeout"
    # 结果已经返回 ⇒ 此刻池子里就不该再有这条连接（不是"稍后被工作线程关掉"）
    assert "demo" not in pool.stats()["running"]

    # 被隔离的子进程必须真的没了：僵尸 server 会攒成句柄泄漏
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.05)
    assert process.poll() is not None, "隔离没把子进程收掉"


def test_timeout_does_not_poison_the_next_call(pool):
    """一次慢调用不许污染同一个 server 上的后续调用。

    这就是 `mcp_stdio` 列最初偶发失败的原形：`read_is_idempotent` 报
    `first=None second='北京：晴，21 度'`——weather 本身没问题，是上一笔 slow
    调用留下的线程/连接把它的回答读走了。这类"上一笔的账记到下一笔头上"最难归因。
    """
    defs = {d.extra["mcp_tool"]: d for d in tooldefs_from_mcp(pool, "demo")}
    assert not McpExecutor(defs["slow"], pool=pool).call(
        "demo__slow", {}, _live_ctx(deadline_ms=300)).ok

    after = McpExecutor(defs["weather"], pool=pool).call(
        "demo__weather", {"city": "北京"}, _live_ctx())
    assert after.ok and after.output == "北京：晴，21 度", after.error


def test_concurrent_calls_on_one_server_never_steal_each_others_answers(pool):
    """同一个 session 上并发调用：每个调用必须拿到**自己**那条回答。

    一根管道只能有一个读者。`McpSession` 的文档一直写着"读循环在锁内串行"，
    而 `_request` 并没真的加锁——去掉那把锁这个测试就会（偶发地）失败，
    失败现象是"另一个城市的答案"或"server 没有响应"，不是"并发不支持"。
    """
    session = pool.session("demo")
    cities = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "西安"]
    answers: dict[str, str] = {}

    def ask(city: str) -> None:
        raw = session.call_tool("weather", {"city": city}, timeout=8.0)
        answers[city] = " ".join(
            str(b.get("text") or "") for b in raw.get("content") or [] if isinstance(b, dict)
        )

    threads = [threading.Thread(target=ask, args=(c,)) for c in cities]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=20.0)
        assert not t.is_alive(), "并发调用卡住了：说明有读者在等一条永远不会来的回答"

    assert set(answers) == set(cities)
    for city in cities:
        assert answers[city].startswith(f"{city}："), f"{city} 拿到了别人的答案: {answers[city]}"


def test_abandoned_read_does_not_swallow_the_next_answer(pool):
    """超时的读不能留下一个还在抢管道的读者。

    这里**故意不隔离**（直接调 session），就是为了单独测传输层的性质：
    旧实现每次 `read()` 起一个线程、超时后不再管它，那个线程会把后到的整行读进
    一个没人看的队列。现象是"下一次调用没有响应"，而慢的永远是**下一笔**。
    """
    session = pool.session("demo")
    with pytest.raises(ToolTimeout):
        session.call_tool("slow", {}, timeout=0.3)
    raw = session.call_tool("weather", {"city": "南京"}, timeout=9.0)
    text = " ".join(str(b.get("text") or "") for b in raw.get("content") or [])
    assert text == "南京：晴，21 度", f"迟到回答没被按 id 丢弃，或被孤儿读者吞了：{text!r}"


def test_session_raises_tool_timeout_directly(pool):
    """直接看协议层：超时是 `ToolTimeout`，不是笼统的 RuntimeError。"""
    session = pool.session("demo")
    with pytest.raises(ToolTimeout):
        session.call_tool("slow", {}, timeout=0.3)


def test_reference_server_does_not_import_onyx():
    """这份 server 的可分性就是它的全部价值：它必须能被裸解释器跑起来。

    它一旦被 import 到 onyx 里，`(sys.executable, 文件路径)` 这种方式就会因为
    PYTHONPATH 不同而失败——而那次失败的现象与"MCP 执行器坏了"一模一样，
    排查的人会先去查执行器。所以这条不许漂。
    """
    import ast

    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    assert "onyx" not in imported, f"server 反向依赖了 onyx：{sorted(imported)}"
    assert imported <= {"__future__", "contextlib", "json", "sys", "time"}, sorted(imported)


def test_server_starts_from_a_foreign_cwd(tmp_path):
    """从别的目录起进程也要能成：契约矩阵是人在任何工作目录下跑的 CLI。"""
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(SERVER), "--help"], capture_output=True, text=True,
        encoding="utf-8", timeout=10, cwd=str(tmp_path), input="",
    )
    # server 不认 flag：stdin 关闭就正常退出（0），关键是它**起得来**
    assert proc.returncode == 0, proc.stderr[:300]
    assert "starting" in proc.stderr
