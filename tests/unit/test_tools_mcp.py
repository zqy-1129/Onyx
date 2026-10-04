"""S16d：MCP 客户端与执行器。

这里测的是**协议边界**，不依赖任何真实 MCP server（有一个专门的集成测试
用真子进程验证 stdio 帧，见 `tests/integration/test_mcp_stdio.py`）。

盯的四件事，都是"看起来能用、实际会坑"的位置：
1. 帧与 id 配对：服务器会发通知、也会发迟到的回答；认错了 id 就把两次调用
   的结果拼在一起，比报错更糟；
2. 非协议行与 stderr：不少 server 往 stdout 打日志、往 stderr 打进度。
   前者必须跳过，后者必须持续排空（不然管道满时死锁）；
3. 副作用默认值：MCP 的 `annotations` 是服务器自报的提示，未标注必须当成有副作用；
4. 评测期零真实副作用：`mock_policy=fixture` 下一个子进程都不许起。
"""

from __future__ import annotations

import json
import re
import time

import pytest

from onyx.core.errors import ToolRuntime, ToolTimeout, ToolUnknown
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.executors.mcp import McpExecutor, config_path, default_pool, parse_impl_ref
from onyx.tools.mcp import (
    MAX_OUTPUT_CHARS,
    PROTOCOL_VERSION,
    McpError,
    McpPool,
    McpSession,
    ServerSpec,
    load_servers,
    load_servers_file,
    payload_from_mcp,
    tooldefs_from_mcp,
)
from onyx.tools.spec import SideEffect, ToolDef, ToolKind

DEMO = ServerSpec(name="demo", command=("python", "demo.py"))


class FakeTransport:
    """脚本化的假传输：write 记账，read 按脚本给。"""

    def __init__(self, script: list[dict | None]) -> None:
        self.script = list(script)
        self.sent: list[dict] = []
        self.closed = False
        self.hang = False          # True ⇒ read 直接超时，模拟 server 不响应
        self.sleep_s = 0.0

    def write(self, payload: dict) -> None:
        self.sent.append(payload)

    def read(self, timeout: float):
        if self.sleep_s:
            time.sleep(self.sleep_s)
        if self.hang or not self.script:
            raise ToolTimeout("fake: 没有更多回答", detail={"timeout": timeout})
        item = self.script.pop(0)
        return item

    def close(self) -> None:
        self.closed = True


def _init_result(version: str = PROTOCOL_VERSION) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "result": {
        "protocolVersion": version,
        "serverInfo": {"name": "demo-server", "version": "1.2.3"},
        "capabilities": {"tools": {}},
    }}


def _session(*script: dict | None) -> tuple[McpSession, FakeTransport]:
    transport = FakeTransport(list(script))
    return McpSession(DEMO, transport=transport), transport


# ── 握手与帧 ──────────────────────────────────────────────────────
def test_initialize_sends_the_handshake_then_the_notification():
    session, transport = _session(_init_result())
    result = session.initialize()
    assert result["serverInfo"]["name"] == "demo-server"
    assert session.protocol_version == PROTOCOL_VERSION
    request, notification = transport.sent[0], transport.sent[1]
    assert request["method"] == "initialize" and request["id"] == 1
    assert request["params"]["protocolVersion"] == PROTOCOL_VERSION
    assert request["params"]["clientInfo"]["name"] == "onyx"
    # initialized 是通知：有 method 没 id。带 id 发出去会被当成请求，server 会回一条
    assert notification == {"jsonrpc": "2.0", "method": "notifications/initialized"}


def test_unsupported_protocol_version_is_refused_not_guessed():
    session, _ = _session(_init_result(version="1999-01-01"))
    with pytest.raises(McpError, match="协议版本"):
        session.initialize()


def test_notifications_and_late_answers_are_skipped_not_matched():
    """认错 id = 把两次调用的结果拼在一起，比直接报错更难查。"""
    session, transport = _session(
        {"jsonrpc": "2.0", "method": "notifications/message", "params": {"x": 1}},
        {"jsonrpc": "2.0", "id": 99, "result": {"tools": []}},        # 迟到的别人的回答
        {"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "ok", "description": "d"}]}},
    )
    session._next_id = 1  # 下一个请求的 id 将是 2
    tools = session.list_tools()
    assert [t["name"] for t in tools] == ["ok"]
    assert transport.sent[0]["id"] == 2


def test_jsonrpc_error_becomes_mc_error_with_code():
    session, _ = _session(
        _init_result(),
        {"jsonrpc": "2.0", "id": 2, "error": {"code": -32601, "message": "method not found"}},
    )
    session._next_id = 1
    with pytest.raises(McpError) as exc:
        session.list_tools()
    assert exc.value.code == -32601
    assert "method not found" in str(exc.value)


def test_non_protocol_lines_are_skipped_not_treated_as_answers():
    """read 返回 None 表示"这不是 JSON"，请求循环必须继续等而不是当空回答。"""
    session, transport = _session(_init_result(), None, {"jsonrpc": "2.0", "id": 2,
                                                          "result": {"tools": []}})
    session.initialize()
    session._next_id = 1
    assert session.list_tools() == []
    assert len(transport.sent) == 3


def test_timeout_propagates_as_tool_timeout():
    session, transport = _session(_init_result())
    session.initialize()
    transport.hang = True
    session._next_id = 1
    with pytest.raises(ToolTimeout):
        session.list_tools()


# ── 配置 ──────────────────────────────────────────────────────────
def test_config_accepts_the_common_shapes():
    servers = load_servers({"mcpServers": {
        "demo": {"command": "python", "args": ["s.py"], "env": ["PATH"], "timeout_s": 5},
    }})
    assert servers["demo"].argv == ("python", "s.py")
    assert servers["demo"].env == ("PATH",)


def test_config_rejects_empty_and_broken_entries():
    with pytest.raises(ValueError, match="command"):
        load_servers({"mcpServers": {"demo": {"args": ["x"]}}})
    with pytest.raises(ValueError, match="timeout_s"):
        load_servers({"demo": {"command": ["python"], "timeout_s": 0}})
    with pytest.raises(ValueError, match="对象"):
        load_servers({"demo": "python server.py"})
    with pytest.raises(ValueError, match="没有 server"):
        load_servers({})


def test_config_file_error_names_the_file_and_line(tmp_path):
    bad = tmp_path / "mcp.json"
    bad.write_text('{"mcpServers": {"a": {"command": ["x"],}}}', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("mcp.json")):
        load_servers_file(bad)


# ── 发现 → ToolDef ────────────────────────────────────────────────
def _pool_with(tools: list[dict]) -> McpPool:
    """造一个已经"接上"的池：用假传输，不起进程。"""
    transport = FakeTransport([_init_result(), {"jsonrpc": "2.0", "id": 2,
                                                "result": {"tools": tools}}])
    session = McpSession(DEMO, transport=transport)
    pool = McpPool({"demo": DEMO})
    pool._sessions["demo"] = session        
    pool._next_transport = None  # type: ignore[attr-defined]
    session._next_id = 1
    return pool


def test_discovered_definitions_are_prefixed_and_traceable():
    defs = tooldefs_from_mcp(_pool_with([{
        "name": "fetch", "title": "抓页面", "description": "抓取指定 URL 的内容",
        "inputSchema": {"type": "object", "properties": {"url": {"type": "string"}},
                        "required": ["url"]},
        "annotations": {"readOnlyHint": True},
    }]), "demo")
    assert len(defs) == 1
    definition = defs[0]
    # 两个 server 都有 `fetch` 时，裸名注册会互相覆盖，而"这工具是谁提供的"
    # 恰好是排障要问的第一个问题
    assert definition.name == "demo__fetch"
    assert definition.impl_ref == "mcp:demo:fetch"
    assert definition.extra["mcp_tool"] == "fetch"
    assert definition.kind == ToolKind.MCP
    assert definition.side_effect is SideEffect.READ
    assert definition.parameters["required"] == ["url"]
    assert definition.doc == "抓页面"


@pytest.mark.parametrize("annotations,expected,reason", [
    (None, SideEffect.WRITE, "未标注"),
    ({"readOnlyHint": True}, SideEffect.READ, "自报只读"),
    ({"readOnlyHint": True, "destructiveHint": True}, SideEffect.WRITE, "destructive 优先"),
    ({"readOnlyHint": False}, SideEffect.WRITE, "自报非只读"),
    ("不是对象", SideEffect.WRITE, "形状不对"),
])
def test_side_effect_defaults_to_conservative(annotations, expected, reason):
    tool = {"name": "t", "description": "x" * 30, "inputSchema": {"type": "object"}}
    if annotations is not None:
        tool["annotations"] = annotations
    definition = tooldefs_from_mcp(_pool_with([tool]), "demo")[0]
    assert definition.side_effect is expected, reason
    assert definition.extra["side_effect_reason"]


def test_discovery_rejects_a_server_with_no_tools():
    with pytest.raises(McpError, match="没有报告任何工具"):
        tooldefs_from_mcp(_pool_with([]), "demo")


def test_discovered_defs_pass_the_existing_audit():
    """发现的定义必须过同一套契约审计，否则"接入 MCP"就变成往注册表里灌不合格定义。"""
    from onyx.tools.spec import audit_many

    defs = tooldefs_from_mcp(_pool_with([{
        "name": "weather", "description": "查询指定城市的当前天气情况与温度",
        "inputSchema": {"type": "object", "properties": {"city": {"type": "string",
                      "description": "城市名，如北京"}}, "required": ["city"]},
        "annotations": {"readOnlyHint": True},
    }]), "demo")
    report = audit_many(defs)
    errors = [f for items in report.values() for f in items if f.severity.value == "error"]
    assert not errors, f"发现的 MCP 定义没过审计: {errors}"


# ── 结果映射 ──────────────────────────────────────────────────────
def test_non_utf8_bytes_do_not_kill_the_reader():
    """真机撞到的：中文 Windows 上子进程的 stdout 可能是 GBK。

    text 模式下 `readline()` 会抛 UnicodeDecodeError，**读线程当场死掉**，
    父进程再也等不到回答——现象是"卡到超时"而不是"编码错了"。
    现在按二进制读、显式解码：坏字节替换，解不出 JSON 就当非协议行跳过。
    """
    from onyx.tools.mcp import decode_line

    raw = ("北京：" + "晴").encode("gbk") + b"\n"
    text = decode_line(raw)             # 不许抛
    assert "\n" not in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text.strip())        # 不是合法 JSON ⇒ 上层按非协议行跳过


def test_text_blocks_are_joined_and_images_are_not_inlined():
    payload = payload_from_mcp({"content": [
        {"type": "text", "text": "北京 晴"},
        {"type": "text", "text": "21 度"},
        {"type": "image", "mimeType": "image/png", "data": "A" * 4000},
    ]})
    assert payload == "北京 晴\n21 度"


def test_image_only_results_expose_size_not_bytes():
    result = payload_from_mcp({"content": [{"type": "image", "mimeType": "image/png",
                                            "data": "A" * 2048}]})
    blocks = result["blocks"] if isinstance(result, dict) else []
    assert blocks[0]["type"] == "image"
    assert blocks[0]["bytes"] == 2048, "base64 内联会把几百 KB 塞进上下文，而上下文开销正是被测对象"
    assert "data" not in blocks[0]


def test_structured_content_wins_when_present():
    assert payload_from_mcp({
        "content": [{"type": "text", "text": "ignore me"}],
        "structuredContent": {"temp_c": 21},
    }) == {"temp_c": 21}


def test_is_error_raises_tool_runtime_not_arg_error():
    """`isError` 是 server 说这次调用失败了 ⇒ `error` 档。

    记成 `arg_error` 会把 server 的 bug 算成模型的错，而修法完全相反。
    """
    with pytest.raises(ToolRuntime, match="工具执行失败") as exc:
        payload_from_mcp({"content": [{"type": "text", "text": "工具执行失败"}],
                          "isError": True})
    assert exc.value.detail["kind"] == "mcp_tool_error"


def test_long_output_is_truncated_with_the_original_length_stated():
    text = "x" * (MAX_OUTPUT_CHARS + 1000)
    out = payload_from_mcp({"content": [{"type": "text", "text": text}]})
    assert "截断" in out and str(len(text)) in out, "截断必须留下原长，否则读者以为那就是全部输出"


# ── 执行器 ────────────────────────────────────────────────────────
def _mcp_def(effect: SideEffect = SideEffect.READ, **kw) -> ToolDef:
    return ToolDef(
        name="demo__weather", description="查询指定城市的当前天气",
        parameters={"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]},
        kind=ToolKind.MCP, side_effect=effect, impl_ref="mcp:demo:weather", **kw,
    )


def _ctx(**kw) -> ToolCtx:
    from onyx.tools.sandbox import SandboxPolicy

    policy = kw.pop("policy", None) or SandboxPolicy(
        allowed_side_effects=frozenset({SideEffect.READ}),
        require_approval=frozenset({SideEffect.WRITE}),
        allowed_impl_prefixes=("mcp:demo:",),
    )
    return ToolCtx(policy=policy, **kw)


def _executor(call_result: dict, **kw) -> tuple[McpExecutor, FakeTransport]:
    """已经握过手的会话：下一次请求的 id 是 2，脚本里就按 id=2 回答。

    如果两边 id 错开，`_init_result()` 会被当成 `tools/call` 的回答，
    测试就会"通过"在一次根本没有发生的调用上——那比失败更糟。
    """
    transport = FakeTransport([{"jsonrpc": "2.0", "id": 2, "result": call_result}])
    session = McpSession(DEMO, transport=transport)
    session.protocol_version = PROTOCOL_VERSION
    session._next_id = 1                    
    pool = McpPool({"demo": DEMO})
    pool._sessions["demo"] = session        
    return McpExecutor(kw.pop("definition", _mcp_def()), pool=pool), transport


def test_real_call_returns_the_tool_output():
    executor, transport = _executor({"content": [{"type": "text", "text": "晴 21 度"}]})
    result = executor.call("demo__weather", {"city": "北京"}, _ctx(
        mock_policy=MockPolicy.LIVE))
    assert result.ok and result.output == "晴 21 度"
    call = transport.sent[-1]
    assert call["method"] == "tools/call"
    assert call["params"] == {"name": "weather", "arguments": {"city": "北京"}}
    assert executor.real_calls == 1


def test_wrong_tool_name_is_unknown_tool():
    executor, transport = _executor({"content": []})
    result = executor.call("other__weather", {"city": "北京"}, _ctx(mock_policy=MockPolicy.LIVE))
    assert not result.ok and result.error_kind == "unknown_tool"
    assert transport.sent == [], "路由错不该打到 server 上"


def test_bad_args_never_reach_the_server():
    """参数校验在 `guarded_call` 里，先于任何外部副作用——这条必须成立，
    否则模型可以用一次调用去探测 server。"""
    executor, transport = _executor({"content": []})
    result = executor.call("demo__weather", {}, _ctx(mock_policy=MockPolicy.LIVE))
    assert not result.ok and result.error_kind == "arg_error"
    assert transport.sent == []


def test_write_side_effect_is_refused_before_any_process_starts():
    executor, transport = _executor({"content": []})
    definition = _mcp_def(effect=SideEffect.WRITE)
    result = McpExecutor(definition, pool=executor._pool).call(
        definition.name, {"city": "北京"}, _ctx(mock_policy=MockPolicy.LIVE))
    # 副作用类工具在策略层就被拒：一次都没打到 server，也就没起子进程
    assert not result.ok and result.error_kind == "rejected"
    assert transport.sent == []


def test_impl_ref_outside_the_allow_list_is_refused():
    from onyx.tools.sandbox import SandboxPolicy

    policy = SandboxPolicy(allowed_side_effects=frozenset({SideEffect.READ}),
                           allowed_impl_prefixes=("mcp:trusted:",))
    executor, transport = _executor({"content": []})
    result = executor.call("demo__weather", {"city": "北京"},
                           _ctx(policy=policy, mock_policy=MockPolicy.LIVE))
    assert not result.ok and result.error_kind == "rejected"
    assert transport.sent == [], "白名单不过 ⇒ 连请求都不许发"


def test_unconfigured_server_is_a_clear_error():
    """impl_ref 白名单**过了**、但配置里没有这个 server ⇒ 说清楚是配置缺失。

    顺序是有意义的：白名单不过时在策略层就拒（连请求都不发），
    这两种失败的修法完全不同（改策略 vs 改配置文件）。
    """
    from onyx.tools.sandbox import SandboxPolicy

    executor, _ = _executor({"content": []})
    executor.definition = ToolDef(
        name="demo__weather", description="x" * 30, kind=ToolKind.MCP,
        impl_ref="mcp:stranger:weather",
        parameters={"type": "object", "properties": {"city": {"type": "string"}},
                    "required": ["city"]},
    )
    ctx = _ctx(
        mock_policy=MockPolicy.LIVE,
        policy=SandboxPolicy(allowed_side_effects=frozenset({SideEffect.READ}),
                             allowed_impl_prefixes=("mcp:",)),
    )
    result = executor.call("demo__weather", {"city": "北京"}, ctx)
    assert not result.ok and result.error_kind == "error", (
        "定义与实现脱节属于 `error` 档，不许记成 `unknown_tool`（那是模型的错）"
    )
    assert "未配置的 MCP server" in result.error
    assert "demo" in result.error or result.extra.get("detail", {}).get("known") == ["demo"]


def test_fixture_policy_starts_nothing():
    """评测的零副作用保证：`mock_policy=fixture` 时有桩就用桩，**一个字节都不发**。"""
    executor, transport = _executor({"content": [{"type": "text", "text": "real"}]})
    ctx = _ctx(mock_policy=MockPolicy.FIXTURE,
               fixtures={"demo__weather": {"stub": "晴 25 度"}})
    result = executor.call("demo__weather", {"city": "北京"}, ctx)
    assert result.ok and result.mocked and result.output == {"stub": "晴 25 度"}
    assert transport.sent == [], "评测期打到真实 server 会让分数取决于对方当时的心情"


def test_missing_fixture_refuses_to_run_live():
    executor, transport = _executor({"content": [{"type": "text", "text": "real"}]})
    result = executor.call("demo__weather", {"city": "北京"}, _ctx(mock_policy=MockPolicy.FIXTURE))
    assert not result.ok and result.error_kind == "skipped"
    assert transport.sent == [], "没有桩就退回真跑，等于把'评测可复现'变成口号"


def test_server_error_maps_to_error_kind_not_arg_error():
    executor, _ = _executor({
        "content": [{"type": "text", "text": "upstream blew up"}], "isError": True,
    })
    result = executor.call("demo__weather", {"city": "北京"}, _ctx(mock_policy=MockPolicy.LIVE))
    assert not result.ok
    assert result.error_kind == "error", "server 的 bug 不许记成模型的错"
    assert "upstream blew up" in result.error


def test_deadline_becomes_timeout_kind():
    executor, transport = _executor({"content": []})
    transport.hang = True          # server 收了请求但不回答（进程还在，只是慢）
    result = executor.call("demo__weather", {"city": "北京"},
                           _ctx(mock_policy=MockPolicy.LIVE, deadline_ms=5))
    assert not result.ok and result.error_kind == "timeout", (
        "超时要归到 timeout 档（可重试），笼统记成 error 就丢掉了修法"
    )


@pytest.mark.parametrize("ref", ["", "demo:weather", "mcp:demo", "mcp::weather",
                                 "http://x", "mcp::"])
def test_impl_ref_shape_is_enforced(ref):
    """只接受 `mcp:<server>:<tool>`。

    工具定义是从文件导入、可以被改写的；如果这里允许任意 argv，
    一份坏定义就等于让模型决定跑什么命令。
    """
    with pytest.raises(ToolUnknown, match=re.escape("mcp:<server>:<tool>")):
        parse_impl_ref(ref)


def test_parse_impl_ref_splits_on_the_last_colon_only():
    assert parse_impl_ref("mcp:demo:weather") == ("demo", "weather")
    assert parse_impl_ref("mcp:demo:we:ather") == ("demo", "we:ather")


# ── 配置入口与池缓存 ─────────────────────────────────────────────
def test_default_pool_missing_config_says_how_to_fix_it(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_MCP_CONFIG", str(tmp_path / "nope.json"))
    from onyx.tools.executors.mcp import reset_pools

    reset_pools()
    with pytest.raises(ToolUnknown) as exc:
        default_pool()
    assert "mcpServers" in exc.value.detail["hint"]
    assert exc.value.detail["env"] == "ONYX_MCP_CONFIG"
    reset_pools()


def test_default_pool_is_cached_per_path(tmp_path, monkeypatch):
    from onyx.tools.executors.mcp import reset_pools

    config = tmp_path / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"demo": {"command": ["python", "s.py"]}}}),
                      encoding="utf-8")
    monkeypatch.setenv("ONYX_MCP_CONFIG", str(config))
    reset_pools()
    first = default_pool()
    assert config_path() == config
    assert default_pool() is first, "一个 server 一个子进程：重复建池会攒出一堆僵尸"
    assert list(first.servers) == ["demo"]
    reset_pools()


def test_pool_rejects_unknown_server_with_the_known_list():
    pool = McpPool({"demo": DEMO})
    with pytest.raises(ToolUnknown) as exc:
        pool.session("stranger")
    assert "demo" in exc.value.detail["known"]
