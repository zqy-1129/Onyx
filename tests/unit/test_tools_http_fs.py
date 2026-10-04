"""S11 验收：http 执行器与 fs_read 的安全边界。

http 这一列的关键不是"能发请求"，而是两件容易被跳过的事：
1. **deadline 真的传到了 socket**——否则 `run_with_deadline` 放弃等待后，
   卡住的请求会永久占用线程池 worker，现象是"越来越慢"而不是报错。
2. **离线可测**——用 MockTransport 断言零真实网络，评测才可复现。
"""

from __future__ import annotations

import httpx
import pytest

from onyx.core.errors import ToolSandboxDenied, ToolUnknown
from onyx.tools.builtin.fs_read import read_file
from onyx.tools.contract import CONTRACT_NAMES, run_contracts, summarize
from onyx.tools.executor import MockPolicy, ToolCtx
from onyx.tools.executors import EXECUTOR_KINDS, executor_for
from onyx.tools.executors.http import (
    HttpExecutor,
    _effective_timeout_ms,
    _read_config,
    contract_target,
)
from onyx.tools.executors.python_fn import PythonFnExecutor
from onyx.tools.sandbox import PERMISSIVE_POLICY, SandboxPolicy
from onyx.tools.spec import SideEffect, ToolDef, ToolKind

#: `.invalid` 是 RFC 2606 保留的永不解析域名：万一 transport 注入失效，
#: 请求会 DNS 失败而不是打到某个真实服务上
BASE = "http://contract.invalid"


def _http_def(path: str = "/weather", **extra_http) -> ToolDef:
    return ToolDef(
        name="http_demo",
        description="http 执行器测试用的工具定义，端点写死在定义里",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，例如「北京」"}},
            "required": ["city"],
        },
        kind=ToolKind.HTTP,
        side_effect=SideEffect.NETWORK,
        extra={"http": {"url": BASE + path, "method": "GET", "body": "query", **extra_http}},
    )


def _client(handler) -> HttpExecutor:
    return HttpExecutor(_http_def(), transport=httpx.MockTransport(handler))


def _live_ctx(**kw) -> ToolCtx:
    return ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY, **kw)


# ── 正常路径 ──────────────────────────────────────────────────────
def test_query_args_reach_the_url_and_the_body_parses_as_json():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"city": request.url.params.get("city"), "temp_c": 21})

    result = _client(handler).call("http_demo", {"city": "北京"}, _live_ctx())
    assert result.ok and not result.mocked
    assert result.output["status"] == 200
    assert result.output["body"] == {"city": "北京", "temp_c": 21}
    assert result.output["truncated"] is False
    # 参数进 query（GET 的默认 body 模式），URL 主机由定义决定、模型改不了
    assert seen["url"].startswith(BASE + "/weather?")
    assert "city=" in seen["url"]


def test_json_body_mode_posts_the_arguments():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["json"] = request.read().decode("utf-8")
        return httpx.Response(201, json={"id": 7})

    definition = _http_def("/create", method="POST", body="json")
    executor = HttpExecutor(definition, transport=httpx.MockTransport(handler))
    result = executor.call("http_demo", {"city": "北京"}, _live_ctx())
    assert result.ok and result.output["status"] == 201
    assert seen["method"] == "POST"
    assert '"city"' in str(seen["json"])


def test_non_2xx_is_a_tool_side_failure_with_the_status_in_detail():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream gone")

    result = _client(handler).call("http_demo", {"city": "北京"}, _live_ctx())
    assert not result.ok and result.error_kind == "error"
    assert result.extra["detail"]["kind"] == "http_status"
    assert result.extra["detail"]["status"] == 503
    assert "upstream gone" in result.extra["detail"]["body"]


def test_connection_error_is_distinguished_from_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    result = _client(handler).call("http_demo", {"city": "北京"}, _live_ctx())
    assert not result.ok and result.error_kind == "error"
    assert result.extra["detail"]["kind"] == "unreachable"


def test_body_is_truncated_instead_of_eating_the_context_window():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 200_000)

    result = _client(handler).call("http_demo", {"city": "北京"}, _live_ctx())
    assert result.ok
    assert result.output["truncated"] is True
    assert len(result.output["body"]) == 65_536


# ── deadline 传播：这一步真正要证明的东西 ─────────────────────────
def test_deadline_reaches_the_socket_layer():
    """httpx 把有效超时放在 `request.extensions["timeout"]`（已实测）。

    所以这里能直接检查"1ms 的 deadline 有没有交给 socket"，
    而不是只检查"超时之后我们有没有映射对"。
    """
    recorded: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded.append((request.extensions.get("timeout") or {}).get("read"))
        return httpx.Response(200, json={"ok": True})

    executor = _client(handler)
    executor.call("http_demo", {"city": "北京"}, _live_ctx(deadline_ms=1))
    assert recorded == [pytest.approx(0.001)]


def test_effective_timeout_is_the_minimum_of_every_constraint():
    """取最小值：任何一个更紧的约束都必须生效，取最大值等于让最松的说了算。"""
    definition = _http_def(timeout_ms=3000)
    assert _effective_timeout_ms(_read_config(definition), definition, _live_ctx()) == 3000
    assert _effective_timeout_ms(
        _read_config(definition), definition, _live_ctx(deadline_ms=50)
    ) == 50
    # 谁都没写时，落到策略默认（10s）而不是 http 的 15s——两个候选里更紧的那个赢。
    # 关键是**永远不会无限等待**：一个卡住的上游会永久占着线程池 worker
    loose = _http_def()
    assert _effective_timeout_ms(_read_config(loose), loose, _live_ctx()) == 10_000


def test_socket_timeout_maps_to_the_timeout_kind():
    """超时是**可重试**的，5xx 不是——两者混成一档就没法决定要不要重跑。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("socket read timed out", request=request)

    result = _client(handler).call("http_demo", {"city": "北京"}, _live_ctx(deadline_ms=1))
    assert not result.ok and result.error_kind == "timeout"
    assert result.extra["detail"]["kind"] == "http_timeout"


def test_unpropagated_deadline_is_detected_as_a_failure():
    """注入缺陷：执行器无视 ctx.deadline_ms，只用默认 15s。

    handler 检查 socket 层的实际超时，发现没传下来就返回 500，
    于是 error_kind 变成 error 而不是 timeout——契约断言随即变红。
    这正是 IMPLEMENTATION.md S11 要求"失败形态也要能复现"的那一条。
    """

    class DeadlineIgnoringHttp(HttpExecutor):
        def _send(self, clean, ctx):
            return super()._send(clean, ToolCtx(
                trace_id=ctx.trace_id, deadline_ms=None, dry_run=ctx.dry_run,
                mock_policy=ctx.mock_policy, fixtures=ctx.fixtures, replay=ctx.replay,
                policy=SandboxPolicy(
                    allowed_side_effects=PERMISSIVE_POLICY.allowed_side_effects,
                    require_approval=frozenset(), default_timeout_ms=15_000,
                ),
                now=ctx.now, extra=dict(ctx.extra),
            ))

    def handler(request: httpx.Request) -> httpx.Response:
        effective = (request.extensions.get("timeout") or {}).get("read")
        if effective is None or effective > 0.05:
            return httpx.Response(500, json={"error": "deadline 未传播", "read": effective})
        raise httpx.ReadTimeout("simulated", request=request)

    executor = DeadlineIgnoringHttp(_http_def("/slow"), transport=httpx.MockTransport(handler))
    result = executor.call("http_demo", {"city": "北京"}, _live_ctx(deadline_ms=1))
    assert not result.ok
    assert result.error_kind == "error", "缺陷没被检出——handler 的自检失效了"
    assert "deadline 未传播" in result.extra["detail"]["body"]

    # 对照：正常的执行器在同一条 handler 下必须报 timeout
    healthy = HttpExecutor(_http_def("/slow"), transport=httpx.MockTransport(handler))
    fixed = healthy.call("http_demo", {"city": "北京"}, _live_ctx(deadline_ms=1))
    assert fixed.error_kind == "timeout"


# ── 配置校验 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("extra_http,reason", [
    ({}, "missing_http_config"),
    ({"url": "ftp://x/y"}, "bad_url"),
    ({"url": "/relative"}, "bad_url"),
    ({"url": BASE, "method": "TRACE"}, "bad_method"),
    ({"url": BASE, "body": "form"}, "bad_body_mode"),
    ({"url": BASE, "headers": "not-a-dict"}, "bad_headers"),
])
def test_bad_http_config_fails_at_construction_not_at_call_time(extra_http, reason):
    """配置错必须在构造时就炸。否则错误会出现在某条 trace 里，看起来像模型的问题。"""
    definition = ToolDef(
        name="broken", description="配置非法的 http 工具，用来验证构造期校验",
        parameters={"type": "object", "properties": {}},
        kind=ToolKind.HTTP, side_effect=SideEffect.NETWORK, extra={"http": extra_http},
    )
    with pytest.raises(ToolUnknown) as exc:
        HttpExecutor(definition)
    assert exc.value.detail["kind"] == reason


def test_missing_http_key_entirely_is_also_a_construction_error():
    definition = ToolDef(name="nohttp", description="完全没有 extra.http 的定义",
                         kind=ToolKind.HTTP, side_effect=SideEffect.NETWORK)
    with pytest.raises(ToolUnknown) as exc:
        HttpExecutor(definition)
    assert exc.value.detail["kind"] == "missing_http_config"


# ── 沙箱与 mock ───────────────────────────────────────────────────
def test_network_side_effect_is_denied_by_the_default_policy():
    executor = _client(lambda request: httpx.Response(200, json={"ok": True}))
    result = executor.call(
        "http_demo", {"city": "北京"},
        ToolCtx(mock_policy=MockPolicy.LIVE, policy=SandboxPolicy()),
    )
    assert not result.ok and result.error_kind == "rejected"
    assert executor.real_calls == 0, "被拒绝的调用不许发出请求"


def test_fixture_policy_makes_zero_real_requests():
    """real_calls 是实测计数，不是声明——这条断言因此有牙。"""
    executor = _client(lambda request: httpx.Response(200, json={"ok": True}))
    ctx = ToolCtx(
        mock_policy=MockPolicy.FIXTURE, policy=PERMISSIVE_POLICY,
        fixtures={"http_demo": {"temp_c": 21}},
    )
    result = executor.call("http_demo", {"city": "北京"}, ctx)
    assert result.ok and result.mocked and result.output == {"temp_c": 21}
    assert executor.real_calls == 0


def test_http_executor_passes_the_full_contract_offline():
    target = contract_target()
    assert target is not None, "httpx 未安装，这一列应显示为未知而不是通过"
    sample, factory, valid_args, synth = target
    results = run_contracts(factory, sample, valid_args=valid_args, synth=synth)
    counts = summarize(results)
    assert counts["failed"] == 0, {i.name: i.detail for i in results if not i.passed}
    assert counts == {"passed": len(CONTRACT_NAMES), "failed": 0, "not_applicable": 0}


def test_http_contract_sample_never_hits_the_network():
    """contract_target 用的是 MockTransport + .invalid 域名，所以整条命令零外网。"""
    sample, factory, valid_args, _synth = contract_target()
    executor = factory(sample)
    assert isinstance(executor, HttpExecutor)
    assert executor._transport is not None
    assert executor.call(sample.name, dict(valid_args), _live_ctx()).ok
    assert executor.real_calls == 1  # 计数指的是"执行器发出的请求"，不是"打到外网"


# ── fs_read：路径穿越 ─────────────────────────────────────────────
@pytest.fixture
def tree(tmp_path):
    # newline="\n"：read_file 读的是原始字节，Windows 默认的 CRLF 翻译会让
    # 断言变成"在测平台的换行习惯"而不是在测工具
    (tmp_path / "notes.txt").write_text("第一行\n第二行\n", encoding="utf-8", newline="\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "deep.md").write_text("# 标题", encoding="utf-8", newline="\n")
    (tmp_path / "big.txt").write_text("y" * 5000, encoding="utf-8", newline="\n")
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("do not read", encoding="utf-8", newline="\n")
    return tmp_path, secret


def test_fs_read_returns_text_within_the_root(tree):
    root, _ = tree
    result = read_file("notes.txt", str(root))
    assert result["text"] == "第一行\n第二行\n"
    assert result["lines"] == 2 and result["truncated"] is False
    assert read_file("nested/deep.md", str(root))["text"] == "# 标题"


@pytest.mark.parametrize("path", [
    "../secret.txt",
    "../../secret.txt",
    "nested/../../secret.txt",
    "./nested/../../secret.txt",
])
def test_fs_read_blocks_traversal(tree, path):
    root, secret = tree
    with pytest.raises(ToolSandboxDenied) as exc:
        read_file(path, str(root))
    assert exc.value.detail["kind"] == "path_traversal"
    assert secret.read_text(encoding="utf-8") == "do not read"


def test_fs_read_blocks_absolute_paths_outside_the_root(tree):
    root, secret = tree
    with pytest.raises(ToolSandboxDenied):
        read_file(str(secret), str(root))


def test_fs_read_blocks_symlink_escape(tree):
    """resolve() 会展开符号链接，所以这条与 ../ 走同一个检查。"""
    root, secret = tree
    link = root / "link.txt"
    try:
        link.symlink_to(secret)
    except OSError:  # pragma: no cover - Windows 无权限建符号链接时跳过
        pytest.skip("当前环境不允许创建符号链接")
    with pytest.raises(ToolSandboxDenied):
        read_file("link.txt", str(root))


def test_fs_read_refuses_when_root_is_not_configured():
    """root 缺失是部署配置问题：报 rejected，不是 arg_error（那会指向模型）。"""
    with pytest.raises(ToolSandboxDenied) as exc:
        read_file("notes.txt", "")
    assert exc.value.detail["kind"] == "root_not_configured"


def test_fs_read_rejects_binary_and_missing_files(tree):
    root, _ = tree
    from onyx.core.errors import ToolArgError

    (root / "blob.png").write_bytes(b"\x89PNG")
    with pytest.raises(ToolArgError) as exc:
        read_file("blob.png", str(root))
    assert exc.value.detail["kind"] == "binary_file"
    with pytest.raises(ToolArgError) as exc:
        read_file("nope.txt", str(root))
    assert exc.value.detail["kind"] == "not_a_file"
    with pytest.raises(ToolArgError) as exc:
        read_file("nested", str(root))
    assert exc.value.detail["kind"] == "not_a_file"


def test_fs_read_truncates_large_files(tree):
    root, _ = tree
    result = read_file("big.txt", str(root), max_bytes=100)
    assert result["truncated"] is True
    assert result["returned_bytes"] == 100 and result["bytes"] == 5000


# ── constants：定义级、模型不可覆盖 ───────────────────────────────
FS_READ = ToolDef(
    name="fs_read",
    description="读取 root 之下的一个文本文件，越界会被沙箱拒绝",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对 root 的文件路径"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    kind=ToolKind.PYTHON_FN,
    side_effect=SideEffect.READ,
    impl_ref="onyx.tools.builtin.fs_read:read_file",
    extra={"constants": {"root": "PLACEHOLDER"}},
)


def test_constants_are_injected_and_the_model_cannot_override_them(tree):
    root, secret = tree
    definition = ToolDef(
        name=FS_READ.name, description=FS_READ.description, parameters=FS_READ.parameters,
        kind=FS_READ.kind, side_effect=FS_READ.side_effect, impl_ref=FS_READ.impl_ref,
        extra={"constants": {"root": str(root)}},
    )
    executor = PythonFnExecutor(definition)
    ok = executor.call("fs_read", {"path": "notes.txt"}, _live_ctx())
    assert ok.ok and "第一行" in ok.output["text"]

    # additionalProperties=False ⇒ 模型连试都试不了
    denied = executor.call("fs_read", {"path": "notes.txt", "root": "/"}, _live_ctx())
    assert not denied.ok and denied.error_kind == "arg_error"
    assert denied.extra["detail"]["kind"] == "unexpected_field"
    assert secret.exists()


def test_exposing_a_constant_in_the_schema_is_reported_as_a_definition_defect(tree):
    """同名即缺陷：静默让常量覆盖参数会把配置错误藏到运行时。"""
    root, _ = tree
    leaky = ToolDef(
        name="fs_read", description=FS_READ.description,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "相对 root 的文件路径"},
                "root": {"type": "string", "description": "允许根目录（不该暴露给模型）"},
            },
            "required": ["path"],
        },
        kind=ToolKind.PYTHON_FN, side_effect=SideEffect.READ,
        impl_ref="onyx.tools.builtin.fs_read:read_file",
        extra={"constants": {"root": str(root)}},
    )
    result = PythonFnExecutor(leaky).call("fs_read", {"path": "notes.txt"}, _live_ctx())
    assert not result.ok and result.error_kind == "error"
    assert result.extra["detail"]["kind"] == "constant_exposed"
    assert result.extra["detail"]["fields"] == ["root"]


def test_constants_must_be_an_object():
    bad = ToolDef(
        name="fs_read", description=FS_READ.description, parameters=FS_READ.parameters,
        kind=ToolKind.PYTHON_FN, impl_ref=FS_READ.impl_ref, extra={"constants": ["root"]},
    )
    result = PythonFnExecutor(bad).call("fs_read", {"path": "x"}, _live_ctx())
    assert not result.ok and result.extra["detail"]["kind"] == "bad_constants"


# ── 分发 ──────────────────────────────────────────────────────────
def test_http_is_registered_and_only_builtin_tools_remain_pending():
    assert EXECUTOR_KINDS == ("fixture", "http", "mcp", "python_fn")
    assert isinstance(executor_for(_http_def()), HttpExecutor)
    assert isinstance(executor_for(_http_def(), kind="fixture").spec(), ToolDef)
    # mcp 在 S16d 落地；还剩引擎内建工具，它需要先有 P21 的实测结论才知道怎么调
    for pending in ("ollama_builtin",):
        definition = ToolDef(name="x", description="y" * 30, kind=ToolKind(pending))
        with pytest.raises(ToolUnknown, match="S16"):
            executor_for(definition)
