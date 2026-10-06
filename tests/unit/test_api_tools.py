"""S25 验收：Tool Bench 的读端点。

盯三件事：
- **与 CLI 同源**：注册表装配与契约矩阵都走同一个构建处，界面不许自己算一套开销
  （那个量要和 trace 归因的 `part=tool_defs` 对齐）。
- **未核算 ≠ 0 token**：`tokens=None` 表示还没算过，界面上会显示「—」；填 0 就是造一个假数据。
- **矩阵里"未知"与"通过"必须能区分**：装不上的执行器进 `unavailable` 并带原因，
  没实现的进 `pending`，两者都不许出现在 ✓ 里。
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.runtime import build_runtime, build_tool_registry
from onyx.settings import load_settings
from onyx.store.records import ToolRunRecord
from onyx.store.repos import ToolRepo
from onyx.tools.spec import ToolDef

SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "要回显的文本内容"}},
    "required": ["text"],
    "additionalProperties": False,
}


def _def(name: str = "echo", *, description: str = "", **kw) -> ToolDef:
    return ToolDef(
        name=name,
        description=description or f"{name} 的用途说明，足够长以免触发描述过短的审计规则。",
        kind=kw.pop("kind", "python_fn"),
        side_effect=kw.pop("side_effect", "read"),
        impl_ref=kw.pop("impl_ref", "onyx.tools.builtin.echo:echo"),
        parameters=kw.pop("parameters", SCHEMA),
        examples=kw.pop("examples", [{"instruction": "回显一句话",
                                      "expect": {"name": name, "arguments": {"text": "hi"}}}]),
        **kw,
    )


@pytest.fixture
def app(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
    )
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")) as client:
        client.runtime = runtime
        yield client
    runtime.close()


def _register(runtime, *definitions: ToolDef) -> None:
    registry = build_tool_registry(runtime.db, provider_id="mock-local")
    registry.register_many(definitions if definitions else (_def(),))


# ── 注册表 ────────────────────────────────────────────────────────
def test_empty_registry_is_empty_not_zero_tools(app):
    assert app.get("/api/tools").json() == []


def test_registered_tools_show_version_costs_and_hash(app):
    client = app
    _register(client.runtime, _def("echo"), _def("weather_query", kind="http",
                                           side_effect="read", impl_ref=""))
    rows = {row["name"]: row for row in client.get("/api/tools").json()}
    assert set(rows) == {"echo", "weather_query"}
    echo = rows["echo"]
    assert echo["version"] >= 1 and echo["kind"] == "python_fn"
    assert echo["side_effect"] == "read", "副作用位必须看得见：它是沙箱白名单的依据"
    assert echo["hash"], "定义要有内容 hash 版本，否则无法知道模型看到的是哪一份"
    assert echo["n_examples"] == 1
    # 没跑过 refresh_costs 时是 None（"未核算"），不是 0
    assert echo["tokens"] is None or isinstance(echo["tokens"], int)
    assert rows["weather_query"]["kind"] == "http"


def test_enabled_and_kind_filters_are_honoured(app):
    client = app
    _register(client.runtime, _def("echo"), _def("weather_query", kind="http", impl_ref=""))
    assert len(client.get("/api/tools", params={"kind": "http"}).json()) == 1
    assert len(client.get("/api/tools", params={"enabled_only": True}).json()) == 2

    repo = ToolRepo(client.runtime.db)
    repo.set_enabled(repo.find_by_name("echo").id, False)
    disabled = client.get("/api/tools", params={"enabled_only": True}).json()
    assert [row["name"] for row in disabled] == ["weather_query"], \
        "禁用位要能被过滤，否则「启用了哪些」这个问题没有答案"
    still_listed = client.get("/api/tools").json()
    assert len(still_listed) == 2, "不启用不等于不存在：历史版本行要能查得到"


# ── 审计 ──────────────────────────────────────────────────────────
def test_audit_reports_findings_with_their_fix(app):
    client = app
    # 三种各有其害的形态：参数没描述（模型只能靠名字猜）、描述过短（等于没写）、
    # 没标注副作用（沙箱无法决策）
    no_param_desc = _def("bare", parameters={
        "type": "object", "properties": {"text": {"type": "string"}},
        "required": ["text"], "additionalProperties": False,
    })
    terse = _def("terse", description="回显")
    untagged = _def("untagged", extra={"side_effect_untagged": True})
    _register(client.runtime, no_param_desc, terse, untagged)

    body = client.get("/api/tools/audit").json()
    by_rule = {item["rule"] for item in body["findings"]}
    assert "DESC_MISSING" in by_rule and "DESC_TOO_SHORT" in by_rule, by_rule
    for item in body["findings"]:
        assert item["tool"] and item["message"]
        assert item["fix"], "只报问题不报告修法是半成品"
        assert item["meaning"], "规则说明与 CLI 共用同一份文案"
    assert sum(body["counts"].values()) == len(body["findings"])
    # 未标注副作用是 error 级：沙箱靠它决策，默认当成只读等于把 write/exec 工具放行
    assert body["counts"]["error"] > 0 and "SIDE_EFFECT_UNTAGGED" in by_rule
    assert body["note"], "要声明这是现算的，不是历史快照"


def test_audit_of_a_good_definition_is_clean(app):
    client = app
    _register(client.runtime, _def("echo"))
    body = client.get("/api/tools/audit").json()
    assert body["counts"]["error"] == 0, body["findings"]


# ── 开销 ──────────────────────────────────────────────────────────
def test_cost_separates_json_from_template_scaffold(app):
    """P17：只报 JSON 大小会把优化方向引到"精简描述"，而实测大头来自模板注入。"""
    client = app
    _register(client.runtime, _def("echo"))

    plain = client.get("/api/tools/cost").json()
    assert plain["json_tokens"] > 0
    assert plain["count_source"] == "heuristic", "没指定模型时不能假装是标定档"
    assert plain["hint"], "heuristic 档必须自带一句提醒"
    assert plain["template_overhead_tokens"] == 0
    assert plain["template_share"] is None, "没传开销时不编造占比"

    with_scaffold = client.get("/api/tools/cost", params={"overhead": 220}).json()
    assert with_scaffold["effective_tokens"] == with_scaffold["json_tokens"] + 220
    assert with_scaffold["template_share"] > 0.5, "模板占比要真的算出来"
    assert with_scaffold["tools"][0]["tokens"] >= 0
    # 开销排序按 token 降序：界面第一行就该是"最贵的工具"
    assert with_scaffold["tools"][0]["name"] == "echo"


# ── 契约矩阵 ──────────────────────────────────────────────────────
def test_matrix_shape_matches_the_cli_json(app):
    client = app
    _register(client.runtime, _def("echo"))
    body = client.get("/api/tools/matrix").json()
    assert body["tool"] == "echo"
    assert body["source"] in ("registry", "builtin")
    assert len(body["assertions"]) == 8
    #: 端点与 CLI 必须是同一张矩阵（含 S35 的真子进程第五列）——两处列数不同，
    #: 就说明有人给界面单独搭了一套"更容易通过"的样本。
    assert set(body["executors"]) == {"python_fn", "mock", "http", "mcp", "mcp_stdio"}
    assert body["samples"]["http"] == "contract_http", "每列测的定义要报得出出处"
    assert body["sample_notes"]["http"]
    assert body["failed"] == 0, body["executors"]
    assert "ollama_builtin" in body["pending"], "没实现的执行器种类要显式列出"
    assert json.dumps(body, ensure_ascii=False), "形状必须能直接进 JSON 响应"


def test_registered_definition_drives_the_matrix_sample_args(app):
    """覆盖内置的同名工具时，参数必须取自注册表那一份——否则测的不是模型实际看到的定义。"""
    client = app
    _register(client.runtime, _def("echo", description="注册表里的覆盖版，用来验证参数取自哪一份定义。",
                                   examples=[{"instruction": "回显一句来自注册表的话",
                                              "expect": {"name": "echo",
                                                        "arguments": {"text": "来自注册表"}}}]))
    body = client.get("/api/tools/matrix").json()
    assert body["source"] == "registry"
    assert body["valid_args"] == {"text": "来自注册表"}


def test_matrix_rejects_bad_arguments(app):
    client = app
    assert client.get("/api/tools/matrix", params={"tool": "does_not_exist"}).status_code == 404
    assert client.get("/api/tools/matrix", params={"args": "[1,2]"}).status_code == 422
    assert client.get("/api/tools/matrix", params={"args": "{oops"}).status_code == 422
    assert client.get("/api/tools/matrix", params={"args": "x" * 9000}).status_code == 413


def test_matrix_is_readable_on_a_read_only_board(tmp_path):
    """Tool Bench 全是读端点：只读看板也该能看审计与矩阵，否则共享的人只能自己开 CLI。"""
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "ro.sqlite", event_log=False,
    )
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "g.lock",
                               token="t0ken", read_only=True)) as client:
        client.headers.update({"Authorization": "Bearer t0ken"})
        for path in ("/api/tools", "/api/tools/audit", "/api/tools/cost",
                     "/api/tools/matrix", "/api/tools/runs"):
            assert client.get(path).status_code == 200, path


# ── 运行历史 ──────────────────────────────────────────────────────
def test_runs_carry_the_trace_they_came_from(app):
    """每个 fire/run 结果都要点得回那次真实请求，否则"工具跑成什么样"又是一句无据之言。"""
    client = app
    repo = ToolRepo(client.runtime.db)
    repo.insert_run(ToolRunRecord(
        id="run-1", tool_id="tool-def-1", tool_def_hash="sha256:aa", test_id="t1",
        trace_id="trace-1", started_at="2026-10-04T00:00:00+00:00", status="ok",
        latency_ms=12.5, deterministic=True, idempotent=True,
    ))
    repo.insert_run(ToolRunRecord(
        id="run-2", tool_id="tool-def-1", started_at="2026-10-04T00:01:00+00:00",
        status="timeout", error="deadline 12ms 用完",
    ))

    rows = client.get("/api/tools/runs").json()
    assert [row["id"] for row in rows] == ["run-2", "run-1"], "新的在前"
    ok = next(row for row in rows if row["id"] == "run-1")
    assert ok["trace_id"] == "trace-1" and ok["latency_ms"] == 12.5
    assert ok["output_ref"] is None, "没有引用就是没有，不许显示空串"
    bad = next(row for row in rows if row["id"] == "run-2")
    assert bad["status"] == "timeout" and "deadline" in bad["error"]
    assert bad["deterministic"] is None, "没测过 ≠ 不确定"

    assert len(client.get("/api/tools/runs", params={"tool_id": "nope"}).json()) == 0
    assert len(client.get("/api/tools/runs", params={"limit": 1}).json()) == 1
