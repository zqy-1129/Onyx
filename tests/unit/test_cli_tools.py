"""S11 验收：CLI 侧的工具执行与契约矩阵。

这些测试同时是"自测命令"的可执行版本——IMPLEMENTATION.md 里写的
`onyx tools contract` / `onyx tools run` 必须真的能跑出预期输出，
否则文档里的验收步骤就是假的。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from onyx.cli import app

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "tools.yaml"

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """所有命令都落在临时数据目录，绝不碰仓库里的 .data。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


def _import_builtin():
    result = _run("tools", "import", "--builtin")
    assert result.exit_code == 0, result.output
    return result


# ── import / ls / audit ───────────────────────────────────────────
def test_import_builtin_registers_the_reference_tools():
    result = _import_builtin()
    for name in ("echo", "calculator", "time_now"):
        assert name in result.output
    assert "已导入 3 个工具" in result.output

    listed = _run("tools", "ls")
    assert listed.exit_code == 0
    for name in ("echo", "calculator", "time_now"):
        assert name in listed.output


def test_import_requires_a_file_or_the_builtin_flag():
    result = _run("tools", "import")
    assert result.exit_code == 2
    assert "--builtin" in result.output


def test_builtin_definitions_pass_audit():
    _import_builtin()
    result = _run("tools", "audit")
    assert result.exit_code == 0, result.output
    assert "error=0 warn=0 info=0" in result.output


# ── contract ──────────────────────────────────────────────────────
def test_contract_matrix_covers_every_implemented_executor():
    result = _run("tools", "contract")
    assert result.exit_code == 0, result.output
    for implemented in ("python_fn", "mock", "http"):
        assert implemented in result.output
    # 未实现的执行器必须**显式列出**并写明落地里程碑，不能悄悄从矩阵里消失
    for pending in ("mcp", "ollama_builtin"):
        assert pending in result.output
    assert "S16" in result.output
    assert "失败 0" in result.output
    # 每条 n/a 都必须带原因
    assert "无法证明零真实调用" in result.output
    assert "deadline 无从生效" in result.output
    # http 列用的是离线样本，这一点必须写出来，否则读者会以为它打了真实网络
    assert "离线 MockTransport" in result.output


def test_contract_can_use_a_registered_tool_as_sample():
    """已注册工具没有内置的合法参数，必须由 examples/schema 自动构造。"""
    assert _run("tools", "import", str(EXAMPLES)).exit_code == 0
    result = _run("tools", "contract", "--tool", "calculator")
    assert result.exit_code == 0, result.output
    # 样本出处那一节会把定义从哪来写清楚（注册表覆盖内置时尤其重要）
    assert "calculator （注册表/内置）" in result.output


def test_registered_definition_wins_over_the_builtin_of_the_same_name(tmp_path):
    """出处必须是注册表：用户显式导入的那份才是模型实际会看到的定义。

    若内置定义抢先命中，`tools run echo` 会执行与注册版本不同的实现——
    这正是"数字对不上出处"要避免的情况。
    """
    _import_builtin()
    override = tmp_path / "echo_override.yaml"
    override.write_text(
        """
tools:
  - name: echo
    description: 覆盖内置 echo 的注册版本，用来验证解析顺序是注册表优先。
    kind: python_fn
    side_effect: read
    impl_ref: onyx.tools.builtin.echo:echo
    parameters:
      type: object
      properties:
        text:
          type: string
          description: 需要回显的文本内容，这里刻意与内置版本不同以便区分。
      required: [text]
      additionalProperties: false
    examples:
      - instruction: 回显一句来自注册表的话
        expect:
          name: echo
          arguments: { text: 来自注册表 }
""",
        encoding="utf-8",
    )
    assert _run("tools", "import", str(override)).exit_code == 0

    payload = json.loads(_run("tools", "contract", "--tool", "echo", "--json").output)
    assert payload["source"] == "registry"
    assert payload["valid_args"] == {"text": "来自注册表"}
    assert payload["samples"]["python_fn"] == "echo"
    assert payload["summary"]["python_fn"]["failed"] == 0

    run = _run("tools", "run", "echo", "--args", '{"text": "来自注册表"}')
    assert run.exit_code == 0, run.output
    assert "出处=registry" in run.output


def test_contract_rejects_an_unknown_tool():
    result = _run("tools", "contract", "--tool", "does_not_exist")
    assert result.exit_code == 2
    assert "找不到工具" in result.output


def test_contract_json_output_is_machine_readable():
    result = _run("tools", "contract", "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tool"] == "echo"
    assert payload["source"] == "builtin"
    assert set(payload["executors"]) == {"python_fn", "mock", "http", "mcp"}
    assert len(payload["assertions"]) == 8
    assert payload["summary"]["python_fn"] == {"passed": 7, "failed": 0, "not_applicable": 1}
    # http 列有自己的离线样本，8 条断言全部适用
    assert payload["samples"]["http"] == "contract_http"
    assert payload["summary"]["http"] == {"passed": 8, "failed": 0, "not_applicable": 0}
    # mcp 列同样自带离线样本（假连接，不起子进程）：契约矩阵不许依赖外部 server，
    # 否则这条命令变成"装了东西才跑得动"，就没人经常跑了
    assert payload["samples"]["mcp"] == "contract_mcp"
    assert payload["summary"]["mcp"] == {"passed": 8, "failed": 0, "not_applicable": 0}


def test_pending_kinds_are_derived_not_hardcoded():
    """"还剩哪些执行器没实现"必须从注册表推导。

    写在 CLI 里的版本会在实现完成后继续宣称"未实现"，而那句话看起来永远合理——
    S16d 落地 mcp 之后，矩阵里那一行就该自动消失。
    """
    from onyx.tools.executors import EXECUTOR_KINDS, PENDING_KINDS

    assert "mcp" in EXECUTOR_KINDS and "mcp" not in PENDING_KINDS
    assert set(PENDING_KINDS) == {"ollama_builtin"}
    assert all(v for v in PENDING_KINDS.values()), "每个未实现的种类都要写明计划与前置探针"
    result = _run("tools", "contract")
    assert "mcp] 未实现" not in result.output
    assert "[ollama_builtin] 未实现" in result.output


# ── run ───────────────────────────────────────────────────────────
def test_run_echo_returns_structured_success():
    result = _run("tools", "run", "echo", "--args", '{"text": "hi", "times": 2}')
    assert result.exit_code == 0, result.output
    assert "✓ ok" in result.output
    assert '"chars": 2' in result.output
    assert "耗时" in result.output


def test_run_reports_arg_error_with_a_nonzero_exit():
    result = _run("tools", "run", "echo", "--args", "{}")
    assert result.exit_code == 1
    assert "arg_error" in result.output
    assert "缺少必填参数" in result.output


def test_run_surfaces_the_calculator_whitelist_rejection():
    """RCE 尝试必须是 arg_error 且带 kind，不能是崩溃也不能被执行。"""
    result = _run(
        "tools", "run", "calculator", "--args", '{"expr": "__import__(\'os\').system(\'id\')"}'
    )
    assert result.exit_code == 1
    assert "arg_error" in result.output
    assert "unsafe_expression" in result.output


def test_run_deny_policy_skips_without_executing():
    result = _run("tools", "run", "echo", "--args", '{"text": "hi"}', "--mock", "deny")
    assert result.exit_code == 1
    assert "skipped" in result.output
    assert "mocked" in result.output


def test_run_fixture_policy_returns_the_stub():
    result = _run(
        "tools", "run", "echo", "--args", '{"text": "hi"}',
        "--mock", "fixture", "--fixture", '{"echo": "stub", "times": 1, "chars": 4}',
    )
    assert result.exit_code == 0, result.output
    assert "stub" in result.output and "mocked" in result.output


def test_run_dry_run_allows_read_tools():
    result = _run("tools", "run", "time_now", "--args", '{"fmt": "date"}', "--dry-run")
    assert result.exit_code == 0, result.output
    assert "✓ ok" in result.output


def test_run_rejects_an_invalid_allow_value():
    result = _run("tools", "run", "echo", "--allow", "bogus")
    assert result.exit_code == 2
    assert "--allow" in result.output


def test_sandbox_denies_network_until_explicitly_allowed():
    """默认策略只放 read。这两条一起看才说明 --allow 真的在起作用。"""
    assert _run("tools", "import", str(EXAMPLES)).exit_code == 0

    denied = _run("tools", "run", "get_weather", "--args", '{"city": "北京"}')
    assert denied.exit_code == 1
    assert "rejected" in denied.output

    # 放开 network 之后策略不再是拦路虎，但示例的 impl 并不存在——
    # 这时必须报 error（工具缺陷），而不是又变回 rejected
    allowed = _run("tools", "run", "get_weather", "--args", '{"city": "北京"}', "--allow", "network")
    assert allowed.exit_code == 1
    assert "error" in allowed.output
    assert "无法导入" in allowed.output


def test_run_rejects_an_unknown_tool():
    result = _run("tools", "run", "nope")
    assert result.exit_code == 2
    assert "找不到工具" in result.output


# ── cost ──────────────────────────────────────────────────────────
def test_cost_reports_the_template_share_warning():
    _import_builtin()
    result = _run("tools", "cost", "--overhead", "291")
    assert result.exit_code == 0, result.output
    assert "每次请求实付" in result.output
    assert "模板占比" in result.output


# ── fire（模型侧 fire-and-verify）─────────────────────────────────
FIRE = ["tools", "fire", "把 hello 原样回显一次", "--model", "mock/echo",
        "--provider", "mock", "--tools", "echo"]


def test_fire_derives_the_expectation_from_the_tool_examples():
    _import_builtin()
    result = _run(*FIRE)
    assert result.exit_code == 1, result.output
    # 期望参数来自 examples[0]，不需要用户再手抄一遍
    assert '{"text": "hello", "times": 1}' in result.output
    # mock provider 的默认剧本不调工具，所以判定必须是 NO_CALL 而不是 PASS
    assert "NO_CALL" in result.output
    assert "没有发起任何工具调用" in result.output
    assert "停止原因: final" in result.output


def test_fire_accepts_an_explicit_expectation():
    _import_builtin()
    result = _run(*FIRE, "--expect", "echo", "--expect-args", '{"text": "x"}')
    assert result.exit_code == 1
    assert '{"text": "x"}' in result.output


def test_fire_rejects_an_expectation_outside_the_tool_set():
    _import_builtin()
    result = _run(*FIRE, "--expect", "calculator")
    assert result.exit_code == 2
    assert "不在本次工具集里" in result.output


def test_fire_requires_examples_or_explicit_args(tmp_path):
    """没有 examples 又没有 --expect-args 时必须说清楚怎么修，而不是给个空期望。"""
    bare = tmp_path / "bare.yaml"
    bare.write_text(
        """
tools:
  - name: bare_tool
    description: 一个没有 examples 的工具，用来验证 fire 的错误提示是否可行动。
    kind: python_fn
    side_effect: read
    impl_ref: onyx.tools.builtin.echo:echo
    parameters:
      type: object
      properties:
        text:
          type: string
          description: 需要回显的文本内容，这里只是为了让 schema 合法。
      required: [text]
      additionalProperties: false
""",
        encoding="utf-8",
    )
    assert _run("tools", "import", str(bare)).exit_code == 0
    result = _run("tools", "fire", "随便说点什么", "--model", "mock/echo",
                  "--provider", "mock", "--tools", "bare_tool")
    assert result.exit_code == 2
    assert "NO_EXAMPLE" in result.output
    assert "--expect-args" in result.output


def test_fire_needs_at_least_one_tool():
    result = _run("tools", "fire", "回显", "--model", "mock/echo", "--provider", "mock")
    assert result.exit_code == 2
    assert "tools import --builtin" in result.output


def test_fire_json_output_is_machine_readable():
    _import_builtin()
    result = _run(*FIRE, "--json")
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["verdict"] == "no_call"
    assert payload["expected"] == {"tool": "echo", "arguments": {"text": "hello", "times": 1},
                                  "exact": False}
    assert payload["called"] == []
    assert payload["stop_reason"] == "final"
    assert payload["mocked"] is None


def test_fire_runs_under_deny_policy_without_touching_real_tools():
    """deny 是纯观测档：即使模型真要调工具，也只记 skipped，不产生任何副作用。

    mock provider 的默认剧本不发起调用，所以这里断言的是"命令能跑通且判定明确"；
    "用桩跑出的 PASS 必须标 mocked"由 tests/unit/test_verify.py 覆盖。
    """
    _import_builtin()
    result = _run(*FIRE, "--mock", "deny", "--json")
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["verdict"] == "no_call"
    assert payload["called"] == []
