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
    assert set(payload["executors"]) == {"python_fn", "mock", "http"}
    assert len(payload["assertions"]) == 8
    assert payload["summary"]["python_fn"] == {"passed": 7, "failed": 0, "not_applicable": 1}
    # http 列有自己的离线样本，8 条断言全部适用
    assert payload["samples"]["http"] == "contract_http"
    assert payload["summary"]["http"] == {"passed": 8, "failed": 0, "not_applicable": 0}
    assert payload["pending"] == {"mcp": "S16", "ollama_builtin": "S16"}


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
