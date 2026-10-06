"""契约矩阵的构建处（`onyx/tools/matrix.py`）。

这张矩阵是"执行器可替换"这句话的唯一证据，所以它的形状与措辞都要能被测试钉住：
CLI 打印的、`--json` 输出的与 `GET /api/tools/matrix` 返回的必须来自同一个构建处。
"""

from __future__ import annotations

import pytest

from onyx.core.errors import ToolUnknown
from onyx.tools.contract import CONTRACT_NAMES
from onyx.tools.matrix import (
    MOCK_CONTRACT_EXEMPTIONS,
    SAMPLE_SOURCE_NOTE,
    build_matrix,
    default_valid_args,
)


@pytest.fixture
def echo_def():
    from onyx.tools.builtin.defs import builtin_def

    definition = builtin_def("echo")
    assert definition is not None
    return definition


def test_matrix_covers_every_implemented_column(echo_def):
    matrix = build_matrix(echo_def, source="builtin")
    #: `mcp_stdio` 是 S35 加的第五列：真子进程 + 真管道。它的存在本身就是要被断言的——
    #: "只有离线假连接"这件事曾经被写成 ROADMAP 里的缺口，因为矩阵不会为它红。
    assert set(matrix.columns) == {"python_fn", "mock", "http", "mcp", "mcp_stdio"}
    assert matrix.failed == 0, [row for row in matrix.failures()]
    # 每列都要有全部 8 条断言的结果，缺一条就是"看起来全过"
    for column in matrix.columns:
        assert set(matrix.results[column]) == set(CONTRACT_NAMES)


def test_as_dict_is_the_same_shape_the_cli_prints(echo_def):
    """`--json` 与 API 共用这个形状：两处各造一份就会在某个时刻报出两个版本的矩阵。"""
    payload = build_matrix(echo_def, source="builtin").as_dict()
    assert payload["tool"] == "echo" and payload["source"] == "builtin"
    assert payload["assertions"] == list(CONTRACT_NAMES)
    assert set(payload["executors"]) == set(payload["samples"])
    # samples 保持"列名 → 样本定义名"的原始形状，出处说明另放 sample_notes
    assert payload["samples"]["http"] == "contract_http"
    assert payload["sample_notes"]["http"] == SAMPLE_SOURCE_NOTE["http"]
    assert payload["summary"]["python_fn"]["failed"] == 0
    assert "ollama_builtin" in payload["pending"], "没实现的执行器种类要显式列出"


def test_registered_sample_wins_over_the_builtin_echo_alias(echo_def):
    """echo 的专用样本参数只在**内置**定义上用。

    注册表里覆盖的同名 echo 必须用它自己的 examples，否则矩阵测的是内置定义，
    而报告说的是注册版本——"数字对不上出处"就是这个意思。
    """
    from onyx.tools.spec import ToolDef

    registered = ToolDef(
        name="echo", description="注册表里的覆盖版，用来验证参数取自哪一份定义。",
        kind="python_fn", side_effect="read",
        impl_ref="onyx.tools.builtin.echo:echo",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的内容，与内置版不同。"}},
            "required": ["text"], "additionalProperties": False,
        },
        examples=[{"instruction": "回显一句来自注册表的话",
                   "expect": {"name": "echo", "arguments": {"text": "来自注册表"}}}],
    )
    assert default_valid_args(echo_def, "builtin") == {"text": "onyx", "times": 1}
    assert default_valid_args(registered, "registry") == {"text": "来自注册表"}

    matrix = build_matrix(registered, source="registry")
    assert matrix.valid_args == {"text": "来自注册表"}
    assert matrix.samples["python_fn"] == "echo"


def test_unknown_definition_lists_the_builtin_options():
    with pytest.raises(ToolUnknown) as exc:
        build_matrix(None, tool_label="nope")
    assert "找不到工具" in exc.value.message
    assert "echo" in exc.value.message, "报错要给出可选项"
    assert exc.value.code == "TOOL_UNKNOWN"


def test_stdio_column_is_hermetic_and_its_source_is_stated(echo_def):
    """这一列必须自己说清"我起的是真进程"，否则读矩阵的人会以为它和 mcp 列是同一件事。"""
    matrix = build_matrix(echo_def, source="builtin")
    column = matrix.results["mcp_stdio"]
    assert all(item.passed for item in column.values()), [
        (name, item.detail) for name, item in column.items() if not item.passed]
    #: 这一列一条豁免都不许有。豁免在这本项目里是"结构上做不到"的意思，
    #: 而真子进程做得到——给它加 n/a 就是把它降格成"另一份假连接"，还是起了进程的那种。
    assert not [name for name, item in column.items() if not item.applicable], \
        "真 stdio 列出现不适用格子：要么是真缺陷，要么是在用豁免遮"
    assert matrix.samples["mcp_stdio"] == "reference__weather", "样本该来自真发现，不是手写定义"
    assert matrix.as_dict()["sample_notes"]["mcp_stdio"] == "（真子进程 + 真管道）"
    assert SAMPLE_SOURCE_NOTE["mcp_stdio"] == "（真子进程 + 真管道）"


def test_a_stdio_server_that_cannot_start_is_unknown_not_passed(echo_def, monkeypatch):
    """起不来就是"未知"：矩阵不许因为环境问题崩掉，也不许把没测过的格子留成 ✓。

    这一列跑的是真子进程，所以在只读沙箱、没有可写 temp、或打包成 single-file 的环境里
    都可能起不来。那些情况下**其余四列仍然要能给出结论**，而这一列必须写明为什么没测。
    """
    import onyx.tools.matrix as matrix_module

    def _boom():
        raise OSError("[WinError 2] 系统找不到指定的文件")

    monkeypatch.setattr(matrix_module, "stdio_contract_target", _boom)
    matrix = build_matrix(echo_def, source="builtin")
    assert "mcp_stdio" not in matrix.columns
    assert "mcp_stdio" in matrix.unavailable
    reason = matrix.unavailable["mcp_stdio"]
    assert "未知，不是通过" in reason and "WinError" in reason
    assert set(matrix.columns) == {"python_fn", "mock", "http", "mcp"}, "其余列不许被牵连"


def test_a_timed_out_stdio_call_does_not_poison_the_next_one(echo_def):
    """真进程列曾经抓到的东西：一笔超时之后，被放弃的线程还在读管道，会偷走下一笔的回答。

    离线假连接测不到这件事（它同步回答），所以这条断言在第五列出现前是**空白**。
    """
    import onyx.tools.matrix as matrix_module

    sample, factory, args, synth, close = matrix_module.stdio_contract_target()
    from onyx.tools.contract import run_contracts
    try:
        results = {item.name: item for item in run_contracts(factory, sample, valid_args=args,
                                                            synth=synth)}
    finally:
        close()
    assert results["timeout_is_reported"].passed, "慢工具必须先真的超时"
    assert results["read_is_idempotent"].passed, (
        f"超时之后的调用被污染了：{results['read_is_idempotent'].detail}")


def test_missing_httpx_is_unknown_not_passed(echo_def, monkeypatch):
    """装不上 httpx 时那一列必须显示"未知"。当成通过等于把缺依赖写成质量保证。"""
    import onyx.tools.matrix as matrix_module

    monkeypatch.setattr(matrix_module, "http_contract_target", lambda: None)
    payload = build_matrix(echo_def, source="builtin").as_dict()
    assert "http" not in payload["executors"]
    assert "httpx" in payload["unavailable"]["http"]
    assert "未知，不是通过" in payload["unavailable"]["http"]
    # 未知不能同时出现在 pending 里：同一列既"没装"又"没实现"就没人知道该信哪个
    assert "http" not in payload["pending"]


def test_every_exemption_carries_a_reason(echo_def):
    """n/a 不许静默：每条豁免都要写明为什么，并指出谁替代它保证这件事。"""
    assert MOCK_CONTRACT_EXEMPTIONS
    for assertion, reason in MOCK_CONTRACT_EXEMPTIONS.items():
        assert assertion in CONTRACT_NAMES
        assert len(reason) > 20

    matrix = build_matrix(echo_def, source="builtin")
    not_applicable = [row for row in matrix.failures() if row["status"] == "not_applicable"]
    assert any(row["executor"] == "mock" for row in not_applicable)
    assert any("deadline 无从生效" in row["detail"] for row in not_applicable)


def test_pending_kinds_never_claim_an_implemented_executor(echo_def):
    from onyx.tools.executors import EXECUTOR_KINDS, PENDING_KINDS

    payload = build_matrix(echo_def, source="builtin").as_dict()
    assert set(payload["pending"]) <= set(PENDING_KINDS)
    for kind in payload["pending"]:
        assert kind not in EXECUTOR_KINDS, f"{kind} 已实现却还宣称未实现"
        assert payload["pending"][kind]
