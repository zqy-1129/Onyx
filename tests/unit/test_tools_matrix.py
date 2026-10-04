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
    assert set(matrix.columns) == {"python_fn", "mock", "http", "mcp"}
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
