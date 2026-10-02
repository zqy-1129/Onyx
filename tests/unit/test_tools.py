"""S10 验收：工具定义、契约审计、版本化、上下文开销核算。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from onyx.store.db import Database
from onyx.store.repos import ToolRepo
from onyx.tools.registry import ToolRegistry, defs_from_payload
from onyx.tools.spec import (
    Severity,
    SideEffect,
    ToolDef,
    ToolKind,
    audit,
    audit_many,
    content_hash,
    cost_report,
    openai_json,
)

GOOD = ToolDef(
    name="get_weather",
    description="查询指定城市的当前天气，包括温度与天气状况。仅在用户询问实时天气时使用。",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名称，例如「北京」或「Beijing」。"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    side_effect=SideEffect.NETWORK,
    examples=({"instruction": "北京天气", "expect": {"name": "get_weather"}},),
)


def _rules(findings) -> set[str]:
    return {f.rule for f in findings}


# ── 定义与 hash ────────────────────────────────────────────────────
def test_good_definition_passes_audit():
    findings = audit(GOOD, count_fn=len)
    assert _rules(findings) <= {"ADDITIONAL_PROPS_UNSET"}, f"意外告警: {findings}"
    assert not any(f.severity is Severity.ERROR for f in findings)


def test_openai_json_is_the_cost_basis():
    """开销必须基于**注入上下文的实际文本**，而不是 Python 对象的大小。"""
    payload = openai_json(GOOD)
    assert payload.startswith('{"type":"function"')
    assert '"get_weather"' in payload and '"city"' in payload
    assert len(payload.encode("utf-8")) > 100


def test_hash_tracks_semantic_changes_only():
    base = content_hash(GOOD)
    assert content_hash(replace(GOOD, description=GOOD.description + "改")) != base
    assert content_hash(replace(GOOD, owner="someone-else")) == base, \
        "owner 不影响注入内容，不该触发版本变更"
    assert content_hash(replace(GOOD, tags=("x",))) == base


# ── 审计规则 ───────────────────────────────────────────────────────
def test_illegal_name_is_an_error():
    findings = audit(ToolDef(name="get weather!", description=GOOD.description,
                             parameters=GOOD.parameters), count_fn=len)
    assert "NAME_PATTERN" in _rules(findings)
    assert any(f.severity is Severity.ERROR and f.rule == "NAME_PATTERN" for f in findings)


def test_missing_description_is_flagged_with_actionable_fix():
    findings = audit(ToolDef(name="t", parameters=GOOD.parameters), count_fn=len)
    hit = next(f for f in findings if f.rule == "DESC_MISSING" and f.path == "description")
    assert hit.severity is Severity.WARN
    assert "什么时候该用它" in hit.fix, "修法必须可行动，不能只说不合规"


def test_short_description_is_flagged():
    findings = audit(ToolDef(name="t", description="查天气", parameters=GOOD.parameters), count_fn=len)
    assert "DESC_TOO_SHORT" in _rules(findings)


def test_required_must_exist_in_properties():
    bad = ToolDef(
        name="t", description=GOOD.description,
        parameters={"type": "object", "properties": {"a": {"type": "string", "description": "参数 a 的说明文字"}},
                    "required": ["a", "ghost"]},
    )
    findings = audit(bad, count_fn=len)
    hit = next(f for f in findings if f.rule == "REQUIRED_MISMATCH")
    assert hit.severity is Severity.ERROR and "ghost" in hit.message


def test_missing_additional_properties_is_info_not_error():
    findings = audit(ToolDef(
        name="t", description=GOOD.description,
        parameters={"type": "object", "properties": {"a": {"type": "string", "description": "参数 a 的说明文字"}}},
    ), count_fn=len)
    hit = next(f for f in findings if f.rule == "ADDITIONAL_PROPS_UNSET")
    assert hit.severity is Severity.INFO


def test_param_without_description_is_flagged():
    findings = audit(ToolDef(
        name="t", description=GOOD.description,
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    ), count_fn=len)
    assert any(f.rule == "DESC_MISSING" and f.path.endswith("city.description") for f in findings)


def test_no_examples_blocks_fire_and_verify():
    findings = audit(ToolDef(name="t", description=GOOD.description,
                             parameters=GOOD.parameters), count_fn=len)
    assert "NO_EXAMPLE" in _rules(findings)


def test_description_budget_uses_the_given_counter():
    """预算按 token 算，所以必须传入与 gateway 同源的 count_fn。"""
    verbose = ToolDef(
        name="t", description="很长很长的说明" * 200, parameters=GOOD.parameters,
        examples=GOOD.examples,
    )
    assert "DESCRIPTION_BUDGET" in _rules(audit(verbose, count_fn=lambda text: len(text) // 2))
    assert "DESCRIPTION_BUDGET" not in _rules(audit(verbose, count_fn=lambda text: 10))


def test_unverified_schema_is_reported_not_silently_passed():
    """jsonschema 不可用时必须显式说"未校验"，不能当成通过。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("no jsonschema")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        findings = audit(GOOD, count_fn=len)
    finally:
        builtins.__import__ = real_import
    assert "SCHEMA_UNVERIFIED" in _rules(findings)
    assert "SCHEMA_VALID" not in _rules(findings), "没校验就不该声称合法"


def test_duplicate_names_detected_across_a_set():
    results = audit_many([GOOD, GOOD], count_fn=len)
    assert any(f.rule == "DUPLICATE_NAME" for f in results["get_weather"])


# ── 开销核算（P17）────────────────────────────────────────────────
def test_cost_report_includes_template_overhead():
    report = cost_report([GOOD], count_fn=lambda text: 78, template_overhead=213)
    assert report["json_tokens"] == 78
    assert report["effective_tokens"] == 291
    assert report["template_share"] == pytest.approx(213 / 291, abs=1e-3)
    assert report["template_share"] > 0.7, "P17：模板脚手架才是开销大头"


def test_cost_report_sorted_and_zero_overhead():
    small = ToolDef(name="a", description="x" * 40, parameters={"type": "object", "properties": {}})
    big = ToolDef(name="b", description="y" * 400, parameters={"type": "object", "properties": {}})
    report = cost_report([small, big], count_fn=len)
    assert [t["name"] for t in report["tools"]] == ["b", "a"], "按开销降序，最贵的排最前"
    assert report["template_overhead_tokens"] == 0
    assert report["effective_tokens"] == report["json_tokens"]


# ── 注册表：版本化 ─────────────────────────────────────────────────
@pytest.fixture
def registry(tmp_path):
    with Database(tmp_path / "t.sqlite") as db:
        yield ToolRegistry(ToolRepo(db), count_fn=len)


def test_register_assigns_v1_and_persists(registry):
    record = registry.register(GOOD)
    assert record.version == "1" and record.tokens and record.bytes
    assert record.side_effect == "network" and record.kind == "python_fn"
    fetched = registry.get("get_weather")
    assert fetched is not None and fetched.description == GOOD.description
    assert fetched.parameters["required"] == ["city"]
    assert fetched.side_effect is SideEffect.NETWORK


def test_same_content_does_not_bump_version(registry):
    first = registry.register(GOOD)
    again = registry.register(replace(GOOD, owner="other"))
    assert again.version == first.version == "1"
    assert again.owner == "other", "非语义字段仍应更新"


def test_changed_schema_bumps_version_and_keeps_id(registry):
    first = registry.register(GOOD)
    changed = replace(GOOD, description=GOOD.description + " 支持多城市")
    second = registry.register(changed)
    assert second.version == "2"
    assert second.id == first.id, "同名工具必须是同一行，否则旧 trace 的 tool_def_hash 找不到归属"
    assert second.hash != first.hash


def test_refresh_costs_keeps_version(registry):
    registry.register(GOOD)
    registry.count_fn = lambda text: len(text) * 2
    assert registry.refresh_costs() == 1
    record = registry.repo.find_by_name("get_weather")
    assert record.version == "1", "换标定不该改版本号（内容没变）"
    assert record.tokens == len(openai_json(GOOD)) * 2


def test_specs_only_include_enabled(registry):
    registry.register(GOOD)
    registry.register(ToolDef(name="disabled_tool", description="x" * 40,
                              parameters=GOOD.parameters, enabled=False))
    names = {s.name for s in registry.specs()}
    assert names == {"get_weather"}
    assert {s.name for s in registry.specs(["disabled_tool"])} == set()


def test_audit_all_reads_from_repo(registry):
    registry.register(ToolDef(name="bad name!", description="短", parameters={}))
    results = registry.audit_all()
    assert "NAME_PATTERN" in _rules(results["bad name!"])


# ── 外部输入解析 ───────────────────────────────────────────────────
def test_defs_from_payload_accepts_both_shapes():
    raw = {"name": "t", "description": "d", "parameters": {"type": "object"}}
    assert defs_from_payload([raw])[0].name == "t"
    assert defs_from_payload({"tools": [raw]})[0].name == "t"


def test_defs_from_payload_rejects_bad_enums_with_context():
    with pytest.raises(ValueError, match="kind 非法"):
        defs_from_payload([{"name": "t", "kind": "carrier_pigeon"}])
    with pytest.raises(ValueError, match="side_effect 非法") as ei:
        defs_from_payload([{"name": "t", "side_effect": "maybe"}])
    assert "t" in str(ei.value), "报错必须指名是哪个工具"


def test_defs_from_payload_requires_name():
    with pytest.raises(ValueError, match="缺少 name"):
        defs_from_payload([{"description": "d"}])
    with pytest.raises(ValueError, match="必须是列表"):
        defs_from_payload("not a list")


def test_defs_from_payload_keeps_unknown_keys_in_extra():
    defs = defs_from_payload([
        {"name": "t", "description": "d", "side_effect": "read", "mcp_server": "weather-srv"},
    ])
    assert defs[0].extra == {"mcp_server": "weather-srv"}, "新字段不许丢（原则 6）"


def test_defs_from_payload_marks_missing_side_effect():
    """缺 side_effect 不能静默取默认 read——沙箱靠它决定要不要拒绝或审批。"""
    defs = defs_from_payload([{"name": "t", "description": "d"}])
    assert defs[0].side_effect is SideEffect.READ
    assert defs[0].extra["side_effect_untagged"] is True


def test_example_yaml_file_is_importable(tmp_path):
    """examples/tools.yaml 必须真的能导入——示例文件坏了比没有示例更糟。"""
    import json
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "examples" / "tools.yaml"
    assert path.exists(), f"缺少示例文件: {path}"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    defs = defs_from_payload(payload)
    assert {d.name for d in defs} == {
        "get_weather", "calculator", "db_query", "fs_read", "http_demo",
    }
    for definition in defs:
        errors = [f for f in audit(definition, count_fn=len) if f.severity is Severity.ERROR]
        assert not errors, f"{definition.name} 有 error 级问题: {errors}"
    assert json.loads(openai_json(defs[0]))["function"]["name"] == "get_weather"

    by_name = {d.name: d for d in defs}
    # constants / http 是执行器配置，不进 schema，所以必须落在 extra 里
    assert by_name["fs_read"].extra["constants"] == {"root": "./examples"}
    assert "root" not in by_name["fs_read"].parameters["properties"]
    assert by_name["http_demo"].extra["http"]["url"].startswith("http://")
    assert by_name["http_demo"].side_effect is SideEffect.NETWORK


def test_tool_kind_coverage():
    assert {str(k) for k in ToolKind} == {"python_fn", "http", "mcp", "ollama_builtin", "fixture"}
