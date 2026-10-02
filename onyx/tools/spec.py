"""工具定义与契约审计。

审计的意义不是"格式检查"，而是**可诊断性**：本地小模型调不对工具时，
一半以上的原因在定义本身（描述太短、参数无说明、required 与 properties 不一致）。
所以每条规则都对应一个可行动的修法，而不是只报"不合规"。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

#: 单工具描述超过这个 token 数就该警惕：工具库开销是**每次请求**都要付的（P17）
DESCRIPTION_TOKEN_BUDGET = 400
#: 描述短于此长度基本等于没写，模型只能靠名字猜
MIN_DESCRIPTION_CHARS = 20


class ToolKind(StrEnum):
    PYTHON_FN = "python_fn"
    HTTP = "http"
    MCP = "mcp"
    OLLAMA_BUILTIN = "ollama_builtin"
    FIXTURE = "fixture"


class SideEffect(StrEnum):
    """沙箱决策依据。未标注就是审计错误——不能默认当成只读。"""

    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    EXEC = "exec"


class Severity(StrEnum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    kind: ToolKind = ToolKind.PYTHON_FN
    side_effect: SideEffect = SideEffect.READ
    impl_ref: str = ""
    version: str = "1"
    tags: tuple[str, ...] = ()
    owner: str = ""
    enabled: bool = True
    timeout_ms: int | None = None
    doc: str = ""
    examples: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return content_hash(self)

    def to_spec(self):
        """转成 L1 的 `ToolSpec`（发给引擎的形状）。"""
        from onyx.core.types import ToolSpec

        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    def as_openai(self) -> dict[str, Any]:
        return to_openai_tool(self)

    @classmethod
    def from_openai(cls, raw: Mapping[str, Any], **kw: Any) -> ToolDef:
        function = raw.get("function") or raw
        parameters = dict(function.get("parameters") or {})
        return cls(
            name=str(function.get("name", "")),
            description=str(function.get("description", "") or ""),
            parameters=parameters,
            **kw,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str = ""
    error_kind: str = ""
    mocked: bool = False
    bytes: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.mocked:
            return "mocked"
        return "ok" if self.ok else (self.error_kind or "error")


def to_openai_tool(definition: ToolDef) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.parameters or {"type": "object", "properties": {}},
        },
    }


def openai_json(definition: ToolDef) -> str:
    """注入上下文的**实际文本**：开销核算必须基于它，而不是基于 Python 对象的大小。"""
    return json.dumps(to_openai_tool(definition), ensure_ascii=False, separators=(",", ":"))


def content_hash(definition: ToolDef) -> str:
    """内容 hash：schema/描述变化即新版本。

    旧 trace 通过 tool_def_hash 仍能定位当时的定义——否则"换工具描述前后模型表现对比"
    这个最常见的调优动作就没有可比性。
    """
    payload = json.dumps(
        {
            "name": definition.name,
            "description": definition.description,
            "parameters": definition.parameters,
            "kind": str(definition.kind),
            "side_effect": str(definition.side_effect),
        },
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class AuditFinding:
    rule: str
    severity: Severity
    message: str
    path: str = ""
    fix: str = ""


#: 规则说明（CLI 与看板共用同一份文案，避免前后端各写一套）
RULES: dict[str, str] = {
    "NAME_PATTERN": "工具名必须匹配 ^[a-zA-Z0-9_-]{1,64}$",
    "SCHEMA_VALID": "parameters 必须是合法的 JSON Schema（draft 2020-12）",
    "SCHEMA_UNVERIFIED": "未安装 jsonschema，合法性未能校验（不是通过，是未知）",
    "DESC_MISSING": "工具或参数缺少 description，模型只能靠名字猜",
    "DESC_TOO_SHORT": "描述过短，等于没写",
    "DESCRIPTION_BUDGET": "单工具注入上下文过大——这笔开销每次请求都要付（P17）",
    "SIDE_EFFECT_UNTAGGED": "未标注副作用，沙箱无法决策；不许默认当成只读",
    "NO_EXAMPLE": "没有 examples，无法做 fire-and-verify",
    "REQUIRED_MISMATCH": "required 里出现了 properties 未定义的字段",
    "ADDITIONAL_PROPS_UNSET": "未声明 additionalProperties，模型多给字段时行为不确定",
    "DUPLICATE_NAME": "工具名重复",
    "NO_PARAMETERS": "没有 parameters 字段（无参工具也应显式写空 object）",
}


def audit(
    definition: ToolDef,
    *,
    count_fn: Callable[[str], int] | None = None,
    token_budget: int = DESCRIPTION_TOKEN_BUDGET,
) -> list[AuditFinding]:
    """审计单个工具定义。`count_fn` 决定开销核算的精度档位（与 gateway 共用同一套）。"""
    findings: list[AuditFinding] = []

    def add(rule: str, severity: Severity, message: str, *, path: str = "", fix: str = "") -> None:
        findings.append(AuditFinding(
            rule=rule, severity=severity, message=message, path=path,
            fix=fix or RULES.get(rule, ""),
        ))

    if not NAME_PATTERN.match(definition.name or ""):
        add("NAME_PATTERN", Severity.ERROR, f"非法工具名: {definition.name!r}", path="name")

    if not definition.description.strip():
        add("DESC_MISSING", Severity.WARN, "工具缺少 description", path="description",
            fix="补一句「什么时候该用它」——这比参数说明更影响选择正确率")
    elif len(definition.description.strip()) < MIN_DESCRIPTION_CHARS:
        add("DESC_TOO_SHORT", Severity.WARN,
            f"描述仅 {len(definition.description.strip())} 字符", path="description")

    if not definition.parameters:
        add("NO_PARAMETERS", Severity.WARN, "缺少 parameters", path="parameters",
            fix='无参工具也应显式写 {"type":"object","properties":{}}')
    else:
        findings.extend(_audit_schema(definition, add, count_fn, token_budget))

    if not definition.examples:
        add("NO_EXAMPLE", Severity.INFO, "没有 examples", path="examples",
            fix="至少给 1 个 (指令 → 期望调用) 样本，才能做 fire-and-verify")

    return findings


def _audit_schema(
    definition: ToolDef,
    add: Callable[..., None],
    count_fn: Callable[[str], int] | None,
    token_budget: int,
) -> list[AuditFinding]:
    out: list[AuditFinding] = []
    schema = definition.parameters

    try:
        import jsonschema
    except ImportError:
        add("SCHEMA_UNVERIFIED", Severity.WARN, "jsonschema 未安装", path="parameters",
            fix="uv sync --extra runtime")
        jsonschema = None  # type: ignore[assignment]
    if jsonschema is not None:
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except Exception as exc:  # noqa: BLE001 - 任何 schema 库异常都转成一条审计结论
            add("SCHEMA_VALID", Severity.ERROR, f"JSON Schema 非法: {exc}"[:300], path="parameters")

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        add("REQUIRED_MISMATCH", Severity.ERROR, "parameters.properties 缺失或不是对象", path="parameters")
        return out

    required = schema.get("required") or []
    if not isinstance(required, list):
        add("REQUIRED_MISMATCH", Severity.ERROR, "required 必须是数组", path="parameters.required")
        required = []
    missing = [name for name in required if name not in properties]
    if missing:
        add("REQUIRED_MISMATCH", Severity.ERROR, f"required 未在 properties 中定义: {missing}",
            path="parameters.required")

    if "additionalProperties" not in schema:
        add("ADDITIONAL_PROPS_UNSET", Severity.INFO, "未声明 additionalProperties",
            path="parameters.additionalProperties")

    for name, prop in properties.items():
        if not isinstance(prop, dict):
            add("SCHEMA_VALID", Severity.ERROR, f"参数 {name} 的定义不是对象", path=f"properties.{name}")
            continue
        description = str(prop.get("description") or "").strip()
        if not description:
            add("DESC_MISSING", Severity.WARN, f"参数 {name} 缺少 description",
                path=f"properties.{name}.description")
        elif len(description) < MIN_DESCRIPTION_CHARS:
            add("DESC_TOO_SHORT", Severity.WARN, f"参数 {name} 的描述仅 {len(description)} 字符",
                path=f"properties.{name}.description")

    payload = openai_json(definition)
    tokens = count_fn(payload) if count_fn else None
    if tokens is not None and tokens > token_budget:
        add("DESCRIPTION_BUDGET", Severity.WARN,
            f"该工具注入 {tokens} token（预算 {token_budget}）", path="parameters",
            fix="精简 description 或拆分工具；注意 P17：真正的开销大头可能是模板脚手架而非 JSON")
    return out


def audit_many(
    definitions: Iterable[ToolDef], *, count_fn: Callable[[str], int] | None = None
) -> dict[str, list[AuditFinding]]:
    """批量审计，额外检查跨工具的重名。"""
    results: dict[str, list[AuditFinding]] = {}
    seen: dict[str, int] = {}
    for definition in definitions:
        findings = audit(definition, count_fn=count_fn)
        seen[definition.name] = seen.get(definition.name, 0) + 1
        results[definition.name] = findings
    for name, count in seen.items():
        if count > 1:
            results[name].append(AuditFinding(
                rule="DUPLICATE_NAME", severity=Severity.ERROR,
                message=f"工具名 {name!r} 出现 {count} 次", path="name", fix=RULES["DUPLICATE_NAME"],
            ))
    return results


def cost_report(
    definitions: Sequence[ToolDef],
    *,
    count_fn: Callable[[str], int],
    template_overhead: int = 0,
) -> dict[str, Any]:
    """工具库的上下文开销报告（DESIGN §6.2 的核心指标）。

    `template_overhead` 是该模型的模板脚手架开销（由归因残差实测得到）。
    **必须一起报**：P17 实测一个 78 token 的 JSON 实吃约 291 token，
    其中 73% 是模板注入的说明文本。只报 JSON 大小会把优化方向引到
    "精简描述"，而真正有效的是换模板/换模型。
    """
    per_tool = []
    for definition in definitions:
        payload = openai_json(definition)
        per_tool.append({
            "name": definition.name,
            "tokens": count_fn(payload),
            "bytes": len(payload.encode("utf-8")),
            "kind": str(definition.kind),
            "side_effect": str(definition.side_effect),
        })
    per_tool.sort(key=lambda item: item["tokens"], reverse=True)
    total = sum(item["tokens"] for item in per_tool)
    return {
        "tools": per_tool,
        "json_tokens": total,
        "json_bytes": sum(item["bytes"] for item in per_tool),
        "template_overhead_tokens": template_overhead,
        "effective_tokens": total + template_overhead,
        "template_share": round(template_overhead / (total + template_overhead), 4)
        if (total + template_overhead) else None,
        "count_source": getattr(count_fn, "source_name", "unknown"),
    }
