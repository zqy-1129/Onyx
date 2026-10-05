"""JSON 解析与 schema 合规 grader。

结构化输出的评测必须把三件事分开报，否则"JSON 合法率"这一个数会骗人：
1. **能不能解析**（模型有没有吐出合法 JSON，含被 ```json 包裹的情况）
2. **合不合规**（解析出来了，但字段缺了/类型错了）
3. **字段级 EM**（合规了，但值是错的）

`jsonschema` 缺失时报 `verified=False`（**未知，不是通过**）——
与 `tools/spec.py` 的 `SCHEMA_UNVERIFIED` 同一条纪律。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.eval.graders.normalize import strip_code_fence


@dataclass(frozen=True, slots=True)
class JsonCheck:
    parsed: bool
    value: Any = None
    error: str = ""
    #: 解析前的原文（截断）。诊断"为什么没解析出来"只能看原文
    raw: str = ""

    @property
    def as_object(self) -> Mapping[str, Any] | None:
        return self.value if isinstance(self.value, Mapping) else None


@dataclass(frozen=True, slots=True)
class SchemaCheck:
    valid: bool
    #: schema 本身有没有被校验过。False 表示"未知"，不是"通过"
    verified: bool
    errors: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    type_errors: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


def parse_json(text: Any, *, strict_object: bool = False) -> JsonCheck:
    """解析模型输出里的 JSON。

    先剥 ```json 围栏，再试整体解析；失败后**只**尝试截取最外层的花括号/方括号，
    用来救"前面加了一句『好的，结果如下：』"这种常见形态。
    不做更激进的修复（补引号、去尾逗号）——那会把模型的格式能力问题藏起来。
    """
    raw = "" if text is None else str(text)
    stripped = strip_code_fence(raw)
    last_error = "没有找到可解析的 JSON"
    for candidate in _candidates(stripped):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if strict_object and not isinstance(value, Mapping):
            return JsonCheck(False, None, f"期望 JSON 对象，实际是 {type(value).__name__}",
                             raw[:500])
        return JsonCheck(True, value, "", raw[:500])
    return JsonCheck(False, None, last_error, raw[:500])


def clean_json_object(text: Any) -> bool:
    """输出的**形态**是否干净：整段就是一个对象，没有围栏也没有前后缀说明。

    能解析不等于干净——`好的，结果如下：{...}` 会被 `parse_json` 救回来，
    但那是没听话的形态，下游按 JSON 直读的代码会炸。
    所以"解析成功率"与"格式合法率"是两个数（DESIGN §9.4），这里管后者。
    """
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        return False
    return stripped.startswith("{") and stripped.endswith("}")


def _candidates(text: str) -> Sequence[str]:
    yield text
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start != -1 and end > start:
            yield text[start : end + 1]


def check_schema(value: Any, schema: Mapping[str, Any] | None) -> SchemaCheck:
    """校验 value 是否符合 schema。

    优先用 `jsonschema`（draft 2020-12）；没装就退回自带的浅层检查，
    并把 `verified=False` 标出来——退回档只查 required / 顶层类型，
    查不了 anyOf / 嵌套 / 数值范围，所以它给出的是"没发现问题"而不是"合规"。
    """
    if not schema:
        return SchemaCheck(True, verified=False, detail={"reason": "没有 schema 可校验"})

    try:
        import jsonschema
    except ImportError:
        return _shallow_check(value, schema)

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except Exception as exc:  # noqa: BLE001 - schema 库的任何异常都转成一条结论
        return SchemaCheck(
            False, verified=False, errors=(f"schema 本身非法: {exc}"[:300],),
            detail={"kind": "bad_schema"},
        )
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
    if not errors:
        return SchemaCheck(True, verified=True)
    return SchemaCheck(
        False, verified=True,
        errors=tuple(f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"[:200]
                     for e in errors[:10]),
        missing=tuple(
            str(p) for e in errors if e.validator == "required"
            for p in _missing_names(e.message)
        ),
        type_errors=tuple(
            '/'.join(str(p) for p in e.absolute_path) or "<root>"
            for e in errors if e.validator == "type"
        ),
        detail={"count": len(errors)},
    )


def _missing_names(message: str) -> Sequence[str]:
    """从 jsonschema 的 required 报错里取出字段名。"""
    import re

    match = re.search(r"'([^']+)' is a required property", message)
    return (match.group(1),) if match else ()


def _shallow_check(value: Any, schema: Mapping[str, Any]) -> SchemaCheck:
    """jsonschema 缺失时的退回档：只查 required 与顶层类型。"""
    if not isinstance(value, Mapping):
        return SchemaCheck(
            False, verified=False, errors=(f"期望对象，实际 {type(value).__name__}",),
            detail={"fallback": True},
        )
    required = [name for name in (schema.get("required") or []) if isinstance(name, str)]
    missing = tuple(name for name in required if name not in value)
    properties = schema.get("properties") or {}
    type_errors: tuple[str, ...] = ()
    if isinstance(properties, Mapping):
        for name, prop in properties.items():
            if name not in value or not isinstance(prop, Mapping):
                continue
            declared = prop.get("type")
            if isinstance(declared, str) and not _type_ok(value[name], declared):
                type_errors += (f"{name}: 期望 {declared}",)
    unexpected: tuple[str, ...] = ()
    if schema.get("additionalProperties") is False and isinstance(properties, Mapping):
        unexpected = tuple(sorted(set(value) - set(properties)))
    ok = not (missing or type_errors or unexpected)
    return SchemaCheck(
        ok, verified=False, missing=missing, type_errors=type_errors, unexpected=unexpected,
        errors=() if ok else tuple([*(f"缺字段 {m}" for m in missing), *type_errors,
                                    *(f"多余字段 {u}" for u in unexpected)]),
        detail={"fallback": True,
                "note": "jsonschema 未安装：只查了 required 与顶层类型，"
                        "嵌套/anyOf/数值范围未校验，所以这是『没发现问题』而非『合规』"},
    )


def _type_ok(value: Any, declared: str) -> bool:
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "array":
        return isinstance(value, list | tuple)
    if declared == "object":
        return isinstance(value, Mapping)
    if declared == "null":
        return value is None
    return True  # 不认识的 type 关键字不拦，交给 jsonschema


def field_em(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    """字段级精确匹配率。返回逐字段结果，便于定位到底哪个字段总是错。"""
    per_field = {name: _field_ok(want, actual.get(name)) for name, want in expected.items()}
    matched = sum(1 for ok in per_field.values() if ok)
    return {
        "per_field": per_field,
        "matched": matched,
        "total": len(per_field),
        "score": matched / len(per_field) if per_field else None,
        "missing": sorted(name for name in expected if name not in actual),
        "unexpected": sorted(name for name in actual if name not in expected),
    }


def _field_ok(want: Any, got: Any) -> bool:
    """单个字段的比较。数值字段按**数值**比。

    `500` 与 `500.0` 是同一笔钱：按字符串比会把 JSON 的写法差异算成抽取错误，
    而真实跑一次就是这样——qwen3.5:9b 的 amount 有 19 条可判定样本，
    按字符串比只有 4 条对，其中大部分掉的正是写成整数的 500 / 1500。
    反过来，`"500元"` 不算对：单位没去掉是真的没抽对（那一层由 schema 先拦）。
    """
    from onyx.eval.graders.exact import exact, numeric

    if isinstance(want, bool) or isinstance(got, bool):
        return exact(want, got)
    if isinstance(want, int | float) and isinstance(got, int | float):
        return numeric(want, got)
    return exact(want, got)
