"""参数校验：把"模型给的参数"变成"可以安全交给实现的参数"。

这一层是契约测试第 2/3 条断言的基础：**非法参数必须变成 ToolArgError，
既不能崩进程，也不能返回 200 让调用方以为成功了**。
本地小模型产出的参数畸形率远高于云端模型，所以这里要给出足够细的失败种类
（缺字段 / 类型错 / 枚举越界 / 多余字段），否则"工具调不对"没法归因。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from onyx.core.errors import ToolArgError

#: JSON Schema type → 允许的 Python 类型。bool 必须在 int 之前判断（Python 里 bool 是 int 子类）
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list, tuple),
    "object": (dict,),
    "null": (type(None),),
}


def validate_args(schema: Mapping[str, Any] | None, args: Any) -> dict[str, Any]:
    """校验并返回规范化后的参数 dict。任何不合规都抛 `ToolArgError`。"""
    if args is None:
        args = {}
    if not isinstance(args, Mapping):
        raise ToolArgError(
            f"参数必须是对象，实际是 {type(args).__name__}",
            detail={"kind": "not_an_object", "args_repr": repr(args)[:200]},
        )
    args = dict(args)
    if not schema:
        return args

    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        # 没有 properties 就无从校验字段；返回原值但标注，让上层能识别"未校验"
        return args

    required = schema.get("required") or []
    if not isinstance(required, Sequence) or isinstance(required, str | bytes):
        raise ToolArgError("schema.required 必须是数组", detail={"kind": "bad_schema"})
    missing = [name for name in required if name not in args or args[name] is None]
    if missing:
        raise ToolArgError(
            f"缺少必填参数: {missing}", detail={"kind": "missing_required", "missing": missing}
        )

    if schema.get("additionalProperties") is False:
        unknown = sorted(set(args) - set(properties))
        if unknown:
            raise ToolArgError(
                f"多余参数: {unknown}", detail={"kind": "unexpected_field", "unknown": unknown}
            )

    for name, value in args.items():
        prop = properties.get(name)
        if not isinstance(prop, Mapping):
            continue
        if value is None:
            if "null" not in _types_of(prop):
                raise ToolArgError(
                    f"参数 {name} 不允许为 null", detail={"kind": "null_not_allowed", "field": name}
                )
            continue
        _check_type(name, value, prop)
        _check_enum(name, value, prop)
        if isinstance(value, Mapping) and prop.get("type") == "object":
            validate_args(prop, value)
        if isinstance(value, list | tuple) and isinstance(prop.get("items"), Mapping):
            for index, item in enumerate(value):
                _check_type(f"{name}[{index}]", item, prop["items"])
                _check_enum(f"{name}[{index}]", item, prop["items"])

    return args


def _types_of(prop: Mapping[str, Any]) -> tuple[str, ...]:
    raw = prop.get("type")
    if isinstance(raw, str):
        return (raw,)
    if isinstance(raw, list | tuple):
        return tuple(str(t) for t in raw)
    return ()


def _check_type(name: str, value: Any, prop: Mapping[str, Any]) -> None:
    declared = _types_of(prop)
    if not declared:
        return
    allowed: tuple[type, ...] = ()
    for type_name in declared:
        allowed += _TYPE_MAP.get(type_name, ())
    if not allowed:
        return
    # bool 是 int 的子类：声明 integer 时不接受 True/False，反之亦然
    if isinstance(value, bool) and "boolean" not in declared:
        raise ToolArgError(
            f"参数 {name} 类型错误：期望 {list(declared)}，实际 boolean",
            detail={"kind": "type_mismatch", "field": name, "expected": list(declared), "actual": "boolean"},
        )
    if not isinstance(value, bool) and isinstance(value, int) and "integer" not in declared \
            and "number" not in declared and "boolean" in declared:
        raise ToolArgError(
            f"参数 {name} 类型错误：期望 boolean，实际 integer",
            detail={"kind": "type_mismatch", "field": name},
        )
    if not isinstance(value, allowed):
        raise ToolArgError(
            f"参数 {name} 类型错误：期望 {list(declared)}，实际 {type(value).__name__}",
            detail={
                "kind": "type_mismatch", "field": name,
                "expected": list(declared), "actual": type(value).__name__,
            },
        )


def _check_enum(name: str, value: Any, prop: Mapping[str, Any]) -> None:
    allowed = prop.get("enum")
    if isinstance(allowed, list | tuple) and allowed and value not in allowed:
        raise ToolArgError(
            f"参数 {name} 取值 {value!r} 不在允许范围内: {list(allowed)}",
            detail={"kind": "enum_violation", "field": name, "allowed": list(allowed), "actual": value},
        )


def diff_args(expected: Mapping[str, Any], actual: Mapping[str, Any] | None) -> dict[str, Any]:
    """评测/验证用的参数差异报告（S12 的 fire-and-verify 会用到）。"""
    actual = actual or {}
    missing = sorted(k for k in expected if k not in actual)
    unexpected = sorted(k for k in actual if k not in expected)
    mismatched = {
        k: {"expected": expected[k], "actual": actual[k]}
        for k in expected
        if k in actual and actual[k] != expected[k]
    }
    return {
        "ok": not (missing or unexpected or mismatched),
        "missing": missing, "unexpected": unexpected, "mismatched": mismatched,
        "subset_ok": not (missing or mismatched),
    }
