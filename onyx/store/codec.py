"""JSON 列编解码。

`default=str` 是刻意的：落库路径上**绝不允许因为一个不可序列化对象而丢掉整条 trace**。
观测系统的第一要务是把证据存下来，而不是保证类型纯洁。
"""

from __future__ import annotations

import json
from typing import Any


def dumps(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if not obj:  # {} / [] / () 一律存 NULL，省空间也让 IS NULL 查询有意义
        return None
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def loads(raw: str | None, default: Any = None) -> Any:
    if raw is None or raw == "":
        return default
    # 幂等：已经解码过的 dict/list 原样返回。
    # 曾经在这里对 dict 再调一次 json.loads 直接抛 TypeError（tool_repo 的嵌套字段）。
    if isinstance(raw, dict | list):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def loads_list(raw: str | None) -> list[Any]:
    value = loads(raw, default=[])
    return value if isinstance(value, list) else []


def loads_dict(raw: str | None) -> dict[str, Any]:
    value = loads(raw, default={})
    return value if isinstance(value, dict) else {}
