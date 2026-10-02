"""时间工具：非确定性工具的样本，用于验证 deterministic/idempotent 标记。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from onyx.core.errors import ToolArgError

MAX_OFFSET_HOURS = 24


def time_now(tz_offset_hours: float = 0.0, fmt: str = "iso") -> dict[str, Any]:
    if not isinstance(tz_offset_hours, int | float) or isinstance(tz_offset_hours, bool):
        raise ToolArgError(f"tz_offset_hours 必须是数值，实际 {type(tz_offset_hours).__name__}")
    if abs(tz_offset_hours) > MAX_OFFSET_HOURS:
        raise ToolArgError(f"时区偏移超出 ±{MAX_OFFSET_HOURS} 小时: {tz_offset_hours}")
    tz = timezone(timedelta(hours=float(tz_offset_hours)))
    now = datetime.now(tz)
    if fmt == "iso":
        text = now.isoformat(timespec="seconds")
    elif fmt == "date":
        text = now.strftime("%Y-%m-%d")
    elif fmt == "time":
        text = now.strftime("%H:%M:%S")
    elif fmt == "unix":
        text = str(int(now.timestamp()))
    else:
        raise ToolArgError(f"不支持的 fmt: {fmt!r}（可选 iso/date/time/unix）")
    return {"now": text, "tz_offset_hours": float(tz_offset_hours), "fmt": fmt}
