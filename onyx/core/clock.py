"""时钟抽象。

延迟指标（TTFT / prefill / decode）必须用单调时钟计算，落库时间戳用墙钟。
把两者收在一个可替换对象里，测试才能用假时钟复现"慢请求""冷启动"等时序场景。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


def utc_now_iso() -> str:
    """ISO-8601 UTC，秒级以下保留 6 位小数。SQLite TEXT 列可直接按字典序排序。"""
    return datetime.now(UTC).isoformat(timespec="microseconds")


@runtime_checkable
class Clock(Protocol):
    def wall_iso(self) -> str: ...
    def monotonic_ns(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SystemClock:
    def wall_iso(self) -> str:
        return utc_now_iso()

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


@dataclass(slots=True)
class FakeClock:
    """测试用：手动推进墙钟，单调钟按固定步长前进。"""

    wall: str = "2026-01-01T00:00:00.000000+00:00"
    mono_ns: int = 0
    step_ns: int = 1_000_000
    reads: list[str] = field(default_factory=list)

    def wall_iso(self) -> str:
        self.reads.append("wall")
        return self.wall

    def monotonic_ns(self) -> int:
        self.reads.append("mono")
        self.mono_ns += self.step_ns
        return self.mono_ns

    def advance_ms(self, ms: float) -> None:
        self.mono_ns += int(ms * 1_000_000)


SYSTEM_CLOCK: Clock = SystemClock()
