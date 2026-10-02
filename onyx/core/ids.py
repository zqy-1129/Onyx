"""时间可排序 id（ULID 形状，纯 stdlib 实现）。

为什么不用 uuid4：trace 列表天然按时间倒序查询，可排序 id 让 `ORDER BY id` 与
`WHERE id > :cursor` 的游标分页都能走索引，不必额外维护 started_at 排序键。
"""

from __future__ import annotations

import random
import threading
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TIME_CHARS = 10  # 48 bits of ms
_RANDOM_CHARS = 16  # 80 bits
_ID_LEN = _TIME_CHARS + _RANDOM_CHARS

_lock = threading.Lock()
_last_ms = 0
_counter = 0


def _b32(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_CROCKFORD[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def _b32_decode(text: str) -> int:
    value = 0
    for ch in text:
        idx = _CROCKFORD.find(ch)
        if idx < 0:
            raise ValueError(f"非法 base32 字符: {ch!r}")
        value = (value << 5) | idx
    return value


def new_trace_id(*, now_ms: int | None = None) -> str:
    """返回 26 字符 id：前 10 位是毫秒时间，后 16 位含单调计数器 + 随机位。

    同一毫秒内连续调用严格递增（靠 `_counter`），跨进程靠随机位避免碰撞。
    """
    global _last_ms, _counter

    ms = now_ms if now_ms is not None else time.time_ns() // 1_000_000
    with _lock:
        if ms <= _last_ms:
            ms = _last_ms
            _counter += 1
        else:
            _last_ms = ms
            _counter = random.getrandbits(32)
        counter = _counter & 0xFFFFFFFF
    rand = (counter << 48) | random.getrandbits(48)
    return _b32(ms, _TIME_CHARS) + _b32(rand, _RANDOM_CHARS)


def id_timestamp_ms(trace_id: str) -> int:
    """从 id 反解生成时刻（毫秒）。用于按 id 做时间范围过滤而无需读 started_at。"""
    if len(trace_id) != _ID_LEN:
        raise ValueError(f"id 长度应为 {_ID_LEN}，实际 {len(trace_id)}")
    return _b32_decode(trace_id[:_TIME_CHARS])


def is_valid_id(trace_id: str) -> bool:
    if len(trace_id) != _ID_LEN:
        return False
    return all(ch in _CROCKFORD for ch in trace_id)


def reset_state(*, last_ms: int = 0) -> None:
    """清空单调状态。

    仅测试需要：`new_trace_id(now_ms=...)` 传一个比进程内历史更早的时间时，
    钳制逻辑会把它抬到 `_last_ms`（这是正确行为，否则 id 不再单调），
    于是"给定时间 → 反解时间"的断言必须在干净状态下做。
    """
    global _last_ms, _counter
    with _lock:
        _last_ms = last_ms
        _counter = 0
