"""SSE 事件广播。

gateway 在**同步线程**里产生事件，SSE 在 **asyncio** 里消费。
用 `queue.Queue`（本身线程安全）解耦，避免 `run_coroutine_threadsafe` 那类跨环调用的复杂度。

背压策略：队列满就**丢弃最旧的并计数**。看板宁可掉几个增量帧，
也不能因为一个慢客户端把 gateway 的发送路径阻塞住——那会污染我们要测量的延迟。
"""

from __future__ import annotations

import asyncio
import json
import queue
import uuid
from collections.abc import AsyncIterator
from typing import Any

from onyx.core.event import TraceEvent

DEFAULT_QUEUE_SIZE = 500
DEFAULT_MAX_SUBSCRIBERS = 64


class SseBroker:
    def __init__(self, *, queue_size: int = DEFAULT_QUEUE_SIZE,
                 max_subscribers: int = DEFAULT_MAX_SUBSCRIBERS) -> None:
        self._subscribers: dict[str, queue.Queue[str]] = {}
        self._queue_size = queue_size
        self._max_subscribers = max_subscribers
        self.published = 0
        self.dropped = 0

    # ── 订阅 ──────────────────────────────────────────────────────
    def subscribe(self) -> tuple[str, queue.Queue[str]]:
        if len(self._subscribers) >= self._max_subscribers:
            oldest = next(iter(self._subscribers))
            self._subscribers.pop(oldest, None)
        sub_id = uuid.uuid4().hex
        self._subscribers[sub_id] = queue.Queue(maxsize=self._queue_size)
        return sub_id, self._subscribers[sub_id]

    def unsubscribe(self, sub_id: str) -> None:
        self._subscribers.pop(sub_id, None)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # ── 发布（同步线程调用）───────────────────────────────────────
    def publish(self, event: TraceEvent) -> None:
        """实现 EventSink 协议，可直接挂进 EventFanout。"""
        self.published += 1
        line = _sse_frame(event.to_dict())
        for queue_ in list(self._subscribers.values()):
            try:
                queue_.put_nowait(line)
            except queue.Full:
                self.dropped += 1
                try:
                    queue_.get_nowait()  # 丢最旧的一帧，保住实时性
                    queue_.put_nowait(line)
                except queue.Empty:
                    pass

    def flush(self, timeout: float = 1.0) -> None: ...
    def close(self) -> None:
        self._subscribers.clear()

    name = "sse"

    # ── 消费（asyncio）────────────────────────────────────────────
    async def stream(self, sub_id: str, queue_: queue.Queue[str], *,
                     heartbeat: float = 15.0) -> AsyncIterator[str]:
        yield _sse_frame({"type": "hello", "sub_id": sub_id})
        while True:
            try:
                item = await asyncio.to_thread(_get_with_timeout, queue_, heartbeat)
            except asyncio.CancelledError:  # 客户端断开
                self.unsubscribe(sub_id)
                raise
            if item is None:
                yield ": heartbeat\n\n"  # 保活，防止代理层掐断空闲连接
                continue
            yield item


def _get_with_timeout(queue_: queue.Queue[str], timeout: float) -> str | None:
    try:
        return queue_.get(timeout=timeout)
    except queue.Empty:
        return None


def _sse_frame(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"
