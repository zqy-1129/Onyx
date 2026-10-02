"""SQLite 记录 sink：后台单写线程 + 批量提交。

为什么异步：请求主链路不能被磁盘 IO 拖慢（TTFT 是我们要测量的对象，
不能反过来被观测本身污染）。
为什么不无脑异步到底：队列满时**丢弃并计数**，绝不阻塞请求——
观测系统宁可丢样本也不能把被测系统拖垮，但丢了多少必须可见（`dropped`）。
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable, Iterable, Sequence

from onyx.store.db import Database
from onyx.store.records import (
    AnomalyRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)
from onyx.store.repos import ModelRepo, TraceRepo, UsageRepo

log = logging.getLogger("onyx.store.sqlite")

_Op = Callable[[], None]


class SqliteRecordSink:
    name = "sqlite"

    def __init__(
        self,
        db: Database,
        *,
        batch_size: int = 64,
        idle_wait: float = 0.05,
        max_queue: int = 20_000,
    ) -> None:
        self.db = db
        self.traces = TraceRepo(db)
        self.usage = UsageRepo(db)
        self.models = ModelRepo(db)
        self._queue: queue.Queue[_Op | None] = queue.Queue(maxsize=max_queue)
        self._batch_size = batch_size
        self._idle_wait = idle_wait
        self._pending = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._stop = threading.Event()
        self.written = 0
        self.dropped = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="onyx-sqlite-writer", daemon=True)
        self._thread.start()

    # ── RecordSink 接口 ───────────────────────────────────────────
    def write_trace(self, rec: TraceRecord) -> None:
        self._enqueue(lambda: self.traces.upsert(rec))

    def finish_trace(self, trace_id: str, **fields: object) -> None:
        self._enqueue(lambda: self.traces.finish(trace_id, **fields))

    def mark_first_token(self, trace_id: str, at: str) -> None:
        self._enqueue(lambda: self.traces.mark_first_token(trace_id, at))

    def write_usage(
        self,
        rec: UsageRecord,
        *,
        alts: Sequence[UsageAltRecord] = (),
        parts: Iterable[TokenPartRecord] = (),
    ) -> None:
        alts = list(alts)
        parts = list(parts)

        def _write() -> None:
            with self.db.transaction():
                self.usage.upsert(rec)
                if alts:
                    self.usage.upsert_alts(alts)
                if parts:
                    self.usage.replace_parts(rec.trace_id, parts)

        self._enqueue(_write)

    def write_tool_call(self, rec: ToolCallRecord) -> None:
        self._enqueue(lambda: self.traces.insert_tool_call(rec))

    def write_anomaly(self, rec: AnomalyRecord) -> None:
        self._enqueue(lambda: self.traces.insert_anomaly(rec))

    # ── 队列控制 ──────────────────────────────────────────────────
    def _enqueue(self, op: _Op) -> None:
        with self._lock:
            self._pending += 1
            self._idle.clear()
        try:
            self._queue.put_nowait(op)
        except queue.Full:
            with self._lock:
                self._pending -= 1
                self.dropped += 1
                if self._pending == 0:
                    self._idle.set()
            log.warning("记录队列已满，丢弃 1 条写入（累计 %d）", self.dropped)

    def _run(self) -> None:
        while True:
            batch: list[_Op] = []
            try:
                first = self._queue.get(timeout=self._idle_wait)
            except queue.Empty:
                if self._stop.is_set():
                    break
                continue
            if first is None:
                break
            batch.append(first)
            while len(batch) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    self._stop.set()
                    break
                batch.append(item)
            for op in batch:
                try:
                    op()
                    self.written += 1
                except Exception as exc:  # noqa: BLE001 - 单条写失败不能中断整个写入线程
                    self.errors += 1
                    log.warning("写入失败: %s", exc)
                finally:
                    with self._lock:
                        self._pending -= 1
                        if self._pending == 0:
                            self._idle.set()
            if self._stop.is_set() and self._queue.empty():
                break

    def flush(self, timeout: float = 1.0) -> None:
        self._idle.wait(timeout)

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=5.0)
        self.flush(1.0)

    def stats(self) -> dict[str, int]:
        return {"written": self.written, "dropped": self.dropped, "errors": self.errors}
