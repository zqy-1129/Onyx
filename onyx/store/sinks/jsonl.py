"""NDJSON 原始事件日志。

用途：① 事件契约的回归证据（重放旧日志能验证 visitor 是否兼容）
     ② 出问题时脱离 DB 直接看原始流
     ③ 未来导出到 Langfuse/OTLP 的中间格式
"""

from __future__ import annotations

import threading
from pathlib import Path

from onyx.core.event import TraceEvent


class JsonlEventSink:
    name = "jsonl"

    def __init__(self, path: Path | str, *, buffer_limit: int = 256) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[str] = []
        self._lock = threading.Lock()
        self._buffer_limit = max(1, buffer_limit)
        self.written = 0

    def emit(self, event: TraceEvent) -> None:
        line = event.to_json()
        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self._buffer_limit:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(self._buffer) + "\n")
        self.written += len(self._buffer)
        self._buffer.clear()

    def flush(self, timeout: float = 1.0) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        self.flush()

    def __len__(self) -> int:
        return self.written
