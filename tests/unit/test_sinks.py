from __future__ import annotations

import json
import threading

import pytest

from onyx.core.clock import FakeClock
from onyx.core.event import EventType, TraceEvent, make_event
from onyx.core.ids import new_trace_id
from onyx.store.db import Database
from onyx.store.records import (
    AnomalyRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)
from onyx.store.repos import TraceRepo, UsageRepo
from onyx.store.sinks import (
    EventFanout,
    JsonlEventSink,
    NullEventSink,
    NullRecordSink,
    RecordFanout,
    SqliteRecordSink,
)


class BoomSink:
    name = "boom"

    def emit(self, event: TraceEvent) -> None:
        raise RuntimeError("下游炸了")

    def flush(self, timeout: float = 1.0) -> None:
        raise RuntimeError("flush 也炸")

    def close(self) -> None:
        raise RuntimeError("close 也炸")


def _event(clock: FakeClock | None = None) -> TraceEvent:
    return make_event(
        EventType.TRACE_END, new_trace_id(), {"status": "ok", "wall_ms": 12.0},
        clock=clock or FakeClock(),
    )


# ── JSONL 事件日志 ─────────────────────────────────────────────────
def test_jsonl_sink_writes_ndjson(tmp_path):
    path = tmp_path / "events.ndjson"
    sink = JsonlEventSink(path, buffer_limit=2)
    for _ in range(5):
        sink.emit(_event())
    sink.flush()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert first["type"] == "trace_end" and first["payload"]["wall_ms"] == 12.0
    assert first["v"] == 1
    assert len(sink) == 5


def test_jsonl_sink_creates_parent_dirs(tmp_path):
    sink = JsonlEventSink(tmp_path / "nested" / "dir" / "e.ndjson")
    sink.emit(_event())
    sink.close()
    assert (tmp_path / "nested" / "dir" / "e.ndjson").exists()


# ── Fanout 错误隔离 ────────────────────────────────────────────────
def test_event_fanout_isolates_failures():
    """一个 sink 崩了，其余 sink 必须照常收到事件（原则 2）。"""
    good = NullEventSink()
    fanout = EventFanout([BoomSink(), good])
    for _ in range(3):
        fanout.emit(_event())
    fanout.flush()
    fanout.close()
    assert good.count == 3
    assert fanout.errors["boom"] == 5  # 3 emit + 1 flush + 1 close


class BoomRecordSink(NullRecordSink):
    name = "boom"

    def write_trace(self, rec: TraceRecord) -> None:
        raise RuntimeError("炸")


def test_record_fanout_isolates_failures():
    good = NullRecordSink()
    fanout = RecordFanout([BoomRecordSink(), good])
    fanout.write_trace(TraceRecord(id="t1", kind="generation", purpose="chat", started_at="x"))
    fanout.write_anomaly(AnomalyRecord(id="a1", code="C", severity="warn"))
    assert good.traces == 1 and good.anomalies == 1
    assert fanout.errors["boom"] == 1


# ── SQLite sink ────────────────────────────────────────────────────
@pytest.fixture
def sink(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    writer = SqliteRecordSink(db, batch_size=32, idle_wait=0.01)
    yield writer, db
    writer.close()
    db.close()


def test_sqlite_sink_concurrent_writes_never_lose_rows(sink):
    """8 线程 × 63 = 504 条并发写入 → 落库必须恰好 504 条（不丢不重）。"""
    writer, db = sink
    traces = TraceRepo(db)
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            for _ in range(n):
                writer.write_trace(TraceRecord(
                    id=new_trace_id(), kind="generation", purpose="chat",
                    started_at="2026-10-02T10:00:00.000000+00:00",
                ))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(63,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.flush(5.0)

    assert not errors
    assert writer.stats()["dropped"] == 0, "队列未满时不许丢弃"
    assert traces.count() == 504, f"实际 {traces.count()} 条"


def test_sqlite_sink_writes_usage_bundle(sink):
    writer, db = sink
    usage_repo = UsageRepo(db)
    trace_id = new_trace_id()
    writer.write_trace(TraceRecord(
        id=trace_id, kind="generation", purpose="eval:tool_selection",
        started_at="2026-10-02T10:00:00.000000+00:00", eval_run_id="run-1", case_id="c-1",
    ))
    writer.write_usage(
        UsageRecord(trace_id=trace_id, source="engine", confidence="high", in_tokens=100, out_tokens=20),
        alts=[
            UsageAltRecord(trace_id=trace_id, source="engine", in_tokens=100, out_tokens=20),
            UsageAltRecord(trace_id=trace_id, source="hf_tokenizer", in_tokens=103, out_tokens=20),
        ],
        parts=[TokenPartRecord(trace_id=trace_id, part="tool_defs", ord=0, tokens=60)],
    )
    writer.write_tool_call(ToolCallRecord(
        id="tc1", trace_id=trace_id, step=1, parse_status="ok", name="weather",
    ))
    writer.write_anomaly(AnomalyRecord(id="an1", code="TOKEN_DRIFT", severity="warn", trace_id=trace_id))
    writer.flush(5.0)

    bundle = usage_repo.fetch(trace_id)
    assert bundle.usage.in_tokens == 100
    assert len(bundle.alts) == 2
    assert bundle.part_tokens("tool_defs") == 60
    assert len(TraceRepo(db).list_tool_calls(trace_id)) == 1
    assert TraceRepo(db).anomaly_counts() == {"TOKEN_DRIFT": 1}
    assert writer.stats()["errors"] == 0


def test_sqlite_sink_survives_bad_record(sink):
    """单条写入失败只计数，不能带崩写入线程。"""
    writer, db = sink
    writer.write_trace(TraceRecord(id="t1", kind="generation", purpose="chat", started_at="x"))
    writer.write_usage(UsageRecord(trace_id="不存在的trace", source="engine", confidence="high"))
    writer.write_trace(TraceRecord(id="t2", kind="generation", purpose="chat", started_at="y"))
    writer.flush(5.0)
    stats = writer.stats()
    assert stats["errors"] == 1, "外键约束失败必须被记录为 error 而不是静默通过"
    assert TraceRepo(db).count() == 2


def test_sink_close_is_idempotent(tmp_path):
    db = Database(tmp_path / "t.sqlite")
    writer = SqliteRecordSink(db)
    writer.close()
    writer.close()
    db.close()
