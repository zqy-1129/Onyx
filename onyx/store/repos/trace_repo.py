from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

from onyx.core.clock import utc_now_iso
from onyx.store.codec import dumps, loads_dict
from onyx.store.db import Database
from onyx.store.records import AnomalyRecord, ToolCallRecord, TraceRecord

_TRACE_COLUMNS: tuple[str, ...] = (
    "id", "parent_id", "root_id", "kind", "purpose", "eval_run_id", "case_id", "sample_seq",
    "provider_id", "model_id", "model_name", "started_at", "first_token_at", "finished_at",
    "status", "error", "params_json", "messages_ref", "tools_ref", "rendered_prompt_ref",
    "output_ref", "raw_request_ref", "raw_response_ref", "finish_reason",
    "engine_latency_json", "gpu_json", "keep_alive", "contract_version", "extra_json",
)


class TraceRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ── trace ─────────────────────────────────────────────────────
    def upsert(self, rec: TraceRecord) -> None:
        values = self._params(rec)
        placeholders = ",".join("?" * len(_TRACE_COLUMNS))
        updates = ", ".join(f"{c}=excluded.{c}" for c in _TRACE_COLUMNS if c != "id")
        self.db.execute(
            f"INSERT INTO trace({','.join(_TRACE_COLUMNS)}) VALUES({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            values,
        )

    def finish(
        self,
        trace_id: str,
        *,
        status: str,
        finished_at: str | None = None,
        error: str | None = None,
        finish_reason: str | None = None,
        engine_latency: dict | None = None,
        output_ref: str | None = None,
        raw_response_ref: str | None = None,
        gpu: dict | None = None,
    ) -> None:
        """请求结束时补齐字段。只写非 None，避免把已有值抹掉。"""
        sets: list[str] = ["status=?", "finished_at=?"]
        params: list[object] = [status, finished_at or utc_now_iso()]
        for column, value in (
            ("error", error),
            ("finish_reason", finish_reason),
            ("engine_latency_json", dumps(engine_latency)),
            ("output_ref", output_ref),
            ("raw_response_ref", raw_response_ref),
            ("gpu_json", dumps(gpu)),
        ):
            if value is not None:
                sets.append(f"{column}=?")
                params.append(value)
        params.append(trace_id)
        self.db.execute(f"UPDATE trace SET {', '.join(sets)} WHERE id=?", params)

    def mark_first_token(self, trace_id: str, at: str) -> None:
        self.db.execute(
            "UPDATE trace SET first_token_at=? WHERE id=? AND first_token_at IS NULL", (at, trace_id)
        )

    def get(self, trace_id: str) -> TraceRecord | None:
        row = self.db.query_one("SELECT * FROM trace WHERE id=?", (trace_id,))
        return self._from_row(row) if row else None

    def list(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
        purpose: str | None = None,
        model_id: str | None = None,
        status: str | None = None,
        since: str | None = None,
    ) -> list[TraceRecord]:
        """游标分页：id 本身按时间排序，`WHERE id < cursor` 直接走主键索引。"""
        clauses: list[str] = []
        params: list[object] = []
        if cursor:
            clauses.append("id < ?")
            params.append(cursor)
        if purpose:
            clauses.append("purpose = ?")
            params.append(purpose)
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if since:
            clauses.append("started_at >= ?")
            params.append(since)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 1000)))
        rows = self.db.query(f"SELECT * FROM trace{where} ORDER BY id DESC LIMIT ?", params)
        return [self._from_row(r) for r in rows]

    def count(self, *, since: str | None = None, status: str | None = None) -> int:
        clauses: list[str] = []
        params: list[object] = []
        if since:
            clauses.append("started_at>=?")
            params.append(since)
        if status:
            clauses.append("status=?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return int(self.db.scalar(f"SELECT COUNT(*) FROM trace{where}", params, 0))

    # ── tool calls ────────────────────────────────────────────────
    def insert_tool_call(self, rec: ToolCallRecord) -> None:
        self.db.execute(
            """INSERT INTO tool_call(id, trace_id, step, call_id, name, args_json, args_raw,
                                     parse_status, parse_source, result_status, result_ref,
                                     result_bytes, started_at, latency_ms, tool_id, tool_def_hash,
                                     executed_by, extra_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 args_json=excluded.args_json, args_raw=excluded.args_raw,
                 parse_status=excluded.parse_status, parse_source=excluded.parse_source,
                 result_status=excluded.result_status, result_ref=excluded.result_ref,
                 result_bytes=excluded.result_bytes, latency_ms=excluded.latency_ms,
                 executed_by=excluded.executed_by, extra_json=excluded.extra_json""",
            (
                rec.id, rec.trace_id, rec.step, rec.call_id, rec.name, dumps(rec.args), rec.args_raw,
                rec.parse_status, rec.parse_source, rec.result_status, rec.result_ref,
                rec.result_bytes, rec.started_at, rec.latency_ms, rec.tool_id, rec.tool_def_hash,
                rec.executed_by, dumps(rec.extra),
            ),
        )

    def list_tool_calls(self, trace_id: str) -> list[ToolCallRecord]:
        rows = self.db.query("SELECT * FROM tool_call WHERE trace_id=? ORDER BY step, id", (trace_id,))
        return [
            ToolCallRecord(
                id=r["id"], trace_id=r["trace_id"], step=r["step"], parse_status=r["parse_status"],
                name=r["name"], call_id=r["call_id"], args=loads_dict(r["args_json"]) or None,
                args_raw=r["args_raw"], parse_source=r["parse_source"],
                result_status=r["result_status"], result_ref=r["result_ref"],
                result_bytes=r["result_bytes"], started_at=r["started_at"], latency_ms=r["latency_ms"],
                tool_id=r["tool_id"], tool_def_hash=r["tool_def_hash"], executed_by=r["executed_by"],
                extra=loads_dict(r["extra_json"]),
            )
            for r in rows
        ]

    # ── anomalies ─────────────────────────────────────────────────
    def insert_anomaly(self, rec: AnomalyRecord) -> None:
        self.db.execute(
            """INSERT INTO anomaly(id, trace_id, code, severity, detail_json, created_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET severity=excluded.severity, detail_json=excluded.detail_json""",
            (rec.id, rec.trace_id, rec.code, rec.severity, dumps(rec.detail), rec.created_at or utc_now_iso()),
        )

    def list_anomalies(self, *, limit: int = 50, code: str | None = None) -> list[AnomalyRecord]:
        if code:
            rows = self.db.query(
                "SELECT * FROM anomaly WHERE code=? ORDER BY id DESC LIMIT ?", (code, limit)
            )
        else:
            rows = self.db.query("SELECT * FROM anomaly ORDER BY id DESC LIMIT ?", (limit,))
        return [
            AnomalyRecord(
                id=r["id"], code=r["code"], severity=r["severity"], trace_id=r["trace_id"],
                detail=loads_dict(r["detail_json"]), created_at=r["created_at"],
            )
            for r in rows
        ]

    def anomaly_counts(self, *, since: str | None = None) -> dict[str, int]:
        sql = "SELECT code, COUNT(*) AS n FROM anomaly"
        params: Sequence[object] = ()
        if since:
            sql += " WHERE created_at>=?"
            params = (since,)
        sql += " GROUP BY code ORDER BY n DESC"
        return {str(r["code"]): int(r["n"]) for r in self.db.query(sql, params)}

    # ── 内部 ──────────────────────────────────────────────────────
    def _params(self, rec: TraceRecord) -> tuple:
        data = asdict(rec)
        return (
            data["id"], data["parent_id"], data["root_id"] or data["id"], data["kind"], data["purpose"],
            data["eval_run_id"], data["case_id"], data["sample_seq"], data["provider_id"],
            data["model_id"], data["model_name"], data["started_at"], data["first_token_at"],
            data["finished_at"], data["status"], data["error"], dumps(data["params"]),
            data["messages_ref"], data["tools_ref"], data["rendered_prompt_ref"], data["output_ref"],
            data["raw_request_ref"], data["raw_response_ref"], data["finish_reason"],
            dumps(data["engine_latency"]), dumps(data["gpu"]), data["keep_alive"],
            data["contract_version"], dumps(data["extra"]),
        )

    def _from_row(self, row) -> TraceRecord:
        return TraceRecord(
            id=row["id"], kind=row["kind"], purpose=row["purpose"], started_at=row["started_at"],
            status=row["status"], parent_id=row["parent_id"], root_id=row["root_id"],
            eval_run_id=row["eval_run_id"], case_id=row["case_id"], sample_seq=row["sample_seq"],
            provider_id=row["provider_id"], model_id=row["model_id"], model_name=row["model_name"],
            first_token_at=row["first_token_at"], finished_at=row["finished_at"], error=row["error"],
            params=loads_dict(row["params_json"]), messages_ref=row["messages_ref"],
            tools_ref=row["tools_ref"], rendered_prompt_ref=row["rendered_prompt_ref"],
            output_ref=row["output_ref"], raw_request_ref=row["raw_request_ref"],
            raw_response_ref=row["raw_response_ref"], finish_reason=row["finish_reason"],
            engine_latency=loads_dict(row["engine_latency_json"]), gpu=loads_dict(row["gpu_json"]),
            keep_alive=row["keep_alive"], contract_version=row["contract_version"],
            extra=loads_dict(row["extra_json"]),
        )
