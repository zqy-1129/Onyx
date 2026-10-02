from __future__ import annotations

from onyx.core.clock import utc_now_iso
from onyx.core.ids import new_trace_id
from onyx.store.codec import dumps, loads_dict, loads_list
from onyx.store.db import Database
from onyx.store.records import ToolDefRecord, ToolRunRecord, ToolTestRecord


class ToolRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ── 定义 ──────────────────────────────────────────────────────
    def upsert_def(self, rec: ToolDefRecord) -> str:
        now = utc_now_iso()
        self.db.execute(
            """INSERT INTO tool_def(id, name, version, kind, schema_json, impl_ref, hash, tokens,
                                    bytes, tags, owner, enabled, side_effect, timeout_ms, doc,
                                    examples_json, extra_json, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET
                 version=excluded.version, kind=excluded.kind, schema_json=excluded.schema_json,
                 impl_ref=excluded.impl_ref, hash=excluded.hash, tokens=excluded.tokens,
                 bytes=excluded.bytes, tags=excluded.tags, owner=excluded.owner,
                 enabled=excluded.enabled, side_effect=excluded.side_effect,
                 timeout_ms=excluded.timeout_ms, doc=excluded.doc,
                 examples_json=excluded.examples_json, extra_json=excluded.extra_json,
                 updated_at=excluded.updated_at""",
            (
                rec.id, rec.name, rec.version, rec.kind,
                dumps({"description": rec.description, "parameters": rec.schema_json}),
                rec.impl_ref, rec.hash, rec.tokens, rec.bytes, ",".join(rec.tags), rec.owner,
                int(rec.enabled), rec.side_effect, rec.timeout_ms, rec.doc,
                dumps([dict(e) for e in rec.examples]),
                dumps(rec.extra), rec.created_at or now, now,
            ),
        )
        row = self.db.query_one("SELECT id FROM tool_def WHERE name=?", (rec.name,))
        return str(row["id"])

    def find_by_name(self, name: str) -> ToolDefRecord | None:
        row = self.db.query_one("SELECT * FROM tool_def WHERE name=?", (name,))
        return self._def_from_row(row) if row else None

    def get_def(self, tool_id: str) -> ToolDefRecord | None:
        row = self.db.query_one("SELECT * FROM tool_def WHERE id=?", (tool_id,))
        return self._def_from_row(row) if row else None

    def list_defs(self, *, enabled_only: bool = False, kind: str | None = None) -> list[ToolDefRecord]:
        clauses, params = [], []
        if enabled_only:
            clauses.append("enabled=1")
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(f"SELECT * FROM tool_def{where} ORDER BY name", params)
        return [self._def_from_row(r) for r in rows]

    def set_enabled(self, tool_id: str, enabled: bool) -> None:
        self.db.execute(
            "UPDATE tool_def SET enabled=?, updated_at=? WHERE id=?",
            (int(enabled), utc_now_iso(), tool_id),
        )

    def delete_def(self, tool_id: str) -> None:
        self.db.execute("DELETE FROM tool_def WHERE id=?", (tool_id,))

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM tool_def", default=0))

    def _def_from_row(self, row) -> ToolDefRecord:
        schema_blob = loads_dict(row["schema_json"])
        parameters = schema_blob.get("parameters")
        return ToolDefRecord(
            id=row["id"], name=row["name"], version=row["version"], kind=row["kind"],
            schema_json=dict(parameters) if isinstance(parameters, dict) else {},
            description=str(schema_blob.get("description") or ""),
            hash=row["hash"], impl_ref=row["impl_ref"] or "", tokens=row["tokens"],
            bytes=row["bytes"], tags=tuple(t for t in (row["tags"] or "").split(",") if t),
            owner=row["owner"] or "", enabled=bool(row["enabled"]),
            side_effect=row["side_effect"] or "read", timeout_ms=row["timeout_ms"],
            doc=row["doc"] or "", examples=tuple(loads_list(row["examples_json"])),
            extra=loads_dict(row["extra_json"]),
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    # ── 契约测试 ──────────────────────────────────────────────────
    def add_test(self, rec: ToolTestRecord) -> str:
        self.db.execute(
            """INSERT INTO tool_test(id, tool_id, name, args_json, expect_json, checks_json, live, created_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 args_json=excluded.args_json, expect_json=excluded.expect_json,
                 checks_json=excluded.checks_json, live=excluded.live""",
            (
                rec.id, rec.tool_id, rec.name, dumps(rec.args), dumps(rec.expect),
                dumps([dict(c) for c in rec.checks]), int(rec.live), rec.created_at or utc_now_iso(),
            ),
        )
        return rec.id

    def list_tests(self, tool_id: str | None = None) -> list[ToolTestRecord]:
        if tool_id:
            rows = self.db.query("SELECT * FROM tool_test WHERE tool_id=? ORDER BY name", (tool_id,))
        else:
            rows = self.db.query("SELECT * FROM tool_test ORDER BY tool_id, name")
        return [
            ToolTestRecord(
                id=r["id"], tool_id=r["tool_id"], name=r["name"], args=loads_dict(r["args_json"]),
                expect=loads_dict(r["expect_json"]) or None,
                checks=tuple(loads_list(r["checks_json"])), live=bool(r["live"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    # ── 执行记录 ──────────────────────────────────────────────────
    def insert_run(self, rec: ToolRunRecord) -> str:
        self.db.execute(
            """INSERT INTO tool_run(id, tool_id, tool_def_hash, test_id, trace_id, started_at,
                                    latency_ms, status, output_ref, error, deterministic,
                                    idempotent, extra_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.id, rec.tool_id, rec.tool_def_hash, rec.test_id, rec.trace_id,
                rec.started_at or utc_now_iso(), rec.latency_ms, rec.status, rec.output_ref,
                rec.error,
                None if rec.deterministic is None else int(rec.deterministic),
                None if rec.idempotent is None else int(rec.idempotent),
                dumps(rec.extra),
            ),
        )
        return rec.id

    def list_runs(self, *, tool_id: str | None = None, limit: int = 50) -> list[ToolRunRecord]:
        if tool_id:
            rows = self.db.query(
                "SELECT * FROM tool_run WHERE tool_id=? ORDER BY id DESC LIMIT ?", (tool_id, limit)
            )
        else:
            rows = self.db.query("SELECT * FROM tool_run ORDER BY id DESC LIMIT ?", (limit,))
        return [
            ToolRunRecord(
                id=r["id"], status=r["status"], started_at=r["started_at"], tool_id=r["tool_id"],
                tool_def_hash=r["tool_def_hash"], test_id=r["test_id"], trace_id=r["trace_id"],
                latency_ms=r["latency_ms"], output_ref=r["output_ref"], error=r["error"],
                deterministic=None if r["deterministic"] is None else bool(r["deterministic"]),
                idempotent=None if r["idempotent"] is None else bool(r["idempotent"]),
                extra=loads_dict(r["extra_json"]),
            )
            for r in rows
        ]

    @staticmethod
    def new_id() -> str:
        return new_trace_id()
