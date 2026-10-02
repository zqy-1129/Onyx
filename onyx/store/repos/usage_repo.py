from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from onyx.store.codec import dumps, loads_dict
from onyx.store.db import Database
from onyx.store.records import TokenPartRecord, UsageAltRecord, UsageRecord


@dataclass(frozen=True, slots=True)
class UsageBundle:
    """一条 trace 的完整计量视图：采信值 + 各来源明细 + 归因。"""

    usage: UsageRecord | None = None
    alts: tuple[UsageAltRecord, ...] = ()
    parts: tuple[TokenPartRecord, ...] = ()

    def alt(self, source: str) -> UsageAltRecord | None:
        return next((a for a in self.alts if str(a.source) == source), None)

    def part_tokens(self, part: str) -> int:
        return sum(p.tokens for p in self.parts if p.part == part)

    @property
    def has_attribution(self) -> bool:
        return bool(self.parts)


@dataclass(slots=True)
class UsageSummary:
    """聚合结果（Token Ledger 页用）。"""

    traces: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    thinking_tokens: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    by_confidence: dict[str, int] = field(default_factory=dict)
    drift_samples: list[tuple[str, float]] = field(default_factory=list)


class UsageRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert(self, rec: UsageRecord) -> None:
        self.db.execute(
            """INSERT INTO usage(trace_id, in_tokens, out_tokens, thinking_tokens, cached_tokens,
                                 source, confidence, ttft_ms, prefill_tps, decode_tps, wall_ms,
                                 bytes_out, drift_pct, extra_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(trace_id) DO UPDATE SET
                 in_tokens=excluded.in_tokens, out_tokens=excluded.out_tokens,
                 thinking_tokens=excluded.thinking_tokens, cached_tokens=excluded.cached_tokens,
                 source=excluded.source, confidence=excluded.confidence, ttft_ms=excluded.ttft_ms,
                 prefill_tps=excluded.prefill_tps, decode_tps=excluded.decode_tps,
                 wall_ms=excluded.wall_ms, bytes_out=excluded.bytes_out,
                 drift_pct=excluded.drift_pct, extra_json=excluded.extra_json""",
            (
                rec.trace_id, rec.in_tokens, rec.out_tokens, rec.thinking_tokens, rec.cached_tokens,
                str(rec.source), str(rec.confidence), rec.ttft_ms, rec.prefill_tps, rec.decode_tps,
                rec.wall_ms, rec.bytes_out, rec.drift_pct, dumps(rec.extra),
            ),
        )

    def upsert_alts(self, records: Iterable[UsageAltRecord]) -> None:
        self.db.executemany(
            """INSERT INTO usage_alt(trace_id, source, in_tokens, out_tokens, thinking_tokens,
                                     cached_tokens, ok, confidence, note)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(trace_id, source) DO UPDATE SET
                 in_tokens=excluded.in_tokens, out_tokens=excluded.out_tokens,
                 thinking_tokens=excluded.thinking_tokens, cached_tokens=excluded.cached_tokens,
                 ok=excluded.ok, confidence=excluded.confidence, note=excluded.note""",
            [
                (
                    r.trace_id, str(r.source), r.in_tokens, r.out_tokens, r.thinking_tokens,
                    r.cached_tokens, int(r.ok), None if r.confidence is None else str(r.confidence), r.note,
                )
                for r in records
            ],
        )

    def replace_parts(self, trace_id: str, parts: Iterable[TokenPartRecord]) -> None:
        """归因整体替换：重算时必须清掉旧分段，否则同一 part 会重复累加。"""
        rows = [(trace_id, p.part, p.ord, p.tokens, p.bytes) for p in parts]
        with self.db.transaction():
            self.db.execute("DELETE FROM token_part WHERE trace_id=?", (trace_id,))
            self.db.executemany(
                "INSERT INTO token_part(trace_id, part, ord, tokens, bytes) VALUES(?,?,?,?,?)", rows
            )

    def fetch(self, trace_id: str) -> UsageBundle:
        row = self.db.query_one("SELECT * FROM usage WHERE trace_id=?", (trace_id,))
        usage = None
        if row:
            usage = UsageRecord(
                trace_id=row["trace_id"], source=row["source"], confidence=row["confidence"],
                in_tokens=row["in_tokens"], out_tokens=row["out_tokens"],
                thinking_tokens=row["thinking_tokens"], cached_tokens=row["cached_tokens"],
                ttft_ms=row["ttft_ms"], prefill_tps=row["prefill_tps"], decode_tps=row["decode_tps"],
                wall_ms=row["wall_ms"], bytes_out=row["bytes_out"], drift_pct=row["drift_pct"],
                extra=loads_dict(row["extra_json"]),
            )
        alts = tuple(
            UsageAltRecord(
                trace_id=r["trace_id"], source=r["source"], in_tokens=r["in_tokens"],
                out_tokens=r["out_tokens"], thinking_tokens=r["thinking_tokens"],
                cached_tokens=r["cached_tokens"], ok=bool(r["ok"]), confidence=r["confidence"],
                note=r["note"] or "",
            )
            for r in self.db.query("SELECT * FROM usage_alt WHERE trace_id=? ORDER BY source", (trace_id,))
        )
        parts = tuple(
            TokenPartRecord(
                trace_id=r["trace_id"], part=r["part"], ord=r["ord"], tokens=r["tokens"], bytes=r["bytes"]
            )
            for r in self.db.query(
                "SELECT * FROM token_part WHERE trace_id=? ORDER BY ord, part", (trace_id,)
            )
        )
        return UsageBundle(usage=usage, alts=alts, parts=parts)

    def summarize(self, *, since: str | None = None, model_id: str | None = None) -> UsageSummary:
        where: list[str] = []
        params: list[object] = []
        if since:
            where.append("t.started_at>=?")
            params.append(since)
        if model_id:
            where.append("t.model_id=?")
            params.append(model_id)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        join = "FROM usage u JOIN trace t ON t.id=u.trace_id"
        row = self.db.query_one(
            f"""SELECT COUNT(u.trace_id) AS n,
                       COALESCE(SUM(u.in_tokens),0)  AS i,
                       COALESCE(SUM(u.out_tokens),0) AS o,
                       COALESCE(SUM(u.thinking_tokens),0) AS th
                {join}{clause}""",
            params,
        )
        summary = UsageSummary(
            traces=int(row["n"]), in_tokens=int(row["i"]), out_tokens=int(row["o"]),
            thinking_tokens=int(row["th"]),
        )
        for key_column, target in (("source", summary.by_source), ("confidence", summary.by_confidence)):
            for r in self.db.query(
                f"SELECT u.{key_column} AS k, COUNT(*) AS n {join}{clause} GROUP BY u.{key_column}",
                params,
            ):
                target[str(r["k"])] = int(r["n"])
        drift_where = [*where, "u.drift_pct IS NOT NULL"]
        summary.drift_samples = [
            (str(r["trace_id"]), float(r["drift_pct"]))
            for r in self.db.query(
                f"SELECT u.trace_id, u.drift_pct {join} WHERE {' AND '.join(drift_where)}", params
            )
        ]
        return summary
