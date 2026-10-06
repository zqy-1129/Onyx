"""性能基线的读写（S36）。

这张表只回答两个问题：**这次在什么条件下测**、**每格测出来多少**。
写入方只有 `onyx perf run` 一处，读取方是 `ls` / `show` / `compare`。

两条纪律：
- `status != "measured"` 的格子照样落一行（带 `reason`）。"没测到"是一行可查的记录，
  不是缺失的行——缺了就没法回答"这次到底欠哪几格"，而半截的基线必须能被看出欠多少。
- `metrics_json` 里的 `None` 保持 `null`。落库时补 0 等于把"未知"写成"零延迟"，
  那是这个项目反复踩的同一颗坑。
"""

from __future__ import annotations

from collections.abc import Sequence

from onyx.core.ids import new_trace_id
from onyx.store.codec import dumps, loads_dict, loads_list
from onyx.store.db import Database
from onyx.store.records import PerfCellRecord, PerfRunRecord

_RUN_COLUMNS: tuple[str, ...] = (
    "id", "started_at", "finished_at", "status", "provider_id", "engine_version", "model",
    "quantization", "device", "num_ctx", "keep_alive", "stream", "timing_source", "env_hash",
    "comparable", "conditions_json", "grid_json", "elapsed_s", "n_requests", "app_version",
    "git_rev", "note", "error",
)
_CELL_COLUMNS: tuple[str, ...] = (
    "run_id", "cell_key", "phase", "prompt_chars", "target_tokens", "concurrency", "repeat",
    "status", "reason", "n_requests", "n_measured", "metrics_json", "trace_ids_json",
)


class PerfRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def new_id() -> str:
        return new_trace_id()

    # ── 写 ────────────────────────────────────────────────────────
    def insert_run(self, rec: PerfRunRecord) -> None:
        self.db.execute(
            f"""INSERT INTO perf_run({",".join(_RUN_COLUMNS)})
                VALUES({",".join("?" * len(_RUN_COLUMNS))})
                ON CONFLICT(id) DO UPDATE SET
                  finished_at=excluded.finished_at, status=excluded.status,
                  elapsed_s=excluded.elapsed_s, n_requests=excluded.n_requests,
                  note=excluded.note, error=excluded.error""",
            (
                rec.id, rec.started_at, rec.finished_at, rec.status, rec.provider_id,
                rec.engine_version, rec.model, rec.quantization, rec.device, rec.num_ctx,
                rec.keep_alive, 1 if rec.stream else 0, rec.timing_source, rec.env_hash,
                1 if rec.comparable else 0,
                # 条件快照是这张表的立身之本：空也要落成 `{}` 而不是 NULL，
                # 否则"没记条件"与"条件为空"两种情况会混成一种
                dumps(rec.conditions) or "{}", dumps(rec.grid) or "{}",
                rec.elapsed_s, rec.n_requests, rec.app_version, rec.git_rev, rec.note, rec.error,
            ),
        )

    def insert_cells(self, recs: Sequence[PerfCellRecord]) -> None:
        for rec in recs:
            self.db.execute(
                f"""INSERT INTO perf_cell({",".join(_CELL_COLUMNS)})
                    VALUES({",".join("?" * len(_CELL_COLUMNS))})
                    ON CONFLICT(run_id, cell_key) DO UPDATE SET
                      status=excluded.status, reason=excluded.reason,
                      n_requests=excluded.n_requests, n_measured=excluded.n_measured,
                      metrics_json=excluded.metrics_json, trace_ids_json=excluded.trace_ids_json""",
                (
                    rec.run_id, rec.cell_key, rec.phase, rec.prompt_chars, rec.target_tokens,
                    rec.concurrency, rec.repeat, rec.status, rec.reason,
                    rec.n_requests, rec.n_measured,
                    dumps(rec.metrics) or "{}", dumps(list(rec.trace_ids)) or "[]",
                ),
            )

    # ── 读 ────────────────────────────────────────────────────────
    def list_runs(self, *, model: str | None = None, limit: int = 20) -> list[PerfRunRecord]:
        sql = "SELECT * FROM perf_run"
        params: list[object] = []
        if model:
            sql += " WHERE model=?"
            params.append(model)
        sql += " ORDER BY started_at DESC, id DESC LIMIT ?"
        params.append(limit)
        return [self._run(row) for row in self.db.query(sql, tuple(params))]

    def get_run(self, run_id: str) -> PerfRunRecord | None:
        rows = self.db.query("SELECT * FROM perf_run WHERE id=?", (run_id,))
        return self._run(rows[0]) if rows else None

    def cells(self, run_id: str) -> list[PerfCellRecord]:
        rows = self.db.query(
            "SELECT * FROM perf_cell WHERE run_id=? ORDER BY phase, prompt_chars,"
            " target_tokens, concurrency", (run_id,))
        return [self._cell(row) for row in rows]

    def find_by_env(self, env_hash: str, *, before: str | None = None) -> list[PerfRunRecord]:
        """同条件的历史基线（新→旧）。`before` 用来排除自己那次。"""
        sql = "SELECT * FROM perf_run WHERE env_hash=?"
        params: list[object] = [env_hash]
        if before:
            sql += " AND id<>?"
            params.append(before)
        sql += " ORDER BY started_at DESC, id DESC LIMIT 50"
        return [self._run(row) for row in self.db.query(sql, tuple(params))]

    # ── 行 → 记录 ─────────────────────────────────────────────────
    @staticmethod
    def _run(row) -> PerfRunRecord:
        return PerfRunRecord(
            id=row["id"], started_at=row["started_at"], finished_at=row["finished_at"],
            status=row["status"], model=row["model"], provider_id=row["provider_id"],
            engine_version=row["engine_version"], quantization=row["quantization"],
            device=row["device"], num_ctx=row["num_ctx"], keep_alive=row["keep_alive"],
            stream=bool(row["stream"]), timing_source=row["timing_source"],
            env_hash=row["env_hash"], comparable=bool(row["comparable"]),
            conditions=loads_dict(row["conditions_json"]), grid=loads_dict(row["grid_json"]),
            elapsed_s=float(row["elapsed_s"] or 0.0), n_requests=int(row["n_requests"] or 0),
            app_version=row["app_version"], git_rev=row["git_rev"],
            note=row["note"], error=row["error"],
        )

    @staticmethod
    def _cell(row) -> PerfCellRecord:
        return PerfCellRecord(
            run_id=row["run_id"], cell_key=row["cell_key"], phase=row["phase"],
            prompt_chars=int(row["prompt_chars"]), target_tokens=int(row["target_tokens"]),
            concurrency=int(row["concurrency"]), repeat=int(row["repeat"]),
            status=row["status"], reason=row["reason"],
            n_requests=int(row["n_requests"]), n_measured=int(row["n_measured"]),
            metrics=loads_dict(row["metrics_json"]),
            trace_ids=tuple(loads_list(row["trace_ids_json"])),
        )
