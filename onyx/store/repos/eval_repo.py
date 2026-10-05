"""评测存储：数据集、样本、任务、运行、逐条 grade。

两条查询语义值得单独说明：
- `list_graded_case_ids` 支撑 `--resume`：靠它跳过已评过的 case，
  断点续跑才不会重复计费（每次重复都是真实的 GPU 时间）。
- `grade` 上有 `UNIQUE(eval_run_id, case_id, seq)`，所以重跑同一 case 是 upsert
  而不是追加——中断后重来不会产生两份分数，聚合值也就不会被悄悄稀释。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from onyx.core.clock import utc_now_iso
from onyx.core.ids import new_trace_id
from onyx.store.codec import dumps, loads_dict, loads_list
from onyx.store.db import Database
from onyx.store.records import (
    CaseRecord,
    DatasetRecord,
    GradeRecord,
    RunRecord,
    TaskRecord,
)


class EvalRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ── 数据集 ────────────────────────────────────────────────────
    def upsert_dataset(self, rec: DatasetRecord) -> str:
        self.db.execute(
            """INSERT INTO dataset(id, upstream, revision, license, split_json, n_cases,
                                   loader, notes, imported_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 upstream=excluded.upstream, revision=excluded.revision,
                 license=excluded.license, split_json=excluded.split_json,
                 n_cases=excluded.n_cases, loader=excluded.loader, notes=excluded.notes,
                 imported_at=excluded.imported_at""",
            (rec.id, rec.upstream, rec.revision, rec.license, dumps(rec.splits), rec.n_cases,
             rec.loader, rec.notes, rec.imported_at or utc_now_iso()),
        )
        return rec.id

    def get_dataset(self, dataset_id: str) -> DatasetRecord | None:
        row = self.db.query_one("SELECT * FROM dataset WHERE id=?", (dataset_id,))
        return self._dataset_from_row(row) if row else None

    def list_datasets(self) -> list[DatasetRecord]:
        return [self._dataset_from_row(r) for r in
                self.db.query("SELECT * FROM dataset ORDER BY imported_at DESC")]

    # ── 样本 ──────────────────────────────────────────────────────
    def upsert_case(self, rec: CaseRecord) -> str:
        self.db.execute(
            """INSERT INTO eval_case(id, dataset_id, ord, kind, input_json, tools_json,
                                     expect_json, fixture_json, meta_json, tags)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 dataset_id=excluded.dataset_id, ord=excluded.ord, kind=excluded.kind,
                 input_json=excluded.input_json, tools_json=excluded.tools_json,
                 expect_json=excluded.expect_json, fixture_json=excluded.fixture_json,
                 meta_json=excluded.meta_json, tags=excluded.tags""",
            (rec.id, rec.dataset_id, rec.ord, rec.kind, dumps(rec.input),
             dumps([dict(t) for t in rec.tools]) if rec.tools else None,
             dumps(rec.expect), dumps(rec.fixture) if rec.fixture else None,
             dumps(rec.meta), ",".join(rec.tags)),
        )
        return rec.id

    def upsert_cases(self, records: Iterable[CaseRecord]) -> int:
        count = 0
        with self.db.transaction():
            for rec in records:
                self.upsert_case(rec)
                count += 1
        return count

    def list_cases(
        self,
        dataset_id: str,
        *,
        kind: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[CaseRecord]:
        clauses = ["dataset_id=?"]
        params: list[object] = [dataset_id]
        if kind:
            clauses.append("kind=?")
            params.append(kind)
        sql = f"SELECT * FROM eval_case WHERE {' AND '.join(clauses)} ORDER BY ord, id"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        return [self._case_from_row(r) for r in self.db.query(sql, tuple(params))]

    def count_cases(self, dataset_id: str) -> int:
        row = self.db.query_one("SELECT COUNT(*) AS n FROM eval_case WHERE dataset_id=?",
                                (dataset_id,))
        return int(row["n"]) if row else 0

    def prune_stale_cases(self, dataset_id: str, keep_ids: Iterable[str]) -> int:
        """删掉这份数据集里既不属于当前样本集、也**没有被任何 grade 引用**的旧行。

        case id 是内容哈希，所以生成器一改就是一批新 id。旧行全留着，`list_cases` 就会把两个
        版本的样本混在一起数；旧行乱删，历史分数就点不回它那条样本（这正是"删除数据集"没做的原因）。
        所以判据是引用完整性，不是"看起来旧"。
        """
        keep = [str(case_id) for case_id in keep_ids]
        where = ["dataset_id=?", "id NOT IN (SELECT case_id FROM grade)"]
        params: list[object] = [dataset_id]
        if keep:
            where.insert(1, f"id NOT IN ({','.join('?' * len(keep))})")
            params.extend(keep)
        cursor = self.db.execute(f"DELETE FROM eval_case WHERE {' AND '.join(where)}", tuple(params))
        return int(cursor.rowcount or 0)

    # ── 任务 ──────────────────────────────────────────────────────
    def upsert_task(self, rec: TaskRecord) -> str:
        self.db.execute(
            """INSERT INTO eval_task(id, name, dataset_id, metrics_json, grader_json,
                                     sample_params_json, k, budget_json, sandbox, extra_json)
               VALUES(?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 name=excluded.name, dataset_id=excluded.dataset_id,
                 metrics_json=excluded.metrics_json, grader_json=excluded.grader_json,
                 sample_params_json=excluded.sample_params_json, k=excluded.k,
                 budget_json=excluded.budget_json, sandbox=excluded.sandbox,
                 extra_json=excluded.extra_json""",
            (rec.id, rec.name, rec.dataset_id, dumps(list(rec.metrics)), dumps(rec.grader),
             dumps(rec.sample_params), rec.k, dumps(rec.budget), int(rec.sandbox),
             dumps(rec.extra)),
        )
        return rec.id

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self.db.query_one("SELECT * FROM eval_task WHERE id=?", (task_id,))
        return self._task_from_row(row) if row else None

    # ── 运行 ──────────────────────────────────────────────────────
    def insert_run(self, rec: RunRecord) -> str:
        self.db.execute(
            """INSERT INTO eval_run(id, task_id, model_id, provider_id, started_at, finished_at,
                                    status, seed, app_version, git_rev, params_snapshot_json,
                                    config_json, n_cases, n_done, n_error, n_skipped,
                                    aggregate_json, cost_json, notes,
                                    dataset_id, dataset_revision)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (rec.id, rec.task_id, rec.model_id, rec.provider_id, rec.started_at, rec.finished_at,
             rec.status, rec.seed, rec.app_version, rec.git_rev, dumps(rec.params_snapshot),
             dumps(rec.config), rec.n_cases, rec.n_done, rec.n_error, rec.n_skipped,
             dumps(rec.aggregate), dumps(rec.cost), rec.notes,
             rec.dataset_id, rec.dataset_revision),
        )
        return rec.id

    def update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        finished_at: str | None = None,
        n_done: int | None = None,
        n_error: int | None = None,
        n_skipped: int | None = None,
        n_cases: int | None = None,
        aggregate: dict | None = None,
        cost: dict | None = None,
        notes: str | None = None,
    ) -> None:
        sets: list[str] = []
        params: list[object] = []
        for column, value in (
            ("status", status), ("finished_at", finished_at), ("n_done", n_done),
            ("n_error", n_error), ("n_skipped", n_skipped), ("n_cases", n_cases),
            ("notes", notes),
        ):
            if value is not None:
                sets.append(f"{column}=?")
                params.append(value)
        if aggregate is not None:
            sets.append("aggregate_json=?")
            params.append(dumps(aggregate))
        if cost is not None:
            sets.append("cost_json=?")
            params.append(dumps(cost))
        if not sets:
            return
        params.append(run_id)
        self.db.execute(f"UPDATE eval_run SET {', '.join(sets)} WHERE id=?", tuple(params))

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self.db.query_one("SELECT * FROM eval_run WHERE id=?", (run_id,))
        return self._run_from_row(row) if row else None

    def list_runs(
        self,
        *,
        task_id: str | None = None,
        model_id: str | None = None,
        dataset_id: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[RunRecord]:
        clauses, params = [], []
        for column, value in (("task_id", task_id), ("model_id", model_id),
                              ("dataset_id", dataset_id), ("status", status)):
            if value:
                clauses.append(f"{column}=?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self.db.query(
            f"SELECT * FROM eval_run{where} ORDER BY started_at DESC LIMIT ?", tuple(params)
        )
        return [self._run_from_row(r) for r in rows]

    def run_dataset_ids(self, run_id: str) -> set[str]:
        """一个 run 实际用到的数据集 id（从 grade 反推）。

        对比前必须问这个：跨数据集的"回归"看着像模型变差，实际是换了考卷。
        """
        rows = self.db.query(
            """SELECT DISTINCT c.dataset_id FROM grade g
               JOIN eval_case c ON c.id = g.case_id
               WHERE g.eval_run_id=? AND c.dataset_id IS NOT NULL""",
            (run_id,),
        )
        return {str(r["dataset_id"]) for r in rows}

    # ── grade ─────────────────────────────────────────────────────
    def upsert_grade(self, rec: GradeRecord) -> str:
        self.db.execute(
            """INSERT INTO grade(id, eval_run_id, case_id, seq, trace_id, score, passed, verdict,
                                 invalid_format, out_of_set, metrics_json, error, judge_model_id,
                                 judge_usage_json, extra_json, graded_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(eval_run_id, case_id, seq) DO UPDATE SET
                 trace_id=excluded.trace_id, score=excluded.score, passed=excluded.passed,
                 verdict=excluded.verdict, invalid_format=excluded.invalid_format,
                 out_of_set=excluded.out_of_set, metrics_json=excluded.metrics_json,
                 error=excluded.error, judge_model_id=excluded.judge_model_id,
                 judge_usage_json=excluded.judge_usage_json, extra_json=excluded.extra_json,
                 graded_at=excluded.graded_at""",
            (rec.id, rec.eval_run_id, rec.case_id, rec.seq, rec.trace_id, rec.score,
             None if rec.passed is None else int(rec.passed), rec.verdict,
             int(rec.invalid_format), int(rec.out_of_set), dumps(rec.metrics), rec.error,
             rec.judge_model_id, dumps(rec.judge_usage), dumps(rec.extra),
             rec.graded_at or utc_now_iso()),
        )
        return rec.id

    def upsert_grades(self, records: Iterable[GradeRecord]) -> int:
        count = 0
        with self.db.transaction():
            for rec in records:
                self.upsert_grade(rec)
                count += 1
        return count

    def list_grades(
        self, run_id: str, *, verdict: str | None = None, limit: int | None = None
    ) -> list[GradeRecord]:
        clauses = ["eval_run_id=?"]
        params: list[object] = [run_id]
        if verdict:
            clauses.append("verdict=?")
            params.append(verdict)
        sql = f"SELECT * FROM grade WHERE {' AND '.join(clauses)} ORDER BY case_id, seq"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._grade_from_row(r) for r in self.db.query(sql, tuple(params))]

    def list_graded_case_ids(self, run_id: str) -> set[str]:
        """`--resume` 的依据：这些 case 已经花过 GPU 时间了，不许再跑一遍。"""
        rows = self.db.query(
            "SELECT DISTINCT case_id FROM grade WHERE eval_run_id=?", (run_id,)
        )
        return {str(r["case_id"]) for r in rows}

    def verdict_counts(self, run_id: str) -> dict[str, int]:
        rows = self.db.query(
            "SELECT verdict, COUNT(*) AS n FROM grade WHERE eval_run_id=? GROUP BY verdict",
            (run_id,),
        )
        return {str(r["verdict"]): int(r["n"]) for r in rows}

    # ── 行 → 记录 ─────────────────────────────────────────────────
    def _dataset_from_row(self, row) -> DatasetRecord:
        return DatasetRecord(
            id=row["id"], imported_at=row["imported_at"], upstream=row["upstream"] or "",
            revision=row["revision"] or "", license=row["license"] or "",
            splits={str(k): int(v) for k, v in loads_dict(row["split_json"]).items()},
            n_cases=row["n_cases"], loader=row["loader"] or "", notes=row["notes"] or "",
        )

    def _case_from_row(self, row) -> CaseRecord:
        return CaseRecord(
            id=row["id"], dataset_id=row["dataset_id"], ord=int(row["ord"] or 0),
            kind=row["kind"] or "single", input=loads_dict(row["input_json"]),
            expect=loads_dict(row["expect_json"]),
            tools=tuple(loads_list(row["tools_json"])),
            fixture=loads_dict(row["fixture_json"]), meta=loads_dict(row["meta_json"]),
            tags=tuple(t for t in (row["tags"] or "").split(",") if t),
        )

    def _task_from_row(self, row) -> TaskRecord:
        return TaskRecord(
            id=row["id"], name=row["name"], dataset_id=row["dataset_id"],
            metrics=tuple(loads_list(row["metrics_json"])), grader=loads_dict(row["grader_json"]),
            sample_params=loads_dict(row["sample_params_json"]), k=int(row["k"] or 1),
            budget=loads_dict(row["budget_json"]), sandbox=bool(row["sandbox"]),
            extra=loads_dict(row["extra_json"]),
        )

    def _run_from_row(self, row) -> RunRecord:
        return RunRecord(
            id=row["id"], task_id=row["task_id"], model_id=row["model_id"],
            started_at=row["started_at"], status=row["status"] or "running",
            provider_id=row["provider_id"], finished_at=row["finished_at"], seed=row["seed"],
            app_version=row["app_version"] or "", git_rev=row["git_rev"] or "",
            params_snapshot=loads_dict(row["params_snapshot_json"]),
            config=loads_dict(row["config_json"]),
            n_cases=int(row["n_cases"] or 0), n_done=int(row["n_done"] or 0),
            n_error=int(row["n_error"] or 0), n_skipped=int(row["n_skipped"] or 0),
            aggregate=loads_dict(row["aggregate_json"]), cost=loads_dict(row["cost_json"]),
            notes=row["notes"] or "",
            dataset_id=row["dataset_id"], dataset_revision=row["dataset_revision"] or "",
        )

    def _grade_from_row(self, row) -> GradeRecord:
        passed = row["passed"]
        return GradeRecord(
            id=row["id"], eval_run_id=row["eval_run_id"], case_id=row["case_id"],
            score=float(row["score"]), verdict=row["verdict"], graded_at=row["graded_at"],
            seq=int(row["seq"] or 0), trace_id=row["trace_id"],
            passed=None if passed is None else bool(passed),
            invalid_format=bool(row["invalid_format"]), out_of_set=bool(row["out_of_set"]),
            metrics=loads_dict(row["metrics_json"]), error=row["error"],
            judge_model_id=row["judge_model_id"],
            judge_usage=loads_dict(row["judge_usage_json"]), extra=loads_dict(row["extra_json"]),
        )

    @staticmethod
    def new_id() -> str:
        return new_trace_id()


def grades_by_case(grades: Sequence[GradeRecord]) -> dict[str, list[GradeRecord]]:
    """按 case 分组，pass^k / pass@k 要用。"""
    out: dict[str, list[GradeRecord]] = {}
    for grade in grades:
        out.setdefault(grade.case_id, []).append(grade)
    for items in out.values():
        items.sort(key=lambda g: g.seq)
    return out
