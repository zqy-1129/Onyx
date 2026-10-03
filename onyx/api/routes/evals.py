"""评测 API：数据集、运行列表、运行详情、GPU 锁状态。

只做只读查询 —— 发起评测是 CLI / 后台任务的事。
在 API 里跑一次 236 条的评测会把请求线程占住几分钟，
而 HTTP 超时会让调用方以为失败了，实际 GPU 还在跑。

一个刻意的设计：`grade` 里带 `trace_id` 而不是只带分数。
"每个分数都能点进一条真实 trace"是整个系统的立足点（DESIGN §15），
如果 API 不暴露 trace_id，前端就做不出下钻，这条主张就只写在文档里。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from onyx.api.deps import AppState, get_state
from onyx.store.repos import EvalRepo

router = APIRouter(prefix="/api", tags=["evals"])


class DatasetView(BaseModel):
    id: str
    n_cases: int | None = None
    upstream: str = ""
    revision: str = ""
    license: str = ""
    loader: str = ""
    splits: dict[str, int] = {}
    imported_at: str = ""
    notes: str = ""


class RunView(BaseModel):
    id: str
    task_id: str
    model_id: str
    status: str
    started_at: str
    finished_at: str | None = None
    seed: int | None = None
    app_version: str = ""
    git_rev: str = ""
    params_snapshot: dict[str, Any] = {}
    config: dict[str, Any] = {}
    n_cases: int = 0
    n_done: int = 0
    n_error: int = 0
    n_skipped: int = 0
    #: 汇总指标。CI 是 dict（落库前已被 metrics.jsonable 转过），
    #: macro_f1 为 None 表示"没有可判定样本"，不是 0 分
    aggregate: dict[str, Any] = {}
    cost: dict[str, Any] = {}


class GradeView(BaseModel):
    case_id: str
    seq: int
    score: float
    verdict: str
    passed: bool | None
    invalid_format: bool
    out_of_set: bool
    trace_id: str | None = None
    error: str | None = None
    metrics: dict[str, Any] = {}


class GpuStatusView(BaseModel):
    busy: bool
    owner: str | None = None
    progress: str | None = None
    eta_s: float | None = None
    holder_host: str | None = None


def _dataset_view(record: Any) -> DatasetView:
    return DatasetView(
        id=record.id, n_cases=record.n_cases, upstream=record.upstream,
        revision=record.revision, license=record.license, loader=record.loader,
        splits=record.splits, imported_at=record.imported_at, notes=record.notes,
    )


def _run_view(record: Any) -> RunView:
    return RunView(
        id=record.id, task_id=record.task_id, model_id=record.model_id,
        status=record.status, started_at=record.started_at,
        finished_at=record.finished_at, seed=record.seed,
        app_version=record.app_version, git_rev=record.git_rev,
        params_snapshot=record.params_snapshot, config=record.config,
        n_cases=record.n_cases, n_done=record.n_done, n_error=record.n_error,
        n_skipped=record.n_skipped, aggregate=record.aggregate, cost=record.cost,
    )


def _grade_view(record: Any) -> GradeView:
    return GradeView(
        case_id=record.case_id, seq=record.seq, score=record.score,
        verdict=record.verdict, passed=record.passed,
        invalid_format=record.invalid_format, out_of_set=record.out_of_set,
        trace_id=record.trace_id, error=record.error, metrics=record.metrics,
    )


@router.get("/datasets", response_model=list[DatasetView])
def list_datasets(state: AppState = Depends(get_state)) -> list[DatasetView]:
    return [_dataset_view(item) for item in EvalRepo(state.runtime.db).list_datasets()]


@router.get("/runs", response_model=list[RunView])
def list_runs(
    task: str | None = Query(default=None),
    model: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    state: AppState = Depends(get_state),
) -> list[RunView]:
    return [_run_view(item) for item in
            EvalRepo(state.runtime.db).list_runs(task_id=task, model_id=model, limit=limit)]


@router.get("/runs/{run_id}", response_model=dict)
def get_run(run_id: str, state: AppState = Depends(get_state)) -> dict[str, Any]:
    repo = EvalRepo(state.runtime.db)
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"找不到 run {run_id!r}")
    return {
        "run": _run_view(run).model_dump(),
        "verdict_counts": repo.verdict_counts(run_id),
    }


@router.get("/runs/{run_id}/grades", response_model=list[GradeView])
def list_grades(
    run_id: str,
    verdict: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=2000),
    state: AppState = Depends(get_state),
) -> list[GradeView]:
    repo = EvalRepo(state.runtime.db)
    if repo.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail=f"找不到 run {run_id!r}")
    return [_grade_view(item) for item in repo.list_grades(run_id, verdict=verdict, limit=limit)]


@router.get("/gpu", response_model=GpuStatusView)
def gpu_status(state: AppState = Depends(get_state)) -> GpuStatusView:
    """谁在占 GPU、跑到哪、预计还要多久。

    只读锁文件，不参与竞争。看板用它把"排队中"显示成有 ETA 的状态，
    否则用户只会看到界面卡住，然后去 kill 进程。
    """
    from onyx.core.clock import utc_now_iso

    lock = state.gpu_lock
    info = lock.peek()
    if info is None:
        return GpuStatusView(busy=False)
    return GpuStatusView(
        busy=lock.is_busy(), owner=info.owner, progress=info.progress,
        eta_s=info.eta_s(utc_now_iso()), holder_host=info.host or None,
    )
