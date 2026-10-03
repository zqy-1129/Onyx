"""评测 API：数据集、运行、分数下钻、矩阵/配对对比与 GPU 锁状态。

只做只读查询 —— 发起评测是 CLI / 后台任务的事。
在 API 里跑一次 236 条的评测会把请求线程占住几分钟，
而 HTTP 超时会让调用方以为失败了，实际 GPU 还在跑。

两个刻意的形状：
- `grade` 里带 `trace_id` 而不是只带分数。"每个分数都能点进一条真实 trace"
  是整个系统的立足点（DESIGN §15），API 不暴露它，前端就做不出下钻。
- 矩阵与对比在这里返回**派生后的结论 + 警告**，而不是让前端自己算差值：
  配对 CI 只在 Python 侧算一次，界面、CLI 与导出报告才不会互相打脸。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from onyx.api.deps import AppState, get_state
from onyx.eval.compare import CompareError, compare_runs
from onyx.report.eval_report import build_matrix
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
    #: 数据集来历。矩阵与对比都要靠它回答"这两次是不是同一份考卷"
    dataset_id: str | None = None
    dataset_revision: str = ""
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
        # 不传这两个字段的话，界面永远显示「—」，而库里其实有值——
        # "声明了却没填"的字段比缺字段更坏，因为它看起来像是数据缺失
        dataset_id=record.dataset_id, dataset_revision=record.dataset_revision,
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


@router.get("/matrix", response_model=dict)
def matrix(
    task: str | None = Query(default=None),
    model: str | None = Query(default=None),
    dataset: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=2000),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """模型 × 任务矩阵。每格是**该组合最新一次 done 运行**的主分数。

    取"最新"而不是"最好"：挑过的分数就不是测量了。
    未完成（running/cancelled）的运行不进网格——半截分数没有可比性，
    放进一样的格子里会让人以为它一样可信。
    """
    runs = EvalRepo(state.runtime.db).list_runs(
        task_id=task, model_id=model, dataset_id=dataset, limit=limit
    )
    return build_matrix(runs).as_dict()


@router.get("/compare", response_model=dict)
def compare(
    base: str = Query(..., description="基准 run_id"),
    target: str = Query(..., description="对比 run_id；差值方向是 target − base"),
    eps: float = Query(default=0.0, ge=0.0, le=1.0),
    seed: int = Query(default=0),
    iterations: int = Query(default=2000, ge=100, le=20000),
    with_cases: bool = Query(default=True, description="是否带上逐 case 明细"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """两次运行的配对对比：净改善/净劣化 + 配对 bootstrap CI + 劣化清单。

    404 而不是 400：run 不存在与"根本没法比"（任务不同、交集为空）是同一类
    "这个组合不成立"，前端只需处理一种失败文案。
    """
    try:
        result = compare_runs(EvalRepo(state.runtime.db), base, target,
                              eps=eps, seed=seed, iterations=iterations)
    except CompareError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    payload = result.as_dict()
    if not with_cases:
        payload["cases"] = []
    return payload
