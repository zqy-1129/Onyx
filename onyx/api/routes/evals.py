"""评测 API：数据集、运行、分数下钻、矩阵/配对对比、GPU 锁状态，以及发起评测。

发起评测走的是服务进程内的那条单飞队列（`onyx.eval.service`），不在请求线程里跑：
一次 236 条的评测会把请求线程占住几分钟，而 HTTP 超时会让调用方以为失败了，
实际 GPU 还在跑。POST 只入队并立刻返回 run_id，进度靠 `…/progress` 读。

三个刻意的形状：
- `grade` 里带 `trace_id` 而不是只带分数。"每个分数都能点进一条真实 trace"
  是整个系统的立足点（DESIGN §15），API 不暴露它，前端就做不出下钻。
- 矩阵与对比在这里返回**派生后的结论 + 警告**，而不是让前端自己算差值：
  配对 CI 只在 Python 侧算一次，界面、CLI 与导出报告才不会互相打脸。
- 进度必须说明**出处**（`source`）：界面发起的运行有逐条进度和取消开关，
  CLI 发起的运行只有库里的检查点。把后者显示成"可以取消"是撒谎。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from onyx.api.deps import AppState, get_state
from onyx.core.clock import utc_now_iso
from onyx.eval.compare import CompareError, compare_runs
from onyx.eval.datasets.loader import DatasetError, parse_jsonl, register_dataset
from onyx.eval.service import JobView, SubmitRequest
from onyx.eval.tasks import build_task, builtin_dataset_names, specs
from onyx.report.eval_report import build_matrix
from onyx.store.repos import EvalRepo

router = APIRouter(prefix="/api", tags=["evals"])

#: 内存快照里"还没结束"的两个状态；其余都是终态
LIVE_STATES = ("queued", "running")

#: 一次导入的 JSONL 上限。超了直接拒——请求线程要把它解析成对象，
#  几十 MB 的上传等于"任何人用一个 POST 就能把看板钉住"
MAX_UPLOAD_BYTES = 4_000_000


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
    #: 界面能不能直接选它跑评测。库里登记的导入数据集目前还没有"读回"载入器，
    #: 选它只会在 worker 里变成一条 error，所以这里就标成不可选
    selectable: bool = False


class TaskView(BaseModel):
    id: str
    name: str
    requires: list[str] = []
    metrics: list[str] = []
    labels: list[str] = []
    default_dataset: str = ""
    dataset_revision: str = ""
    n_cases: int | None = None
    splits: dict[str, int] = {}
    max_tokens: int | None = None
    temperature: float | None = None
    #: 任务构造失败的原因。列不出来的任务比列得少更坏：那看起来像"没有任务"
    error: str = ""


class RunRequest(BaseModel):
    """与 `onyx eval run` 的 flag 一一对应；范围校验在 service 里做，不在这里重复一套。"""

    task: str
    model: str
    k: int = 1
    limit: int | None = None
    split: str = "default"
    seed: int | None = None
    dataset: str | None = None
    max_tokens: int | None = None
    max_wall_ms: float | None = None
    #: 续跑一个被中断的 run（写回同一个 id，已评过的 case 不重复计费）
    resume_run_id: str | None = None
    #: 拿不到 GPU 锁时最多等多久。0 = 不排队直接失败（对应 CLI 的 --no-queue）
    lock_timeout: float | None = None
    unload_others: bool = False
    notes: str = ""


class SubmitView(BaseModel):
    run_id: str
    state: str
    position: int
    task: str
    model: str


class ProgressView(BaseModel):
    run_id: str
    #: service = 本进程发起的运行；db = 只有库里的记录（CLI 发起，或已被裁剪）
    source: str
    state: str
    task: str = ""
    model: str = ""
    done: int = 0
    total: int = 0
    case_id: str = ""
    verdict: str = ""
    position: int = 0
    holder: str = ""
    eta_s: float | None = None
    waited_s: float = 0.0
    error: str = ""
    reason: str = ""
    queued_at: str = ""
    started_at: str = ""
    finished_at: str | None = None
    #: 能不能真的取消：只有本进程发起的运行可以
    cancellable: bool = False
    n_error: int = 0
    dataset_id: str | None = None


class QueueView(BaseModel):
    max_pending: int
    jobs: list[ProgressView]


class CancelView(BaseModel):
    run_id: str
    state: str
    cancelled: bool
    message: str


class DatasetImportRequest(BaseModel):
    """JSONL 文本而不是文件路径：看板可能在另一台机器上，路径没有意义，
    而且"路径来自请求体"本身就是任意文件读的入口（S21 的共享姿态）。"""

    jsonl: str
    id: str | None = None
    name: str = "uploaded"
    upstream: str = ""
    revision: str = ""
    license: str = ""
    notes: str = ""
    #: 覆盖已有 id 必须显式确认：历史 grade 指着这个 id，悄悄换内容会让"能不能比"变成玄学
    allow_replace: bool = False


class ImportView(BaseModel):
    id: str
    n_cases: int
    upstream: str
    revision: str
    license: str
    splits: dict[str, int] = {}
    replaced: bool = False
    warnings: list[str] = []


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
        # 与提交服务的校验保持一致：内置写法，或已导入且真有样本的 id。
        # 登记过但一条样本都没有的行选上去只会在 worker 里变成一条 error
        selectable=record.id in set(builtin_dataset_names()) or bool(record.n_cases),
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


def _progress_view(job: JobView | None, row: Any, state: AppState) -> ProgressView:
    """把"内存里的任务状态"与"库里的 run 记录"合成一个视图。

    内存优先于库，但只在任务还活着的时候优先：终态一律以库为准，
    因为 runner 收尾写的 n_done/n_cases 才是这次运行的最终事实。
    """
    info = state.gpu_lock.peek()
    eta = info.eta_s(utc_now_iso()) if info is not None else None
    base = ProgressView(
        run_id=row.id if row is not None else (job.run_id if job else ""),
        source="db", state=str(row.status) if row is not None else "unknown",
        task=row.task_id if row is not None else "",
        model=row.model_id if row is not None else "",
        done=int(row.n_done) if row is not None else 0,
        total=int(row.n_cases) if row is not None else 0,
        started_at=row.started_at if row is not None else "",
        finished_at=row.finished_at if row is not None else None,
        n_error=int(row.n_error) if row is not None else 0,
        dataset_id=row.dataset_id if row is not None else None,
        holder=info.owner if info is not None else "", eta_s=eta,
    )
    if job is None:
        return base
    if job.state in LIVE_STATES:
        return ProgressView(**{
            **base.model_dump(), "source": "service", "state": job.state,
            "task": job.task, "model": job.model, "cancellable": True,
            "done": job.done, "case_id": job.case_id, "verdict": job.verdict,
            "position": job.position, "waited_s": job.waited_s,
            "queued_at": job.queued_at,
            # holder 只在"真的在等锁"时才有意义。持有者就是自己时说"GPU 正被 api-eval:… 占用"，
            # 界面会读成"有人在跟我抢 GPU"，而事实是我的评测正在正常跑
            "holder": job.holder,
            "eta_s": base.eta_s if job.holder else None,
            # job.total 在第一条样本之前是 0，而 run 行一写出来就有 n_cases：
            # 已经开跑的任务要用库里那个更可信的总数，否则进度条显示 0/0
            "total": job.total or base.total,
            "started_at": job.started_at or base.started_at,
        })
    # 已结束：状态与进度以库为准，但只有服务侧才知道的失败原因在内存里。
    # 没有 run 行时（排队中就被取消）库里根本没有这件事的痕迹，只能报内存状态
    return ProgressView(**{
        **base.model_dump(), "source": "service", "cancellable": False,
        "state": base.state if row is not None else job.state,
        "task": job.task, "model": job.model,
        "error": job.error, "reason": job.reason, "position": 0,
        "queued_at": job.queued_at, "holder": "", "eta_s": None,
        "started_at": job.started_at or base.started_at,
        "finished_at": base.finished_at or job.finished_at,
    })


def _job_view(job: JobView, state: AppState) -> ProgressView:
    row = EvalRepo(state.runtime.db).get_run(job.run_id)
    return _progress_view(job, row, state)


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


# ── 发起评测（S23）─────────────────────────────────────────────────
@router.get("/tasks", response_model=list[TaskView])
def list_tasks() -> list[TaskView]:
    """可跑的评测任务，以及它们默认用哪份数据、产出哪些指标。

    选项从 `specs()` 现读而不是在前端硬编码：硬编码的清单会在新装一个任务插件时
    变成"界面看不到、CLI 却能跑"的那种裂缝。
    """
    out: list[TaskView] = []
    for task_id in sorted(specs()):
        try:
            task = build_task(task_id, model="")
        except (KeyError, ValueError, TypeError) as exc:
            # 一个坏任务不该把整个列表打空：那会让人以为"一个任务都没有"
            out.append(TaskView(id=task_id, name=task_id, error=f"{type(exc).__name__}: {exc}"[:200]))
            continue
        dataset = getattr(task, "dataset", None)
        out.append(TaskView(
            id=task_id, name=getattr(task, "name", task_id),
            requires=sorted(str(cap) for cap in getattr(task, "requires", ()) or ()),
            metrics=list(getattr(task, "metric_names", ()) or ()),
            labels=[str(label) for label in getattr(task, "labels", ()) or ()],
            default_dataset=dataset.id if dataset is not None else "",
            dataset_revision=dataset.revision if dataset is not None else "",
            n_cases=len(dataset) if dataset is not None else None,
            splits=dataset.splits() if dataset is not None else {},
            max_tokens=getattr(task, "max_tokens", None),
            temperature=getattr(task, "temperature", None),
        ))
    return out


@router.post("/runs", response_model=SubmitView, status_code=202)
def submit_run(body: RunRequest, state: AppState = Depends(get_state)) -> SubmitView:
    """发起一次评测：**入队并立刻返回 run_id**，不等结果。

    与 `onyx eval run` 共用同一个 runner、同一把 GPU 锁、同一份任务注册表，
    所以界面发起与命令行发起的差异只有"谁点的按钮"这一件事（记在 run 的 config.trigger 里）。
    校验失败返回 422 并列出可选项；队列已满返回 429。
    只读看板（`--read-only`）上这个端点返回 403。
    """
    view = state.eval_service.submit(SubmitRequest(
        task=body.task, model=body.model, k=body.k, limit=body.limit, split=body.split,
        seed=body.seed, dataset=body.dataset, max_tokens=body.max_tokens,
        max_wall_ms=body.max_wall_ms, lock_timeout=body.lock_timeout,
        resume_run_id=body.resume_run_id,
        unload_others=body.unload_others, notes=body.notes,
    ))
    return SubmitView(
        run_id=view.run_id, state=view.state, position=view.position,
        task=view.task, model=view.model,
    )


@router.get("/queue", response_model=QueueView)
def queue(state: AppState = Depends(get_state)) -> QueueView:
    """本进程的任务队列：谁在跑、谁在等、前面还有几个。

    只反映这个服务发起的任务；CLI 的任务通过 `/api/gpu` 的持有者看到。
    """
    return QueueView(
        max_pending=state.eval_service.max_pending,
        jobs=[_job_view(job, state) for job in state.eval_service.jobs()],
    )


@router.get("/runs/{run_id}/progress", response_model=ProgressView)
def run_progress(run_id: str, state: AppState = Depends(get_state)) -> ProgressView:
    """一条运行的进度。内存快照优先，其次库里的 run 记录。

    `source` 必须读出来：界面发起的运行有逐条进度和取消开关，
    CLI 发起的运行只有 runner 每 10 条落一次库的检查点。
    把后者显示成"可以取消"是撒谎。
    """
    job = state.eval_service.snapshot(run_id)
    row = EvalRepo(state.runtime.db).get_run(run_id)
    if job is None and row is None:
        raise HTTPException(status_code=404, detail=f"找不到 run {run_id!r}")
    return _progress_view(job, row, state)


@router.post("/runs/{run_id}/cancel", response_model=CancelView)
def cancel_run(run_id: str, state: AppState = Depends(get_state)) -> CancelView:
    """请求取消一条运行。`cancelled=true` 表示**取消已登记**，终态要接着读 progress。

    取消不打断进行中的那条请求：半截请求的 trace 比没有 trace 更难解释，
    所以运行会在两条样本之间停下。
    """
    job = state.eval_service.cancel(run_id)
    if job is not None:
        if job.state in LIVE_STATES:
            return CancelView(
                run_id=run_id, state=job.state, cancelled=True,
                message="取消已登记，会在下一条样本前停下"
                + ("（还在等 GPU 锁，会立刻退出排队）" if job.holder else ""),
            )
        return CancelView(
            run_id=run_id, state=job.state, cancelled=job.state == "cancelled",
            message=f"这条运行已经是终态（{job.state}），取消是幂等的：什么都没改",
        )

    row = EvalRepo(state.runtime.db).get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"找不到 run {run_id!r}")
    if row.status != "running":
        return CancelView(
            run_id=run_id, state=str(row.status), cancelled=False,
            message=f"这条运行已经是终态（{row.status}），不需要取消",
        )
    # 库里有 running 行而这个进程没有对应任务 ⇒ 它是别的进程（CLI）在跑。
    # 我们既没有它的取消开关，也不该去动它的锁：说"取消不了"比假装取消成功诚实
    holder = state.gpu_lock.peek()
    raise HTTPException(
        status_code=409,
        detail=f"这条 run 由 {holder.owner if holder else '另一个进程'} 在跑，不是本服务发起的，"
               "取消要回到那个进程（Ctrl-C）",
    )


@router.post("/datasets", response_model=ImportView, status_code=201)
def import_dataset(
    body: DatasetImportRequest, state: AppState = Depends(get_state)
) -> ImportView:
    """导入 JSONL 数据集，并把来历（upstream/revision/license）一起落库。

    与 `onyx eval import` 共用同一份解析（`parse_jsonl`）与同一个落库函数
    （`register_dataset`）：两个入口对"第 37 行坏了"必须给出行号并给出同一个结果，
    对"覆盖了旧 revision"必须说同一句警告。

    两条刻意的限制：
    - **收文本不收路径**：请求体里的路径就是任意文件读，而这个看板是可以共享的（S21）。
    - **覆盖已有 id 要显式确认**：历史 grade 指向这个 id，悄悄换内容会让
      "这两次分数能不能比"从"能"变成"不知道"，而那正是评测最贵的一种失效。
    """
    payload = body.jsonl or ""
    if len(payload.encode("utf-8")) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"导入内容 {len(payload)} 字节，上限 {MAX_UPLOAD_BYTES} 字节；"
                   "请分批判次导入（一次导入一大坨也没法核对第几行坏了）",
        )
    if not payload.strip():
        raise HTTPException(status_code=422, detail="jsonl 是空的：没有任何样本可导入")
    name = (body.name or "uploaded").strip()[:80] or "uploaded"
    # 空串按"没填"处理：默认 id 由 name 决定，而 revision 是内容 hash
    dataset_id = (body.id or "").strip()[:80] or None

    repo = EvalRepo(state.runtime.db)
    if dataset_id is not None and repo.get_dataset(dataset_id) is not None and not body.allow_replace:
        raise HTTPException(
            status_code=409,
            detail=f"数据集 {dataset_id!r} 已经登记过；覆盖会改变历史分数指向的考卷。"
                   "确认要覆盖就带 allow_replace=true，或换一个 id",
        )

    try:
        dataset = parse_jsonl(
            payload, name=name, dataset_id=dataset_id,
            upstream=body.upstream.strip(), revision=body.revision.strip(),
            license=body.license.strip(), notes=body.notes.strip(),
        )
    except DatasetError as exc:
        # 行号是这里唯一有用的信息："第 37 行不是合法 JSON"能直接改，"格式错误"不能
        raise HTTPException(status_code=422, detail=str(exc)) from None

    replaced = repo.get_dataset(dataset.id) is not None
    warnings = register_dataset(state.runtime.db, dataset, repo=repo)
    return ImportView(
        id=dataset.id, n_cases=len(dataset), upstream=dataset.upstream,
        revision=dataset.revision, license=dataset.license,
        splits=dataset.splits(), replaced=replaced, warnings=warnings,
    )
