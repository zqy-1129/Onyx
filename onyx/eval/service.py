"""评测提交服务：进程内单飞队列、取消与进度快照（S23）。

为什么不在 HTTP 请求里直接跑评测：一次 236 条的评测要占住 GPU 几十分钟，
请求会超时，调用方以为失败了而 GPU 还在跑。所以 POST 只做"提交"，
立刻返回 run_id，真正的执行在这**一个** worker 线程里排队进行。

四条纪律：

1. **只有一条评测路径**。这里不重新实现任何循环，只是把 `onyx eval run` 用的那套
   `load_dataset` / `build_task` / `EvalRunner` 换个触发方式。界面与 CLI 各一套语义，
   迟早有一边在撒谎。
2. **单飞**。本地一块 GPU 是独占资源（DESIGN §8.5）：并发不报错，只污染数字。
   GPU 锁管跨进程，这里的队列管本进程——缺任何一个都会撞车。
3. **run 记录从"真的开始跑"那一刻才存在**。runner 是在拿到 GPU 锁之后才 insert 的，
   所以排队中/等锁中被取消的任务一条样本都没跑，库里不该多出这么一条 run。
4. **状态词表不新增**。库里只有 running / done / cancelled / skipped / error：
   worker 崩了记 error，服务重启留下的僵尸也记 error，而不是发明一个新状态
   让每个读这张表的下游都重新学一遍。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.core.errors import EvalError, EvalQueueFull
from onyx.core.ids import new_trace_id
from onyx.eval.gpu_lock import DEFAULT_GPU_STALE_AFTER_S, GpuLock, LockInfo, default_lock_path
from onyx.eval.runner import EvalRunner, ProgressCB, RunConfig
from onyx.eval.tasks import build_task, builtin_dataset_names, load_dataset, specs
from onyx.llm.gateway import Gateway
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, ModelRepo

#: 同时排队（尚未开跑）的任务上限。超了直接拒绝而不是无限排：
#: 一次评测可能是几十分钟 GPU 时间，默默排在 30 个任务之后等于永远不会有结果
DEFAULT_MAX_PENDING = 8
#: 内存里保留多少个已结束任务的状态。grade 与 run 记录在库里永久保留，
#: 这里只决定"进度条还能回看多远"
DEFAULT_KEEP_JOBS = 50
WORKER_POLL_S = 0.25
#: 单次提交的采样次数/样本数上限。挡住一个手滑的 `limit=100000` 把 GPU 锁住几天
MAX_K = 8
MAX_LIMIT = 20_000

#: 从界面发起评测时数据集训允许的写法（理由见 `_validate_dataset`）
NO_FILE_DATASET_HINT = (
    "界面不能读服务器磁盘上的文件（file:<路径> 只给 CLI 用）；"
    "请先导入数据集：用页面上的「数据集」面板，或 onyx eval import"
)

LIVE_STATES = ("queued", "running")


class EvalCancelled(Exception):
    """等待 GPU 锁期间被取消。

    只在本进程内部流转（worker 捕获后落成 cancelled 状态），所以不进 `core.errors`。
    它必须存在：runner 的取消检查只在 case 之间，而 worker 此刻阻塞在 `acquire()` 里——
    不在这里抛出去，"排队中"那一屏的取消按钮就是死的。
    """


@dataclass(frozen=True, slots=True)
class SubmitRequest:
    """一次提交的全部参数，与 `onyx eval run` 的 flag 一一对应。

    少一个字段就是"CLI 能做的界面做不到"，多一个就是两边口径分裂。
    """

    task: str
    model: str
    k: int = 1
    limit: int | None = None
    split: str = "default"
    seed: int | None = None
    dataset: str | None = None
    max_tokens: int | None = None
    max_wall_ms: float | None = None
    #: 拿不到 GPU 锁时最多等多久；0 = 不排队直接失败，None = 一直等（与 CLI 同语义）
    lock_timeout: float | None = None
    unload_others: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class JobView:
    """任务状态快照：全是可 JSON 化的标量，路由直接 `**asdict(view)`。"""

    run_id: str
    state: str
    task: str
    model: str
    done: int
    total: int
    case_id: str = ""
    verdict: str = ""
    #: 前面还有几个待跑任务；0 = 正在跑或已结束
    position: int = 0
    #: 等待 GPU 锁时，当前持有者是谁（配合 `/api/gpu` 的 ETA）
    holder: str = ""
    waited_s: float = 0.0
    error: str = ""
    reason: str = ""
    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""


@dataclass(slots=True)
class _Job:
    run_id: str
    request: SubmitRequest
    state: str = "queued"
    done: int = 0
    total: int = 0
    case_id: str = ""
    verdict: str = ""
    holder: str = ""
    waited_s: float = 0.0
    error: str = ""
    reason: str = ""
    queued_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    cancelled: threading.Event = field(default_factory=threading.Event)

    @property
    def live(self) -> bool:
        return self.state in LIVE_STATES

    def view(self, position: int = 0) -> JobView:
        return JobView(
            run_id=self.run_id, state=self.state, task=self.request.task,
            model=self.request.model, done=self.done, total=self.total,
            case_id=self.case_id, verdict=self.verdict, position=position,
            holder=self.holder, waited_s=round(self.waited_s, 1), error=self.error,
            reason=self.reason, queued_at=self.queued_at, started_at=self.started_at,
            finished_at=self.finished_at,
        )


class EvalService:
    """把提交进来的评测排成一条队，用一个线程串行跑完。"""

    def __init__(
        self,
        gateway: Gateway,
        db: Database,
        *,
        gpu_lock_path: Any = None,
        gpu_stale_after_s: float = DEFAULT_GPU_STALE_AFTER_S,
        max_pending: int = DEFAULT_MAX_PENDING,
        keep_jobs: int = DEFAULT_KEEP_JOBS,
    ) -> None:
        self.gateway = gateway
        self.db = db
        self.repo = EvalRepo(db)
        self.max_pending = max_pending
        self.keep_jobs = keep_jobs
        # 锁路径由装配层注入（`AppState` 已经握着 serve 用的那把锁，同一个文件才能互斥）。
        # 默认仍是机器级路径：数据目录可以按实例覆盖，而 GPU 是整台机器一块。
        self.lock_path = gpu_lock_path or default_lock_path()
        self.stale_after_s = gpu_stale_after_s
        self._jobs: dict[str, _Job] = {}
        self._order: list[str] = []
        # 可重入：cancel/snapshot 持锁时要调 `_position`，普通 Lock 会自己把自己锁死
        self._queue: queue.Queue[_Job] = queue.Queue()
        self._mutex = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stopping = threading.Event()

    # ── 提交 ──────────────────────────────────────────────────────
    def submit(self, request: SubmitRequest) -> JobView:
        """校验并排队，立刻返回 run_id。

        校验放在提交时而不是 worker 里：模型名打错要当场说，
        而不是等 GPU 排完队、每条样本都失败一遍再告诉你。
        """
        self._validate(request)
        job = _Job(run_id=new_trace_id(), request=request, queued_at=utc_now_iso())
        with self._mutex:
            if self._stopping.is_set():
                raise EvalQueueFull("这个服务正在关停，不再接受新的评测")
            waiting = sum(1 for item in self._jobs.values() if item.state == "queued")
            if waiting >= self.max_pending:
                raise EvalQueueFull(
                    f"排队中的评测已经有 {waiting} 个（上限 {self.max_pending}）",
                    detail={"max_pending": self.max_pending},
                )
            self._jobs[job.run_id] = job
            self._order.append(job.run_id)
            self._trim()
            position = sum(1 for rid in self._order
                           if self._jobs.get(rid) is not None and self._jobs[rid].state == "queued")
            # 持锁启动：两个并发提交各自看到 `_worker is None` 就会起两个 worker，
            # 单飞纪律当场失效，而且现象是"两个评测互相污染数字"而不是报错
            self._start_worker()
        self._queue.put(job)
        return job.view(position)

    def _validate(self, request: SubmitRequest) -> None:
        if not request.task:
            raise EvalError("必须指定任务", detail={"available": sorted(specs())})
        if request.task not in specs():
            raise EvalError(f"未知任务 {request.task!r}", detail={"available": sorted(specs())})
        if not request.model.strip():
            raise EvalError("必须指定模型")
        if not 1 <= request.k <= MAX_K:
            raise EvalError(f"k 必须在 1..{MAX_K}", detail={"k": request.k})
        if request.limit is not None and not 1 <= request.limit <= MAX_LIMIT:
            raise EvalError(f"limit 必须在 1..{MAX_LIMIT}", detail={"limit": request.limit})
        if not request.split.strip():
            raise EvalError("split 不能是空字符串")
        self._validate_dataset(request.dataset)
        self._validate_model(request.model)

    def _validate_dataset(self, dataset_id: str | None) -> None:
        """只接受 `load_dataset(..., db=)` 真能载入的写法。

        两道闸各自的理由：
        - **不接受 `file:<路径>`**：那是 CLI 在本机的便利。把它开放给 HTTP 就等于
          "请求体里写什么路径就读什么文件"，而这个看板是可以带 token 共享出去的（S21）。
        - **接受内置写法与库里已登记的 id**：解析顺序与 CLI 完全一致（内置 → file: → 库里），
          校验必须与 worker 的能力一致，否则会出现"提交成功、跑的时候死在 worker 里"。
        """
        if dataset_id is None:
            return
        if dataset_id.startswith("file:"):
            raise EvalError(NO_FILE_DATASET_HINT, detail={"dataset": dataset_id})
        if dataset_id in set(builtin_dataset_names()):
            return
        record = self.repo.get_dataset(dataset_id)
        if record is not None and record.n_cases:
            return
        # 只有 dataset 行、没有样本的登记不算可选：跑起来会是 `status=done, n_total=0`
        # 这种"看起来完全正常"的东西，而它其实是导入被中断的残骸
        known = set(builtin_dataset_names()) | {
            item.id for item in self.repo.list_datasets() if item.n_cases
        }
        raise EvalError(
            f"未知数据集 {dataset_id!r}；界面可选: 内置数据集或已导入且有样本的数据集",
            detail={"available": sorted(known)},
        )

    def _validate_model(self, model: str) -> None:
        """库里已经同步过模型清单时才做严格校验。

        一张空的 model 表只说明还没 `onyx models sync`，"查不到"并不等于"不存在"——
        把它当错误会挡掉合法的首次使用，那等于让界面比 CLI 更挑环境。
        """
        provider_id = str(getattr(self.gateway.provider, "id", "") or "")
        known = [record.name for record in ModelRepo(self.db).list_models(provider_id or None)]
        if not known:
            return
        if model not in known:
            raise EvalError(
                f"模型 {model!r} 不在已同步的清单里",
                detail={"available": sorted(known)[:20], "hint": "先同步模型清单，或检查名字"},
            )

    # ── 取消 ──────────────────────────────────────────────────────
    def cancel(self, run_id: str) -> JobView | None:
        """标记取消；返回 None 表示这个进程从没发起过这条 run。

        取消是**标记**而不是立即停止：worker 可能正在等 GPU 锁（锁的轮询回调里看到
        标记就退出），也可能正在跑一条样本（runner 只在 case 之间检查）。
        半截请求的 trace 比没有 trace 更难解释，所以不打断进行中的请求。
        """
        with self._mutex:
            job = self._jobs.get(run_id)
            if job is None:
                return None
            job.cancelled.set()
            return job.view(self._position(job.run_id))

    # ── 查询 ──────────────────────────────────────────────────────
    def snapshot(self, run_id: str) -> JobView | None:
        with self._mutex:
            job = self._jobs.get(run_id)
            return job.view(self._position(job.run_id)) if job else None

    def jobs(self) -> list[JobView]:
        """排队/在跑的在前（按提交顺序），已结束的按结束时间倒序。"""
        with self._mutex:
            known = [self._jobs[rid] for rid in self._order if rid in self._jobs]
            live = [job for job in known if job.live]
            settled = [job for job in known if not job.live]
            settled.sort(key=lambda job: job.finished_at or job.queued_at, reverse=True)
            return [job.view(self._position(job.run_id)) for job in [*live, *settled]]

    def _position(self, run_id: str) -> int:
        job = self._jobs.get(run_id)
        if job is None or not job.live:
            return 0
        ahead = 0
        for rid in self._order:
            other = self._jobs.get(rid)
            if other is job:
                break
            if other is not None and other.state == "queued":
                ahead += 1
        return ahead

    def _trim(self) -> None:
        """只丢**已结束**且最旧的那些：排队中与正在跑的永远留着。"""
        settled = [rid for rid in self._order
                   if rid in self._jobs and not self._jobs[rid].live]
        for rid in settled[:max(0, len(settled) - self.keep_jobs)]:
            self._jobs.pop(rid, None)
            self._order.remove(rid)

    # ── worker ────────────────────────────────────────────────────
    def _start_worker(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._work, name="onyx-eval-worker", daemon=True)
        self._worker.start()

    def _work(self) -> None:
        while not self._stopping.is_set():
            try:
                # 带超时而不是无限 get()：关停时队列是空的，`get()` 会永远挂着，
                # join() 也就永远等不到——那正是"服务关不掉"的经典形状
                job = self._queue.get(timeout=WORKER_POLL_S)
            except queue.Empty:
                continue
            try:
                self._execute(job)
            except Exception as exc:  # noqa: BLE001 - worker 死了整个看板的评测入口会静默失效
                self._fail(job, exc)

    def _execute(self, job: _Job) -> None:
        if job.cancelled.is_set():
            # 排队期间就被取消：一条样本都没跑，库里不该出现这条 run
            job.state, job.finished_at = "cancelled", utc_now_iso()
            return
        job.state, job.started_at = "running", utc_now_iso()
        request = job.request
        try:
            dataset = load_dataset(request.dataset, task_id=request.task, db=self.db)
            overrides: dict[str, Any] = {}
            if request.max_tokens:
                overrides["max_tokens"] = request.max_tokens
            task = build_task(request.task, model=request.model, dataset=dataset, **overrides)
        except (KeyError, ValueError, OSError) as exc:
            self._fail(job, exc)
            return

        runner = EvalRunner(
            self.gateway, self.repo, task, dataset=dataset,
            on_progress=self._progress(job), gpu_lock=self._make_lock(job),
        )
        try:
            report = runner.run(RunConfig(
                model=request.model, k=request.k, seed=request.seed, limit=request.limit,
                split=request.split, run_id=job.run_id, trigger="api",
                max_wall_ms=request.max_wall_ms, notes=request.notes,
                should_stop=job.cancelled.is_set, lock_timeout=request.lock_timeout,
                unload_others=request.unload_others,
            ))
        except EvalCancelled:
            job.state, job.finished_at = "cancelled", utc_now_iso()
            return
        except Exception as exc:  # noqa: BLE001 - 见 _fail：库里可能已经有一条 running 行
            self._fail(job, exc)
            return
        job.state, job.finished_at = report.status, utc_now_iso()
        job.total = report.n_cases
        job.done = report.n_done
        job.reason = report.skip_reason

    def _make_lock(self, job: _Job) -> GpuLock:
        return GpuLock(
            self.lock_path, owner=f"api-eval:{job.request.task}@{job.request.model}",
            stale_after_s=self.stale_after_s, on_wait=self._waiting(job),
        )

    def _waiting(self, job: _Job) -> Callable[[float, LockInfo | None], None]:
        def on_wait(waited: float, info: LockInfo | None) -> None:
            if job.cancelled.is_set():
                raise EvalCancelled(job.run_id)
            job.waited_s = waited
            # 只记持有者是谁，不记它的进度：把别人的 done/total 写进这个任务，
            # 界面会显示成"我们的评测跑到 3/236 了"
            job.holder = info.owner if info is not None else ""

        return on_wait

    @staticmethod
    def _progress(job: _Job) -> ProgressCB:
        def on_progress(done: int, total: int, case_id: str, grade: Any) -> None:
            job.done, job.total, job.case_id = done, total, case_id
            job.verdict = str(grade.verdict)

        return on_progress

    def _fail(self, job: _Job, exc: BaseException) -> None:
        job.state = "error"
        job.error = f"{type(exc).__name__}: {exc}"[:500]
        job.finished_at = utc_now_iso()
        # 库里如果已经有 running 行（runner 开跑之后才写），必须给它一个终态：
        # 留在 running 就是"界面显示这条还在跑，而线程早就退出了"
        row = self.repo.get_run(job.run_id)
        if row is not None and row.status == "running":
            self.repo.update_run(
                job.run_id, status="error", finished_at=job.finished_at,
                aggregate={**row.aggregate, "failure": {"reason": job.error}},
            )

    # ── 生命周期 ──────────────────────────────────────────────────
    def settle(self, timeout: float = 30.0) -> bool:
        """等到没有 queued/running 的任务为止。测试与优雅关停都用它。

        自己轮询而不是 `Queue.join()`：join 没有超时参数，worker 卡死时它会一直挂着，
        而那恰恰是最需要看清"卡在哪个任务上"的时刻。
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._has_live():
                return True
            time.sleep(0.02)
        return not self._has_live()

    def _has_live(self) -> bool:
        with self._mutex:
            return any(job.live for job in self._jobs.values())

    def shutdown(self, timeout: float = 5.0) -> None:
        """停止接受新任务，并尽力让在跑的那条收尾。

        不强杀线程：正在跑的评测握着 GPU 锁、写着 grade，硬停会让库里留在 running。
        超时就放手——线程是 daemon，进程退出时它自然消失，锁靠心跳过期被回收，
        那条 run 由下一次服务启动时的 `reclaim_orphans` 标成 error。
        """
        self._stopping.set()
        with self._mutex:
            for job in self._jobs.values():
                job.cancelled.set()
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            if job.state == "queued":
                # 还排在队里而 worker 已经不会再取：留在 queued 就是撒谎，
                # 它一条样本都没跑过，也不会有人再来跑它
                job.state, job.finished_at = "cancelled", utc_now_iso()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=timeout)

    def reclaim_orphans(self) -> list[str]:
        """把上次服务崩溃/重启留下的 running 行标成 error，并写明原因。

        两条判据必须一起用，缺一不可：
        - **出处是 api**（`config.trigger == "api"`）：CLI 发起的 running 属于另一个进程，
          它可能正好好地在跑，我们没有任何权限改它的状态。
        - **GPU 锁没人持有**：runner 只在拿到锁之后才写 running 行，并且每条样本打心跳。
          锁空闲就说明那个持有者已经不在了。
        宁可少标不可错标：错标会把一次真实进行中的评测写成失败历史。
        """
        if GpuLock(self.lock_path, stale_after_s=self.stale_after_s).is_busy():
            return []
        with self._mutex:
            owned = set(self._jobs)
        reclaimed: list[str] = []
        for row in self.repo.list_runs(status="running", limit=200):
            if row.id in owned or str(row.config.get("trigger") or "") != "api":
                continue
            # finished_at 留空：只知道"它已经不在了"，不知道它什么时候没的。
            # 填当前时间会被读成"它跑到这一刻"，那是编造
            self.repo.update_run(row.id, status="error", aggregate={
                **row.aggregate,
                "interrupted": {
                    "reason": "服务重启或进程退出，这条运行没有跑完",
                    "seen_at": utc_now_iso(),
                },
            })
            reclaimed.append(row.id)
        return reclaimed
