"""评测 runner：`build` → **gateway 发** → `grade` → `aggregate`，外加调度与断点续跑。

runner 刻意很薄。它不做任何计量、不发任何 HTTP、不解释任何分数——
那些分别属于 gateway、provider 和 task。这样"加一个任务"永远不需要改 runner，
"换观测实现"也永远不需要改任务（DESIGN §9.1）。

三条运行时纪律：
1. **单条样本失败不许中断整轮**。一条 case 抛异常就记 `verdict=error` 继续跑；
   跑到一半崩掉会丢掉已经花掉的 GPU 时间，而那是本地评测最贵的资源。
2. **中断后已完成的样本全部保留**。`--resume` 靠 `list_graded_case_ids` 跳过，
   `grade` 上的 `UNIQUE(eval_run_id, case_id, seq)` 保证重跑是覆盖而不是追加。
3. **评测自身的开销必须计入 `cost`**。judge 也是本地模型时同样吃 GPU 与时间，
   不计入就会低估一次评测的真实代价（DESIGN R13）。
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from onyx import __version__
from onyx.core.clock import SYSTEM_CLOCK, Clock, utc_now_iso
from onyx.core.ids import new_trace_id
from onyx.core.types import Cap, TraceContext, TracePurpose
from onyx.eval.datasets.loader import Dataset
from onyx.eval.metrics import jsonable
from onyx.eval.task import EvalTask, Grade, Skip, Verdict, check_capabilities
from onyx.llm.gateway import Gateway
from onyx.store.records import GradeRecord, RunRecord, TaskRecord
from onyx.store.repos.eval_repo import EvalRepo

#: 进度回调：(已完成数, 总数, 当前 case_id, 最新 grade)
ProgressCB = Callable[[int, int, str, Grade], None]


@dataclass(frozen=True, slots=True)
class RunConfig:
    model: str
    k: int = 1
    seed: int | None = None
    limit: int | None = None
    split: str = "default"
    #: 续跑：跳过该 run 里已经评过的 case
    resume_run_id: str | None = None
    max_wall_ms: float | None = None
    purpose: TracePurpose = TracePurpose.EVAL
    notes: str = ""
    #: 外部取消信号（Ctrl-C / API 取消按钮）。runner 只在 case 之间检查，
    #: 不打断进行中的请求——半截请求的 trace 会很难解释
    should_stop: Callable[[], bool] | None = None


@dataclass(frozen=True, slots=True)
class RunReport:
    run_id: str
    task_id: str
    model: str
    status: str
    aggregate: dict[str, Any] = field(default_factory=dict)
    grades: tuple[Grade, ...] = ()
    skipped: tuple[Skip, ...] = ()
    cost: dict[str, Any] = field(default_factory=dict)
    n_cases: int = 0
    n_done: int = 0
    n_error: int = 0
    n_skipped: int = 0
    started_at: str = ""
    finished_at: str = ""
    #: 整个任务因能力不足被跳过时的原因（不是"跑了但都失败"）
    skip_reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "done"


class EvalRunner:
    def __init__(
        self,
        gateway: Gateway,
        repo: EvalRepo,
        task: EvalTask,
        *,
        dataset: Dataset | None = None,
        clock: Clock = SYSTEM_CLOCK,
        on_progress: ProgressCB | None = None,
        caps: frozenset[Cap] | None = None,
    ) -> None:
        self.gateway = gateway
        self.repo = repo
        self.task = task
        self.dataset = dataset
        self.clock = clock
        self.on_progress = on_progress
        self._caps = caps

    # ── 能力 ──────────────────────────────────────────────────────
    @property
    def capabilities(self) -> frozenset[Cap]:
        if self._caps is not None:
            return self._caps
        caps = getattr(self.gateway.provider, "capabilities", None)
        return frozenset(caps()) if callable(caps) else frozenset()

    # ── 主流程 ────────────────────────────────────────────────────
    def run(self, config: RunConfig) -> RunReport:
        started_at = utc_now_iso()
        started_ns = self.clock.monotonic_ns()

        self._ensure_persisted()
        skip = check_capabilities(self.task, self.capabilities)
        run_id = config.resume_run_id or new_trace_id()
        resuming = config.resume_run_id is not None and self.repo.get_run(run_id) is not None

        if skip is not None:
            # 能力不足：整个任务不跑，但**必须留下一条记录并写明原因**。
            # 静默不跑会让看板上"这个模型没有分数"，与"跑了但 0 分"无法区分
            if not resuming:
                self.repo.insert_run(RunRecord(
                    id=run_id, task_id=self.task.id, model_id=config.model,
                    started_at=started_at, finished_at=utc_now_iso(), status="skipped",
                    seed=config.seed, app_version=__version__, n_skipped=1,
                    aggregate={"skip": {"reason": skip.reason, "missing": list(skip.missing)}},
                    notes=config.notes,
                ))
            return RunReport(
                run_id=run_id, task_id=self.task.id, model=config.model, status="skipped",
                skipped=(skip,), n_skipped=1, skip_reason=skip.reason,
                started_at=started_at, finished_at=utc_now_iso(),
                aggregate={"skip": {"reason": skip.reason, "missing": list(skip.missing)}},
            )

        cases = list(self.task.load(split=config.split, limit=config.limit))
        already: set[str] = set()
        if resuming:
            already = self.repo.list_graded_case_ids(run_id)
            cases = [case for case in cases if case.id not in already]
        elif not resuming and config.resume_run_id:
            run_id = new_trace_id()

        total = len(cases) * max(1, config.k)
        if not resuming:
            self.repo.insert_run(RunRecord(
                id=run_id, task_id=self.task.id, model_id=config.model,
                provider_id=getattr(self.gateway.provider, "id", "") or None,
                started_at=started_at, status="running", seed=config.seed,
                app_version=__version__, git_rev=_git_rev(),
                params_snapshot=_params_snapshot(self.task),
                config={"k": config.k, "limit": config.limit, "split": config.split,
                        "resumed_from": config.resume_run_id,
                        "already_graded": len(already)},
                n_cases=total, notes=config.notes,
            ))

        grades: list[Grade] = []
        cost = {"in_tokens": 0, "out_tokens": 0, "requests": 0,
                "in_tokens_unknown": 0, "wall_ms": 0.0}
        done = errors = 0
        status = "done"
        cancelled = False

        def elapsed_ms() -> float:
            return (self.clock.monotonic_ns() - started_ns) / 1e6

        def stop_requested() -> bool:
            """预算与取消都只在 case 之间检查：打断进行中的请求会留下半截 trace，
            那种 trace 比没有 trace 更难解释。

            墙钟必须**当场**读，不能读 `cost["wall_ms"]`——那个字段要到循环结束后
            才更新，用它判断的话 wall 预算永远不会触发（曾静默失效过一轮）。
            """
            if config.should_stop and config.should_stop():
                return True
            return bool(config.max_wall_ms and elapsed_ms() > config.max_wall_ms)

        try:
            for case in cases:
                if stop_requested():
                    status, cancelled = "cancelled", True
                    break
                for seq in range(max(1, config.k)):
                    grade = self._one(run_id, case, seq, config, cost)
                    grades.append(grade)
                    self.repo.upsert_grade(_to_record(run_id, grade))
                    done += 1
                    if grade.verdict is Verdict.ERROR:
                        errors += 1
                    if self.on_progress:
                        self.on_progress(done, total, case.id, grade)
                # k 次采样属于同一个 case，中途停下会留下半组样本，
                # pass^k 会把它当成"k 次里只对了这几次的全部"而虚高
                if stop_requested():
                    status, cancelled = "cancelled", True
                    break
        except KeyboardInterrupt:
            # Ctrl-C：已完成的样本全部保留，状态标 cancelled 而不是 done。
            # 标成 done 会让一次半截的运行看起来像完整结果
            status, cancelled = "cancelled", True
        finally:
            cost["wall_ms"] = round(elapsed_ms(), 1)

        # 续跑时要把**之前那些** grade 一起纳入聚合，否则分数只反映新跑的部分
        all_records = self.repo.list_grades(run_id)
        # 立刻转成 JSON-safe：报告里看到的与库里存的必须是同一个形状。
        # 否则刚跑完时 CI 是 dataclass、`eval show` 读回来是字符串，
        # 同一次运行的置信区间在两个入口一个显示一个消失
        aggregate = jsonable(
            self.task.aggregate(_records_to_grades(all_records), seed=config.seed or 0)
        )
        aggregate["cancelled"] = cancelled
        aggregate["resumed"] = resuming
        aggregate["already_graded_before"] = len(already)
        skipped_count = aggregate.get("verdicts", {}).get(Verdict.SKIPPED.value, 0)

        self.repo.update_run(
            run_id, status=status, finished_at=utc_now_iso(), n_done=done,
            n_error=errors, n_skipped=skipped_count, n_cases=len(all_records),
            aggregate=aggregate, cost=cost,
        )
        return RunReport(
            run_id=run_id, task_id=self.task.id, model=config.model, status=status,
            aggregate=aggregate, grades=tuple(grades),
            cost=cost, n_cases=len(all_records), n_done=done, n_error=errors,
            n_skipped=skipped_count, started_at=started_at, finished_at=utc_now_iso(),
        )

    # ── 单条样本 ──────────────────────────────────────────────────
    def _one(
        self, run_id: str, case: Any, seq: int, config: RunConfig, cost: dict[str, Any]
    ) -> Grade:
        request = self.task.build(case)
        context = TraceContext(
            purpose=config.purpose, eval_run_id=run_id, case_id=case.id,
            sample_seq=seq, root_trace_id=run_id,
        )
        try:
            result = self.gateway.generate(
                request, purpose=config.purpose, context=context
            )
        except Exception as exc:  # noqa: BLE001 - 单条失败不许中断整轮（见模块文档）
            return Grade(
                case_id=case.id, seq=seq, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=f"{type(exc).__name__}: {exc}"[:500],
                extra={"stage": "generate"},
            )

        usage = result.usage
        cost["requests"] += 1
        if usage is None or usage.in_tokens is None or usage.out_tokens is None:
            # 不知道就是不知道：记一个计数，而不是把 None 当 0 加进成本
            cost["in_tokens_unknown"] += 1
        else:
            cost["in_tokens"] += usage.in_tokens
            cost["out_tokens"] += usage.out_tokens

        try:
            grade = self.task.grade(case, result.generation)
        except Exception as exc:  # noqa: BLE001 - grader 崩了也不能丢掉这条 trace
            return Grade(
                case_id=case.id, seq=seq, score=0.0, verdict=Verdict.ERROR, passed=None,
                trace_id=result.trace_id,
                error=f"grader 崩溃: {type(exc).__name__}: {exc}"[:400],
                extra={"stage": "grade"},
            )
        # 每个分数都必须能点进一条真实 trace（DESIGN §15）
        return _with_trace(grade, result.trace_id, seq)

    # ── 落库前置 ──────────────────────────────────────────────────
    def _ensure_persisted(self) -> None:
        dataset = self.dataset
        if dataset is not None and self.repo.get_dataset(dataset.id) is None:
            dataset_record, cases = dataset.to_records()
            self.repo.upsert_dataset(dataset_record)
            self.repo.upsert_cases(cases)
        self.repo.upsert_task(TaskRecord(
            id=self.task.id, name=getattr(self.task, "name", self.task.id),
            metrics=tuple(getattr(self.task, "metric_names", ())),
            grader={"labels": list(getattr(self.task, "labels", ()))},
            dataset_id=dataset.id if dataset else None,
            sample_params=_params_snapshot(self.task),
            k=int(getattr(self.task, "k", 1) or 1),
            extra={"requires": sorted(str(c) for c in self.task.requires)},
        ))


def _with_trace(grade: Grade, trace_id: str, seq: int) -> Grade:
    return replace(grade, trace_id=trace_id or grade.trace_id, seq=seq)


def _to_record(run_id: str, grade: Grade) -> GradeRecord:
    return GradeRecord(
        id=new_trace_id(), eval_run_id=run_id, case_id=grade.case_id, seq=grade.seq,
        trace_id=grade.trace_id or None, score=grade.score,
        verdict=str(grade.verdict), graded_at=utc_now_iso(), passed=grade.passed,
        invalid_format=grade.invalid_format, out_of_set=grade.out_of_set,
        metrics=grade.metrics, error=grade.error or None,
        judge_model_id=grade.judge_model_id, judge_usage=grade.judge_usage,
        extra=grade.extra,
    )


def _records_to_grades(records: Sequence[GradeRecord]) -> list[Grade]:
    return [
        Grade(
            case_id=r.case_id, score=r.score, verdict=Verdict(r.verdict), seq=r.seq,
            passed=r.passed, invalid_format=r.invalid_format, out_of_set=r.out_of_set,
            metrics=r.metrics, trace_id=r.trace_id or "", error=r.error or "",
            judge_model_id=r.judge_model_id or "", judge_usage=r.judge_usage, extra=r.extra,
        )
        for r in records
    ]


def _params_snapshot(task: EvalTask) -> dict[str, Any]:
    """参数快照必须落库：否则换了 temperature 之后的分数差异无法解释。"""
    return {
        key: getattr(task, key)
        for key in ("max_tokens", "temperature", "split", "labels", "fallback_label")
        if hasattr(task, key)
    }


def _git_rev() -> str:
    """当前 commit。分数没有版本锚点就无法回答"这个退化是哪次改动引入的"。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def wait_for(
    progress: ProgressCB,
    interval: float = 0.5,
    *,
    now: Callable[[], float] = time.monotonic,
) -> ProgressCB:
    """给进度回调加节流：每条都打印会把 200 条的日志刷满，反而看不见异常。

    第一条与最后一条**永远放行**：第一条是"跑起来了"的信号，
    最后一条是终态；节流掉任何一条都会让人误判进度。
    """
    state: dict[str, float | None] = {"last": None}

    def wrapper(done: int, total: int, case_id: str, grade: Grade) -> None:
        current = now()
        last = state["last"]
        if last is None or done >= total or current - last >= interval:
            state["last"] = current
            progress(done, total, case_id, grade)

    return wrapper
