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

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from onyx import __version__
from onyx.core.clock import SYSTEM_CLOCK, Clock, utc_now_iso
from onyx.core.ids import new_trace_id
from onyx.core.types import Cap, EmbedRequest, TraceContext, TracePurpose
from onyx.eval.datasets.loader import Dataset
from onyx.eval.gpu_lock import GpuLock
from onyx.eval.metrics import jsonable
from onyx.eval.task import EvalTask, Grade, Skip, Verdict, check_capabilities
from onyx.llm.caps import ENGINE_CAP_MAP
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
    #: 预先分配的 run_id。服务侧需要"提交即拿到 id"，而 runner 内部生成的 id
    #: 调用方永远拿不到。与 `resume_run_id` 是两件事：那个带"跳过已评 case"的语义，
    #: 挪用它会让一次新运行莫名其妙地继承别人的进度。两者都给时 `resume_run_id` 赢。
    run_id: str | None = None
    #: 谁发起的这次运行（cli / api）。进程重启后要判断"哪条 running 是僵尸"，
    #: 靠的就是这个出处：不知道是谁发起的，就无法把它和"另一个进程正在跑"分开。
    trigger: str = "cli"
    max_wall_ms: float | None = None
    purpose: TracePurpose = TracePurpose.EVAL
    notes: str = ""
    #: 外部取消信号（Ctrl-C / API 取消按钮）。runner 只在 case 之间检查，
    #: 不打断进行中的请求——半截请求的 trace 会很难解释
    should_stop: Callable[[], bool] | None = None
    #: 拿不到 GPU 锁时最多等多久（秒）。None = 一直等
    lock_timeout: float | None = None
    #: 开跑前把**其它**已载入的模型卸掉。DESIGN §8.5：两个模型同时驻留会触发
    #: CPU offload，吞吐差一个数量级但数字看起来"正常"——最难发现的污染
    unload_others: bool = False
    #: 每跑完 N 条把进度写回 run 记录。每条都写要多一次 UPDATE，而崩溃/中断时
    #: 需要知道的只是"跑到哪了"，量级上 10 条一次完全够
    progress_every: int = 10


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
    #: 数据集来历。对比与报告的每个结论都默认"同一份数据"，所以这个必须随结果走
    dataset_id: str = ""
    dataset_revision: str = ""

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
        gpu_lock: GpuLock | None = None,
    ) -> None:
        self.gateway = gateway
        self.repo = repo
        self.task = task
        self.dataset = dataset
        self.clock = clock
        self.on_progress = on_progress
        self.gpu_lock = gpu_lock
        self._caps = caps

    # ── 能力 ──────────────────────────────────────────────────────
    @property
    def capabilities(self) -> frozenset[Cap]:
        if self._caps is not None:
            return self._caps
        caps = getattr(self.gateway.provider, "capabilities", None)
        return frozenset(caps()) if callable(caps) else frozenset()

    def capabilities_for(self, model: str) -> frozenset[Cap]:
        """provider 级能力 ∪ **这个模型自报**的能力。

        只看 provider 是不够的：Ollama 的 provider 级能力是**通道基线**，
        而 embedding 是模型级事实（对 qwen3.5:9b 调 `/api/embed` 会被引擎拒掉，
        对 qwen3-embedding:0.6b 就正常）。用 provider 级判，向量任务在真实 Ollama 上
        永远 skip——而这看起来像"这个模型不行"。
        显式传入的 `caps`（测试与服务侧）优先，不再问引擎。
        """
        base = self.capabilities
        if self._caps is not None or not model:
            return base
        show = getattr(self.gateway.provider, "show_model", None)
        if not callable(show):
            return base
        try:
            detail = show(model)
        except Exception as exc:  # noqa: BLE001 - 问不到模型能力就用 provider 级，但要说出来
            logging.getLogger("onyx.eval").warning(
                "取模型 %s 的能力失败，只用 provider 级判定：%s: %s",
                model, type(exc).__name__, exc,
            )
            return base
        declared = tuple(getattr(detail, "capabilities", ()) or ())
        return base | {cap for name in declared if (cap := ENGINE_CAP_MAP.get(str(name)))}

    @property
    def _dataset_id(self) -> str:
        return self.dataset.id if self.dataset is not None else ""

    @property
    def _dataset_revision(self) -> str:
        return self.dataset.revision if self.dataset is not None else ""

    # ── 主流程 ────────────────────────────────────────────────────
    def run(self, config: RunConfig) -> RunReport:
        started_at = utc_now_iso()
        started_ns = self.clock.monotonic_ns()

        self._ensure_persisted()
        skip = check_capabilities(self.task, self.capabilities_for(config.model))
        run_id = config.resume_run_id or config.run_id or new_trace_id()
        previous = self.repo.get_run(run_id) if config.resume_run_id else None
        resuming = previous is not None

        if skip is not None:
            # 能力不足：整个任务不跑，但**必须留下一条记录并写明原因**。
            # 静默不跑会让看板上"这个模型没有分数"，与"跑了但 0 分"无法区分
            if not resuming:
                self.repo.insert_run(RunRecord(
                    id=run_id, task_id=self.task.id, model_id=config.model,
                    started_at=started_at, finished_at=utc_now_iso(), status="skipped",
                    seed=config.seed, app_version=__version__, n_skipped=1,
                    aggregate={"skip": {"reason": skip.reason, "missing": list(skip.missing)}},
                    notes=config.notes, dataset_id=self._dataset_id,
                    dataset_revision=self._dataset_revision,
                ))
            return RunReport(
                run_id=run_id, task_id=self.task.id, model=config.model, status="skipped",
                skipped=(skip,), n_skipped=1, skip_reason=skip.reason,
                started_at=started_at, finished_at=utc_now_iso(),
                aggregate={"skip": {"reason": skip.reason, "missing": list(skip.missing)}},
                dataset_id=self._dataset_id, dataset_revision=self._dataset_revision,
            )

        cases = list(self.task.load(split=config.split, limit=config.limit))
        already: set[str] = set()
        # 续跑时"这个 run 已经完成了多少"必须以库里已有的 grade 为准，
        # 不能用本段计数器——否则收尾会把 n_done 写成本段的 0，
        # 一个跑完的 run 看起来一条都没跑（`n_error` 同理）
        base_records: list[Any] = []
        if resuming:
            already = self.repo.list_graded_case_ids(run_id)
            cases = [case for case in cases if case.id not in already]
            base_records = self.repo.list_grades(run_id)
        elif not resuming and config.resume_run_id:
            run_id = new_trace_id()

        total = len(cases) * max(1, config.k)
        grades: list[Grade] = []
        cost: dict[str, Any] = {
            "in_tokens": 0, "out_tokens": 0, "requests": 0,
            "in_tokens_unknown": 0, "wall_ms": 0.0, "unloaded_models": [],
        }
        if resuming:
            # 续跑必须**接着算**之前那一段的花费：`_one` 只统计本次发的请求，
            # 不接上就会出现"0 tok · 0 请求 · 0 ms"，而之前那一段真的花掉了十几分钟 GPU。
            # 卸载记录不接续：它描述的是"这一段开跑前清掉了什么"
            cost.update(_carried_cost(previous.cost if previous else {}))
        done = errors = 0
        status = "done"
        cancelled = False

        if self.gpu_lock is not None:
            # 拿不到锁就抛 GpuLockBusy（带持有者与 ETA）。这一步必须在写 run 记录**之前**：
            # 否则库里会留下一条 status=running 却永远不动的记录，比没有记录更难解释。
            self.gpu_lock.acquire(timeout=config.lock_timeout)

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
            if not resuming:
                self.repo.insert_run(RunRecord(
                    id=run_id, task_id=self.task.id, model_id=config.model,
                    provider_id=getattr(self.gateway.provider, "id", "") or None,
                    started_at=started_at, status="running", seed=config.seed,
                    app_version=__version__, git_rev=_git_rev(),
                    params_snapshot=_params_snapshot(self.task),
                    config={"k": config.k, "limit": config.limit, "split": config.split,
                            "resumed_from": config.resume_run_id,
                            "already_graded": len(already),
                            "trigger": config.trigger,
                            "unload_others": config.unload_others},
                    n_cases=total, notes=config.notes,
                    dataset_id=self._dataset_id, dataset_revision=self._dataset_revision,
                ))
            if self.gpu_lock is not None and config.unload_others:
                cost["unloaded_models"] = self._unload_other_models(config.model)

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
                    if self.gpu_lock is not None:
                        # 每条样本后刷新心跳：排队者的 ETA 完全依赖它，
                        # 而且进程崩了之后正是靠心跳过期才能回收锁
                        self.gpu_lock.heartbeat(done, total)
                    if config.progress_every and done % config.progress_every == 0:
                        # 只在样本之间落一次进度：中断/崩溃时 `eval ls` 得说得出跑到哪了。
                        # 否则那一次的 n_done 是 0，而库里其实已经有几百条 grade——
                        # 恰好是最需要知道进度的时候它最错
                        self.repo.update_run(
                            run_id, n_done=len(base_records) + done,
                            n_error=_error_count(base_records) + errors,
                        )
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
            # 墙钟是"累计值"：续跑时之前那一段已经花掉的时间不能假装没花
            cost["wall_ms"] = round(float(cost.get("wall_ms") or 0.0) + elapsed_ms(), 1)
            if self.gpu_lock is not None:
                # 心跳写不出去意味着"排队者看到的 ETA 在变旧、锁可能被判过期"。
                # 不为它中断评测，但必须留在 cost 里，否则这段数字无法解释
                cost["gpu_heartbeat_errors"] = self.gpu_lock.heartbeat_errors
                if self.gpu_lock.last_heartbeat_error:
                    cost["gpu_heartbeat_error"] = self.gpu_lock.last_heartbeat_error
                self.gpu_lock.release()

        # 续跑时要把**之前那些** grade 一起纳入聚合，否则分数只反映新跑的部分
        all_records = self.repo.list_grades(run_id)
        # 完成数与错误数一律以"这个 run 库里现在有多少 grade"为准：
        # 用本段计数器的话，一次什么都没新跑的续跑会把 n_done 写成 0，
        # 一个跑完的 run 看起来一条都没跑过
        n_done = len(all_records)
        n_error = _error_count(all_records)
        # n_cases 是**考卷大小**（计划要评多少条），不是已评条数。
        # 收尾拿 grade 条数覆盖它，被中断在 26 条的 run 就会写成 26/26 ——
        # 与跑完那行长得一模一样，而"这条还欠 210 条"恰恰是最需要看出来的信息
        planned = previous.n_cases if previous is not None and previous.n_cases else total
        n_cases = max(planned, n_done)
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
            run_id, status=status, finished_at=utc_now_iso(), n_done=n_done,
            n_error=n_error, n_skipped=skipped_count, n_cases=n_cases,
            aggregate=aggregate, cost=cost,
        )
        return RunReport(
            run_id=run_id, task_id=self.task.id, model=config.model, status=status,
            aggregate=aggregate, grades=tuple(grades),
            cost=cost, n_cases=n_cases, n_done=n_done, n_error=n_error,
            n_skipped=skipped_count, started_at=started_at, finished_at=utc_now_iso(),
            # 跑完的 run 也要带上考卷来历：字段声明的就是"每个结果都必须能回答
            # "这是哪份数据考出来的""，只有 skip 路径带上的话，
            # 进程内直接用 report 的调用方（服务侧、脚本）拿到的是空串
            dataset_id=self._dataset_id, dataset_revision=self._dataset_revision,
        )

    def _unload_other_models(self, keep: str) -> list[str]:
        """卸掉除目标模型以外已载入的模型。

        失败不中断评测：卸载只是**降低污染概率**的优化，不是正确性前提。
        但结果必须记进 cost，否则"这次基准是不是被别的模型挤了显存"无从判断。
        """
        running = getattr(self.gateway.provider, "running", None)
        unload = getattr(self.gateway.provider, "unload", None)
        if not callable(running) or not callable(unload):
            return []
        unloaded: list[str] = []
        try:
            loaded = running()
        except Exception:  # noqa: BLE001 - 采样失败不能影响评测
            return []
        for item in loaded:
            name = getattr(item, "name", "") or getattr(item, "model", "")
            if not name or name == keep:
                continue
            try:
                unload(name)
                unloaded.append(name)
            except Exception:  # noqa: BLE001 - 同上
                continue
        return unloaded

    # ── 单条样本 ──────────────────────────────────────────────────
    def _one(
        self, run_id: str, case: Any, seq: int, config: RunConfig, cost: dict[str, Any]
    ) -> Grade:
        request = self.task.build(case)
        context = TraceContext(
            purpose=config.purpose, eval_run_id=run_id, case_id=case.id,
            sample_seq=seq, root_trace_id=run_id,
        )
        #: 走哪条入口由 `build()` 返回的**请求类型**决定，不看任务 id。
        #: 写成 `if self.task.id == "semantic_similarity"` 的话，第二个非生成任务又得改这里。
        embed_call = isinstance(request, EmbedRequest)
        call = self.gateway.embed if embed_call else self.gateway.generate
        stage = "embed" if embed_call else "generate"
        try:
            result = call(request, purpose=config.purpose, context=context)
        except Exception as exc:  # noqa: BLE001 - 单条失败不许中断整轮（见模块文档）
            return Grade(
                case_id=case.id, seq=seq, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=f"{type(exc).__name__}: {exc}"[:500],
                extra={"stage": stage},
            )

        usage = result.usage
        cost["requests"] += 1
        #: 向量调用**没有**输出 token：那是事实，不是"引擎没报"。
        #: 沿用"进出都必须有数"的判据会把每条 embed 都记成 `in_tokens_unknown`，
        #: 于是新通路在自己的成本报表里被抹成"什么都没测到"。
        out_known = embed_call or (usage is not None and usage.out_tokens is not None)
        if usage is None or usage.in_tokens is None or not out_known:
            # 不知道就是不知道：记一个计数，而不是把 None 当 0 加进成本
            cost["in_tokens_unknown"] += 1
        else:
            cost["in_tokens"] += usage.in_tokens
            cost["out_tokens"] += int(usage.out_tokens or 0)

        #: 判分拿到的样本与发起的请求同类型：生成任务收 Generation，向量化任务收 Embedding
        sample = result.embedding if embed_call else result.generation
        try:
            grade = self.task.grade(case, sample)
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
        if dataset is not None:
            dataset_record, cases = dataset.to_records()
            existing = self.repo.get_dataset(dataset.id)
            # 只在"库里没有"时写是不够的：case id 是内容哈希，生成器一改就是一批新 id，
            # 而 dataset_id 不变。真机踩过——S32 给长上下文加干扰项之后，那轮 9 条 grade
            # 有一条都点不回自己的样本，`dataset.revision` 还停在旧串。
            # revision 变了就必须重写；旧样本只删掉**没被任何 grade 引用过**的那些，
            # 因为历史分数还要靠它们回答"当时考的是哪一份"。
            if existing is None or existing.revision != dataset.revision:
                self.repo.upsert_dataset(dataset_record)
                self.repo.upsert_cases(cases)
                self.repo.prune_stale_cases(dataset.id, [rec.id for rec in cases])
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


def _error_count(records: Sequence[Any]) -> int:
    """库里这些 grade 里有多少判成 error。

    读回来的是字符串（`verdict` 列），所以比对 `Verdict.ERROR.value` 而不是枚举本身。
    """
    return sum(1 for record in records if record.verdict == Verdict.ERROR.value)


def _carried_cost(previous: dict[str, Any] | None) -> dict[str, Any]:
    """续跑时要接着算的那些累计字段。

    只接这四个：`unloaded_models` 描述的是"这一段开跑前清掉了什么"，
    把历史段的一路带过来会读成"刚才又卸了一次"。
    """
    source = previous or {}
    carried = {key: int(source.get(key) or 0) for key in (
        "in_tokens", "out_tokens", "requests", "in_tokens_unknown")}
    carried["wall_ms"] = float(source.get("wall_ms") or 0.0)
    return carried


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
