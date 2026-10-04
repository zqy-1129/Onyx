"""S13 验收：评测 runner。

runner 刻意很薄，所以这里测的全是**运行时纪律**，而不是算法：
- 单条样本失败不许中断整轮（已经花掉的 GPU 时间是本地评测最贵的资源）
- 中断后已完成的样本全部保留，`--resume` 不重复计费
- 每条 grade 都有 trace_id，且指向一条真实存在的 trace
- 能力不足要留一条 skipped 记录并写明原因，不是静默不跑
- 评测自身的开销（含 judge）必须计入 cost
"""

from __future__ import annotations

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.types import Cap
from onyx.eval.datasets.loader import Dataset
from onyx.eval.runner import EvalRunner, RunConfig, wait_for
from onyx.eval.task import Grade, Verdict
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "mock/classifier"


def _dataset(n: int = 6, labels=("转账",)) -> Dataset:
    cases = []
    for index in range(n):
        label = labels[index % len(labels)]
        cases.append({
            "id": f"c{index}", "ord": index,
            "input": {"instruction": f"第 {index} 条：{label}"},
            "expect": {"label": label}, "tags": [], "kind": "single",
        })
    return Dataset(id="tiny", cases=tuple(cases), upstream="test", revision="r1")


def _scripts(*texts: str) -> list[MockScript]:
    return [MockScript(text=text, done_reason="stop", in_tokens=100, out_tokens=5)
            for text in texts]


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _runner(env, scripts, *, dataset=None, task=None, clock=None, on_progress=None, caps=None,
             gpu_lock=None):
    _, _, observer, blobs = env
    data = dataset or _dataset()
    provider = MockProvider(scripts={MODEL: list(scripts)}, models=(MODEL,))
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=clock or FakeClock())
    instance = task or IntentClassification(data, model=MODEL)
    runner = EvalRunner(gateway, EvalRepo(env[0]), instance, dataset=data,
                        clock=clock or FakeClock(), on_progress=on_progress, caps=caps,
                        gpu_lock=gpu_lock)
    return runner, provider


# ── 正常路径 ──────────────────────────────────────────────────────
def test_run_completes_and_persists_everything(env):
    # 所有样本都期望「转账」，剧本也只有一条「转账」⇒ 全对。
    # 剧本是多条时 MockProvider 会按调用次数推进并在耗尽后 clamp 到最后一条，
    # 于是 accuracy 变成"两个序列的巧合"，测不出东西
    runner, _ = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL, seed=42))

    assert report.status == "done" and report.ok
    assert report.n_cases == 6 and report.n_done == 6 and report.n_error == 0
    assert report.aggregate["n_total"] == 6
    assert report.aggregate["accuracy"] == pytest.approx(1.0)

    repo = EvalRepo(env[0])
    stored = repo.get_run(report.run_id)
    assert stored is not None and stored.status == "done"
    assert stored.n_done == 6
    assert stored.aggregate["macro_f1"] is not None
    assert stored.seed == 42
    # 参数快照必须落库，否则换了 temperature 之后的分数差异无法解释
    assert stored.params_snapshot.get("temperature") == 0.0
    assert stored.app_version
    assert len(repo.list_grades(report.run_id)) == 6


def test_every_grade_points_at_a_real_trace(env):
    """DoD：每条 grade 有 trace_id。分数点不进去就等于不可验证。"""
    _, sink, _, _ = env
    runner, _ = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL))
    sink.flush(5.0)

    repo = TraceRepo(env[0])
    assert report.grades
    for grade in report.grades:
        assert grade.trace_id, f"{grade.case_id} 没有 trace_id"
        assert repo.get(grade.trace_id) is not None, f"{grade.trace_id} 在库里不存在"
    # 每条 trace 都标了它属于哪次评测、哪条样本、第几次采样
    records = [repo.get(g.trace_id) for g in report.grades]
    assert {r.eval_run_id for r in records} == {report.run_id}
    assert {r.case_id for r in records} == {g.case_id for g in report.grades}
    # purpose_label 对评测会带上 run id，这样按 purpose 聚合时不同评测不会混在一起
    assert all(r.purpose == f"eval:{report.run_id}" for r in records)
    assert all(r.root_id == report.run_id for r in records)


def test_dataset_is_persisted_once_not_on_every_run(env):
    """236 条样本每次跑都重写一遍是纯浪费，而且会刷掉 imported_at。"""
    runner, _ = _runner(env, _scripts("转账"))
    first = runner.run(RunConfig(model=MODEL))
    repo = EvalRepo(env[0])
    imported_at = repo.get_dataset("tiny").imported_at

    second = runner.run(RunConfig(model=MODEL))
    assert second.run_id != first.run_id
    assert repo.get_dataset("tiny").imported_at == imported_at, "数据集被重复导入"
    assert repo.count_cases("tiny") == 6, "样本被重复写入"


def test_k_sampling_produces_k_traces_per_case(env):
    """引擎不支持 n，所以 k 次采样是 k 次独立请求、k 条独立 trace。"""
    _, sink, _, _ = env
    runner, provider = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL, k=3))

    sink.flush(5.0)
    assert report.n_done == 18, "6 条样本 × 3 次采样"
    assert len(provider.calls) == 18
    grades = EvalRepo(env[0]).list_grades(report.run_id)
    assert len(grades) == 18
    assert {g.seq for g in grades} == {0, 1, 2}
    assert report.aggregate["k"] == 3
    # trace 数 = case 数 × k，这是"resume 不重复计费"的判据
    assert len({g.trace_id for g in grades}) == 18


def test_pass_hat_k_and_pass_at_k_are_both_reported(env):
    """剧本交替返回对/错：同一 case 的多次采样里必然有错 ⇒ pass^k < pass@k。"""
    data = _dataset(2, labels=("转账", "查余额"))
    runner, _ = _runner(env, _scripts("转账", "查余额"), dataset=data,
                        task=IntentClassification(data, model=MODEL))
    report = runner.run(RunConfig(model=MODEL, k=3))
    aggregate = report.aggregate
    # 样本标签按 转账/查余额 交替，脚本也按 转账/查余额 交替，所以约一半对一半错
    assert aggregate["pass_at_k"] is not None
    assert aggregate["pass_hat_k"] is not None
    assert aggregate["pass_hat_k"] <= aggregate["pass_at_k"]


# ── 韧性 ──────────────────────────────────────────────────────────
def test_one_failing_case_does_not_abort_the_run(env):
    """跑到一半崩掉会丢掉已经花掉的 GPU 时间——那是本地评测最贵的资源。"""
    scripts = _scripts("转账")
    scripts[0] = MockScript(raise_exc=RuntimeError("引擎炸了"))
    runner, _ = _runner(env, scripts)
    report = runner.run(RunConfig(model=MODEL))

    assert report.status == "done", "单条失败不该让整轮变成 error"
    assert report.n_done == 6
    assert report.n_error >= 1
    errored = [g for g in report.grades if g.verdict is Verdict.ERROR]
    assert errored and "generate" in errored[0].extra.get("stage", "")
    assert errored[0].passed is None, "引擎失败是未判定，不是判错"
    assert errored[0].attributable is False


def test_a_crashing_grader_is_recorded_and_the_run_continues(env):
    class BadTask(IntentClassification):
        def grade(self, case, sample):
            if case.id == "c1":
                raise ValueError("grader 自己崩了")
            return super().grade(case, sample)

    runner, _ = _runner(env, _scripts("转账"), task=BadTask(_dataset(), model=MODEL))
    report = runner.run(RunConfig(model=MODEL))

    assert report.status == "done"
    crashed = next(g for g in report.grades if g.case_id == "c1")
    assert crashed.verdict is Verdict.ERROR
    assert "grader 崩溃" in crashed.error
    assert crashed.trace_id, "grader 崩了也不能丢掉那条 trace 的关联"


def test_cost_accounting_counts_requests_and_tokens(env):
    runner, _ = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL))
    cost = report.cost
    assert cost["requests"] == 6
    assert cost["in_tokens"] == 600 and cost["out_tokens"] == 30
    assert cost["in_tokens_unknown"] == 0
    assert cost["wall_ms"] >= 0


def test_fallback_estimates_still_count_towards_cost(env):
    """引擎没报计数时保真阶梯会退回 heuristic 档，成本照样要计入。"""
    scripts = [MockScript(text="转账", report_usage=False) for _ in range(6)]
    runner, _ = _runner(env, scripts)
    report = runner.run(RunConfig(model=MODEL))
    assert report.cost["requests"] == 6
    assert report.cost["in_tokens"] > 0, "退回档的估计值也是真实付出的成本"
    assert report.cost["in_tokens_unknown"] == 0


def test_truly_unmeasured_usage_is_counted_not_faked_as_zero(env):
    """连退回档都给不出数字时记一个"未知"计数，而不是把 None 当 0 加进成本。

    用桩 gateway 造这个场景：真实 gateway 的保真阶梯几乎总能给出某个档位的数字，
    所以这条路径只能靠桩覆盖——而它恰恰是"成本被低报"的唯一入口。
    """
    class NoUsageGateway:
        provider = MockProvider(models=(MODEL,))

        def generate(self, request, **kw):
            from onyx.core.types import Generation
            from onyx.llm.gateway import GatewayResult

            return GatewayResult(
                trace_id=kw.get("trace_id") or "t-none",
                generation=Generation(text="转账", status=Status_OK),
                usage=None, anomalies=(), latency={}, record_refs={},
            )

    from onyx.core.types import Status as Status_OK

    data = _dataset(3)
    runner = EvalRunner(
        NoUsageGateway(), EvalRepo(env[0]),
        IntentClassification(data, model=MODEL), dataset=data,
    )
    report = runner.run(RunConfig(model=MODEL))
    assert report.cost["requests"] == 3
    assert report.cost["in_tokens"] == 0
    assert report.cost["in_tokens_unknown"] == 3, "不知道就是不知道，必须单独计数"


# ── 中断与续跑 ────────────────────────────────────────────────────
def test_cancel_preserves_completed_grades(env):
    runner, provider = _runner(env, _scripts("转账"))
    # 判据用"已经发了几个请求"，不依赖 should_stop 被调用几次
    # （runner 在 case 前后各检查一次，所以调用次数是 case 数的两倍）
    report = runner.run(RunConfig(model=MODEL, should_stop=lambda: len(provider.calls) >= 3))
    assert report.status == "cancelled"
    assert report.ok is False
    assert report.n_done == 3
    assert len(provider.calls) == 3
    # 已完成的必须留着，并且聚合只基于真实跑过的那部分
    assert len(EvalRepo(env[0]).list_grades(report.run_id)) == 3
    assert report.aggregate["cancelled"] is True
    assert report.aggregate["n_total"] == 3


class _GradeCrashRepo(EvalRepo):
    """在第 N 条 grade 写入时崩掉，模拟"进程跑到一半被 kill / 断电"。"""

    def __init__(self, db, *, boom_at: int) -> None:
        super().__init__(db)
        self.boom_at = boom_at
        self.upserts = 0

    def upsert_grade(self, rec):
        self.upserts += 1
        if self.upserts == self.boom_at:
            raise RuntimeError(f"模拟崩在第 {self.boom_at} 条 grade")
        return super().upsert_grade(rec)


def test_a_crash_mid_run_leaves_a_truthful_progress_row(env):
    """进度必须周期性落库，而不是只在结束时写。

    只在最后写 `n_done` 的话，崩在中途的那一行是 `status=running, n_done=0`，
    而库里其实已经有三条 grade——恰恰在最需要知道进度的时候它最错。
    """
    db, _, observer, blobs = env
    data = _dataset(8)
    provider = MockProvider(scripts={MODEL: _scripts("转账")}, models=(MODEL,))
    gateway = Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())
    runner = EvalRunner(gateway, _GradeCrashRepo(db, boom_at=4),
                        IntentClassification(data, model=MODEL), dataset=data)

    with pytest.raises(RuntimeError, match="崩在第 4 条"):
        runner.run(RunConfig(model=MODEL, progress_every=2))

    row = EvalRepo(db).list_runs()[0]
    assert row.status == "running", "没有收尾就不许冒充跑完了"
    assert row.n_done == 2, "落库的是崩溃前那个检查点（done=2），不是 0"
    assert len(EvalRepo(db).list_grades(row.id)) == 3, "grade 本身一条都没丢"


def test_wall_budget_stops_the_run(env):
    clock = FakeClock(step_ns=50_000_000)  # 每次读钟前进 50ms
    runner, _ = _runner(env, _scripts("转账"), clock=clock)
    report = runner.run(RunConfig(model=MODEL, max_wall_ms=100))
    assert report.status == "cancelled"
    assert report.n_done < 6


def test_resume_carries_the_previous_segments_cost(env):
    """续跑不许把已经花掉的 GPU 时间清零。

    本地评测最贵的资源就是 GPU 时间；cost 变成 0 之后，"这次评测花了多少"
    再也没有答案，而 run 记录看起来完全正常（status=done、分数齐全）。
    """
    runner, provider = _runner(env, _scripts("转账"))
    partial = runner.run(RunConfig(model=MODEL, should_stop=lambda: len(provider.calls) >= 2))
    assert partial.cost["requests"] == 2 and partial.cost["in_tokens"] == 200

    resumed = runner.run(RunConfig(model=MODEL, resume_run_id=partial.run_id))
    row = EvalRepo(env[0]).get_run(resumed.run_id)
    assert resumed.status == "done"
    assert row.cost["requests"] == 6, "落库的是两段的总和，不是本次那一段"
    assert row.cost["in_tokens"] == 600 and row.cost["out_tokens"] == 30
    assert row.cost["unloaded_models"] == [], "卸载记录属于当前这段，不接续历史"


def test_resume_skips_already_graded_cases(env):
    """DoD：跑一半 kill → 再跑不重复计费（trace 数 = case 数 × k）。"""
    _, sink, _, _ = env
    runner, provider = _runner(env, _scripts("转账"))

    partial = runner.run(RunConfig(model=MODEL, should_stop=lambda: len(provider.calls) >= 2))
    assert partial.status == "cancelled" and partial.n_done == 2
    first_round_requests = len(provider.calls)

    resumed = runner.run(RunConfig(model=MODEL, resume_run_id=partial.run_id))
    assert resumed.run_id == partial.run_id, "续跑必须写回同一个 run"
    assert resumed.status == "done"
    assert resumed.n_done == 6, "n_done 是整个 run 的完成数（2 条已评 + 新跑 4 条）"
    assert len(provider.calls) - first_round_requests == 4, "已评过的 case 不许再发请求"

    sink.flush(5.0)
    grades = EvalRepo(env[0]).list_grades(resumed.run_id)
    assert len(grades) == 6, "两轮加起来正好 6 条，没有重复"
    assert len({g.trace_id for g in grades}) == 6
    assert len({(g.case_id, g.seq) for g in grades}) == 6
    # 聚合必须包含**全部** 6 条，而不只是本轮新跑的 4 条
    assert resumed.aggregate["n_total"] == 6
    assert resumed.aggregate["resumed"] is True
    assert resumed.aggregate["already_graded_before"] == 2


def test_rerunning_a_completed_case_overwrites_instead_of_duplicating(env):
    """`UNIQUE(eval_run_id, case_id, seq)`：重跑是覆盖，不是追加。

    否则中断后重来会产生两份分数，聚合值被悄悄稀释，而报告上看不出来。
    """
    runner, _ = _runner(env, _scripts("转账"))
    first = runner.run(RunConfig(model=MODEL, limit=2))
    again = runner.run(RunConfig(model=MODEL, limit=2, resume_run_id=first.run_id))
    grades = EvalRepo(env[0]).list_grades(first.run_id)
    assert len(grades) == 2, "全部已评过 ⇒ 一条都不该新增"
    assert again.n_done == 2, "n_done 是这个 run 的总完成数，不是本段新增数"


def test_resume_with_an_unknown_run_id_starts_fresh(env):
    runner, _ = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL, resume_run_id="does-not-exist"))
    assert report.status == "done"
    assert report.n_done == 6
    assert report.run_id != "does-not-exist"


# ── 能力跳过 ──────────────────────────────────────────────────────
def test_capability_skip_leaves_a_record_with_a_reason(env):
    """静默不跑会让看板上"这个模型没有分数"，与"跑了但 0 分"无法区分。"""

    class NeedsTools(IntentClassification):
        id = "tool_selection"
        name = "工具选择"
        requires = frozenset({Cap.TOOLS, Cap.TOOL_CHOICE})

    data = _dataset(2)
    runner, provider = _runner(
        env, _scripts("转账"), task=NeedsTools(data, model=MODEL), caps=frozenset({Cap.CHAT}),
    )
    report = runner.run(RunConfig(model=MODEL))

    assert report.status == "skipped"
    assert report.skip_reason and "不做隐式降级" in report.skip_reason
    assert len(provider.calls) == 0, "能力不足就不该发任何请求"
    stored = EvalRepo(env[0]).get_run(report.run_id)
    assert stored.status == "skipped"
    assert stored.aggregate["skip"]["missing"] == ["tool_choice", "tools"]


def test_task_runs_when_capabilities_are_met(env):
    class NeedsTools(IntentClassification):
        requires = frozenset({Cap.TOOLS})

    data = _dataset(2)
    runner, _ = _runner(env, _scripts("转账"),
                        task=NeedsTools(data, model=MODEL), caps=frozenset({Cap.CHAT, Cap.TOOLS}))
    assert runner.run(RunConfig(model=MODEL)).status == "done"


def test_capabilities_are_read_from_the_provider_by_default(env):
    runner, _ = _runner(env, _scripts("转账"))
    assert isinstance(runner.capabilities, frozenset)
    assert Cap.CHAT in runner.capabilities, "MockProvider 声明了 chat 能力"


# ── 进度回调 ──────────────────────────────────────────────────────
def test_progress_callback_sees_every_grade(env):
    seen: list[tuple[int, int, str]] = []
    runner, _ = _runner(env, _scripts("转账"),
                        on_progress=lambda done, total, case_id, grade: seen.append(
                            (done, total, case_id)))
    runner.run(RunConfig(model=MODEL))
    assert [item[0] for item in seen] == list(range(1, 7))
    assert all(item[1] == 6 for item in seen)


def test_wait_for_throttles_but_keeps_the_first_and_last():
    """第一条是"跑起来了"的信号，最后一条是终态，两条都不许被节流掉。"""
    events: list[int] = []
    clock = {"t": 0.0}
    throttled = wait_for(
        lambda done, total, case_id, grade: events.append(done),
        interval=1.0, now=lambda: clock["t"],
    )
    grade = Grade(case_id="c", score=1.0, verdict=Verdict.CORRECT)
    throttled(1, 5, "c", grade)      # 第一条，放行
    clock["t"] = 0.2
    throttled(2, 5, "c", grade)      # 间隔未到，节流
    clock["t"] = 1.5
    throttled(3, 5, "c", grade)      # 超过间隔，放行
    clock["t"] = 1.6
    throttled(5, 5, "c", grade)      # 最后一条，永远放行
    assert events == [1, 3, 5]


# ── GPU 锁 ────────────────────────────────────────────────────────
def test_runner_heartbeats_and_releases_the_gpu_lock(env, tmp_path):
    """排队者的 ETA 完全依赖心跳；跑完还必须释放，否则 GPU 永久"被占"。"""
    from onyx.eval.gpu_lock import GpuLock

    lock = GpuLock(tmp_path / "gpu.lock", owner="eval:test", poll_s=0.01)
    runner, _ = _runner(env, _scripts("转账"), gpu_lock=lock)
    report = runner.run(RunConfig(model=MODEL))
    assert report.status == "done"
    assert lock.held is False, "跑完没释放锁"
    assert lock.peek() is None


def test_runner_queues_behind_a_live_holder(env, tmp_path):
    """锁被占时不许开跑：两个评测同时占一块 GPU，现象不是报错而是数字被污染。"""
    from onyx.eval.gpu_lock import GpuLock

    holder = GpuLock(tmp_path / "gpu.lock", owner="other-eval", poll_s=0.01, stale_after_s=60)
    holder.acquire()
    holder.heartbeat(1, 10)
    try:
        runner, provider = _runner(env, _scripts("转账"),
                                   gpu_lock=GpuLock(tmp_path / "gpu.lock", owner="me",
                                                    poll_s=0.01, stale_after_s=60))
        with pytest.raises(Exception) as exc:
            runner.run(RunConfig(model=MODEL, lock_timeout=0.05))
        assert exc.value.__class__.__name__ == "GpuLockBusy"
        assert len(provider.calls) == 0, "没拿到锁就不该发任何请求"
        # 也不该留下一条 status=running 却永远不动的记录
        assert EvalRepo(env[0]).list_runs(limit=5) == []
    finally:
        holder.release()


def test_lock_is_released_even_when_a_case_raises(env, tmp_path):
    """中途抛异常也必须放锁。泄漏的锁会让后面所有评测干等，
    而且唯一的表现是"所有人都排队"，比崩溃更难归因。"""
    from onyx.eval.gpu_lock import GpuLock

    class Boom(IntentClassification):
        def grade(self, case, sample):
            raise RuntimeError("grader 炸了") if case.id == "c0" else super().grade(case, sample)

    lock = GpuLock(tmp_path / "gpu.lock", owner="me", poll_s=0.01, stale_after_s=60)
    data = _dataset(2)
    runner = EvalRunner(
        _runner(env, _scripts("转账"))[0].gateway, EvalRepo(env[0]),
        Boom(data, model=MODEL), dataset=data, gpu_lock=lock,
    )
    report = runner.run(RunConfig(model=MODEL))
    assert report.status == "done"  # 单条失败被吸收成 grade=error
    assert lock.held is False and lock.peek() is None

    after = GpuLock(tmp_path / "gpu.lock", owner="next", poll_s=0.01, stale_after_s=60)
    after.acquire(timeout=0.2)          # 能被下一个进程拿到，说明锁真的放了
    after.release()


def test_unload_others_records_what_it_evicted(env, monkeypatch):
    """卸载结果必须进 cost：否则"这次基准是不是被别的模型挤了显存"无从判断。"""
    from onyx.eval.gpu_lock import GpuLock

    data = _dataset(1)
    runner, provider = _runner(env, _scripts("转账"), dataset=data,
                               task=IntentClassification(data, model=MODEL))
    provider.running = lambda: [type("M", (), {"name": "other-model", "model": "other-model"})()]
    unloaded: list[str] = []
    provider.unload = lambda name: unloaded.append(name)

    lock = GpuLock("unused", owner="x")
    runner2 = EvalRunner(
        runner.gateway, EvalRepo(env[0]), IntentClassification(data, model=MODEL),
        dataset=data, gpu_lock=lock,
    )
    report = runner2.run(RunConfig(model=MODEL, unload_others=True))
    assert unloaded == ["other-model"], "该卸的没卸"
    assert report.cost["unloaded_models"] == ["other-model"]


# ── 提交即拿到 id（S23：界面发起评测）──────────────────────────────
def test_explicit_run_id_is_used_for_the_run(env):
    """服务侧要在样本跑起来之前就返回 id，否则界面无法展示"已提交"的这条运行。"""
    _, sink, _, _ = env
    runner, _ = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL, run_id="pre-allocated-1", trigger="api"))
    assert report.run_id == "pre-allocated-1"
    row = EvalRepo(env[0]).get_run("pre-allocated-1")
    assert row is not None and row.status == "done"
    sink.flush(5.0)
    # 每条 grade 的 trace 都挂在预先分配的那个 run 上，而不是另一个新 id
    traces = {TraceRepo(env[0]).get(g.trace_id).eval_run_id
              for g in EvalRepo(env[0]).list_grades("pre-allocated-1")}
    assert traces == {"pre-allocated-1"}


def test_resume_run_id_beats_an_explicit_run_id(env):
    """两个都给时必须是续跑语义赢：把新 id 写进 resume 路径会产生一个空壳 run。"""
    runner, _ = _runner(env, _scripts("转账"))
    first = runner.run(RunConfig(model=MODEL, limit=2))
    again = runner.run(RunConfig(model=MODEL, resume_run_id=first.run_id, run_id="ignored"))
    assert again.run_id == first.run_id
    assert EvalRepo(env[0]).get_run("ignored") is None, "不该凭空造出一条从没跑过的 run"


def test_trigger_is_persisted_as_provenance(env):
    """running 行是不是僵尸，取决于它是谁发起的；出处必须在库里，不能只在内存里。"""
    runner, _ = _runner(env, _scripts("转账"))
    report = runner.run(RunConfig(model=MODEL, trigger="api"))
    assert EvalRepo(env[0]).get_run(report.run_id).config["trigger"] == "api"

    other = runner.run(RunConfig(model=MODEL))
    assert EvalRepo(env[0]).get_run(other.run_id).config["trigger"] == "cli"
