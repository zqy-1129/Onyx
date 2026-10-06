"""S23 验收：从界面发起评测的服务侧（进程内单飞队列 + 取消 + 进度）。

这里测的全是**并发与状态**纪律，而不是评测算法（算法在 test_eval_runner.py）：
- 提交必须立刻返回，不能把调用方占在 GPU 上
- 同一进程内也只许一个评测在跑：两个评测同时占一块 GPU，现象不是报错而是数字被污染
- 取消要真的取消，而且"一条样本都没跑"的取消不许在库里留下 run
- worker 崩了不能让库里永远留在 running
- 服务重启留下的僵尸要能解释，但**不许碰别的进程发起的运行**
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import pytest

from onyx.core.content import FileBlobStore
from onyx.core.errors import EvalError, EvalQueueFull
from onyx.eval.gpu_lock import GpuLock
from onyx.eval.service import EvalService, SubmitRequest
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.records import ModelRecord, ProviderRecord, RunRecord
from onyx.store.repos import EvalRepo, ModelRepo
from onyx.store.sinks import SqliteRecordSink

MODEL = "mock/echo"
TASK = "intent_classification"
LIVE = ("queued", "running")


class _Counting(MockProvider):
    """带耗时与并发计数的 mock：单飞纪律只有"让两个任务真的可能重叠"才测得出来。"""

    def __init__(self, *args, delay: float = 0.0, **kw) -> None:
        super().__init__(*args, **kw)
        self.delay = delay
        self.active = 0
        self.max_active = 0
        self._mu = threading.Lock()

    def generate(self, req, *, trace_id: str = "", on_event=None):  # type: ignore[override]
        with self._mu:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            return super().generate(req, trace_id=trace_id, on_event=on_event)
        finally:
            with self._mu:
                self.active -= 1


@dataclass
class _Env:
    db: Database
    service: EvalService
    provider: _Counting
    gateway: Gateway
    tmp_path: object


def _wait_until(pred, timeout: float = 8.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return pred()


@pytest.fixture
def env(tmp_path):
    """一个离线 service + 它下面的 db。锁路径指到 tmp：默认的机器级锁锁的是真 GPU。"""
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=8, idle_wait=0.005)
    provider = _Counting(
        id="mock-local", scripts={MODEL: [MockScript(text="转账", in_tokens=100, out_tokens=4)]},
        models=(MODEL,),
    )
    gateway = Gateway(provider, observer=ObserverEngine(record_sink=sink),
                      blobs=FileBlobStore(tmp_path / "blobs"))
    service = EvalService(gateway, db, gpu_lock_path=tmp_path / "gpu.lock")
    yield _Env(db=db, service=service, provider=provider, gateway=gateway, tmp_path=tmp_path)
    service.shutdown(timeout=15.0)
    sink.close()
    db.close()


def _req(**kw) -> SubmitRequest:
    base = dict(task=TASK, model=MODEL, limit=3)
    base.update(kw)
    return SubmitRequest(**base)


def _run_once(env: _Env) -> str:
    """真的跑完一个任务，返回它的 task_id 所在的 run（后面的用例要往库里插假 run）。"""
    run_id = env.service.submit(_req(limit=1)).run_id
    assert env.service.settle(20.0), "任务没在 20s 内跑完"
    return run_id


# ── 提交与进度 ────────────────────────────────────────────────────
def test_submit_returns_at_once_and_the_run_lands_in_the_db(env):
    started = time.monotonic()
    view = env.service.submit(_req())
    elapsed = time.monotonic() - started
    # 不在这里断言 state=="queued"：worker 可能已经把它取走了，那正是它该有的速度
    assert view.state in LIVE
    assert elapsed < 0.5, "提交必须立刻返回，不能把调用方占在 GPU 上等结果"
    assert env.service.settle(20.0)

    row = EvalRepo(env.db).get_run(view.run_id)
    assert row is not None and row.status == "done"
    # 出处必须落库：进程重启后"这条 running 是不是僵尸"全靠它
    assert row.config["trigger"] == "api"
    final = env.service.snapshot(view.run_id)
    assert final is not None and final.state == "done"
    assert final.done == final.total == 3


def test_progress_snapshot_advances_case_by_case(env):
    env.provider.delay = 0.12
    view = env.service.submit(_req(limit=3))
    seen: list[int] = []
    for _ in range(600):
        snap = env.service.snapshot(view.run_id)
        assert snap is not None, "刚提交的任务在快照里消失了"
        if snap.state not in LIVE:
            break
        # 没在等锁就不许报持有者：把"自己的评测正在跑"显示成"有人在抢 GPU"是谎报竞争
        assert snap.holder == "", f"没人占 GPU 却报出持有者：{snap.holder}"
        if not seen or snap.done != seen[-1]:
            seen.append(snap.done)
        time.sleep(0.005)
    assert env.service.settle(20.0)
    # 一条一条推进而不是 0 → 3 跳变：进度条要能让人看出"它还在动"
    assert seen == sorted(seen) and len(seen) >= 3, f"进度读数不是递增推进的：{seen}"

    final = env.service.snapshot(view.run_id)
    assert final.state == "done" and final.done == final.total == 3
    assert final.case_id and final.verdict, "进度要能说出跑到哪条、判成什么"


def test_queue_is_single_flight(env):
    """同进程内也必须串行：并发跑两个评测不会报错，只会让所有数字失真。"""
    env.provider.delay = 0.02
    first = env.service.submit(_req(limit=4))
    second = env.service.submit(_req(limit=4))
    assert env.service.settle(30.0)
    assert env.provider.max_active == 1, f"两个评测重叠了（峰值并发 {env.provider.max_active}）"
    assert env.service.snapshot(first.run_id).state == "done"
    assert env.service.snapshot(second.run_id).state == "done"


def test_jobs_lists_live_before_settled(env):
    old = _run_once(env)
    holder = GpuLock(env.service.lock_path, owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    try:
        live = env.service.submit(_req(limit=1))
        assert _wait_until(lambda: env.service.snapshot(live.run_id).state == "running")
        views = env.service.jobs()
    finally:
        holder.release()
    ids = [view.run_id for view in views]
    assert ids.index(live.run_id) < ids.index(old), "还在跑的必须排在已结束的前面"
    assert ids.index(live.run_id) == 0


# ── 取消 ──────────────────────────────────────────────────────────
def test_cancel_while_waiting_for_the_gpu_lock_leaves_no_run(env):
    """等锁中的取消：一条样本都没跑，库里不能出现这条 run。

    这一条同时验证取消按钮在"等锁"那一屏是活的——worker 此刻阻塞在 acquire() 里，
    而 runner 的取消检查只在 case 之间。
    """
    holder = GpuLock(env.service.lock_path, owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    holder.heartbeat(1, 10)
    try:
        view = env.service.submit(_req())
        assert _wait_until(
            lambda: env.service.snapshot(view.run_id).state == "running"
            and env.service.snapshot(view.run_id).holder == "cli-eval"
        ), "读不到持有者，界面就只能显示干巴巴的一句「排队中」"
        env.service.cancel(view.run_id)
        assert env.service.settle(20.0)
        # 必须在放 holder 之前查：取消一个等锁中的任务，绝不能顺手把别人的锁删了
        still = GpuLock(env.service.lock_path, poll_s=0.02).peek()
        assert still is not None and still.owner == "cli-eval", "取消时把别人的锁抢了"
    finally:
        holder.release()

    after = env.service.snapshot(view.run_id)
    assert after is not None and after.state == "cancelled"
    assert EvalRepo(env.db).get_run(view.run_id) is None, "没跑过就不该有 run"
    assert len(env.provider.calls) == 0, "没拿到锁就不该发任何请求"


def test_cancel_mid_run_keeps_completed_grades(env):
    env.provider.delay = 0.05
    view = env.service.submit(_req(limit=6))
    assert _wait_until(lambda: env.service.snapshot(view.run_id).done >= 1), \
        "没等到任何进度就取消，测不到「跑了一半」这条路"
    env.service.cancel(view.run_id)
    assert env.service.settle(20.0)

    after = env.service.snapshot(view.run_id)
    assert after is not None and after.state == "cancelled"
    repo = EvalRepo(env.db)
    row = repo.get_run(view.run_id)
    assert row is not None and row.status == "cancelled"
    graded = repo.list_grades(view.run_id)
    assert 1 <= len(graded) < 6, "已完成的样本必须全部留着"
    assert len(graded) == row.n_done, "内存进度与库里的完成数必须是同一件事"


def test_cancel_a_never_submitted_run_returns_none(env):
    assert env.service.cancel("no-such-run") is None


# ── 校验 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("bad", [
    dict(task="no-such-task"),
    dict(task=""),
    dict(model="   "),
    dict(k=0),
    dict(k=99),
    dict(limit=0),
    dict(limit=99_999),
    dict(split="  "),
    dict(dataset="file:/etc/passwd"),
    dict(dataset="no_such_dataset"),
])
def test_invalid_requests_are_rejected_at_submit_time(env, bad):
    with pytest.raises(EvalError) as exc:
        env.service.submit(_req(**bad))
    assert exc.value.message
    assert len(env.provider.calls) == 0, "校验没过就不该发任何请求"
    assert EvalRepo(env.db).list_runs(limit=5) == [], "被拒绝的提交不许留下任何痕迹"


def test_unknown_dataset_error_lists_what_you_can_use(env):
    with pytest.raises(EvalError) as exc:
        env.service.submit(_req(dataset="intent_zh-v9"))
    assert "intent_zh" in exc.value.detail["available"], "报错要给出可选项，否则只能靠猜"


def test_model_catalog_is_only_checked_once_it_is_synced(env):
    """空 model 表只说明还没同步过模型清单，不等于模型不存在。

    在这里把它当错误拦掉，界面就比 CLI 更挑环境——而首次使用恰好都是还没 sync 的状态。
    """
    assert env.service.submit(_req(limit=1)).run_id

    repo = ModelRepo(env.db)
    repo.upsert_provider(ProviderRecord(
        id="mock-local", kind="mock", base_url="mock://", api_style="native"
    ))
    repo.upsert_model(ModelRecord(id="mock-local/mock/echo", provider_id="mock-local", name=MODEL))
    assert repo.find_by_name("mock-local", MODEL) is not None, "fixture 得真的把清单建起来"
    with pytest.raises(EvalError, match="不在已同步的清单里"):
        env.service.submit(_req(model="mock/typo"))
    # 清单里的名字必须放行，否则校验就从"防打错"变成了"只许用某一个模型"
    assert env.service.submit(_req(limit=1)).run_id


def test_queue_cap_rejects_instead_of_silently_queueing(env):
    """默默排在第 30 位等于永远不会有结果，而界面上只写着"排队中"。"""
    service = EvalService(env.gateway, env.db, max_pending=1,
                          gpu_lock_path=env.tmp_path / "cap.lock")
    holder = GpuLock(env.tmp_path / "cap.lock", owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    try:
        first = service.submit(_req(limit=1))
        assert _wait_until(lambda: service.snapshot(first.run_id).state == "running")
        second = service.submit(_req(limit=1))
        assert second.state == "queued" and second.position == 1
        with pytest.raises(EvalQueueFull):
            service.submit(_req(limit=1))
    finally:
        # 先让 worker 散伙再放锁：反过来会在放锁那一刻把第一个任务放行去真跑
        service.shutdown(timeout=20.0)
        holder.release()
    assert EvalRepo(env.db).list_runs(limit=5) == [], "两个任务都没开跑，库里不该有 run"


# ── 崩溃与僵尸 ────────────────────────────────────────────────────
class _CrashRepo(EvalRepo):
    def __init__(self, db, *, boom_at: int) -> None:
        super().__init__(db)
        self.boom_at = boom_at
        self.upserts = 0

    def upsert_grade(self, rec):
        self.upserts += 1
        if self.upserts == self.boom_at:
            raise RuntimeError(f"模拟崩在第 {self.boom_at} 条 grade")
        return super().upsert_grade(rec)


def test_worker_crash_gives_the_run_row_a_terminal_status(env):
    """worker 崩了之后库里不能永远留在 running。

    "界面显示这条还在跑、而线程早就退出了"是本服务最坏的失效方式：
    用户会一直等一个不会来的结果。
    """
    env.service.repo = _CrashRepo(env.db, boom_at=2)
    view = env.service.submit(_req(limit=5))
    assert env.service.settle(20.0)

    after = env.service.snapshot(view.run_id)
    assert after is not None and after.state == "error"
    assert "崩在第 2 条" in after.error
    row = EvalRepo(env.db).get_run(view.run_id)
    assert row is not None and row.status == "error", "库里也必须是终态"
    assert row.aggregate["failure"]["reason"] == after.error


class _SlowTerminalRepo(EvalRepo):
    """崩在第 N 条 grade，且把"写终态"那一步放慢——用来把 settle 与库里的顺序顶出来。"""

    def __init__(self, db, *, boom_at: int, delay: float) -> None:
        super().__init__(db)
        self.boom_at = boom_at
        self.delay = delay
        self.upserts = 0
        self.terminal_written = threading.Event()

    def upsert_grade(self, rec):
        self.upserts += 1
        if self.upserts == self.boom_at:
            raise RuntimeError("模拟崩在写终态之前")
        return super().upsert_grade(rec)

    def update_run(self, run_id, **kw):
        if kw.get("status") in ("error", "done", "cancelled", "skipped"):
            time.sleep(self.delay)
            self.terminal_written.set()
        return super().update_run(run_id, **kw)


def test_settle_does_not_report_idle_before_the_row_is_terminal(env):
    """`settle()` 说"没有活了"的那一刻，库里必须已经有终态——否则它在撒谎。

    `_fail` 原本先改内存里的 `state`（`live` 就是从它推导的）再写库，于是中间有一个窗口。
    快机器上是微秒级，谁都看不见；套件的覆盖率追踪阶段慢 5–10 倍，它就变成一次偶发失败
    （测试读到 running）。看板读的是库，所以"哪个是真"必须有答案，而不是靠机器快。
    """
    repo = _SlowTerminalRepo(env.db, boom_at=2, delay=0.3)
    env.service.repo = repo
    view = env.service.submit(_req(limit=5))
    assert env.service.settle(20.0)

    row = EvalRepo(env.db).get_run(view.run_id)
    assert row is not None and row.status == "error", (
        "settle 已经返回，库里却还停在 running：人会在界面前等一个不会来的结果")
    assert repo.terminal_written.is_set()


def test_a_failing_terminal_write_does_not_wedge_the_worker(env):
    """连写终态都失败时：worker 线程必须活着，这条 job 也不能永远 live。

    `_fail` 是被 worker 的 `except` 调用的。它要是把异常抛出去，线程当场死掉，
    之后每一条从界面发起的评测都只会留在 queued——"评测入口静默失效"就是这么发生的。
    """
    class _BadWrite(EvalRepo):
        """grade 一定写崩（触发 `_fail`），而 `_fail` 要写的那个终态也失败。"""

        def upsert_grade(self, rec):
            raise RuntimeError("模拟 grade 写不下去")

        def update_run(self, run_id, **kw):  # type: ignore[override]
            if kw.get("status") == "error":
                raise RuntimeError("模拟写终态失败")
            return super().update_run(run_id, **kw)

    env.service.repo = _BadWrite(env.db)
    view = env.service.submit(_req(limit=5))
    assert env.service.settle(8.0), "job 永远 live ⇒ worker 线程已经死了"

    after = env.service.snapshot(view.run_id)
    assert after is not None and after.state == "error"
    assert "库里没能写下终态" in after.error, "库里没落终态这件事必须写在人看得见的地方"
    assert EvalRepo(env.db).get_run(view.run_id).status == "running", \
        "这条测试的前提：终态确实没写进去（真实库里留下的就是这种行，由重启后的 reclaim 兜底）"

    # 关键不是这条 job 的状态，而是**入口还在**：下一条照样跑得完
    env.service.repo = EvalRepo(env.db)
    second = env.service.submit(_req(limit=2))
    assert env.service.settle(8.0)
    assert env.service.snapshot(second.run_id).state == "done", "worker 死了就再没人取队列"


def test_reclaim_marks_only_api_owned_rows(env):
    """服务重启留下的僵尸要能解释，但不能碰别的进程发起的运行。"""
    _run_once(env)
    repo = EvalRepo(env.db)
    task_id = repo.list_runs(limit=1)[0].task_id
    for run_id, trigger in (("zombie-api", "api"), ("someone-elses-cli", "cli")):
        repo.insert_run(RunRecord(
            id=run_id, task_id=task_id, model_id=MODEL,
            started_at="2026-10-04T00:00:00+00:00", status="running",
            config={"trigger": trigger}, n_cases=10, n_done=3,
        ))

    assert env.service.reclaim_orphans() == ["zombie-api"]

    row = repo.get_run("zombie-api")
    assert row.status == "error"
    assert "服务重启" in row.aggregate["interrupted"]["reason"]
    assert row.finished_at is None, "不知道它什么时候没的，填当前时间就是编造"
    assert row.n_done == 3, "已经跑出来的进度一条都不许抹掉"
    assert repo.get_run("someone-elses-cli").status == "running", "别人进程的运行不许动"


def test_reclaim_stays_out_when_someone_holds_the_lock(env):
    """锁被活着的持有者占着时，running 行完全可能是真的在跑。"""
    _run_once(env)
    repo = EvalRepo(env.db)
    repo.insert_run(RunRecord(
        id="maybe-live", task_id=repo.list_runs(limit=1)[0].task_id, model_id=MODEL,
        started_at="2026-10-04T00:00:00+00:00", status="running", config={"trigger": "api"},
    ))
    holder = GpuLock(env.service.lock_path, owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    holder.heartbeat(1, 10)
    try:
        assert env.service.reclaim_orphans() == []
    finally:
        holder.release()
    assert repo.get_run("maybe-live").status == "running"


def test_trimming_keeps_live_jobs(env):
    """内存只留最近 N 个已结束任务，但排队中与正在跑的永远留着——
    裁掉正在跑的任务等于把它的取消开关扔了。"""
    service = EvalService(env.gateway, env.db, keep_jobs=1,
                          gpu_lock_path=env.tmp_path / "trim.lock")
    first = service.submit(_req(limit=1)).run_id
    assert service.settle(20.0)
    second = service.submit(_req(limit=1)).run_id
    assert service.settle(20.0)

    holder = GpuLock(env.tmp_path / "trim.lock", owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    try:
        waiting = service.submit(_req(limit=1)).run_id
        assert _wait_until(lambda: service.snapshot(waiting) is not None
                           and service.snapshot(waiting).state == "running")
        assert service.snapshot(first) is None, "已结束又超出保留数的旧任务该被裁掉"
        assert service.snapshot(second) is not None, "只裁最旧的，最近那个要留着看进度"
        assert service.snapshot(waiting) is not None, "正在跑的任务永远不许被裁"
    finally:
        service.shutdown(timeout=20.0)
        holder.release()


# ── 续跑（S26：被中断的 run 要在界面上能接上）─────────────────────
def test_resume_writes_back_into_the_same_run(env):
    """续跑的 id 必须沿用原来那条：新样本写回同一个 run，进度条也不用换目标。"""
    env.provider.delay = 0.05
    interrupted = env.service.submit(_req(limit=6))
    _wait_until(lambda: env.service.snapshot(interrupted.run_id).done >= 1)
    env.service.cancel(interrupted.run_id)
    assert env.service.settle(20.0)
    repo = EvalRepo(env.db)
    run_id = interrupted.run_id
    assert repo.get_run(run_id).status == "cancelled"
    graded_before = len(repo.list_grades(run_id))
    calls_before = len(env.provider.calls)

    view = env.service.submit(_req(limit=6, resume_run_id=run_id))
    assert view.run_id == run_id, "续跑另起 id 等于把分数历史劈成两半"
    assert env.service.settle(30.0)

    row = repo.get_run(run_id)
    final = env.service.snapshot(run_id)
    assert final is not None and final.state == "done"
    assert row.status == "done" and row.n_done == 6
    assert len(repo.list_grades(run_id)) == 6
    assert len(env.provider.calls) - calls_before == 6 - graded_before, \
        "已评过的 case 不许再发一遍：那是最贵的重复"
    assert row.config["trigger"] == "api"
    # 续跑这件事记在 aggregate 里（config 是"这条 run 当初怎么建起来的"，不该被后一段改写）
    assert row.aggregate["resumed"] is True
    assert row.aggregate["already_graded_before"] == graded_before
    # 续跑段的成本必须接在原来的成本上：只看这一段会显示"这次没花钱"，
    # 而之前那几秒 GPU 真的花掉了
    assert row.cost["requests"] == 6


def test_resume_rejects_a_run_that_does_not_exist(env):
    with pytest.raises(EvalError, match="找不到要续跑的 run"):
        env.service.submit(_req(resume_run_id="never-existing"))


def test_resume_rejects_a_finished_run(env):
    done = _run_once(env)
    with pytest.raises(EvalError, match="已经跑完"):
        env.service.submit(_req(limit=1, resume_run_id=done))


def test_resume_rejects_a_run_still_being_written(env):
    """running 行说明有进程在写它：这时接上去就是两个人跑同一个 run、重复计费。"""
    _run_once(env)
    repo = EvalRepo(env.db)
    task_id = repo.list_runs(limit=1)[0].task_id
    repo.insert_run(RunRecord(
        id="someone-else", task_id=task_id, model_id=MODEL,
        started_at="2026-10-04T00:00:00+00:00", status="running",
        config={"trigger": "cli"},
    ))
    with pytest.raises(EvalError, match="别的进程"):
        env.service.submit(_req(limit=1, resume_run_id="someone-else"))


def test_resume_requires_the_same_task_and_model(env):
    run_id = _run_once(env)
    row = EvalRepo(env.db).get_run(run_id)
    with pytest.raises(EvalError, match="任务与模型"):
        env.service.submit(SubmitRequest(
            task="tool_selection", model=row.model_id, limit=1, resume_run_id=run_id,
        ))
    with pytest.raises(EvalError, match="任务与模型"):
        env.service.submit(SubmitRequest(
            task=row.task_id, model="other/model", limit=1, resume_run_id=run_id,
        ))


def test_resume_cannot_switch_the_dataset(env):
    """换考卷还接在同一份分数历史里，回归对比就废了。"""
    env.provider.delay = 0.0
    view = env.service.submit(_req(limit=6))
    _wait_until(lambda: env.service.snapshot(view.run_id).done >= 1)
    env.service.cancel(view.run_id)
    assert env.service.settle(20.0)
    with pytest.raises(EvalError, match="数据集必须与原 run 一致"):
        env.service.submit(_req(limit=6, resume_run_id=view.run_id, dataset="tool_calls_zh"))


def test_resume_without_a_dataset_inherits_the_original_one(env):
    """省略 dataset 的意思是"照旧"，不是"换成任务当前默认"。

    界面会从 run 行预填，CLI/API 调用方却不填才是常态；那时如果按默认集载入，
    新分数就接在另一批样本上，回归对比读起来仍然自洽，却是错的。
    """
    env.provider.delay = 0.05
    view = env.service.submit(_req(limit=6))
    _wait_until(lambda: env.service.snapshot(view.run_id).done >= 1)
    env.service.cancel(view.run_id)
    assert env.service.settle(20.0)
    repo = EvalRepo(env.db)
    original = repo.get_run(view.run_id)
    assert original.status == "cancelled", "前提：这条真的被中断了，否则续跑会被「已经跑完」拒掉"
    assert original.dataset_id, "前提：原 run 记下了它用的是哪份数据集"

    resumed = env.service.submit(_req(limit=6, resume_run_id=view.run_id))
    assert env.service.settle(30.0)
    after = repo.get_run(resumed.run_id)
    assert after.dataset_id == original.dataset_id
    assert after.status == "done" and after.n_done == 6
    # 考卷大小不能因为"这段只新跑了 4 条"或"上次只计划 2 条"而缩水
    assert after.n_cases == 6


# ── 关停 ──────────────────────────────────────────────────────────
def test_shutdown_settles_the_queue_and_refuses_new_work(env):
    service = EvalService(env.gateway, env.db,
                          gpu_lock_path=env.tmp_path / "down.lock")
    holder = GpuLock(env.tmp_path / "down.lock", owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    try:
        running = service.submit(_req(limit=1))
        assert _wait_until(lambda: service.snapshot(running.run_id).state == "running")
        queued = service.submit(_req(limit=1))
        # 在锁还被别人占着的时候关停：两个任务都没开跑，收尾必须是 cancelled 而不是"跑完"
        service.shutdown(timeout=20.0)
    finally:
        holder.release()

    assert service.snapshot(running.run_id).state == "cancelled"
    assert service.snapshot(queued.run_id).state == "cancelled", "排队中的任务留在 queued 就是撒谎"
    with pytest.raises(EvalQueueFull):
        service.submit(_req(limit=1))
    assert EvalRepo(env.db).list_runs(limit=5) == []
