"""S14 验收：GPU 独占锁。

单 GPU 上两个评测同时跑，现象**不是报错**而是数字被污染：
两个模型同时驻留触发 CPU offload，吞吐差一个数量级却看起来"正常"。
所以这把锁的正确性与它的可观测性同样重要——排队者必须知道要等多久，
否则人会去 kill 进程，而那正好会留下半截运行。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from onyx.core.errors import GpuLockBusy
from onyx.eval.gpu_lock import DEFAULT_STALE_AFTER_S, GpuLock, LockInfo, read_lock


@pytest.fixture
def lock_path(tmp_path) -> Path:
    return tmp_path / "gpu.lock"


def _lock(path, owner="a", **kw) -> GpuLock:
    return GpuLock(path, owner=owner, poll_s=0.01, **kw)


# ── 互斥 ──────────────────────────────────────────────────────────
def test_second_holder_is_blocked_while_the_first_runs(lock_path):
    first = _lock(lock_path, "eval:intent@qwen")
    first.acquire()
    try:
        second = _lock(lock_path, "eval:tool@other")
        with pytest.raises(GpuLockBusy):
            second.acquire(timeout=0.05)
        assert second.held is False
        assert first.held is True
    finally:
        first.release()


def test_lock_is_reacquirable_after_release(lock_path):
    first = _lock(lock_path, "a")
    first.acquire()
    first.release()
    assert read_lock(lock_path) is None

    second = _lock(lock_path, "b")
    second.acquire(timeout=0.5)
    assert second.held and second.peek().owner == "b"
    second.release()


def test_release_only_removes_its_own_lock(lock_path):
    """被判死之后锁被别人接管了，这时原来的持有者不许把别人的锁删掉。"""
    first = _lock(lock_path, "a", stale_after_s=0.05)
    first.acquire()
    time.sleep(0.15)

    second = _lock(lock_path, "b", stale_after_s=0.05)
    second.acquire(timeout=0.5)
    assert second.peek().owner == "b"

    first.release()  # 迟到的释放
    assert read_lock(lock_path) is not None, "b 的锁被 a 删掉了"
    assert read_lock(lock_path).owner == "b"
    second.release()
    assert read_lock(lock_path) is None


def test_context_manager_releases_on_exception(lock_path):
    lock = _lock(lock_path, "a")
    with pytest.raises(RuntimeError), lock:
        raise RuntimeError("炸了")
    assert read_lock(lock_path) is None
    assert lock.held is False


# ── 排队与 ETA ────────────────────────────────────────────────────
def test_busy_error_carries_owner_progress_and_eta(lock_path):
    """排队者必须知道**谁在跑、跑到哪、还要多久**，否则人会去 kill 进程。"""
    holder = _lock(lock_path, "eval:intent_classification@qwen3.5:9b")
    holder.acquire()
    holder.heartbeat(60, 240)

    waiter = _lock(lock_path, "eval:tool_selection@other")
    with pytest.raises(GpuLockBusy) as exc:
        waiter.acquire(timeout=0.05)
    detail = exc.value.detail
    assert detail["owner"] == "eval:intent_classification@qwen3.5:9b"
    assert detail["progress"] == "60/240"
    assert detail["eta_s"] is not None and detail["eta_s"] > 0
    assert "60/240" in exc.value.message
    holder.release()


def test_wait_callback_reports_progress_while_queued(lock_path):
    holder = _lock(lock_path, "a")
    holder.acquire()
    holder.heartbeat(1, 10)

    seen: list[tuple[float, str | None]] = []
    waiter = GpuLock(
        lock_path, owner="b", poll_s=0.01,
        on_wait=lambda waited, info: seen.append((waited, info.owner if info else None)),
    )
    with pytest.raises(GpuLockBusy):
        waiter.acquire(timeout=0.06)
    assert len(seen) >= 2, "排队期间必须持续汇报，不能静默干等"
    assert all(owner == "a" for _, owner in seen)
    assert seen[-1][0] > seen[0][0], "已等待时长要递增"
    holder.release()


def test_eta_is_none_when_progress_is_unknown(lock_path):
    """还没开始或没有总数时 ETA 是 None，不是 0——0 会被读成"马上就好"。"""
    holder = _lock(lock_path, "a")
    holder.acquire()
    info = holder.peek()
    assert info.eta_s(_now()) is None
    assert info.progress == "—"

    holder.heartbeat(0, 100)
    assert holder.peek().eta_s(_now()) is None, "一条都没跑完，无从估算"
    holder.release()


def test_eta_is_computed_from_the_holders_own_rate(lock_path):
    holder = _lock(lock_path, "a")
    holder.acquire()
    holder.heartbeat(10, 100)
    time.sleep(0.05)
    eta = holder.peek().eta_s(_now())
    assert eta is not None and eta > 0
    # 剩下 90 条，已跑 10 条 ⇒ ETA 大约是已用时间的 9 倍
    elapsed = holder.peek().elapsed_s(_now())
    assert eta == pytest.approx(elapsed * 9, rel=0.35)
    holder.release()


def test_completed_run_reports_no_eta(lock_path):
    holder = _lock(lock_path, "a")
    holder.acquire()
    holder.heartbeat(10, 10)
    assert holder.peek().eta_s(_now()) is None
    assert holder.peek().progress == "10/10"
    holder.release()


# ── 死锁回收 ──────────────────────────────────────────────────────
def test_stale_lock_is_reclaimed(lock_path):
    """持有者崩了就不会再写心跳，超过阈值必须能被接管。

    否则一次 Ctrl-C 之后 GPU 就"永久被占"，而唯一的表现是所有人都排队。
    """
    dead = _lock(lock_path, "crashed-run", stale_after_s=0.05)
    dead.acquire()
    assert _lock(lock_path, "b", stale_after_s=30).is_busy() is True

    time.sleep(0.12)
    fresh = _lock(lock_path, "b", stale_after_s=0.05)
    assert fresh.is_busy() is False, "心跳过期后不该再算占用"
    fresh.acquire(timeout=0.5)
    assert fresh.peek().owner == "b"
    fresh.release()


def test_heartbeat_keeps_a_long_run_from_being_reclaimed(lock_path):
    """长跑必须靠心跳续命，否则会被别的进程当成死锁抢走。"""
    holder = _lock(lock_path, "long-run", stale_after_s=0.2)
    holder.acquire()
    for index in range(4):
        time.sleep(0.08)
        holder.heartbeat(index, 10)
        assert _lock(lock_path, "other", stale_after_s=0.2).is_busy() is True
    holder.release()


def test_a_corrupt_lock_file_is_treated_as_free(lock_path):
    """读到坏文件只可能是写入方崩在半路；当成没有锁，让下一个进程接管。"""
    lock_path.write_text('{"owner": "a", "pid": 1, "started', encoding="utf-8")
    assert read_lock(lock_path) is None
    lock = _lock(lock_path, "b")
    lock.acquire(timeout=0.5)
    assert lock.peek().owner == "b"
    lock.release()


def test_lock_file_is_valid_json_at_every_read(lock_path):
    """写入用 os.replace 原子替换，所以排队者永远读不到半截 JSON。"""
    holder = _lock(lock_path, "a")
    holder.acquire()
    for index in range(20):
        holder.heartbeat(index, 100)
        raw = lock_path.read_text(encoding="utf-8")
        payload = json.loads(raw)  # 抛异常就说明读到了半截写入
        assert payload["owner"] == "a"
    holder.release()


def test_heartbeat_without_holding_is_a_noop(lock_path):
    lock = _lock(lock_path, "a")
    lock.heartbeat(1, 10)
    assert read_lock(lock_path) is None, "没拿到锁就不该写出锁文件"


def test_lock_info_records_the_host(lock_path):
    holder = _lock(lock_path, "a", host="test-host")
    holder.acquire()
    assert holder.peek().host == "test-host"
    holder.release()


def test_defaults_are_documented_constants():
    assert DEFAULT_STALE_AFTER_S == 180.0
    info = LockInfo(owner="a", pid=1, started_at=_now(), heartbeat_at=_now(),
                    done=5, total=10)
    assert info.progress == "5/10"


# ── 锁路径的作用域 ────────────────────────────────────────────────
def test_default_lock_path_ignores_the_data_dir(tmp_path, monkeypatch):
    """锁路径**不许**跟着 `ONYX_DATA_DIR` 走。

    数据目录是可以按实例覆盖的（多实例并存、测试隔离都靠它），
    而 GPU 是整台机器只有一块的资源。
    """
    from onyx.eval.gpu_lock import default_lock_path

    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path / "instance-a"))
    first = default_lock_path()
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path / "instance-b"))
    assert default_lock_path() == first, "换数据目录等于换锁，两个实例就各自为政了"
    assert first.parent != tmp_path


def test_instances_with_different_data_dirs_still_exclude_each_other(tmp_path, monkeypatch):
    """把上面那条变成行为断言：两个"数据目录不同"的实例必须仍然互斥。

    这是这把锁存在的唯一理由，只测路径相等不够——
    真正要保证的是"两个进程即使各用一份 .data，也不会同时占显存"。
    """
    from onyx.eval.gpu_lock import default_lock_path

    root = tmp_path / "temp"
    root.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(root))
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path / "a"))
    first = _lock(default_lock_path(), "run-a")
    first.acquire()
    try:
        monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path / "b"))
        second = _lock(default_lock_path(), "run-b")
        with pytest.raises(GpuLockBusy):
            second.acquire(timeout=0.05)
    finally:
        first.release()


def _now() -> str:
    from onyx.core.clock import utc_now_iso

    return utc_now_iso()


def test_takeover_is_exclusive_even_when_both_see_a_stale_lock(lock_path):
    """两个接管者都看到同一把过期锁时，也只能有一个赢。

    独占性来自 `os.rename`（要求源存在）与 `O_CREAT|O_EXCL` 这一对，
    不是来自"覆盖后回读确认"——回读对**顺序**执行的接管者不设防：
    后一个回读看到的是自己，而前一个早已返回，于是两个持有者并存。
    两个持有者比没有锁更危险，因为它看起来是安全的。
    """
    dead = _lock(lock_path, "dead", stale_after_s=0.02)
    dead.acquire()
    dead.heartbeat(1, 10)
    time.sleep(0.08)

    a, b = _lock(lock_path, "a", stale_after_s=0.02), _lock(lock_path, "b", stale_after_s=0.02)
    wins = [a._try_write(a.peek()), b._try_write(b.peek())]
    assert sum(wins) == 1, f"两个接管者都以为自己拿到了锁: {wins}"
    assert read_lock(lock_path).owner == ("a" if wins[0] else "b")


def test_transient_rename_denial_still_allows_takeover(lock_path, monkeypatch):
    """Windows 上 `rename` 会因"文件正被别的句柄打开"短暂 EACCES（杀软/索引器）。

    一次失败就放弃 = **没人能接管死锁**，于是评测无限排队等一个已经死掉的持有者。
    这比误抢一次更糟，所以接管必须重试，而"实在不行就覆盖"依然不许（那会打破互斥）。
    """
    dead = _lock(lock_path, "dead", stale_after_s=0.02)
    dead.acquire()
    dead.heartbeat(1, 10)
    time.sleep(0.08)

    real_rename = os.rename
    attempts = {"n": 0}

    def flaky(src, dst):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise PermissionError(13, "另一个程序正在使用此文件")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", flaky)
    info = _lock(lock_path, "taker", stale_after_s=0.02).acquire(timeout=1.0)
    assert info.owner == "taker"
    assert attempts["n"] >= 2, "第一次被拒后必须重试，而不是直接排队等一个死者"


def test_a_live_holder_that_heartbeats_cannot_be_displaced(lock_path):
    """反向保证：只要持有者按节奏打心跳，排队者就不该把它当成死锁抢走。"""
    # 阈值取得比等待时长大一个量级：否则"等的时候刚好过期"是被接管的正确条件，
    # 测不出我想测的东西
    holder = _lock(lock_path, "holder", stale_after_s=2.0)
    holder.acquire()

    waiter = _lock(lock_path, "waiter", stale_after_s=2.0)
    deadline = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        holder.heartbeat(1, 100)          # 持续续命
        time.sleep(0.02)
        info = waiter.peek()
        if info is not None and waiter._is_stale(info):
            pytest.fail("持续心跳的持有者被判成了死锁")

    with pytest.raises(GpuLockBusy):
        waiter.acquire(timeout=0.05)
    assert read_lock(lock_path).owner == "holder"
    holder.release()


def test_takeover_leaves_no_stray_files_behind(lock_path):
    """搬走过期锁的临时文件必须被清掉，否则目录会被 .stolen.* 堆满。"""
    dead = _lock(lock_path, "dead", stale_after_s=0.02)
    dead.acquire()
    time.sleep(0.08)

    fresh = _lock(lock_path, "fresh", stale_after_s=0.02)
    fresh.acquire(timeout=0.5)
    # 接管过程中搬走东西用的临时文件必须在释放前就清掉，
    # 否则长期跑下来目录里会堆满 .stolen.*
    leftovers = [p.name for p in lock_path.parent.iterdir() if ".stolen." in p.name]
    assert leftovers == [], f"遗留了搬走过期锁的临时文件: {leftovers}"
    fresh.release()
    assert list(lock_path.parent.iterdir()) == []


# ── 心跳写不进去（Windows 真实形态）───────────────────────────────
def test_transient_replace_denial_does_not_lose_a_heartbeat(lock_path, monkeypatch):
    """看板每秒 `peek()` 一次锁文件，Windows 上这会让 os.replace 短暂 ACCESS_DENIED。

    一次就放弃等于"有人在看进度"就能把一轮评测判死——而看进度正是它唯一的用途。
    """
    holder = _lock(lock_path, "holder")
    holder.acquire()
    real_replace = os.replace
    state = {"n": 0}

    def flaky(src, dst):
        state["n"] += 1
        if state["n"] == 1:
            raise PermissionError(5, "另一个程序正在使用此文件，进程无法访问")
        return real_replace(src, dst)

    try:
        monkeypatch.setattr(os, "replace", flaky)
        holder.heartbeat(3, 10)
    finally:
        monkeypatch.undo()

    assert state["n"] >= 2, "第一次被拒后必须重试"
    assert holder.heartbeat_errors == 0
    info = read_lock(lock_path)
    assert info is not None and info.done == 3, "重试成功后心跳要真的写进去"
    holder.release()


def test_persistent_replace_denial_only_costs_a_heartbeat(lock_path, monkeypatch):
    """一直写不进去时：评测继续跑，但失败必须被数出来，而且不留垃圾文件。"""
    holder = _lock(lock_path, "holder")
    holder.acquire()
    before = holder.peek().heartbeat_at

    def deny(src, dst):
        raise PermissionError(5, "拒绝访问")

    monkeypatch.setattr(os, "replace", deny)
    holder.heartbeat(7, 10)          # 不许抛：拿不到文件句柄就把整轮 GPU 时间判死，代价不对等
    monkeypatch.undo()

    assert holder.heartbeat_errors == 1
    assert "PermissionError" in holder.last_heartbeat_error
    assert holder.peek().heartbeat_at == before, "写不出去就不该假装刷新过"
    assert [p.name for p in lock_path.parent.iterdir() if p.name.endswith(".tmp")] == []

    # 锁本身还在手上：心跳失败不等于丢锁，release 照样要能删掉它
    holder.release()
    assert read_lock(lock_path) is None


def test_heartbeat_failures_reach_the_run_record(lock_path, monkeypatch):
    """心跳长期写不出去意味着锁可能被别人判过期接管——那段数字要能被解释，
    所以失败次数必须进 run 记录，而不是只活在内存里直到进程结束。"""
    from onyx.core.content import FileBlobStore
    from onyx.eval.datasets.loader import Dataset
    from onyx.eval.runner import EvalRunner, RunConfig
    from onyx.eval.tasks.intent_classification import IntentClassification
    from onyx.llm.gateway import Gateway
    from onyx.llm.providers.mock import MockProvider, MockScript
    from onyx.obs.engine import ObserverEngine
    from onyx.store.db import Database
    from onyx.store.repos import EvalRepo
    from onyx.store.sinks import SqliteRecordSink

    monkeypatch.setattr("onyx.eval.gpu_lock.WRITE_RETRIES", 1)  # 别真的退避 30ms×N 次
    db = Database(lock_path.parent / "run.sqlite")
    sink = SqliteRecordSink(db, batch_size=4, idle_wait=0.005)
    provider = MockProvider(scripts={"m": MockScript(text="转账", in_tokens=10, out_tokens=2)},
                            models=("m",))
    gateway = Gateway(provider, observer=ObserverEngine(record_sink=sink),
                      blobs=FileBlobStore(lock_path.parent / "blobs"))
    cases = tuple(
        {"id": f"c{i}", "ord": i, "input": {"instruction": "转账"}, "expect": {"label": "转账"},
         "tags": [], "kind": "single"} for i in range(4)
    )
    data = Dataset(id="tiny-lock", cases=cases, upstream="test", revision="r1")
    lock = _lock(lock_path, "runner")
    real_replace = os.replace

    def deny(src, dst):
        if str(src).endswith(".tmp"):
            raise PermissionError(5, "拒绝访问")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", deny)
    try:
        report = EvalRunner(gateway, EvalRepo(db), IntentClassification(data, model="m"),
                            dataset=data, gpu_lock=lock).run(RunConfig(model="m"))
    finally:
        monkeypatch.undo()
        sink.close()
        db.close()

    assert report.status == "done", "心跳失败不该把一轮跑完的评测判成失败"
    assert report.cost["gpu_heartbeat_errors"] >= 1
    assert "PermissionError" in report.cost["gpu_heartbeat_error"]
    assert lock.held is False, "跑完还是要放锁"


# ── 释放（Windows 上 unlink 会被读者挡住）─────────────────────────
def test_transient_unlink_denial_still_removes_the_lock(lock_path, monkeypatch):
    """看板的 `/api/gpu` 每秒 `read_text` 一次这个文件，Windows 的 unlink 会短暂被拒。

    删不动又不能改写的话，锁会以"心跳新鲜"的姿态留在原地，
    后面所有人白等 `stale_after_s`（默认 600 秒）——现象只是"一直排队"。
    """
    holder = _lock(lock_path, "holder")
    holder.acquire()
    holder.heartbeat(3, 10)
    real_unlink = os.unlink
    state = {"n": 0}

    def flaky(path):
        state["n"] += 1
        if state["n"] == 1:
            raise PermissionError(5, "另一个程序正在使用此文件")
        return real_unlink(path)

    monkeypatch.setattr(os, "unlink", flaky)
    holder.release()
    monkeypatch.undo()

    assert state["n"] >= 2, "第一次被拒后必须重试"
    assert read_lock(lock_path) is None, "重试成功后锁文件要真的没了"
    assert holder.release_failed is False


def test_persistent_unlink_denial_marks_the_lock_released(lock_path):
    """删不掉就改写心跳：让下一个等待者立刻能接管，而不是等 600 秒。"""
    holder = _lock(lock_path, "holder")
    holder.acquire()
    holder.heartbeat(4, 10)

    real_unlink = os.unlink
    os.unlink = lambda path: (_ for _ in ()).throw(PermissionError(5, "拒绝访问"))
    try:
        holder.release()
    finally:
        os.unlink = real_unlink

    info = read_lock(lock_path)
    assert info is not None, "删不掉时保留文件，但内容必须改成「已释放」"
    assert holder.release_failed is False
    assert info.extra.get("released_at"), "要写得出不撒谎的「我已经走了」"
    assert holder.is_busy() is False, "心跳被推到过去 ⇒ 不能还算忙的"

    # 下一个持有者不必等 stale 窗口就能接管
    _lock(lock_path, "next", stale_after_s=60).acquire(timeout=0.5)
    assert read_lock(lock_path).owner == "next"


def test_release_reports_when_neither_path_works(lock_path):
    """两条路都走不通时必须留下痕迹：静默失败等于让下一个人莫名其妙地等。"""
    holder = _lock(lock_path, "holder")
    holder.acquire()
    holder.heartbeat(1, 5)

    real_unlink, real_replace = os.unlink, os.replace

    def deny_unlink(path):
        raise PermissionError(5, "拒绝访问")

    def deny_replace(src, dst):
        if str(src).endswith(".tmp"):
            raise PermissionError(5, "拒绝访问")
        return real_replace(src, dst)

    os.unlink = deny_unlink
    os.replace = deny_replace
    try:
        holder.release()
    finally:
        os.unlink, os.replace = real_unlink, real_replace

    assert holder.release_failed is True
    assert "PermissionError" in holder.last_release_error
    assert holder.held is False, "放锁的状态机不能因为写盘失败就以为还持着"
