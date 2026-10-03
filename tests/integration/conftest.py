"""真机测试的 GPU 互斥。

为什么需要这个文件：`pytest -m live` 曾经完全不参与 GPU 锁，
于是它可以和 `onyx eval run` 同时打同一个引擎。后果不是失败，而是**数字是错的**——
两个模型同时驻留会触发 CPU offload，吞吐差一个数量级，
而基准测试照样返回一个看起来很正常的 t/s。

这一次真的撞上了：并发起 eval 与 live 套件时 `test_unload_releases_model` 失败
（模型被并发请求占着，卸不掉），看起来像产品 bug，实际是测试装置没互斥。

所以这里在**整个 session 开始前**拿一次机器级 GPU 锁：
- 拿不到就 fail 并说明谁在占，而不是让测试跑出一堆无意义的数字；
- 用 `default_lock_path()`，与 CLI / `onyx serve` 锁的是同一个文件。

本目录下的测试全都带 `live` 标记，所以 autouse 不会把离线单元测试串起来。
"""

from __future__ import annotations

import pytest

from onyx.core.errors import GpuLockBusy
from onyx.eval.gpu_lock import GpuLock, default_lock_path

#: live 套件要连着跑几十分钟，给排队留足时间
QUEUE_TIMEOUT_S = 300.0
#: 单条真实生成可能跑几分钟，阈值必须明显大于它，否则活着的持有者会被排队者误判成死锁
STALE_AFTER_S = 900.0


class _LiveGpu:
    """锁 + 进度计数。

    计数是必须的：排队者算 ETA 靠的就是心跳里的 done/total，
    session 级的锁如果只在开头打一次心跳，跑满 `stale_after_s` 之后
    就会被排队者当成死锁抢走。
    """

    def __init__(self, lock: GpuLock, total: int) -> None:
        self.lock = lock
        self.total = total
        self.done = 0

    def beat(self) -> None:
        self.done += 1
        self.lock.heartbeat(self.done, self.total)


@pytest.fixture(scope="session", autouse=True)
def live_gpu(request: pytest.FixtureRequest) -> _LiveGpu:
    """整个 session 独占 GPU；拿不到就中止，不跑注定失真的基准。"""
    lock = GpuLock(default_lock_path(), owner="pytest-live", stale_after_s=STALE_AFTER_S)
    try:
        lock.acquire(timeout=QUEUE_TIMEOUT_S)
    except GpuLockBusy as exc:
        info = lock.peek()
        held = f"，当前由 {info.owner} 占用（进度 {info.progress}）" if info else ""
        pytest.fail(f"拿不到 GPU 锁，中止 live 测试：并发跑评测会让所有延迟与吞吐数字失真{held}"
                    f"（等待上限 {QUEUE_TIMEOUT_S:.0f}s）：{exc}")
    gpu = _LiveGpu(lock, max(request.session.testscollected, 1))
    gpu.beat()
    try:
        yield gpu
    finally:
        lock.release()


@pytest.fixture(autouse=True)
def _heartbeat(live_gpu: _LiveGpu) -> None:
    """每个测试续一次命。"""
    live_gpu.beat()
