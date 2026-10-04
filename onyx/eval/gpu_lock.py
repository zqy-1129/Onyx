"""GPU 独占锁（DESIGN §8.5）。

单 GPU 上 eval、benchmark、playground 会互相踩，而且踩出来的现象**不是报错**：
两个模型同时驻留会触发 CPU offload，吞吐差一个数量级，但数字看起来"正常"，
于是你会把一个被污染的基准当成模型能力记下来。所以必须串行，而且必须显式。

实现选择：
- **文件锁 + 心跳**，不用 OS 级 advisory lock。跨平台（Windows 没有 flock），
  而且排队者能读到持有者的**进度**，从而算出 ETA——只有"锁住了/没锁住"两个状态
  的锁，排队者只能干等。
- **靠心跳过期判死锁，不靠 PID 存活**。判断一个 PID 是否还活着在 Windows 上
  要么需要 pywin32，要么用 `os.kill(pid, 0)`（在 Windows 上语义不同且可能误伤）。
  心跳过期是纯文件语义，可移植且可测：进程崩了就不再写心跳，超过阈值自动可回收。
- 写入用 `os.replace` 原子替换，所以排队者永远读不到半截 JSON。
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from onyx.core.clock import SYSTEM_CLOCK, Clock, utc_now_iso
from onyx.core.errors import GpuLockBusy


def default_lock_path() -> Path:
    """机器级全局的锁路径。

    **不能放在数据目录里**：`ONYX_DATA_DIR` 可以按实例覆盖（多实例并存、测试隔离），
    而 GPU 是整台机器只有一块的资源。锁跟着数据目录走的话，
    两个实例会各自锁各自的文件，然后照样同时往显存里塞模型——
    这把锁存在的唯一理由就没了。
    """
    return Path(tempfile.gettempdir()) / "onyx-gpu.lock"


#: 心跳超过这个秒数没更新就认为持有者已经死了，锁可回收。
#: **必须明显大于单条样本的最长耗时**：runner 在每条样本前后各打一次心跳，
#: 所以这个值的实际下界是"一次请求的耗时"。27B 模型上长 prompt 可能跑几分钟，
#: 调小它会误伤活着的持有者，让两个评测同时占 GPU——那正是这把锁要防的事。
DEFAULT_STALE_AFTER_S = 180.0
DEFAULT_POLL_S = 0.5
#: 接管死锁时搬走旧锁文件的重试次数。Windows 的 rename 会被"文件正被别的句柄打开"
#: 短暂拒绝（杀软/索引器），一次就放弃等于没人能接管死锁。
CLAIM_RETRIES = 3


@dataclass(frozen=True, slots=True)
class LockInfo:
    """锁文件的内容。排队者靠它算 ETA，所以进度与心跳都必须在里面。"""

    owner: str
    pid: int
    started_at: str
    heartbeat_at: str
    done: int = 0
    total: int = 0
    host: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def elapsed_s(self, now_iso: str) -> float | None:
        return _seconds_between(self.started_at, now_iso)

    def eta_s(self, now_iso: str) -> float | None:
        """剩余时间的估计。**样本数为 0 或还没开始时返回 None，不返回 0**。

        返回 0 会被读成"马上就好"，而真相是"还不知道"。
        """
        if self.done <= 0 or self.total <= self.done:
            return None
        elapsed = self.elapsed_s(now_iso)
        if elapsed is None or elapsed <= 0:
            return None
        return elapsed / self.done * (self.total - self.done)

    @property
    def progress(self) -> str:
        if not self.total:
            return "—"
        return f"{self.done}/{self.total}"


#: 排队时的通知回调：(已等待秒数, 当前持有者信息 或 None)
WaitCB = Callable[[float, LockInfo | None], None]


class GpuLock:
    """一把可排队、有心跳、能报 ETA 的文件锁。"""

    def __init__(
        self,
        path: Path | str,
        *,
        owner: str = "",
        poll_s: float = DEFAULT_POLL_S,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
        clock: Clock = SYSTEM_CLOCK,
        on_wait: WaitCB | None = None,
        host: str = "",
    ) -> None:
        self.path = Path(path)
        self.owner = owner or f"pid:{os.getpid()}"
        self.poll_s = poll_s
        self.stale_after_s = stale_after_s
        self.clock = clock
        self.on_wait = on_wait
        self.host = host or _hostname()
        self._held = False

    # ── 状态 ──────────────────────────────────────────────────────
    @property
    def held(self) -> bool:
        return self._held

    def peek(self) -> LockInfo | None:
        """只看不拿。看板用它显示"谁在跑 GPU、还要多久"。"""
        return read_lock(self.path)

    def is_busy(self) -> bool:
        info = self.peek()
        return info is not None and not self._is_stale(info)

    # ── 获取与释放 ────────────────────────────────────────────────
    def acquire(self, *, timeout: float | None = None) -> LockInfo:
        """阻塞获取。超时抛 `GpuLockBusy`（带持有者与 ETA，好让人知道该等多久）。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        waited = 0.0
        while True:
            info = self.peek()
            if (info is None or self._is_stale(info)) and self._try_write(info):
                self._held = True
                return self._current()
            if deadline is not None and time.monotonic() >= deadline:
                raise GpuLockBusy(
                    f"GPU 被 {info.owner if info else '未知持有者'} 占用"
                    + (f"，进度 {info.progress}" if info else "")
                    + (f"，预计还需 {_fmt_eta(info.eta_s(utc_now_iso()))}"
                       if info and info.eta_s(utc_now_iso()) is not None else ""),
                    detail={"owner": info.owner if info else "",
                            "progress": info.progress if info else "",
                            "eta_s": info.eta_s(utc_now_iso()) if info else None,
                            "stale": bool(info and self._is_stale(info))},
                )
            if self.on_wait is not None:
                self.on_wait(waited, info)
            time.sleep(self.poll_s)
            waited += self.poll_s

    def heartbeat(self, done: int, total: int, **extra: Any) -> None:
        """刷新心跳与进度。**排队者的 ETA 完全依赖这个调用**，所以每条样本后都要打。"""
        if not self._held:
            return
        self._write(LockInfo(
            owner=self.owner, pid=os.getpid(), started_at=self._started_at,
            heartbeat_at=utc_now_iso(), done=done, total=total,
            host=self.host, extra=extra,
        ))

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        info = self.peek()
        # 只删自己的锁：如果它已被别人（在我们被判死之后）接管，删掉就等于释放别人的锁
        if info is not None and info.pid == os.getpid() and info.owner == self.owner:
            with contextlib.suppress(OSError):
                self.path.unlink()

    def __enter__(self) -> GpuLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.release()

    # ── 内部 ──────────────────────────────────────────────────────
    @property
    def _started_at(self) -> str:
        return getattr(self, "_start_iso", None) or utc_now_iso()

    def _current(self) -> LockInfo:
        return LockInfo(
            owner=self.owner, pid=os.getpid(), started_at=self._start_iso,
            heartbeat_at=utc_now_iso(), done=0, total=0, host=self.host,
        )

    def _try_write(self, previous: LockInfo | None) -> bool:
        """原子地占锁。

        两条路径：
        - 锁文件不存在 ⇒ `O_CREAT|O_EXCL` 直接创建，内核保证同一时刻只有一个赢家。
        - 要接管（心跳过期 / 内容坏到读不出来）⇒ 先 `os.rename` 把它**搬走**。
          rename 要求源文件存在，所以并发的接管者里只有一个能成功，其余全部拿到
          FileNotFoundError 回去重新排队。搬走之后还要再确认内容确实过期——
          上一步的 peek 与这一次 rename 之间有窗口，搬走的可能是别人刚写的新锁。

        为什么不用"直接覆盖 + 回读确认"：那样两个接管者**顺序**执行时都会成功——
          后一个的回读看到的是自己，而前一个早已返回。两个持有者比没有锁更危险，
          因为它看起来是安全的。
        """
        self._start_iso = utc_now_iso()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._reclaimable():
            return False
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(self._current()), ensure_ascii=False))
        return True

    def _reclaimable(self) -> bool:
        """当前锁文件是否可以被我接管；可接管时把它搬走并清掉。"""
        existing = self.peek()
        if existing is not None and not self._is_stale(existing):
            return False
        if not self.path.exists():
            return True

        token = f"{os.getpid()}.{self.clock.monotonic_ns()}"
        moved = self.path.with_name(f"{self.path.name}.stolen.{token}")
        # Windows 上 `rename` 会因为别的句柄正打开这个文件而短暂 EACCES（杀软/索引器
        # 很常见）。把它当成"别人先搬走了"就放弃，后果是**没人能接管一把死锁**——
        # 评测会无限排队等一个已经死掉的持有者，这是锁最坏的失效方式。
        for attempt in range(CLAIM_RETRIES):
            try:
                os.rename(self.path, moved)
                break
            except FileNotFoundError:
                return False  # 别人先搬走了，或文件刚好消失 ⇒ 回去重新排队
            except OSError:
                if attempt == CLAIM_RETRIES - 1:
                    return False  # 始终搬不动就排队，绝不"实在不行就覆盖"
                time.sleep(0.005 * (attempt + 1))
        try:
            stale = read_lock(moved)
            if stale is not None and not self._is_stale(stale):
                # 搬走的是别人刚写的新锁：原样放回去，自己排队
                os.replace(moved, self.path)
                return False
            return True
        finally:
            # 清理失败不影响锁语义：搬走的那个文件已经不在锁路径上了
            with contextlib.suppress(OSError):
                moved.unlink()

    def _write(self, info: LockInfo) -> None:
        """原子写：先写临时文件再 `os.replace`，排队者永远读不到半截 JSON。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(asdict(info), ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def _is_stale(self, info: LockInfo) -> bool:
        age = _seconds_between(info.heartbeat_at, utc_now_iso())
        return age is not None and age > self.stale_after_s


def read_lock(path: Path | str) -> LockInfo | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # 读到坏文件只可能是写入方崩溃在半路；当成没有锁，让下一个进程接管
        return None
    if not isinstance(payload, dict):
        return None
    known = {f for f in LockInfo.__slots__}  # type: ignore[attr-defined]
    return LockInfo(**{k: v for k, v in payload.items() if k in known})


def _seconds_between(start_iso: str, end_iso: str) -> float | None:
    from datetime import datetime

    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
    except (TypeError, ValueError):
        return None
    return (end - start).total_seconds()


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "未知"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _hostname() -> str:
    import socket

    try:
        return socket.gethostname()
    except OSError:  # pragma: no cover
        return ""
