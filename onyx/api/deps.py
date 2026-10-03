"""请求上下文与依赖注入。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from onyx.api.sse import SseBroker
from onyx.eval.gpu_lock import GpuLock, default_lock_path
from onyx.runtime import Runtime
from onyx.store.repos import ModelRepo, TraceRepo, UsageRepo


@dataclass
class AppState:
    runtime: Runtime
    broker: SseBroker
    traces: TraceRepo
    usage: UsageRepo
    models: ModelRepo
    #: 本地单 GPU 是独占资源（DESIGN §8.5）。
    #: 必须是**跨进程**的文件锁：评测跑在 `onyx eval` 这个进程里，看板跑在
    #: `onyx serve` 里，用 threading.Lock 的话两边各自锁各自的，什么都没防住。
    gpu_lock: GpuLock

    @classmethod
    def of(
        cls, runtime: Runtime, broker: SseBroker | None = None,
        *, gpu_lock_path: Path | str | None = None,
    ) -> AppState:
        return cls(
            runtime=runtime,
            broker=broker or SseBroker(),
            traces=TraceRepo(runtime.db),
            usage=UsageRepo(runtime.db),
            models=ModelRepo(runtime.db),
            # 默认机器级路径；只有测试与"确实要换一台 GPU"的部署才覆盖它
            gpu_lock=GpuLock(
                gpu_lock_path or default_lock_path(),
                owner=f"serve:{runtime.provider.id}",
                # 阈值必须大于单次请求的最长耗时，否则一个还在正常生成的长请求
                # 会被排队者判成死锁并抢走 GPU。这里对齐 provider 的 read 超时（600s）
                stale_after_s=600.0,
            ),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.onyx
