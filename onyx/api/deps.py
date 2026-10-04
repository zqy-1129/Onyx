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
        gpu_stale_after_s: float | None = None,
    ) -> AppState:
        from onyx.eval.gpu_lock import DEFAULT_GPU_STALE_AFTER_S

        return cls(
            runtime=runtime,
            broker=broker or SseBroker(),
            traces=TraceRepo(runtime.db),
            usage=UsageRepo(runtime.db),
            models=ModelRepo(runtime.db),
            # 默认机器级路径；只有测试与"确实要换一台 GPU"的部署才覆盖它。
            # 阈值由调用方注入（配置文件 [gpu].stale_after_s）而不是在这里读全局配置：
            # 装配层决定"用哪把锁、多快算死"，这里只负责照做。
            gpu_lock=GpuLock(
                gpu_lock_path or default_lock_path(),
                owner=f"serve:{runtime.provider.id}",
                stale_after_s=(
                    DEFAULT_GPU_STALE_AFTER_S if gpu_stale_after_s is None else gpu_stale_after_s
                ),
            ),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.onyx
