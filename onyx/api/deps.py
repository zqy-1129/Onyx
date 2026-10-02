"""请求上下文与依赖注入。"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from fastapi import Request

from onyx.api.sse import SseBroker
from onyx.runtime import Runtime
from onyx.store.repos import ModelRepo, TraceRepo, UsageRepo


@dataclass
class AppState:
    runtime: Runtime
    broker: SseBroker
    traces: TraceRepo
    usage: UsageRepo
    models: ModelRepo
    #: 本地单 GPU 是独占资源（DESIGN §8.5）。Playground 与评测共用这把锁，
    #: 否则两个请求互相踩，所有延迟数字都失去意义。S13 会换成带队列的调度器。
    gpu_lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def of(cls, runtime: Runtime, broker: SseBroker | None = None) -> AppState:
        return cls(
            runtime=runtime,
            broker=broker or SseBroker(),
            traces=TraceRepo(runtime.db),
            usage=UsageRepo(runtime.db),
            models=ModelRepo(runtime.db),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.onyx
