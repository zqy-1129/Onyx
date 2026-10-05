"""请求上下文与依赖注入。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request

from onyx.api.sse import SseBroker
from onyx.eval.gpu_lock import GpuLock, default_lock_path
from onyx.eval.service import EvalService
from onyx.obs.alerts.service import AlertService
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
    #: 界面发起的评测走这条进程内单飞队列（S23）。它用的是**同一把锁的路径**，
    #: 所以"看板发起的评测"与"CLI 发起的评测"照样互斥——只是本进程内再多排一层队。
    eval_service: EvalService
    #: 告警轮询（S27）。`None` = 这个进程没被装配成会发通知——
    #: 判不出来就装作"有出口"，会让人以为没收到通知是网络问题。
    alert_service: Any = None

    @classmethod
    def of(
        cls, runtime: Runtime, broker: SseBroker | None = None,
        *, gpu_lock_path: Path | str | None = None,
        gpu_stale_after_s: float | None = None,
        alert_rule: Any = None, alert_channels: Sequence[Any] = (),
    ) -> AppState:
        from onyx.eval.gpu_lock import DEFAULT_GPU_STALE_AFTER_S

        lock = GpuLock(
            gpu_lock_path or default_lock_path(),
            owner=f"serve:{runtime.provider.id}",
            stale_after_s=(
                DEFAULT_GPU_STALE_AFTER_S if gpu_stale_after_s is None else gpu_stale_after_s
            ),
        )
        return cls(
            runtime=runtime,
            broker=broker or SseBroker(),
            traces=TraceRepo(runtime.db),
            usage=UsageRepo(runtime.db),
            models=ModelRepo(runtime.db),
            # 默认机器级路径；只有测试与"确实要换一台 GPU"的部署才覆盖它。
            # 阈值由调用方注入（配置文件 [gpu].stale_after_s）而不是在这里读全局配置：
            # 装配层决定"用哪把锁、多快算死"，这里只负责照做。
            gpu_lock=lock,
            eval_service=EvalService(
                runtime.gateway, runtime.db,
                gpu_lock_path=lock.path, gpu_stale_after_s=lock.stale_after_s,
            ),
            # 规则与出口都由装配层（cli serve）注入：deps 不读全局配置文件，
            # 否则测试里 create_app(...) 会悄悄往自己的临时目录写通知
            alert_service=(AlertService(runtime.db, rule=alert_rule, channels=alert_channels)
                           if alert_rule is not None else None),
        )


def get_state(request: Request) -> AppState:
    return request.app.state.onyx
