"""FastAPI 应用工厂。

用工厂而不是模块级 `app`：测试要能注入 MockProvider 的 runtime，
生产要能换数据目录，两者都不该靠 monkeypatch 全局变量。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from onyx import __version__
from onyx.api.deps import AppState
from onyx.core.errors import (
    CapabilityMissing,
    EvalError,
    EvalQueueFull,
    OnyxError,
    ProviderRejected,
    ProviderUnreachable,
    RequestTimeout,
)
from onyx.runtime import Runtime, build_runtime

log = logging.getLogger("onyx.api")

#: OnyxError → HTTP 状态码。集中在一处，避免每个路由各写一套。
#: 这里是**精确类型**查表（见 `_onyx_error`），新增错误族成员要一起登记，
#: 否则它会静默落到 500，前端看到的是"服务出错"而不是"任务名写错了"。
_ERROR_STATUS: dict[type[OnyxError], int] = {
    ProviderUnreachable: 503,
    RequestTimeout: 504,
    ProviderRejected: 502,
    CapabilityMissing: 422,
    EvalError: 422,
    EvalQueueFull: 429,
}


def iso_before(seconds: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds)).isoformat(timespec="microseconds")


def create_app(
    runtime: Runtime | None = None,
    *,
    base_url: str = "http://127.0.0.1:11434",
    provider_kind: str = "ollama",
    provider_id: str = "ollama-local",
    db_path: Path | str | None = None,
    sample_gpu: bool = True,
    gpu_lock_path: Path | str | None = None,
    gpu_stale_after_s: float | None = None,
    token: str | None = None,
    read_only: bool = False,
    event_sinks: tuple[str, ...] = (),
    alert_rule: Any = None,
    alert_channels: tuple[Any, ...] = (),
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173"),
) -> FastAPI:
    owns_runtime = runtime is None
    resolved = runtime or build_runtime(
        base_url=base_url, provider_kind=provider_kind, provider_id=provider_id,
        db_path=db_path, sample_gpu=sample_gpu, event_log=False, event_sinks=event_sinks,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        # 上次进程退出/崩溃留下的 running 行：不标出来，看板会一直显示"这条还在跑"，
        # 而实际上已经没有任何线程在为它工作
        orphans = state.eval_service.reclaim_orphans()
        if orphans:
            log.warning("服务启动时发现 %d 条没跑完的评测，已标为 error：%s",
                        len(orphans), ", ".join(orphans[:5]))
        if state.alert_service is not None:
            state.alert_service.start()
            log.info("告警轮询已启动：出口 %s · 每 %.1fs 问一次库",
                     ", ".join(c.name for c in state.alert_service.channels),
                     state.alert_service.poll_s)
        yield
        # 先停后台线程再关库：反过来会让正在写的线程对着一个已关闭的连接报错
        if state.alert_service is not None:
            state.alert_service.stop()
        state.eval_service.shutdown()
        if owns_runtime:
            resolved.close()

    app = FastAPI(
        title="Onyx", version=__version__,
        description="本地大模型观测与评测看板 API",
        docs_url="/api/docs", openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=list(cors_origins), allow_methods=["*"], allow_headers=["*"],
    )
    if token:
        # 闸装在 CORS 之后：预检请求不带 Authorization，先拦就会把 CORS 自己挡死
        from onyx.api.auth import install_auth

        install_auth(app, token=token, read_only=read_only)

    state = AppState.of(resolved, gpu_lock_path=gpu_lock_path,
                        gpu_stale_after_s=gpu_stale_after_s,
                        alert_rule=alert_rule, alert_channels=alert_channels)
    # gateway 的事件同时进 SSE 广播
    resolved.events.add(state.broker)
    app.state.onyx = state
    app.state.owns_runtime = owns_runtime

    from onyx.api.routes import alerts, evals, fleet, playground, tools, traces

    app.include_router(fleet.router)
    app.include_router(traces.router)
    app.include_router(playground.router)
    app.include_router(evals.router)
    app.include_router(tools.router)
    app.include_router(alerts.router)

    @app.exception_handler(OnyxError)
    async def _onyx_error(_: Request, exc: OnyxError) -> JSONResponse:
        status = _ERROR_STATUS.get(type(exc), 500)
        # 只回传 code + message + detail，绝不回传堆栈（会泄漏路径与内部结构）
        payload: dict[str, Any] = {"error": {"code": exc.code, "message": exc.message}}
        if exc.detail:
            payload["error"]["detail"] = exc.detail
        if status >= 500:
            log.warning("未分类错误 %s: %s", exc.code, exc.message, exc_info=True)
        return JSONResponse(status_code=status, content=payload)

    return app
