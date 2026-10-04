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
    OnyxError,
    ProviderRejected,
    ProviderUnreachable,
    RequestTimeout,
)
from onyx.runtime import Runtime, build_runtime

log = logging.getLogger("onyx.api")

#: OnyxError → HTTP 状态码。集中在一处，避免每个路由各写一套。
_ERROR_STATUS: dict[type[OnyxError], int] = {
    ProviderUnreachable: 503,
    RequestTimeout: 504,
    ProviderRejected: 502,
    CapabilityMissing: 422,
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
    event_sinks: tuple[str, ...] = (),
    cors_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173"),
) -> FastAPI:
    owns_runtime = runtime is None
    resolved = runtime or build_runtime(
        base_url=base_url, provider_kind=provider_kind, provider_id=provider_id,
        db_path=db_path, sample_gpu=sample_gpu, event_log=False, event_sinks=event_sinks,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> Any:
        yield
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

    state = AppState.of(resolved, gpu_lock_path=gpu_lock_path)
    # gateway 的事件同时进 SSE 广播
    resolved.events.add(state.broker)
    app.state.onyx = state
    app.state.owns_runtime = owns_runtime

    from onyx.api.routes import evals, fleet, playground, traces

    app.include_router(fleet.router)
    app.include_router(traces.router)
    app.include_router(playground.router)
    app.include_router(evals.router)

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
