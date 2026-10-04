"""局域网共享时的一道闸：token + 可选只读。

默认绑 127.0.0.1 是"隐形安全边界"——它从来不是鉴权，只是碰巧没人能连上。
一旦 `--host 0.0.0.0`（或任何非回环地址），同网段的任何人都能：
读走你的全部 prompt 与原始 body、往你的 GPU 上打请求、unload 你正在用的模型。
所以 `onyx serve` 在非回环绑定且没有 token 时**拒绝启动**，而不是"先起来再说"。

设计取舍：
- **只闸 /api**：SPA 外壳与静态资源不含数据，拦住它们只会让页面白屏而不会保护任何东西；
  数据出口全在 /api 下（含 `/api/docs` 与 `/api/openapi.json`，它们会暴露能力面，所以也在闸内）。
- **token 比较用 `hmac.compare_digest`**：普通 `==` 会按字节短路返回，
  在局域网里足够让一个反复请求的客户端逐字节猜出 token。
- **SSE 允许 `?token=`**：`EventSource` 不能设请求头。这不是偷懒的口子——
  它写在文档里，并明确告知 query 参数会进访问日志与 Referer，能走 header 就走 header。
- **只读模式**是给"给别人看一眼看板"的场景：GET/HEAD 放行，写操作 403 并说明怎么放开。
  它替代不了鉴权，只是把"共享看板"和"共享操作台"分开。
"""

from __future__ import annotations

import hmac
import ipaddress
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

#: 例外：`localhost` 是名字不是地址，`ipaddress` 解析不了它，所以单独认。
#: `0.0.0.0` / `::` 明确**不算**回环——它们是"所有网卡"。
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_loopback(host: str) -> bool:
    """只有"字面上就是回环地址"才算回环。

    不能用 `startswith("127.")`：`127.0.0.1.evil.com` 也满足它，而它是攻击者控制的域名。
    所以拿 `ipaddress` 判，判不出 IP 的一律当作非回环（要 token）——宁可多问一次。
    """
    value = (host or "").strip().strip("[]")
    if not value:
        return False
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _deny(status: int, code: str, message: str, **detail: object) -> JSONResponse:
    """错误体形状与 `api/routes` 一致：前端只认 `{error:{code,message,detail}}`。

    形状不一致的话，看板上会出现"请求失败了但不知道为什么"，
    而未授权恰恰是最需要说清下一步的一种失败。
    """
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "detail": detail}},
        headers={"WWW-Authenticate": "Bearer"} if status == 401 else {},
    )


def _supplied_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    # EventSource 设不了请求头，所以 SSE 只能走 query。文档里写明：能走 header 就走 header
    query = request.query_params.get("token")
    return query or None


def install_auth(app: FastAPI, *, token: str, read_only: bool = False) -> None:
    """给 app 装上 token 闸（可选再加只读）。必须在启动前调用。"""
    if not token:
        raise ValueError("install_auth 需要非空 token；空 token 装出来的闸等于没装")

    @app.middleware("http")
    async def _guard(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse | object]]
    ):
        if not request.url.path.startswith("/api"):
            return await call_next(request)  # type: ignore[return-value]
        if not hmac.compare_digest(_supplied_token(request) or "", token):
            return _deny(
                401, "UNAUTHORIZED",
                "这个 Onyx 看板需要 token 才能读数据",
                hint="带 Authorization: Bearer <token>；SSE 可用 ?token=（会进访问日志）",
            )
        if read_only and request.method.upper() not in SAFE_METHODS:
            return _deny(
                403, "READ_ONLY",
                f"看板以只读模式运行，{request.method} 被拒绝",
                hint="需要操作（Playground / unload / 导入）时，用 `--no-read-only` 重新启动 serve",
            )
        return await call_next(request)  # type: ignore[return-value]
