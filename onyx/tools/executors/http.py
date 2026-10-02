"""http 执行器：把一个人工审核过的 HTTP 端点变成工具，不需要写 Python。

两个刻意的设计决定：

1. **URL 只来自定义，参数只能进 query/body。**
   所以模型无法把请求指向别的主机。这也是不提供 `http_get(url=...)`
   这种"通用抓取"工具的理由——参数来自模型输出，等价于把 SSRF 开放给模型。

2. **deadline 必须传播到 socket。**
   `run_with_deadline` 到点只是放弃等待，底层线程还活着（Python 杀不掉线程）。
   若 httpx 自己没有一个同样紧的超时，一个卡住的上游会永久占用线程池里的一个 worker；
   多来几次整个工具子系统就瘫了，而且现象是"越来越慢"而不是"报错"——最难查的那种。
   所以这里取 `min(调用方 deadline, 定义 timeout_ms, 默认)` 直接塞进 httpx.Timeout。

每次调用新建一个 `httpx.Client`（用完即关）。工具调用不是热路径，
换来的是没有连接池生命周期要管——契约测试会为每条断言新建执行器，
持有长连接会直接泄漏。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from onyx.core.errors import ToolRuntime, ToolTimeout, ToolUnknown
from onyx.tools.contract import text_schema
from onyx.tools.executor import ToolCtx, guarded_call
from onyx.tools.spec import SideEffect, ToolDef, ToolKind, ToolResult

#: 响应体截断上限。工具输出会进上下文，一个 50MB 的 JSON 能直接把窗口吃光
MAX_BODY_BYTES = 65_536
DEFAULT_HTTP_TIMEOUT_MS = 15_000
ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})
BODY_MODES = frozenset({"json", "query", "none"})


def _httpx():
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - 取决于安装的 extras
        raise ToolUnknown(
            "http 执行器需要 httpx：uv sync --extra runtime",
            detail={"kind": "missing_extra", "extra": "runtime"},
        ) from exc
    return httpx


class HttpExecutor:
    kind = ToolKind.HTTP
    #: 隔离证据类型：`real_calls` 是真实发出的请求数，所以这条是**实测**而非声明
    ISOLATION_PROOF = "counter"

    def __init__(
        self,
        definition: ToolDef,
        *,
        transport: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._httpx = _httpx()
        self._definition = definition
        self._config = _read_config(definition)
        self._transport = transport
        self._extra_headers = dict(headers or {})
        #: 真的发出去的请求数。契约断言用它证明 mock 策略下零网络
        self.real_calls = 0

    def spec(self) -> ToolDef:
        return self._definition

    def call(self, name: str, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        definition = self._definition
        if name != definition.name:
            return ToolResult(
                ok=False, error_kind="unknown_tool",
                error=f"http 执行器绑定的是 {definition.name!r}，收到 {name!r}",
            )
        return guarded_call(definition, args, ctx, lambda clean: self._send(clean, ctx), name=name)

    # ── 实际请求 ──────────────────────────────────────────────────
    def _send(self, clean: dict[str, Any], ctx: ToolCtx) -> dict[str, Any]:
        httpx = self._httpx
        config = self._config
        timeout_ms = _effective_timeout_ms(config, self._definition, ctx)

        params = clean if config["body"] == "query" else None
        payload = clean if config["body"] == "json" else None
        headers = {**config["headers"], **self._extra_headers}

        started = time.monotonic()
        self.real_calls += 1
        try:
            with httpx.Client(
                transport=self._transport, timeout=httpx.Timeout(timeout_ms / 1000),
                headers=headers, follow_redirects=False,
            ) as client:
                response = client.request(
                    config["method"], config["url"], params=params, json=payload
                )
        except httpx.TimeoutException as exc:
            # 必须是 ToolTimeout 而不是笼统的 error：超时是可重试的，5xx 不是
            raise ToolTimeout(
                f"{config['method']} {config['url']} 超过 {timeout_ms}ms",
                detail={"kind": "http_timeout", "timeout_ms": timeout_ms, "url": config["url"]},
            ) from exc
        except httpx.HTTPError as exc:
            raise ToolRuntime(
                f"无法连接 {config['url']}: {type(exc).__name__}: {exc}"[:400],
                detail={"kind": "unreachable", "url": config["url"]},
            ) from exc
        elapsed_ms = round((time.monotonic() - started) * 1000, 3)

        output = _decode(response)
        if response.status_code >= 400:
            # 上游 4xx/5xx 是**工具侧**的失败。至于是不是模型参数引起的，
            # 交给 S12 的 fire-and-verify 结合 status 判断，这里不猜
            raise ToolRuntime(
                f"上游返回 {response.status_code}",
                detail={"kind": "http_status", "status": response.status_code,
                        "body": output["body"] if isinstance(output["body"], str)
                        else json.dumps(output["body"], ensure_ascii=False)[:500],
                        "elapsed_ms": elapsed_ms},
            )
        return output


def _decode(response: Any) -> dict[str, Any]:
    """把响应变成输出。

    **不放耗时**：`guarded_call` 已经在 `extra.latency_ms` 里记了，而输出会被
    契约断言 `read_is_idempotent` 逐字段比较——把一个每次都变的数字放进去，
    等于让所有 http 工具天生"不幂等"，那条断言就再也测不出东西了。
    """
    raw = response.content or b""
    truncated = len(raw) > MAX_BODY_BYTES
    text = raw[:MAX_BODY_BYTES].decode(response.encoding or "utf-8", "replace")
    try:
        body: Any = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        body = text
    return {
        "status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "body": body,
        "truncated": truncated,
    }


def _effective_timeout_ms(config: dict[str, Any], definition: ToolDef, ctx: ToolCtx) -> int:
    """取三者最小值：调用方 deadline、定义 timeout_ms、http 配置 timeout_ms。

    任何一个更紧的约束都必须生效——取最大值等于让最松的那个说了算。
    """
    candidates = [
        value for value in (
            ctx.deadline_ms, definition.timeout_ms, config.get("timeout_ms"),
            ctx.policy.default_timeout_ms, DEFAULT_HTTP_TIMEOUT_MS,
        )
        if isinstance(value, int) and value > 0
    ]
    return min(candidates) if candidates else DEFAULT_HTTP_TIMEOUT_MS


def _read_config(definition: ToolDef) -> dict[str, Any]:
    """从 `definition.extra["http"]` 读端点配置。

    配置错必须在**构造时**就炸，而不是等到第一次调用——否则错误会出现在
    某条 trace 里，看起来像模型的问题。
    """
    raw = definition.extra.get("http")
    if not isinstance(raw, dict) or not raw:
        # 空配置与缺失配置报同一种：都是"这个工具还没配好"，
        # 报 bad_url 会把人引去查 URL 写法，方向就错了
        raise ToolUnknown(
            f"工具 {definition.name} 缺少 extra.http 配置",
            detail={"kind": "missing_http_config", "expected": ["url", "method", "body"]},
        )
    url = str(raw.get("url") or "")
    if not url.startswith(("http://", "https://")):
        raise ToolUnknown(
            f"extra.http.url 必须是 http(s) 绝对地址，实际 {url!r}",
            detail={"kind": "bad_url", "url": url[:200]},
        )
    method = str(raw.get("method") or ("GET" if not raw.get("body") else "POST")).upper()
    if method not in ALLOWED_METHODS:
        raise ToolUnknown(
            f"不支持的 HTTP 方法 {method!r}",
            detail={"kind": "bad_method", "allowed": sorted(ALLOWED_METHODS)},
        )
    body = str(raw.get("body") or ("query" if method in {"GET", "HEAD", "DELETE"} else "json"))
    if body not in BODY_MODES:
        raise ToolUnknown(
            f"extra.http.body 必须是 {sorted(BODY_MODES)} 之一，实际 {body!r}",
            detail={"kind": "bad_body_mode"},
        )
    headers = raw.get("headers") or {}
    if not isinstance(headers, dict):
        raise ToolUnknown("extra.http.headers 必须是对象", detail={"kind": "bad_headers"})
    return {
        "url": url,
        "method": method,
        "body": body,
        "headers": {str(k): str(v) for k, v in headers.items()},
        "timeout_ms": raw.get("timeout_ms"),
    }


#: 契约样本用的假主机名：`.invalid` 是 RFC 2606 保留的永不解析域名，
#: 万一 transport 注入失效，请求会 DNS 失败而不是打到某个真实服务上
CONTRACT_HOST = "http://contract.invalid"


def contract_target() -> tuple[ToolDef, Callable[[ToolDef], Any], dict[str, Any], Any] | None:
    """离线 http 契约样本：本地 MockTransport 应答，**零真实网络**。

    返回 None 表示没装 httpx（`runtime` extra）。调用方必须显示"未安装"，
    不能显示"通过"——未知就是未知（UI_DESIGN R2 的同一条纪律）。

    慢工具那条断言在这里是**真的**在测 deadline 传播：httpx 会把有效超时放进
    `request.extensions["timeout"]`（已实测），所以 handler 能直接检查
    "执行器有没有把 1ms 的 deadline 交给 socket"。若没传播，handler 返回 500，
    契约断言随即变红——这正是 IMPLEMENTATION.md S11 要求复现的那个失败形态。

    返回 `(样本定义, 执行器工厂, 合法参数, synth)`，直接喂给 `contract.run_contracts`。
    """
    try:
        import httpx
    except ImportError:  # pragma: no cover - 取决于安装的 extras
        return None

    sample = ToolDef(
        name="contract_http",
        description="契约测试用的 HTTP 工具，由本地 MockTransport 应答，不发真实请求",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，例如「北京」"}},
            "required": ["city"],
        },
        kind=ToolKind.HTTP,
        side_effect=SideEffect.NETWORK,
        extra={"http": {"url": f"{CONTRACT_HOST}/weather", "method": "GET", "body": "query"}},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        effective = (request.extensions.get("timeout") or {}).get("read")
        if request.url.path == "/slow":
            if effective is None or effective > 0.05:
                return httpx.Response(
                    500, json={"error": "deadline 未传播到 socket", "read_timeout": effective}
                )
            raise httpx.ReadTimeout("模拟 socket 读超时", request=request)
        if request.url.path == "/write":
            return httpx.Response(200, json={"written": True})
        return httpx.Response(200, json={"city": request.url.params.get("city"), "temp_c": 21})

    transport = httpx.MockTransport(handler)

    def synth(name: str, role: str) -> ToolDef:
        path = "/slow" if role == "slow" else "/write"
        return ToolDef(
            name=name,
            description="契约测试用的合成 HTTP 工具，验证超时传播与沙箱拒绝",
            parameters=text_schema(),
            kind=ToolKind.HTTP,
            side_effect=SideEffect.READ if role == "slow" else SideEffect.WRITE,
            extra={"http": {"url": CONTRACT_HOST + path, "method": "GET", "body": "query"}},
        )

    return sample, lambda d: HttpExecutor(d, transport=transport), {"city": "北京"}, synth
