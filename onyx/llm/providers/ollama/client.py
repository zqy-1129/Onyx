"""Ollama HTTP 客户端：只做传输与错误归一，不含业务语义。

所有第三方异常在这里被翻译成 `onyx.core.errors`，上层永远看不到 httpx 的栈。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any

import httpx

from onyx.core.errors import ProviderRejected, ProviderUnreachable, RequestTimeout

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        *,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers=headers or {}, transport=transport
        )

    # ── 传输 ──────────────────────────────────────────────────────
    def get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._send("GET", path, params=params)

    def post_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._send("POST", path, json_body=payload or {})

    def delete_json(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        return self._send("DELETE", path, json_body=payload or {})

    def post_ndjson(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """流式 POST，逐行产出 JSON 对象。

        Ollama 原生流是 application/x-ndjson；空行与非 JSON 行直接跳过，
        但**解析失败的行要保留原文**塞进 `_unparsed` 键，供事后诊断。
        """
        started = time.monotonic()
        try:
            with self._client.stream("POST", path, json=payload) as resp:
                self._raise_for_status(resp, started)
                for line in resp.iter_lines():
                    if not line or not line.strip():
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        yield {"_unparsed": line}
        except httpx.TimeoutException as exc:
            raise RequestTimeout(
                f"流式请求超时: {path}", detail={"path": path, "elapsed_ms": _ms(started)}
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(
                f"无法连接 Ollama: {type(exc).__name__}: {exc}",
                base_url=self.base_url, elapsed_ms=_ms(started),
            ) from exc

    def _send(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        started = time.monotonic()
        try:
            resp = self._client.request(method, path, params=params, json=json_body)
        except httpx.TimeoutException as exc:
            raise RequestTimeout(
                f"{method} {path} 超时", detail={"path": path, "elapsed_ms": _ms(started)}
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(
                f"无法连接 Ollama（{type(exc).__name__}）。服务是否已启动？",
                base_url=self.base_url, elapsed_ms=_ms(started),
            ) from exc
        self._raise_for_status(resp, started)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise ProviderRejected(
                f"响应不是合法 JSON: {path}", status=resp.status_code, body=resp.text
            ) from exc

    def _raise_for_status(self, resp: httpx.Response, started: float) -> None:
        if resp.status_code < 400:
            return
        body = _safe_text(resp)
        message = body
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and parsed.get("error"):
                message = str(parsed["error"])
        except json.JSONDecodeError:
            pass
        raise ProviderRejected(
            f"Ollama 返回 {resp.status_code}: {message[:300]}",
            status=resp.status_code,
            body=body,
            detail={"elapsed_ms": _ms(started)},
        )

    # ── 生命周期 ──────────────────────────────────────────────────
    def is_reachable(self, timeout: float = 3.0) -> bool:
        try:
            httpx.get(f"{self.base_url}/api/version", timeout=timeout).raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 2)


def _safe_text(resp: httpx.Response) -> str:
    try:
        return resp.text
    except Exception:  # noqa: BLE001 - 流式响应体可能已不可读，错误信息不能因此丢失
        return "<unreadable body>"
