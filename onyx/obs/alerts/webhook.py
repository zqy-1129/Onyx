"""webhook 出口（S28）：把命中推到机器外面。

**这是第三个网络例外**（前两个是 `executors.http` 与 `sinks.otlp`），已按契约要求
在 `pyproject.toml` 里逐条登记。用 httpx 而不是 stdlib `urllib.request` 是刻意的：
门禁的意义在于"新增一个顺手发请求的模块会立刻让 lint 失败"，
用标准库绕过去就等于把这条可见性换成省事。模块只在真的配了 URL 时才被 import。
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from onyx.obs.alerts.channels import Alert, ChannelResult

log = logging.getLogger("onyx.alerts")

#: 单次请求的超时。通知不是事务：宁可标成 failed 让人去查，也不要占着轮询线程不放。
DEFAULT_TIMEOUT_S = 5.0
#: 重试上限。断网几小时是本地机器的常态，无限重试会把"哪些异常没通知"变成不可查的问题，
#: 而失败必须落成一行 `status=failed` 的记录——那是它唯一的可追溯形态。
DEFAULT_RETRIES = 2
RETRY_BACKOFF_S = (0.5, 2.0)


def _real_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def redact(url: str) -> str:
    """去掉 query 与 fragment 的 URL。webhook 地址的 query 里通常就挂着 secret，
    而 detail 会进库、会出现在终端里。
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return "（URL 无法解析）"
    if not parts.scheme or not parts.netloc:
        return url[:80]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


class WebhookChannel:
    """POST 一段 JSON 到任意接收端。载荷是 `alert.as_dict()`——与文件出口同一个形状，
    两处各拼一份的话，"文件里有建议、webhook 里没有"这种分歧迟早会出现。
    """

    name = "webhook"

    def __init__(
        self,
        url: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        transport: Any = None,
        sleep: Any = None,
    ) -> None:
        import httpx

        if not url or not url.lower().startswith(("http://", "https://")):
            # 配错 URL 要在装配当场说，而不是等到第一次出事时发现"从来没发出去过"
            raise ValueError(f"webhook URL 必须是 http(s) 地址，实际是 {redact(url)!r}")
        self.url = url
        self.timeout = timeout
        self.retries = max(0, retries)
        self._client = httpx.Client(timeout=timeout, transport=transport)
        #: 退避怎么等：测试要能不倒退真时间
        self._sleep = sleep if sleep is not None else _real_sleep

    def close(self) -> None:
        self._client.close()

    def deliver(self, alert: Alert) -> ChannelResult:
        body = json.dumps(alert.as_dict(), ensure_ascii=False).encode("utf-8")
        last = ""
        for attempt in range(self.retries + 1):
            if attempt:
                self._sleep(RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)])
            try:
                resp = self._client.post(
                    self.url, content=body,
                    headers={"content-type": "application/json", "user-agent": "onyx-alerts"},
                )
            except Exception as exc:  # noqa: BLE001 - 网络失败要变成投递结果，不是异常
                last = f"{type(exc).__name__}: {exc}"[:160]
                continue
            if 200 <= resp.status_code < 300:
                return ChannelResult(True, f"{redact(self.url)} ← {resp.status_code}"
                                            f"{'（重试 ' + str(attempt) + ' 次）' if attempt else ''}")
            last = f"HTTP {resp.status_code}：{resp.text[:120]}"
            # 4xx 是"对方不收"，重试只是刷日志；5xx 与网络错误才值得再试一次
            if resp.status_code < 500:
                break
        return ChannelResult(False, f"{redact(self.url)} 投递失败：{last}"[:400])
