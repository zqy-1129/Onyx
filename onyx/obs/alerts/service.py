"""告警服务：轮询 → 判定 → 投递 → 落库留痕。

跑在 `serve` 的后台线程里，**不在请求路径上**：在被观测的那次请求里发外部 HTTP
等于让观测者影响被测（webhook 慢会进 TTFT，webhook 挂了会在 trace 里留一条
与模型无关的 error），而观测层的异常隔离又会把这种失败吞成一行 warning。

状态尽量不放在进程里：窗口计数读 `anomaly` 表，cooldown 读 `alert_trigger` 表，
所以重启既不会重复通知，也不会忘记刚才通知过什么。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.core.ids import new_trace_id
from onyx.obs.alerts.channels import Alert, AlertChannel, ChannelResult
from onyx.obs.alerts.rules import (
    DEFAULT_ALERT_SCAN_LIMIT,
    AlertRule,
    Hit,
    Row,
    evaluate,
)
from onyx.obs.anomalies import SPECS
from onyx.store.db import Database
from onyx.store.records import AlertTriggerRecord
from onyx.store.repos import AlertRepo, TraceRepo

log = logging.getLogger("onyx.alerts")


def message_for(hit: Hit) -> str:
    """通知正文 = 事实 + 规格里的"下一步做什么"。

    文案只从 `obs/anomalies.py` 的 SPECS 取，前端与 CLI 不许各写一份——
    一个只会说"出事了"的通知需要人再去查一遍，而那一步通常不会发生。
    """
    spec = SPECS.get(hit.code)
    meaning = spec.meaning if spec else f"未登记的异常码 {hit.code}"
    head = f"{hit.code}（{hit.severity}）：{meaning} · {hit.count_text()} 次 / {hit.window_s}s"
    if spec:
        head += f" · 建议：{spec.action}"
    return head + "（计数是下限：窗口内异常超过扫描上限）" if hit.truncated else head


class AlertService:
    def __init__(
        self,
        db: Database,
        *,
        rule: AlertRule,
        channels: Sequence[AlertChannel] = (),
        poll_s: float | None = None,
        scan_limit: int = DEFAULT_ALERT_SCAN_LIMIT,
    ) -> None:
        self.traces = TraceRepo(db)
        self.alerts = AlertRepo(db)
        self.rule = rule
        self.channels = tuple(channels)
        #: 轮询周期跟着规则走，但允许调用方覆盖（测试要跑得快）
        self.poll_s = poll_s if poll_s is not None else rule.poll_s
        self.scan_limit = scan_limit
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        #: 线程死过一次就知道"通知系统自己坏了"这件事必须能被外部看见，
        #: 而不是等到人发现从来没收到过通知才回头猜
        self.last_error = ""
        self.ticks = 0

    # ── 生命周期 ──────────────────────────────────────────────────
    def start(self) -> None:
        if not self.rule.enabled or not self.channels or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="onyx-alerts", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        # 出口自己拥有的东西（httpx 连接池）由关停负责回收。
        # 这里不让异常冒出去：关不上连接池不该让 serve 的死过程变成一场崩。
        for channel in self.channels:
            close = getattr(channel, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001 - 回收失败只记一行
                log.warning("告警出口 %s 关闭失败：%s", channel.name, exc)

    def _loop(self) -> None:
        while not self._stop.is_set():
            # 等待与判定分开计秒：poll_s 是"多久问一次库"，不是"占用多久"
            if self._stop.wait(self.poll_s):
                return
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - 后台线程绝不能带着服务一起死
                self.last_error = f"{type(exc).__name__}: {exc}"[:200]
                log.warning("告警轮询出错（%s），下一轮继续", self.last_error)

    # ── 一次完整的 扫描 → 判定 → 投递 → 留痕 ──────────────────────
    def tick(self, *, now: str | None = None) -> list[AlertTriggerRecord]:
        """跑一轮。返回这一轮写下的触发行（测试与 CLI 用它来断言，不靠睡）。"""
        current = now or utc_now_iso()
        since = _shift(current, -self.rule.window_s)
        with self._lock:
            self.ticks += 1
            rows = self.traces.anomalies_in_window(
                since=since, codes=self.rule.codes or None, limit=self.scan_limit
            )
            truncated = len(rows) >= self.scan_limit
            result = evaluate(
                [Row(id=r.id, code=r.code, severity=r.severity,
                     created_at=r.created_at, trace_id=r.trace_id) for r in rows],
                rule=self.rule,
                now=current,
                last_triggered_at=self.alerts.last_real_trigger_times(),
                truncated=truncated,
            )
        if result.undated:
            log.warning("告警窗口里有 %d 条异常的时间戳解析不出来，未计入次数", result.undated)
        return [rec for hit in result.hits for rec in self._deliver(hit, current)]

    def _deliver(self, hit: Hit, now: str) -> list[AlertTriggerRecord]:
        if hit.suppressed:
            # 抑制不写行：那是"没发消息"，而这张表记的是"发过什么、结果如何"。
            # 写成行的话，一条持续坏的异常会把审计淹没成心跳日志。
            return []
        alert = Alert(
            code=hit.code, severity=hit.severity, message=message_for(hit),
            n=hit.n, window_s=hit.window_s, triggered_at=now,
            rule=self.rule.as_dict(), first_anomaly_id=hit.first_id,
            last_anomaly_id=hit.last_id, trace_ids=hit.trace_ids, truncated=hit.truncated,
        )
        out: list[AlertTriggerRecord] = []
        for channel in self.channels:
            result = self._deliver_to(channel, alert)
            out.append(self._record(hit, channel.name, result, now))
        return out

    def _deliver_to(self, channel: AlertChannel, alert: Alert) -> ChannelResult:
        try:
            return channel.deliver(alert)
        except Exception as exc:  # noqa: BLE001 - 渠道坏不能带坏别的渠道
            return ChannelResult(False, f"渠道自身抛错 {type(exc).__name__}: {exc}"[:200])

    def _record(self, hit: Hit, channel: str, result: ChannelResult,
                now: str, *, is_test: bool = False) -> AlertTriggerRecord:
        rec = AlertTriggerRecord(
            id=new_trace_id(), created_at=now, code=hit.code, severity=hit.severity,
            rule=self.rule.as_dict(), n_in_window=hit.n, window_s=hit.window_s,
            first_anomaly_id=hit.first_id, last_anomaly_id=hit.last_id,
            trace_ids=hit.trace_ids, channel=channel,
            status="sent" if result.ok else "failed",
            detail=result.detail[:400], is_test=is_test,
        )
        self.alerts.insert_trigger(rec)
        if not result.ok:
            log.warning("告警投递失败（%s → %s）：%s", hit.code, channel, result.detail)
        return rec

    # ── 供 CLI / 测试直接走一遍出口 ───────────────────────────────
    def send_test(self, *, code: str = "CONTEXT_OVERFLOW", channel_filter: str | None = None,
                  message: str = "") -> list[AlertTriggerRecord]:
        """造一条**标明是测试**的通知走一遍出口。

        它落库时 `is_test=1`，因此不会污染 cooldown，也不会让人以为真出过事。
        """
        now = utc_now_iso()
        spec = SPECS.get(code)
        # 用真实那条消息的形状（含 SPECS 的"建议"）来测出口：
        # 只发一句"test"的话，通了也证明不了真出事时人能看到有用的话
        probe = Hit(code=code, severity=(spec.severity if spec else "error"), n=0,
                    window_s=self.rule.window_s, first_id="", last_id="")
        alert = Alert(
            code=code, severity=probe.severity,
            message=message or f"【测试】{message_for(probe)}（这不是真实异常，只验出口通不通）",
            n=0, window_s=self.rule.window_s, triggered_at=now, rule=self.rule.as_dict(),
            is_test=True,
        )
        chosen = [c for c in self.channels
                  if channel_filter is None or c.name == channel_filter]
        out: list[AlertTriggerRecord] = []
        for channel in chosen:
            result = self._deliver_to(channel, alert)
            out.append(self._record(
                Hit(code=code, severity=alert.severity, n=0, window_s=alert.window_s,
                    first_id="", last_id=""),
                channel.name, result, now, is_test=True,
            ))
        return out

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.rule.enabled,
            "channels": [c.name for c in self.channels],
            "poll_s": self.poll_s,
            "ticks": self.ticks,
            "last_error": self.last_error,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }


def _shift(iso: str, seconds: float) -> str:
    """把 ISO 时间平移若干秒；解析不出来就退回"现在 + 偏移"。

    窗口边界不能因为一条坏时间戳而整个失配，但也不许假装按原意算过——
    失配时调用方仍然能从 `undated` 计数里知道有东西坏了。
    """
    try:
        base = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        base = datetime.now(UTC)
    return (base + timedelta(seconds=seconds)).isoformat()


def build_channels(cfg: Any, data_dir: Path) -> list[AlertChannel]:
    """按配置装配出口。**没配的渠道不装**，而不是装一个"什么都不发"的渠道——
    后者会让 `alerts ls` 显示"有出口"，而它从来没成功投递过任何东西。

    文件出口先装、webhook 后装：本地那条是"一定能落"的兜底，这个顺序保证
    webhook 挂了也不影响本地留下那一行。

    装配放在这里而不是 `onyx/config.py`：config 被 `settings` 引进 tools 层，
    import-linter 会顺着链把 `webhook -> httpx` 也算成"tools 碰了网络"。
    这条链的报告正是那道门禁值钱的地方。
    """
    from onyx.config import alert_channels_enabled, alert_file_path, alert_webhook_url
    from onyx.obs.alerts.channels import FileChannel

    if not alert_channels_enabled(cfg):
        return []
    out: list[AlertChannel] = [FileChannel(alert_file_path(cfg, data_dir))]
    url = alert_webhook_url(cfg)
    if url:
        # 惰性导入：没配 URL 时连模块都不 import，`onyx.obs` 整体仍在零三方依赖下可导入
        from onyx.obs.alerts.webhook import WebhookChannel

        out.append(WebhookChannel(url))
    return out
