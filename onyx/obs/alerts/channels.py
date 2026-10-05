"""告警出口：协议 + 本地文件出口。

渠道**不许把异常抛给轮询线程**：一个出口坏了不能让别的出口也不投，
也不能让 serve 的后台线程死掉——那时候"没有通知"和"通知系统自己崩了"就再也分不开了。
失败以 `ChannelResult(ok=False, detail=原因)` 返回，由调用方落成 `status=failed` 的行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from onyx.core.clock import utc_now_iso


@dataclass(frozen=True, slots=True)
class Alert:
    """一条要送出去的消息。字段是**结论 + 定位信息**，不含 trace 正文。

    正文里可能有用户输入，而 webhook 是往机器外面送的东西；
    定位信息（anomaly id / trace id）足够让人在看板上点开看现场。
    """

    code: str
    severity: str
    message: str
    n: int
    window_s: int
    triggered_at: str
    rule: dict[str, Any]
    first_anomaly_id: str = ""
    last_anomaly_id: str = ""
    trace_ids: tuple[str, ...] = ()
    truncated: bool = False
    is_test: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "n": self.n,
            "n_is_lower_bound": self.truncated,
            "window_s": self.window_s,
            "triggered_at": self.triggered_at,
            "rule": self.rule,
            "first_anomaly_id": self.first_anomaly_id,
            "last_anomaly_id": self.last_anomaly_id,
            "trace_ids": list(self.trace_ids),
            "is_test": self.is_test,
        }


@dataclass(frozen=True, slots=True)
class ChannelResult:
    ok: bool
    detail: str = ""


class AlertChannel(Protocol):
    name: str

    def deliver(self, alert: Alert) -> ChannelResult:
        """投递一条消息。返回结果，不抛异常。"""
        ...


def default_alert_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / "alerts" / "alerts.jsonl"


class FileChannel:
    """往本地文件追加一行 JSONL。零依赖、零外部可用性要求，所以它是"一定能落"的那条出口。

    一天一个文件不是洁癖：通知是拿来回看"上周三那天到底出了什么"的，
    单文件会一直长，而按天分片让人可以直接删旧的。
    """

    name = "file"

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _target(self, when: str) -> Path:
        try:
            day = datetime.fromisoformat(when).strftime("%Y%m%d")
        except (TypeError, ValueError):
            day = utc_now_iso()[:10].replace("-", "")
        return self.path.parent / f"{self.path.stem}-{day}{self.path.suffix}"

    def deliver(self, alert: Alert) -> ChannelResult:
        target = self._target(alert.triggered_at)
        line = json.dumps(alert.as_dict(), ensure_ascii=False, separators=(",", ":"))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            return ChannelResult(False, f"{type(exc).__name__}: {exc}"[:200])
        return ChannelResult(True, f"写入 {target}")
