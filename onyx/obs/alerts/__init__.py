"""告警（M10 / G4）：把"库里已经记下来的异常"变成"人会知道"。

三个模块各司其职，便于各自被单测钉住：
- `rules` 纯判定（窗口 / 阈值 / cooldown），不碰库也不发消息
- `channels` 出口协议与实现，失败以结果返回而不抛异常
- `service` 轮询与留痕：判定 → 逐渠道投递 → 每个 (命中, 渠道) 落一行
"""

from onyx.obs.alerts.channels import (
    Alert,
    AlertChannel,
    ChannelResult,
    FileChannel,
    default_alert_path,
)
from onyx.obs.alerts.rules import (
    DEFAULT_ALERT_COOLDOWN_S,
    DEFAULT_ALERT_MIN_COUNT,
    DEFAULT_ALERT_POLL_S,
    DEFAULT_ALERT_SCAN_LIMIT,
    DEFAULT_ALERT_SEVERITIES,
    DEFAULT_ALERT_WINDOW_S,
    AlertRule,
    EvalResult,
    Hit,
    Row,
    evaluate,
)
from onyx.obs.alerts.service import AlertService, build_channels, message_for

__all__ = [
    "DEFAULT_ALERT_COOLDOWN_S",
    "DEFAULT_ALERT_MIN_COUNT",
    "DEFAULT_ALERT_POLL_S",
    "DEFAULT_ALERT_SCAN_LIMIT",
    "DEFAULT_ALERT_SEVERITIES",
    "DEFAULT_ALERT_WINDOW_S",
    "Alert",
    "AlertChannel",
    "AlertRule",
    "AlertService",
    "ChannelResult",
    "EvalResult",
    "FileChannel",
    "Hit",
    "Row",
    "build_channels",
    "default_alert_path",
    "evaluate",
    "message_for",
]
