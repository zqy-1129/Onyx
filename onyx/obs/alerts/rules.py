"""告警规则的判定：**纯函数，不碰库也不发任何东西**。

判据全部来自库里已有的 anomaly 行，而不是进程内的计数——服务重启不该让同一条异常
立刻再通知一遍，而"刚才那个窗口里出现过几条"这件事只有库能回答。
把判定做成纯函数是为了让窗口/阈值/cooldown 三条规则能在离线单测里穷举，
不需要真的等 5 秒或造一个假时钟。

默认值的家在这里（`onyx/config.py` 的 `effective()` 反向导入它），
和 gpu_lock / retention / sandbox 同一个规矩：默认值不抄两份。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from onyx.obs.anomalies import Severity

#: 只盯 error 级：本地这块卡上 warn 级（低置信计数、缓存命中）是**常态**，
#: 把它们推给人会训练出"忽略通知"的习惯，而那等于没有通知。
DEFAULT_ALERT_SEVERITIES: tuple[str, ...] = (Severity.ERROR,)
#: 窗口 300s 内出现 1 次 error 就通知：本地评测一旦出事，最需要的是立刻停手，
#: 而不是等它连续坏五次（每次 60s 的 GPU 时间）。
DEFAULT_ALERT_WINDOW_S = 300
DEFAULT_ALERT_MIN_COUNT = 1
#: 同一条规则命中后 10 分钟内不再重复。冷却期存在的理由是"一件事说一遍"，
#: 不是省流量：一条一直坏的异常每小时会推 6 次同样的话。
DEFAULT_ALERT_COOLDOWN_S = 600
#: 轮询周期。DoD 是"1 分钟内收到通知"，5 秒有余量，也让一次评测里的连续异常能进同一个窗口。
DEFAULT_ALERT_POLL_S = 5.0
#: 一次窗口扫描最多读多少条异常。命中上限时计数只能报成下限（`Hit.truncated`），
#: 把下限当精确数报出去，等于在一个诊断系统里编造精确。
DEFAULT_ALERT_SCAN_LIMIT = 2000

#: 触发历史里留几个样本 trace_id：够下钻就行，不是全量清单
SAMPLE_TRACE_IDS = 3


@dataclass(frozen=True, slots=True)
class AlertRule:
    """一条生效的筛选与阈值。`source` 是它的出处（flag/env/file/default），会一起落库。"""

    severities: tuple[str, ...] = DEFAULT_ALERT_SEVERITIES
    codes: tuple[str, ...] = ()
    exclude_codes: tuple[str, ...] = ()
    window_s: int = DEFAULT_ALERT_WINDOW_S
    min_count: int = DEFAULT_ALERT_MIN_COUNT
    cooldown_s: int = DEFAULT_ALERT_COOLDOWN_S
    poll_s: float = DEFAULT_ALERT_POLL_S
    enabled: bool = True
    source: str = "default"

    def __post_init__(self) -> None:
        # 规则本身不许是"配了但永远命中不了"的形状：那种配置比没配更危险
        if self.window_s <= 0:
            raise ValueError("window_s 必须大于 0")
        if self.min_count < 1:
            raise ValueError("min_count 至少是 1（0 表示'没异常也通知'，那不是告警）")
        if self.cooldown_s < 0:
            raise ValueError("cooldown_s 不能是负数")

    def matches(self, code: str, severity: str) -> bool:
        """码白名单与级别是**两条独立的闸**，都过才算盯这条。

        单独看 codes 会漏掉"盯 CONTEXT_OVERFLOW 但只要 error 级"这种收窄；
        单独看 severities 又没法把噪音码请出去。
        """
        if self.codes and code not in self.codes:
            return False
        if code in self.exclude_codes:
            return False
        return severity in self.severities

    def as_dict(self) -> dict[str, Any]:
        """落库快照。事后解释"当时为什么触发"用的是这一份，不是现在的配置。"""
        return {
            "severities": list(self.severities),
            "codes": list(self.codes),
            "exclude_codes": list(self.exclude_codes),
            "window_s": self.window_s,
            "min_count": self.min_count,
            "cooldown_s": self.cooldown_s,
            "poll_s": self.poll_s,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Hit:
    """一次命中。`suppressed` 非空表示"本该通知但在 cooldown 里"，调用方不该重复投递。"""

    code: str
    severity: str
    n: int
    window_s: int
    first_id: str
    last_id: str
    trace_ids: tuple[str, ...] = ()
    #: 扫描被 limit 截断 ⇒ n 是**下限**。这句话必须跟着走，否则一个"≥12"会被读成"正好 12"
    truncated: bool = False
    suppressed: str = ""

    def count_text(self) -> str:
        return f"{self.n}{'+' if self.truncated else ''}"

    def summary(self) -> str:
        head = f"{self.code} 在 {self.window_s}s 窗口里出现 {self.count_text()} 次"
        return f"{head}（{self.suppressed}）" if self.suppressed else head


@dataclass(frozen=True, slots=True)
class Row:
    """判定要的最小行形状。用它而不是 AnomalyRecord，是为了让单测能就地造数据。"""

    id: str
    code: str
    severity: str
    created_at: str
    trace_id: str | None = None


def _age_seconds(created_at: str, now: str) -> float | None:
    """这条异常距 now 多少秒；解析不出来返回 None（调用方必须报告，不许静默丢）。"""
    try:
        born = datetime.fromisoformat(created_at)
        here = datetime.fromisoformat(now)
    except (TypeError, ValueError):
        return None
    return (here - born).total_seconds()


@dataclass(frozen=True, slots=True)
class EvalResult:
    hits: tuple[Hit, ...] = ()
    #: 时间戳解析不出来、因此**没有**进入窗口计数的行数。
    #: 它必须被报告而不是被丢掉——"库里有一批异常的时间是坏的"这件事本身值得知道。
    undated: int = 0


def evaluate(
    rows: list[Row],
    *,
    rule: AlertRule,
    now: str,
    last_triggered_at: dict[str, str],
    truncated: bool = False,
) -> EvalResult:
    """按 code 分组，判窗口、阈值与 cooldown。

    `last_triggered_at` 是"每条 code 上次真的投递的时间"（来自库，不是进程）。
    cooldown 命中时仍返回一个带 `suppressed` 的 Hit —— 让调用方知道"这件事被规则挡了"，
    而不是让它凭空消失。
    """
    if not rule.enabled:
        return EvalResult()

    groups: dict[str, list[Row]] = {}
    undated = 0
    for row in rows:
        if not rule.matches(row.code, row.severity):
            continue
        age = _age_seconds(row.created_at, now)
        if age is None:
            undated += 1
            continue
        if age < 0 or age > rule.window_s:
            continue  # 未来的或出窗的
        groups.setdefault(row.code, []).append(row)

    hits: list[Hit] = []
    for code, group in groups.items():
        n = len(group)
        if n < rule.min_count:
            continue
        severity = group[-1].severity
        prior = last_triggered_at.get(code)
        suppressed = ""
        if prior:
            since = _age_seconds(prior, now)
            if since is not None and since < rule.cooldown_s:
                suppressed = f"cooldown：距上次投递 {int(since)}s < {rule.cooldown_s}s"
        hits.append(Hit(
            code=code, severity=severity, n=n, window_s=rule.window_s,
            first_id=group[0].id, last_id=group[-1].id,
            trace_ids=tuple(
                r.trace_id for r in group[-SAMPLE_TRACE_IDS:] if r.trace_id
            ),
            truncated=truncated, suppressed=suppressed,
        ))
    return EvalResult(hits=tuple(hits), undated=undated)
