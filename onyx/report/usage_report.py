"""跨时间的 token 用量汇总：`onyx report usage` 与 `/api/usage/summary` 的共同实现（S38）。

为什么要抽这一份：看板的聚合里原本手写着两件小事——
`ordered[len(ordered)//2]` 当作 p50、`v > 0.10` 当作漂移阈值。
而 p50 的定义在 `llm/measurement/stats.py` 只有一份，阈值在 `reconciler` 只有一个常量
（`DEFAULT_DRIFT_THRESHOLD`，异常判据用的就是它）。CLI 再加一份的话，
同一份库会在看板和终端给出两个"漂移超阈率"——那是本项目最不该出现的双事实。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from onyx.llm.measurement.reconciler import DEFAULT_DRIFT_MIN_TOKENS, DEFAULT_DRIFT_THRESHOLD
from onyx.llm.measurement.stats import median

#: CSV / markdown 的列序。**测试把它钉住**：列序漂了下游脚本会静默读错列，
#: 而 diff 里只看得出"换了个顺序"，看不出读错。
CSV_COLUMNS: tuple[str, ...] = (
    "bucket", "traces", "in_tokens", "out_tokens", "decode_tps",
    "cold_prefill_tps", "warm_prefill_tps",
)


@dataclass(frozen=True, slots=True)
class UsageOverview:
    since: str | None
    model: str | None
    bucket_minutes: int
    traces: int
    in_tokens: int
    out_tokens: int
    thinking_tokens: int
    by_source: dict[str, int] = field(default_factory=dict)
    by_confidence: dict[str, int] = field(default_factory=dict)
    by_prefill_mode: dict[str, int] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    timeseries: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "since": self.since, "model": self.model, "bucket_minutes": self.bucket_minutes,
            "traces": self.traces, "in_tokens": self.in_tokens, "out_tokens": self.out_tokens,
            "thinking_tokens": self.thinking_tokens,
            "by_source": dict(self.by_source), "by_confidence": dict(self.by_confidence),
            "by_prefill_mode": dict(self.by_prefill_mode),
            "drift": dict(self.drift), "timeseries": list(self.timeseries),
        }


_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def normalize_since(raw: str | None) -> str | None:
    """把调用方给的 `--since` 折成库里那种定长 UTC 串；不认的写法**报错而不是查空**。

    库里的 `started_at` 全是 `2026-10-03T03:26:51.211036+00:00` 这一个形状，过滤靠字典序
    比较。所以 `--since 7d` 不会炸：`'7' > '2'`，它比所有行都大，于是静静地返回 0 条，
    再被读成"这几天没有请求"。带时区的写法同理——`+08:00` 的串和 `+00:00` 的串比大小
    不是同一回事，必须先折算成 UTC 再比。
    """
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    if _DATE_ONLY.match(text):
        try:
            day = date.fromisoformat(text)      # 正则只认形状，`2026-13-45` 得靠它拦
        except ValueError:
            raise _since_error(raw) from None
        return f"{day.isoformat()}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise _since_error(raw) from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


def _since_error(raw: str) -> ValueError:
    today = datetime.now(UTC).date()
    return ValueError(
        f"--since 只认 `YYYY-MM-DD` 或 ISO 时间（naive 按 UTC 理解），收到 {raw!r}。"
        f"今天是 {today.isoformat()}，要最近 7 天就写 --since {(today - timedelta(days=7)).isoformat()}。"
    )


def build_overview(
    usage_repo: Any, *, since: str | None = None, model: str | None = None,
    bucket_minutes: int = 60,
) -> UsageOverview:
    """从 `UsageRepo` 出一份汇总。CLI 与 API 都走这里，不再各算一遍。

    升 `ValueError`：`since` 不是可解析的时间（宁可让两个入口都响，也不要静默出一份空表）。
    """
    since = normalize_since(since)
    summary = usage_repo.summarize(since=since, model_id=model)
    values = sorted(value for _, value in summary.drift_samples if value is not None)
    drift = {
        "n": len(values),
        "max": values[-1] if values else None,
        "p50": median(values) if values else None,
        "over_threshold": sum(1 for value in values if value > DEFAULT_DRIFT_THRESHOLD),
        "threshold": DEFAULT_DRIFT_THRESHOLD,
        # 口径必须自己说清：`TOKEN_DRIFT` 异常用的是 `pct > 阈值 且绝对差 ≥ min_tokens`，
        # 这里的"超阈条数"只有前半条。真机上 2298 个样本里相对差超阈 721 条、异常 711 条——
        # **差 10 条不代表它们是同一个量**：短 prompt 上两个门槛会分得很开（如 in=120、pct=0.15
        # 的那几条，绝对差只有 16~18 tok，异常不报而这里照计）。把它们当同一个数才是 bug。
        "rule": "只按相对差判（pct > threshold），不含绝对差门槛 ⇒ 比 TOKEN_DRIFT 异常数宽",
        "min_tokens_for_anomaly": DEFAULT_DRIFT_MIN_TOKENS,
    }
    return UsageOverview(
        since=since, model=model, bucket_minutes=bucket_minutes,
        traces=summary.traces, in_tokens=summary.in_tokens, out_tokens=summary.out_tokens,
        thinking_tokens=summary.thinking_tokens,
        by_source=dict(summary.by_source), by_confidence=dict(summary.by_confidence),
        by_prefill_mode=dict(usage_repo.by_prefill_mode(since=since)),
        drift=drift,
        timeseries=list(usage_repo.timeseries(bucket_minutes=bucket_minutes, since=since)),
    )


# ── 渲染 ──────────────────────────────────────────────────────────
#: 这三列是"速率"。S39 之前 `UsageRepo.timeseries` 用 `COALESCE(AVG(...),0)` 聚合，于是
#: **"这一格没有延迟数据"与"测到了 0 t/s"在数据里长得一模一样**；现在 SQL 直接出 NULL，
#: 而 0 仍按"没测到"处理——decode 为 0 意味着没吐出字，那不是一个可画的速率。
#: 反过来 token 数的 0 是**真 0**（空输出），所以这条规则只作用于这几列：
#: 把已知的 0 渲染成「—」是把已知说成未知，与把未知说成 0 一样坏。
RATE_COLUMNS = frozenset({"decode_tps", "cold_prefill_tps", "warm_prefill_tps"})


def _missing(name: str, value: Any) -> bool:
    return value is None or (name in RATE_COLUMNS and not value)


def render_csv(overview: UsageOverview) -> str:
    rows = [",".join(CSV_COLUMNS)]
    for bucket in overview.timeseries:
        rows.append(",".join(_csv_cell(name, bucket.get(name)) for name in CSV_COLUMNS))
    return "\n".join(rows) + "\n"


def _csv_cell(name: str, value: Any) -> str:
    if _missing(name, value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    # 逗号或引号出现时必须按 CSV 转义，否则一列会变两列——静默错位比报错更坏
    return f'"{text}"' if any(ch in text for ch in (',', '"', "\n")) else text


def render_markdown(overview: UsageOverview) -> str:
    lines = [
        f"# token 用量汇总（{'全部时间' if not overview.since else f'自 {overview.since}'}"
        + (f" · 模型 {overview.model}" if overview.model else "") + "）",
        "",
        f"- trace 数：**{overview.traces}**｜in **{overview.in_tokens:,}** tok｜"
        f"out **{overview.out_tokens:,}** tok｜thinking {overview.thinking_tokens:,} tok",
        f"- 采信出处：{_kv(overview.by_source)}",
        f"- 置信度：{_kv(overview.by_confidence)}",
        f"- prefill 冷热：{_kv(overview.by_prefill_mode)}"
        "（两者混算会得到一个既不代表冷启动也不代表稳态的数，见 PROBES P11）",
        _drift_line(overview.drift),
        "",
        "| " + " | ".join(CSV_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(CSV_COLUMNS)) + "|",
    ]
    for bucket in overview.timeseries:
        lines.append("| " + " | ".join(_md_cell(name, bucket.get(name)) for name in CSV_COLUMNS) + " |")
    if not overview.timeseries:
        lines.append("| （空） | | | | | | |")   # 空表也要打出来：没有 ≠  zero
    return "\n".join(lines) + "\n"


def render_table(overview: UsageOverview) -> list[str]:
    """纯文本表（无 rich 依赖，测试与管道都用它）。"""
    head = (f"状态：traces={overview.traces} in={overview.in_tokens:,} "
            f"out={overview.out_tokens:,} thinking={overview.thinking_tokens:,}")
    lines = [head, f"出处：{_kv(overview.by_source)}",
             f"置信度：{_kv(overview.by_confidence)}",
             f"冷热：{_kv(overview.by_prefill_mode)}", _drift_line(overview.drift),
             f"{'bucket':<18} {'traces':>6} {'in':>10} {'out':>10} {'decode':>8} "
             f"{'cold_pf':>8} {'warm_pf':>8}"]
    for bucket in overview.timeseries:
        lines.append(f"{str(bucket.get('bucket', ''))[:18]:<18} {bucket.get('traces', 0):>6} "
                     f"{bucket.get('in_tokens', 0):>10,} {bucket.get('out_tokens', 0):>10,} "
                     f"{_num(bucket.get('decode_tps')):>8} {_num(bucket.get('cold_prefill_tps')):>8} "
                     f"{_num(bucket.get('warm_prefill_tps')):>8}")
    if not overview.timeseries:
        lines.append("（这个范围里没有 trace —— 不是 0 花费，是没有任何请求）")
    return lines


def _kv(mapping: dict[str, int]) -> str:
    return "、".join(f"{key} {value}" for key, value in sorted(mapping.items())) or "—"


def _drift_line(drift: dict[str, Any]) -> str:
    if not drift.get("n"):
        return "- 对账漂移：没得比（只有一路计数或没有多路样本）"
    return (f"- 对账漂移：n={drift['n']} p50={drift['p50']:.2%} max={drift['max']:.2%} "
            f"pct 超阈（>{drift['threshold']:.0%}）{drift['over_threshold']} 条"
            f"｜{drift.get('rule', '')}")


def _num(value: Any) -> str:
    """速率列的「—」规则（纯文本表里只用于那三列，与 `_missing` 同源）。"""
    return "—" if not value else f"{float(value):.1f}"


def _md_cell(name: str, value: Any) -> str:
    if _missing(name, value):
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
