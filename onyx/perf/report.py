"""渲染与落库形状：`onyx perf` 的人类输出、`--json` 与库里三处同源（S36）。

放在一处而不是各写一遍，理由和契约矩阵一样：一旦 CLI 与库里的措辞分开算，
"终端说这一格没测到、看板显示 0"这种分歧就没法排查——而这条命令的全部价值
恰恰是"没测到"和"测出来很慢"能被区分开。

所有格式化都遵守同一条：**`None` 是「—」，不是 0，也不是空串**。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from onyx.perf.bench import BenchOutcome

#: 表格里按这个顺序展示；不在名单里的键仍然进 `--json`，只是不占一行
METRIC_COLUMNS: tuple[str, ...] = (
    "prompt_tokens", "out_tokens", "ttft_ms", "wall_ms", "decode_tps",
    "prefill_tps_warm", "prefill_tps_cold", "aggregate_tps", "load_ms",
)
DASH = "—"


def fmt(value: Any, *, unit: str = "") -> str:
    """一个数的显示形状。None / 缺 ⇒ 「—」。"""
    if value is None or value == "":
        return DASH
    if isinstance(value, float):
        text = f"{value:,.1f}" if abs(value) >= 10 else f"{value:.3g}"
    else:
        text = f"{value:,}" if isinstance(value, int) else str(value)
    return f"{text}{unit}" if unit else text


def fmt_spread(spread: Mapping[str, Any] | None) -> str:
    """落点摘要 → `中位/p95 (n=3)`；没有值就是「—」，附带说明时把说明带出来。"""
    if not spread:
        return DASH
    if "median" not in spread:
        return str(spread.get("note") or DASH)
    core = f"{fmt(spread.get('median'))} / {fmt(spread.get('p95'))} (n={spread.get('n')})"
    note = spread.get("note")
    return f"{core}；{note}" if note else core


def status_symbol(status: str) -> str:
    return {"measured": "✓", "skipped": "没测到", "error": "失败"}.get(status, status)


def cell_line(key: str, status: str, reason: str, metrics: Mapping[str, Any] | None) -> str:
    """一格的单行摘要。真机跑完与从库里读回**共用这一个函数**——
    两份写法迟早会分叉，而分叉的表现是"终端说没测到、库里看着有数"。"""
    head = f"{key:<26} {status_symbol(status):<7}"
    if status != "measured":
        return f"{head} {reason}"
    metrics = metrics or {}
    counts = (f"发 {metrics.get('n_requests', 0)}"
              f"· 计 {metrics.get('n_measured', 0)}"
              f"· 败 {metrics.get('n_error', 0)}"
              f"· 裁 {metrics.get('n_truncated', 0)}")
    decode = fmt_spread(metrics.get("decode_tps"))
    aggregate = fmt_spread(metrics.get("aggregate_tps"))
    ttft = fmt_spread(metrics.get("ttft_ms"))
    return f"{head}{counts} | decode {decode} | 合计 {aggregate} | TTFT {ttft}"


def condition_lines(conditions: Mapping[str, Any]) -> list[str]:
    """把"这次在什么条件下测"打成可读的几行，顺序固定，方便眼睛比对。"""
    return [
        f"引擎：{conditions.get('provider_id') or DASH} "
        f"版本 {conditions.get('engine_version') or DASH}"
        f"｜模型 {conditions.get('model') or DASH}"
        f"（量化 {conditions.get('quantization') or DASH}）",
        f"条件：设备 {conditions.get('device') or DASH}｜num_ctx "
        f"{fmt(conditions.get('num_ctx'))}｜keep_alive {conditions.get('keep_alive')}"
        f"｜流式 {'是' if conditions.get('stream') else '否'}"
        f"｜温度 {fmt(conditions.get('temperature'))}｜seed {fmt(conditions.get('seed'))}",
        f"时序出处：{conditions.get('timing_source')}"
        + ("" if conditions.get("timing_source") == "engine_ns"
                   else "（引擎没给纳秒分段 ⇒ 吞吐列会是「—」而不是 0）"),
        f"代码：{conditions.get('app_version') or DASH} @ {conditions.get('git_rev') or DASH}"
        "（刻意不进指纹：换代码正是基线要对比的东西）",
    ]


def run_lines(outcome: BenchOutcome) -> list[str]:
    head = (f"状态 {outcome.status}｜{outcome.n_requests} 发｜用时 {outcome.elapsed_s:.1f}s"
            f"｜指纹 {outcome.env_hash[:12]}｜可比 "
            f"{'是' if outcome.comparable else '否（认不出引擎身份）'}")
    return [head, *condition_lines(outcome.conditions), "",
            *[cell_line(item.cell.key, item.status, item.reason, item.metrics)
              for item in outcome.cells]]


def record_lines(run: Any, cells: Sequence[Any]) -> list[str]:
    """从库里读回一条基线时的渲染。与 `run_lines` 共用 `condition_lines` 与 `cell_line`。"""
    head = (f"状态 {run.status}｜{run.n_requests} 发｜用时 {run.elapsed_s:.1f}s"
            f"｜指纹 {run.env_hash[:12]}｜可比 "
            f"{'是' if run.comparable else '否（认不出引擎身份）'}")
    tail = [f"备注：{run.note}"] if run.note else []
    return [head, *condition_lines(run.conditions), *([f"错误：{run.error}"] if run.error else []),
            "", *tail,
            *[cell_line(item.cell_key, item.status, item.reason, item.metrics)
              for item in cells]]


def unmeasured_lines(outcome: BenchOutcome) -> list[str]:
    """"还欠哪几格"。半截的基线必须能被看出欠多少。"""
    return [f"{item.cell.key}：{item.reason or item.status}" for item in outcome.unmeasured]


# ── 落库形状 ──────────────────────────────────────────────────────
def outcome_to_rows(
    outcome: BenchOutcome, *, run_id: str, started_at: str, finished_at: str, note: str = ""
) -> tuple[Any, list[Any]]:
    """`BenchOutcome` → `(PerfRunRecord, [PerfCellRecord])`。

    数字**抄进表里**而不是只留 trace 指针：trace 会被保留策略摘走，
    而一条基线的意义正是"半年后还能拿来比"。指针留着，只用于下钻。
    """
    from onyx.store.records import PerfCellRecord, PerfRunRecord

    conditions = outcome.conditions
    run = PerfRunRecord(
        id=run_id, started_at=started_at, finished_at=finished_at, status=outcome.status,
        provider_id=str(conditions.get("provider_id") or ""),
        engine_version=str(conditions.get("engine_version") or ""),
        model=str(conditions.get("model") or ""),
        quantization=str(conditions.get("quantization") or ""),
        device=str(conditions.get("device") or ""),
        num_ctx=conditions.get("num_ctx"),
        keep_alive=str(conditions.get("keep_alive") or ""),
        stream=bool(conditions.get("stream")),
        timing_source=str(conditions.get("timing_source") or "unknown"),
        env_hash=outcome.env_hash, comparable=outcome.comparable,
        conditions=conditions, grid=outcome.plan.as_dict(),
        elapsed_s=round(outcome.elapsed_s, 3), n_requests=outcome.n_requests,
        app_version=str(conditions.get("app_version") or ""),
        git_rev=str(conditions.get("git_rev") or ""), note=note,
    )
    cells = [
        PerfCellRecord(
            run_id=run_id, cell_key=item.cell.key, phase=item.cell.phase,
            prompt_chars=item.cell.prompt_chars, target_tokens=item.cell.target_tokens,
            concurrency=item.cell.concurrency, repeat=item.cell.repeat,
            status=item.status, reason=item.reason,
            n_requests=int(item.metrics.get("n_requests", len(item.samples))),
            n_measured=int(item.metrics.get("n_measured", 0)),
            metrics=item.metrics,
            trace_ids=tuple(s.trace_id for s in item.samples if s.trace_id),
        )
        for item in outcome.cells
    ]
    return run, cells


def compare_lines(
    left: Mapping[str, Any], right: Mapping[str, Any], diffs: Sequence[Mapping[str, Any]]
) -> list[str]:
    """两次运行的条件差异。逐字段列，而不是留一句"条件不同"让人猜。"""
    out = [f"甲 {left.get('id')}  乙 {right.get('id')}"]
    for item in diffs:
        out.append(f"  · {item['field']}：甲={item['a'] if item['a'] not in (None, '') else DASH}"
                   f" / 乙={item['b'] if item['b'] not in (None, '') else DASH}")
    return out
