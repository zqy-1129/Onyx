"""报告层（L6）：把 run 记录整理成可脱离看板阅读的产物。

只读，不发起任何请求。矩阵/对比的所有数字都直接来自 `eval_run.aggregate_json`
与 `grade` 表，**这里绝不重算任何统计量**——两处各算一套必然漂移，
而漂移出来的差别看起来像"模型变了"。

导出格式：
- `csv` 给表格软件与回归脚本；
- `md` 给 PR 描述、issue 与聊天窗口；
- `html` 是自包含单文件（内联样式与 SVG），可以直接发给别人，不依赖服务在跑。
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from onyx.eval.task import headline_of
from onyx.store.records import RunRecord

#: 低于这个任务数，雷达图会退化成一条线——那种图比表格更容易误导人
MIN_AXES_FOR_RADAR = 3
#: 主分数覆盖的样本比例低于它，这一格就必须在矩阵里点名。
#: 触发场景是真的：模型输出不守格式时，`macro_f1` 只在"格式合法的那些题"上算，
#: 于是 8/236 个样本能算出 1.000——那是真话，但读起来像"这个模型意图识别很强"
THIN_COVERAGE = 0.5


@dataclass(frozen=True, slots=True)
class Cell:
    """矩阵里的一格：某个模型在某个任务上的一次运行。"""

    run_id: str
    model_id: str
    task_id: str
    metric: str
    value: float | None
    ci: dict[str, Any] | None
    n: int | None
    low_confidence: bool
    #: 主分数**真正统计到**的样本数（任务自己报；intent 用 `n_judged`）。
    #: 与 `n_total` 一起看才有意义：格式不合规会把样本挤出可判定集合，
    #: 于是 `macro_f1` 可以在 8/236 个样本上算出 1.000——那是真话，但不是这场考试的成绩
    n_judged: int | None
    n_total: int | None
    status: str
    started_at: str
    dataset_id: str
    dataset_revision: str
    #: 这次运行自己花的钱，报告里要能看见"这个分数值多少 GPU 时间"
    cost: dict[str, Any]

    @property
    def coverage(self) -> float | None:
        """主分数覆盖的样本比例。None = 任务没报可判定数，无从判断。"""
        if not self.n_judged or not self.n_total:
            return None
        return self.n_judged / self.n_total


@dataclass(frozen=True, slots=True)
class Matrix:
    models: tuple[str, ...]
    tasks: tuple[str, ...]
    cells: tuple[Cell, ...]
    #: 出现过的数据集来历。多于一个 ⇒ 这些格子不是同一份考卷
    provenance: tuple[str, ...]
    warnings: tuple[str, ...]

    def cell(self, model_id: str, task_id: str) -> Cell | None:
        return next((c for c in self.cells
                     if c.model_id == model_id and c.task_id == task_id), None)

    @property
    def thin_cells(self) -> tuple[Cell, ...]:
        """主分数只覆盖到少数样本的格子。

        这类格子必须点名：`macro_f1 1.000` 建立在 8/236 个可判定样本上时，
        它是"格式合法的那些题全对"，不是"这个模型意图识别很强"。
        """
        return tuple(cell for cell in self.cells
                     if cell.coverage is not None and cell.coverage < THIN_COVERAGE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "models": list(self.models), "tasks": list(self.tasks),
            "cells": [_cell_dict(c) for c in self.cells],
            "provenance": list(self.provenance), "warnings": list(self.warnings),
        }


def build_matrix(runs: Sequence[RunRecord]) -> Matrix:
    """每个 (模型, 任务) 取**最新一次**运行。

    取"最新"而不是"最好"：分数被挑过以后就不再是测量，而是宣传。
    未完成（running）的 run 不参与矩阵——半截运行的分数没有可比性，
    把它放进网格里会让那一列看起来和别的列一样可信。
    """
    latest: dict[tuple[str, str], RunRecord] = {}
    for run in runs:
        if run.status != "done":
            continue
        key = (run.model_id, run.task_id)
        current = latest.get(key)
        if current is None or (run.started_at or "") > (current.started_at or ""):
            latest[key] = run

    cells = tuple(
        _cell(run) for run in sorted(latest.values(), key=lambda r: (r.model_id, r.task_id))
    )
    models = tuple(sorted({run.model_id for run in latest.values()}))
    tasks = tuple(sorted({run.task_id for run in latest.values()}))
    provenance = tuple(sorted({
        f"{run.dataset_id or '未知'}@{run.dataset_revision or '未知'}"
        for run in latest.values()
    }))

    warnings: list[str] = []
    if len(provenance) > 1:
        warnings.append(
            f"矩阵里混了 {len(provenance)} 份数据（{', '.join(provenance)}）："
            "跨列比较没有意义，同列内比较也要先看版本是否一致"
        )
    missing = [(c.model_id, c.task_id) for c in cells if c.value is None]
    if missing:
        warnings.append(
            f"{len(missing)} 个格子没有可判定样本（显示为「—」），它们不是 0 分"
        )
    thin = [c for c in cells if (c.coverage or 1.0) < THIN_COVERAGE]
    if thin:
        examples = "、".join(f"`{c.model_id}@{c.task_id}` {c.n_judged}/{c.n_total}"
                             for c in thin[:3])
        warnings.append(
            f"{len(thin)} 格的主分数只覆盖不到一半样本（{examples}）："
            "那一格读的是**可判定子集**，不是整场考试——先对照同一次运行的 format_valid_rate"
        )
    return Matrix(models=models, tasks=tasks, cells=cells,
                  provenance=provenance, warnings=tuple(warnings))


def _cell(run: RunRecord) -> Cell:
    headline = headline_of(run.aggregate)
    metric, value = headline if headline else ("—", None)
    ci = run.aggregate.get(f"{metric}_ci") if metric else None
    aggregate = run.aggregate
    return Cell(
        run_id=run.id, model_id=run.model_id, task_id=run.task_id, metric=str(metric),
        value=value if isinstance(value, (int, float)) or value is None else None,
        ci=ci if isinstance(ci, dict) else None,
        n=(ci.get("n") if isinstance(ci, dict) else None) or run.n_cases,
        low_confidence=bool(aggregate.get("low_confidence")),
        #: `n_judged` 是任务的口径（intent：格式合法且标签在集合内才算）；
        #: 没有这个键的任务（tool_selection 的全集都是可判定的）就留 None，
        #: 宁可"不知道"也不编一个比例出来
        n_judged=aggregate.get("n_judged"),
        n_total=aggregate.get("n_total") or run.n_cases,
        status=run.status, started_at=run.started_at,
        dataset_id=run.dataset_id or "", dataset_revision=run.dataset_revision,
        cost=run.cost,
    )


def _cell_dict(cell: Cell) -> dict[str, Any]:
    return {
        "run_id": cell.run_id, "model_id": cell.model_id, "task_id": cell.task_id,
        "metric": cell.metric, "value": cell.value, "ci": cell.ci, "n": cell.n,
        "n_judged": cell.n_judged, "n_total": cell.n_total, "coverage": cell.coverage,
        "low_confidence": cell.low_confidence, "status": cell.status,
        "started_at": cell.started_at, "dataset_id": cell.dataset_id,
        "dataset_revision": cell.dataset_revision, "cost": cell.cost,
    }


# ── 渲染 ──────────────────────────────────────────────────────────
def render_csv(matrix: Matrix) -> str:
    """一行一格。CI 拆成两列，方便表格软件直接画误差棒。"""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["model", "task", "metric", "value", "ci_low", "ci_high", "n",
                     "n_judged", "n_total", "coverage", "low_confidence", "status",
                     "dataset", "revision", "in_tokens", "out_tokens", "requests",
                     "wall_ms", "run_id", "started_at"])
    for cell in matrix.cells:
        writer.writerow([
            cell.model_id, cell.task_id, cell.metric,
            "" if cell.value is None else f"{cell.value:.6f}",
            "" if not cell.ci or cell.ci.get("low") is None else f"{cell.ci['low']:.6f}",
            "" if not cell.ci or cell.ci.get("high") is None else f"{cell.ci['high']:.6f}",
            cell.n, cell.n_judged, cell.n_total,
            "" if cell.coverage is None else f"{cell.coverage:.4f}",
            int(cell.low_confidence), cell.status,
            cell.dataset_id, cell.dataset_revision,
            cell.cost.get("in_tokens", ""), cell.cost.get("out_tokens", ""),
            cell.cost.get("requests", ""), cell.cost.get("wall_ms", ""),
            cell.run_id, cell.started_at,
        ])
    return buffer.getvalue()


def render_markdown(matrix: Matrix, comparisons: Sequence[Any] = ()) -> str:
    """给 PR / issue / 聊天窗口的版本。表格 + 来历 + 配对结论。"""
    lines: list[str] = ["# Onyx 评测报告", ""]
    lines.append(f"数据来源：`{len(matrix.cells)}` 次运行 · 数据集 "
                 + ("、".join(f"`{p}`" for p in matrix.provenance) or "未知"))
    lines.append("")
    if matrix.warnings:
        lines.append("> **可比性警告**")
        for note in matrix.warnings:
            lines.append(f"> - {note}")
        lines.append("")

    header = "| 模型 | " + " | ".join(matrix.tasks) + " |"
    lines.append("## 模型 × 任务矩阵")
    lines.append(header)
    lines.append("|---" * (len(matrix.tasks) + 1) + "|")
    for model in matrix.models:
        row = [f"`{model}`"]
        for task in matrix.tasks:
            cell = matrix.cell(model, task)
            row.append("—" if cell is None else _cell_text(cell))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("每格是**该模型在该任务上最新一次 done 运行**的主分数与 95% CI；"
                 "「—」表示没有可判定样本或没跑过，两种情况都绝不是 0 分。")
    lines.append("")

    if comparisons:
        lines.append("## 配对对比")
        for item in comparisons:
            lines.extend(_comparison_lines(item))

    lines.append("## 每次运行的成本")
    lines.append("| 模型 | 任务 | 请求 | in tok | out tok | 墙钟 | run_id |")
    lines.append("|---|---|---|---|---|---|---|")
    for cell in matrix.cells:
        lines.append(
            f"| `{cell.model_id}` | {cell.task_id} | {cell.cost.get('requests', '—')} "
            f"| {cell.cost.get('in_tokens', '—')} | {cell.cost.get('out_tokens', '—')} "
            f"| {_ms(cell.cost.get('wall_ms'))} | `{cell.run_id[:8]}…` |"
        )
    lines.append("")
    return "\n".join(lines)


def _cell_text(cell: Cell) -> str:
    value = "—" if cell.value is None else f"{cell.value:.3f}"
    ci = cell.ci or {}
    low, high = ci.get("low"), ci.get("high")
    span = "" if low is None or high is None else f" [{low:.3f}–{high:.3f}]"
    flag = " ⚠" if cell.low_confidence else ""
    # 覆盖率只在你没看全的时候才加——全量覆盖时多写一句"236/236"是噪声
    cover = (f"（可判定 {cell.n_judged}/{cell.n_total}）"
             if (cell.coverage or 1.0) < THIN_COVERAGE else "")
    return f"{cell.metric} {value}{span}{cover}{flag}"


def _comparison_lines(item: Any) -> list[str]:
    """Comparison → md 行。接受 `eval.compare.Comparison` 或它的 as_dict()。"""
    data = item if isinstance(item, dict) else item.as_dict()
    base, target = data["base"], data["target"]
    ci = data.get("delta_ci") or {}
    out = [
        f"### `{base['model_id']}` → `{target['model_id']}` · {base['task_id']}",
        "",
        f"- 配对样本 **{data['n_paired']}** 条"
        + (f"，覆盖率 {data['coverage']:.0%}" if data.get("coverage") is not None else ""),
        f"- 净变化：改善 {data['improved']} / 劣化 {data['regressed']}"
        f" / 不变 {data['unchanged']}",
        f"- 均值差 {data['mean_delta']:+.4f}"
        + (f"，95% CI [{ci['low']:+.4f}, {ci['high']:+.4f}]（n={ci.get('n')}）"
           if ci.get("low") is not None else "（配对样本太少，给不出区间）"),
    ]
    if data.get("warnings"):
        out.append("- 可比性：" + "；".join(data["warnings"]))
    worse = [c for c in data["cases"] if c["delta"] < 0][:10]
    if worse:
        out += ["", "| case | 变化 | base | target |", "|---|---|---|---|"]
        out += [f"| `{c['case_id']}` | {c['delta']:+.2f} | {c['verdict_base']} "
                f"| {c['verdict_target']} |" for c in worse]
    out.append("")
    return out


def render_html(matrix: Matrix, comparisons: Sequence[Any] = ()) -> str:
    """自包含单文件。样式是 tokens.css 的那套颜色，脱离服务也能读。"""
    rows: list[str] = []
    for model in matrix.models:
        cells = "".join(
            f"<td>{_html_cell(matrix.cell(model, task))}</td>" for task in matrix.tasks
        )
        rows.append(f"<tr><th>{_esc(model)}</th>{cells}</tr>")
    head = "".join(f"<th>{_esc(task)}</th>" for task in matrix.tasks)

    warn_html = ""
    if matrix.warnings:
        warn_html = ('<div class="warn"><b>可比性警告</b><ul>'
                     + "".join(f"<li>{_esc(w)}</li>" for w in matrix.warnings) + "</ul></div>")

    radar = _radar_svg(matrix) if len(matrix.tasks) >= MIN_AXES_FOR_RADAR else ""
    radar_block = (
        f'<section><h2>能力形状</h2>{radar}'
        f'<p class="note">每条轴按 [0,1] 归一；轴数少于 {MIN_AXES_FOR_RADAR} 时不画雷达图，'
        "因为两点连成的\"形状\"没有任何信息量。</p></section>"
        if radar else ""
    )

    comp_html = "".join(_comparison_html(item) for item in comparisons)
    cost_rows = "".join(
        f"<tr><td>{_esc(c.model_id)}</td><td>{_esc(c.task_id)}</td>"
        f"<td>{_esc(c.cost.get('requests'))}</td><td>{_esc(c.cost.get('in_tokens'))}</td>"
        f"<td>{_esc(c.cost.get('out_tokens'))}</td><td>{_esc(_ms(c.cost.get('wall_ms')))}</td>"
        f"<td><code>{_esc(c.run_id)}</code></td></tr>"
        for c in matrix.cells
    )
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Onyx 评测报告</title>
<style>
:root {{
  --bg:#0b0e14; --panel:#11151d; --elev:#171c26; --border:#1f2733;
  --text:#e6e9ef; --dim:#9aa4b2; --muted:#626c7a;
  --ok:#3fb950; --warn:#d29922; --err:#f85149; --accent:#a371f7;
}}
body {{ background:var(--bg); color:var(--text); margin:0; padding:24px;
  font:14px/1.5 Inter, system-ui, 'PingFang SC', sans-serif; }}
h1 {{ font-size:18px; margin:0 0 4px; }}
h2 {{ font-size:14px; margin:24px 0 8px; color:var(--accent); }}
.meta {{ color:var(--muted); font-family:'JetBrains Mono', monospace; font-size:12px; }}
table {{ border-collapse:collapse; width:100%; margin-top:8px; }}
th, td {{ border-bottom:1px solid var(--border); padding:6px 8px; text-align:left;
  font-variant-numeric:tabular-nums; }}
thead th {{ color:var(--dim); font-weight:600; }}
tbody th {{ color:var(--dim); }}
.ci {{ color:var(--muted); font-size:12px; }}
.flag {{ color:var(--warn); }}
.bad {{ color:var(--err); }} .good {{ color:var(--ok); }}
code {{ font-family:'JetBrains Mono', monospace; font-size:12px; color:var(--dim); }}
.warn {{ background:var(--elev); border:1px solid var(--border); border-left:3px solid var(--warn);
  padding:8px 12px; margin:12px 0; }}
.note {{ color:var(--muted); font-size:12px; }}
section {{ margin-top:8px; }}
</style></head><body>
<h1>Onyx 评测报告</h1>
<div class="meta">{_esc(len(matrix.cells))} 次运行 · 数据集 {_esc('、'.join(matrix.provenance) or '未知')}
 · 生成于 {_esc(datetime.now().strftime('%Y-%m-%d %H:%M'))}</div>
{warn_html}
<section><h2>模型 × 任务矩阵</h2>
<table><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>
<p class="note">每格是该组合**最新一次 done 运行**的主分数与 95% CI；「—」表示没有可判定样本或没跑过，
不是 0 分。</p></section>
{radar_block}
{comp_html}
<section><h2>每次运行的成本</h2>
<table><thead><tr><th>模型</th><th>任务</th><th>请求</th><th>in tok</th><th>out tok</th>
<th>墙钟</th><th>run_id</th></tr></thead><tbody>{cost_rows}</tbody></table></section>
</body></html>
"""


def _html_cell(cell: Cell | None) -> str:
    if cell is None or cell.value is None:
        return '<span class="ci">—</span>'
    ci = cell.ci or {}
    span = ""
    if ci.get("low") is not None and ci.get("high") is not None:
        span = f' <span class="ci">[{ci["low"]:.3f}–{ci["high"]:.3f}]</span>'
    flag = ' <span class="flag">⚠低样本</span>' if cell.low_confidence else ""
    cover = (f' <span class="flag">可判定 {cell.n_judged}/{cell.n_total}</span>'
             if (cell.coverage or 1.0) < THIN_COVERAGE else "")
    return f"{_esc(cell.metric)} {cell.value:.3f}{span}{cover}{flag}"


def _comparison_html(item: Any) -> str:
    data = item if isinstance(item, dict) else item.as_dict()
    base, target = data["base"], data["target"]
    ci = data.get("delta_ci") or {}
    interval = ("配对样本太少，给不出区间" if ci.get("low") is None else
                f"均值差 {data['mean_delta']:+.4f}，95% CI "
                f"[{ci['low']:+.4f}, {ci['high']:+.4f}]（n={ci.get('n')}）")
    worse = [c for c in data["cases"] if c["delta"] < 0][:20]
    rows = "".join(
        f"<tr><td><code>{_esc(c['case_id'])}</code></td>"
        f"<td class=\"bad\">{c['delta']:+.2f}</td>"
        f"<td>{_esc(c['verdict_base'])}</td><td>{_esc(c['verdict_target'])}</td>"
        f"<td><code>{_esc((c.get('trace_base') or '')[:12])}…</code></td>"
        f"<td><code>{_esc((c.get('trace_target') or '')[:12])}…</code></td></tr>"
        for c in worse
    )
    return f"""<section><h2>配对对比 · {_esc(base['model_id'])} → {_esc(target['model_id'])}</h2>
<p class="note">配对 {data['n_paired']} 条 · 改善 <span class="good">{data['improved']}</span> /
劣化 <span class="bad">{data['regressed']}</span> / 不变 {data['unchanged']} · {_esc(interval)}</p>
<p class="note">{_esc('；'.join(data.get('warnings') or []))}</p>
<table><thead><tr><th>劣化 case</th><th>Δ</th><th>base 判定</th><th>target 判定</th>
<th>base trace</th><th>target trace</th></tr></thead><tbody>{rows}</tbody></table></section>"""


def _radar_svg(matrix: Matrix, *, size: int = 320) -> str:
    """手写 SVG 雷达图：每个模型一条多边形。

    刻意不用图表库（与前端同一决定）：这里的图形形态固定，
    手写既可控又零依赖，报告文件还能直接发给别人。
    """
    import math

    cx = cy = size / 2
    radius = size / 2 - 44
    axes = list(matrix.tasks)
    count = len(axes)

    def point(index: int, value: float) -> tuple[float, float]:
        angle = -math.pi / 2 + index * 2 * math.pi / count
        capped = min(max(value, 0.0), 1.0)
        return cx + math.cos(angle) * radius * capped, cy + math.sin(angle) * radius * capped

    def polygon(values: Sequence[float]) -> str:
        return " ".join(
            f"{point(i, value)[0]:.1f},{point(i, value)[1]:.1f}"
            for i, value in enumerate(values)
        )

    grid = "".join(
        f'<polygon points="{polygon([step] * count)}" fill="none" stroke="var(--border)"/>'
        for step in (0.25, 0.5, 0.75, 1.0)
    )
    spokes = "".join(
        f'<line x1="{cx}" y1="{cy}" x2="{point(i, 1.0)[0]:.1f}" y2="{point(i, 1.0)[1]:.1f}"'
        f' stroke="var(--border)"/>'
        f'<text x="{point(i, 1.2)[0]:.1f}" y="{point(i, 1.2)[1]:.1f}" fill="var(--dim)"'
        f' font-size="11" text-anchor="middle">{_esc(task)}</text>'
        for i, task in enumerate(axes)
    )
    palette = ("var(--ok)", "var(--info)", "var(--accent)", "var(--warn)", "var(--err)")
    series = []
    for index, model in enumerate(matrix.models):
        color = palette[index % len(palette)]
        # 没有格子的轴按 0 画，但**图例与表格仍然显示真实值**：
        # 缺格的位置在图上凹陷是"没跑过"的形状线索，不能靠补 0 假装跑过
        values = [
            (cell.value if cell is not None and cell.value is not None else 0.0)
            for cell in (matrix.cell(model, task) for task in axes)
        ]
        series.append(f'<polygon points="{polygon(values)}" fill="{color}" '
                      f'fill-opacity="0.12" stroke="{color}" stroke-width="1.5"/>')
    legend = "".join(
        f'<text x="8" y="{size - 8 - 14 * index}" fill="{palette[index % len(palette)]}"'
        f' font-size="11">{_esc(model)}</text>'
        for index, model in enumerate(matrix.models)
    )
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="模型 × 任务雷达图">{grid}{spokes}{"".join(series)}{legend}</svg>')


def _esc(value: Any) -> str:
    """HTML 转义。任务名、模型名、警告文本都会进 HTML，一处漏转就是一处注入面。"""
    text = "" if value is None else str(value)
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace('"', "&quot;"))


def _ms(value: Any) -> str:
    if value is None:
        return "—"
    seconds = float(value) / 1000
    return f"{seconds:.1f}s" if seconds < 120 else f"{seconds / 60:.1f}min"
