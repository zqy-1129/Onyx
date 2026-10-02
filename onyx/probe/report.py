"""能力矩阵渲染：控制台（rich）与 markdown（追加进 docs/PROBES.md）。

单元格三态：`✓ 确认` / `✗ 不支持` / `? 未实测`。
把 `✗` 与 `?` 画成同一个符号是这类看板最常见的误导——前者要 skip，后者要先跑探针。
"""

from __future__ import annotations

from dataclasses import dataclass

from onyx.core.types import Cap
from onyx.llm.caps import CapReport

#: 低于这个宽度就切紧凑布局：13 列的矩阵在 80 列终端里只会渲染成乱码
WIDE_LAYOUT_MIN_WIDTH = 120

#: 矩阵列顺序（按"决定能跑什么评测"的重要性排）
MATRIX_COLUMNS: tuple[tuple[str, Cap], ...] = (
    ("tools", Cap.TOOLS),
    ("tool_ch", Cap.TOOL_CHOICE),
    ("think", Cap.THINKING),
    ("struct", Cap.STRUCTURED_OUTPUT),
    ("stream", Cap.STREAM_USAGE),
    ("vision", Cap.VISION),
    ("embed", Cap.EMBED),
    ("n>1", Cap.N_SAMPLING),
    ("admin", Cap.ADMIN),
)

#: 窄终端用的短标签（整行必须能塞进 80 列）
COMPACT_COLUMNS: tuple[tuple[str, Cap], ...] = (
    ("tool", Cap.TOOLS),
    ("tch", Cap.TOOL_CHOICE),
    ("thk", Cap.THINKING),
    ("str", Cap.STRUCTURED_OUTPUT),
    ("strm", Cap.STREAM_USAGE),
    ("vis", Cap.VISION),
    ("emb", Cap.EMBED),
    ("n>1", Cap.N_SAMPLING),
    ("adm", Cap.ADMIN),
)


@dataclass(frozen=True, slots=True)
class MatrixRow:
    name: str
    caps: CapReport
    parameter_size: str = ""
    quantization: str = ""
    size_gb: float = 0.0
    tool_format: str = "unknown"
    ctx_train: int | None = None
    ctx_loaded: int | None = None
    probed: bool = False
    usage_ratio: float | None = None
    usage_ratio_n: int = 0

    @property
    def ratio_text(self) -> str:
        if not self.usage_ratio or self.usage_ratio_n < 30:
            return "未标定"
        return f"{self.usage_ratio:.3f}+{self.usage_ratio_n}"


def render_markdown(rows: list[MatrixRow], *, provider_version: str = "", provider_id: str = "") -> str:
    header = ["模型", "参数", "量化", "磁盘", *(name for name, _ in MATRIX_COLUMNS),
              "tool_format", "ctx(载入/训练)", "tokens/char"]
    lines = [
        f"### 能力矩阵 · {provider_id or 'provider'}"
        + (f" · 引擎 v{provider_version}" if provider_version else ""),
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for row in rows:
        cells = [
            f"`{row.name}`", row.parameter_size or "—", row.quantization or "—",
            f"{row.size_gb}GB",
            *[row.caps.symbol(cap) for _, cap in MATRIX_COLUMNS],
            row.tool_format, f"{row.ctx_loaded or '—'} / {row.ctx_train or '—'}", row.ratio_text,
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "图例：`✓` 确认支持 · `✗` 确认不支持（评测应 skip 并写明原因）· "
        "`?` 未实测（**不等于不支持**，应先跑 `onyx probe run`）",
        "",
    ]
    unknown = sorted({
        str(cap) for row in rows for cap in row.caps.unknown
    })
    if unknown:
        lines += [f"未实测的能力位：{', '.join(unknown)}", ""]
    return "\n".join(lines)


def render_console(
    rows: list[MatrixRow],
    *,
    provider_version: str = "",
    provider_id: str = "",
    console: object | None = None,
) -> None:
    from rich.console import Console

    # 宽度可注入：默认终端宽度会把列挤到无法阅读（也无法测试）
    console = console or Console()
    title = f"能力矩阵 · {provider_id or 'provider'}"
    if provider_version:
        title += f" · 引擎 v{provider_version}"
    width = getattr(console, "width", 80) or 80
    if width < WIDE_LAYOUT_MIN_WIDTH:
        _render_compact(rows, title=title, console=console)
    else:
        _render_wide(rows, title=title, console=console)
    console.print(
        "[dim]✓ 确认 · ✗ 不支持（评测 skip）· ? 未实测（≠不支持，先跑 onyx probe run）[/dim]"
    )


def compact_caps(row: MatrixRow) -> str:
    """窄终端用：短标签 + 符号，压成一行 `tool✓ tch✗ str?`。

    标签必须短到整行能塞进 80 列，否则 rich 会截断成 `admi…`——
    被截断的能力位比不显示更糟，因为用户会以为那一列不存在。
    """
    color = {"confirmed": "green", "missing": "red", "unknown": "yellow"}
    return " ".join(
        f"[{color[row.caps.state(cap)]}]{name}{row.caps.symbol(cap)}[/]"
        for name, cap in COMPACT_COLUMNS
    )


def _render_compact(rows: list[MatrixRow], *, title: str, console: object) -> None:
    from rich.table import Table

    overview = Table(title=title, pad_edge=False, expand=False)
    overview.add_column("模型", no_wrap=True, style="bold")
    overview.add_column("参数", justify="right")
    overview.add_column("量化")
    overview.add_column("磁盘", justify="right")
    overview.add_column("ctx 载入/训练", justify="right", no_wrap=True)
    overview.add_column("tokens/char", justify="right", no_wrap=True)
    for row in rows:
        overview.add_row(
            row.name, row.parameter_size or "—", row.quantization or "—", f"{row.size_gb}GB",
            f"{row.ctx_loaded or '—'}/{row.ctx_train or '—'}", row.ratio_text,
        )
    console.print(overview)

    caps_table = Table(title="能力位", pad_edge=False, expand=False)
    caps_table.add_column("模型", no_wrap=True, style="bold")
    caps_table.add_column("能力", no_wrap=True)
    caps_table.add_column("tool_format")
    for row in rows:
        caps_table.add_row(
            row.name, compact_caps(row),
            row.tool_format if row.probed else f"[dim]{row.tool_format}[/dim]",
        )
    console.print(caps_table)


def _render_wide(rows: list[MatrixRow], *, title: str, console: object) -> None:
    from rich.table import Table

    table = Table(title=title, pad_edge=False, show_lines=False)
    table.add_column("模型", style="bold", no_wrap=True)
    table.add_column("参数", justify="right")
    table.add_column("量化")
    table.add_column("磁盘", justify="right")
    for name, _ in MATRIX_COLUMNS:
        table.add_column(name, justify="center")
    table.add_column("tool_format")
    table.add_column("ctx 载入/训练", justify="right")
    table.add_column("tokens/char")

    color = {"confirmed": "green", "missing": "red", "unknown": "yellow"}
    for row in rows:
        cells = [row.name, row.parameter_size or "—", row.quantization or "—", f"{row.size_gb}GB"]
        for _, cap in MATRIX_COLUMNS:
            state = row.caps.state(cap)
            cells.append(f"[{color[state]}]{row.caps.symbol(cap)}[/]")
        cells += [
            row.tool_format if row.probed else f"[dim]{row.tool_format}[/dim]",
            f"{row.ctx_loaded or '—'} / {row.ctx_train or '—'}",
            row.ratio_text,
        ]
        table.add_row(*cells)
    console.print(table)
