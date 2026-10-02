"""Onyx CLI 入口。

命令按子系统分组，随里程碑逐步补齐（见 IMPLEMENTATION.md S7）。
当前已实现：`version` / `db init` / `db info` / `doctor`。
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

from onyx import __version__
from onyx.settings import Settings, load_settings
from onyx.store.db import Database

app = typer.Typer(
    name="onyx",
    help="本地大模型观测与评测看板",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
db_app = typer.Typer(help="数据库：迁移、体检、备份", no_args_is_help=True)
app.add_typer(db_app, name="db")
probe_app = typer.Typer(help="语义实测探针：把引擎行为变成可复现的结论", no_args_is_help=True)
app.add_typer(probe_app, name="probe")


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def _settings() -> Settings:
    return load_settings().ensure_dirs()


def _db_path(settings: Settings, db: Path | None) -> Path:
    """`--db` 是**数据库文件路径**，不是数据目录——两者别混。"""
    return Path(db).resolve() if db else settings.db_path


def _echo_checks(checks: list[CheckResult]) -> bool:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Onyx doctor · v{__version__}", show_lines=False, pad_edge=False)
    table.add_column("", width=2)
    table.add_column("检查项", style="bold")
    table.add_column("结果")
    table.add_column("修复建议", style="dim")
    for c in checks:
        table.add_row(
            "[green]✓[/]" if c.ok else "[red]✗[/]",
            c.name,
            c.detail,
            "" if c.ok else c.hint,
        )
    console.print(table)
    return all(c.ok for c in checks)


# ── version ────────────────────────────────────────────────────────
@app.command()
def version() -> None:
    """打印版本与运行环境。"""
    typer.echo(f"onyx {__version__} · python {platform.python_version()} · {sys.platform}")


# ── db ─────────────────────────────────────────────────────────────
@db_app.command("init")
def db_init(
    db: Path = typer.Option(None, "--db", help="数据库路径，默认 <repo>/.data/onyx.sqlite"),
) -> None:
    """创建/迁移数据库（幂等）。"""
    settings = _settings()
    path = _db_path(settings, db)
    database = Database(path)
    typer.echo(f"已就绪: {path} (schema_version={database.version()})")
    database.close()


@db_app.command("info")
def db_info(
    db: Path = typer.Option(None, "--db", help="数据库路径"),
) -> None:
    """打印 schema 版本、表清单与行数。"""
    settings = _settings()
    path = _db_path(settings, db)
    if not path.exists():
        typer.echo(f"数据库不存在: {path}（先跑 onyx db init）", err=True)
        raise typer.Exit(1)
    database = Database(path)
    typer.echo(f"path           : {path}")
    typer.echo(f"schema_version : {database.version()}")
    typer.echo(f"size           : {path.stat().st_size / 1024:.1f} KiB")
    typer.echo(f"blobs          : {_blob_count(settings)}")
    typer.echo("tables         :")
    for name in database.table_names():
        if name == "schema_version":
            continue
        # 表名来自 sqlite_master，非用户输入
        count = database.scalar(f"SELECT COUNT(*) FROM {name}")
        typer.echo(f"  {name:<14} {count}")
    database.close()


def _blob_count(settings: Settings) -> str:
    if not settings.blob_dir.exists():
        return "0 files / 0 B"
    files = [p for p in settings.blob_dir.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    return f"{len(files)} files / {total / 1024:.1f} KiB"


# ── doctor ─────────────────────────────────────────────────────────
@app.command()
def doctor(
    db: Path = typer.Option(None, "--db", help="数据库路径"),
    ollama_url: str = typer.Option("http://127.0.0.1:11434", "--ollama-url", help="Ollama base url"),
    skip_network: bool = typer.Option(False, "--skip-network", help="跳过网络检查"),
) -> None:
    """体检：环境、数据库、blob 一致性、Ollama 可达性。"""
    settings = _settings()
    path = _db_path(settings, db)
    checks: list[CheckResult] = []

    checks.append(CheckResult(
        "Python ≥ 3.12", sys.version_info >= (3, 12),
        platform.python_version(), "uv python install 3.12",
    ))
    checks.append(CheckResult(
        "数据目录可写", _writable(settings.data_dir), str(settings.data_dir),
        "检查磁盘权限或设置 ONYX_DATA_DIR",
    ))

    db_ok = path.exists()
    detail = "未初始化"
    version_no = 0
    if db_ok:
        database = Database(path)
        version_no = database.version()
        detail = f"schema_version={version_no}"
        database.close()
    checks.append(CheckResult("数据库已迁移", db_ok and version_no >= 1, detail, "onyx db init"))

    if db_ok:
        database = Database(path)
        refs = [
            row[0]
            for row in database.query(
                "SELECT raw_response_ref FROM trace WHERE raw_response_ref IS NOT NULL"
            )
        ]
        database.close()
        from onyx.core.content import FileBlobStore

        store = FileBlobStore(settings.blob_dir)
        missing = [r for r in refs if not store.exists(r)]
        checks.append(CheckResult(
            "blob 引用完整", not missing,
            f"{len(refs) - len(missing)}/{len(refs)} 可解析",
            "原始证据缺失，检查 .data/blobs 是否被清理",
        ))

    if not skip_network:
        checks.append(_check_ollama(ollama_url))

    raise typer.Exit(0 if _echo_checks(checks) else 1)


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _check_ollama(base_url: str) -> CheckResult:
    try:
        import httpx
    except ImportError:
        return CheckResult("Ollama 可达", False, "httpx 未安装", "uv sync --extra runtime")
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/version", timeout=3.0)
        resp.raise_for_status()
        return CheckResult("Ollama 可达", True, f"{base_url} · v{resp.json().get('version', '?')}")
    except Exception as exc:  # noqa: BLE001 - 体检要把任何失败都变成一行可读结论
        return CheckResult(
            "Ollama 可达", False, f"{base_url} 无响应（{type(exc).__name__}）",
            "启动 Ollama 服务，或用 --skip-network 跳过",
        )


# ── probe ──────────────────────────────────────────────────────────
@probe_app.command("list")
def probe_list() -> None:
    """列出已注册的探针。"""
    from onyx.probe import registered_probes

    for name in registered_probes():
        typer.echo(name)


@probe_app.command("run")
def probe_run(
    model: str = typer.Option(..., "--model", help="要实测的模型名"),
    suite: str = typer.Option("all", "--suite", help="逗号分隔的探针名，或 all"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url", help="Ollama base url"),
    write: Path = typer.Option(None, "--append-to", help="把 markdown 结论追加到该文件"),
) -> None:
    """对真实引擎跑语义实验。单 GPU 独占 ⇒ 探针串行执行。"""
    from onyx.llm.providers.ollama import OllamaProvider
    from onyx.probe import registered_probes, render_markdown, run_suite

    names = registered_probes() if suite == "all" else [s.strip() for s in suite.split(",") if s.strip()]
    provider = OllamaProvider(base_url=url)
    if not provider.client.is_reachable():
        typer.echo(f"Ollama 不可达: {url}", err=True)
        raise typer.Exit(1)
    version = provider.info().version
    report, ctx = run_suite(provider, model, names, provider_version=version)
    provider.close()

    markdown = render_markdown(report, ctx)
    if write:
        existing = write.read_text(encoding="utf-8") if write.exists() else ""
        write.write_text(existing.rstrip() + "\n\n---\n\n" + markdown, encoding="utf-8")
        typer.echo(f"已追加到 {write}")
    for finding in report.findings:
        flag = "?" if finding.unknown else "✓"
        typer.echo(f"{flag} {finding.probe:<16} {finding.verdict}")
        for key, value in finding.evidence.items():
            typer.echo(f"    {key} = {value}")
    unknown = [f.probe for f in report.findings if f.unknown]
    if unknown:
        typer.echo(f"\n以下探针未得出确定结论（UI 应显示「—」而不是 0）: {unknown}")


# ── models ─────────────────────────────────────────────────────────
models_app = typer.Typer(help="模型资产：同步、清单、载入状态", no_args_is_help=True)
app.add_typer(models_app, name="models")
traces_app = typer.Typer(help="trace：列表、详情、重放", no_args_is_help=True)
app.add_typer(traces_app, name="traces")


def _runtime(url: str, db: Path | None, *, sample_gpu: bool = True):
    from onyx.runtime import build_runtime

    return build_runtime(
        base_url=url, db_path=str(db) if db else None, sample_gpu=sample_gpu, event_log=False
    )


@models_app.command("sync")
def models_sync(
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
) -> None:
    """从引擎拉取模型清单并落库。"""
    from onyx.runtime import sync_models

    runtime = _runtime(url, db)
    try:
        count = sync_models(runtime)
        runtime.flush()
        typer.echo(f"已同步 {count} 个模型 → {runtime.settings.db_path}")
    finally:
        runtime.close()


@models_app.command("ls")
def models_ls(
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
) -> None:
    """列出已安装模型与当前载入状态（显存 / 上下文 / keep-alive 剩余）。"""
    from rich.console import Console
    from rich.table import Table

    runtime = _runtime(url, db)
    try:
        loaded = {m.name: m for m in runtime.provider.running()}
        table = Table(title="模型", pad_edge=False)
        for column in ("模型", "参数", "量化", "磁盘", "能力", "状态", "显存", "ctx(载入/训练)"):
            table.add_column(column)
        for card in runtime.provider.list_models():
            live = loaded.get(card.name)
            table.add_row(
                card.name, card.parameter_size or "—", card.quantization or "—",
                f"{card.size_gb}GB", ",".join(card.capabilities) or "—",
                f"[green]已载入[/] 剩 {_remaining(live.expires_at)}" if live else "[dim]未载入[/]",
                f"{live.size_vram / 1e9:.2f}GB" if live else "—",
                f"{live.context_length if live else '—'} / {card.context_length or '—'}",
            )
        Console().print(table)
    finally:
        runtime.close()


def _remaining(expires_at: str) -> str:
    from datetime import datetime

    try:
        target = datetime.fromisoformat(expires_at)
    except ValueError:
        return "—"
    seconds = (target - datetime.now(target.tzinfo)).total_seconds()
    return f"{int(seconds)}s" if seconds > 0 else "已过期"


# ── chat ───────────────────────────────────────────────────────────
DEMO_TOOLS = {
    "weather": ("get_weather", "查询指定城市当前天气", {"city": "string"}),
}


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="用户消息"),
    model: str = typer.Option(..., "--model", "-m"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
    stream: bool = typer.Option(False, "--stream"),
    max_tokens: int = typer.Option(512, "--max-tokens"),
    thinking: bool = typer.Option(None, "--thinking/--no-thinking"),
    tool: str = typer.Option("", "--tool", help="逗号分隔的演示工具名，如 weather"),
) -> None:
    """发一次真实对话并把它完整落库（token / 延迟 / 工具 / 原始证据）。"""
    from rich.console import Console
    from rich.table import Table

    from onyx.core.types import GenerationRequest, GenParams, ToolSpec
    from onyx.runtime import sync_models

    console = Console()
    runtime = _runtime(url, db)
    try:
        sync_models(runtime)
        tools = tuple(
            ToolSpec(
                name=DEMO_TOOLS[name][0], description=DEMO_TOOLS[name][1],
                parameters={
                    "type": "object",
                    "properties": {k: {"type": v} for k, v in DEMO_TOOLS[name][2].items()},
                    "required": list(DEMO_TOOLS[name][2]),
                },
            )
            for name in (t.strip() for t in tool.split(",")) if name in DEMO_TOOLS
        )
        req = GenerationRequest.of(
            model, prompt,
            params=GenParams(max_tokens=max_tokens, temperature=0.0),
            thinking=thinking, stream=stream, tools=tools,
        )
        result = runtime.gateway.generate(req)
        runtime.flush()

        gen = result.generation
        if gen.thinking:
            console.print(f"[dim]💭 thinking ({len(gen.thinking)} 字符)[/dim]")
            console.print(gen.thinking[:400] + ("…" if len(gen.thinking) > 400 else ""))
        console.print(gen.text or "[yellow]（正文为空）[/yellow]")
        for call in gen.tool_calls:
            console.print(f"[cyan]→ 工具调用[/cyan] {call.name} {call.arguments} "
                          f"[dim]({call.parse_status})[/dim]")

        usage = result.usage
        table = Table(title=f"trace {result.trace_id}", pad_edge=False)
        table.add_column("指标")
        table.add_column("值", justify="right")
        if usage:
            rows = [
                ("输入 token", usage.in_tokens), ("输出 token", usage.out_tokens),
                ("来源 / 置信度", f"{usage.source} / {usage.confidence}"),
                ("drift", None if usage.drift_pct is None else f"{usage.drift_pct:.2%}"),
                ("工具定义 token", sum(p.tokens for p in usage.parts if p.part == "tool_defs") or "—"),
                ("模板控制符", sum(p.tokens for p in usage.parts if p.part == "template_ctl") or "—"),
            ]
        else:
            rows = [("usage", "无")]
        rows += [
            ("TTFT", f"{result.latency.get('ttft_ms'):.1f}ms" if result.latency.get("ttft_ms") else "—"),
            ("prefill 模式", result.latency.get("prefill_mode") or "—"),
            ("prefill TPS", _fmt(result.latency.get("prefill_tps"))),
            ("decode TPS", _fmt(result.latency.get("decode_tps"))),
            ("wall", _fmt(result.latency.get("wall_ms"), "ms")),
            ("异常", ", ".join(sorted({c for c, _, _ in result.anomalies})) or "无"),
        ]
        for name, value in rows:
            table.add_row(str(name), str(value))
        console.print(table)
        console.print(f"[dim]onyx traces show {result.trace_id}[/dim]")
    finally:
        runtime.close()


def _fmt(value: object, suffix: str = "") -> str:
    return f"{value:.1f}{suffix}" if isinstance(value, int | float) else "—"


# ── traces ─────────────────────────────────────────────────────────
@traces_app.command("ls")
def traces_ls(
    limit: int = typer.Option(20, "--limit"),
    db: Path = typer.Option(None, "--db"),
    purpose: str = typer.Option(None, "--purpose"),
) -> None:
    """最近的 trace 列表。"""
    from rich.console import Console
    from rich.table import Table

    from onyx.store.repos import TraceRepo, UsageRepo

    settings = _settings()
    with Database(_db_path(settings, db)) as database:
        traces, usage_repo = TraceRepo(database), UsageRepo(database)
        rows = traces.list(limit=limit, purpose=purpose)
        # 一次性取出相关异常再分组，避免每行一次查询
        wanted = {r.id for r in rows}
        anomaly_counts: dict[str, int] = {}
        for item in traces.list_anomalies(limit=1000):
            if item.trace_id in wanted:
                anomaly_counts[item.trace_id] = anomaly_counts.get(item.trace_id, 0) + 1
        table = Table(title=f"最近 {limit} 条 trace", pad_edge=False)
        for column in ("id", "purpose", "model", "in", "out", "src", "prefill", "tps", "状态", "工具", "异常"):
            table.add_column(column)
        for record in rows:
            bundle = usage_repo.fetch(record.id)
            u = bundle.usage
            table.add_row(
                record.id[-8:], record.purpose, record.model_name or "—",
                str(u.in_tokens) if u and u.in_tokens else "—",
                str(u.out_tokens) if u and u.out_tokens else "—",
                u.source if u else "—", (u.prefill_mode or "—") if u else "—",
                _fmt(u.decode_tps) if u else "—", record.status,
                str(len(traces.list_tool_calls(record.id))),
                str(anomaly_counts.get(record.id, 0)),
            )
        Console().print(table)


@traces_app.command("show")
def traces_show(
    trace_id: str = typer.Argument(...),
    db: Path = typer.Option(None, "--db"),
) -> None:
    """一条 trace 的完整证据：多来源计数、分段归因、工具调用、异常、原始 body。"""
    from rich.console import Console
    from rich.table import Table

    from onyx.store.repos import TraceRepo, UsageRepo

    console = Console()
    settings = _settings()
    with Database(_db_path(settings, db)) as database:
        record = TraceRepo(database).get(trace_id)
        if record is None:
            typer.echo(f"trace 不存在: {trace_id}", err=True)
            raise typer.Exit(1)
        bundle = UsageRepo(database).fetch(trace_id)
        calls = TraceRepo(database).list_tool_calls(trace_id)
        anomalies = [a for a in TraceRepo(database).list_anomalies(limit=500) if a.trace_id == trace_id]

        console.print(f"[bold]trace[/bold] {record.id}  [{record.status}] {record.purpose}")
        console.print(f"model={record.model_name} provider={record.provider_id} "
                      f"started={record.started_at} finish={record.finish_reason}")
        if record.error:
            console.print(f"[red]error: {record.error}[/red]")

        if bundle.usage:
            u = bundle.usage
            console.print(f"\n[bold]采信[/bold] in={u.in_tokens} out={u.out_tokens} "
                          f"thinking={u.thinking_tokens} source={u.source} conf={u.confidence} "
                          f"drift={u.drift_pct}")
            console.print(f"[bold]延迟[/bold] ttft={_fmt(u.ttft_ms, 'ms')} prefill={u.prefill_mode} "
                          f"({_fmt(u.prefill_ms_per_token, 'ms/tok')}) "
                          f"prefill_tps={_fmt(u.prefill_tps)} decode_tps={_fmt(u.decode_tps)}")
        alt_table = Table(title="各来源计数（对账）", pad_edge=False)
        for column in ("source", "in", "out", "thinking", "cached", "ok", "note"):
            alt_table.add_column(column)
        for alt in bundle.alts:
            alt_table.add_row(str(alt.source), str(alt.in_tokens), str(alt.out_tokens),
                              str(alt.thinking_tokens), str(alt.cached_tokens),
                              "✓" if alt.ok else "✗", alt.note[:40])
        console.print(alt_table)

        if bundle.parts:
            part_table = Table(title="分段归因（Σ分段 + template_ctl = 引擎计数）", pad_edge=False)
            for column in ("part", "tokens", "bytes"):
                part_table.add_column(column)
            for part in bundle.parts:
                part_table.add_row(part.part, str(part.tokens), str(part.bytes if part.bytes else "—"))
            console.print(part_table)

        if calls:
            call_table = Table(title="工具调用", pad_edge=False)
            for column in ("step", "name", "parse", "args", "result", "ms"):
                call_table.add_column(column)
            for call in calls:
                call_table.add_row(str(call.step), call.name or "—", call.parse_status,
                                   (call.args_raw or str(call.args or ""))[:60],
                                   call.result_status or "未执行", _fmt(call.latency_ms))
            console.print(call_table)

        if anomalies:
            for item in anomalies:
                color = {"error": "red", "warn": "yellow"}.get(item.severity, "dim")
                console.print(f"[{color}]⚠ {item.code}[/{color}] {item.detail}")

        console.print(f"\n[dim]messages={record.messages_ref}\noutput={record.output_ref}[/dim]")


@traces_app.command("replay")
def traces_replay(
    trace_id: str = typer.Argument(...),
    db: Path = typer.Option(None, "--db"),
    dry_run: bool = typer.Option(True, "--dry-run/--send"),
) -> None:
    """从原始证据重建请求。默认只打印不发送——重放会真实占用 GPU。"""
    import json as _json

    from onyx.core.content import FileBlobStore
    from onyx.store.repos import TraceRepo

    settings = _settings()
    with Database(_db_path(settings, db)) as database:
        record = TraceRepo(database).get(trace_id)
        if record is None:
            typer.echo(f"trace 不存在: {trace_id}", err=True)
            raise typer.Exit(1)
        blobs = FileBlobStore(settings.blob_dir)
        messages = blobs.get_json(record.messages_ref) if record.messages_ref else []
        tools = blobs.get_json(record.tools_ref) if record.tools_ref else []
    payload = {"model": record.model_name, "messages": messages, "params": record.params}
    if tools:
        payload["tools"] = tools
    typer.echo(_json.dumps(payload, ensure_ascii=False, indent=2))
    if dry_run:
        typer.echo("[dry-run] 未发送。加 --send 才会真的打引擎。")


@probe_app.command("matrix")
def probe_matrix(
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
    markdown: Path = typer.Option(None, "--markdown", help="把矩阵追加到该 markdown 文件"),
) -> None:
    """逐模型输出能力矩阵：✓ 确认 / ✗ 不支持 / ? 未实测（三者必须可区分）。"""
    from onyx.llm.caps import CapReport, infer_caps
    from onyx.probe.report import MatrixRow, render_console, render_markdown
    from onyx.runtime import sync_models
    from onyx.store.repos import ModelRepo

    runtime = _runtime(url, db, sample_gpu=False)
    try:
        sync_models(runtime)
        info = runtime.provider.info()
        repo = ModelRepo(runtime.db)
        loaded = {m.name: m for m in runtime.provider.running()}
        rows: list[MatrixRow] = []
        for card in runtime.provider.list_models():
            record = repo.find_by_name(info.id, card.name)
            findings = (record.probe if record else {}) or {}
            caps = CapReport.from_dict(record.extra.get("caps") if record else None) \
                if record and record.extra.get("caps") else \
                infer_caps(engine_capabilities=card.capabilities, probe_findings=findings,
                           api_style=info.api_style, provider_kind=info.kind)
            live = loaded.get(card.name)
            rows.append(MatrixRow(
                name=card.name, caps=caps, parameter_size=card.parameter_size,
                quantization=card.quantization, size_gb=card.size_gb,
                tool_format=(record.tool_format if record else "unknown"),
                ctx_train=card.context_length, ctx_loaded=live.context_length if live else None,
                probed=bool(findings),
                usage_ratio=record.usage_ratio if record else None,
                usage_ratio_n=record.usage_ratio_n or 0 if record else 0,
            ))
        render_console(rows, provider_version=info.version, provider_id=info.id)
        if markdown:
            text = render_markdown(rows, provider_version=info.version, provider_id=info.id)
            existing = markdown.read_text(encoding="utf-8") if markdown.exists() else ""
            markdown.write_text(existing.rstrip() + "\n\n" + text, encoding="utf-8")
            typer.echo(f"已追加到 {markdown}")
    finally:
        runtime.close()


@probe_app.command("write-back")
def probe_write_back(
    model: str = typer.Option(..., "--model"),
    suite: str = typer.Option("all", "--suite"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
) -> None:
    """跑探针并把结论回灌到模型档案（tool_format / capabilities / 三态能力位）。"""
    from onyx.llm.caps import infer_caps
    from onyx.probe import registered_probes, run_suite
    from onyx.runtime import sync_models
    from onyx.store.repos import ModelRepo

    runtime = _runtime(url, db, sample_gpu=False)
    try:
        sync_models(runtime)
        info = runtime.provider.info()
        names = registered_probes() if suite == "all" else [s.strip() for s in suite.split(",") if s]
        report, _ = run_suite(runtime.provider, model, names, provider_version=info.version)
        findings = {f.probe: f.verdict for f in report.findings}
        card = next((c for c in runtime.provider.list_models() if c.name == model), None)
        caps = infer_caps(
            engine_capabilities=card.capabilities if card else (),
            probe_findings=findings, api_style=info.api_style, provider_kind=info.kind,
        )
        repo = ModelRepo(runtime.db)
        record = repo.find_by_name(info.id, model)
        if record is None:
            typer.echo(f"模型未落库: {model}", err=True)
            raise typer.Exit(1)
        tool_format = findings.get("tool_format", "unknown").split("_")[0]
        repo.update_model(
            record.id,
            probe_json=findings,
            tool_format=tool_format if tool_format in {"native", "xml", "json", "chat"} else "unknown",
            capabilities_json=list(caps.confirmed | caps.missing | caps.unknown),
            extra_json={**record.extra, "caps": caps.as_dict(),
                        "probe_version": info.version},
        )
        runtime.flush()
        typer.echo(f"已回灌 {model}: tool_format={tool_format}")
        for name, verdict in findings.items():
            typer.echo(f"  {name:<16} {verdict}")
        typer.echo(f"  能力位 ✓{len(caps.confirmed)} ✗{len(caps.missing)} ?{len(caps.unknown)}")
    finally:
        runtime.close()


# ── calibrate ──────────────────────────────────────────────────────
@app.command()
def calibrate(
    model: str = typer.Option(..., "--model"),
    n: int = typer.Option(40, "--n", help="样本数（不同长度的 prompt）"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
    write: bool = typer.Option(True, "--write/--no-write", help="把标定结果写回模型档案"),
) -> None:
    """标定该模型的 tokens/char 比值与模板固定开销（fitted 档）。

    用引擎自报的 `prompt_eval_count` 作真值反推——这样即使模型没有可用的
    chat template（PROBES P9：3 个模型里 2 个没有），也能把估计误差压到个位数百分比。
    """
    from onyx.core.types import GenerationRequest, GenParams, TokenSource
    from onyx.llm.measurement.calibrate import CalibrationSample, fit_by_script
    from onyx.runtime import sync_models
    from onyx.store.repos import ModelRepo

    # 中英文交替拼接：**任何长度的前缀都保持相同的语系比例**。
    # 若把中文段与英文段分开排布，按长度截断就会让短样本几乎纯中文、长样本含大量英文，
    # 拟合出的"比值"反映的是采样方式而不是模型属性（实测 R²=0.94 但最大相对误差 66%）。
    zh = "本地大模型推理需要精确的 token 计量，缓存命中会让吞吐数字虚高。"
    en = "Accurate token accounting requires the engine's own counts. "
    corpus = (zh + en) * 8
    runtime = _runtime(url, db, sample_gpu=False)
    try:
        sync_models(runtime)
        provider = runtime.provider
        provider.generate(GenerationRequest.of(  # 预热：把冷启动载入排除在样本外
            model, "hi", params=GenParams(max_tokens=1), thinking=False
        ))
        samples: list[CalibrationSample] = []
        lengths = [max(8, int(len(corpus) * (i + 1) / max(1, n))) for i in range(n)]
        for index, length in enumerate(lengths):
            prompt = corpus[:length]
            gen = provider.generate(GenerationRequest.of(
                model, prompt, params=GenParams(max_tokens=1, temperature=0.0), thinking=False,
                keep_alive="10m",
            ))
            engine = gen.usage_from(TokenSource.ENGINE)
            if engine is None or engine.in_tokens is None:
                typer.echo(f"样本 {index} 无引擎计数，跳过", err=True)
                continue
            samples.append(CalibrationSample(
                chars=len(prompt), tokens=engine.in_tokens, text=prompt,
                prompt_eval_ms=gen.latency.ms("prompt_eval") if gen.latency else None,
                cold=gen.latency.is_cold if gen.latency else None,
            ))
            if (index + 1) % 10 == 0:
                typer.echo(f"  已采集 {index + 1}/{n}")

        result = fit_by_script(samples)
        typer.echo(f"\n模型 {model} 标定结果（n={result.n}）")
        typer.echo(f"  中文 token/字    : {result.cjk_ratio if result.cjk_ratio is not None else '—'}")
        typer.echo(f"  其他 token/字    : "
                   f"{result.other_ratio if result.other_ratio is not None else result.ratio}")
        typer.echo(f"  模板固定开销     : {result.intercept} token/请求")
        typer.echo(f"  R²               : {result.r2}")
        typer.echo(f"  最大相对误差     : {result.max_rel_error:.2%}")
        typer.echo(f"  冷 prefill       : {result.cold_ms_per_token} ms/token")
        typer.echo(f"  热 prefill       : {result.warm_ms_per_token} ms/token")
        if not result.usable:
            typer.echo("  [!] 未达可用门槛（n≥30 且 R²≥0.90 且最大相对误差≤20%）⇒ 不写回。"
                       "fitted 档保持失效，而不是用一个看起来可信的坏标定")
        if write and result.usable:
            repo = ModelRepo(runtime.db)
            info = provider.info()
            record = repo.find_by_name(info.id, model)
            if record:
                repo.update_model(
                    record.id, usage_ratio=result.ratio, usage_ratio_n=result.n,
                    extra_json={**record.extra,
                                "fitted_intercept": result.intercept,
                                "fitted_cjk_ratio": result.cjk_ratio,
                                "fitted_other_ratio": result.other_ratio,
                                "fitted_r2": result.r2,
                                "fitted_max_rel_error": result.max_rel_error,
                                "cold_ms_per_token": result.cold_ms_per_token,
                                "warm_ms_per_token": result.warm_ms_per_token},
                )
                runtime.flush()
                typer.echo("  已写回模型档案（fitted 档生效）")
    finally:
        runtime.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
