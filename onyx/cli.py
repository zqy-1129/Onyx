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


def _runtime(url: str, db: Path | None, *, sample_gpu: bool = True, provider_kind: str = "ollama"):
    """构造运行时。

    `provider_kind` 是 DESIGN §13 的扩展点在 CLI 上的出口：换成 `mock` 就能在没有
    引擎的机器上跑通整条链路（含工具循环与评测），这也是离线测试的依据。
    """
    from onyx.runtime import build_runtime

    return build_runtime(
        provider_kind=provider_kind, base_url=url,
        db_path=str(db) if db else None,
        sample_gpu=sample_gpu and provider_kind == "ollama", event_log=False,
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


# ── serve ──────────────────────────────────────────────────────────
@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url", help="Ollama base url"),
    db: Path = typer.Option(None, "--db"),
    gpu_lock_path: Path = typer.Option(
        None, "--gpu-lock",
        help="GPU 锁文件路径；默认机器级路径。多卡机器用它给每个服务一条锁"
    ),
) -> None:
    """启动 REST + SSE 服务（看板后端）。"""
    import uvicorn

    from onyx.api.app import create_app

    app_obj = create_app(base_url=url, db_path=str(db) if db else None,
                         gpu_lock_path=gpu_lock_path)
    typer.echo(f"Onyx API: http://{host}:{port}/api/docs")
    uvicorn.run(app_obj, host=host, port=port, log_level="info")


# ── tools ──────────────────────────────────────────────────────────
tools_app = typer.Typer(help="工具注册表：导入、审计、上下文开销核算", no_args_is_help=True)
app.add_typer(tools_app, name="tools")


def _tool_registry(db: Path | None, model: str | None):
    """构造注册表。给了 --model 就用**该模型标定过的计数档位**，
    与 gateway 归因走同一个 `text_counter`——开销数字必须与 trace 里的
    `part=tool_defs` 同源，不许两处各算一套。"""
    from onyx.llm.measurement.fidelity import text_counter
    from onyx.runtime import counter_ctx_factory
    from onyx.store.repos import ToolRepo
    from onyx.tools.registry import ToolRegistry

    settings = _settings()
    database = Database(_db_path(settings, db))
    count_fn = None
    if model:
        ctx = counter_ctx_factory(database, "ollama-local")(model)
        count_fn = text_counter(ctx)
        count_fn.source_name = (  # type: ignore[attr-defined]
            "gguf_vocab" if ctx.tokenizer is not None
            else "fitted" if ctx.fitted_ratio and ctx.fitted_n >= 30
            else "heuristic"
        )
    return ToolRegistry(ToolRepo(database), count_fn=count_fn), database


@tools_app.command("import")
def tools_import(
    file: Path = typer.Argument(None, exists=True, dir_okay=False, readable=True),
    db: Path = typer.Option(None, "--db"),
    builtin: bool = typer.Option(
        False, "--builtin", help="导入内置参照工具（echo/calculator/time_now），不需要文件"
    ),
) -> None:
    """从 YAML/JSON 导入工具定义（内容 hash 变化即版本递增）。"""
    import json as _json

    from onyx.tools.registry import defs_from_payload

    if builtin:
        from onyx.tools.builtin.defs import BUILTIN_DEFS

        definitions = list(BUILTIN_DEFS)
    else:
        if file is None:
            typer.echo("要么给出定义文件，要么用 --builtin 导入内置工具", err=True)
            raise typer.Exit(2)
        text = file.read_text(encoding="utf-8")
        if file.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            payload = yaml.safe_load(text)
        else:
            payload = _json.loads(text)
        definitions = defs_from_payload(payload)
    registry, database = _tool_registry(db, None)
    try:
        records = registry.register_many(definitions)
        for record in records:
            typer.echo(f"  {record.name:<24} v{record.version} {record.kind} "
                       f"{record.side_effect} tokens={record.tokens} hash={record.hash}")
        typer.echo(f"已导入 {len(records)} 个工具")
    finally:
        database.close()


@tools_app.command("ls")
def tools_ls(db: Path = typer.Option(None, "--db")) -> None:
    """列出已注册工具。"""
    from rich.console import Console
    from rich.table import Table

    registry, database = _tool_registry(db, None)
    try:
        records = registry.repo.list_defs()
        table = Table(title=f"工具（{len(records)}）", pad_edge=False)
        for column in ("工具", "版本", "类型", "副作用", "tokens", "bytes", "启用", "hash"):
            table.add_column(column)
        for record in records:
            table.add_row(
                record.name, f"v{record.version}", record.kind, record.side_effect,
                str(record.tokens if record.tokens is not None else "—"),
                str(record.bytes if record.bytes is not None else "—"),
                "✓" if record.enabled else "✗", record.hash,
            )
        Console().print(table)
    finally:
        database.close()


@tools_app.command("audit")
def tools_audit(
    db: Path = typer.Option(None, "--db"),
    model: str = typer.Option(None, "--model", help="用该模型的标定档位核算描述预算"),
) -> None:
    """契约审计：逐条规则列出问题与修法。"""
    from rich.console import Console
    from rich.table import Table

    registry, database = _tool_registry(db, model)
    try:
        results = registry.audit_all()
        table = Table(title="工具契约审计", pad_edge=False)
        for column in ("工具", "规则", "级别", "问题", "修法"):
            table.add_column(column)
        counts = {"error": 0, "warn": 0, "info": 0}
        for name, findings in results.items():
            for finding in findings:
                counts[str(finding.severity)] = counts.get(str(finding.severity), 0) + 1
                table.add_row(name, finding.rule, str(finding.severity), finding.message, finding.fix)
        Console().print(table)
        typer.echo(f"error={counts['error']} warn={counts['warn']} info={counts['info']}")
        if counts["error"]:
            raise typer.Exit(1)
    finally:
        database.close()


@tools_app.command("cost")
def tools_cost(
    db: Path = typer.Option(None, "--db"),
    model: str = typer.Option(None, "--model", help="按该模型的计数档位核算（推荐）"),
    overhead: int = typer.Option(0, "--overhead", help="模板脚手架开销，取 trace 归因的 template_ctl"),
) -> None:
    """工具库上下文开销：JSON 本身 + 模板脚手架。

    P17：只报 JSON 会把优化方向引到"精简描述"，而实测 73% 的开销来自模板注入的说明文本。
    """
    from rich.console import Console
    from rich.table import Table

    registry, database = _tool_registry(db, model)
    try:
        report = registry.cost(template_overhead=overhead)
        table = Table(title=f"工具库开销 · count_source={report['count_source']}", pad_edge=False)
        for column in ("工具", "tokens", "bytes", "类型", "副作用"):
            table.add_column(column, justify="right" if column in {"tokens", "bytes"} else "left")
        for item in report["tools"]:
            table.add_row(item["name"], str(item["tokens"]), str(item["bytes"]),
                          item["kind"], item["side_effect"])
        Console().print(table)
        typer.echo(f"JSON 本身      : {report['json_tokens']} token / {report['json_bytes']} bytes")
        typer.echo(f"模板脚手架     : {report['template_overhead_tokens']} token"
                   f"{'（用 --overhead 传入 trace 归因的 template_ctl）' if not overhead else ''}")
        typer.echo(f"每次请求实付   : {report['effective_tokens']} token")
        if report["template_share"] is not None:
            typer.echo(f"模板占比       : {report['template_share']:.1%}"
                       "  ← 换模板/换模型比精简描述更有效")
        if report["count_source"] == "heuristic":
            typer.echo("[!] 用的是 heuristic 档（未标定/未指定模型），"
                       "绝对值仅供比较，加 --model 可得到标定后的数字")
    finally:
        database.close()


#: mock 执行器结构上不触达真实实现，deadline 无从生效——豁免必须写明原因（不许静默跳过）
_MOCK_CONTRACT_EXEMPTIONS = {
    "timeout_is_reported": (
        "mock 执行器不导入也不调用真实实现，deadline 无从生效；"
        "这条保证由 python_fn 上的同名断言覆盖"
    )
}


def _http_contract_target():
    """http 列的离线契约样本（MockTransport + `.invalid` 域名，零真实网络）。

    惰性导入：httpx 属于 `runtime` extra，没装时这一列必须显示"未安装"，
    而不是让整个 `tools contract` 命令崩掉。
    """
    try:
        from onyx.tools.executors.http import contract_target
    except ImportError:  # pragma: no cover - 取决于安装的 extras
        return None
    return contract_target()


def _resolve_tool_def(tool: str, db: Path | None):
    """查定义：**注册表优先，内置兜底**。返回 (定义, 出处, 需要关闭的 Database 或 None)。

    顺序不能反。用户显式 `tools import` 进来的定义才是模型实际会看到的那一份；
    若内置定义抢先命中，`tools run` 就会执行与注册版本不同的实现——
    这正是本项目要避免的"数字对不上出处"。
    """
    from onyx.tools.builtin.defs import builtin_def

    registry, database = _tool_registry(db, None)
    registered = registry.get(tool)
    if registered is not None:
        return registered, "registry", database
    database.close()
    return builtin_def(tool), "builtin", None


@tools_app.command("contract")
def tools_contract(
    tool: str = typer.Option("echo", "--tool", help="python_fn/mock 两列用的样本工具"),
    args: str = typer.Option(None, "--args", help="JSON 对象，覆盖自动构造的合法参数"),
    db: Path = typer.Option(None, "--db"),
    json_out: bool = typer.Option(False, "--json", help="输出机器可读结果"),
) -> None:
    """执行器契约矩阵：同一套断言跑在所有已实现的执行器上。

    "可替换"不是文档里的一句话，而是这张矩阵逼出来的。
    断言比对的是**失败的种类**（arg_error/rejected/timeout/unknown_tool/skipped/error），
    种类混淆就等于放弃归因能力。

    http 列始终用离线 MockTransport 样本（`contract.invalid` 是 RFC 2606 保留的
    永不解析域名），所以这条命令不发任何真实网络请求。
    """
    import json as _json

    from rich.console import Console
    from rich.table import Table

    from onyx.tools.builtin.defs import BUILTIN_DEFS, CONTRACT_SAMPLE_ARGS
    from onyx.tools.contract import CONTRACT_NAMES, run_contracts, sample_args, summarize
    from onyx.tools.executors import MockReplayExecutor, PythonFnExecutor

    definition, source, database = _resolve_tool_def(tool, db)
    try:
        if definition is None:
            names = ", ".join(item.name for item in BUILTIN_DEFS)
            typer.echo(f"找不到工具 {tool!r}；内置可选: {names}，或用 tools import 先注册", err=True)
            raise typer.Exit(2)
        if args:
            valid_args = _json.loads(args)
        elif definition.name == "echo" and source == "builtin":
            valid_args = dict(CONTRACT_SAMPLE_ARGS)
        else:
            valid_args = sample_args(definition)

        # 每一列是 (标签, 样本定义, 工厂, 合法参数, fixtures, 豁免, synth)
        targets: list[tuple] = [
            ("python_fn", definition, PythonFnExecutor, valid_args, None, {}, None),
            ("mock", definition, MockReplayExecutor, valid_args,
             {definition.name: {"__contract_mock__": True, "tool": definition.name}},
             _MOCK_CONTRACT_EXEMPTIONS, None),
        ]
        http = _http_contract_target()
        unavailable: dict[str, str] = {}
        if http is None:
            unavailable["http"] = "未安装 httpx（uv sync --extra runtime）——未知，不是通过"
        else:
            http_sample, http_factory, http_args, http_synth = http
            targets.append(("http", http_sample, http_factory, http_args, None, {}, http_synth))
        pending = {"mcp": "S16", "ollama_builtin": "S16"}

        matrix: dict[str, dict[str, object]] = {}
        samples: dict[str, str] = {}
        for name, sample, factory, target_args, fixtures, exemptions, synth in targets:
            results = run_contracts(
                factory, sample, valid_args=target_args,
                fixtures=fixtures, exemptions=exemptions, synth=synth,
            )
            matrix[name] = {item.name: item for item in results}
            samples[name] = sample.name

        if json_out:
            typer.echo(_json.dumps({
                "tool": definition.name, "source": source, "valid_args": valid_args,
                "assertions": list(CONTRACT_NAMES),
                "samples": samples,
                "executors": {
                    name: {
                        item: {
                            "passed": matrix[name][item].passed,
                            "applicable": matrix[name][item].applicable,
                            "detail": matrix[name][item].detail,
                        }
                        for item in CONTRACT_NAMES
                    }
                    for name in matrix
                },
                "unavailable": unavailable,
                "pending": pending,
                "summary": {name: summarize(list(matrix[name].values())) for name in matrix},
            }, ensure_ascii=False, indent=2, default=str))
            return

        console = Console()
        table = Table(title="执行器契约矩阵", pad_edge=False)
        table.add_column("断言", style="bold")
        for name in matrix:
            table.add_column(name, justify="center")
        for name in (*unavailable, *pending):
            table.add_column(name, justify="center", style="dim")
        for item in CONTRACT_NAMES:
            row = [item]
            for name in matrix:
                result = matrix[name][item]
                row.append("✓" if result.passed else ("n/a" if not result.applicable else "✗"))
            row.extend("—" for _ in range(len(unavailable) + len(pending)))
            table.add_row(*row)
        console.print(table)

        typer.echo("样本出处（每一列测的定义可能不同，必须写清楚）：")
        for name, sample_name in samples.items():
            note = "（注册表/内置）" if name in {"python_fn", "mock"} else "（离线 MockTransport）"
            typer.echo(f"  {name:<12} {sample_name} {note}")

        failed = 0
        for name in matrix:
            counts = summarize(list(matrix[name].values()))
            failed += counts["failed"]
            typer.echo(
                f"{name:<12} 通过 {counts['passed']} · 失败 {counts['failed']} · "
                f"不适用 {counts['not_applicable']}"
            )
        for name in matrix:
            for item in CONTRACT_NAMES:
                result = matrix[name][item]
                if not result.passed:
                    prefix = "不适用" if not result.applicable else "失败"
                    typer.echo(f"  [{name}] {prefix} {item}: {result.detail}")
        for name, reason in unavailable.items():
            typer.echo(f"  [{name}] {reason}")
        for name, milestone in pending.items():
            typer.echo(f"  [{name}] 未实现，计划在 {milestone}")
        if failed:
            raise typer.Exit(1)
    finally:
        if database is not None:
            database.close()


@tools_app.command("run")
def tools_run(
    tool: str = typer.Argument(..., help="工具名（内置或已注册）"),
    args: str = typer.Option("{}", "--args", help="JSON 对象形式的调用参数"),
    db: Path = typer.Option(None, "--db"),
    mock: str = typer.Option("live", "--mock", help="live | fixture | replay | deny"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只允许 read 类工具"),
    deadline: int = typer.Option(None, "--deadline-ms", help="覆盖定义里的超时"),
    allow: str = typer.Option(
        None, "--allow", help="额外放开的副作用，逗号分隔：write,network,exec"
    ),
    fixture: str = typer.Option(None, "--fixture", help="mock=fixture 时的返回值 JSON"),
) -> None:
    """不经模型直接调用一次工具，打印结构化结果。

    走的是与模型调用**完全相同**的 `guarded_call` 管线，所以这里能看到
    模型触发时会得到的同一种 error_kind——排查"工具调不对"时先看这条，
    能立刻分开是工具的问题还是模型的问题。
    """
    import json as _json

    from onyx.tools.executor import MockPolicy, ToolCtx
    from onyx.tools.executors import executor_for

    definition, source, database = _resolve_tool_def(tool, db)
    try:
        if definition is None:
            typer.echo(f"找不到工具 {tool!r}（注册表与内置定义里都没有）", err=True)
            raise typer.Exit(2)
        call_args = _json.loads(args)
        policy = _policy_from_cli(allow)
        stubs = {tool: _json.loads(fixture)} if fixture else {}
        mock_policy = MockPolicy(mock)
        ctx = ToolCtx(
            mock_policy=mock_policy, dry_run=dry_run, deadline_ms=deadline,
            fixtures=stubs, replay=stubs, policy=policy,
        )
        # 非 live 一律走 mock 执行器：它结构上不可能触达真实实现，
        # 比"相信 python_fn 会短路"要可靠得多
        kind = None if mock_policy is MockPolicy.LIVE else "fixture"
        executor = executor_for(definition, kind=kind, responses=stubs or None)
        result = executor.call(tool, call_args, ctx)

        typer.echo(f"工具      : {definition.name} v{definition.version} "
                   f"({definition.kind}, {definition.side_effect}) 出处={source}")
        typer.echo(f"执行器    : {type(executor).__name__} mock={mock} dry_run={dry_run}")
        typer.echo(f"结果      : {'✓ ok' if result.ok else '✗ ' + (result.error_kind or 'error')}"
                   f"{'（mocked）' if result.mocked else ''}")
        if result.ok:
            typer.echo(f"输出      : {_json.dumps(result.output, ensure_ascii=False, default=str)[:800]}")
            typer.echo(f"字节      : {result.bytes}")
        else:
            typer.echo(f"错误      : {result.error}")
            if result.extra.get("detail"):
                typer.echo(f"细节      : {_json.dumps(result.extra['detail'], ensure_ascii=False, default=str)}")
        typer.echo(f"耗时      : {result.extra.get('latency_ms', '—')} ms")
        if not result.ok:
            raise typer.Exit(1)
    finally:
        if database is not None:
            database.close()


@tools_app.command("fire")
def tools_fire(
    instruction: str = typer.Argument(..., help="给模型的指令"),
    model: str = typer.Option(..., "--model"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
    provider: str = typer.Option("ollama", "--provider", help="ollama | mock（离线跑通整条链路）"),
    tools: str = typer.Option(None, "--tools", help="逗号分隔的工具名；不给就用全部已启用的"),
    expect: str = typer.Option(None, "--expect", help="期望调用的工具名；默认取 --tools 的第一个"),
    expect_args: str = typer.Option(None, "--expect-args", help="期望参数 JSON；默认取该工具 examples[0]"),
    exact: bool = typer.Option(False, "--exact", help="严格档：不允许多余字段"),
    mock: str = typer.Option(
        "fixture", "--mock", help="live | fixture | replay | deny。默认 fixture：评测期零真实副作用"
    ),
    fixture: str = typer.Option(None, "--fixture", help='桩返回值 JSON：{"工具名": {...}}'),
    max_steps: int = typer.Option(6, "--max-steps"),
    total_tokens: int = typer.Option(None, "--max-total-tokens", help="各步 in+out 之和的上限"),
    deadline: int = typer.Option(None, "--deadline-ms", help="单次工具调用的超时"),
    allow: str = typer.Option(None, "--allow", help="额外放开的副作用：write,network,exec"),
    thinking: bool = typer.Option(False, "--thinking/--no-thinking"),
    max_new_tokens: int = typer.Option(512, "--max-new-tokens"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """模型侧 fire-and-verify：一条指令 → 期望的调用 → 实际发生的调用。

    判定分六种，因为它们的修法完全不同（详见 verify.py 的表）：
    NO_CALL 改提示词/工具描述 · WRONG_TOOL 改工具之间的区分度 ·
    BAD_ARGS 改参数描述或加 max_tokens · **TOOL_FAILED 改工具，不是改模型** ·
    LOOP_BROKEN 改工具返回值 · ERROR 先跑 onyx doctor。

    默认 `--mock fixture`：不给桩就不执行真实工具（报 skipped），
    这样同一条指令重跑多少次都可复现。要真跑加 `--mock live`。
    """
    import json as _json

    from rich.console import Console

    from onyx.core.types import GenerationRequest, GenParams, TraceContext, TracePurpose
    from onyx.runtime import sync_models
    from onyx.store.repos import ToolRepo
    from onyx.tools.executor import MockPolicy, ToolCtx
    from onyx.tools.loop import LoopBudget, ToolLoop
    from onyx.tools.registry import ToolRegistry
    from onyx.tools.verify import Verdict, verify_case

    console = Console()
    runtime = _runtime(url, db, provider_kind=provider)
    try:
        sync_models(runtime)
        registry = ToolRegistry(ToolRepo(runtime.db))
        wanted = [t.strip() for t in tools.split(",")] if tools else None
        definitions = _fire_definitions(registry, wanted)
        if not definitions:
            typer.echo(
                "没有可用的工具定义。先跑 `onyx tools import --builtin`，"
                "或用 --tools 指定已注册的名字。", err=True,
            )
            raise typer.Exit(2)

        expected_tool = expect or (wanted[0] if wanted else definitions[0].name)
        definition = next((d for d in definitions if d.name == expected_tool), None)
        expectation = _fire_expectation(definition, expected_tool, expect_args, exact)
        if expectation is None:
            raise typer.Exit(2)

        stubs = _json.loads(fixture) if fixture else {}
        ctx = ToolCtx(
            mock_policy=MockPolicy(mock), deadline_ms=deadline,
            fixtures=stubs, replay=stubs, policy=_policy_from_cli(allow),
        )
        loop = ToolLoop(
            runtime.gateway, definitions,
            budget=LoopBudget(max_steps=max_steps, max_total_tokens=total_tokens),
            ctx=ctx,
        )
        request = GenerationRequest.of(
            model, instruction,
            params=GenParams(max_tokens=max_new_tokens, temperature=0.0),
            thinking=thinking, tools=tuple(d.to_spec() for d in definitions),
            context=TraceContext(purpose=TracePurpose.TOOL_TEST),
        )
        result = verify_case(loop, expectation, request)
        runtime.flush()

        if json_out:
            typer.echo(_json.dumps({
                "instruction": instruction, "model": model,
                "expected": {"tool": expectation.tool, "arguments": dict(expectation.arguments),
                             "exact": expectation.exact},
                "verdict": str(result.verdict), "detail": result.detail,
                "called": list(result.called), "actual_args": result.actual_args,
                "args_diff": result.args_diff, "steps": result.steps,
                "stop_reason": result.stop_reason, "codes": list(result.codes),
                "mocked": result.mocked,
            }, ensure_ascii=False, indent=2, default=str))
            raise typer.Exit(0 if result.ok else 1)

        typer.echo(f"指令      : {instruction}")
        typer.echo(f"模型      : {model}   工具: {', '.join(d.name for d in definitions)}")
        typer.echo(f"期望      : {expectation.tool} "
                   f"{_json.dumps(dict(expectation.arguments), ensure_ascii=False)}"
                   f"{'（严格档）' if expectation.exact else ''}")
        typer.echo(f"实际      : {', '.join(result.called) or '（无调用）'}"
                   f"{'' if result.actual_args is None else ' ' + _json.dumps(result.actual_args, ensure_ascii=False)}")
        if result.args_diff:
            typer.echo(f"参数差异  : {_json.dumps(result.args_diff, ensure_ascii=False, default=str)}")
        typer.echo(f"步数      : {result.steps}   停止原因: {result.stop_reason}")
        if result.codes:
            typer.echo(f"异常码    : {', '.join(result.codes)}")
        label = {
            Verdict.PASS: "[green]PASS[/green]",
            Verdict.TOOL_FAILED: "[yellow]TOOL_FAILED[/yellow]",
            Verdict.ERROR: "[red]ERROR[/red]",
        }.get(result.verdict, f"[red]{str(result.verdict).upper()}[/red]")
        typer.echo(f"判定      : {label}")
        console.print(f"[dim]{result.detail}[/dim]")
        if result.mocked:
            typer.echo("[!] 工具用桩执行（--mock 非 live）：这验证的是模型会不会调，不是工具能不能跑")
        raise typer.Exit(0 if result.ok else 1)
    finally:
        runtime.close()


def _fire_definitions(registry, wanted: list[str] | None) -> list:
    """解析要参与本次验证的工具定义。

    与 `tools run` 同一条规矩：**注册表优先，内置兜底**。内置只是没注册时的便利，
    一旦用户导入过同名定义，模型看到的和执行的就必须是同一份。
    """
    from onyx.tools.builtin.defs import builtin_def

    if not wanted:
        return registry.list(enabled_only=True)
    out = []
    for name in wanted:
        definition = registry.get(name) or builtin_def(name)
        if definition is None:
            typer.echo(f"找不到工具 {name!r}（注册表与内置定义里都没有）", err=True)
            continue
        out.append(definition)
    return out


def _fire_expectation(definition, expected_tool: str, expect_args: str | None, exact: bool):
    """构造期望。没给 --expect-args 时从 `ToolDef.examples[0]` 取。

    examples 本来就是为 fire-and-verify 准备的（审计规则 NO_EXAMPLE 催的就是它），
    所以这里不需要用户再手抄一遍参数。
    """
    import json as _json

    from onyx.tools.verify import Expectation

    if expect_args:
        return Expectation("", expected_tool, _json.loads(expect_args), exact=exact)
    if definition is None:
        typer.echo(
            f"--expect {expected_tool!r} 不在本次工具集里，且没有 --expect-args 可用", err=True
        )
        return None
    if not definition.examples:
        typer.echo(
            f"工具 {expected_tool} 没有 examples，无法自动取期望参数；"
            "请用 --expect-args 显式给出，或给定义补一条 examples（审计规则 NO_EXAMPLE）",
            err=True,
        )
        return None
    expectation = Expectation.from_example(definition.examples[0], exact=exact)
    if expectation.tool != expected_tool:
        # examples[0] 里写的工具名与 --expect 不一致，说明定义自相矛盾
        typer.echo(
            f"[!] {expected_tool} 的 examples[0] 期望调用的是 {expectation.tool!r}，"
            "以 examples 为准", err=True,
        )
    return expectation


def _policy_from_cli(allow: str | None):

    """默认最严（只允许 read）。`--allow` 是显式的、逐次生效的放开。"""
    from onyx.tools.sandbox import PERMISSIVE_POLICY, SandboxPolicy
    from onyx.tools.spec import SideEffect

    if not allow:
        return SandboxPolicy()
    wanted = {piece.strip() for piece in allow.split(",") if piece.strip()}
    try:
        effects = {SideEffect(piece) for piece in wanted}
    except ValueError:
        typer.echo("--allow 取值非法（可选 read/write/network/exec）", err=True)
        raise typer.Exit(2) from None
    return SandboxPolicy(
        allowed_side_effects=frozenset({SideEffect.READ, *effects}),
        # CLI 是人在操作，--allow 本身就是审批动作；不再二次弹窗
        require_approval=frozenset(),
        allowed_impl_prefixes=PERMISSIVE_POLICY.allowed_impl_prefixes,
    )


# ── eval ─────────────────────────────────────────────────────────
eval_app = typer.Typer(help="评测：数据集、任务、运行与分数下钻", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


def _eval_runtime(url: str, db: Path | None, provider_kind: str = "ollama"):
    """评测跑在真实 runtime 上：分数与 trace 共用同一套存储与观测。"""
    return _runtime(url, db, sample_gpu=provider_kind == "ollama", provider_kind=provider_kind)


@eval_app.command("ls")
def eval_ls(
    db: Path = typer.Option(None, "--db"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """列出数据集、内置任务与最近的运行。"""
    from rich.console import Console
    from rich.table import Table

    from onyx.eval.tasks import task_ids
    from onyx.store.repos import EvalRepo

    console = Console()
    settings = _settings()
    database = Database(_db_path(settings, db))
    try:
        repo = EvalRepo(database)
        datasets = repo.list_datasets()
        table = Table(title=f"数据集（{len(datasets)}）", pad_edge=False)
        for column in ("id", "条数", "来源", "revision", "子集"):
            table.add_column(column)
        for item in datasets:
            table.add_row(
                item.id, str(item.n_cases if item.n_cases is not None else "—"),
                item.upstream or "—", item.revision or "—",
                ", ".join(f"{k}:{v}" for k, v in item.splits.items()) or "—",
            )
        console.print(table)

        typer.echo(f"内置任务: {', '.join(task_ids())}")

        runs = repo.list_runs(limit=limit)
        run_table = Table(title=f"最近 {len(runs)} 次运行", pad_edge=False)
        for column in ("run_id", "任务", "模型", "状态", "n", "主分数", "开始"):
            run_table.add_column(column)
        for run in runs:
            run_table.add_row(
                run.id, run.task_id, run.model_id, run.status, str(run.n_cases),
                _headline_score(run.aggregate), run.started_at,
            )
        console.print(run_table)
    finally:
        database.close()


#: 主分数的候选指标，按优先级。**只显示任务声明过的第一个指标**，
#: 绝不因为它是 None 就悄悄换成另一个——那会让"macro_f1 未知"显示成"得了 0 分"，
#: 而这两个结论的修法完全相反（前者是根本没有可判定样本，后者是模型不行）
_HEADLINE_METRICS = ("macro_f1", "accuracy", "must_call_acc", "pass_hat_k", "score")


def _headline_score(aggregate: dict) -> str:
    """列表页只显示一个数，但必须**带上指标名**，且未知就显示「—」。"""
    if not aggregate:
        return "—"
    if aggregate.get("skip"):
        return "skipped"
    for key in _HEADLINE_METRICS:
        if key not in aggregate:
            continue
        value = aggregate.get(key)
        if value is None:
            return f"{key} —（无可判定样本）"
        text = f"{key} {_fmt(value)}"
        # 区间按「指标名 + _ci」找，不写死 macro_f1：头号指标一换（工具任务是
        # must_call_acc），写死的版本就会把区间整个丢掉，只剩一个孤零零的数
        ci = aggregate.get(f"{key}_ci")
        low, high = _ci_get(ci, "low"), _ci_get(ci, "high")
        if low is not None and high is not None:
            text += f" [{_fmt(low)}–{_fmt(high)}]"
        if aggregate.get("low_confidence"):
            text += " ⚠低样本"
        return text
    return "—"


@eval_app.command("import")
def eval_import(
    file: Path = typer.Argument(None, exists=True, dir_okay=False, readable=True),
    id: str = typer.Option(None, "--id", help="数据集 id；默认 JSONL 取文件名、bfcl 取 bfcl-<子集>"),
    builtin: str = typer.Option(None, "--builtin", help="导入内置数据集，例如 intent_zh"),
    source: str = typer.Option("auto", "--source", help="auto | jsonl | bfcl"),
    answers: Path = typer.Option(
        None, "--answers", exists=True, dir_okay=False, readable=True,
        help="bfcl：答案文件，与问题文件按 id 配对（不按行号）"
    ),
    subset: str = typer.Option("v1", "--subset", help="bfcl 子集名，写进 upstream 和 tags"),
    db: Path = typer.Option(None, "--db"),
    upstream: str = typer.Option("", "--upstream"),
    revision: str = typer.Option("", "--revision"),
    license: str = typer.Option("", "--license", help="上游许可证；转载数据集必须记"),
) -> None:
    """导入数据集（JSONL、内置生成器或 BFCL 风格文件），并把来历一并落库。

    来历（upstream/revision/license）不是元数据装饰：换了数据集版本之后
    两次评测的分数不可比，不记 revision 就永远发现不了这件事。

    外部数据源只支持"从本地文件导入"，不替你下载：下载会在评测路径上引入网络依赖，
    于是"离线复现一次评测"就做不到了，而这正是本地评测相对云 API 的主要优势。
    """
    from onyx.eval.datasets.loader import DatasetError, load_builtin, load_jsonl
    from onyx.eval.datasets.sources import describe, import_bfcl, supported
    from onyx.store.repos import EvalRepo

    # 参数组合错了要报错，不能静默挑一个：`--builtin` 配 `--source bfcl` 如果被
    # 忽略，使用者会以为自己导入的是 BFCL，而实际导入的是内置集
    if builtin and (answers is not None or source not in ("auto", "jsonl")):
        typer.echo("--builtin 自带加载器，与 --source/--answers 互斥", err=True)
        raise typer.Exit(2)
    if answers is not None and source not in ("auto", "bfcl"):
        typer.echo("--answers 只在 bfcl 源有意义（BFCL 的问题与答案分两个文件）", err=True)
        raise typer.Exit(2)
    if source == "auto":
        source = "bfcl" if answers is not None else "jsonl"

    try:
        if not builtin:
            supported(source=source)
        if builtin:
            dataset = load_builtin(builtin, dataset_id=id, upstream=upstream or None,
                                   revision=revision or None, license=license or None)
        elif file is not None:
            dataset = (
                import_bfcl(file, answers, dataset_id=id or f"bfcl-{subset}",
                            subset=subset, upstream=upstream, revision=revision,
                            license=license)
                if source == "bfcl" else
                load_jsonl(file, dataset_id=id, upstream=upstream or None,
                           revision=revision or None, license=license or None)
            )
        else:
            typer.echo("要么给出数据集文件，要么用 --builtin intent_zh", err=True)
            raise typer.Exit(2)
    except DatasetError as exc:
        typer.echo(f"数据集错误: {exc}", err=True)
        raise typer.Exit(1) from None

    settings = _settings()
    database = Database(_db_path(settings, db))
    try:
        repo = EvalRepo(database)
        record, cases = dataset.to_records()
        repo.upsert_dataset(record)
        repo.upsert_cases(cases)
        typer.echo(describe(dataset))
        # notes 里装着"跳过了多少条"这类信息；不打印的话样本变少这件事就没人知道
        if dataset.notes:
            typer.echo(f"  {dataset.notes}")
        for split, count in sorted(dataset.splits().items()):
            typer.echo(f"  子集 {split:<12} {count} 条")
    finally:
        database.close()


@eval_app.command("run")
def eval_run(
    task: str = typer.Option("intent_classification", "--task"),
    model: str = typer.Option(..., "--model"),
    url: str = typer.Option("http://127.0.0.1:11434", "--url"),
    db: Path = typer.Option(None, "--db"),
    provider: str = typer.Option("ollama", "--provider", help="ollama | mock（离线跑通管道）"),
    dataset: str = typer.Option(None, "--dataset", help="数据集 id 或 file:<路径>；默认用任务自带的"),
    k: int = typer.Option(1, "--k", help="每条样本采样次数（pass^k / pass@k）"),
    limit: int = typer.Option(None, "--limit", help="只跑前 N 条"),
    split: str = typer.Option("default", "--split", help="子集，例如 hard"),
    seed: int = typer.Option(None, "--seed"),
    resume: str = typer.Option(None, "--resume", help="续跑指定 run_id，跳过已评过的 case"),
    max_wall_ms: float = typer.Option(None, "--max-wall-ms"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="覆盖任务的生成预算"),
    gpu_lock_path: Path = typer.Option(
        None, "--gpu-lock", help="GPU 锁文件路径；默认用机器级路径，多实例才会互斥"
    ),
    no_queue: bool = typer.Option(
        False, "--no-queue", help="拿不到 GPU 锁就直接失败，不排队等"
    ),
    lock_timeout: float = typer.Option(
        None, "--lock-timeout", help="排队等锁的最长秒数；不给就一直等"
    ),
    unload_others: bool = typer.Option(
        False, "--unload-others",
        help="开跑前卸掉其它已载入模型，避免 size_vram 叠加触发 CPU offload"
    ),
    json_out: bool = typer.Option(False, "--json"),
    quiet: bool = typer.Option(False, "--quiet", help="不打进度"),
) -> None:
    """跑一次评测。请求全部走 gateway，所以**每个分数都能点进一条真实 trace**。

    默认拿机器级 GPU 锁排队（DESIGN §8.5）：单 GPU 上两个评测同时跑，
    现象不是报错而是数字被污染——两个模型同时驻留会触发 CPU offload，
    吞吐差一个数量级却看起来"正常"。这把锁是跨进程的文件锁，
    所以 `onyx serve` 的 Playground 也会与它互斥。

    能力不足时整个任务会被 skip 并写明原因，不做隐式降级：
    用提示词模拟工具调用得到的分数，无法与原生支持的模型比较，
    却看不出区别——那比没有分数更糟。
    """
    import json as _json

    from rich.console import Console

    from onyx.core.errors import GpuLockBusy
    from onyx.eval.gpu_lock import GpuLock, default_lock_path
    from onyx.eval.metrics import jsonable
    from onyx.eval.runner import EvalRunner, RunConfig, wait_for
    from onyx.eval.tasks import build_task, load_dataset
    from onyx.store.repos import EvalRepo

    console = Console()
    try:
        loaded = load_dataset(dataset, task_id=task)
        overrides = {"max_tokens": max_tokens} if max_tokens else {}
        instance = build_task(task, model=model, dataset=loaded, **overrides)
    except KeyError as exc:
        typer.echo(str(exc).strip("'"), err=True)
        raise typer.Exit(2) from None

    runtime = _eval_runtime(url, db, provider)
    try:
        repo = EvalRepo(runtime.db)
        progress = None
        if not quiet and not json_out:
            def progress(done: int, total: int, case_id: str, grade) -> None:
                typer.echo(f"\r  {done}/{total}  {case_id[:18]}  {grade.verdict}", nl=False)

        # 默认用机器级路径：GPU 是整台机器一块，锁跟着可覆盖的数据目录走就锁不住多实例
        lock_path = Path(gpu_lock_path) if gpu_lock_path else default_lock_path()
        lock = GpuLock(
            lock_path, owner=f"eval:{task}@{model}",
            stale_after_s=600.0,  # 必须大于单条样本的最长耗时，否则会误伤活着的持有者
        )
        runner = EvalRunner(
            runtime.gateway, repo, instance, dataset=loaded,
            on_progress=wait_for(progress) if progress else None, gpu_lock=lock,
        )
        config = RunConfig(
            model=model, k=k, seed=seed, limit=limit, split=split,
            resume_run_id=resume, max_wall_ms=max_wall_ms,
            lock_timeout=0.0 if no_queue else lock_timeout, unload_others=unload_others,
        )
        if not quiet and not json_out:
            info = lock.peek()
            # 用 is_busy 而不是"锁文件存在"：崩掉的持有者会留下一个心跳过期的文件，
            # 那时并没有人在占 GPU，说"当前由 X 占用，排队中"会让人等一个不会来的释放
            if info is not None and lock.is_busy():
                verdict = "不排队，直接失败" if no_queue else "排队中…"
                typer.echo(f"[i] GPU 当前由 {info.owner} 占用（进度 {info.progress}），{verdict}")
        try:
            report = runner.run(config)
        except GpuLockBusy as exc:
            typer.echo(str(exc), err=True)
            eta = exc.detail.get("eta_s")
            # ETA 可能是 None（持有者还没开始跑），这时要写"未知"而不是 0s：
            # "预计还需 0s"会被读成"马上就轮到我"，然后人会一直等下去
            typer.echo(
                f"[i] 详情: {exc.detail.get('owner')} 进度 {exc.detail.get('progress')} "
                f"预计还需 {'未知' if eta is None else f'{eta:.0f}s'}；"
                "不想排队可以用 --no-queue 立刻失败，或 --gpu-lock 换一个锁文件",
                err=True,
            )
            raise typer.Exit(3) from None
        runtime.flush()
        if progress:
            typer.echo("")

        if json_out:
            typer.echo(_json.dumps({
                "run_id": report.run_id, "task": report.task_id, "model": report.model,
                "status": report.status, "skip_reason": report.skip_reason,
                "n_cases": report.n_cases, "n_done": report.n_done,
                "n_error": report.n_error, "cost": report.cost,
                "aggregate": jsonable(report.aggregate),
            }, ensure_ascii=False, indent=2, default=str))
            raise typer.Exit(0 if report.status == "done" else 1)

        _print_run_report(console, report, instance, k=k, seed=seed, split=split)
        raise typer.Exit(0 if report.status == "done" else 1)
    finally:
        runtime.close()


#: (指标, 分组)。分组只是显示用的；聚合里没有的指标就不打印，
#: 所以新增任务不必改这里——但指标名要与 `EvalTask.metric_names` 对得上
_REPORT_METRICS = (
    ("macro_f1", "内容"), ("accuracy", "内容"), ("balanced_accuracy", "内容"),
    ("must_call_acc", "内容"),
    ("hit_at_1", "选择"), ("set_f1", "选择"),
    ("no_call_rate", "选择"), ("wrong_tool_rate", "选择"),
    ("false_call_rate", "选择"), ("refusal_rate", "选择"),
    ("args_exact_rate", "参数"), ("args_subset_rate", "参数"),
    ("args_field_rate", "参数"), ("args_relaxed_share", "参数"),
    ("format_valid_rate", "格式"), ("invalid_format_rate", "格式"),
    ("out_of_label_rate", "格式"), ("hallucinated_tool_rate", "格式"),
    ("parse_fail_rate", "格式"),
    ("pass_hat_k", "稳定性"), ("pass_at_k", "稳定性"), ("stability_gap", "稳定性"),
)


def _print_run_report(console, report, task, *, k: int, seed: int | None, split: str) -> None:
    """人读的报告。

    刻意把**内容维度**与**格式维度**分开打印（DESIGN §9.4）：
    API-only 只能生成式打分，模型会因为"输出格式不听话"额外掉分，
    混成一个正确率就会把格式问题误读成能力问题，而两者修法完全相反。
    """

    aggregate = report.aggregate
    if report.status == "skipped":
        console.print(f"[yellow]SKIPPED[/yellow] {report.task_id} · {report.model}")
        console.print(f"  原因: {report.skip_reason}")
        return

    header = (
        f"{report.task_id} · {report.model} · n={report.n_cases} · "
        f"split={split} · k={k}" + (f" · seed={seed}" if seed is not None else "")
        + (f" · [yellow]{report.status}[/yellow]" if report.status != "done" else "")
    )
    console.print(f"[bold]{header}[/bold]")
    if aggregate.get("resumed"):
        console.print(
            f"[dim]续跑自 {aggregate.get('already_graded_before')} 条已有 grade，"
            "聚合包含全部样本[/dim]"
        )

    # 只打印这次评测**真的产出了**的指标。
    # 这里原本硬编码了 intent 任务的 macro_f1/acc/format_valid，
    # 于是 tool_selection 的报告整行全是「—」——看着像评测坏了，
    # 实际是报告模板与任务对不上，正是本项目最该避免的"数字对不上口径"
    groups: dict[str, list[str]] = {}
    for key, group in _REPORT_METRICS:
        if key not in aggregate:
            continue
        text = f"{key} {_fmt(aggregate.get(key))}"
        # CI 跟着指标名走，而不是只对 macro_f1 特判：任何指标算出了区间就必须看得见，
        # 否则 `must_call_acc` 这类头号指标会显示成一个孤零零的数（DESIGN §9.3）
        ci = aggregate.get(f"{key}_ci")
        low, high, units = _ci_get(ci, "low"), _ci_get(ci, "high"), _ci_get(ci, "n")
        if low is not None or high is not None:
            text += f" [95% CI {_fmt(low)}–{_fmt(high)}]"
            if units is not None:
                text += f"（n={units} case）"
        groups.setdefault(group, []).append(text)
    for group in ("内容", "选择", "参数", "格式", "稳定性"):
        if group in groups:
            console.print(f"  [{group}] " + "   ".join(groups[group]))
    if aggregate.get("low_confidence"):
        console.print("[yellow]  ⚠ 样本量低于 100，CI 只说明测过了，不足以支撑决策[/yellow]")
    console.print(f"  [dim]打分口径: {aggregate.get('scoring', '—')}"
                  "（API-only 拿不到受约束 logprob，分数不可与公开 leaderboard 直接比较）[/dim]")

    per_class = aggregate.get("per_class_f1") or {}
    if per_class:
        console.print("  逐类 F1: " + " / ".join(
            f"{label} {_fmt(value)}" for label, value in sorted(per_class.items())
        ))
    confusions = aggregate.get("top_confusions") or []
    if confusions:
        console.print("  混淆集中: " + ", ".join(
            f"{item['expected']}→{item['actual']} ×{item['count']}" for item in confusions
        ))
    if k > 1:
        console.print(
            f"  [稳定性] pass^{k} {_fmt(aggregate.get('pass_hat_k'))}"
            f"   pass@{k} {_fmt(aggregate.get('pass_at_k'))}"
            f"   缺口 {_fmt(aggregate.get('stability_gap'))}"
            "  ← 缺口大＝可用但不可靠"
        )

    verdicts = aggregate.get("verdicts") or {}
    if verdicts:
        console.print("  判定分布: " + " / ".join(
            f"{name} {count}" for name, count in sorted(verdicts.items(), key=lambda kv: -kv[1])
        ))
    cost = report.cost or {}
    unknown = cost.get("in_tokens_unknown") or 0
    console.print(
        f"  成本: in {cost.get('in_tokens', 0):,} tok · out {cost.get('out_tokens', 0):,} tok · "
        f"{cost.get('requests', 0)} 请求 · {cost.get('wall_ms', 0):,.0f} ms"
        + (f" · [yellow]{unknown} 条无计数[/yellow]" if unknown else "")
    )
    console.print(
        f"  run_id [bold]{report.run_id}[/bold]   "
        f"下钻: onyx eval show {report.run_id}"
    )


def _fmt(value, digits: int = 3) -> str:
    """未知显示「—」，绝不显示 0（UI_DESIGN R2）。"""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return str(value)


def _ci_get(ci, key: str):
    """读置信区间的一个端点。

    必须同时接受 `CI` 对象与 dict：刚跑完时它是内存里的 dataclass，
    经 `eval show` 从库里读回来则是 JSON 解码后的 dict。只支持一种的话，
    另一条路径会在运行时才炸。
    """
    if ci is None:
        return None
    if isinstance(ci, dict):
        return ci.get(key)
    return getattr(ci, key, None)


def _call_brief(value: object) -> str:
    """把期望/实际的调用列表压成一行。

    直接 `str()` 会把整份参数 dict 印进表格，一行放不下；而空列表必须显示成
    **（不调用）**而不是空白——空白看起来像"这个 grade 没有期望值"。
    """
    if value is None:
        return "—"
    if not isinstance(value, (list, tuple)):
        return str(value)
    if not value:
        return "（不调用）"
    parts: list[str] = []
    for item in value:
        if isinstance(item, dict) and item.get("name"):
            args = item.get("arguments") or {}
            tail = "" if not args else "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"
            parts.append(f"{item['name']}{tail}")
        else:
            parts.append(str(item))
    return " + ".join(parts)


@eval_app.command("show")
def eval_show(
    run_id: str = typer.Argument(..., help="eval run 输出的 run_id"),
    db: Path = typer.Option(None, "--db"),
    limit: int = typer.Option(20, "--limit", help="列出多少条 grade"),
    verdict: str = typer.Option(None, "--verdict", help="只看某个判定，例如 out_of_label"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """看一次运行的汇总与逐条 grade（含 trace_id，可直接跳进那条 trace）。"""
    import json as _json

    from rich.console import Console
    from rich.table import Table

    from onyx.store.repos import EvalRepo

    console = Console()
    settings = _settings()
    database = Database(_db_path(settings, db))
    try:
        repo = EvalRepo(database)
        run = repo.get_run(run_id)
        if run is None:
            typer.echo(f"找不到 run {run_id!r}", err=True)
            raise typer.Exit(2)
        grades = repo.list_grades(run_id, verdict=verdict, limit=limit)
        if json_out:
            typer.echo(_json.dumps({
                "run": {
                    "id": run.id, "task_id": run.task_id, "model_id": run.model_id,
                    "status": run.status, "started_at": run.started_at,
                    "finished_at": run.finished_at, "seed": run.seed,
                    "app_version": run.app_version, "git_rev": run.git_rev,
                    "params_snapshot": run.params_snapshot, "config": run.config,
                    "n_cases": run.n_cases, "n_done": run.n_done, "n_error": run.n_error,
                    "aggregate": run.aggregate, "cost": run.cost,
                },
                "grades": [
                    {"case_id": g.case_id, "seq": g.seq, "verdict": g.verdict,
                     "score": g.score, "passed": g.passed, "trace_id": g.trace_id,
                     "invalid_format": g.invalid_format, "out_of_set": g.out_of_set,
                     "metrics": g.metrics, "error": g.error}
                    for g in grades
                ],
            }, ensure_ascii=False, indent=2, default=str))
            return

        console.print(
            f"[bold]{run.task_id}[/bold] · {run.model_id} · {run.status} · "
            f"n={run.n_cases} done={run.n_done} error={run.n_error}"
        )
        console.print(
            f"[dim]seed={run.seed if run.seed is not None else '—'} "
            f"app={run.app_version or '—'} git={run.git_rev or '—'} "
            f"params={run.params_snapshot or '—'}[/dim]"
        )
        console.print(f"主分数: {_headline_score(run.aggregate)}")
        table = Table(title=f"grade（前 {len(grades)} 条）", pad_edge=False)
        for column in ("case", "seq", "判定", "分", "格式", "期望", "预测", "trace"):
            table.add_column(column)
        for grade in grades:
            metrics = grade.metrics or {}
            table.add_row(
                grade.case_id[:20], str(grade.seq), grade.verdict, f"{grade.score:.2f}",
                "✗" if grade.invalid_format else "✓",
                _call_brief(metrics.get("expected")),
                _call_brief(metrics.get("predicted", metrics.get("actual"))),
                (grade.trace_id or "—")[:12],
            )
        console.print(table)
        if grades:
            # 表里的 trace 是 12 个字符（列宽就那么多），跳进去要用完整 id
            first = next((g.trace_id for g in grades if g.trace_id), None)
            console.print(
                f"[dim]下钻: onyx traces show {first}[/dim]" if first else
                "[dim]这些 grade 没有关联 trace[/dim]"
            )
        if grade_errors := [g for g in grades if g.error]:
            console.print("[dim]错误样例:[/dim]")
            for grade in grade_errors[:3]:
                console.print(f"  {grade.case_id[:20]} [{grade.verdict}] {grade.error[:120]}")
    finally:
        database.close()


def _force_utf8_stdio() -> None:
    """Windows 中文环境下 stdout 默认 gbk，`✓ ✗ —` 这类字符会直接抛 UnicodeEncodeError。

    看板与 CLI 大量使用这些符号（UI_DESIGN R2：未知必须显示「—」而不是 0），
    所以在入口统一切到 UTF-8。管道重定向时同样生效——否则 `onyx tools contract | head`
    这种最普通的用法就会崩。
    """
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # 已关闭或被替换成不支持编码的流（例如测试里的替身）时静默跳过
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def main() -> None:
    _force_utf8_stdio()
    app()


if __name__ == "__main__":
    main()
