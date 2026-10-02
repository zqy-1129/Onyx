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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
