"""可验证的备份：`onyx db backup` / `db verify-backup`。

为什么这不是"cp 两个文件"：
- **WAL 模式下直接 cp `.sqlite` 是错的**。最近的事务可能还在 `-wal` 里，
  拷出来的库打开不报错、看起来一切正常，但少了最后一段数据。必须用 SQLite 的
  在线备份 API（`Connection.backup`）取一份一致快照。
- **只备库不算备过**。观测数据是"证据"，而证据在 blob 目录里：库里有引用、blob 没备份，
  恢复出来的是一个引用全是空洞的看板。所以备份装的是**被引用到的** blob
  （孤儿不占备份体积），verify 检查"备份库里的每个引用都能在备份里解析"。
- **备份会在没人看的时候坏**。所以 verify 不只看行数：每个 blob 都重算一遍 sha256
  与文件名比对——内容寻址在这里正好免费提供了一个完整性校验器。

`manifest.json` 记的是备份**当时**的样子：备完之后当前库又长了数据不该让 verify 失败，
但要能说出"备到哪儿了"（`drift`）。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.core.content import FileBlobStore
from onyx.core.errors import BlobNotFound, InvalidBlobRef
from onyx.store.db import Database
from onyx.store.retention import ALL_REF_COLUMNS, referenced_refs

DB_NAME = "onyx.sqlite"
BLOBS_NAME = "blobs"
MANIFEST_NAME = "manifest.json"
#: `schema_version` 的行数没有信息量（迁移条数是固定的），比对时跳过。
TABLES_EXCLUDED = frozenset({"schema_version"})


@dataclass(frozen=True, slots=True)
class BackupInfo:
    root: Path
    created_at: str
    schema_version: int
    tables: dict[str, int]
    blobs: int
    blob_bytes: int
    digest: str
    db_bytes: int
    unresolved: tuple[str, ...] = ()

    def as_manifest(self, *, app_version: str, source_db: str) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "app_version": app_version,
            "source_db": source_db,
            "schema_version": self.schema_version,
            "db_bytes": self.db_bytes,
            "tables": dict(self.tables),
            "blobs": self.blobs,
            "blob_bytes": self.blob_bytes,
            "digest": self.digest,
            # 备份当时就解析不了的引用：不拦备份，但必须留在清单里，
            # 否则"备份缺证据"会看起来像"备份做错了"，而真正坏的是更早的清理。
            "unresolved_refs": list(self.unresolved),
        }


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class Report:
    root: Path
    checks: tuple[Check, ...]
    #: 备份之后当前库多出来的东西（不是错误，是"备到哪儿了"）
    drift: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in self.checks],
            "drift": dict(self.drift),
        }


def _table_counts(db: Database) -> dict[str, int]:
    """每张表多少行。`schema_version` 跳过：迁移条数是固定的，比了没信息量。"""
    return {
        name: int(db.scalar(f"SELECT COUNT(*) FROM {name}", (), 0))
        for name in db.table_names()
        if name not in TABLES_EXCLUDED
    }


def _digest_of(refs: dict[str, int]) -> str:
    """对「哪些 ref、各多大」整体取摘要。

    不含 ref 内容：内容寻址已经让文件名等于摘要，逐字节校验在 verify 里做，
    清单只回答"备份当时是不是这个样子"。
    """
    lines = "\n".join(f"{ref}:{size}" for ref, size in sorted(refs.items()))
    return "sha256:" + hashlib.sha256(lines.encode("utf-8")).hexdigest()


def create_backup(
    db: Database, store: FileBlobStore, to: Path | str, *, app_version: str = ""
) -> BackupInfo:
    """把库与被引用的 blob 复制进 `to/`，并写一份 manifest。

    目标目录里已有备份时拒绝覆盖：备份被"顺手覆盖一次"等于丢掉一个恢复点，
    而这种事通常要等到真需要恢复的时候才被发现。
    """
    root = Path(to).resolve()
    if (root / MANIFEST_NAME).exists():
        raise FileExistsError(f"目录里已经有一份备份：{root}（换目录，或先确认要不要覆盖）")
    (root / BLOBS_NAME).mkdir(parents=True, exist_ok=True)

    db.backup_to(root / DB_NAME)

    copied: dict[str, int] = {}
    unresolved: list[str] = []
    for ref in sorted(referenced_refs(db)):
        try:
            stat = store.stat(ref)
        except (BlobNotFound, InvalidBlobRef):
            # 备份当时就悬空：继续把能备的备上，verify 会把它报成一条具名检查。
            unresolved.append(ref)
            continue
        target = _blob_path(root / BLOBS_NAME, ref)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stat.path, target)
        copied[ref] = stat.size

    with sqlite3.connect(root / DB_NAME) as conn:
        pages = int(conn.execute("PRAGMA page_count").fetchone()[0])
        page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])

    info = BackupInfo(
        root=root,
        created_at=utc_now_iso(),
        schema_version=db.version(),
        tables=_table_counts(db),
        blobs=len(copied),
        blob_bytes=sum(copied.values()),
        digest=_digest_of(copied),
        db_bytes=pages * page_size,
        unresolved=tuple(unresolved),
    )
    manifest = info.as_manifest(app_version=app_version, source_db=str(db.path))
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return info


def _blob_path(blobs_dir: Path, ref: str) -> Path:
    """备份目录沿用同样的两级分片布局：`blobs/ab/cd/<hex>`。

    ref 先严格校验才参与路径拼接——备份目录是要被人当恢复源用的，
    这里放过一个 `../` 等于给了一个任意写文件的入口。
    """
    if not ref.startswith("sha256:") or len(ref) != 71:
        raise ValueError(f"非法 blob ref，不参与备份路径拼接: {ref!r}")
    hexdig = ref[7:]
    if any(ch not in "0123456789abcdef" for ch in hexdig):
        raise ValueError(f"非法 blob ref（不是小写十六进制）: {ref!r}")
    return blobs_dir / hexdig[:2] / hexdig[2:4] / hexdig


def _refs_in_conn(conn: sqlite3.Connection) -> set[str]:
    """备份库里引用的 blob。列清单沿用 `retention.ALL_REF_COLUMNS`：同一份定义，
    避免"什么算引用"在两处各写一套而悄悄漂移。"""
    found: set[str] = set()
    for table, column in ALL_REF_COLUMNS:
        try:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM {table} WHERE {column} LIKE 'sha256:%'"
            ).fetchall()
        except sqlite3.Error:
            continue  # 表都不在了 ⇒ 结构已坏，前面的完整性检查会报出来
        found.update(str(row[0]) for row in rows)
    return found


def verify_backup(root: Path | str, *, live: Database | None = None) -> Report:
    """证明一份备份**可用**，而不只是"存在"。

    检查顺序：结构 → 库 → blob 内容 → 交叉一致性。备份坏掉时人要做的决定是
    "还来得及重做吗"，所以每条失败都得带具体数字与名字。
    以 `mode=ro` 打开备份：验证一个备份不该把它改写成 WAL 模式。
    """
    root = Path(root).resolve()
    checks: list[Check] = []
    manifest_path = root / MANIFEST_NAME
    db_path = root / DB_NAME
    blobs_dir = root / BLOBS_NAME

    present = ((MANIFEST_NAME, manifest_path.exists()), (DB_NAME, db_path.exists()),
               (BLOBS_NAME + "/", blobs_dir.is_dir()))
    missing = [name for name, ok in present if not ok]
    checks.append(Check(
        "备份结构完整", not missing,
        "manifest / 库 / blobs 齐备" if not missing else f"缺 {', '.join(missing)}",
    ))
    if missing:
        return Report(root=root, checks=tuple(checks))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        checks.append(Check("manifest 可解析", False, f"JSON 坏了：{exc}"))
        return Report(root=root, checks=tuple(checks))
    checks.append(Check("manifest 可解析", True, f"备于 {manifest.get('created_at', '?')}"))

    try:
        with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as conn:
            integrity = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
            try:
                version = int(
                    conn.execute("SELECT COALESCE(MAX(version),0) FROM schema_version").fetchone()[0]
                )
            except sqlite3.Error:
                version = -1
            recorded = dict(manifest.get("tables", {}))
            observed: dict[str, int] = {}
            for name in recorded:
                try:
                    observed[name] = int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
                except sqlite3.Error:
                    observed[name] = -1
            wanted = _refs_in_conn(conn)
    except sqlite3.DatabaseError as exc:
        # 库打不开时后面几条没有一条能被评估——写"未知"而不是让它们全绿。
        checks.append(Check(
            "库能打开且自检通过", False,
            f"读不了这份 sqlite 文件：{type(exc).__name__}: {exc}",
        ))
        return Report(root=root, checks=tuple(checks))
    checks.append(Check(
        "库能打开且自检通过", integrity == "ok",
        "ok" if integrity == "ok" else f"integrity_check={integrity}",
    ))
    checks.append(Check(
        "schema 版本与清单一致", version == int(manifest.get("schema_version", -1)),
        f"备份 v{version} · 清单 v{manifest.get('schema_version')}",
    ))
    mismatched = {n: (c, int(recorded[n])) for n, c in observed.items() if c != int(recorded[n])}
    checks.append(Check(
        "每张表的行数与清单一致", not mismatched,
        f"{len(observed)} 张表全部对上" if not mismatched else "; ".join(
            f"{n} 备份里 {c} / 清单 {m}" for n, (c, m) in mismatched.items()
        ),
    ))

    sizes: dict[str, int] = {}
    corrupt: list[str] = []
    for path in sorted(p for p in blobs_dir.rglob("*") if p.is_file()):
        data = path.read_bytes()
        ref = "sha256:" + path.name
        if hashlib.sha256(data).hexdigest() != path.name:
            corrupt.append(ref)
        else:
            sizes[ref] = len(data)
    checks.append(Check(
        "每个 blob 的 sha256 都对得上文件名", not corrupt,
        f"{len(sizes)} 个全部一致" if not corrupt else
        f"{len(corrupt)} 个已损坏：" + ", ".join(r[:18] for r in corrupt[:3]),
    ))
    digest = _digest_of(sizes)
    matches = digest == manifest.get("digest")
    checks.append(Check(
        "blob 集合与清单一致", matches,
        f"{len(sizes)} 个 / {sum(sizes.values())} B" if matches else
        f"清单 {manifest.get('blobs')} 个 / {manifest.get('blob_bytes')} B，"
        f"实际 {len(sizes)} 个 / {sum(sizes.values())} B（备份目录被动过，或备份没做完）",
    ))

    unresolved = sorted(wanted - set(sizes))
    checks.append(Check(
        # 只备库不备证据的"备份"就是靠这一步暴露的
        "备份里的引用全部可解析", not unresolved,
        f"{len(wanted)} 个引用全在" if not unresolved else
        f"{len(unresolved)} 个引用缺文件：" + ", ".join(r[:18] for r in unresolved[:3]),
    ))

    drift: dict[str, Any] = {}
    if live is not None:
        grown = {
            name: int(live.scalar(f"SELECT COUNT(*) FROM {name}", (), 0)) - int(recorded[name])
            for name in recorded
            if name in live.table_names()
        }
        grown = {n: d for n, d in grown.items() if d > 0}
        drift = {"tables_grown": grown, "current_refs": len(referenced_refs(live))}
    return Report(root=root, checks=tuple(checks), drift=drift)
