"""SQLite 连接与迁移。

设计要点：
- **WAL + busy_timeout**：看板读、gateway 写并发；WAL 让读不阻塞写。
- **迁移幂等且防篡改**：每个 `.sql` 文件一个版本号，落 `schema_version` 并记 checksum；
  已应用的迁移文件若被改动 → 直接报 `MigrationError`，绝不静默重放。
- 单写者：所有写通过 `Database` 的同一连接（`check_same_thread=False` + 内部锁），
  避免 SQLite 多写者锁竞争（DESIGN R11）。
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.core.errors import MigrationError

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA synchronous=NORMAL",
    "PRAGMA foreign_keys=ON",
    "PRAGMA busy_timeout=5000",
    "PRAGMA temp_store=MEMORY",
)


class Database:
    def __init__(self, path: Path | str, *, migrate: bool = True) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._tx_failed = False
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=5.0
        )
        self._conn.row_factory = sqlite3.Row
        for pragma in _PRAGMAS:
            self._conn.execute(pragma)
        if migrate:
            self.migrate()

    # ── 生命周期 ──────────────────────────────────────────────────
    @classmethod
    def open(cls, path: Path | str, *, migrate: bool = True) -> Database:
        return cls(path, migrate=migrate)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── 迁移 ──────────────────────────────────────────────────────
    def migrate(self, migrations_dir: Path | None = None) -> list[int]:
        """按序应用未应用的迁移，返回本次应用的版本号。"""
        directory = migrations_dir or MIGRATIONS_DIR
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS schema_version(
                       version     INTEGER PRIMARY KEY,
                       name        TEXT NOT NULL,
                       checksum    TEXT NOT NULL,
                       applied_at  TEXT NOT NULL
                   )"""
            )
            applied = {
                int(row["version"]): str(row["checksum"])
                for row in self._conn.execute("SELECT version, checksum FROM schema_version")
            }
            done: list[int] = []
            for version, name, sql in discover_migrations(directory):
                digest = hashlib.sha256(sql.encode("utf-8")).hexdigest()
                if version in applied:
                    if applied[version] != digest:
                        raise MigrationError(
                            f"已应用的迁移 {name} 内容被修改（checksum 不一致）。"
                            "请新增一个迁移文件，不要改历史。",
                            detail={"version": version, "name": name},
                        )
                    continue
                try:
                    self._conn.executescript(self._migration_script(version, name, digest, sql))
                except sqlite3.Error as exc:
                    if self._conn.in_transaction:
                        self._conn.execute("ROLLBACK")
                    raise MigrationError(f"迁移 {name} 失败: {exc}", detail={"version": version}) from exc
                done.append(version)
            return done

    @staticmethod
    def _migration_script(version: int, name: str, checksum: str, sql: str) -> str:
        """把版本记账与 DDL 拼进**同一个显式事务**。

        不能用 `BEGIN` + `executescript`：CPython 的 executescript 会先隐式提交
        当前事务（实测 3.12.15 / SQLite 3.53），于是 DDL 落在自动提交模式里，
        迁移失败就会留下半套 schema。把 BEGIN 写进脚本内部才有原子性。
        字面量来自我们自己的文件名与 sha256，仍做单引号转义以防意外。
        """
        def lit(value: object) -> str:
            return "'" + str(value).replace("'", "''") + "'"

        return (
            "BEGIN IMMEDIATE;\n"
            "INSERT INTO schema_version(version, name, checksum, applied_at) "
            f"VALUES({int(version)}, {lit(name)}, {lit(checksum)}, {lit(utc_now_iso())});\n"
            f"{sql.rstrip().rstrip(';')};\n"
            "COMMIT;\n"
        )

    def version(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(version),0) AS v FROM schema_version").fetchone()
        return int(row["v"])

    def table_names(self) -> list[str]:
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        return sorted(str(r["name"]) for r in rows)

    # ── 读写 ──────────────────────────────────────────────────────
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> None:
        rows = list(seq)
        if not rows:
            return
        # 不能用 `with self._conn`：sqlite3 的连接上下文管理器会提交/回滚当前事务，
        # 这会破坏调用方通过 transaction() 显式开启的事务边界。
        with self._lock:
            self._conn.executemany(sql, rows)

    def transaction(self):
        """`with db.transaction(): ...` —— 单写者下的显式事务。"""
        return _Transaction(self)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def query_one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Sequence[Any] = (), default: Any = None) -> Any:
        row = self.query_one(sql, params)
        if row is None:
            return default
        return row[0]


class _Transaction:
    """可重入事务。

    为什么必须可重入：repo 方法（如 `UsageRepo.replace_parts`）既要能独立调用，
    也会被上层（`SqliteRecordSink.write_usage`）包在事务里调用。若内层再发一次
    `BEGIN`，sqlite3 直接抛错——而如果此时锁已获取却没释放，整个进程死锁。
    所以只有最外层真正 BEGIN/COMMIT，内层只计数；任何一层失败都让最外层回滚。
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def __enter__(self) -> Database:
        db = self._db
        db._lock.acquire()
        db._tx_depth += 1
        if db._tx_depth == 1:
            try:
                db._tx_failed = False
                db._conn.execute("BEGIN IMMEDIATE")
            except BaseException:
                db._tx_depth -= 1
                db._lock.release()
                raise
        return db

    def __exit__(self, exc_type: object, *_: object) -> None:
        db = self._db
        try:
            if exc_type is not None:
                db._tx_failed = True
            db._tx_depth -= 1
            if db._tx_depth == 0:
                db._conn.execute("ROLLBACK" if db._tx_failed else "COMMIT")
                db._tx_failed = False
        finally:
            db._lock.release()


def discover_migrations(directory: Path) -> list[tuple[int, str, str]]:
    """`NNNN_name.sql` → (version, name, sql)。版本号冲突或命名不合规直接报错。"""
    found: dict[int, tuple[str, str]] = {}
    for path in sorted(directory.glob("*.sql")):
        stem = path.stem
        head, _, tail = stem.partition("_")
        if not head.isdigit() or not tail:
            raise MigrationError(f"迁移文件名必须是 NNNN_name.sql，实际: {path.name}")
        version = int(head)
        if version in found:
            raise MigrationError(f"迁移版本号 {version} 重复: {found[version][0]} 与 {path.name}")
        found[version] = (stem, path.read_text(encoding="utf-8"))
    return [(v, name, sql) for v, (name, sql) in sorted(found.items())]
