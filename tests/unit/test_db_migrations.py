from __future__ import annotations

import shutil

import pytest

from onyx.core.errors import MigrationError
from onyx.store.db import MIGRATIONS_DIR, Database, discover_migrations


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "t.sqlite")


def test_fresh_db_applies_migrations(db):
    assert db.version() == 2
    tables = set(db.table_names())
    assert {"provider", "model", "trace", "usage", "usage_alt", "token_part", "tool_call", "anomaly"} <= tables
    columns = {r["name"] for r in db.query("PRAGMA table_info(usage)")}
    assert {"prefill_mode", "prefill_ms_per_token"} <= columns, "P11 的冷/热分列必须落库"


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "t.sqlite"
    first = Database(path)
    assert first.migrate() == []  # 已在构造时应用
    first.close()
    second = Database(path)
    assert second.version() == 2
    assert second.migrate() == []
    second.close()


def test_wal_and_foreign_keys_enabled(db):
    assert db.scalar("PRAGMA journal_mode") == "wal"
    assert int(db.scalar("PRAGMA foreign_keys")) == 1


def test_tampering_applied_migration_is_refused(tmp_path):
    migrations = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations)
    path = tmp_path / "t.sqlite"
    db = Database(path, migrate=False)
    db.migrate(migrations)

    target = migrations / "0001_init.sql"
    target.write_text(target.read_text(encoding="utf-8") + "\n-- 偷改历史\n", encoding="utf-8")
    with pytest.raises(MigrationError, match="内容被修改"):
        db.migrate(migrations)
    db.close()


def test_incremental_migration_applies_only_new(tmp_path):
    """真实场景：已有数据的库，新增一个迁移文件后重开，只应用新的那个且旧数据完好。"""
    migrations = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations)
    path = tmp_path / "t.sqlite"

    with Database(path, migrate=False) as db:
        assert db.migrate(migrations) == [1, 2]
        db.execute(
            "INSERT INTO provider(id, kind, base_url, api_style, created_at) VALUES('p','mock','','native','')"
        )
        assert db.version() == 2

    # 之后新增一个迁移
    (migrations / "0003_add_tool_def.sql").write_text(
        "CREATE TABLE IF NOT EXISTS tool_def(id TEXT PRIMARY KEY, name TEXT NOT NULL);", encoding="utf-8"
    )
    with Database(path, migrate=False) as db:
        assert db.migrate(migrations) == [3], "只应用新增的迁移"
        assert db.version() == 3
        assert "tool_def" in db.table_names()
        assert db.scalar("SELECT COUNT(*) FROM provider") == 1, "既有数据必须完好"
        assert db.migrate(migrations) == []


def test_bad_migration_filename_rejected(tmp_path):
    migrations = tmp_path / "m"
    migrations.mkdir()
    (migrations / "init.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match=r"NNNN_name\.sql"):
        discover_migrations(migrations)


def test_duplicate_version_rejected(tmp_path):
    migrations = tmp_path / "m"
    migrations.mkdir()
    (migrations / "0001_a.sql").write_text("SELECT 1;", encoding="utf-8")
    (migrations / "0001_b.sql").write_text("SELECT 1;", encoding="utf-8")
    with pytest.raises(MigrationError, match="重复"):
        discover_migrations(migrations)


def test_failed_migration_rolls_back(tmp_path):
    """迁移中途失败必须整体回滚：绝不允许留下半套 schema。"""
    good = tmp_path / "m_good"
    good.mkdir()
    shutil.copy(MIGRATIONS_DIR / "0001_init.sql", good / "0001_init.sql")

    broken = tmp_path / "m_broken"
    broken.mkdir()
    shutil.copy(MIGRATIONS_DIR / "0001_init.sql", broken / "0001_init.sql")
    (broken / "0002_broken.sql").write_text(
        "CREATE TABLE ok_table(x INT); THIS IS NOT SQL;", encoding="utf-8"
    )

    with Database(tmp_path / "t.sqlite", migrate=False) as db:
        assert db.migrate(good) == [1]

    with Database(tmp_path / "t2.sqlite", migrate=False) as db2:
        with pytest.raises(MigrationError, match="失败"):
            db2.migrate(broken)
        assert "ok_table" not in db2.table_names(), "失败迁移的副作用必须被回滚"
        assert db2.version() == 1, "0001 已成功，只回滚失败的 0002"
        assert db2.migrate(good) == [], "重跑时不许重复应用 0001"


def test_memory_db_supported():
    with Database(":memory:") as db:
        assert db.version() == 2
        db.execute("INSERT INTO provider(id,kind,base_url,api_style,created_at) VALUES('p','mock','','native','')")
        assert db.scalar("SELECT COUNT(*) FROM provider") == 1
