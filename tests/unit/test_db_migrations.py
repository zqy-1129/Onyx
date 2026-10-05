from __future__ import annotations

import shutil

import pytest

from onyx.core.errors import MigrationError
from onyx.store.db import MIGRATIONS_DIR, Database, discover_migrations


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "t.sqlite")


#: 仓库里的迁移数量。新增迁移时这个数会变，测试随之更新——
#: 它是"迁移有没有被意外删掉/改名"的一道哨兵
EXPECTED_VERSION = 7


def test_fresh_db_applies_migrations(db):
    assert db.version() == EXPECTED_VERSION
    tables = set(db.table_names())
    assert {"provider", "model", "trace", "usage", "usage_alt", "token_part", "tool_call",
            "anomaly", "tool_def", "tool_test", "tool_run",
            "dataset", "eval_case", "eval_task", "eval_run", "grade",
            "retention_run", "alert_trigger"} <= tables
    columns = {r["name"] for r in db.query("PRAGMA table_info(usage)")}
    assert {"prefill_mode", "prefill_ms_per_token"} <= columns, "P11 的冷/热分列必须落库"


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "t.sqlite"
    first = Database(path)
    assert first.migrate() == []  # 已在构造时应用
    first.close()
    second = Database(path)
    assert second.version() == EXPECTED_VERSION
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
        assert db.migrate(migrations) == list(range(1, EXPECTED_VERSION + 1))
        db.execute(
            "INSERT INTO provider(id, kind, base_url, api_style, created_at) VALUES('p','mock','','native','')"
        )
        assert db.version() == EXPECTED_VERSION

    # 之后新增一个迁移（版本号必须接在现有迁移之后）
    next_version = EXPECTED_VERSION + 1
    (migrations / f"{next_version:04d}_add_probe_log.sql").write_text(
        "CREATE TABLE IF NOT EXISTS probe_log(id TEXT PRIMARY KEY, model TEXT NOT NULL);",
        encoding="utf-8",
    )
    with Database(path, migrate=False) as db:
        assert db.migrate(migrations) == [next_version], "只应用新增的迁移"
        assert db.version() == next_version
        assert "probe_log" in db.table_names()
        assert db.scalar("SELECT COUNT(*) FROM provider") == 1, "既有数据必须完好"
        assert db.migrate(migrations) == []


def test_migration_writes_a_rollback_snapshot_first(tmp_path):
    """升级真的动了 schema 之前必须先留一份能退回去的库。

    没有这一步，一次失败的迁移留下半套 schema，而唯一的选择是手改数据库——
    那是观测系统里最不该发生的"只能靠人记着怎么修"的时刻。
    """
    migrations = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS_DIR, migrations)
    path = tmp_path / "t.sqlite"

    with Database(path, migrate=False) as db:
        db.migrate(migrations)
        db.execute(
            "INSERT INTO provider(id, kind, base_url, api_style, created_at) VALUES('p','mock','','native','')"
        )

    (migrations / f"{EXPECTED_VERSION + 1:04d}_wider.sql").write_text(
        "CREATE TABLE IF NOT EXISTS extra_note(id TEXT PRIMARY KEY, text TEXT NOT NULL);",
        encoding="utf-8",
    )
    with Database(path, migrate=False) as db:
        assert db.migrate(migrations) == [EXPECTED_VERSION + 1]

    snapshot = tmp_path / "backups" / f"pre-migration-v{EXPECTED_VERSION}.sqlite"
    assert snapshot.exists(), "升级前必须留快照"
    with Database(snapshot, migrate=False) as old:
        assert old.version() == EXPECTED_VERSION, "快照是**升级前**的版本"
        assert old.scalar("SELECT COUNT(*) FROM provider") == 1, "快照里数据完好才叫能退回"

    # 再开一次不该又复制一份：同名快照就是那个回滚点
    with Database(path, migrate=False) as db:
        assert db.migrate(migrations) == []
    assert sorted(p.name for p in (tmp_path / "backups").iterdir()) == [
        f"pre-migration-v{EXPECTED_VERSION}.sqlite"
    ]


def test_empty_database_gets_no_snapshot(tmp_path):
    """空库没有可回滚的东西：造一个 backups/ 目录只是噪音。"""
    with Database(tmp_path / "fresh.sqlite") as db:
        assert db.version() == EXPECTED_VERSION
    assert not (tmp_path / "backups").exists()


def test_memory_database_skips_snapshot(tmp_path, monkeypatch):
    """`:memory:` 没有文件可拷——它必须在拼路径之前就返回，而不是往 cwd 写东西。"""
    cwd_before = set(p.name for p in tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)

    with Database(":memory:") as db:
        assert db.version() == EXPECTED_VERSION

    assert set(p.name for p in tmp_path.iterdir()) == cwd_before, "内存库不该留下文件"


def test_backfill_only_runs_when_the_source_version_is_reachable(tmp_path):
    """回填型迁移的边界：跳级升级时它根本不会执行（前一个版本已被裁剪）。

    这类迁移不能假装"重跑一次就好了"，所以这里断言它只在目标场景里跑。
    """
    from onyx.store.db import discover_migrations

    assert len(discover_migrations(MIGRATIONS_DIR)) == EXPECTED_VERSION


def test_dataset_provenance_is_backfilled_for_existing_runs(tmp_path):
    """0005 的回填必须真的把历史 run 的数据集补上。

    回填是迁移最容易糊弄过去的一步：新库看起来一切正常，老库升级之后
    `dataset_id` 全空，于是"这两次跑的是不是同一份数据"重新变成无法回答的问题
    ——而 M5 的每个对比结论都建立在这件事上。
    """
    from onyx.store.records import CaseRecord, DatasetRecord
    from onyx.store.repos import EvalRepo

    migrations = tmp_path / "migrations"
    migrations.mkdir()
    # 只放 0001–0004：模拟"升级前"的库
    for path in sorted(MIGRATIONS_DIR.glob("*.sql"))[:4]:
        shutil.copy(path, migrations / path.name)

    path = tmp_path / "t.sqlite"
    with Database(path, migrate=False) as db:
        assert db.migrate(migrations) == [1, 2, 3, 4]
        repo = EvalRepo(db)
        repo.upsert_dataset(DatasetRecord(id="intent_zh-v1", imported_at="2026-10-03T00:00:00+00:00",
                                          upstream="builtin:intent_zh",
                                          revision="seed=20261003", n_cases=2))
        repo.upsert_cases([
            CaseRecord(id="izh-1", dataset_id="intent_zh-v1", ord=0,
                       input={"instruction": "a"}, expect={"label": "转账"}),
            CaseRecord(id="izh-2", dataset_id="intent_zh-v1", ord=1,
                       input={"instruction": "b"}, expect={"label": "查余额"}),
        ])
        db.execute(
            """INSERT INTO eval_task(id, name, dataset_id, metrics_json)
               VALUES('intent_classification','意图识别','intent_zh-v1','[]')"""
        )
        # 旧数据必须用**旧 schema** 写：`EvalRepo.insert_run` 现在带 dataset_id 列，
        # 而 0005 之前那一列还不存在
        db.execute(
            """INSERT INTO eval_run(id, task_id, model_id, started_at, status, n_cases)
               VALUES('run-old','intent_classification','qwen3.5:9b',
                      '2026-10-03T00:00:00+00:00','done',2)"""
        )
        for index, case_id in enumerate(("izh-1", "izh-2")):
            db.execute(
                """INSERT INTO grade(id, eval_run_id, case_id, seq, score, verdict,
                                    invalid_format, out_of_set, graded_at)
                   VALUES(?,?,?,0,1.0,'correct',0,0,'2026-10-03T00:00:01+00:00')""",
                (f"g{index}", "run-old", case_id),
            )

    with Database(path, migrate=False) as db:
        assert db.migrate(MIGRATIONS_DIR) == list(range(5, EXPECTED_VERSION + 1)), \
            "0005 回填 + 之后的每张留痕表都要应用上"
        run = EvalRepo(db).get_run("run-old")
        assert run is not None
        assert run.dataset_id == "intent_zh-v1", "历史 run 的数据集必须被回填出来"
        assert run.dataset_revision == "seed=20261003", "版本号也要回填，否则不可比性无从判断"


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
        assert db.version() == EXPECTED_VERSION
        db.execute("INSERT INTO provider(id,kind,base_url,api_style,created_at) VALUES('p','mock','','native','')")
        assert db.scalar("SELECT COUNT(*) FROM provider") == 1
