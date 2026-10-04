"""S18 备份与"备份可验证恢复"。

备份最危险的失败不是报错，而是**安静地少一块**：少了一个 blob、少了一段 WAL 里的事务、
或者只备了库没备证据——恢复出来是个看起来完全正常的空看板。
所以这里的断言都在两件事上：备份必须包含 WAL 之后的数据，verify 必须能指名道姓抓出缺什么。
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from onyx.cli import app
from onyx.core.content import FileBlobStore
from onyx.core.ids import new_trace_id
from onyx.settings import load_settings
from onyx.store.backup import (
    DB_NAME,
    MANIFEST_NAME,
    _blob_path,
    create_backup,
    verify_backup,
)
from onyx.store.db import Database
from onyx.store.records import TraceRecord
from onyx.store.repos import TraceRepo

NOW = "2026-10-04T00:00:00.000000+00:00"


@pytest.fixture
def live(tmp_path):
    """一个有数据的小库：3 条 trace + 各自 blob + 1 个没人引用的孤儿 blob。"""
    database = Database(tmp_path / "live.sqlite")
    store = FileBlobStore(tmp_path / "blobs")
    refs = {}
    for index in range(3):
        ref = store.put_json({"role": "user", "content": f"证据 {index}" * 20})
        refs[ref] = index
        TraceRepo(database).upsert(
            TraceRecord(id=new_trace_id(), kind="request", purpose="chat",
                        started_at=NOW, messages_ref=ref)
        )
    orphan = store.put_text("没人引用的孤儿 blob")
    return database, store, refs, orphan


def _names(report) -> dict[str, bool]:
    return {check.name: check.ok for check in report.checks}


def _detail(report, name: str) -> str:
    return next(c.detail for c in report.checks if c.name == name)


# ── 备份内容 ───────────────────────────────────────────────────────
def test_backup_contains_rows_still_in_the_wal(live, tmp_path):
    """刚写进去、还没 checkpoint 的行必须在备份里。

    这是 `cp .sqlite` 会失败而 `Connection.backup` 不会的地方：WAL 模式下最近的
    事务还在 `-wal` 文件里，只拷主文件等于拷一个旧的库——而且它打开还不报错。
    """
    database, store, _refs, _ = live
    fresh = new_trace_id()
    fresh_ref = store.put_json({"fresh": True})
    TraceRepo(database).upsert(
        TraceRecord(id=fresh, kind="request", purpose="chat", started_at=NOW,
                    messages_ref=fresh_ref)
    )
    assert (tmp_path / "live.sqlite-wal").exists(), "没有 WAL 就没有这条测试的前提"

    info = create_backup(database, store, tmp_path / "backup")
    with sqlite3.connect(info.root / DB_NAME) as conn:
        ids = {row[0] for row in conn.execute("SELECT id FROM trace")}

    assert fresh in ids, "备份必须包含 WAL 里还没落主文件的数据"
    assert len(ids) == 4


def test_backup_only_carries_referenced_blobs(live, tmp_path):
    database, store, refs, orphan = live
    info = create_backup(database, store, tmp_path / "backup")

    assert info.blobs == len(refs)
    backup_blobs = {p.name for p in (info.root / "blobs").rglob("*") if p.is_file()}
    assert backup_blobs == {ref.split(":")[1] for ref in refs}
    assert orphan.split(":")[1] not in backup_blobs, "孤儿不该占备份体积"
    # 备份沿用同样的两级分片布局，恢复时能直接并回去
    assert _blob_path(info.root / "blobs", next(iter(refs))).exists()


def test_backup_refuses_to_overwrite_an_existing_one(live, tmp_path):
    """覆盖一个恢复点，通常要等到真需要恢复时才被发现——所以直接拒绝。"""
    database, store, _, _ = live
    target = tmp_path / "backup"
    create_backup(database, store, target)

    with pytest.raises(FileExistsError, match="已经有一份备份"):
        create_backup(database, store, target)


def test_backup_records_refs_that_were_already_missing(live, tmp_path):
    """备份当时就悬空的引用：不拦备份，但要如实写进清单。"""
    database, store, refs, _ = live
    gone = next(iter(refs))
    store.delete(gone)

    info = create_backup(database, store, tmp_path / "backup")

    assert info.unresolved == (gone,)
    manifest = json.loads((info.root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["unresolved_refs"] == [gone]


def test_blob_path_rejects_traversal(live, tmp_path):
    """ref 参与路径拼接之前必须严格校验——备份目录是要被人当恢复源用的。"""
    for bad in ("sha256:../" + "../" * 6 + "etc/passwd", "sha256:zz" + "0" * 62,
                "md5:" + "a" * 32, "sha256:" + "A" * 64):
        with pytest.raises(ValueError, match="非法 blob ref"):
            _blob_path(tmp_path / "blobs", bad)


# ── verify：每一份"坏"都要报得出名字 ───────────────────────────────
def test_verify_accepts_a_good_backup(live, tmp_path):
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")

    report = verify_backup(info.root, live=database)

    assert report.ok, [f"{c.name}: {c.detail}" for c in report.checks if not c.ok]
    assert _names(report)["备份里的引用全部可解析"] is True
    assert report.drift["tables_grown"] == {}


def test_verify_does_not_rewrite_the_backup(live, tmp_path):
    """验证一个备份不该把它改成 WAL 模式——否则"备份"会被验证动作本身动过。"""
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")

    verify_backup(info.root)

    assert not (info.root / (DB_NAME + "-wal")).exists()
    assert not (info.root / (DB_NAME + "-shm")).exists()


def test_backup_is_a_self_contained_single_file(live, tmp_path):
    """备份得"一个文件就能恢复"。

    源库是 WAL 模式，复制出来的头一页也写着 WAL——那等于宣称这个文件可恢复，
    而它其实还依赖一个没被拷走的 `-wal`。所以落地时切回 DELETE 日志模式。
    """
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")

    assert not (info.root / (DB_NAME + "-wal")).exists()
    with sqlite3.connect(info.root / DB_NAME) as conn:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]) == "delete"


def test_verify_reports_growth_since_the_backup(live, tmp_path):
    """备完之后当前库又长了数据，不是备份的错：verify 要报增量而不是失败。"""
    database, store, refs, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    TraceRepo(database).upsert(
        TraceRecord(id=new_trace_id(), kind="request", purpose="chat", started_at=NOW,
                    messages_ref=next(iter(refs)))
    )

    report = verify_backup(info.root, live=database)

    assert report.ok
    assert report.drift["tables_grown"] == {"trace": 1}


def test_verify_detects_a_missing_blob_file(live, tmp_path):
    """少一个文件 ⇒ 集合摘要对不上，且那个引用在备份里解析不了。"""
    database, store, refs, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    victim = _blob_path(info.root / "blobs", next(iter(refs)))
    victim.unlink()

    report = verify_backup(info.root)

    flags = _names(report)
    assert report.ok is False
    assert flags["blob 集合与清单一致"] is False
    assert flags["备份里的引用全部可解析"] is False
    assert "1 个引用缺文件" in _detail(report, "备份里的引用全部可解析")


def test_verify_detects_a_tampered_blob(live, tmp_path):
    """内容被改过一个字节就再也对不上文件名——内容寻址免费给了一个校验器。"""
    database, store, refs, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    victim = _blob_path(info.root / "blobs", next(iter(refs)))
    victim.write_bytes("被人改过的证据".encode())

    report = verify_backup(info.root)

    assert _names(report)["每个 blob 的 sha256 都对得上文件名"] is False
    assert "已损坏" in _detail(report, "每个 blob 的 sha256 都对得上文件名")


def test_verify_detects_dropped_rows_in_the_backup(live, tmp_path):
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    with sqlite3.connect(info.root / DB_NAME) as conn:
        conn.execute("DELETE FROM trace WHERE rowid = (SELECT MIN(rowid) FROM trace)")
        conn.commit()

    report = verify_backup(info.root)

    assert _names(report)["每张表的行数与清单一致"] is False
    assert "trace 备份里 2 / 清单 3" in _detail(report, "每张表的行数与清单一致")


def test_verify_detects_a_corrupt_backup_database(live, tmp_path):
    """库读不了 ⇒ 报这一条并停下，而不是让后面每条检查都"没发现失败"地绿过去。"""
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    (info.root / DB_NAME).write_bytes("这不是 sqlite 文件".encode())

    report = verify_backup(info.root)

    assert report.ok is False
    assert _names(report)["库能打开且自检通过"] is False
    assert "读不了" in _detail(report, "库能打开且自检通过")
    assert len(report.checks) == 3, "库打不开时 blob 与引用检查无法评估，不该装作通过"


def test_verify_detects_a_corrupt_manifest(live, tmp_path):
    """manifest 读不了就没得比：报这一条并停下，不要拿空清单去"通过"其余检查。"""
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    (info.root / MANIFEST_NAME).write_text("{不是 json", encoding="utf-8")

    report = verify_backup(info.root)

    assert report.ok is False
    assert _names(report)["manifest 可解析"] is False
    assert len(report.checks) == 2, "清单读不了就不往下猜了"


def test_verify_detects_missing_tables_in_the_backup(live, tmp_path):
    """备份库少了表：行数、schema 版本、引用清单三处都得如实失败，而不是各自装作没看见。"""
    database, store, _, _ = live
    info = create_backup(database, store, tmp_path / "backup")
    with sqlite3.connect(info.root / DB_NAME) as conn:
        conn.execute("DROP TABLE trace")
        conn.execute("DROP TABLE schema_version")
        conn.commit()

    report = verify_backup(info.root)

    flags = _names(report)
    assert flags["每张表的行数与清单一致"] is False
    assert "trace 备份里 -1" in _detail(report, "每张表的行数与清单一致")
    assert flags["schema 版本与清单一致"] is False
    assert "备份 v-1" in _detail(report, "schema 版本与清单一致")


def test_verify_reports_an_incomplete_directory(tmp_path):
    """半途而废的备份（只有 manifest）必须一眼看出来，而不是后面每条检查都莫名其妙失败。"""
    root = tmp_path / "backup"
    root.mkdir()
    (root / MANIFEST_NAME).write_text("{}", encoding="utf-8")

    report = verify_backup(root)

    assert _names(report)["备份结构完整"] is False
    assert "onyx.sqlite" in _detail(report, "备份结构完整")
    assert len(report.checks) == 1, "结构不全就不再往下猜了"


# ── CLI ────────────────────────────────────────────────────────────
@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "240")
    return load_settings().ensure_dirs()


def _cli(*argv: str):
    from typer.testing import CliRunner

    return CliRunner().invoke(app, list(argv))


def _seed(data_dir) -> None:
    database = Database(data_dir.db_path)
    store = FileBlobStore(data_dir.blob_dir)
    ref = store.put_json({"content": "真实证据 " * 30})
    TraceRepo(database).upsert(
        TraceRecord(id=new_trace_id(), kind="request", purpose="chat",
                    started_at=NOW, messages_ref=ref)
    )
    database.close()


def test_cli_backup_then_verify_is_green(data_dir):
    _seed(data_dir)
    target = data_dir.data_dir / "backup"

    made = _cli("db", "backup", "--to", str(target))
    assert made.exit_code == 0, made.output
    assert "只装被引用的" in made.output

    checked = _cli("db", "verify-backup", str(target))
    assert checked.exit_code == 0, checked.output
    assert "备份里的引用全部可解析" in checked.output


def test_cli_verify_fails_loudly_when_evidence_is_missing(data_dir):
    """恢复手段本身坏了要非 0 退出——否则人会以为"有备份"就等于"能恢复"。"""
    _seed(data_dir)
    target = data_dir.data_dir / "backup"
    _cli("db", "backup", "--to", str(target))
    blobs = [p for p in (target / "blobs").rglob("*") if p.is_file()]
    blobs[0].unlink()

    checked = _cli("db", "verify-backup", str(target))

    assert checked.exit_code == 1
    assert "缺文件" in checked.output


def test_cli_backup_json_is_machine_readable(data_dir):
    _seed(data_dir)
    target = data_dir.data_dir / "backup"
    _cli("db", "backup", "--to", str(target))

    result = _cli("db", "verify-backup", str(target), "--json")

    body = json.loads(result.output)
    assert body["ok"] is True
    assert {c["name"] for c in body["checks"]} >= {"备份结构完整", "库能打开且自检通过"}


def test_cli_backup_refuses_to_overwrite(data_dir):
    _seed(data_dir)
    target = data_dir.data_dir / "backup"
    assert _cli("db", "backup", "--to", str(target)).exit_code == 0

    again = _cli("db", "backup", "--to", str(target))

    assert again.exit_code == 1
    assert "拒绝覆盖" in (again.output + getattr(again, "stderr", ""))
