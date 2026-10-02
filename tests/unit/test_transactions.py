"""事务语义测试。

这里有一条回归：曾经 `write_usage` 外层开事务、内层 `replace_parts` 再开一次，
sqlite3 抛 "cannot start a transaction within a transaction"，而锁在 `__enter__`
里已经获取却没人释放 —— 整个进程死锁（pytest 挂死，CPU 0%）。
"""

from __future__ import annotations

import pytest

from onyx.store.db import Database


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "t.sqlite") as database:
        yield database


def _insert_provider(db, pid: str) -> None:
    db.execute(
        "INSERT INTO provider(id, kind, base_url, api_style, created_at) VALUES(?,?,?,?,?)",
        (pid, "mock", "", "native", "2026-01-01T00:00:00+00:00"),
    )


def test_transaction_commits(db):
    with db.transaction():
        _insert_provider(db, "p1")
    assert db.scalar("SELECT COUNT(*) FROM provider") == 1


def test_transaction_rolls_back_on_error(db):
    with pytest.raises(ValueError, match="boom"), db.transaction():
        _insert_provider(db, "p1")
        raise ValueError("boom")
    assert db.scalar("SELECT COUNT(*) FROM provider") == 0


def test_nested_transaction_commits_once(db):
    """嵌套不许发第二次 BEGIN；全部成功时一次性提交。"""
    with db.transaction():
        _insert_provider(db, "p1")
        with db.transaction():
            _insert_provider(db, "p2")
    assert db.scalar("SELECT COUNT(*) FROM provider") == 2


def test_inner_failure_rolls_back_outer_even_if_swallowed(db):
    """内层失败被外层 catch 住时，外层也不许提交半套数据。"""
    with db.transaction():
        _insert_provider(db, "p1")
        try:
            with db.transaction():
                _insert_provider(db, "p2")
                raise RuntimeError("inner failed")
        except RuntimeError:
            pass
    assert db.scalar("SELECT COUNT(*) FROM provider") == 0, "内层失败必须污染整个外层事务"


def test_lock_released_after_nested_failure(db):
    """死锁回归：嵌套事务失败后，锁必须已释放，后续操作不能阻塞。"""
    with pytest.raises(RuntimeError), db.transaction(), db.transaction():
        raise RuntimeError("x")
    # 若锁泄漏，下面这行会永久阻塞（faulthandler_timeout 会 dump 栈）
    _insert_provider(db, "after")
    assert db.scalar("SELECT COUNT(*) FROM provider") == 1


def test_failed_transaction_does_not_leave_sqlite_in_tx(db):
    with pytest.raises(ValueError), db.transaction():
        _insert_provider(db, "p1")
        raise ValueError
    assert db._conn.in_transaction is False
    with db.transaction():
        _insert_provider(db, "p2")
    assert db.scalar("SELECT COUNT(*) FROM provider") == 1
