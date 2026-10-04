"""S17 数据生命周期：`onyx rotate` 的保留策略。

断言的方向都是"删错"的代价——原始 body 删了就回不来：
1. 默认什么都不删（dry-run），但必须报得出"会删多少"，否则没人敢按 `--apply`；
2. 摘的是重引用，留下的必须够下钻（行本身 + messages/output/计数）；
3. 被结论引用的 trace 行永不消失；共享 blob 只要还有别的行引用就不能删；
4. 每次运行都留痕，且**留痕写的是事实**：dry-run/被拦下时"摘除数"必须是 0。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from onyx.cli import app
from onyx.core.content import FileBlobStore
from onyx.core.ids import new_trace_id
from onyx.settings import load_settings
from onyx.store.db import Database
from onyx.store.records import (
    AnomalyRecord,
    DatasetRecord,
    GradeRecord,
    RunRecord,
    TaskRecord,
    TokenPartRecord,
    ToolCallRecord,
    ToolRunRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)
from onyx.store.repos import EvalRepo, ToolRepo, TraceRepo, UsageRepo
from onyx.store.retention import (
    MAX_RECLAIM_RATIO,
    disk_report,
    footprint_trend,
    history,
    parse_window,
    referenced_refs,
    sweep,
)

NOW = "2026-10-04T00:00:00+00:00"


def _ago(days: int) -> str:
    """相对 `NOW` 的时刻，格式与落库的 started_at 一致（TEXT 按字典序比较）。"""
    reference = datetime.fromisoformat(NOW).astimezone(UTC)
    return (reference - timedelta(days=days)).isoformat(timespec="microseconds")


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "t.sqlite") as database:
        yield database


@pytest.fixture
def store(tmp_path):
    return FileBlobStore(tmp_path / "blobs")


def _trace(db, *, age_days: int = 60, purpose: str = "chat", **refs) -> str:
    trace_id = new_trace_id()
    TraceRepo(db).upsert(
        TraceRecord(id=trace_id, kind="request", purpose=purpose, started_at=_ago(age_days), **refs)
    )
    return trace_id


def _apply(db, store, **kw):
    """真的删。统一带 `force=True`：这些用例不测回收上限那道门槛（见
    `test_reclaim_above_the_cap_needs_force_and_deletes_nothing`），
    而小样本里"要回收的"天然就超过 60%，不让过就每个用例都得解释一遍门槛。"""
    return sweep(db, store, now=NOW, dry_run=False, force=True, **kw)


def _eval_run(db, trace_ids: list[str], run_id: str = "run1") -> None:
    """造一条真实形状的评测记录：dataset → task → run → grade(指向 trace)。"""
    evals = EvalRepo(db)
    evals.upsert_dataset(DatasetRecord(id="ds1", imported_at=_ago(300), revision="r1"))
    evals.upsert_task(TaskRecord(id="task1", name="意图识别", dataset_id="ds1",
                                 metrics=("accuracy",)))
    evals.insert_run(
        RunRecord(id=run_id, task_id="task1", model_id="m", started_at=_ago(200), dataset_id="ds1")
    )
    for seq, trace_id in enumerate(trace_ids):
        evals.upsert_grade(
            GradeRecord(
                id=f"g{seq}", eval_run_id=run_id, case_id=f"c{seq}", score=1.0,
                verdict="pass", graded_at=_ago(200), trace_id=trace_id,
            )
        )


# ── 窗口写法 ───────────────────────────────────────────────────────
def test_parse_window_accepts_units_and_rejects_bare_numbers():
    assert parse_window("90d") == timedelta(days=90)
    assert parse_window("24h") == timedelta(hours=24)
    assert parse_window("2w") == timedelta(days=14)
    for bad in ("30", "90x", "", "d", None):
        with pytest.raises(ValueError, match=r"d\|h\|w"):
            parse_window(bad)


# ── dry-run：报得出数字，但一个字节都不碰 ──────────────────────────
def test_dry_run_reports_the_numbers_but_touches_nothing(db, store):
    raw = store.put_text("原始响应 body " * 80)
    rendered = store.put_text("渲染后的 prompt " * 80)
    messages = store.put_json([{"role": "user", "content": "你好"}])
    trace_id = _trace(
        db, age_days=60, raw_response_ref=raw, rendered_prompt_ref=rendered, messages_ref=messages
    )

    result = sweep(db, store, now=NOW)

    assert result.dry_run and not result.applied
    assert result.per_rule["trace.raw_response_ref"] == 1
    assert sum(result.per_rule.values()) == 2, "messages_ref 不在摘除范围内"
    assert result.reclaimable_bytes == store.stat(raw).size + store.stat(rendered).size
    assert result.bytes_after == result.bytes_before
    assert result.freed_bytes == 0 and result.blobs_deleted == 0
    # 库里什么都没变
    record = TraceRepo(db).get(trace_id)
    assert record.raw_response_ref == raw and record.messages_ref == messages
    assert store.exists(raw) and store.exists(rendered)
    # dry-run 也留痕，且事实列是 0（不是"预计值"）
    run = history(db)[0]
    assert run["dry_run"] == 1
    assert run["refs_cleared"] == 0
    assert run["orphans_found"] == 2


def test_plan_and_apply_agree_on_the_same_numbers(db, store):
    """dry-run 说能回收 5000 B，真跑就必须回收 5000 B——两者共用一段代码。"""
    heavy = store.put_text("x" * 5000)
    _trace(db, age_days=60, raw_request_ref=heavy)

    planned = sweep(db, store, now=NOW)
    applied = _apply(db, store)

    assert planned.reclaimable_bytes == applied.freed_bytes
    assert sum(planned.per_rule.values()) == sum(applied.per_rule.values()) == 1


# ── apply：摘重引用，留下钻要用的东西 ──────────────────────────────
def test_apply_clears_heavy_refs_and_keeps_what_drill_down_needs(db, store):
    heavy = {
        name: store.put_text(f"{name} payload " * 120)
        for name in ("raw_request", "raw_response", "rendered_prompt")
    }
    keep = {
        "messages": store.put_json([{"role": "user", "content": "hi"}]),
        "output": store.put_json({"text": "好的"}),
        "tools": store.put_json([{"type": "function"}]),
    }
    trace_id = _trace(
        db, age_days=60,
        raw_request_ref=heavy["raw_request"], raw_response_ref=heavy["raw_response"],
        rendered_prompt_ref=heavy["rendered_prompt"],
        messages_ref=keep["messages"], output_ref=keep["output"], tools_ref=keep["tools"],
    )
    expected = sum(store.stat(ref).size for ref in heavy.values())
    before = store.total_bytes()

    result = _apply(db, store)

    assert result.applied and result.refs_cleared == 3
    assert result.freed_bytes == expected
    # 口径必须一致：报"释放了多少"和"盘上小了多少"是同一件事
    assert result.bytes_after == before - result.freed_bytes == before - expected
    assert result.reclaimable_bytes == result.freed_bytes
    record = TraceRepo(db).get(trace_id)
    assert record is not None, "行必须留着：删行会让 grade.trace_id 断线"
    assert (record.raw_request_ref, record.raw_response_ref, record.rendered_prompt_ref) == (
        None, None, None
    )
    assert record.messages_ref == keep["messages"] and record.output_ref == keep["output"]
    assert any(store.exists(ref) is False for ref in heavy.values())
    assert all(store.exists(ref) for ref in keep.values())


def test_young_traces_are_left_alone(db, store):
    heavy = store.put_text("y" * 2000)
    trace_id = _trace(db, age_days=3, raw_response_ref=heavy)

    result = _apply(db, store)

    assert result.reclaimed == 0
    assert TraceRepo(db).get(trace_id).raw_response_ref == heavy
    assert store.exists(heavy)


def test_shared_blob_survives_while_a_younger_row_still_references_it(db, store):
    """内容寻址会去重：老 trace 摘了引用，新 trace 还在引用同一个 blob ⇒ 不能删文件。"""
    shared = store.put_text("很长的 system prompt " * 60)
    old = _trace(db, age_days=60, raw_request_ref=shared)
    young = _trace(db, age_days=1, raw_request_ref=shared)

    result = _apply(db, store)

    assert result.per_rule["trace.raw_request_ref"] == 1
    assert result.blobs_deleted == 0, "还有一个活着的人在引用它"
    assert store.exists(shared)
    assert TraceRepo(db).get(old).raw_request_ref is None
    assert TraceRepo(db).get(young).raw_request_ref == shared
    assert referenced_refs(db) == {shared}


def test_tool_payloads_follow_the_same_window(db, store):
    """工具返回值也进 raw 窗口：事实（状态/延迟/参数）留着，重 payload 摘掉。"""
    result_ref = store.put_json({"rows": [{"i": n} for n in range(400)]})
    run_ref = store.put_text("工具输出 " * 300)
    trace_id = _trace(db, age_days=60, raw_request_ref=store.put_text("req " * 500))
    TraceRepo(db).insert_tool_call(
        ToolCallRecord(id="tc1", trace_id=trace_id, step=0, parse_status="ok",
                       result_status="ok", result_ref=result_ref, started_at=_ago(60))
    )
    ToolRepo(db).insert_run(
        ToolRunRecord(id="tr1", status="ok", started_at=_ago(60), trace_id=trace_id,
                      output_ref=run_ref)
    )

    result = _apply(db, store)

    assert result.per_rule["tool_call.result_ref"] == 1
    assert result.per_rule["tool_run.output_ref"] == 1
    call = TraceRepo(db).list_tool_calls(trace_id)[0]
    assert call.result_ref is None and call.result_status == "ok"
    assert not store.exists(result_ref) and not store.exists(run_ref)


# ── 保护关系：分数指向的 trace 行不能消失 ──────────────────────────
def test_protected_trace_loses_raw_bodies_but_never_its_row(db, store):
    heavy = store.put_text("评测原始响应 " * 200)
    trace_id = _trace(db, age_days=200, purpose="eval", raw_response_ref=heavy)
    _eval_run(db, [trace_id])

    result = _apply(db, store, purge_traces=True)

    assert result.protected_kept == 1
    assert result.traces_deleted == 0, "分数指向的 trace 被删掉，评测历史就断线了"
    assert result.refs_cleared == 1, "评测跑批才是 .data 涨得最快的一路，原始 body 照样要有期限"
    assert TraceRepo(db).get(trace_id) is not None
    assert not store.exists(heavy)


def test_purge_deletes_children_first_and_unlinks_a_protected_child(db, store):
    parent = _trace(db, age_days=200)
    child = _trace(db, age_days=200, parent_id=parent)
    _eval_run(db, [child])

    usage = UsageRepo(db)
    usage.upsert(UsageRecord(trace_id=parent, source="engine", confidence="high",
                             in_tokens=5, out_tokens=7))
    usage.upsert_alts([UsageAltRecord(trace_id=parent, source="engine", in_tokens=5)])
    usage.replace_parts(parent, [TokenPartRecord(trace_id=parent, part="msg:0", tokens=3)])
    TraceRepo(db).insert_tool_call(
        ToolCallRecord(id="tc1", trace_id=parent, step=0, parse_status="ok")
    )
    TraceRepo(db).insert_anomaly(
        AnomalyRecord(id="an1", code="DRIFT", severity="warn", trace_id=parent, created_at=_ago(200))
    )

    result = _apply(db, store, purge_traces=True)

    assert result.traces_deleted == 1
    assert result.child_rows_deleted == 5
    assert TraceRepo(db).get(parent) is None
    kept = TraceRepo(db).get(child)
    assert kept is not None and kept.parent_id is None, "父行没了链接只能断，否则外键拒绝删除"
    assert usage.fetch(parent).usage is None, "子表不能留下悬空的 trace_id"


def test_purge_is_off_unless_explicitly_asked(db, store):
    _trace(db, age_days=200)
    result = _apply(db, store)
    assert result.traces_deleted == 0
    assert result.traces_aged == 1, "超窗的 trace 数要说得出，即使默认不删"


# ── 孤儿与悬空引用 ─────────────────────────────────────────────────
def test_crash_orphan_is_reclaimed_and_dangling_reference_is_reported(db, store):
    orphan = store.put_text("半截写入的产物 " * 30)
    kept = store.put_text("还在被引用 " * 30)
    gone = store.put_text("会被人为删掉 " * 30)
    _trace(db, age_days=1, messages_ref=kept)
    _trace(db, age_days=1, output_ref=gone)
    store.delete(gone)  # 制造"库里有引用、盘上没有"：doctor 的 blob 完整性项抓的就是它

    result = _apply(db, store)

    assert result.orphans_found == 1 and result.blobs_deleted == 1
    assert not store.exists(orphan)
    assert store.exists(kept)
    assert result.dangling_refs == 1, "引用悬空必须报出来，不能被当成「没东西要删」"


def test_second_run_has_nothing_left_to_do(db, store):
    heavy = store.put_text("x" * 3000)
    _trace(db, age_days=60, raw_request_ref=heavy)
    assert _apply(db, store).reclaimed > 0

    again = _apply(db, store)

    assert again.reclaimed == 0
    assert again.freed_bytes == 0
    assert again.bytes_after == again.bytes_before


# ── 回收上限这道门槛 ───────────────────────────────────────────────
def test_reclaim_above_the_cap_needs_force_and_deletes_nothing(db, store):
    heavy = store.put_text("z" * 8000)
    trace_id = _trace(db, age_days=60, raw_response_ref=heavy)

    blocked = sweep(db, store, now=NOW, dry_run=False)

    assert blocked.blocked and not blocked.applied
    assert f"{MAX_RECLAIM_RATIO:.0%}" in blocked.blocked
    assert store.exists(heavy), "拦住就必须一个字节都没动"
    assert TraceRepo(db).get(trace_id).raw_response_ref == heavy
    assert blocked.refs_cleared == 0 and sum(blocked.per_rule.values()) == 1
    run = history(db)[0]
    assert run["dry_run"] == 0 and run["refs_cleared"] == 0 and run["blobs_deleted"] == 0
    assert json.loads(run["detail_json"])["blocked"]

    done = sweep(db, store, now=NOW, dry_run=False, force=True)
    assert done.applied and not store.exists(heavy)


# ── 体积报告与留痕表自己 ───────────────────────────────────────────
def test_disk_report_describes_the_data_dir(db, store):
    heavy = store.put_text("w" * 1500)
    _trace(db, age_days=60, raw_response_ref=heavy)
    _apply(db, store)

    report = disk_report(db, store)

    assert report.blob_files == 0 and report.blob_bytes == 0
    assert report.traces == 1
    assert report.oldest_trace_at == _ago(60)
    assert report.retention_runs == 1
    assert report.last_run["dry_run"] == 0
    assert report.db_bytes == db.path.stat().st_size
    assert report.total_bytes >= report.blob_bytes


def test_retention_run_itself_is_never_cleaned(db, store):
    """留痕表是审计日志不是数据：跑十次 rotate 之后一条都不能少。"""
    for _ in range(3):
        sweep(db, store, now=NOW, raw_after="0d", dry_run=False, force=True)
    assert int(db.scalar("SELECT COUNT(*) FROM retention_run")) == 3


def test_vacuum_only_runs_when_something_was_deleted(db, store):
    heavy = store.put_text("v" * 4000)
    _trace(db, age_days=60, raw_response_ref=heavy)
    before = db.path.stat().st_size

    result = _apply(db, store, vacuum=True)

    assert result.vacuum is True
    assert db.path.stat().st_size <= before, "VACUUM 之后文件不该变大"
    idle = _apply(db, store, vacuum=True)
    assert idle.vacuum is False, "什么都没删就不该重写整个数据库文件"


# ── 体积曲线（`onyx db sizes` 的数据来源）──────────────────────────
def _seed_run(db, *, started_at: str, before: int, after: int) -> None:
    db.execute(
        "INSERT INTO retention_run(id, started_at, finished_at, dry_run,"
        " trace_after_d, raw_after_d, purge_traces, bytes_before, bytes_after)"
        " VALUES(?,?,?,0,90,30,0,?,?)",
        (new_trace_id(), started_at, started_at, before, after),
    )


def _hours_ago(hours: float) -> str:
    reference = datetime.fromisoformat(NOW).astimezone(UTC)
    return (reference - timedelta(hours=hours)).isoformat(timespec="microseconds")


def test_trend_refuses_a_slope_from_a_two_minute_window(db):
    """相隔 72 秒的两个点也"能"算出每天多少字节，但那是噪声除以时间。

    一个荒谬的斜率比"问不出来"更有害：它看起来像测量结果。
    """
    _seed_run(db, started_at=_hours_ago(0.02), before=1_000, after=1_000)
    _seed_run(db, started_at=NOW, before=1_000, after=1_400)

    trend = footprint_trend(db)

    assert trend.samples == 2
    assert 0 < trend.span_days < 1, "跨度算对了，只是不够长"
    assert trend.enough is False


def test_trend_refuses_to_invent_a_slope(db):
    """0 / 1 个采样点、以及同一时刻的两个点，都给不出日均增速。

    硬算会得到 ±无穷大或一个凭空的斜率；而"照这个速度还能撑 N 天"这种话，
    说错比不说更有害。
    """
    assert footprint_trend(db).enough is False

    _seed_run(db, started_at=NOW, before=1000, after=1000)
    assert footprint_trend(db).enough is False

    _seed_run(db, started_at=NOW, before=2000, after=9000)
    same_instant = footprint_trend(db)
    assert same_instant.samples == 2 and same_instant.enough is False
    assert same_instant.span_days == 0.0


def test_trend_computes_a_daily_rate_and_keeps_the_latest_size(db):
    _seed_run(db, started_at=_ago(10), before=1_000, after=1_000)
    _seed_run(db, started_at=NOW, before=1_000, after=11_000)

    trend = footprint_trend(db)

    assert trend.enough is True
    assert trend.samples == 2
    assert trend.latest_bytes == 11_000
    assert trend.per_day_bytes == pytest.approx(1_000.0)
    assert trend.points[0][0] == _ago(10), "点必须按时间正序，否则曲线是反的"


def test_trend_reports_shrinking_as_a_negative_rate(db):
    _seed_run(db, started_at=_ago(5), before=9_000, after=9_000)
    _seed_run(db, started_at=NOW, before=9_000, after=3_000)

    assert footprint_trend(db).per_day_bytes == pytest.approx(-1_200.0)


# ── CLI：onyx rotate / db info / db sizes ──────────────────────────
@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """数据目录进 tmp，命令默认的 <repo>/.data 一个字节都不许碰。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("COLUMNS", "240")
    return load_settings().ensure_dirs()


def _cli(*argv: str):
    from typer.testing import CliRunner

    return CliRunner().invoke(app, list(argv))


def _seed_cli(data_dir) -> str:
    """一条 60 天前的 trace + 一个重 blob，落在命令默认的路径上。"""
    database = Database(data_dir.db_path)
    store = FileBlobStore(data_dir.blob_dir)
    ref = store.put_text("原始 body " * 600)
    _trace(database, age_days=60, raw_response_ref=ref)
    database.close()
    return ref


def test_cli_rotate_without_apply_deletes_nothing(data_dir):
    ref = _seed_cli(data_dir)
    store = FileBlobStore(data_dir.blob_dir)

    result = _cli("rotate")

    assert result.exit_code == 0, result.output
    assert "dry-run" in result.output
    assert "预计" in result.output, "dry-run 的数字必须标成预计，不能写得像已经删了"
    assert "未删除任何东西" in result.output
    assert store.exists(ref)


def test_cli_rotate_apply_frees_the_bytes(data_dir):
    ref = _seed_cli(data_dir)
    store = FileBlobStore(data_dir.blob_dir)

    result = _cli("rotate", "--apply", "--force")

    assert result.exit_code == 0, result.output
    assert "已" in result.output and "未删除任何东西" not in result.output
    assert not store.exists(ref)


def test_cli_rotate_blocks_an_oversized_single_reclaim(data_dir):
    """一个 blob 就是全部体积 ⇒ 超过单次上限 ⇒ 退出码非 0 且什么都没删。"""
    ref = _seed_cli(data_dir)
    store = FileBlobStore(data_dir.blob_dir)

    result = _cli("rotate", "--apply")

    assert result.exit_code == 1
    assert "--force" in result.output
    assert store.exists(ref), "被拦住就必须一个字节都没动"


def test_cli_rotate_rejects_a_bare_number_window(data_dir):
    _seed_cli(data_dir)
    result = _cli("rotate", "--raw-after", "30")
    assert result.exit_code == 2
    assert "窗口" in (result.output + getattr(result, "stderr", ""))


def test_cli_rotate_json_is_machine_readable(data_dir):
    _seed_cli(data_dir)
    result = _cli("rotate", "--json")
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["dry_run"] is True and body["refs_cleared"] == 0
    assert sum(body["per_rule"].values()) == 1


def test_db_info_says_whether_rotate_ever_ran(data_dir):
    """没跑过 rotate 要说"从没跑过"，而不是显示一行空白让人以为已经清理过了。"""
    Database(data_dir.db_path).close()
    first = _cli("db", "info")
    assert "从没跑过 onyx rotate" in first.output, first.output

    _cli("rotate")
    after = _cli("db", "info")
    assert "dry-run" in after.output
    assert "rotate runs" in after.output


def test_db_sizes_refuses_to_guess_a_trend(data_dir):
    """一个采样点、或两个同一时刻的点，都不能推出"照这个速度还能撑 N 天"。"""
    with Database(data_dir.db_path) as database:
        _seed_run(database, started_at=NOW, before=1_000, after=1_000)

    one_point = _cli("db", "sizes")
    assert one_point.exit_code == 0, one_point.output
    assert "问不出来" in one_point.output

    with Database(data_dir.db_path) as database:
        _seed_run(database, started_at=NOW, before=1_000, after=11_000)
    same_instant = _cli("db", "sizes")
    assert "问不出来" in same_instant.output, "两个点但同一时刻仍然给不出斜率"

    with Database(data_dir.db_path) as database:
        _seed_run(database, started_at=_ago(10), before=500, after=500)
    grown = _cli("db", "sizes").output
    assert "+1.0 KiB/天" in grown, grown
    assert "还很充裕" in grown, "日均 1 KiB 不该报出'还能写 N 天'这种假精确"


def test_db_sizes_gives_a_runway_when_the_disk_is_really_filling(data_dir):
    """一天 10 GiB 的速度，任何单机都会在十年内写满——这时必须给出天数。"""
    with Database(data_dir.db_path) as database:
        _seed_run(database, started_at=_ago(2), before=0, after=0)
        _seed_run(database, started_at=_ago(1), before=0, after=10 * 1024 ** 3)

    urgent = _cli("db", "sizes").output

    assert "还能写约" in urgent, urgent
    assert "GiB/天" in urgent


def test_db_sizes_json_is_machine_readable(data_dir):
    with Database(data_dir.db_path) as database:
        _seed_run(database, started_at=_ago(4), before=2_000, after=2_000)
        _seed_run(database, started_at=NOW, before=2_000, after=6_000)

    result = _cli("db", "sizes", "--json")

    body = json.loads(result.output)
    assert body["retention_runs"] == 2
    assert body["trend"]["samples"] == 2
    assert body["trend"]["per_day_bytes"] == pytest.approx(1_000.0)
    assert body["now"]["total_bytes"] > 0
    assert [p["bytes_after"] for p in body["trend"]["points"]] == [2_000, 6_000]
