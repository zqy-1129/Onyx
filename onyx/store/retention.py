"""数据生命周期：分数永久、证据有限期。

为什么默认**摘引用而不是删行**：吃磁盘的是 blob（原始 HTTP body、渲染后的 prompt、
工具返回值），trace 行本身很轻。删行会让 `grade.trace_id` 指向不存在的 trace，
评测历史就此断线；而摘掉三个重引用能收回绝大部分空间，下钻要看的
messages / output / 计数 / 延迟一个都没少。

为什么 dry-run 与 apply **走同一段代码**：dry-run 的全部价值是"报得出会删多少"。
估算与删除各写一套，迟早会对不上（数字好看、真跑炸了）。所以这里让 dry-run
真的执行删除，只在事务末尾回滚，并且一个文件都不碰。

为什么每次运行都写 `retention_run`（dry-run 也写）：事后有人问"三周前那次原始
body 怎么没了"，答案必须是一条记录，而不是"大概跑了 rotate 吧"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.core.content import BlobStore
from onyx.core.ids import new_trace_id
from onyx.store.codec import dumps
from onyx.store.db import Database

DEFAULT_RAW_AFTER = "30d"
DEFAULT_TRACE_AFTER = "90d"

#: 单次回收的上限比例。超过它说明"窗口"或"数据规模"和想象的不一样，
#: 必须有人明确说知道（`--force`）才动手——删掉的原始 body 不会自己回来。
MAX_RECLAIM_RATIO = 0.6

_UNIT = {"h": timedelta(hours=1), "d": timedelta(days=1), "w": timedelta(weeks=1)}
_WINDOW_RE = re.compile(r"^(\d+)([dhw])$")


def parse_window(text: str) -> timedelta:
    """`90d` / `24h` / `2w` → timedelta。

    不接受裸数字：`30` 是 30 天还是 30 秒只有写它的人知道，而猜错方向的代价是永久删证据。
    """
    match = _WINDOW_RE.match((text or "").strip().lower())
    if not match:
        raise ValueError(f"窗口要写成 <整数><d|h|w>，例如 90d / 24h / 2w；实际是 {text!r}")
    return int(match.group(1)) * _UNIT[match.group(2)]


@dataclass(frozen=True, slots=True)
class _Rule:
    """一个"重证据"引用列：超出 raw 窗口就摘掉。表名/列名都是本文件写死的常量。"""

    table: str
    column: str
    #: 这条记录归属的时刻。tool_call 自己的 started_at 可空，所以跟着父 trace 的时间走。
    time_sql: str


RAW_RULES: tuple[_Rule, ...] = (
    _Rule("trace", "raw_request_ref", "started_at"),
    _Rule("trace", "raw_response_ref", "started_at"),
    _Rule("trace", "rendered_prompt_ref", "started_at"),
    _Rule(
        "tool_call",
        "result_ref",
        "(SELECT t.started_at FROM trace t WHERE t.id = tool_call.trace_id)",
    ),
    _Rule("tool_run", "output_ref", "started_at"),
)

#: 所有可能指向 blob 的列。回收的唯一判据是**库里还有没有任何一行引用它**，
#: 所以这张清单必须完整：漏一列，那一列引用的 blob 就会被当孤儿删掉。
#: `tool_def.impl_ref` 刻意不在这里——那是 `mcp:server:tool` 这类实现指针，不是内容寻址引用。
ALL_REF_COLUMNS: tuple[tuple[str, str], ...] = (
    ("trace", "messages_ref"),
    ("trace", "tools_ref"),
    ("trace", "rendered_prompt_ref"),
    ("trace", "output_ref"),
    ("trace", "raw_request_ref"),
    ("trace", "raw_response_ref"),
    ("tool_call", "result_ref"),
    ("tool_run", "output_ref"),
    ("model", "tokenizer_ref"),
)

#: 被结论引用的 trace：行永不删除。grade / tool_run 侧没有外键（见 0004 注释——
#: 故意不设，好让清理 trace 不牵连评测历史），所以保护关系只能显式写在这里。
#: 摘引用不受此限：评测跑批才是 `.data` 增长最快的那一路，留着原始 body 等于没做保留策略。
PROTECTED_SQL = """
    SELECT trace_id FROM grade WHERE trace_id IS NOT NULL
    UNION
    SELECT trace_id FROM tool_run WHERE trace_id IS NOT NULL
    UNION
    SELECT id FROM trace WHERE eval_run_id IS NOT NULL
"""

#: 删 trace 行时必须一起清掉的子表（`trace_id` 列名在各表一致）。
#: 顺序就是这里的顺序：先无外键的 anomaly，再有外键的四张，最后才动 trace 自己。
_CHILD_TABLES: tuple[str, ...] = ("anomaly", "tool_call", "token_part", "usage_alt", "usage")


class _Rollback(Exception):
    """内部信号：dry-run 用真删除算准数字，再靠它把事务撤掉。"""


@dataclass(frozen=True, slots=True)
class Outcome:
    """一次保留策略运行的全部数字（dry-run 报的是"将会是多少"）。"""

    run_id: str
    started_at: str
    dry_run: bool
    applied: bool
    blocked: str
    raw_after: str
    trace_after: str
    raw_cutoff: str
    trace_cutoff: str
    purge_traces: bool
    vacuum: bool
    #: 事实：本次真的摘掉了几处引用。dry-run 与被拦下时是 0——"预计要摘"看 `per_rule`。
    refs_cleared: int
    traces_aged: int
    traces_deleted: int
    child_rows_deleted: int
    protected_kept: int
    blobs_deleted: int
    orphans_found: int
    reclaimable_bytes: int
    freed_bytes: int
    dangling_refs: int
    bytes_before: int
    bytes_after: int
    #: 本次算出的"每一列要摘多少"。dry-run 与被拦下时，这就是全部真相；
    #: 审计列写的是事实，估算只活在这里和 `retention_run.detail_json`。
    per_rule: dict[str, int] = field(default_factory=dict)

    @property
    def reclaimed(self) -> int:
        """这次动到的总量——全 0 就是"无事可做"，报告里要能一眼看出来。"""
        return self.refs_cleared + self.blobs_deleted + self.traces_deleted

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "blocked": self.blocked,
            "windows": {"raw_after": self.raw_after, "trace_after": self.trace_after},
            "cutoffs": {"raw": self.raw_cutoff, "trace": self.trace_cutoff},
            "purge_traces": self.purge_traces,
            "vacuum": self.vacuum,
            "refs_cleared": self.refs_cleared,
            "per_rule": dict(self.per_rule),
            "traces_aged": self.traces_aged,
            "traces_deleted": self.traces_deleted,
            "child_rows_deleted": self.child_rows_deleted,
            "protected_kept": self.protected_kept,
            "blobs_deleted": self.blobs_deleted,
            "orphans_found": self.orphans_found,
            "reclaimable_bytes": self.reclaimable_bytes,
            "freed_bytes": self.freed_bytes,
            "dangling_refs": self.dangling_refs,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
        }


@dataclass(frozen=True, slots=True)
class DiskReport:
    """`.data` 的体积现状：`db info`（以及后续 `doctor` 的磁盘项）从这里取数。"""

    db_bytes: int
    wal_bytes: int
    blob_bytes: int
    blob_files: int
    traces: int
    oldest_trace_at: str | None
    dangling_refs: int
    retention_runs: int
    last_run: dict[str, Any] | None = None

    @property
    def total_bytes(self) -> int:
        return self.db_bytes + self.wal_bytes + self.blob_bytes


def _moment(now: str | None) -> datetime:
    if not now:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(now)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _cutoff(moment: datetime, window: str) -> str:
    """cutoff 必须与 `started_at` 同格式：TEXT 列按字典序比，格式不一致就会比错。"""
    return (moment - parse_window(window)).isoformat(timespec="microseconds")


def _days(window: str) -> int:
    """窗口换算成整型天数落进审计列（小时级窗口原样存在 detail_json 里，不会丢信息）。"""
    return int(parse_window(window).total_seconds() // 86400)


def referenced_refs(db: Database) -> set[str]:
    """库里当前仍在引用的 blob ref。"""
    found: set[str] = set()
    for table, column in ALL_REF_COLUMNS:
        rows = db.query(
            f"SELECT DISTINCT {column} AS ref FROM {table} WHERE {column} LIKE 'sha256:%'"
        )
        found.update(str(row["ref"]) for row in rows)
    return found


def _clear_raw_refs(db: Database, cutoff: str) -> dict[str, int]:
    per_rule: dict[str, int] = {}
    for rule in RAW_RULES:
        cursor = db.execute(
            f"UPDATE {rule.table} SET {rule.column}=NULL "
            f"WHERE {rule.column} IS NOT NULL AND {rule.column}<>'' "
            f"AND {rule.time_sql}<?",
            (cutoff,),
        )
        per_rule[f"{rule.table}.{rule.column}"] = max(0, cursor.rowcount)
    return per_rule


def _purge_traces(db: Database, cutoff: str) -> tuple[int, int]:
    """删除超窗且未被结论引用的 trace，连带子表行。返回 (trace 行数, 子表行数)。

    先置空父子链接再删：被保护的孩子可以活下来，但它父亲的行要没了，不断链外键会直接拒绝。
    待删集合的谓词不含 `parent_id`，所以这一步不会让它自己变化。
    """
    aged = f"(SELECT id FROM trace WHERE started_at<? AND id NOT IN ({PROTECTED_SQL}))"
    db.execute(f"UPDATE trace SET parent_id=NULL WHERE parent_id IN {aged}", (cutoff,))
    children = 0
    for table in _CHILD_TABLES:
        cursor = db.execute(f"DELETE FROM {table} WHERE trace_id IN {aged}", (cutoff,))
        children += max(0, cursor.rowcount)
    cursor = db.execute(
        f"DELETE FROM trace WHERE started_at<? AND id NOT IN ({PROTECTED_SQL})", (cutoff,)
    )
    return max(0, cursor.rowcount), children


def _orphans(db: Database, store: BlobStore) -> tuple[list[str], int, int]:
    """(无人引用的 ref, 它们的字节数, 库里引用了但盘上没有的 ref 数)。"""
    referenced = referenced_refs(db)
    on_disk = set(store.iter_refs())
    found = sorted(on_disk - referenced)
    return found, sum(store.stat(ref).size for ref in found), len(referenced - on_disk)


def _record(db: Database, outcome: Outcome, orphan_sample: list[str]) -> None:
    """留痕写在**事务之外**：dry-run 回滚了删除，但"算过这一次"本身必须留下。"""
    detail = {
        # 窗口原样存一份：24h 落在整型天数列里会变成 0，光看列会误读
        "windows": {"raw_after": outcome.raw_after, "trace_after": outcome.trace_after},
        "per_rule": outcome.per_rule,
        "orphans": orphan_sample[:20],
        "blocked": outcome.blocked,
        "vacuum": outcome.vacuum,
        "freed_bytes": outcome.freed_bytes,
    }
    with db.transaction():
        db.execute(
            """INSERT INTO retention_run(id, started_at, finished_at, dry_run, trace_after_d,
                                        raw_after_d, purge_traces, refs_cleared, traces_deleted,
                                        blobs_deleted, orphans_found, bytes_before, bytes_after,
                                        detail_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                outcome.run_id,
                outcome.started_at,
                utc_now_iso(),
                1 if outcome.dry_run else 0,
                _days(outcome.trace_after),
                _days(outcome.raw_after),
                1 if outcome.purge_traces else 0,
                outcome.refs_cleared,
                outcome.traces_deleted,
                outcome.blobs_deleted,
                outcome.orphans_found,
                outcome.bytes_before,
                outcome.bytes_after,
                dumps(detail),
            ),
        )


def sweep(
    db: Database,
    store: BlobStore,
    *,
    raw_after: str = DEFAULT_RAW_AFTER,
    trace_after: str = DEFAULT_TRACE_AFTER,
    purge_traces: bool = False,
    dry_run: bool = True,
    force: bool = False,
    now: str | None = None,
    vacuum: bool = False,
) -> Outcome:
    """按窗口摘证据 / 删行，然后回收不再被任何一行引用的 blob。

    默认 dry-run：只算不删。`purge_traces` 要显式打开，因为它删的是行——
    `onyx traces ls` 的历史变短是另一个决定，不是清理磁盘的副产品。
    """
    moment = _moment(now)
    raw_cutoff = _cutoff(moment, raw_after)
    trace_cutoff = _cutoff(moment, trace_after)

    bytes_before = store.total_bytes()
    per_rule: dict[str, int] = {}
    traces_aged = 0
    protected_kept = 0
    traces_deleted = 0
    child_rows_deleted = 0
    orphans: list[str] = []
    reclaimable = 0
    dangling = 0
    blocked = ""

    try:
        with db.transaction():
            per_rule = _clear_raw_refs(db, raw_cutoff)
            traces_aged = int(db.scalar("SELECT COUNT(*) FROM trace WHERE started_at<?",
                                        (raw_cutoff,), 0))
            protected_kept = int(db.scalar(
                f"SELECT COUNT(*) FROM trace WHERE started_at<? AND id IN ({PROTECTED_SQL})",
                (trace_cutoff,), 0,
            ))
            if purge_traces:
                traces_deleted, child_rows_deleted = _purge_traces(db, trace_cutoff)
            orphans, reclaimable, dangling = _orphans(db, store)
            if dry_run:
                raise _Rollback
            if bytes_before and reclaimable > MAX_RECLAIM_RATIO * bytes_before and not force:
                blocked = (
                    f"本次将回收 {reclaimable} B，占现有 blob 体积（{bytes_before} B）的 "
                    f"{reclaimable / bytes_before:.0%}，超过单次上限 "
                    f"{MAX_RECLAIM_RATIO:.0%}。确认窗口没写错再加 --force。"
                )
                raise _Rollback
    except _Rollback:
        pass

    freed = 0
    blobs_deleted = 0
    acted = not dry_run and not blocked
    if acted:
        # 事务已提交才动文件：反过来的顺序会留下"引用还在、文件已没了"的证据空洞。
        for ref in orphans:
            freed += store.delete(ref)
        blobs_deleted = len(orphans)
    did_delete = acted and (traces_deleted or sum(per_rule.values()))
    if vacuum and did_delete:
        # 删行不会让 .sqlite 变小——页还躺在文件里，要真收回磁盘必须 VACUUM（重写整文件）。
        db.execute("VACUUM")

    outcome = Outcome(
        run_id=new_trace_id(),
        started_at=moment.isoformat(timespec="microseconds"),
        dry_run=dry_run,
        applied=acted,
        blocked=blocked,
        raw_after=raw_after,
        trace_after=trace_after,
        raw_cutoff=raw_cutoff,
        trace_cutoff=trace_cutoff,
        purge_traces=purge_traces,
        vacuum=bool(vacuum and did_delete),
        # 事实列：回滚了就是 0。"本来要摘多少"看 per_rule，两者搞混会让审计表说谎。
        refs_cleared=sum(per_rule.values()) if acted else 0,
        traces_aged=traces_aged,
        traces_deleted=traces_deleted if acted else 0,
        child_rows_deleted=child_rows_deleted if acted else 0,
        protected_kept=protected_kept,
        blobs_deleted=blobs_deleted,
        orphans_found=len(orphans),
        reclaimable_bytes=reclaimable,
        freed_bytes=freed,
        dangling_refs=dangling,
        bytes_before=bytes_before,
        bytes_after=store.total_bytes(),
        per_rule=per_rule,
    )
    _record(db, outcome, orphans)
    return outcome


def history(db: Database, *, limit: int = 10) -> list[dict[str, Any]]:
    """最近的保留策略运行记录（新→旧）。`retention_run` 自己永不参与清理。"""
    rows = db.query(
        "SELECT * FROM retention_run ORDER BY started_at DESC, id DESC LIMIT ?", (limit,)
    )
    return [dict(row) for row in rows]


def disk_report(db: Database, store: BlobStore) -> DiskReport:
    path = db.path
    wal = Path(str(path) + "-wal")
    stats = db.query_one("SELECT COUNT(*) AS n, MIN(started_at) AS oldest FROM trace")
    runs = int(db.scalar("SELECT COUNT(*) FROM retention_run", (), 0))
    referenced = referenced_refs(db)
    on_disk = set(store.iter_refs())
    return DiskReport(
        db_bytes=path.stat().st_size if path.exists() else 0,
        wal_bytes=wal.stat().st_size if wal.exists() else 0,
        blob_bytes=store.total_bytes(),
        blob_files=len(on_disk),
        traces=int(stats["n"]) if stats else 0,
        oldest_trace_at=str(stats["oldest"]) if stats and stats["oldest"] else None,
        dangling_refs=len(referenced - on_disk),
        retention_runs=runs,
        last_run=history(db, limit=1)[0] if runs else None,
    )
