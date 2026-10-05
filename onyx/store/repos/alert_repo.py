"""告警触发历史的读写。

这张表只回答一个问题：**某天某条规则到底通知没通知**。
所以它记"命中 + 尝试投递"，渠道失败也要留下一行带原因的记录；
被 cooldown 抑制的不写——那是没发生的事，写进去只会把审计淹没成心跳日志。
"""

from __future__ import annotations

from collections.abc import Sequence

from onyx.store.codec import dumps, loads_dict, loads_list
from onyx.store.db import Database
from onyx.store.records import AlertTriggerRecord

_COLUMNS: tuple[str, ...] = (
    "id", "created_at", "code", "severity", "rule_json", "n_in_window", "window_s",
    "first_anomaly_id", "last_anomaly_id", "trace_ids_json", "channel", "status",
    "detail", "is_test",
)


class AlertRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_trigger(self, rec: AlertTriggerRecord) -> None:
        self.db.execute(
            f"""INSERT INTO alert_trigger({','.join(_COLUMNS)})
                VALUES({','.join("?" * len(_COLUMNS))})
                ON CONFLICT(id) DO UPDATE SET status=excluded.status, detail=excluded.detail""",
            (
                rec.id, rec.created_at, rec.code, rec.severity, dumps(rec.rule),
                rec.n_in_window, rec.window_s, rec.first_anomaly_id, rec.last_anomaly_id,
                dumps(list(rec.trace_ids)), rec.channel, rec.status, rec.detail,
                1 if rec.is_test else 0,
            ),
        )

    def list_triggers(
        self,
        *,
        code: str | None = None,
        channel: str | None = None,
        status: str | None = None,
        since: str | None = None,
        include_test: bool = True,
        limit: int = 50,
    ) -> list[AlertTriggerRecord]:
        sql = "SELECT * FROM alert_trigger WHERE 1=1"
        params: list[object] = []
        if code:
            sql += " AND code=?"
            params.append(code)
        if channel:
            sql += " AND channel=?"
            params.append(channel)
        if status:
            sql += " AND status=?"
            params.append(status)
        if since:
            sql += " AND created_at>=?"
            params.append(since)
        if not include_test:
            sql += " AND is_test=0"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [self._row(r) for r in self.db.query(sql, tuple(params))]

    def last_real_trigger_at(self, code: str) -> str | None:
        """这条规则上次**真的投过**的时间（测试行不算）。

        cooldown 以库里的时间为判据而不是进程内的钟：服务重启不该让同一条异常
        立刻再通知一遍，而"重启前刚发过"这件事只有库知道。
        """
        rows = self.db.query(
            "SELECT created_at FROM alert_trigger WHERE code=? AND is_test=0 "
            "ORDER BY id DESC LIMIT 1",
            (code,),
        )
        return str(rows[0]["created_at"]) if rows else None

    def counts_by_status(self, *, since: str | None = None) -> dict[str, int]:
        sql = "SELECT status, channel, COUNT(*) AS n FROM alert_trigger WHERE is_test=0"
        params: Sequence[object] = ()
        if since:
            sql += " AND created_at>=?"
            params = (since,)
        sql += " GROUP BY status, channel"
        return {
            f"{r['status']}:{r['channel']}": int(r["n"]) for r in self.db.query(sql, params)
        }

    @staticmethod
    def _row(r: dict) -> AlertTriggerRecord:
        return AlertTriggerRecord(
            id=str(r["id"]), created_at=str(r["created_at"]),
            code=str(r["code"]), severity=str(r["severity"]),
            rule=loads_dict(r["rule_json"]),
            n_in_window=int(r["n_in_window"]), window_s=int(r["window_s"]),
            first_anomaly_id=r["first_anomaly_id"], last_anomaly_id=r["last_anomaly_id"],
            trace_ids=tuple(str(t) for t in loads_list(r["trace_ids_json"])),
            channel=str(r["channel"]), status=str(r["status"]),
            detail=str(r["detail"] or ""), is_test=bool(r["is_test"]),
        )
