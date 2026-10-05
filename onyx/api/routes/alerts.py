"""告警的读端点：触发历史 + 当前生效的判据与出口（S27）。

第二个端点不是装饰。"我没收到通知"有四种完全不同的原因——没命中、被 cooldown 挡了、
渠道失败、这个进程根本没装配出口——只看历史的人会把它们统统读成"告警坏了"。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from onyx.api.deps import AppState, get_state
from onyx.store.records import AlertTriggerRecord

router = APIRouter(prefix="/api", tags=["alerts"])


class AlertTriggerView(BaseModel):
    id: str
    created_at: str
    code: str
    severity: str
    n_in_window: int
    window_s: int
    channel: str
    status: str
    detail: str
    first_anomaly_id: str
    last_anomaly_id: str
    trace_ids: list[str]
    is_test: bool
    #: 命中当时的判据快照。规则以后会被改，"当时为什么触发"只能用当时那份解释
    rule: dict


class AlertStatusView(BaseModel):
    enabled: bool
    channels: list[str]
    poll_s: float
    ticks: int
    last_error: str
    thread_alive: bool
    rule: dict
    #: 这个进程没被装配成会发通知。必须显式说出来，否则界面只有一片空白
    reason: str = ""


def _view(rec: AlertTriggerRecord) -> AlertTriggerView:
    return AlertTriggerView(
        id=rec.id, created_at=rec.created_at, code=rec.code, severity=rec.severity,
        n_in_window=rec.n_in_window, window_s=rec.window_s, channel=rec.channel,
        status=rec.status, detail=rec.detail, first_anomaly_id=rec.first_anomaly_id or "",
        last_anomaly_id=rec.last_anomaly_id or "", trace_ids=list(rec.trace_ids),
        is_test=rec.is_test, rule=rec.rule,
    )


@router.get("/alerts", response_model=list[AlertTriggerView])
def list_alerts(
    state: AppState = Depends(get_state),
    code: str | None = Query(None),
    channel: str | None = Query(None),
    status: str | None = Query(None, description="sent / failed"),
    since: str | None = Query(None, description="ISO 时间，闭区间"),
    include_test: bool = Query(False, description="`onyx alerts test` 造的行默认不混进来"),
    limit: int = Query(50, ge=1, le=500),
) -> list[AlertTriggerView]:
    from onyx.store.repos import AlertRepo

    rows = AlertRepo(state.runtime.db).list_triggers(
        code=code, channel=channel, status=status, since=since,
        include_test=include_test, limit=limit,
    )
    return [_view(r) for r in rows]


@router.get("/alerts/status", response_model=AlertStatusView)
def alerts_status(state: AppState = Depends(get_state)) -> AlertStatusView:
    service = state.alert_service
    if service is None:
        return AlertStatusView(
            enabled=False, channels=[], poll_s=0.0, ticks=0, last_error="",
            thread_alive=False, rule={},
            reason="这个 serve 进程没有装配告警（规则来自 onyx.toml 的 [alerts]）",
        )
    status = service.status()
    return AlertStatusView(
        **status, rule=service.rule.as_dict(),
        reason="" if status["enabled"] else "[alerts].enabled = false",
    )
