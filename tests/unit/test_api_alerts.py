"""S27 验收：告警的 HTTP 面与 serve 里的轮询线程。

盯三件事：
- **没装配也要能回答**。`alert_service is None` 时 `/api/alerts/status` 必须说"这个进程没装配"，
  而不是 500 或者一片空白——"我没收到通知"的第一种原因就是根本没人发。
- **测试行不混进真实历史**（默认 `include_test=false`）。
- **线程真的在跑**：用 TestClient 走一遍 lifespan，插一条异常，等文件出口落下那一行。
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.llm.providers.mock import MockScript
from onyx.obs.alerts.channels import FileChannel
from onyx.obs.alerts.rules import AlertRule
from onyx.runtime import build_runtime
from onyx.settings import load_settings
from onyx.store.records import AlertTriggerRecord, AnomalyRecord
from onyx.store.repos import AlertRepo, TraceRepo

MODEL = "mock/echo"
SCRIPTS = {MODEL: MockScript(text="转账", in_tokens=10, out_tokens=2, done_reason="stop")}


def _runtime(tmp_path):
    return build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": (MODEL,)},
    )


def _trigger(i: str, *, channel: str = "file", status: str = "sent",
             code: str = "CONTEXT_OVERFLOW", is_test: bool = False,
             at: str = "2026-10-05T03:00:00+00:00") -> AlertTriggerRecord:
    return AlertTriggerRecord(
        id=i, created_at=at, code=code, severity="error",
        rule={"min_count": 1, "window_s": 300, "source": "default"},
        n_in_window=2, window_s=300, first_anomaly_id="a1", last_anomaly_id="a2",
        trace_ids=("t1", "t2"), channel=channel, status=status,
        detail="写入 1 行" if status == "sent" else "ConnectError: 拒绝连接",
        is_test=is_test,
    )


@pytest.fixture
def client(tmp_path):
    runtime = _runtime(tmp_path)
    app = create_app(runtime=runtime, gpu_lock_path=tmp_path / "gpu.lock", sample_gpu=False)
    with TestClient(app) as c:
        c.runtime = runtime  # type: ignore[attr-defined]
        yield c
    runtime.close()


def test_history_is_empty_but_answerable(client):
    body = client.get("/api/alerts").json()
    assert body == []
    assert client.get("/api/alerts").status_code == 200


def test_status_says_this_process_has_no_alerts(client):
    """没装配出口时要说破。空白页会被读成"告警坏了"，而真相是"这个进程从不发通知"。"""
    body = client.get("/api/alerts/status").json()
    assert body["enabled"] is False and body["channels"] == []
    assert "[alerts]" in body["reason"]


def test_rows_come_back_with_the_rule_snapshot(client):
    repo = AlertRepo(client.runtime.db)
    repo.insert_trigger(_trigger("r1"))
    repo.insert_trigger(_trigger("r2", channel="webhook", status="failed"))
    repo.insert_trigger(_trigger("r3", is_test=True))

    rows = client.get("/api/alerts").json()
    assert [r["id"] for r in rows] == ["r2", "r1"], "测试行默认不混进真实历史"
    assert rows[0]["status"] == "failed" and "ConnectError" in rows[0]["detail"]
    assert rows[0]["rule"]["window_s"] == 300, "解释「当时为什么触发」用的是当时的判据"
    assert rows[0]["trace_ids"] == ["t1", "t2"]

    everything = client.get("/api/alerts?include_test=true").json()
    assert len(everything) == 3
    assert client.get("/api/alerts?status=sent").json()[0]["id"] == "r1"
    assert [r["id"] for r in client.get("/api/alerts?channel=webhook").json()] == ["r2"]
    assert client.get("/api/alerts?code=TOOL_LOOP").json() == []


def test_since_filter_and_limit_are_honoured(client):
    repo = AlertRepo(client.runtime.db)
    repo.insert_trigger(_trigger("old", at="2026-10-04T03:00:00+00:00"))
    repo.insert_trigger(_trigger("new", at="2026-10-05T03:00:00+00:00"))
    got = client.get("/api/alerts?since=2026-10-05T00:00:00%2B00:00").json()
    assert [r["id"] for r in got] == ["new"]
    assert len(client.get("/api/alerts?limit=1").json()) == 1


def test_bad_limit_is_rejected_not_silently_clamped(client):
    assert client.get("/api/alerts?limit=0").status_code == 422
    assert client.get("/api/alerts?limit=5000").status_code == 422


# ── 装配起来的那条路径 ────────────────────────────────────────────
def test_assembled_service_reports_its_rule_and_channels(tmp_path):
    runtime = _runtime(tmp_path)
    path = tmp_path / "alerts" / "alerts.jsonl"
    app = create_app(
        runtime=runtime, gpu_lock_path=tmp_path / "g.lock", sample_gpu=False,
        alert_rule=AlertRule(min_count=2, source="file"),
        alert_channels=(FileChannel(path),),
    )
    with TestClient(app) as c:
        body = c.get("/api/alerts/status").json()
    assert body["enabled"] is True and body["channels"] == ["file"]
    assert body["rule"]["min_count"] == 2 and body["rule"]["source"] == "file"
    assert body["poll_s"] == 5.0
    runtime.close()


def test_the_poll_thread_actually_delivers(tmp_path):
    """DoD 的最小版本：异常落库之后，没人点任何东西，文件出口就该在几分钟内多一行。

    cooldown 用默认值（600s），所以一条在窗口里反复存在的异常**只投一次**——
    这条测试同时把这件事钉住：反复投会把人训练成忽略通知。
    """
    runtime = _runtime(tmp_path)
    path = tmp_path / "alerts" / "alerts.jsonl"
    app = create_app(
        runtime=runtime, gpu_lock_path=tmp_path / "g.lock", sample_gpu=False,
        alert_rule=AlertRule(poll_s=0.05),
        alert_channels=(FileChannel(path),),
    )
    with TestClient(app) as c:
        now = datetime.now(UTC)
        TraceRepo(runtime.db).insert_anomaly(AnomalyRecord(
            id="a1", code="CONTEXT_OVERFLOW", severity="error", trace_id="tr-1",
            created_at=now.isoformat(),
        ))
        deadline = time.monotonic() + 10.0
        lines: list[dict] = []
        while time.monotonic() < deadline:
            files = list((tmp_path / "alerts").glob("*.jsonl")) if (tmp_path / "alerts").exists() else []
            lines = [json.loads(text) for f in files for text in
                     f.read_text(encoding="utf-8").splitlines() if text.strip()]
            if lines:
                break
            time.sleep(0.05)
        assert lines, "后台线程没把这条异常通知出去"
        assert lines[0]["code"] == "CONTEXT_OVERFLOW"
        assert "建议" in lines[0]["message"], "通知要带上下一步做什么（文案来自 SPECS，不重写）"

        history = c.get("/api/alerts").json()
        assert [r["status"] for r in history] == ["sent"]
        # 再等几个周期：窗口里那条异常还在，但 cooldown 不该让它再发一遍
        time.sleep(0.4)
        assert len(c.get("/api/alerts").json()) == 1, "cooldown 内重复投递"
    runtime.close()
