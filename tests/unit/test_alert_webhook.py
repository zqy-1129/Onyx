"""S28 验收：webhook 出口。

全程用 `httpx.MockTransport` 打假传输层，不发一个真请求（与契约矩阵的 http 列同一套做法）。
盯的是四件容易在真出事那天才发现的事：
- 失败必须**落成一行记录**而不是重试到永远；
- 4xx 不该重试（对方明说"不收"，重试只是刷日志）；
- URL 的 query 里常挂着 secret，它不许出现在库里的 detail、终端或日志里；
- 载荷与文件出口同形，且**不含 trace 正文**。
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

from onyx.config import load_config
from onyx.obs.alerts.channels import Alert, ChannelResult
from onyx.obs.alerts.service import build_channels
from onyx.obs.alerts.webhook import RETRY_BACKOFF_S, WebhookChannel, redact

URL = "https://hooks.example.test/onyx?key=sup3r-s3cret"


def _alert(**kw) -> Alert:
    base = dict(
        code="CONTEXT_OVERFLOW", severity="error", message="输入超过实际载入上下文",
        n=3, window_s=300, triggered_at="2026-10-05T03:00:00+00:00",
        rule={"min_count": 1, "window_s": 300, "source": "file"},
        first_anomaly_id="a1", last_anomaly_id="a3", trace_ids=("t1", "t2"),
    )
    base.update(kw)
    return Alert(**base)


class _Recorder:
    def __init__(self, responses):
        self.requests: list[httpx.Request] = []
        self._responses = list(responses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        outcome = self._responses.pop(0) if self._responses else httpx.Response(200)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def channel(self, **kw) -> WebhookChannel:
        return WebhookChannel(URL, transport=httpx.MockTransport(self.handler),
                              sleep=lambda _s: None, **kw)


def test_2xx_is_a_successful_delivery_and_hides_the_secret() -> None:
    rec = _Recorder([httpx.Response(204)])
    result = rec.channel().deliver(_alert())
    assert result.ok is True
    assert "204" in result.detail
    assert "sup3r-s3cret" not in result.detail, "detail 会进库、会出现在终端里"
    assert len(rec.requests) == 1


def test_payload_is_the_same_shape_as_the_file_channel_and_carries_no_bodies() -> None:
    rec = _Recorder([httpx.Response(200)])
    rec.channel().deliver(_alert())
    body = json.loads(rec.requests[0].content.decode("utf-8"))
    assert set(body) == {
        "code", "severity", "message", "n", "n_is_lower_bound", "window_s", "triggered_at",
        "rule", "first_anomaly_id", "last_anomaly_id", "trace_ids", "is_test",
    }
    assert body["trace_ids"] == ["t1", "t2"], "给定位信息，不给正文"
    assert not any(k in body for k in ("prompt", "output", "raw_request", "args")), \
        "webhook 是往机器外面送的东西，正文里可能有用户输入"
    assert rec.requests[0].headers["content-type"] == "application/json"


def test_5xx_is_retried_but_bounded() -> None:
    rec = _Recorder([
        httpx.Response(502, text="bad gateway"),
        httpx.Response(502, text="bad gateway"),
        httpx.Response(200),
    ])
    result = rec.channel(retries=2).deliver(_alert())
    assert result.ok is True and len(rec.requests) == 3
    assert "重试 2 次" in result.detail


def test_retries_run_out_and_the_failure_is_reported() -> None:
    rec = _Recorder([httpx.Response(500), httpx.Response(500), httpx.Response(500)])
    result = rec.channel(retries=2).deliver(_alert())
    assert result.ok is False
    assert "HTTP 500" in result.detail
    assert len(rec.requests) == 3, "有上限：无限重试会让「哪些异常没通知」变成不可查"
    assert "sup3r-s3cret" not in result.detail


def test_4xx_is_not_retried() -> None:
    """对方明说"不收"（地址写错、鉴权失败）时重试只是刷日志。"""
    rec = _Recorder([httpx.Response(404, text="not found")])
    result = rec.channel(retries=3).deliver(_alert())
    assert result.ok is False and "404" in result.detail
    assert len(rec.requests) == 1


def test_network_error_is_reported_with_its_type() -> None:
    rec = _Recorder([httpx.ConnectError("connection refused"),
                     httpx.ConnectError("connection refused")])
    result = rec.channel(retries=1).deliver(_alert())
    assert result.ok is False
    assert "ConnectError" in result.detail
    assert len(rec.requests) == 2


def test_a_misconfigured_url_fails_at_assembly_not_at_the_first_incident() -> None:
    for bad in ("", "hooks.example.test/onyx", "ftp://x/y"):
        with pytest.raises(ValueError, match="http"):
            WebhookChannel(bad)


def test_redact_keeps_the_host_but_drops_the_query() -> None:
    assert redact(URL) == "https://hooks.example.test/onyx"
    assert redact("https://a.test/p#frag") == "https://a.test/p"
    assert "secret" not in redact(URL)
    # 不是 URL 的字符串照样截断，不把整段原文带进终端
    assert len(redact("x" * 300)) <= 80
    # urlsplit 对畸形地址会抛 ValueError（端口位置写了非法字符），那也不能把原文吐回去
    assert "无法解析" in redact("http://[::1")


def test_stop_closes_the_webhook_client(tmp_path) -> None:
    """serve 关停时回收 httpx 连接池：`stop()` 不关就等于每次重启漏一个池。

    不去断言"关掉之后再发会报错"——httpx 会自己重开连接池，那是它的内部行为不是我们的契约。
    """
    from onyx.obs.alerts.rules import AlertRule
    from onyx.obs.alerts.service import AlertService
    from onyx.store.db import Database

    closes: list[str] = []

    class _Spy(WebhookChannel):
        def close(self) -> None:
            closes.append(self.name)
            super().close()

    class _BadClose:
        name = "bad"

        def deliver(self, alert: Alert) -> ChannelResult:
            raise NotImplementedError

        def close(self) -> None:
            raise RuntimeError("关不掉")

    db = Database(tmp_path / "s.sqlite")
    try:
        service = AlertService(db, rule=AlertRule(),
                               channels=[_Spy(URL, transport=httpx.MockTransport(
                                   lambda r: httpx.Response(200))), _BadClose()])
        service.stop()  # 一个出口关不掉，不该把关停过程变成崩溃
        assert closes == ["webhook"], "实现了 close() 的出口要被回收；没实现的跳过"
    finally:
        db.close()


def test_the_default_backoff_really_waits(tmp_path) -> None:
    """默认退避走的是真 sleep：测试里注入的 sleep 掩盖不了它写错一次参数。"""
    started = time.monotonic()
    rec = _Recorder([httpx.Response(500), httpx.Response(200)])
    channel = WebhookChannel(URL, transport=httpx.MockTransport(rec.handler), retries=1)
    result = channel.deliver(_alert())
    assert result.ok is True and len(rec.requests) == 2
    assert time.monotonic() - started >= RETRY_BACKOFF_S[0] * 0.6, "第一次重试前该等一下"
    channel.close()


# ── 装配（env 优先于文件；没配 URL 就只有一个出口）──────────────────
def test_webhook_is_only_installed_when_a_url_is_given(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ONYX_ALERT_WEBHOOK_URL", raising=False)
    path = tmp_path / "onyx.toml"
    path.write_text("[alerts]\nwindow_s = 300\n", encoding="utf-8")
    assert [c.name for c in build_channels(load_config(path), tmp_path)] == ["file"]


def test_env_url_beats_the_file_and_is_never_printed(tmp_path, monkeypatch) -> None:
    from onyx.config import ENV_ALERT_WEBHOOK, effective

    path = tmp_path / "onyx.toml"
    path.write_text('[alerts]\nwebhook_url = "https://from-file.test/hook"\n', encoding="utf-8")
    monkeypatch.setenv(ENV_ALERT_WEBHOOK, URL)
    cfg = load_config(path)
    channels = build_channels(cfg, tmp_path)
    assert [c.name for c in channels] == ["file", "webhook"]
    assert channels[1].url == URL, "env 赢过文件（flag > 环境 > 文件 > 默认）"

    row = next(r for r in effective(cfg) if r.key == "alerts.webhook_url")
    assert row.source == "env"
    assert URL not in str(row.value) and "sup3r-s3cret" not in str(row.value)
    assert "已设置" in str(row.value)


def test_file_url_is_used_when_no_env_is_set(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ONYX_ALERT_WEBHOOK_URL", raising=False)
    path = tmp_path / "onyx.toml"
    path.write_text('[alerts]\nwebhook_url = "https://from-file.test/hook"\n', encoding="utf-8")
    channels = build_channels(load_config(path), tmp_path)
    assert channels[1].url == "https://from-file.test/hook"


def test_rule_snapshot_records_env_as_the_provenance(tmp_path, monkeypatch) -> None:
    """`rule.source` 说的是"这套判据从哪来"：文件里写过任何 [alerts] 键就是 file，
    只给了 env URL 是 env，两者都没有是 default。落库的快照里带着它，
    事后才知道该去改哪一层。
    """
    from onyx.config import ENV_ALERT_WEBHOOK, Config, alert_rule

    monkeypatch.setenv(ENV_ALERT_WEBHOOK, URL)
    assert alert_rule(Config()).source == "env"
    monkeypatch.delenv(ENV_ALERT_WEBHOOK)
    assert alert_rule(Config()).source == "default"

    path = tmp_path / "onyx.toml"
    path.write_text("[alerts]\nmin_count = 2\n", encoding="utf-8")
    monkeypatch.setenv(ENV_ALERT_WEBHOOK, URL)
    assert alert_rule(load_config(path)).source == "file"


def test_a_failing_webhook_does_not_stop_the_file_channel(tmp_path) -> None:
    """两条出口的独立性：文件出口先落、webhook 坏了也拦不住它。

    这是"通知一定要留下痕迹"的兜底顺序，也是 `alerts ls` 里那条 failed 行的来处。
    """
    from onyx.obs.alerts.channels import FileChannel
    from onyx.obs.alerts.rules import AlertRule
    from onyx.obs.alerts.service import AlertService
    from onyx.store.db import Database
    from onyx.store.records import AnomalyRecord
    from onyx.store.repos import AlertRepo, TraceRepo

    db = Database(tmp_path / "a.sqlite")
    try:
        broken = _Recorder([httpx.Response(500), httpx.Response(500), httpx.Response(500)])
        service = AlertService(
            db, rule=AlertRule(cooldown_s=0),
            channels=[broken.channel(), FileChannel(tmp_path / "alerts.jsonl")],
        )
        TraceRepo(db).insert_anomaly(AnomalyRecord(
            id="a1", code="CONTEXT_OVERFLOW", severity="error", trace_id="tr-1",
            created_at="2026-10-05T02:59:00+00:00"))
        rows = service.tick(now="2026-10-05T03:00:00+00:00")
        assert {r.channel: r.status for r in rows} == {"webhook": "failed", "file": "sent"}
        written = list(tmp_path.glob("alerts-*.jsonl"))
        assert written, "webhook 失败不能拖累本地那一行"
        assert AlertRepo(db).counts_by_status() == {"failed:webhook": 1, "sent:file": 1}
    finally:
        db.close()
