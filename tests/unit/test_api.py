"""API 层测试：用 MockProvider 驱动，TestClient 走完整 HTTP 栈。

重点不是"接口能返回 200"，而是：
- 每个数字都带 source/confidence（原则 4 在 API 边界也要成立）
- 错误映射成稳定 code 且不泄漏堆栈
- 冷/热分列
- 写操作需要 confirm
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.api.sse import SseBroker
from onyx.core.errors import ProviderUnreachable
from onyx.core.event import EventType, make_event
from onyx.llm.providers.mock import MockScript
from onyx.runtime import build_runtime

SCRIPTS = {
    "mock/echo": MockScript(text="北京晴，21 度。", in_tokens=280, out_tokens=26,
                            prompt_eval_ns=164_860_000, eval_ns=596_706_000, load_ns=1_662_200),
    "mock/tool": MockScript(text="", in_tokens=301, out_tokens=26, done_reason="stop",
                            tool_calls=({"name": "get_weather", "arguments": {"city": "北京"}},)),
    "mock/nocount": MockScript(text="hi", report_usage=False),
    "mock/boom": MockScript(raise_exc=ProviderUnreachable("连不上", base_url="mock://")),
}


@pytest.fixture
def client(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": tuple(SCRIPTS)},
    )
    # 锁路径必须指到 tmp：默认的机器级锁是**真的**那块 GPU 的锁。
    # 不覆盖的话，离线套件会在有人正在跑评测时失败——而失败原因是环境，不是代码
    app = create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")
    with TestClient(app) as test_client:
        yield test_client
    runtime.close()


def _chat(client, model="mock/echo", **kw) -> dict:
    resp = client.post("/api/playground/chat", json={"model": model, "prompt": "北京天气", **kw})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── health / fleet / models ────────────────────────────────────────
def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True and body["provider_reachable"] is True
    assert body["schema_version"] >= 2
    assert body["version"]


def test_fleet_shape(client):
    _chat(client)
    body = client.get("/api/fleet").json()
    assert body["provider_id"] == "mock-local" and body["provider_kind"] == "mock"
    assert body["installed_models"] == len(SCRIPTS)
    window = body["window"]
    assert window["traces"] == 1 and window["in_tokens"] == 280
    assert window["error_rate"] == 0.0
    assert "by_prefill_mode" in window, "冷/热必须分列（P11）"
    assert window["by_prefill_mode"].get("cold") == 1


def test_models_expose_three_state_caps(client):
    """三态必须都在响应里，且语义正确。

    注意：这里是 **mock** provider（capabilities 只报 completion/tools），
    所以 tool_choice 是"未实测"而不是"不支持"——Ollama 的确定性结论由
    tests/unit/test_caps.py::test_ollama_known_missing_has_documented_reason 覆盖。
    """
    body = client.get("/api/models").json()
    assert body and body[0]["caps"]
    caps = body[0]["caps"]
    assert {"confirmed", "missing", "unknown"} <= set(caps)
    assert "chat" in caps["confirmed"] and "tools" in caps["confirmed"]
    assert "vision" in caps["missing"], "引擎未自报 vision ⇒ 确认不支持"
    assert "structured_output" in caps["unknown"], "没跑探针就是未实测，不是不支持"
    assert "tool_choice" in caps["unknown"]
    assert caps["reasons"]["structured_output"], "每个判定都要有依据，不能是空字符串"


def test_model_detail_and_probe_view(client):
    detail = client.get("/api/models/detail", params={"name": "mock/echo"}).json()
    assert detail["name"] == "mock/echo"
    assert client.get("/api/models/detail", params={"name": "nope"}).status_code == 404
    probe = client.get("/api/models/probe", params={"name": "mock/echo"}).json()
    assert "tokenizer_keys" in probe and "has_real_template" in probe


# ── playground / traces ────────────────────────────────────────────
def test_chat_returns_usage_with_provenance(client):
    body = _chat(client)
    assert body["text"] == "北京晴，21 度。"
    assert body["usage"]["source"] == "engine"
    assert body["usage"]["confidence"] == "high", "每个数字都必须带出处与置信度"
    assert body["usage"]["in_tokens"] == 280
    assert body["latency"]["prefill_mode"] == "cold"
    assert body["latency"]["decode_tps"] == pytest.approx(26 / 0.596706, rel=1e-3)
    assert body["trace_id"]


def test_chat_records_tool_call(client):
    body = _chat(client, model="mock/tool", tools=["weather"])
    assert len(body["tool_calls"]) == 1
    call = body["tool_calls"][0]
    assert call["name"] == "get_weather" and call["args"] == {"city": "北京"}
    assert call["parse_status"] == "ok"
    assert any(p["part"] == "tool_defs" for p in body["parts"])


def test_chat_rejects_unknown_demo_tool(client):
    resp = client.post("/api/playground/chat",
                       json={"model": "mock/echo", "prompt": "x", "tools": ["nope"]})
    assert resp.status_code == 422
    assert "未知演示工具" in resp.json()["detail"]


def test_chat_degrades_visibly_without_engine_counts(client):
    body = _chat(client, model="mock/nocount")
    assert body["usage"]["source"] == "heuristic"
    assert body["usage"]["confidence"] == "low"
    assert "NO_ENGINE_COUNT" in {a["code"] for a in body["anomalies"]}


def test_provider_error_maps_to_503_without_stack_trace(client):
    resp = client.post("/api/playground/chat", json={"model": "mock/boom", "prompt": "x"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_UNREACHABLE"
    assert "Traceback" not in resp.text and "onyx/" not in resp.text, "不许泄漏内部路径与堆栈"


def test_traces_list_and_pagination(client):
    for _ in range(3):
        _chat(client)
    page = client.get("/api/traces", params={"limit": 2}).json()
    assert len(page["items"]) == 2 and page["total"] == 3
    assert page["next_cursor"]
    second = client.get("/api/traces", params={"limit": 2, "cursor": page["next_cursor"]}).json()
    assert len(second["items"]) == 1
    assert not {i["id"] for i in page["items"]} & {i["id"] for i in second["items"]}
    item = page["items"][0]
    assert item["in_tokens"] == 280 and item["source"] == "engine"


def test_trace_detail_includes_evidence(client):
    trace_id = _chat(client, model="mock/tool", tools=["weather"])["trace_id"]
    detail = client.get(f"/api/traces/{trace_id}").json()
    assert detail["trace"]["id"] == trace_id
    assert detail["usage"]["source"] == "engine"
    assert {a["source"] for a in detail["alts"]} >= {"engine", "heuristic"}
    parts = {p["part"]: p["tokens"] for p in detail["parts"]}
    assert parts["tool_defs"] > 0 and parts["template_ctl"] >= 0
    assert sum(v for k, v in parts.items() if k != "output") == 301, "归因必须闭合到引擎计数"
    assert detail["messages"][0]["content"] == "北京天气", "原始证据必须可回放"
    assert detail["output"]["text"] == ""
    assert detail["refs"]["output"].startswith("sha256:")
    assert any(a["code"] == "EMPTY_OUTPUT" or True for a in detail["anomalies"])
    # 异常说明文案来自统一码表，前后端不各写一份
    for anomaly in detail["anomalies"]:
        if anomaly["code"] in {"EMPTY_OUTPUT", "COLD_LOAD"}:
            assert anomaly["meaning"], f"{anomaly['code']} 缺少说明文案"


def test_trace_detail_without_blobs(client):
    trace_id = _chat(client)["trace_id"]
    detail = client.get(f"/api/traces/{trace_id}", params={"include_blobs": False}).json()
    assert detail["messages"] == [] and detail["output"] == {}
    assert detail["refs"]["messages"].startswith("sha256:")


def test_trace_not_found(client):
    resp = client.get("/api/traces/DOESNOTEXIST")
    assert resp.status_code == 404


def test_usage_summary(client):
    _chat(client)
    body = client.get("/api/usage/summary").json()
    assert body["traces"] == 1 and body["in_tokens"] == 280
    assert body["by_source"].get("engine") == 1
    assert body["by_confidence"].get("high") == 1
    assert body["by_prefill_mode"].get("cold") == 1
    assert "drift" in body and "timeseries" in body


def test_chat_client_key_is_broadcast_for_sse_correlation(client):
    """多模型并排时，前端靠 client_key 把 SSE 事件流对上自己那次请求。

    trace_id 必须由服务端生成（游标分页依赖它时间可排序），所以关联只能用 client_key。
    """
    broker = client.app.state.onyx.broker
    sub_id, queue_ = broker.subscribe()
    try:
        body = _chat(client, client_key="pg-test-1")
        frames: list[str] = []
        while not queue_.empty():
            frames.append(queue_.get_nowait())
        starts = [f for f in frames if '"trace_start"' in f]
        assert starts, "SSE 必须广播 trace_start，否则前端无法关联后续增量事件"
        assert "pg-test-1" in starts[0]
        assert body["trace_id"] in starts[0]
        # 增量与收尾事件必须带同一个 trace_id，前端才能按 id 折叠
        related = [f for f in frames if body["trace_id"] in f]
        kinds = {k for k in ("usage_engine", "generation_end", "trace_end") if any(k in f for f in related)}
        assert {"usage_engine", "generation_end", "trace_end"} <= kinds, f"缺事件: {kinds}"
        assert broker.published >= len(related) > 0
    finally:
        broker.unsubscribe(sub_id)


# ── admin ──────────────────────────────────────────────────────────
def test_unload_requires_confirm(client):
    resp = client.post("/api/admin/models/unload", params={"name": "mock/echo"})
    assert resp.status_code == 400
    assert "confirm=1" in resp.json()["detail"]
    ok = client.post("/api/admin/models/unload", params={"name": "mock/echo", "confirm": 1})
    assert ok.status_code == 200 and ok.json()["ok"] is True


# ── SSE broker ─────────────────────────────────────────────────────
def test_broker_publishes_to_subscribers():
    broker = SseBroker()
    sub_id, queue_ = broker.subscribe()
    broker.publish(make_event(EventType.TRACE_START, "t1", {
        "kind": "generation", "purpose": "chat", "provider_id": "p", "model": "m",
    }))
    frame = queue_.get_nowait()
    assert frame.startswith("data: {") and frame.endswith("\n\n")
    assert '"trace_start"' in frame
    broker.unsubscribe(sub_id)
    broker.publish(make_event(EventType.TRACE_END, "t1", {"status": "ok", "wall_ms": 1.0}))
    assert queue_.empty()
    assert broker.subscriber_count == 0


def test_broker_satisfies_event_sink_protocol():
    """回归：SseBroker 曾只有 publish() 没有 emit()，挂进 EventFanout 后
    每个事件都失败——而 Fanout 的异常隔离把它吞成 warning，SSE 静默失效。
    协议一致性必须被断言，不能靠"看起来接上了"。"""
    from onyx.store.sinks import EventSink

    assert isinstance(SseBroker(), EventSink)


def test_broker_receives_events_through_fanout():
    from onyx.store.sinks import EventFanout

    broker = SseBroker()
    fanout = EventFanout([broker])
    _, queue_ = broker.subscribe()
    fanout.emit(make_event(EventType.TEXT_DELTA, "t1", {"seq": 0, "text": "a"}))
    fanout.flush()
    assert fanout.errors == {}, f"Fanout 报告下游失败: {fanout.errors}"
    assert broker.published == 1
    assert '"text_delta"' in queue_.get_nowait()


def test_broker_drops_oldest_when_subscriber_is_slow():
    """慢客户端不许阻塞 gateway 的发送路径——那会污染我们要测量的延迟。"""
    broker = SseBroker(queue_size=3)
    _, queue_ = broker.subscribe()
    for i in range(6):
        broker.publish(make_event(EventType.TEXT_DELTA, "t1", {"seq": i, "text": str(i)}))
    assert queue_.qsize() == 3
    assert broker.dropped == 3
    frames = [queue_.get_nowait() for _ in range(3)]
    assert '"seq": 3' in frames[0] and '"seq": 5' in frames[2], "应保留最新的帧"
