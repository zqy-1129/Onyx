from __future__ import annotations

import pytest

from onyx.core.ids import new_trace_id
from onyx.store.db import Database
from onyx.store.records import (
    AnomalyRecord,
    ModelRecord,
    ProviderRecord,
    TokenPartRecord,
    ToolCallRecord,
    TraceRecord,
    UsageAltRecord,
    UsageRecord,
)
from onyx.store.repos import ModelRepo, TraceRepo, UsageRepo


@pytest.fixture
def db(tmp_path):
    with Database(tmp_path / "t.sqlite") as database:
        yield database


@pytest.fixture
def repos(db):
    return ModelRepo(db), TraceRepo(db), UsageRepo(db)


def _provider() -> ProviderRecord:
    return ProviderRecord(
        id="ollama-local", kind="ollama", base_url="http://127.0.0.1:11434",
        api_style="native", caps=("chat", "tools"), version="0.13.0",
    )


def _model(provider_id: str = "ollama-local", name: str = "qwen3:8b", **kw) -> ModelRecord:
    base = dict(
        id=f"{provider_id}/{name}", provider_id=provider_id, name=name,
        remote_model="qwen3:8b", digest="sha256:abc", bytes=5_000_000_000,
        family="qwen3", families=("qwen3",), parameter_size="8B", quantization="Q4_K_M",
        capabilities=("completion", "tools", "thinking"),
        model_info={"tokenizer.ggml.model": "qwen2", "tokenizer.chat_template": "{% raw %}x{% endraw %}"},
    )
    base.update(kw)
    return ModelRecord(**base)


def _trace(trace_id: str | None = None, **kw) -> TraceRecord:
    base = dict(
        kind="generation", purpose="chat", started_at="2026-10-02T10:00:00.000000+00:00",
    )
    base.update(kw)
    return TraceRecord(id=trace_id or new_trace_id(), **base)


# ── provider / model ───────────────────────────────────────────────
def test_provider_upsert_is_idempotent(db):
    models = ModelRepo(db)
    models.upsert_provider(_provider())
    models.upsert_provider(_provider())
    assert len(models.list_providers()) == 1
    got = models.list_providers()[0]
    assert got.caps == ("chat", "tools") and got.enabled is True and not got.config


def test_model_upsert_keeps_single_row_and_stable_id(db):
    models = ModelRepo(db)
    models.upsert_provider(_provider())
    first = models.upsert_model(_model())
    second = models.upsert_model(_model(bytes=6_000_000_000))
    assert first == second
    assert models.count() == 1
    assert models.get_model(first).bytes == 6_000_000_000, "重复同步必须更新而非新增"


def test_model_roundtrip_preserves_nested_metadata(db):
    models = ModelRepo(db)
    models.upsert_provider(_provider())
    mid = models.upsert_model(_model())
    got = models.get_model(mid)
    assert got.model_info["tokenizer.ggml.model"] == "qwen2"
    assert got.capabilities == ("completion", "tools", "thinking")
    assert got.families == ("qwen3",)
    assert models.find_by_name("ollama-local", "qwen3:8b").id == mid
    assert models.find_by_name("ollama-local", "nope") is None


def test_model_update_whitelist(db):
    """探针结论回灌只能写白名单列——防注入，也防误改身份字段。"""
    models = ModelRepo(db)
    models.upsert_provider(_provider())
    mid = models.upsert_model(_model())
    models.update_model(mid, tool_format="native_head", usage_ratio=0.62, usage_ratio_n=210)
    got = models.get_model(mid)
    assert got.tool_format == "native_head" and got.usage_ratio == 0.62 and got.usage_ratio_n == 210
    with pytest.raises(KeyError, match="不允许更新"):
        models.update_model(mid, name="hacked")
    with pytest.raises(KeyError):
        models.update_model(mid, provider_id="x; DROP TABLE model")


def test_probe_conclusion_roundtrip(db):
    models = ModelRepo(db)
    models.upsert_provider(_provider())
    mid = models.upsert_model(_model())
    models.update_model(mid, probe_json={"cache": "counts_full_prompt", "think": "included_in_eval_count"})
    assert models.get_model(mid).probe["cache"] == "counts_full_prompt"


# ── trace ──────────────────────────────────────────────────────────
def test_trace_insert_get_finish(db, repos):
    _, traces, _ = repos
    rec = _trace(model_name="qwen3:8b", params={"temperature": 0.2}, extra={"brand_new_field": 1})
    traces.upsert(rec)
    got = traces.get(rec.id)
    assert got.params == {"temperature": 0.2}
    assert got.extra == {"brand_new_field": 1}, "未知字段必须原样保留（前向兼容）"
    assert got.status == "ok" and got.finished_at is None

    traces.finish(
        rec.id, status="ok", finished_at="2026-10-02T10:00:03.000000+00:00",
        finish_reason="stop", engine_latency={"total_ns": 3_000_000_000, "load_ns": 1_000_000},
        output_ref="sha256:" + "0" * 64, gpu={"size_vram": 5_000_000_000},
    )
    done = traces.get(rec.id)
    assert done.finished_at.endswith("+00:00") and done.finish_reason == "stop"
    assert done.engine_latency["load_ns"] == 1_000_000
    assert done.gpu["size_vram"] == 5_000_000_000


def test_finish_does_not_erase_existing_values(db, repos):
    _, traces, _ = repos
    rec = _trace(error=None)
    traces.upsert(rec)
    traces.finish(rec.id, status="ok", finish_reason="stop")
    traces.finish(rec.id, status="error", error="boom")
    got = traces.get(rec.id)
    assert got.finish_reason == "stop", "第二次 finish 不带该字段时不许抹掉已有值"
    assert got.error == "boom"


def test_first_token_marked_once(db, repos):
    _, traces, _ = repos
    rec = _trace()
    traces.upsert(rec)
    traces.mark_first_token(rec.id, "2026-10-02T10:00:00.140000+00:00")
    traces.mark_first_token(rec.id, "2026-10-02T10:00:09.000000+00:00")
    assert traces.get(rec.id).first_token_at.endswith(".140000+00:00"), "TTFT 只认第一次"


def test_trace_list_cursor_pagination_and_filters(db, repos):
    _, traces, _ = repos
    ids = []
    for i in range(25):
        rec = _trace(
            model_id="m1" if i % 2 else "m2",
            purpose="chat" if i % 5 else "eval:tool_selection",
            status="ok" if i % 7 else "error",
        )
        ids.append(rec.id)
        traces.upsert(rec)

    page1 = traces.list(limit=10)
    assert len(page1) == 10
    assert [t.id for t in page1] == sorted([t.id for t in page1], reverse=True), "必须按时间倒序"
    page2 = traces.list(limit=10, cursor=page1[-1].id)
    assert len(page2) == 10
    assert not {t.id for t in page1} & {t.id for t in page2}, "游标分页不许重叠"
    assert len(traces.list(limit=10, cursor=page2[-1].id)) == 5
    assert traces.count() == 25
    assert all(t.model_id == "m1" for t in traces.list(model_id="m1", limit=50))
    assert all(t.purpose == "eval:tool_selection" for t in traces.list(purpose="eval:tool_selection", limit=50))
    assert all(t.status == "error" for t in traces.list(status="error", limit=50))


def test_eval_context_is_queryable(db, repos):
    """评测样本必须能反查：这是"分数可下钻到 trace"的前提。"""
    _, traces, _ = repos
    rec = _trace(purpose="eval:tool_selection", eval_run_id="run-1", case_id="c-7", sample_seq=2)
    traces.upsert(rec)
    got = traces.list(purpose="eval:tool_selection", limit=10)
    assert got[0].case_id == "c-7" and got[0].sample_seq == 2 and got[0].eval_run_id == "run-1"
    assert got[0].root_id == rec.id


# ── tool calls / anomalies ─────────────────────────────────────────
def test_tool_calls_ordered_by_step(db, repos):
    _, traces, _ = repos
    rec = _trace()
    traces.upsert(rec)
    traces.insert_tool_call(ToolCallRecord(
        id="tc2", trace_id=rec.id, step=2, parse_status="ok", name="calc", args={"x": 2},
        result_status="ok", latency_ms=12.5, executed_by="client",
    ))
    traces.insert_tool_call(ToolCallRecord(
        id="tc1", trace_id=rec.id, step=1, parse_status="truncated", name="weather",
        args=None, args_raw='{"city": "北', result_status=None, executed_by="client",
    ))
    calls = traces.list_tool_calls(rec.id)
    assert [c.id for c in calls] == ["tc1", "tc2"]
    assert calls[0].parse_status == "truncated"
    assert calls[0].args_raw == '{"city": "北', "畸形参数必须保住原文，这是诊断的关键证据"
    assert calls[0].args is None
    assert calls[1].args == {"x": 2}


def test_tool_call_upsert_updates_result(db, repos):
    _, traces, _ = repos
    rec = _trace()
    traces.upsert(rec)
    traces.insert_tool_call(ToolCallRecord(id="tc", trace_id=rec.id, step=1, parse_status="ok", name="a"))
    traces.insert_tool_call(ToolCallRecord(
        id="tc", trace_id=rec.id, step=1, parse_status="ok", name="a",
        result_status="ok", latency_ms=3.0, result_bytes=42,
    ))
    calls = traces.list_tool_calls(rec.id)
    assert len(calls) == 1 and calls[0].result_status == "ok" and calls[0].result_bytes == 42


def test_anomalies(db, repos):
    _, traces, _ = repos
    rec = _trace()
    traces.upsert(rec)
    traces.insert_anomaly(AnomalyRecord(id="a1", code="TOKEN_DRIFT", severity="warn", trace_id=rec.id,
                                        detail={"drift": 0.31}))
    traces.insert_anomaly(AnomalyRecord(id="a2", code="TOKEN_DRIFT", severity="error", trace_id=rec.id))
    traces.insert_anomaly(AnomalyRecord(id="a3", code="TOOL_LOOP", severity="error"))
    assert traces.anomaly_counts() == {"TOKEN_DRIFT": 2, "TOOL_LOOP": 1}
    assert len(traces.list_anomalies(code="TOKEN_DRIFT")) == 2
    assert traces.list_anomalies(limit=10)[0].severity == "error"


def test_anomaly_stats_carries_the_severity_of_each_code(db, repos):
    """看板 chip 的级别来自这里，不是前端自己猜（error 级显示成"提醒"是另一种谎言）。"""
    _, traces, _ = repos
    traces.insert_anomaly(AnomalyRecord(id="a1", code="TOKEN_DRIFT", severity="warn",
                                        created_at="2026-10-05T01:00:00+00:00"))
    traces.insert_anomaly(AnomalyRecord(id="a2", code="TOKEN_DRIFT", severity="error",
                                        created_at="2026-10-05T02:00:00+00:00"))
    traces.insert_anomaly(AnomalyRecord(id="a3", code="TOOL_LOOP", severity="error",
                                        created_at="2026-10-05T03:00:00+00:00"))

    stats = traces.anomaly_stats()
    assert stats["TOKEN_DRIFT"]["n"] == 2
    assert set(stats["TOKEN_DRIFT"]["severities"]) == {"warn", "error"}, "同一码出现过两种级别都要带上"
    assert stats["TOOL_LOOP"] == {"n": 1, "severities": ["error"]}
    assert traces.anomaly_stats(since="2026-10-05T02:30:00+00:00") == {
        "TOOL_LOOP": {"n": 1, "severities": ["error"]}
    }


def test_error_anomaly_summary_points_at_one_real_trace(db, repos):
    _, traces, _ = repos
    traces.insert_anomaly(AnomalyRecord(id="a1", code="CONTEXT_OVERFLOW", severity="error",
                                        trace_id="tr-1", created_at="2026-10-05T01:00:00+00:00"))
    traces.insert_anomaly(AnomalyRecord(id="a2", code="CONTEXT_OVERFLOW", severity="error",
                                        trace_id="tr-2", created_at="2026-10-05T02:00:00+00:00"))
    traces.insert_anomaly(AnomalyRecord(id="a3", code="TOKEN_DRIFT", severity="warn",
                                        trace_id="tr-3", created_at="2026-10-05T03:00:00+00:00"))

    got = traces.error_anomaly_summary(since="2026-10-05T00:00:00+00:00")
    assert got["n"] == 2 and got["by_code"] == {"CONTEXT_OVERFLOW": 2}
    # 最近一条要能点下去：只有计数的那行话无法回答"哪次请求"
    assert got["latest_trace_id"] == "tr-2" and got["latest_code"] == "CONTEXT_OVERFLOW"

    empty = traces.error_anomaly_summary(since="2026-10-06T00:00:00+00:00")
    assert empty["n"] == 0 and empty["latest_trace_id"] == "" and empty["by_code"] == {}


# ── usage ──────────────────────────────────────────────────────────
def test_usage_multi_source_and_attribution(db, repos):
    _, traces, usage = repos
    rec = _trace()
    traces.upsert(rec)
    usage.upsert(UsageRecord(
        trace_id=rec.id, source="engine", confidence="high", in_tokens=1842, out_tokens=213,
        ttft_ms=142.0, prefill_tps=12970.0, decode_tps=31.2, wall_ms=6800.0, drift_pct=0.007,
    ))
    usage.upsert_alts([
        UsageAltRecord(trace_id=rec.id, source="engine", in_tokens=1842, out_tokens=213),
        UsageAltRecord(trace_id=rec.id, source="compat", in_tokens=1842, out_tokens=213),
        UsageAltRecord(trace_id=rec.id, source="hf_tokenizer", in_tokens=1855, out_tokens=209),
        UsageAltRecord(trace_id=rec.id, source="heuristic", in_tokens=1990, out_tokens=205,
                       ok=True, confidence="low", note="chars/4 + CJK"),
        UsageAltRecord(trace_id=rec.id, source="fitted", in_tokens=None, out_tokens=None, ok=False,
                       note="模型未标定"),
    ])
    usage.replace_parts(rec.id, [
        TokenPartRecord(trace_id=rec.id, part="system", ord=0, tokens=120),
        TokenPartRecord(trace_id=rec.id, part="tool_defs", ord=1, tokens=1380),
        TokenPartRecord(trace_id=rec.id, part="msg:0", ord=2, tokens=340),
        TokenPartRecord(trace_id=rec.id, part="gen_prompt", ord=3, tokens=15),
    ])

    bundle = usage.fetch(rec.id)
    assert bundle.usage.source == "engine" and bundle.usage.in_tokens == 1842
    assert len(bundle.alts) == 5
    assert bundle.alt("fitted").ok is False and bundle.alt("fitted").note == "模型未标定"
    assert bundle.alt("nope") is None
    assert bundle.part_tokens("tool_defs") == 1380, "工具定义的上下文开销必须可单独查出"
    assert bundle.has_attribution


def test_replace_parts_does_not_accumulate(db, repos):
    _, traces, usage = repos
    rec = _trace()
    traces.upsert(rec)
    for _ in range(3):
        usage.replace_parts(rec.id, [TokenPartRecord(trace_id=rec.id, part="tool_defs", ord=0, tokens=100)])
    assert usage.fetch(rec.id).part_tokens("tool_defs") == 100, "重算归因必须整体替换，不许累加"


def test_usage_upsert_overwrites(db, repos):
    _, traces, usage = repos
    rec = _trace()
    traces.upsert(rec)
    usage.upsert(UsageRecord(trace_id=rec.id, source="heuristic", confidence="low", in_tokens=100))
    usage.upsert(UsageRecord(trace_id=rec.id, source="engine", confidence="high", in_tokens=120))
    got = usage.fetch(rec.id).usage
    assert got.source == "engine" and got.in_tokens == 120


def test_usage_records_prefill_mode(db, repos):
    """PROBES P11：冷/热必须落库，否则吞吐聚合无法分列，会得到一个看起来合理的错数字。"""
    _, traces, usage = repos
    rec = _trace()
    traces.upsert(rec)
    usage.upsert(UsageRecord(
        trace_id=rec.id, source="engine", confidence="high", in_tokens=644,
        prefill_tps=7722.0, prefill_mode="warm", prefill_ms_per_token=0.129,
    ))
    got = usage.fetch(rec.id).usage
    assert got.prefill_mode == "warm"
    assert got.prefill_ms_per_token == 0.129

    cold = _trace()
    traces.upsert(cold)
    usage.upsert(UsageRecord(
        trace_id=cold.id, source="engine", confidence="high", in_tokens=644,
        prefill_tps=1675.0, prefill_mode="cold", prefill_ms_per_token=0.597,
    ))
    rows = db.query("SELECT prefill_mode, COUNT(*) AS n FROM usage GROUP BY prefill_mode")
    assert {r["prefill_mode"]: r["n"] for r in rows} == {"warm": 1, "cold": 1}


def test_summarize_aggregates(db, repos):
    _, traces, usage = repos
    for i in range(4):
        rec = _trace(model_id="m1", started_at=f"2026-10-0{i + 1}T10:00:00.000000+00:00")
        traces.upsert(rec)
        usage.upsert(UsageRecord(
            trace_id=rec.id, source="engine" if i % 2 else "heuristic",
            confidence="high" if i % 2 else "low",
            in_tokens=100, out_tokens=50, thinking_tokens=10, drift_pct=0.01 * i,
        ))
    summary = usage.summarize()
    assert summary.traces == 4 and summary.in_tokens == 400 and summary.out_tokens == 200
    assert summary.thinking_tokens == 40
    assert summary.by_source == {"engine": 2, "heuristic": 2}
    assert summary.by_confidence == {"high": 2, "low": 2}
    assert len(summary.drift_samples) == 4
    filtered = usage.summarize(since="2026-10-03T00:00:00.000000+00:00")
    assert filtered.traces == 2


def test_timeseries_leaves_unmeasured_rates_as_null_not_zero(db, repos):
    """S39：速率列的"这一格没测到"必须是 NULL。

    原先 SQL 写 `COALESCE(AVG(decode_tps),0)`，于是空桶与"真的 0 t/s"在数据里同形，
    折线会为没数据的桶掉出一个 0 的坑——而 0 t/s 读起来像一个测量结果。
    token 数的 0 是**真 0**（空输出），所以那两列仍然 COALESCE，这条断言把它们隔开。
    """
    _, traces, usage = repos
    rec = _trace(model_id="m1", started_at="2026-10-05T10:00:00.000000+00:00")
    traces.upsert(rec)
    usage.upsert(UsageRecord(trace_id=rec.id, source="engine", confidence="high",
                             in_tokens=0, out_tokens=0))     # 没有任何延迟数据的一次真实请求
    bucket = usage.timeseries(bucket_minutes=60)[0]
    assert bucket["decode_tps"] is None and bucket["cold_prefill_tps"] is None
    assert bucket["warm_prefill_tps"] is None
    assert bucket["in_tokens"] == 0 and bucket["out_tokens"] == 0, "token 的 0 是真的 0"
    assert bucket["traces"] == 1

    second = _trace(model_id="m1", started_at="2026-10-05T11:30:00.000000+00:00")
    traces.upsert(second)
    usage.upsert(UsageRecord(trace_id=second.id, source="engine", confidence="high",
                             in_tokens=10, out_tokens=2, decode_tps=30.0, prefill_mode="cold",
                             prefill_tps=900.0))
    buckets = usage.timeseries(bucket_minutes=60)
    assert len(buckets) == 2
    by_hour = {b["bucket"][-5:]: b for b in buckets}
    assert by_hour["10:00"]["decode_tps"] is None, "补了数据也不能把空桶顶成 0"
    assert by_hour["11:00"]["cold_prefill_tps"] == 900.0
    assert by_hour["11:00"]["warm_prefill_tps"] is None, "这一小时没有 warm 样本，不是 0"
