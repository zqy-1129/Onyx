"""M1 出口判据：一次**真实**对话，DB 里能查到一条完整 trace。

判据（IMPLEMENTATION S5 / M1）：
- token 带来源与置信度
- TTFT / prefill 模式 / decode TPS
- 工具调用与原始参数
- 原始 body 可回放
- Σ分段 + template_ctl == 引擎计数
"""

from __future__ import annotations

import os

import pytest

from onyx.core.content import FileBlobStore
from onyx.core.types import GenerationRequest, GenParams, Message, Role, ToolSpec
from onyx.llm.gateway import Gateway
from onyx.llm.providers.ollama import OllamaProvider
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import TraceRepo, UsageRepo
from onyx.store.sinks import SqliteRecordSink

pytestmark = pytest.mark.live

MODEL = os.environ.get("ONYX_TEST_MODEL", "qwen3.5:9b")
BASE_URL = os.environ.get("ONYX_OLLAMA_URL", "http://127.0.0.1:11434")

WEATHER = ToolSpec(
    name="get_weather", description="查询指定城市当前天气",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)


@pytest.fixture
def runtime(tmp_path):
    provider = OllamaProvider(base_url=BASE_URL)
    if not provider.client.is_reachable():
        pytest.skip(f"Ollama 不可达: {BASE_URL}")
    if MODEL not in {m.name for m in provider.list_models()}:
        pytest.skip(f"模型 {MODEL} 未安装")
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, idle_wait=0.005)
    blobs = FileBlobStore(tmp_path / "blobs")
    observer = ObserverEngine(record_sink=sink)
    gateway = Gateway(provider, observer=observer, blobs=blobs, sample_gpu=True)
    yield gateway, db, sink, blobs
    provider.unload(MODEL)
    sink.close()
    provider.close()
    db.close()


def test_real_conversation_produces_a_complete_trace(runtime):
    gateway, db, sink, blobs = runtime
    result = gateway.generate(GenerationRequest.of(
        MODEL, "用一句话解释什么是 KV 缓存",
        params=GenParams(max_tokens=256, temperature=0.0), thinking=False,
    ))
    sink.flush(10.0)

    record = TraceRepo(db).get(result.trace_id)
    assert record is not None, "trace 必须落库"
    assert record.status == "ok" and record.model_name == MODEL
    assert record.provider_id == "ollama-local"
    assert record.first_token_at is None or record.first_token_at

    bundle = UsageRepo(db).fetch(result.trace_id)
    usage = bundle.usage
    assert usage is not None
    assert usage.source == "engine", f"应以引擎计数为采信来源，实际 {usage.source}"
    assert usage.confidence == "high"
    assert usage.in_tokens and usage.in_tokens > 0
    assert usage.out_tokens and usage.out_tokens > 0
    assert usage.decode_tps and usage.decode_tps > 0
    assert usage.prefill_mode in {"cold", "warm"}, "P11：必须显式判定冷/热"
    assert {a.source for a in bundle.alts} >= {"engine", "heuristic"}, "多来源必须都留下"

    # Σ分段 + template_ctl 必须等于引擎计数（P9 决定的归因口径）
    input_parts = sum(p.tokens for p in bundle.parts if p.part != "output")
    assert input_parts == usage.in_tokens, f"归因不闭合: {input_parts} != {usage.in_tokens}"

    # 原始证据可回放
    assert record.messages_ref and record.output_ref
    output = blobs.get_json(record.output_ref)
    assert output["text"] == result.generation.text
    assert blobs.get_json(record.messages_ref)[0]["content"] == "用一句话解释什么是 KV 缓存"

    # GPU 采样（sample_gpu=True）
    assert record.gpu, "应采到显存/上下文快照"

    print(f"\n[M1] trace={result.trace_id}")
    print(f"[M1] in={usage.in_tokens} out={usage.out_tokens} src={usage.source}/{usage.confidence}")
    print(f"[M1] ttft={usage.ttft_ms} prefill={usage.prefill_mode} "
          f"({usage.prefill_ms_per_token}ms/tok) decode_tps={usage.decode_tps:.1f}")
    print(f"[M1] parts={[(p.part, p.tokens) for p in bundle.parts]}")
    print(f"[M1] gpu={record.gpu}")
    print(f"[M1] anomalies={[(code, sev) for code, sev, _ in result.anomalies]}")


def test_real_tool_call_is_recorded_with_args(runtime):
    gateway, db, sink, _ = runtime
    result = gateway.generate(GenerationRequest(
        model=MODEL,
        messages=(Message(role=Role.USER, content="北京现在天气怎么样？"),),
        tools=(WEATHER,),
        params=GenParams(max_tokens=256, temperature=0.0),
        thinking=False,
    ))
    sink.flush(10.0)

    calls = TraceRepo(db).list_tool_calls(result.trace_id)
    assert calls, f"模型应发起工具调用；正文={result.generation.text[:120]!r}"
    call = calls[0]
    assert call.name == "get_weather"
    assert call.parse_status == "ok", f"参数解析失败: {call.args_raw!r}"
    assert call.args and call.args.get("city")
    assert call.result_status is None, "本步只观测，不执行工具（执行在 S12）"

    bundle = UsageRepo(db).fetch(result.trace_id)
    tool_defs = bundle.part_tokens("tool_defs")
    assert tool_defs > 50, f"工具 schema 开销应被单独归因，实际 {tool_defs}"
    print(f"\n[M1-tool] {call.name} args={call.args} parse={call.parse_status}")
    print(f"[M1-tool] in={bundle.usage.in_tokens} 其中 tool_defs={tool_defs} "
          f"template_ctl={bundle.part_tokens('template_ctl')}")
