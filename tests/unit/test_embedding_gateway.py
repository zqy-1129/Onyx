"""S33：`gateway.embed()` 这条新入口的可观测性契约。

这一层守的是**通路**，不是某个引擎的返回形状（那在 `test_embedding_path.py`）：

1. kind 必须是 `embed`：`TraceContext.kind` 的默认值就是 `generation`，
   所以"读 ctx.kind"会让每条向量化调用都被记成生成调用——按 kind 分的统计全部说谎；
2. 生成专属的异常规则不许作用在向量调用上（否则每条 embed 都挂 `EMPTY_OUTPUT`，
   而"引擎没输出文本"与"模型没答出来"在看板上就成了同一件事）；
3. 没有出处就是 None，不是 0，也不是自称 engine；
4. 引擎没有 `EmbeddingProvider` 时给**明确的 unsupported**：不 TypeError、不假向量、不隐式降级。

这里的 provider 是文件内自己定义的小桩，刻意不用 `MockProvider.embed`：
`EmbeddingProvider` 是**结构化协议**，"照协议写一个实现就能被认出来"本身就是要点
（S16a 的教训：内置实现与内核一起过拟合，只有外部实现能把抽象泄漏顶出来）。
"""

from __future__ import annotations

import json

import pytest

from onyx.core.clock import FakeClock
from onyx.core.content import FileBlobStore
from onyx.core.types import (
    Cap,
    Confidence,
    Embedding,
    EmbedRequest,
    Generation,
    GenerationRequest,
    Status,
    TokenSample,
    TokenSource,
    TraceContext,
    TraceKind,
    TracePurpose,
)
from onyx.llm.gateway import Gateway
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import TraceRepo
from onyx.store.sinks import SqliteRecordSink


class _StubEmbedder:
    """只实现协议要求的成员：`EmbeddingProvider` 靠结构化识别，不靠注册表。"""

    id = "stub-embed"

    def __init__(self, *, vectors: tuple[tuple[float, ...], ...] | None = None,
                 usage: tuple[TokenSample, ...] = ()) -> None:
        self.vectors = vectors
        self.usage = usage
        self.seen: list[EmbedRequest] = []

    def capabilities(self) -> frozenset[Cap]:
        return frozenset({Cap.EMBED})

    def list_models(self) -> list:
        return []

    def show_model(self, name: str):
        raise KeyError(name)

    def running(self) -> list:
        return []

    def generate(self, req: GenerationRequest, **kw: object) -> Generation:
        raise AssertionError("向量入口不许偷偷走 generate")

    def embed(self, req: EmbedRequest, *, trace_id: str = "", on_event=None) -> Embedding:
        from onyx.core.event import EventType, make_event
        from onyx.llm.providers.base import emit

        self.seen.append(req)
        vectors = self.vectors if self.vectors is not None else tuple(
            (0.5, -0.5) for _ in req.inputs
        )
        #: 约定与 generate 侧一致：**计数由 provider 发成事件**（`emit_final_events` 就是干这个的）。
        #: gateway 只负责把事件送进观测，所以"返回了 usage 却没发事件"的实现在这里就会丢数——
        #: 这条测试同时是写给下一个 provider 作者的说明书。
        for sample in self.usage:
            emit(on_event, make_event(EventType.USAGE_ENGINE, trace_id, {
                "in_tokens": sample.in_tokens, "out_tokens": sample.out_tokens,
                "thinking_tokens": None, "cached_tokens": None, "ok": sample.ok,
                "note": sample.note,
                "latency_ns": {"total": None, "load": None, "prompt_eval": None, "eval": None},
                "raw_keys": [],
            }))
        return Embedding(vectors=vectors, model=req.model, usage=self.usage)


@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def _gateway(env, provider) -> Gateway:
    _, _, observer, blobs = env
    return Gateway(provider, observer=observer, blobs=blobs, clock=FakeClock())


def _req(texts=("甲", "乙"), *, model="stub/embed", context=None) -> EmbedRequest:
    return EmbedRequest(
        model=model, inputs=tuple(texts),
        context=context or TraceContext(purpose=TracePurpose.EVAL),
    )


def _codes(db, trace_id: str) -> set[str]:
    return {str(r["code"]) for r in db.query("SELECT code FROM anomaly WHERE trace_id=?",
                                            (trace_id,))}


def _usage(db, trace_id: str) -> list[dict]:
    return [dict(r) for r in db.query(
        "SELECT in_tokens, out_tokens, source FROM usage WHERE trace_id=?", (trace_id,)
    )]


# ── 通路 ────────────────────────────────────────────────────────────
def test_trace_is_recorded_as_an_embed_call(env):
    db, sink, _, blobs = env
    out = _gateway(env, _StubEmbedder()).embed(_req())
    sink.flush(5.0)

    assert out.ok and len(out.embedding.vectors) == 2
    trace = TraceRepo(db).get(out.trace_id)
    assert trace is not None and str(trace.kind) == "embed"
    assert trace.purpose == "eval" and trace.model_name == "stub/embed"
    assert trace.params == {"batch_size": 2}
    #: 向量不落库（没有读者，只会撑大 `.data`），但输入要能下钻回原文
    assert json.loads(blobs.get(trace.messages_ref))["inputs"] == ["甲", "乙"]
    assert out.record_refs["messages"] == trace.messages_ref


def test_context_kind_does_not_override_the_entry_kind(env):
    """`TraceContext(kind=GENERATION)` 是**默认值**，不是"请记成 generation"的请求。"""
    db, sink, _, _ = env
    out = _gateway(env, _StubEmbedder()).embed(_req(
        context=TraceContext(kind=TraceKind.GENERATION, purpose=TracePurpose.PROBE)
    ))
    sink.flush(5.0)
    trace = TraceRepo(db).get(out.trace_id)
    assert str(trace.kind) == "embed" and trace.purpose == "probe"


def test_generation_output_rules_do_not_fire_on_embed(env):
    db, sink, _, _ = env
    out = _gateway(env, _StubEmbedder()).embed(_req())
    sink.flush(5.0)
    codes = _codes(db, out.trace_id)
    assert "EMPTY_OUTPUT" not in codes and "EMPTY_CONTENT_WITH_THINKING" not in codes


def test_missing_engine_counts_are_none_plus_an_anomaly(env):
    """没计数 ⇒ 落一行 None 并说出"缺计数"，而不是自称 engine，也不是补一个 0。"""
    db, sink, _, _ = env
    out = _gateway(env, _StubEmbedder()).embed(_req())
    sink.flush(5.0)
    rows = _usage(db, out.trace_id)
    assert len(rows) == 1
    assert rows[0]["in_tokens"] is None and rows[0]["out_tokens"] is None
    assert rows[0]["source"] != str(TokenSource.ENGINE)
    assert "NO_ENGINE_COUNT" in _codes(db, out.trace_id)


def test_engine_counts_reported_by_the_provider_are_kept(env):
    db, sink, _, _ = env
    provider = _StubEmbedder(usage=(TokenSample(
        source=TokenSource.ENGINE, in_tokens=17, out_tokens=None, confidence=Confidence.HIGH,
    ),))
    out = _gateway(env, provider).embed(_req(("一句话",)))
    sink.flush(5.0)
    rows = _usage(db, out.trace_id)
    assert rows[0]["in_tokens"] == 17 and rows[0]["out_tokens"] is None
    assert rows[0]["source"] == str(TokenSource.ENGINE)


def test_wall_ms_and_status_come_back_on_the_result(env):
    db, sink, _, _ = env
    out = _gateway(env, _StubEmbedder()).embed(_req())
    sink.flush(5.0)
    assert out.embedding.wall_ms is not None and out.embedding.status is Status.OK
    assert str(TraceRepo(db).get(out.trace_id).status) == "ok"


# ── unsupported ────────────────────────────────────────────────────
def test_provider_without_embed_is_refused_not_simulated(env):
    """只实现 `LlmProvider` 的引擎（外部插件就是这一类）要拿到明确的 unsupported。

    两条隐式降级都会留下假事实：`TypeError`（把契约问题变成崩溃），
    或"用 chat 生成一串数字再解析"（那测的就不再是表示质量，而是模型会不会输出 JSON）。
    """
    from example_provider.provider import EchoProvider

    db, sink, _, _ = env
    provider = EchoProvider(base_url="http://stub.local")
    assert not hasattr(provider, "embed"), "前提：这个实现确实没有向量化能力"

    out = _gateway(env, provider).embed(_req())
    sink.flush(5.0)
    assert out.ok is False and out.embedding.vectors == ()
    assert out.embedding.status is Status.ERROR
    assert "EMBED_UNSUPPORTED" in _codes(db, out.trace_id)
    assert "EMBED_UNSUPPORTED" in {code for code, _, _ in out.anomalies}
    detail = next(d for c, _, d in out.anomalies if c == "EMBED_UNSUPPORTED")
    assert detail["cap"] == str(Cap.EMBED) and detail["provider_id"] == provider.id
    assert str(TraceRepo(db).get(out.trace_id).status) == "error", "失败也要留下那条 trace"


def test_embed_request_is_forwarded_verbatim(env):
    """gateway 不拆批、不改模型名：拆多大批是调用方（成本与 GPU 锁）的决定。"""
    db, sink, _, _ = env
    provider = _StubEmbedder()
    req = _req(("a", "b", "c"), model="stub/embed", context=TraceContext(
        purpose=TracePurpose.EVAL, eval_run_id="run-1", case_id="c-1", sample_seq=3))
    _gateway(env, provider).embed(req)
    sink.flush(5.0)
    assert provider.seen == [req]
    trace = TraceRepo(db).get(_gateway(env, provider).embed(req).trace_id)
    assert trace.eval_run_id == "run-1" and trace.case_id == "c-1"
