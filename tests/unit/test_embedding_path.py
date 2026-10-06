"""S33 的 provider 侧实现：`/api/embed` 的解析与 mock 的确定性向量。

通路本身（trace/异常/unsupported）在 `test_embedding_gateway.py`；这里守的是
"从引擎响应到 `Embedding` 这一步会不会留下假事实"：

- 顺序与条数：一批一个请求、`vectors[i]` 对应 `inputs[i]`（2026-10-06 实测顺序保持，
  单条与混在 20 条一批里的向量 cos = 1.0）；
- **少返回一条就是错误**，不是"少一条算了"：少一条会让后面每条都错位，
  而错位的排名看着完全正常；
- 引擎没报的数保持 None（向量没有"输出 token"，那是 None 而不是 0）。
"""

from __future__ import annotations

import json

import httpx

from onyx.core.clock import FakeClock
from onyx.core.types import EmbedRequest, Status, TokenSource
from onyx.llm.providers.mock import MockProvider, mock_vector
from onyx.llm.providers.ollama import embed as embed_impl
from onyx.llm.providers.ollama.client import OllamaClient

DIM = 4


def _engine_response(texts, *, prompt_eval_count=10, with_counts=True):
    vectors = [[round(0.1 * (i + 1) + 0.01 * j, 4) for j in range(DIM)] for i in range(len(texts))]
    body = {"model": "embed/stub", "embeddings": vectors}
    if with_counts:
        body.update({"total_duration": 70_000_000, "load_duration": 5_000_000,
                     "prompt_eval_duration": 60_000_000,
                     "prompt_eval_count": prompt_eval_count})
    return body


def _client(payload, *, capture=None) -> OllamaClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture.append(json.loads(request.read()))
        assert request.url.path == "/api/embed"
        return httpx.Response(200, json=payload)

    return OllamaClient("http://stub.local", transport=httpx.MockTransport(handler))


def _embed(texts, *, payload=None, capture=None):
    """跑一次 provider 侧 embed，同时收事件（出处就藏在事件里）。"""
    body = payload if payload is not None else _engine_response(texts)
    events: list = []
    result = embed_impl.embed(
        _client(body, capture=capture),
        EmbedRequest(model="embed/stub", inputs=tuple(texts)),
        trace_id="t-1", clock=FakeClock(), on_event=events.append,
    )
    return result, events


# ── Ollama 原生通道 ────────────────────────────────────────────────
def test_vectors_keep_request_order_and_one_request_per_batch():
    capture: list = []
    result, _events = _embed(["甲", "乙", "丙"], capture=capture)
    assert result.status is Status.OK and len(result.vectors) == 3
    assert result.dimension == DIM
    assert capture == [{"model": "embed/stub", "input": ["甲", "乙", "丙"]}], "一批只发一个请求"


def test_engine_counts_become_an_engine_sourced_sample():
    result, events = _embed(["甲", "乙"])
    engine = result.usage_from(TokenSource.ENGINE)
    assert engine is not None and engine.in_tokens == 10
    #: 向量没有输出 token：None 而不是 0（填 0 就是凭空造一个可聚合的数字）
    assert engine.out_tokens is None
    assert engine.confidence.value == "high"
    types = [str(e.type) for e in events]
    assert "usage_engine" in types and "model_load" in types, "计数与载入耗时都要有出处事件"


def test_missing_counts_stay_missing_rather_than_zero():
    result, _events = _embed(["甲"], payload=_engine_response(["甲"], with_counts=False))
    assert result.usage == ()
    assert result.latency is not None and result.latency.total_ns is None


def test_keep_alive_is_forwarded_when_given():
    capture: list = []
    events: list = []
    embed_impl.embed(
        _client(_engine_response(["甲"]), capture=capture),
        EmbedRequest(model="embed/stub", inputs=("甲",), keep_alive="30m"),
        trace_id="t-2", clock=FakeClock(), on_event=events.append,
    )
    assert capture[0]["keep_alive"] == "30m"


def test_a_short_vector_list_is_an_error_not_a_silent_drop():
    """引擎少给一条 ⇒ 整批作废。

    这是最坏的一类静默错误：`vectors[1]` 其实是第 2 条 query 的向量却被当成第 2 个候选，
    排名照样是 1..n 里的一个数，看板全绿，而分数与文本已经毫无关系。
    """
    result, _events = _embed(
        ["甲", "乙", "丙"], payload=_engine_response(["甲", "乙"])
    )
    assert result.status is Status.ERROR
    assert result.vectors == () and result.dimension is None
    assert "2" in result.error and "3" in result.error and "对不齐" in result.error
    assert result.extra["requested"] == 3 and result.extra["returned"] == 2


def test_engine_usage_event_carries_the_real_counts():
    _result, events = _embed(["甲", "乙"])
    usage = next(e for e in events if str(e.type) == "usage_engine")
    assert usage.payload["in_tokens"] == 10 and usage.payload["out_tokens"] is None
    assert usage.payload["latency_ns"]["prompt_eval"] == 60_000_000
    assert set(usage.payload["raw_keys"]) == {
        "load_duration", "prompt_eval_count", "prompt_eval_duration", "total_duration"
    }


# ── mock 通道（离线跑任务与契约测试用的就是它）──────────────────────
def test_mock_vectors_are_deterministic_and_overridable():
    """同文本恒等、异文本不等；给了显式向量就用给的——排名判据要靠这个才可测。"""
    hash_of_yi = mock_vector("乙")
    provider = MockProvider(models=("mock/embed",), embeddings={"甲": (1.0,) + (0.0,) * 15})
    req = EmbedRequest(model="mock/embed", inputs=("甲", "甲", "乙"))
    vectors = provider.embed(req).vectors
    assert vectors[0] == vectors[1] == (1.0,) + (0.0,) * 15
    assert vectors[2] == hash_of_yi != vectors[0]
    assert len({len(v) for v in vectors}) == 1
    assert abs(sum(v * v for v in hash_of_yi) - 1.0) < 1e-9, "hash 向量必须是单位向量"
    assert provider.embed_calls == [req], "调用要留痕，否则测不到'一批一个请求'"


def test_mock_reports_no_token_numbers_at_all():
    """mock 没跑真模型 ⇒ 不报计数。报了就等于在测试里冒充"引擎给了出处"。"""
    provider = MockProvider(models=("mock/embed",))
    assert provider.embed(EmbedRequest(model="mock/embed", inputs=("甲",))).usage == ()


def test_ragged_dimensions_are_refused_not_passed_through():
    """一批里维度不一致 ⇒ 报错。放下去，下游的余弦要么抛错要么静默截断。"""
    provider = MockProvider(models=("mock/embed",), embeddings={"甲": (1.0, 0.0)})
    out = provider.embed(EmbedRequest(model="mock/embed", inputs=("甲", "乙")))
    assert out.status is Status.ERROR and out.vectors == ()
    assert "维度" in out.error
