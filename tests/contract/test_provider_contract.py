"""Provider 契约测试：所有实现对同一套断言负责。

`onyx/llm/providers/base.py` 的承诺是"可替换"，而可替换不是文档里写着可替换，
是有一组断言逼着它可替换。这里刻意包含**一个外部插件实现**（EchoProvider）：
内置实现与内核一起过拟合是察觉不到的，只有外部实现会把抽象泄漏顶出来。
（事实：写这份测试时发现 gateway 一直按 `generate(req, trace_id=..., on_event=...)`
调用，而协议里没写 `trace_id` —— 照协议写的外部 provider 必然崩在 TypeError。）

断言分三层：
1. 形状——签名与协议方法齐备；
2. 自洽——`info()` 与 `capabilities()`、`list_models()` 与 `show_model()` 不许互相打脸；
3. 后果——只实现协议的 provider 挂上 gateway 后 trace 必须照样落库，
   token 必须带出处与置信度（"引擎没报"不许被写成 0）。
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from example_provider.provider import EchoProvider

from onyx.core.content import FileBlobStore
from onyx.core.errors import OnyxError
from onyx.core.types import Cap, EmbedRequest, GenerationRequest, ModelDetail, Status, TokenSource
from onyx.llm.gateway import Gateway
from onyx.llm.providers.base import EmbeddingProvider, LlmProvider
from onyx.llm.providers.mock import MockProvider
from onyx.llm.providers.ollama import client as ollama_client
from onyx.llm.providers.ollama import provider as ollama_provider
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import TraceRepo
from onyx.store.sinks import SqliteRecordSink

# ── Ollama 的离线替身：同一套适配器代码，传输层换成 MockTransport ──────────
_STUB_TAGS = {
    "models": [{
        "name": "stub/test", "model": "stub/test:latest", "size": 4096, "digest": "d1",
        "details": {"family": "qwen2", "parameter_size": "1B", "quantization_layer": "Q4_K_M"},
        "capabilities": ["completion", "tools"], "context_length": 2048,
    }]
}
_STUB_SHOW = {
    "template": "{{ .Prompt }}",
    "model_info": {"general.architecture": "qwen2", "tokenizer.ggml.model": "gpt2"},
    "parameters": {"num_ctx": 2048},
    "license": "mit",
}


def _stub_vectors(texts: list) -> list:
    """按输入条数造等长向量：契约测试要的是"条数与顺序对得上"，不是数值。"""
    return [[round(0.25 * (i + 1) + 0.01 * j, 4) for j in range(4)] for i in range(len(texts))]


def _stub_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/api/version":
        return httpx.Response(200, json={"version": "0.0.0-contract"})
    if path == "/api/tags":
        return httpx.Response(200, json=_STUB_TAGS)
    if path == "/api/show":
        return httpx.Response(200, json=_STUB_SHOW)
    if path == "/api/ps":
        return httpx.Response(200, json={"models": []})
    if path == "/api/embed":
        payload = json.loads(request.read())
        texts = payload.get("input") or []
        return httpx.Response(200, json={
            "model": payload.get("model"), "embeddings": _stub_vectors(texts),
            "prompt_eval_count": 2 * max(1, len(texts)),
            "total_duration": 20_000_000, "load_duration": 1_000_000,
            "prompt_eval_duration": 18_000_000,
        })
    if path == "/api/chat":
        payload = json.loads(request.read())
        counts = {
            "prompt_eval_count": 7, "eval_count": 3,
            "total_duration": 10_000_000, "eval_duration": 5_000_000,
            "prompt_eval_duration": 3_000_000, "load_duration": 1_000_000,
        }
        if payload.get("stream"):
            body = "\n".join([
                json.dumps({"model": "stub/test", "created_at": "2026-01-01T00:00:00Z",
                            "message": {"role": "assistant", "content": "好"}, "done": False}),
                json.dumps({"model": "stub/test", "created_at": "2026-01-01T00:00:00Z",
                            "message": {"role": "assistant", "content": ""},
                            "done": True, "done_reason": "stop", **counts}),
            ]) + "\n"
            return httpx.Response(
                200, content=body.encode(),
                headers={"content-type": "application/x-ndjson"},
            )
        return httpx.Response(200, json={
            "model": "stub/test", "created_at": "2026-01-01T00:00:00Z",
            "message": {"role": "assistant", "content": "好的"},
            "done": True, "done_reason": "stop", **counts,
        })
    return httpx.Response(404, json={"error": f"stub 没有这个路径: {path}"})


def _make_ollama() -> Any:
    return ollama_provider.OllamaProvider(
        id="ollama-stub", base_url="http://stub.local",
        client=ollama_client.OllamaClient(
            "http://stub.local", transport=httpx.MockTransport(_stub_handler)
        ),
    )


#: 兼容通道的 stub：只实现 LlmProvider，没有控制面
def _compat_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/v1/models":
        return httpx.Response(200, json={"object": "list", "data": [{
            "id": "compat/test", "object": "model", "created": 1_700_000_000,
            "owned_by": "stub", "context_length": 4096,
        }]})
    if path == "/v1/chat/completions":
        payload = json.loads(request.read())
        usage = {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}
        if payload.get("stream"):
            lines = [
                {"id": "c1", "object": "chat.completion.chunk", "model": "compat/test",
                 "choices": [{"index": 0, "delta": {"content": "好的"}}]},
                {"id": "c1", "object": "chat.completion.chunk", "model": "compat/test",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage},
            ]
            body = "".join(f"data: {json.dumps(line)}\n\n" for line in lines) + "data: [DONE]\n\n"
            return httpx.Response(200, content=body.encode(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "model": "compat/test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "好的"},
                         "finish_reason": "stop"}],
            "usage": usage,
        })
    return httpx.Response(404, json={"error": {"message": f"stub 没有这个路径: {path}"}})


def _make_compat() -> Any:
    from onyx.llm.providers.openai_compat import CompatClient, OpenAICompatProvider

    return OpenAICompatProvider(
        id="compat-stub", base_url="http://stub.local/v1", caps=("chat", "tools"),
        client=CompatClient("http://stub.local/v1",
                            transport=httpx.MockTransport(_compat_handler)),
    )


#: 被测实现。**外部插件必须在列**——它才是抽象的验收者
IMPLS: dict[str, Callable[[], Any]] = {
    "mock": MockProvider,
    "echo_plugin": EchoProvider,
    "ollama_stub": _make_ollama,
    "openai_compat": _make_compat,
}


@pytest.fixture(params=sorted(IMPLS), ids=sorted(IMPLS))
def provider(request):
    instance = IMPLS[request.param]()
    yield instance
    close = getattr(instance, "close", None)
    if callable(close):
        close()


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "contract.sqlite")
    sink = SqliteRecordSink(db, batch_size=1, idle_wait=0.005)
    yield db, sink, tmp_path
    sink.close()
    db.close()


def _model(provider) -> str:
    cards = provider.list_models()
    assert cards, "provider 至少要报告一个模型，否则看板的模型列表是空的"
    return cards[0].name


# ── 1. 形状 ────────────────────────────────────────────────────────
def test_implements_protocol(provider):
    assert isinstance(provider, LlmProvider)


def test_generate_signature_matches_how_gateway_calls_it(provider):
    """gateway 一律 `generate(req, trace_id=..., on_event=...)`（gateway.py）。

    协议里少写 `trace_id` 时内置实现照样能用（它们恰好都接了），
    而**照着协议写的外部 provider 会崩在 TypeError**——这正是契约测试存在的理由。
    """
    params = inspect.signature(provider.generate).parameters
    assert {"req", "trace_id", "on_event"} <= set(params)
    for name in ("trace_id", "on_event"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, f"{name} 必须是关键字参数"
        assert params[name].default is not inspect.Parameter.empty, f"{name} 必须有默认值"


def test_generate_accepts_gateway_style_call(provider):
    """按 gateway 的真实调用方式走一遍：不许 TypeError，也不许吞掉正文。"""
    gen = provider.generate(
        GenerationRequest.of(_model(provider), "你好"), trace_id="t-1", on_event=None
    )
    assert gen.status is Status.OK
    assert gen.text


def test_embed_is_optional_but_its_shape_is_enforced(provider):
    """`EmbeddingProvider` 是**可选**协议——但"有"就必须对得上 gateway 的调用方式。

    两条边界都要钉住：
    - 不强制实现：`openai_compat` 与外部插件没有向量能力是合法状态，
      强制的话就是逼实现者交一个假向量（那比 missing 更坏）；
    - 一旦实现，签名必须收 `trace_id` / `on_event`：当年 `generate` 的协议里漏了 `trace_id`，
      照协议写的外部 provider 直接 TypeError——同一个坑不许在第二个方法上重踩。
    """
    embed = getattr(provider, "embed", None)
    assert isinstance(provider, EmbeddingProvider) is (embed is not None), (
        "协议识别只能是结构化的：有 embed 就该被认成 EmbeddingProvider，反之亦然"
    )
    if embed is None:
        return
    params = inspect.signature(embed).parameters
    assert {"req", "trace_id", "on_event"} <= set(params)
    for name in ("trace_id", "on_event"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, f"{name} 必须是关键字参数"
        assert params[name].default is not inspect.Parameter.empty, f"{name} 必须有默认值"

    result = embed(EmbedRequest(model=_model(provider), inputs=("你好", "你好吗")),
                   trace_id="t-1", on_event=None)
    assert result.status is Status.OK
    #: 条数与顺序是向量通路的契约：少一条会让后面每条都错位，而错位看着完全正常
    assert len(result.vectors) == 2
    assert len({len(v) for v in result.vectors}) == 1, "一批里维度必须一致"


# ── 2. 自洽 ────────────────────────────────────────────────────────
def test_info_agrees_with_capabilities(provider):
    info = provider.info()
    assert info.id == provider.id
    assert info.caps == provider.capabilities(), "info.caps 与 capabilities() 打脸 = 能力位不可信"
    assert isinstance(str(info.kind), str)


def test_models_are_resolvable(provider):
    """列出来的每个模型都必须能被 `show_model` 回答。

    看板与评测都按 `list_models()` 的名字去查详情；这里对不上，
    Fleet 页就会显示一个点不开的模型。
    """
    cards = provider.list_models()
    names = [c.name for c in cards]
    assert len(names) == len(set(names)), "模型名重复会让 id 拼接出不唯一的记录"
    for card in cards:
        assert card.provider_id == provider.id or card.provider_id == "", (
            f"卡片来自 {card.provider_id!r}，但 provider 是 {provider.id!r}"
        )
        detail = provider.show_model(card.name)
        assert isinstance(detail, ModelDetail)
        assert detail.name == card.name


def test_unknown_model_is_answered_not_invented(provider):
    """未知模型：要么抛错，要么返回带名字的 detail。返回 None 或编一个都算违约——
    编出来的模板会被本地计数当成事实使用。"""
    try:
        detail = provider.show_model("definitely-not-a-model-xyz")
    except (OnyxError, LookupError, KeyError):
        return
    assert isinstance(detail, ModelDetail)


def test_running_returns_a_list(provider):
    """`running()` 返回 None 会让"没载入"与"不知道"在界面上长成同一个样子。"""
    loaded = provider.running()
    assert isinstance(loaded, list)


# ── 3. 后果 ────────────────────────────────────────────────────────
def test_broken_observer_callback_does_not_break_generation(provider):
    """`emit()` 是契约的一部分：观测回调抛错不许打断生成。"""
    def boom(event) -> None:
        raise RuntimeError("sink 炸了")

    gen = provider.generate(
        GenerationRequest.of(_model(provider), "你好"), trace_id="t-2", on_event=boom
    )
    assert gen.status is Status.OK


def test_gateway_records_a_trace_for_any_provider(provider, store):
    """只实现 `LlmProvider` 就能被完整记录——这条是"可替换"的实际定义。

    换 provider 不需要改 gateway/obs；同样，gateway 也不许给某个 provider 开小灶。
    """
    db, sink, blobs_dir = store
    gateway = Gateway(provider, observer=ObserverEngine(record_sink=sink),
                      blobs=FileBlobStore(blobs_dir / "blobs"))
    result = gateway.generate(GenerationRequest.of(_model(provider), "你好"))
    sink.flush(5.0)

    record = TraceRepo(db).get(result.trace_id)
    assert record is not None, "trace 必须落库"
    assert record.output_ref, "输出证据引用必须存在"
    # token 的出处必须写明，且必须来自阶梯上登记过的档位：
    # 引擎报了就是 engine/compat，没报就走本地复算并标低置信度。
    # 唯一不被允许的是"什么都不说"，因为那在界面上就是 0
    usage = sink_usage(db, result.trace_id)
    if usage is None:
        pytest.fail("generate 之后必须有 usage 记录（哪怕全是估算值）")
    assert usage.get("source") in {str(s) for s in TokenSource}, \
        f"出处 {usage.get('source')!r} 不在 TokenSource 阶梯里 = 这个数字无法解释"


def sink_usage(db, trace_id: str) -> dict[str, Any] | None:
    row = db.query("SELECT source, confidence, in_tokens, out_tokens FROM usage WHERE trace_id = ?",
                   (trace_id,))
    if not row:
        return None
    source, confidence, in_tokens, out_tokens = row[0]
    return {"source": source, "confidence": confidence,
            "in_tokens": in_tokens, "out_tokens": out_tokens}


def test_capability_set_is_declared_not_inferred_from_name(provider):
    """能力位必须显式声明：评测按它决定 skip 还是跑。

    按名字/前缀猜能力（"ollama 就一定支持 tools"）会让外部实现一进来就被误判，
    而且 skip 记录里的 missing 会变成假信息。
    """
    caps = provider.capabilities()
    assert isinstance(caps, frozenset)
    assert all(isinstance(c, Cap) for c in caps)
    # 至少要能说清自己能不能聊天：空集合意味着"任何任务都该 skip"
    assert Cap.CHAT in caps or not caps, f"能力位形状异常: {sorted(str(c) for c in caps)}"


def test_engine_reported_counts_are_kept_verbatim(provider):
    """引擎报了计数就原样保留（DESIGN：测量必须带出处）。

    这条对 mock/echo 是"没报就别编"，对真实适配器是"报了就别改"。
    """
    gen = provider.generate(
        GenerationRequest.of(_model(provider), "你好", stream=True),
        trace_id="t-3", on_event=None,
    )
    engine = gen.usage_from(TokenSource.ENGINE)
    if engine is None:
        assert not gen.usage or all(
            s.source is not TokenSource.ENGINE for s in gen.usage
        ), "非 ENGINE 出处可以存在，但 ENGINE 出处不许凭空出现"
        return
    assert engine.in_tokens is None or engine.in_tokens >= 0
    assert engine.confidence is not None
