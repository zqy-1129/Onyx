"""OpenAI 兼容通道 provider（vLLM / LM Studio / Xinference / TGI / Ollama 的 `/v1`）。

它的第二个身份是**抽象的验收测试**：只实现 `LlmProvider`，不实现 `AdminProvider`，
就能让看板、评测、导出全部工作。做不到就说明抽象泄漏（由
`scripts/check_extension_boundary.py` 判定，不靠人 review）。

与 Ollama 原生通道的三处真实差异，都在这里显式承认而不是抹平：
1. **计数出处是 `compat` 不是 `engine`**：同一份 prompt 在 `/v1` 与原生通道上的计数
   口径可能不同（S3 的交叉验证就是为了这个），混成一个出处会把差异藏起来；
2. **没有纳秒级分段时序**：`latency` 为空 ⇒ 吞吐、prefill 模式、TTFT 一律显示「—」，
   绝不填 0（DESIGN 原则：未知与 0 是两个相反的事实）；
3. **流式必须显式打开 `stream_options.include_usage`**，否则 usage 恒为 0（R2）。

控制面（加载/卸载/拉取/删除）在这些服务器上**没有统一接口**，所以这里不实现
`AdminProvider`，也不声明 `Cap.ADMIN`：编一个假的 unload 会让"显存已经让出来了"
这种关键判断建立在谎话上。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx

from onyx.core.clock import SYSTEM_CLOCK, Clock
from onyx.core.errors import (
    CapabilityMissing,
    ProviderRejected,
    ProviderUnreachable,
    RequestTimeout,
)
from onyx.core.types import (
    ApiStyle,
    Cap,
    Generation,
    GenerationRequest,
    LoadedModel,
    Message,
    ModelCard,
    ModelDetail,
    ProviderInfo,
    ProviderKind,
    Role,
    ToolSpec,
)
from onyx.llm.params import to_openai_params
from onyx.llm.providers.base import EventCB
from onyx.llm.streaming import StreamAssembler, consume_chunks, emit_final_events

DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)

#: 操作者可以声明的字符串 → 能力位。声明即断言，探针实测会覆盖它（`onyx/llm/caps.py`）。
_CAP_NAMES: dict[str, Cap] = {str(c): c for c in Cap}

#: `req.tool_choice` 里这几个词是 OpenAI 的保留值，其它一律按"函数名"处理
_TOOL_CHOICE_LITERALS = frozenset({"auto", "none", "required"})


class CompatClient:
    """只做传输与错误归一的极简客户端。

    刻意不复用 `OllamaClient`：那是 ollama 包的内部件。provider 之间互相引用，
    扩展点就变成"必须理解内置实现"，而外部实现恰恰是来验证契约够不够用的。
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # 密钥在这里解析而不是在 provider 里：provider 可以自带 client（测试与自定义传输），
        # 若解析放在 provider 层，`client=` + `api_key=` 同时给出时密钥会被静默丢掉
        key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        auth = {"Authorization": f"Bearer {key}"} if key else {}
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, headers={**auth, **(headers or {})},
            transport=transport,
        )

    def get_json(self, path: str) -> Any:
        return self._send("GET", path)

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        return self._send("POST", path, json_body=payload)

    def post_sse(self, path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """`text/event-stream`：只认 `data:` 行，`[DONE]` 是结束标记。

        解析不出 JSON 的行**保留原文**（`_unparsed`）而不是丢掉——"引擎返回了我们看不懂
        的东西"本身就是要被观测的事实。
        """
        started = time.monotonic()
        try:
            with self._client.stream("POST", path, json=payload) as resp:
                self._raise_for_status(resp, started)
                for line in resp.iter_lines():
                    if not line or not line.strip():
                        continue
                    if not line.startswith("data:"):
                        continue  # 事件流里的 `event:`/注释行不是数据
                    body = line[len("data:"):].strip()
                    if body == "[DONE]":
                        return
                    try:
                        yield json.loads(body)
                    except json.JSONDecodeError:
                        yield {"_unparsed": body}
        except httpx.TimeoutException as exc:
            raise RequestTimeout(
                f"流式请求超时: {path}", detail={"path": path}
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(
                f"无法连接 OpenAI 兼容服务: {type(exc).__name__}: {exc}",
                base_url=self.base_url,
            ) from exc

    def _send(self, method: str, path: str, *, json_body: dict[str, Any] | None = None) -> Any:
        started = time.monotonic()
        try:
            resp = self._client.request(method, path, json=json_body)
        except httpx.TimeoutException as exc:
            raise RequestTimeout(f"{method} {path} 超时") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(
                f"无法连接 OpenAI 兼容服务（{type(exc).__name__}）。服务是否已启动？",
                base_url=self.base_url,
            ) from exc
        self._raise_for_status(resp, started)
        if not resp.content:
            return {}
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise ProviderRejected(
                f"响应不是合法 JSON: {path}", status=resp.status_code, body=resp.text
            ) from exc

    def _raise_for_status(self, resp: httpx.Response, started: float) -> None:
        if resp.status_code < 400:
            return
        body = resp.text[:2000]
        message = body
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                message = str((parsed.get("error") or {}).get("message")
                              or parsed.get("error") or message)
        except (json.JSONDecodeError, AttributeError):
            pass
        raise ProviderRejected(
            f"OpenAI 兼容服务返回 {resp.status_code}: {message[:300]}",
            status=resp.status_code, body=body,
            detail={"elapsed_ms": round((time.monotonic() - started) * 1000)},
        )

    def close(self) -> None:
        self._client.close()


def build_payload(
    req: GenerationRequest, *, stream: bool, thinking_via: str = ""
) -> dict[str, Any]:
    """归一化请求 → `/v1/chat/completions` 请求体。

    纪律与原生通道一致：**没设的参数不出现**。填了默认值就等于冻结了一个假设，
    而且评测结果将无法解释"到底是以什么参数跑的"。
    """
    payload: dict[str, Any] = {
        "model": req.model,
        "messages": [parse_message(m) for m in req.messages],
        "stream": stream,
    }
    for key, value in to_openai_params(req.params).items():
        if key == "_dropped_params":
            continue  # 丢弃项由 gateway 记进参数快照（`_would_drop_on_openai_channel`）
        payload[key] = value
    if req.tools:
        payload["tools"] = [
            {"type": "function", "function": {
                "name": t.name, "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            }}
            for t in req.tools
        ]
    if req.tool_choice:
        payload["tool_choice"] = parse_tool_choice(req.tool_choice, req.tools)
    if req.params.json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True,
                            "schema": req.params.json_schema},
        }
    if req.thinking is not None:
        # P23 实测：Ollama 的 `/v1` 既不认 `think` 也不认 `chat_template_kwargs.thinking`
        # （照样产出 reasoning，并把正文挤空）。没有标准开关的通道上，
        # "要求关 thinking 却被静默忽略"会让两次分数不可比，所以要么按声明的方式翻译，
        # 要么显式拒绝——绝不假装设置过了。
        if not thinking_via:
            raise CapabilityMissing(
                "该兼容通道未配置 thinking 开关：请求要求 "
                f"thinking={bool(req.thinking)}，而 /v1 没有标准字段能表达它。"
                "vLLM 之类可在 provider 构造时声明 thinking_via="
                '"chat_template_kwargs.thinking"；否则请改用原生通道',
                detail={"thinking": bool(req.thinking), "thinking_via": thinking_via},
            )
        _assign_path(payload, thinking_via.split("."), bool(req.thinking))
    if stream:
        # 不开这个，流式的 usage 恒为 0（R2）——那会被读成"模型没产 token"
        payload["stream_options"] = {"include_usage": True}
    # 引擎特有参数（vLLM 的 `chat_template_kwargs`、LM Studio 的 `response_format` 变体等）
    # 由调用方经 `req.extra` 显式带入：provider 不猜服务器型号，也不按名字开小灶
    for key, value in (req.extra or {}).items():
        payload[key] = value
    return payload


def _assign_path(target: dict[str, Any], path: list[str], value: Any) -> None:
    """按点号路径写入嵌套键（`chat_template_kwargs.thinking`）。"""
    for part in path[:-1]:
        nxt = target.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise CapabilityMissing(
                f"thinking_via={ '.'.join(path) } 的路径与已有参数冲突: {part}",
                detail={"path": path},
            )
        target = nxt
    target[path[-1]] = value


def parse_tool_choice(choice: str, tools: tuple[ToolSpec, ...]) -> Any:
    if choice in _TOOL_CHOICE_LITERALS:
        return choice
    names = {t.name for t in tools}
    if choice not in names:
        # 指定的工具不在本次请求的工具集里：这必然是配置或提示词写错了。
        # 悄悄降级成 "auto" 会产出一个"看起来强制调用了"的分数，而它其实没有
        raise CapabilityMissing(
            f"tool_choice={choice!r} 不在本次工具集 {sorted(names)} 里",
            detail={"tool_choice": choice, "tools": sorted(names)},
        )
    return {"type": "function", "function": {"name": choice}}


def parse_message(msg: Message) -> dict[str, Any]:
    if msg.media_refs:
        # 兼容通道的图片要 data URL + mime，缺 mime 就是编造事实；
        # 声明 vision 之前必须先把这条路径补上，而不是静默丢图
        raise CapabilityMissing(
            "openai-compat 通道未实现多模态消息：图片会被静默丢弃，"
            "而「模型没看到图」与「模型看到了图但没答对」是两种相反的结论",
            detail={"refs": list(msg.media_refs), "role": str(msg.role)},
        )
    item: dict[str, Any] = {"role": str(msg.role), "content": msg.content or ""}
    if msg.role is Role.TOOL:
        if not msg.tool_call_id:
            raise CapabilityMissing(
                "工具结果消息缺少 tool_call_id，兼容通道无法把它关联到调用",
                detail={"name": msg.name or ""},
            )
        item["tool_call_id"] = msg.tool_call_id
    if msg.name and msg.role is not Role.TOOL:
        item["name"] = msg.name
    if msg.tool_calls:
        item["tool_calls"] = [
            {"id": c.id or f"call_{c.index}", "type": "function",
             "function": {"name": c.name,
                          "arguments": json.dumps(c.arguments or {}, ensure_ascii=False)}}
            for c in msg.tool_calls
        ]
    return item


class OpenAICompatProvider:
    """`/v1/chat/completions` 通道。数据面完整，控制面刻意缺席。"""

    kind = ProviderKind.OPENAI_COMPAT

    def __init__(
        self,
        id: str = "openai-compat",
        base_url: str = "http://127.0.0.1:8000/v1",
        *,
        api_key: str = "",
        caps: tuple[str, ...] | list[str] | frozenset[Cap] = ("chat",),
        thinking_via: str = "",
        clock: Clock = SYSTEM_CLOCK,
        client: CompatClient | None = None,
        timeout: httpx.Timeout | float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
        **_extra: Any,
    ) -> None:
        self.id = id
        self.base_url = base_url.rstrip("/")
        self.clock = clock
        self._caps = self._validate_caps(caps)
        #: 该服务器的 thinking 开关位置（点号路径）。空 = 无法表达，见 `build_payload`
        self.thinking_via = thinking_via
        # api_key 走 `OPENAI_API_KEY`（事实标准），不在 onyx 里另造一套 flag：
        # 再发明一个配置入口就会有两处真值，而密钥最容易在两处都写成明文
        self.client = client or CompatClient(
            self.base_url, api_key=api_key, timeout=timeout, transport=transport
        )
        self.calls: list[GenerationRequest] = []

    @staticmethod
    def _validate_caps(caps: Any) -> frozenset[Cap]:
        """声明能力位时**不许静默吞掉拼写错误**。

        `caps=("chat","tool_chioce")` 若被忽略，评测就会在一个没人支持的能力上
        跑出"正常"的分数；拼错必须当场炸。
        """
        if isinstance(caps, frozenset):
            return caps
        out: set[Cap] = set()
        unknown: list[str] = []
        for raw in caps or ():
            cap = _CAP_NAMES.get(str(raw))
            if cap is None:
                unknown.append(str(raw))
            else:
                out.add(cap)
        if unknown:
            raise ValueError(
                f"未知能力位 {unknown}；可声明的: {sorted(_CAP_NAMES)}"
                "（拼错能力位会让评测在没人支持的能力上产出正常分数）"
            )
        return frozenset(out)

    # ── 元信息 ────────────────────────────────────────────────────
    def info(self) -> ProviderInfo:
        reachable = False
        try:
            raw = self.client.get_json("/models")
            reachable = isinstance(raw, dict)
        except Exception:  # noqa: BLE001 - 体检路径：不可达本身就是合法结论
            pass
        # 版本：OpenAI 兼容通道没有统一版本端点（vLLM 有 /health 与 /v1/models 但不报版本），
        # 拿不到就留空 ⇒ 界面显示「—」，而不是猜一个 "unknown" 进数据库当事实
        return ProviderInfo(
            id=self.id, kind=self.kind, base_url=self.base_url, api_style=ApiStyle.OPENAI,
            version="", reachable=reachable, caps=self.capabilities(),
            extra={"reachable_via": "/v1/models"},
        )

    def capabilities(self) -> frozenset[Cap]:
        return self._caps

    # ── 控制面（只读部分）──────────────────────────────────────────
    def list_models(self) -> list[ModelCard]:
        raw = self.client.get_json("/models")
        items = raw.get("data") or [] if isinstance(raw, dict) else []
        out: list[ModelCard] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = str(item.get("id") or "")
            if not name:
                continue
            meta = item.get("meta") or {}
            out.append(ModelCard(
                provider_id=self.id, name=name, model=str(item.get("owned_by") or ""),
                digest=str(meta.get("digest") or ""),
                # created 是秒级时间戳，转成 ISO 才与原生通道的口径一致
                modified_at=_iso(item.get("created")),
                context_length=_int_or_none(item.get("context_length")
                                           or meta.get("n_ctx")),
                capabilities=tuple(str(c) for c in (item.get("capabilities") or ())),
                extra={"owned_by": str(item.get("owned_by") or "")},
            ))
        return out

    def show_model(self, name: str) -> ModelDetail:
        """兼容通道没有 `/api/show`：模板、GGUF 元数据、量化一律未知。

        未知就标未知。本地 token 复算的 `template_ctl` 一档会因此拿不到模板，
        归因表上必须显示"模板未知"，而不是用一个假模板算出一个像模像样的数
        （那会把归因误差伪装成测量结果）。
        """
        known = {card.name: card for card in self.list_models()}
        card = known.get(name)
        if card is None:
            raise LookupError(
                f"{self.id} 的 /v1/models 里没有 {name!r}；"
                f"有: {sorted(known)[:20]}（服务器可能需要不同的模型名或尚未加载）"
            )
        return ModelDetail(
            name=name, template="", capabilities=card.capabilities, parameters="",
            extra={
                "template_source": "unknown",
                "why": "OpenAI 兼容通道不暴露 chat template；本地归因只能到消息级",
                "owned_by": card.extra.get("owned_by", ""),
            },
        )

    def running(self) -> list[LoadedModel]:
        """驻留状态在这个通道上问不到 ⇒ 不声明 `Cap.ADMIN`，上层显示「未知」。

        返回空列表本身是安全的（没有任何模型被声称"在显存里"），
        但**它必须是"因为不能问所以空"**，而不是"空=没载入"——这正是 ADMIN
        能力位要区分的事。
        """
        return []

    # ── 数据面 ────────────────────────────────────────────────────
    def generate(
        self,
        req: GenerationRequest,
        *,
        trace_id: str = "",
        on_event: EventCB | None = None,
    ) -> Generation:
        self.calls.append(req)
        if req.keep_alive is not None:
            raise CapabilityMissing(
                "keep_alive 是 Ollama 原生概念，兼容通道没有等价物；"
                "驻留策略请交给服务器（vLLM 的 --max-model-len / LM Studio 的 unload）",
                detail={"keep_alive": req.keep_alive},
            )
        if req.stream:
            return consume_chunks(
                self.client.post_sse(
                    "/chat/completions", build_payload(req, stream=True,
                                                      thinking_via=self.thinking_via)
                ),
                trace_id=trace_id, clock=self.clock, style="openai",
                model=req.model, on_event=on_event,
            )
        raw = self.client.post_json(
            "/chat/completions", build_payload(req, stream=False,
                                               thinking_via=self.thinking_via)
        )
        assembler = StreamAssembler(style="openai", model=req.model)
        # 非流式响应也要走同一个 assembler：两条路径的口径必须一致，
        # 否则"流式测出来的工具解析问题"与"非流式测出来的"就不是同一件事
        assembler.feed(raw if isinstance(raw, dict) else {})
        gen = assembler.build(model=req.model)
        emit_final_events(
            gen, raw if isinstance(raw, dict) else {}, trace_id=trace_id,
            clock=self.clock, on_event=on_event, ttft_ms=None,
        )
        return gen

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> OpenAICompatProvider:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _iso(value: Any) -> str:
    """`/v1/models` 的 `created` 是 unix 秒；转成 ISO 才与原生通道对齐。"""
    seconds = _int_or_none(value)
    if seconds is None:
        return ""
    return datetime.fromtimestamp(seconds, UTC).isoformat()


__all__ = [
    "CompatClient",
    "OpenAICompatProvider",
    "build_payload",
    "parse_message",
    "parse_tool_choice",
]
