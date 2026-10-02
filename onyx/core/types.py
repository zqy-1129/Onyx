"""L0 领域类型：provider 无关的规范化数据形状。

三条纪律：
1. **枚举值即持久化值**——字符串一旦落库就是契约，不要改字面量（要改就加新值 + 迁移）。
2. **每个结构都留 `extra: dict`**——引擎返回的新字段先进 extra，需要聚合时再提升为字段（原则 6）。
3. **不做任何 IO / 序列化到具体 API**——`to_ollama_*` 这类映射属于 L2 `llm/params.py`。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


# ── 稳定字符串枚举 ─────────────────────────────────────────────────
def _is_unset(value: Any) -> bool:
    """"未设置"判定：None 与空容器。注意 0 / 0.0 / False 是**有意义的值**，不算未设置。"""
    return value is None or value == () or value == {} or value == []


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    DEVELOPER = "developer"


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    EOS = "eos"
    CANCELLED = "cancelled"
    ERROR = "error"
    UNKNOWN = "unknown"


class Status(StrEnum):
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TokenSource(StrEnum):
    """token 计数的出处（DESIGN §6.1 保真阶梯）。数字没有出处就不许上看板。"""

    ENGINE = "engine"
    COMPAT = "compat"
    HF_TOKENIZER = "hf_tokenizer"
    GGUF_VOCAB = "gguf_vocab"
    FITTED = "fitted"
    HEURISTIC = "heuristic"


#: 采信优先级：越靠前越可信。reconciler 按此顺序挑第一个 ok 的来源。
SOURCE_PRIORITY: tuple[TokenSource, ...] = (
    TokenSource.ENGINE,
    TokenSource.HF_TOKENIZER,
    TokenSource.GGUF_VOCAB,
    TokenSource.FITTED,
    TokenSource.HEURISTIC,
)
#: 只用于交叉验证、不参与采信
CROSSCHECK_SOURCES: frozenset[TokenSource] = frozenset({TokenSource.COMPAT})


class Cap(StrEnum):
    """Provider 能力位。评测按它决定 skip 还是跑（禁止隐式降级）。"""

    CHAT = "chat"
    TOOLS = "tools"
    TOOL_CHOICE = "tool_choice"
    STRUCTURED_OUTPUT = "structured_output"
    THINKING = "thinking"
    VISION = "vision"
    EMBED = "embed"
    STREAM_USAGE = "stream_usage"
    N_SAMPLING = "n_sampling"
    LOGPROBS = "logprobs"
    ADMIN = "admin"


class ProviderKind(StrEnum):
    OLLAMA = "ollama"
    OPENAI_COMPAT = "openai_compat"
    VLLM = "vllm"
    LMSTUDIO = "lmstudio"
    LLAMA_CPP = "llama_cpp"
    MOCK = "mock"


class ApiStyle(StrEnum):
    NATIVE = "native"
    OPENAI = "openai"


class ToolFormat(StrEnum):
    """模型侧工具调用的序列化形态，由 probe/tool_format 实测得出。"""

    NATIVE_HEAD = "native_head"
    CHAT_TEMPLATE = "chat_template"
    XML = "xml"
    JSON = "json"
    UNKNOWN = "unknown"


class ParseStatus(StrEnum):
    OK = "ok"
    JSON_ERROR = "json_error"
    UNKNOWN_TOOL = "unknown_tool"
    MISSING_REQUIRED = "missing_required"
    TYPE_MISMATCH = "type_mismatch"
    TRUNCATED = "truncated"
    NOT_PARSABLE = "not_parsable"


class ExecutedBy(StrEnum):
    CLIENT = "client"
    SERVER = "server"
    MOCK = "mock"
    FIXTURE = "fixture"
    SKIPPED = "skipped"


class TraceKind(StrEnum):
    GENERATION = "generation"
    EMBED = "embed"
    RERANK = "rerank"
    HEALTH = "health"
    ADMIN = "admin"
    JUDGE = "judge"


class TracePurpose(StrEnum):
    CHAT = "chat"
    PLAYGROUND = "playground"
    EVAL = "eval"
    TOOL_TEST = "tool_test"
    PROBE = "probe"
    BENCH = "bench"


# ── 请求侧 ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ToolSpec:
    """OpenAI function-calling 形状的规范化表示。"""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters or {"type": "object", "properties": {}},
            },
        }

    @classmethod
    def from_openai_tool(cls, raw: dict[str, Any]) -> ToolSpec:
        fn = raw.get("function") or raw
        return cls(
            name=str(fn.get("name", "")),
            description=str(fn.get("description", "")),
            parameters=dict(fn.get("parameters") or {}),
            extra={k: v for k, v in raw.items() if k not in {"type", "function"}},
        )


@dataclass(frozen=True, slots=True)
class ToolCall:
    """模型请求的一次工具调用。`arguments_raw` 在解析失败时必须保留原文。"""

    index: int = 0
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] | None = None
    arguments_raw: str = ""
    parse_status: ParseStatus = ParseStatus.OK
    parse_source: Literal["native_head", "text_template", "heuristic", "fixture"] = "native_head"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    thinking: str = ""
    #: 图片等二进制只存 blob 指针，绝不进消息体（DESIGN §7.3）
    media_refs: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.content and not self.tool_calls and not self.media_refs


@dataclass(frozen=True, slots=True)
class GenParams:
    """归一化生成参数。None = 不传给引擎（用引擎默认），不要写死默认值。"""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    seed: int | None = None
    stop: tuple[str, ...] = ()
    num_ctx: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repeat_penalty: float | None = None
    #: 结构化输出：json schema dict；None 表示不强制
    json_schema: dict[str, Any] | None = None
    #: 引擎特有参数（如 ollama 的 min_p / typical_p）原样透传
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, *, drop_none: bool = True) -> dict[str, Any]:
        raw = dataclasses.asdict(self)
        if drop_none:
            raw = {k: v for k, v in raw.items() if not _is_unset(v)}
        if "stop" in raw:
            raw["stop"] = list(raw["stop"])
        return raw

    def merge(self, **overrides: Any) -> GenParams:
        base = dataclasses.asdict(self)
        base["stop"] = tuple(self.stop)
        base.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(GenParams)}
        extra = dict(self.extra)
        for key in list(base):
            if key not in known:
                extra[key] = base.pop(key)
        base["extra"] = extra
        return GenParams(**base)


@dataclass(frozen=True, slots=True)
class TraceContext:
    """把"这条 trace 属于谁"带下去：评测样本、工具测试、探针都要能反查。"""

    kind: TraceKind = TraceKind.GENERATION
    purpose: TracePurpose = TracePurpose.CHAT
    eval_run_id: str | None = None
    case_id: str | None = None
    sample_seq: int | None = None
    parent_trace_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def purpose_label(self) -> str:
        """落库用的 purpose 字符串，评测形如 `eval:tool_selection`。"""
        if self.purpose is TracePurpose.EVAL and self.eval_run_id:
            return f"eval:{self.eval_run_id}"
        return str(self.purpose)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model: str
    messages: tuple[Message, ...]
    params: GenParams = field(default_factory=GenParams)
    tools: tuple[ToolSpec, ...] = ()
    tool_choice: str | None = None
    stream: bool = False
    thinking: bool | None = None
    keep_alive: str | None = None
    context: TraceContext = field(default_factory=TraceContext)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)

    def with_messages(self, messages: tuple[Message, ...] | list[Message]) -> GenerationRequest:
        return dataclasses.replace(self, messages=tuple(messages))

    def with_params(self, **overrides: Any) -> GenerationRequest:
        return dataclasses.replace(self, params=self.params.merge(**overrides))

    @classmethod
    def of(cls, model: str, prompt: str, **kw: Any) -> GenerationRequest:
        """便捷构造：单轮用户消息。"""
        return cls(model=model, messages=(Message(role=Role.USER, content=prompt),), **kw)


# ── 响应侧 ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class EngineLatency:
    """引擎自报的分段耗时（纳秒）。全 None 表示引擎没报——不要填 0。"""

    total_ns: int | None = None
    load_ns: int | None = None
    prompt_eval_ns: int | None = None
    eval_ns: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_cold(self) -> bool:
        """load 超过阈值即视为冷启动，必须与热请求分开聚合（DESIGN §6.4）。"""
        return bool(self.load_ns and self.load_ns > 50_000_000)

    def ms(self, which: Literal["total", "load", "prompt_eval", "eval"]) -> float | None:
        value = getattr(self, f"{which}_ns")
        return None if value is None else value / 1e6


@dataclass(frozen=True, slots=True)
class TokenSample:
    """单一来源的一次计数。`ok=False` 表示该来源本次不可用（note 说明原因）。"""

    source: TokenSource
    in_tokens: int | None = None
    out_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    ok: bool = True
    confidence: Confidence = Confidence.HIGH
    note: str = ""


@dataclass(frozen=True, slots=True)
class TokenPart:
    """归因：只有能渲染 chat template 时才产得出（DESIGN §6.2）。"""

    part: str  # bos|system|tool_defs|msg:<idx>|gen_prompt|image|template_ctl
    ord: int = 0
    tokens: int = 0
    bytes: int = 0


@dataclass(frozen=True, slots=True)
class ReconciledUsage:
    """采信结果：看板只读这个。"""

    source: TokenSource
    confidence: Confidence
    in_tokens: int | None = None
    out_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    drift_pct: float | None = None
    alts: tuple[TokenSample, ...] = ()
    parts: tuple[TokenPart, ...] = ()


@dataclass(frozen=True, slots=True)
class Generation:
    text: str = ""
    thinking: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: FinishReason = FinishReason.UNKNOWN
    status: Status = Status.OK
    error: str = ""
    usage: tuple[TokenSample, ...] = ()
    latency: EngineLatency | None = None
    ttft_ms: float | None = None
    wall_ms: float | None = None
    raw_response_ref: str = ""
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def usage_from(self, source: TokenSource) -> TokenSample | None:
        return next((s for s in self.usage if s.source == source), None)

    @property
    def decode_tps(self) -> float | None:
        sample = self.usage_from(TokenSource.ENGINE)
        if not sample or sample.out_tokens is None or not self.latency or not self.latency.eval_ns:
            return None
        return sample.out_tokens / (self.latency.eval_ns / 1e9)

    @property
    def prefill_tps(self) -> float | None:
        sample = self.usage_from(TokenSource.ENGINE)
        if not sample or sample.in_tokens is None or not self.latency or not self.latency.prompt_eval_ns:
            return None
        return sample.in_tokens / (self.latency.prompt_eval_ns / 1e9)


# ── 控制面 ─────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ModelCard:
    """/api/tags 的规范化形状。"""

    provider_id: str
    name: str
    model: str = ""
    remote_model: str = ""
    remote_host: str = ""
    modified_at: str = ""
    bytes: int = 0
    digest: str = ""
    family: str = ""
    families: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization: str = ""
    format: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size_gb(self) -> float:
        return round(self.bytes / (1024**3), 2)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """/api/ps 的规范化形状：看板 Fleet 页的数据源。"""

    name: str
    model: str = ""
    digest: str = ""
    size: int = 0
    size_vram: int = 0
    context_length: int | None = None
    expires_at: str = ""
    parameter_size: str = ""
    quantization: str = ""
    family: str = ""
    parent_model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def offloaded(self) -> bool:
        """部分权重落在 CPU 上 ⇒ 吞吐数字不可与全 GPU 载入混算。"""
        return self.size > 0 and self.size_vram < self.size

    @property
    def vram_share(self) -> float | None:
        return None if not self.size else round(self.size_vram / self.size, 4)


@dataclass(frozen=True, slots=True)
class ModelDetail:
    """/api/show 的规范化形状。`model_info` 是 raw GGUF 元数据，本地计数的原料。"""

    name: str
    template: str = ""
    capabilities: tuple[str, ...] = ()
    model_info: dict[str, Any] = field(default_factory=dict)
    parameters: str = ""
    license: str = ""
    system: str = ""
    thinking: bool | None = None
    modified_at: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def tokenizer_family(self) -> str:
        """GGUF 的 BPE 家族名（gpt2/llama/qwen2/...），决定本地复算走哪条实现。"""
        return str(self.model_info.get("tokenizer.ggml.model", ""))

    @property
    def chat_template(self) -> str:
        return str(self.model_info.get("tokenizer.chat_template", self.template))


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    id: str
    kind: ProviderKind
    base_url: str
    api_style: ApiStyle
    version: str = ""
    reachable: bool = False
    caps: frozenset[Cap] = frozenset()
    extra: dict[str, Any] = field(default_factory=dict)

    def supports(self, cap: Cap) -> bool:
        return cap in self.caps


@dataclass(frozen=True, slots=True)
class ProbeFinding:
    """一次语义实测的结论。`unknown=True` 表示没测出来——UI 显示「—」而非 0。"""

    probe: str
    subject: str
    verdict: str
    unknown: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
    provider_version: str = ""


@dataclass(frozen=True, slots=True)
class ProbeReport:
    provider_id: str
    findings: tuple[ProbeFinding, ...] = ()
    started_at: str = ""
    finished_at: str = ""

    def by_probe(self, name: str) -> ProbeFinding | None:
        return next((f for f in self.findings if f.probe == name), None)


@dataclass(frozen=True, slots=True)
class AdminResult:
    ok: bool
    action: str
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""
