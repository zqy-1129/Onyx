"""落库记录类型：与表结构一一对应的不可变 dataclass。

为什么不让 repo 直接吃 `core.types`：core 是**领域形状**（provider 无关、面向调用方），
record 是**持久化形状**（含 blob 指针、已归一的字符串列）。两者分开，
改表结构不会波及领域类型，反之亦然。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from onyx.core.types import Confidence, TokenSource


@dataclass(frozen=True, slots=True)
class ProviderRecord:
    id: str
    kind: str
    base_url: str
    api_style: str
    enabled: bool = True
    caps: tuple[str, ...] = ()
    version: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class ModelRecord:
    id: str
    provider_id: str
    name: str
    remote_model: str = ""
    remote_host: str = ""
    digest: str = ""
    bytes: int | None = None
    modified_at: str = ""
    family: str = ""
    families: tuple[str, ...] = ()
    parameter_size: str = ""
    quantization: str = ""
    format: str = ""
    parent_model: str = ""
    ctx_train: int | None = None
    capabilities: tuple[str, ...] = ()
    template: str = ""
    model_info: dict[str, Any] = field(default_factory=dict)
    tool_format: str = "unknown"
    tokenizer_source: str = "none"
    tokenizer_ref: str = ""
    usage_ratio: float | None = None
    usage_ratio_n: int | None = None
    probe: dict[str, Any] = field(default_factory=dict)
    first_seen_at: str = ""
    last_seen_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


#: 允许被"事后回灌"的列（探针结论、tokenizer 标定）。白名单防注入也防误写。
MODEL_UPDATABLE_COLUMNS: frozenset[str] = frozenset(
    {
        "tool_format",
        "tokenizer_source",
        "tokenizer_ref",
        "usage_ratio",
        "usage_ratio_n",
        "probe_json",
        "capabilities_json",
        "model_info_json",
        "template",
        "ctx_train",
        "last_seen_at",
        "extra_json",
    }
)


@dataclass(frozen=True, slots=True)
class TraceRecord:
    id: str
    kind: str
    purpose: str
    started_at: str
    status: str = "ok"
    parent_id: str | None = None
    root_id: str | None = None
    eval_run_id: str | None = None
    case_id: str | None = None
    sample_seq: int | None = None
    provider_id: str | None = None
    model_id: str | None = None
    model_name: str | None = None
    first_token_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    messages_ref: str | None = None
    tools_ref: str | None = None
    rendered_prompt_ref: str | None = None
    output_ref: str | None = None
    raw_request_ref: str | None = None
    raw_response_ref: str | None = None
    finish_reason: str | None = None
    engine_latency: dict[str, Any] = field(default_factory=dict)
    gpu: dict[str, Any] = field(default_factory=dict)
    keep_alive: str | None = None
    contract_version: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageRecord:
    trace_id: str
    source: TokenSource | str
    confidence: Confidence | str
    in_tokens: int | None = None
    out_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    ttft_ms: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    wall_ms: float | None = None
    bytes_out: int | None = None
    drift_pct: float | None = None
    #: cold|warm|unknown —— 吞吐聚合必须按它分列（PROBES P11）
    prefill_mode: str | None = None
    prefill_ms_per_token: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UsageAltRecord:
    trace_id: str
    source: TokenSource | str
    in_tokens: int | None = None
    out_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    ok: bool = True
    confidence: Confidence | str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class TokenPartRecord:
    trace_id: str
    part: str
    ord: int = 0
    tokens: int = 0
    bytes: int | None = None


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    id: str
    trace_id: str
    step: int
    parse_status: str
    name: str | None = None
    call_id: str | None = None
    args: dict[str, Any] | None = None
    args_raw: str | None = None
    parse_source: str | None = None
    result_status: str | None = None
    result_ref: str | None = None
    result_bytes: int | None = None
    started_at: str | None = None
    latency_ms: float | None = None
    tool_id: str | None = None
    tool_def_hash: str | None = None
    executed_by: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnomalyRecord:
    id: str
    code: str
    severity: str
    trace_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
