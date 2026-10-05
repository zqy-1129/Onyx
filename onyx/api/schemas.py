"""API 响应模型。

刻意与 `store.records` 分开：记录是持久化形状，schema 是**对外契约**。
前端只应依赖这里，这样改表结构不会直接击穿 UI。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]


class SourceBadge(BaseModel):
    """每个数字的出处。前端必备组件的数据源（DESIGN 原则 4）。"""

    source: str
    confidence: Confidence
    note: str = ""


class UsageAlt(BaseModel):
    source: str
    in_tokens: int | None = None
    out_tokens: int | None = None
    thinking_tokens: int | None = None
    cached_tokens: int | None = None
    ok: bool = True
    confidence: str | None = None
    note: str = ""


class TokenPart(BaseModel):
    part: str
    ord: int = 0
    tokens: int = 0
    bytes: int | None = None


class LatencyView(BaseModel):
    ttft_ms: float | None = None
    wall_ms: float | None = None
    prefill_mode: Literal["cold", "warm", "unknown"] = "unknown"
    prefill_ms_per_token: float | None = None
    prefill_tps: float | None = None
    decode_tps: float | None = None
    load_ms: float | None = None
    cold_load: bool = False


class ToolCallView(BaseModel):
    id: str
    step: int
    name: str | None = None
    parse_status: str
    parse_source: str | None = None
    args: dict[str, Any] | None = None
    args_raw: str | None = None
    result_status: str | None = None
    latency_ms: float | None = None
    executed_by: str | None = None


class AnomalyView(BaseModel):
    id: str
    code: str
    severity: str
    meaning: str = ""
    action: str = ""
    detail: dict[str, Any] = Field(default_factory=dict)


class TraceSummary(BaseModel):
    id: str
    purpose: str
    kind: str
    model_name: str | None = None
    provider_id: str | None = None
    started_at: str
    status: str
    finish_reason: str | None = None
    in_tokens: int | None = None
    out_tokens: int | None = None
    source: str | None = None
    confidence: str | None = None
    ttft_ms: float | None = None
    decode_tps: float | None = None
    prefill_mode: str | None = None
    tool_calls: int = 0
    anomalies: int = 0
    eval_run_id: str | None = None
    case_id: str | None = None


class TraceDetail(BaseModel):
    trace: TraceSummary
    params: dict[str, Any] = Field(default_factory=dict)
    latency: LatencyView
    usage: UsageAlt | None = None
    alts: list[UsageAlt] = Field(default_factory=list)
    parts: list[TokenPart] = Field(default_factory=list)
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    anomalies: list[AnomalyView] = Field(default_factory=list)
    gpu: dict[str, Any] = Field(default_factory=dict)
    engine_latency: dict[str, Any] = Field(default_factory=dict)
    refs: dict[str, str] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    attribution: dict[str, Any] = Field(default_factory=dict)


class TracePage(BaseModel):
    items: list[TraceSummary]
    next_cursor: str | None = None
    total: int = 0


class LoadedModelView(BaseModel):
    name: str
    size: int = 0
    size_vram: int = 0
    vram_share: float | None = None
    offloaded: bool = False
    context_length: int | None = None
    expires_at: str = ""
    keep_alive_seconds: float | None = None
    quantization: str = ""
    parameter_size: str = ""


class ModelView(BaseModel):
    id: str
    name: str
    provider_id: str
    parameter_size: str = ""
    quantization: str = ""
    size_gb: float | None = None
    capabilities: list[str] = Field(default_factory=list)
    caps: dict[str, Any] = Field(default_factory=dict)
    tool_format: str = "unknown"
    ctx_train: int | None = None
    ctx_loaded: int | None = None
    tokenizer_source: str = "none"
    calibrated: bool = False
    calibration: dict[str, Any] = Field(default_factory=dict)
    probed: bool = False
    #: `None` = 该通道不报告驻留状态（例如 vLLM/LM Studio 的 OpenAI 兼容层）。
    #: 必须与 `False`（问了，答案是"没载入"）区分开——界面把未知画成"未载入"
    #: 会让人去查一个不存在的问题（UI_DESIGN R2）
    loaded: bool | None = None


class FleetView(BaseModel):
    ok: bool
    app_version: str
    provider_id: str
    provider_kind: str
    provider_reachable: bool
    engine_version: str = ""
    base_url: str = ""
    loaded_models: list[LoadedModelView] = Field(default_factory=list)
    #: 该通道能否回答"哪些模型在显存里"。`False` 时 `loaded_models` 是空且**无含义**，
    #: 界面必须显示「驻留状态未知」而不是"当前没有载入任何模型"
    loaded_known: bool = True
    installed_models: int = 0
    window: dict[str, Any] = Field(default_factory=dict)
    #: code → {n, severities}。级别由后端给：chip 在前端自己写死一个 warn 的话，
    #: error 级异常也会显示成"提醒"，而这两种事情的紧急程度完全不同
    anomalies: dict[str, Any] = Field(default_factory=dict)
    #: error 级异常的总览（n / by_code / 最近一条）。级别取异常行上记的那份，
    #: 不回查规格表——降级一个码不该让历史看起来"没出过事"
    error_anomalies: dict[str, Any] = Field(default_factory=dict)
    #: 告警系统的状态（出口、轮询次数、最近错误）。顶部那一行要能同时说清
    #: "出事了" 与 "通知系统本身好不好"，否则两者在界面上长得一样
    alerts: dict[str, Any] = Field(default_factory=dict)


class UsageSummaryView(BaseModel):
    traces: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    thinking_tokens: int = 0
    by_source: dict[str, int] = Field(default_factory=dict)
    by_confidence: dict[str, int] = Field(default_factory=dict)
    by_prefill_mode: dict[str, int] = Field(default_factory=dict)
    drift: dict[str, Any] = Field(default_factory=dict)
    timeseries: list[dict[str, Any]] = Field(default_factory=list)


class ChatRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    thinking: bool | None = False
    stream: bool = False
    tools: list[str] = Field(default_factory=list)
    system: str = ""
    #: 客户端生成的关联键。多模型并排时前端靠它把 SSE 事件流对上自己那次请求。
    #: 刻意不让客户端指定 trace_id：id 必须时间可排序（游标分页依赖它）。
    client_key: str = ""


class ChatResponse(BaseModel):
    trace_id: str
    text: str
    thinking: str = ""
    finish_reason: str
    tool_calls: list[ToolCallView] = Field(default_factory=list)
    usage: UsageAlt | None = None
    latency: LatencyView
    anomalies: list[AnomalyView] = Field(default_factory=list)
    parts: list[TokenPart] = Field(default_factory=list)


class HealthView(BaseModel):
    ok: bool
    version: str
    schema_version: int
    provider_reachable: bool
    engine_version: str = ""
