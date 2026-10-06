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


@dataclass(frozen=True, slots=True)
class ToolDefRecord:
    id: str
    name: str
    version: str
    kind: str
    schema_json: dict[str, Any]
    hash: str
    impl_ref: str = ""
    tokens: int | None = None
    bytes: int | None = None
    tags: tuple[str, ...] = ()
    owner: str = ""
    enabled: bool = True
    side_effect: str = "read"
    timeout_ms: int | None = None
    doc: str = ""
    examples: tuple[dict[str, Any], ...] = ()
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True, slots=True)
class ToolTestRecord:
    id: str
    tool_id: str
    name: str
    args: dict[str, Any]
    expect: dict[str, Any] | None = None
    checks: tuple[dict[str, Any], ...] = ()
    live: bool = False
    created_at: str = ""


@dataclass(frozen=True, slots=True)
class ToolRunRecord:
    id: str
    status: str
    started_at: str
    tool_id: str | None = None
    tool_def_hash: str | None = None
    test_id: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    output_ref: str | None = None
    error: str | None = None
    deterministic: bool | None = None
    idempotent: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ── 评测（L5）─────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class DatasetRecord:
    id: str
    imported_at: str
    upstream: str = ""
    revision: str = ""
    license: str = ""
    splits: dict[str, int] = field(default_factory=dict)
    n_cases: int | None = None
    loader: str = ""
    notes: str = ""


@dataclass(frozen=True, slots=True)
class CaseRecord:
    """一条评测样本的落库形状。与 `eval.task.Case` 一一对应。"""

    id: str
    dataset_id: str
    input: dict[str, Any]
    expect: dict[str, Any]
    ord: int = 0
    kind: str = "single"
    tools: tuple[dict[str, Any], ...] = ()
    fixture: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TaskRecord:
    id: str
    name: str
    metrics: tuple[str, ...] = ()
    grader: dict[str, Any] = field(default_factory=dict)
    dataset_id: str | None = None
    sample_params: dict[str, Any] = field(default_factory=dict)
    k: int = 1
    budget: dict[str, Any] = field(default_factory=dict)
    sandbox: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunRecord:
    id: str
    task_id: str
    model_id: str
    started_at: str
    status: str = "running"
    provider_id: str | None = None
    finished_at: str | None = None
    seed: int | None = None
    app_version: str = ""
    git_rev: str = ""
    params_snapshot: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    n_cases: int = 0
    n_done: int = 0
    n_error: int = 0
    n_skipped: int = 0
    aggregate: dict[str, Any] = field(default_factory=dict)
    #: 评测自身的开销。judge 也是本地模型时同样吃 GPU 与时间，必须计入（DESIGN R13）
    cost: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: 数据集来历（id + revision）。对比与回归的全部结论都建立在"两次跑的是同一份数据"
    #: 这件事上，所以它必须是记录的一部分，而不是靠 case_id 反推出来的推测
    dataset_id: str | None = None
    dataset_revision: str = ""


@dataclass(frozen=True, slots=True)
class GradeRecord:
    id: str
    eval_run_id: str
    case_id: str
    score: float
    verdict: str
    graded_at: str
    seq: int = 0
    trace_id: str | None = None
    passed: bool | None = None
    invalid_format: bool = False
    out_of_set: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    judge_model_id: str | None = None
    judge_usage: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AlertTriggerRecord:
    """一次告警命中在某个渠道上的投递结果。

    存在的理由只有一个：回答"那天到底通知没通知"。所以它记的是**命中 + 尝试投递**，
    被 cooldown 抑制的不写（那是没发生的事），渠道失败必须写并带原因。
    """

    id: str
    created_at: str
    code: str
    severity: str
    #: 生效规则的快照（阈值/窗口/codes/cooldown + 出处）。规则以后会被改，
    #: 而"当时为什么触发"必须以当时那份判据解释，不能拿现在的规则去倒推
    rule: dict[str, Any] = field(default_factory=dict)
    n_in_window: int = 0
    window_s: int = 0
    first_anomaly_id: str | None = None
    last_anomaly_id: str | None = None
    #: 样本 trace_id（够下钻就行，不是全量清单）
    trace_ids: tuple[str, ...] = ()
    channel: str = ""
    status: str = "sent"
    detail: str = ""
    is_test: bool = False


@dataclass(frozen=True, slots=True)
class PerfRunRecord:
    """一次 `onyx perf` 运行的**条件**。

    一张基线能不能被拿来比，全部取决于这里：数字相同而条件不同就是两个不同的事实。
    `env_hash` 是可比性指纹（字段清单在 `onyx.perf.bench.FINGERPRINT_FIELDS`），
    `comparable=False` 表示连"哪台引擎、哪个模型"都没认出来——这时 compare 必须拒绝，
    而不是拿着一个可能张冠李戴的哈希给结论。
    """

    id: str
    started_at: str
    finished_at: str
    status: str
    model: str
    provider_id: str = ""
    engine_version: str = ""
    quantization: str = ""
    device: str = ""
    num_ctx: int | None = None
    keep_alive: str = ""
    stream: bool = True
    timing_source: str = "unknown"
    env_hash: str = ""
    comparable: bool = False
    conditions: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0
    n_requests: int = 0
    app_version: str = ""
    git_rev: str = ""
    note: str = ""
    error: str = ""


@dataclass(frozen=True, slots=True)
class PerfCellRecord:
    """一个格子的结果。`status != "measured"` 时 `metrics` 是空的——
    "没测到"必须是一行带原因的记录，不是一行 0（那会被平均进去）。
    """

    run_id: str
    cell_key: str
    phase: str
    prompt_chars: int
    target_tokens: int
    concurrency: int
    repeat: int
    status: str = "measured"
    reason: str = ""
    n_requests: int = 0
    n_measured: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    trace_ids: tuple[str, ...] = ()
