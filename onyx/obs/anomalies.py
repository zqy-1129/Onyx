"""异常码表：观测层唯一的"什么算不正常"定义。

码字符串是**持久化契约**（落 `anomaly.code`，前端按它分支与配色），
只增不改；要改语义就加新码并把旧码标 deprecated。
"""

from __future__ import annotations

from dataclasses import dataclass


class Severity:
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AnomalySpec:
    code: str
    severity: str
    meaning: str
    action: str


#: code → 规格。UI 的异常说明文案直接从这里取，避免前后端各写一份漂移。
SPECS: dict[str, AnomalySpec] = {
    spec.code: spec
    for spec in (
        # ── 计量口径 ──────────────────────────────────────────────
        AnomalySpec("NO_ENGINE_COUNT", Severity.ERROR,
                    "引擎没有返回 token 计数",
                    "检查是否走了不报计数的通道；流式需确认末事件带 usage"),
        AnomalySpec("LOW_CONFIDENCE_USAGE", Severity.WARN,
                    "采信来源只有低置信档位（heuristic）",
                    "跑 onyx calibrate 标定该模型，或补 tokenizer"),
        AnomalySpec("TOKEN_DRIFT", Severity.WARN,
                    "采信值与独立复算值偏差超阈值",
                    "口径可能分裂；对比 usage_alt 各来源，检查模板或 tokenizer"),
        AnomalySpec("COMPAT_DIVERGENCE", Severity.INFO,
                    "/v1 兼容层计数与原生不一致（P14 已知）",
                    "无需处理，compat 永不采信"),
        AnomalySpec("ATTRIBUTION_CLAMPED", Severity.WARN,
                    "分段计数之和超过引擎计数",
                    "count_fn 高估或引擎发生截断；归因不可信，需换更高保真档位"),
        # ── 缓存与吞吐（P11）──────────────────────────────────────
        AnomalySpec("PREFILL_CACHE_HIT", Severity.INFO,
                    "prefill 命中 KV 缓存，吞吐为等效值而非真实计算吞吐",
                    "聚合时必须与 cold 分列，不要合并成同一个 P50"),
        AnomalySpec("COLD_LOAD", Severity.INFO,
                    "本次请求包含模型载入（冷启动）",
                    "TTFT 与吞吐应归入 cold 分组"),
        AnomalySpec("CONTEXT_NEAR_LIMIT", Severity.WARN,
                    "输入 token 接近实际载入上下文上限",
                    "提高 num_ctx 或裁剪历史/工具定义"),
        AnomalySpec("CONTEXT_OVERFLOW", Severity.ERROR,
                    "输入超过实际载入上下文，引擎可能已截断",
                    "结果不可信；必须提高 num_ctx 后重跑"),
        # ── 工具调用 ──────────────────────────────────────────────
        AnomalySpec("TRUNCATED_TOOL_JSON", Severity.WARN,
                    "工具调用参数 JSON 被截断（多为 max_tokens 不足）",
                    "提高 max_tokens；这是预算问题不是能力问题"),
        AnomalySpec("MALFORMED_TOOL_JSON", Severity.WARN,
                    "工具调用参数不是合法 JSON",
                    "看 args_raw 原文判断是格式能力问题还是模板/停止词配置问题"),
        AnomalySpec("UNKNOWN_TOOL", Severity.WARN,
                    "模型调用了注册表里不存在的工具（幻觉工具名）",
                    "检查工具描述与命名；必要时在 prompt 中限定可用工具"),
        AnomalySpec("ORPHAN_TOOL_CALL", Severity.ERROR,
                    "finish_reason 要求执行工具但没有可执行的调用",
                    "必须补占位 tool 消息，否则后续上下文永久错位"),
        AnomalySpec("TOOL_LOOP", Severity.ERROR,
                    "同一 (工具, 参数) 被重复调用",
                    "熔断并检查提示词；本地小模型常见"),
        AnomalySpec("BUDGET_EXCEEDED", Severity.WARN,
                    "步数/时间/token 预算耗尽", "提高预算或简化任务"),
        AnomalySpec("TOOL_ERROR", Severity.WARN,
                    "工具执行失败", "先跑 onyx tools contract 区分是工具坏了还是模型用错了"),
        # ── 能力与端点 ────────────────────────────────────────────
        AnomalySpec("EMBED_UNSUPPORTED", Severity.ERROR,
                    "向量化请求打到了没有实现 EmbeddingProvider 的引擎",
                    "换有 embedding 能力的模型（Ollama 的 card capabilities 含 embedding）；"
                    "评测侧应整场 skip 并写明原因，而不是改用 chat 拼一个假向量"),
        # ── 输出形态 ──────────────────────────────────────────────
        AnomalySpec("EMPTY_CONTENT_WITH_THINKING", Severity.WARN,
                    "正文为空但产出了推理内容（P5/P12：预算被 thinking 吃光）",
                    "提高 max_tokens 或设 think=false；评测中这不算答错"),
        AnomalySpec("THINKING_LEAK", Severity.WARN,
                    "推理内容混进了正文", "检查流式分栏；会破坏工具 JSON 解析"),
        AnomalySpec("EMPTY_OUTPUT", Severity.WARN, "既无正文也无推理内容",
                    "检查停止词、模板与 max_tokens"),
        # ── 运行时 ────────────────────────────────────────────────
        AnomalySpec("OFFLOADED_TO_CPU", Severity.WARN,
                    "部分权重落在 CPU（size_vram < size）",
                    "吞吐会低一个数量级，不可与全量载入的结果混算"),
        AnomalySpec("UNPARSED_STREAM_LINE", Severity.WARN,
                    "流式响应中出现无法解析的行", "保留原文用于诊断；检查引擎版本"),
        AnomalySpec("PROVIDER_ERROR", Severity.ERROR, "引擎调用失败",
                    "先跑 onyx doctor 确认服务与模型状态"),
        AnomalySpec("OBSERVER_ERROR", Severity.ERROR,
                    "观测组件自身抛错（已被隔离，不影响请求）",
                    "这是 Onyx 的 bug，请带 trace_id 提 issue"),
    )
}


def severity_of(code: str) -> str:
    spec = SPECS.get(code)
    return spec.severity if spec else Severity.WARN


def describe(code: str) -> str:
    spec = SPECS.get(code)
    return spec.meaning if spec else f"未登记的异常码: {code}"


def all_codes() -> tuple[str, ...]:
    return tuple(sorted(SPECS))
