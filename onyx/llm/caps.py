"""能力位推断：把「引擎自报」与「探针实测」合成三态结论。

三态是刚需，不是讲究：`✗ 不支持` 与 `? 没测出来` 在评测里是完全不同的处置——
前者应当 skip 并写明原因，后者应当先跑探针。**合成一个状态就会要么错杀模型、
要么拿一个没验证过的能力去跑评测然后得到无法解释的分数。**
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.core.types import ApiStyle, Cap, ProviderKind

ALL_CAPS: frozenset[Cap] = frozenset(Cap)

#: 引擎 /api/tags 与 /api/show 的 capabilities 字符串 → 我们的能力位
ENGINE_CAP_MAP: dict[str, Cap] = {
    "completion": Cap.CHAT,
    "tools": Cap.TOOLS,
    "thinking": Cap.THINKING,
    "vision": Cap.VISION,
    "embedding": Cap.EMBED,
}

#: 这些能力 Ollama 明确不提供（官方文档列出），无需探针即可判定 missing
OLLAMA_KNOWN_MISSING: frozenset[Cap] = frozenset({Cap.TOOL_CHOICE, Cap.N_SAMPLING, Cap.LOGPROBS})

#: 探针结论 → 能力判定。键是 "<probe>:<verdict>"
PROBE_CAP_RULES: dict[str, tuple[Cap, str]] = {
    "structured:enforced": (Cap.STRUCTURED_OUTPUT, "confirmed"),
    "structured:partially_enforced": (Cap.STRUCTURED_OUTPUT, "missing"),
    "structured:not_enforced_invalid_json": (Cap.STRUCTURED_OUTPUT, "missing"),
    "stream_usage:identical": (Cap.STREAM_USAGE, "confirmed"),
    "stream_usage:stream_final_event_missing_counts": (Cap.STREAM_USAGE, "missing"),
    "stream_usage:divergent": (Cap.STREAM_USAGE, "unknown"),
    "think:thinking_not_produced_by_this_model": (Cap.THINKING, "unknown"),
    "think:thinking_included_in_eval_count": (Cap.THINKING, "confirmed"),
    "think:thinking_excluded_from_eval_count": (Cap.THINKING, "confirmed"),
    "tool_format:native_head": (Cap.TOOLS, "confirmed"),
    "tool_format:plain_text_no_tool_call": (Cap.TOOLS, "unknown"),
    "tool_format:no_output": (Cap.TOOLS, "unknown"),
}


@dataclass(frozen=True, slots=True)
class CapReport:
    confirmed: frozenset[Cap] = frozenset()
    missing: frozenset[Cap] = frozenset()
    unknown: frozenset[Cap] = frozenset()
    reasons: dict[str, str] = field(default_factory=dict)

    def supports(self, cap: Cap) -> bool:
        return cap in self.confirmed

    def state(self, cap: Cap) -> str:
        if cap in self.confirmed:
            return "confirmed"
        if cap in self.missing:
            return "missing"
        return "unknown"

    def symbol(self, cap: Cap) -> str:
        return {"confirmed": "✓", "missing": "✗", "unknown": "?"}[self.state(cap)]

    def skip_reason(self, cap: Cap) -> str:
        """评测 skip 时必须带的原因（DESIGN §9.1：禁止静默降级）。"""
        state = self.state(cap)
        if state == "confirmed":
            return ""
        reason = self.reasons.get(str(cap), "")
        if state == "missing":
            return f"模型不支持 {cap}（{reason}）" if reason else f"模型不支持 {cap}"
        return f"{cap} 未经实测确认（{reason}）；先跑 onyx probe run" if reason else \
            f"{cap} 未经实测确认；先跑 onyx probe run"

    def as_dict(self) -> dict[str, Any]:
        return {
            "confirmed": sorted(str(c) for c in self.confirmed),
            "missing": sorted(str(c) for c in self.missing),
            "unknown": sorted(str(c) for c in self.unknown),
            "reasons": dict(self.reasons),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CapReport:
        raw = raw or {}

        def caps(key: str) -> frozenset[Cap]:
            out = set()
            for item in raw.get(key) or []:
                try:
                    out.add(Cap(str(item)))
                except ValueError:
                    continue
            return frozenset(out)

        return cls(
            confirmed=caps("confirmed"), missing=caps("missing"), unknown=caps("unknown"),
            reasons={str(k): str(v) for k, v in (raw.get("reasons") or {}).items()},
        )


def infer_caps(
    *,
    engine_capabilities: Sequence[str] = (),
    probe_findings: Mapping[str, str] | None = None,
    api_style: ApiStyle | str = ApiStyle.NATIVE,
    provider_kind: ProviderKind | str = ProviderKind.OLLAMA,
) -> CapReport:
    """合成三态能力报告。探针结论**优先于**引擎自报（自报可能撒谎或粒度不够）。"""
    findings = dict(probe_findings or {})
    style = ApiStyle(api_style) if not isinstance(api_style, ApiStyle) else api_style
    kind = ProviderKind(provider_kind) if not isinstance(provider_kind, ProviderKind) else provider_kind

    confirmed: set[Cap] = set()
    missing: set[Cap] = set()
    unknown: set[Cap] = set()
    reasons: dict[str, str] = {}

    def put(cap: Cap, state: str, reason: str) -> None:
        # 必须先从三态里全部移除再落位：探针结论会覆盖引擎自报，
        # 若只是"加进新集合"，同一个 cap 会同时处于两态（而 state() 只按优先级读一个）。
        confirmed.discard(cap)
        missing.discard(cap)
        unknown.discard(cap)
        reasons[str(cap)] = reason
        {"confirmed": confirmed, "missing": missing, "unknown": unknown}[state].add(cap)

    # 1) 引擎自报
    engine_caps = {ENGINE_CAP_MAP[c] for c in engine_capabilities if c in ENGINE_CAP_MAP}
    # 空清单的含义是"**这个通道不上报 capabilities**"（OpenAI 兼容的 /v1/models 就是这样），
    # 不是"上报了且什么都不支持"。把它读成 missing 会显示一排 ✗，
    # 而 ✗ 的处置是"评测直接 skip"，?的处置是"先跑探针"——两者相反（DESIGN §9.1）。
    reported = bool(engine_capabilities)
    for cap in (Cap.CHAT, Cap.TOOLS, Cap.THINKING, Cap.VISION, Cap.EMBED):
        if cap in engine_caps:
            put(cap, "confirmed", f"引擎自报 capabilities 含 {cap}")
        elif reported:
            put(cap, "missing", "引擎 capabilities 未列出")
        else:
            put(cap, "unknown", "该通道不汇报 per-model capabilities，未经实测不能判不支持")

    # 2) 通道/引擎层面的已知事实
    if kind is ProviderKind.OLLAMA and style is ApiStyle.NATIVE:
        for cap in OLLAMA_KNOWN_MISSING:
            put(cap, "missing", "Ollama 官方文档列为不支持")
        put(Cap.ADMIN, "confirmed", "已实现 /api/tags|show|ps|delete|pull 控制面")
    else:
        put(Cap.TOOL_CHOICE, "confirmed" if style is ApiStyle.OPENAI else "unknown",
            "OpenAI 兼容通道支持 tool_choice" if style is ApiStyle.OPENAI else "未实测")
        put(Cap.ADMIN, "unknown", "该通道未实现控制面")
        for cap in (Cap.N_SAMPLING, Cap.LOGPROBS):
            put(cap, "unknown", "未实测")

    # 3) 探针结论覆盖（实测优先）
    for key, verdict in findings.items():
        rule = PROBE_CAP_RULES.get(f"{key}:{verdict}")
        if rule is None:
            continue
        cap, state = rule
        put(cap, state, f"探针 {key} = {verdict}")

    # 4) 需要探针但还没跑的 → unknown（不是 missing）
    for cap, probe_name in ((Cap.STRUCTURED_OUTPUT, "structured"), (Cap.STREAM_USAGE, "stream_usage")):
        if cap in missing and probe_name not in findings:
            missing.discard(cap)
            put(cap, "unknown", f"需探针 {probe_name} 实测，尚未运行")

    leftover = ALL_CAPS - confirmed - missing - unknown
    for cap in leftover:
        put(cap, "unknown", "未覆盖")

    return CapReport(
        confirmed=frozenset(confirmed), missing=frozenset(missing),
        unknown=frozenset(unknown), reasons=reasons,
    )
