"""计量层测试：保真阶梯、归因、对账、标定。

多处直接采用 docs/PROBES.md 的实测数值（644 token / 384.5ms 冷 / 83.4ms 热），
让测试与真机行为绑在一起，而不是绑在我当时的想象上。
"""

from __future__ import annotations

import pytest

from onyx.core.types import (
    Confidence,
    EngineLatency,
    Generation,
    GenerationRequest,
    Message,
    Role,
    TokenPart,
    TokenSample,
    TokenSource,
    ToolSpec,
)
from onyx.llm.measurement import (
    CalibrationSample,
    CompatCounter,
    CounterContext,
    EngineCounter,
    FittedCounter,
    HeuristicCounter,
    attribute,
    default_counters,
    estimate_tokens,
    fit_ratio,
    prefill_mode,
    reconcile,
)
from onyx.llm.measurement.calibrate import estimate_with_ratio
from onyx.llm.measurement.fidelity import visible_input_chars, visible_output_chars
from onyx.llm.measurement.parts import input_segments, part_totals
from onyx.llm.measurement.reconciler import latency_summary

# ── PROBES P11 的实测值 ────────────────────────────────────────────
CACHED_PROMPT_TOKENS = 644
COLD_PROMPT_EVAL_MS = 384.5
WARM_PROMPT_EVAL_MS = 83.4


def _req(*, tools: int = 0, system: str = "", user: str = "北京天气怎么样？") -> GenerationRequest:
    messages = []
    if system:
        messages.append(Message(role=Role.SYSTEM, content=system))
    messages.append(Message(role=Role.USER, content=user))
    specs = tuple(
        ToolSpec(name=f"tool_{i}", description="查询工具", parameters={
            "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
        })
        for i in range(tools)
    )
    return GenerationRequest(model="m", messages=tuple(messages), tools=specs)


def _gen(*, in_tokens=None, out_tokens=None, text="晴，21 度", thinking="", **kw) -> Generation:
    usage = ()
    if in_tokens is not None or out_tokens is not None:
        usage = (TokenSample(source=TokenSource.ENGINE, in_tokens=in_tokens, out_tokens=out_tokens),)
    return Generation(text=text, thinking=thinking, usage=usage, **kw)


# ── heuristic ─────────────────────────────────────────────────────
def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_cjk_not_undercounted_like_chars_div_4():
    """DESIGN R8：chars/4 对中文严重低估。"""
    text = "本地大模型观测与评测看板" * 4  # 48 个中文字符
    assert estimate_tokens(text) >= 40, "中文按 ~1 token/字 计，不能退化成 12"
    assert estimate_tokens("a" * 48) == 12, "拉丁字符仍按 chars/4"


def test_estimate_is_monotonic_and_configurable():
    short, long = "你好", "你好" * 100
    assert estimate_tokens(long) > estimate_tokens(short)
    assert estimate_tokens(long, cjk_tokens_per_char=2.0) == 400


# ── calibrate ─────────────────────────────────────────────────────
def test_fit_ratio_exact():
    samples = [CalibrationSample(chars=c, tokens=int(c * 0.62)) for c in (100, 500, 1000, 5000)]
    result = fit_ratio(samples)
    assert result.ratio == pytest.approx(0.62, abs=1e-3)
    assert result.r2 == pytest.approx(1.0, abs=1e-6)
    assert result.n == 4
    assert result.usable is False, "n<30 不许当 fitted 档用"


def test_fit_ratio_usable_threshold():
    samples = [CalibrationSample(chars=c, tokens=int(c * 0.6)) for c in range(100, 1000, 20)]
    result = fit_ratio(samples)
    assert result.n >= 30 and result.r2 >= 0.9 and result.usable


def test_fit_ratio_records_cold_warm_speed():
    """P11 的冷/热 ms-per-token 必须被标定下来，否则 prefill_mode 只能拍脑袋。"""
    samples = [
        CalibrationSample(chars=1000, tokens=644, prompt_eval_ms=COLD_PROMPT_EVAL_MS, cold=True),
        CalibrationSample(chars=1000, tokens=644, prompt_eval_ms=WARM_PROMPT_EVAL_MS, cold=False),
    ]
    result = fit_ratio(samples)
    assert result.cold_ms_per_token == pytest.approx(0.597, abs=0.01)
    assert result.warm_ms_per_token == pytest.approx(0.129, abs=0.01)
    assert result.cold_ms_per_token / result.warm_ms_per_token == pytest.approx(4.65, abs=0.2)


def test_fit_ratio_handles_garbage():
    assert fit_ratio([]).n == 0
    assert fit_ratio([CalibrationSample(chars=0, tokens=0)]).n == 0
    assert estimate_with_ratio(100, 0.62) == 62


# ── fidelity 阶梯 ─────────────────────────────────────────────────
def test_engine_counter_passthrough():
    gen = _gen(in_tokens=280, out_tokens=26)
    sample = EngineCounter().count(_req(), gen, CounterContext())
    assert sample.in_tokens == 280 and sample.source is TokenSource.ENGINE


def test_engine_counter_absent_returns_none():
    assert EngineCounter().count(_req(), Generation(), CounterContext()) is None


def test_heuristic_counter_includes_tools_and_per_message_overhead():
    plain = HeuristicCounter().count(_req(), _gen(text="晴"), CounterContext())
    with_tools = HeuristicCounter().count(_req(tools=3), _gen(text="晴"), CounterContext())
    assert with_tools.in_tokens > plain.in_tokens, "工具定义必须计入输入成本"
    assert plain.in_tokens >= 2 * CounterContext().per_message_tokens


def test_fitted_counter_refuses_when_uncalibrated():
    """没标定过就不许冒充 fitted —— 宁可退回 heuristic + low。"""
    sample = FittedCounter().count(_req(), _gen(), CounterContext(fitted_ratio=None))
    assert sample.ok is False and sample.confidence is Confidence.LOW
    assert "未标定" in sample.note

    few = FittedCounter().count(_req(), _gen(), CounterContext(fitted_ratio=0.6, fitted_n=5))
    assert few.ok is False


def test_fitted_counter_uses_ratio():
    ctx = CounterContext(fitted_ratio=0.5, fitted_n=200)
    sample = FittedCounter().count(_req(user="a" * 100), _gen(text="b" * 40), ctx)
    assert sample.ok and sample.confidence is Confidence.MEDIUM
    assert sample.in_tokens == 50 + ctx.per_message_tokens
    assert sample.out_tokens == 20


def test_compat_counter_only_present_when_reported():
    assert CompatCounter().count(_req(), _gen(), CounterContext()) is None
    gen = Generation(usage=(TokenSample(source=TokenSource.COMPAT, in_tokens=20),))
    assert CompatCounter().count(_req(), gen, CounterContext()).in_tokens == 20


def test_visible_chars_helpers():
    req = _req(system="你是助手", tools=1)
    assert visible_input_chars(req) > visible_input_chars(_req(system="你是助手"))
    assert visible_output_chars(_gen(text="abc", thinking="def")) == 6


def test_default_counters_order():
    priorities = [c.priority for c in default_counters()]
    assert priorities == sorted(priorities), "阶梯必须按可信度排序"
    assert default_counters()[0].name is TokenSource.ENGINE


# ── reconciler ────────────────────────────────────────────────────
def test_reconcile_prefers_engine():
    samples = (
        TokenSample(source=TokenSource.ENGINE, in_tokens=280, out_tokens=26),
        TokenSample(source=TokenSource.HEURISTIC, in_tokens=270, out_tokens=24),
    )
    result = reconcile(samples)
    assert result.usage.source is TokenSource.ENGINE
    assert result.usage.confidence is Confidence.HIGH
    assert result.usage.in_tokens == 280
    codes = [c for c, _ in result.anomalies]
    assert "TOKEN_DRIFT" not in codes and "NO_ENGINE_COUNT" not in codes
    assert result.usage.drift_pct == pytest.approx(abs(280 - 270) / 280, abs=1e-3)


def test_reconcile_degrades_without_engine_and_reports_it():
    samples = (TokenSample(source=TokenSource.FITTED, in_tokens=270, out_tokens=20),)
    result = reconcile(samples)
    assert result.usage.source is TokenSource.FITTED
    assert result.usage.confidence is Confidence.MEDIUM
    assert "NO_ENGINE_COUNT" in [c for c, _ in result.anomalies]


def test_reconcile_flags_low_confidence():
    result = reconcile((TokenSample(source=TokenSource.HEURISTIC, in_tokens=100),))
    assert result.usage.confidence is Confidence.LOW
    assert "LOW_CONFIDENCE_USAGE" in [c for c, _ in result.anomalies]


def test_reconcile_detects_drift():
    samples = (
        TokenSample(source=TokenSource.ENGINE, in_tokens=1000),
        TokenSample(source=TokenSource.HEURISTIC, in_tokens=600),
    )
    result = reconcile(samples, drift_threshold=0.10)
    codes = {c: d for c, d in result.anomalies}
    assert "TOKEN_DRIFT" in codes
    assert codes["TOKEN_DRIFT"]["drift_pct"] == pytest.approx(0.4, abs=1e-3)


def test_reconcile_ignores_small_absolute_drift():
    """真机实测：19 token 的短 prompt 上启发式给 16，百分比 16% 纯噪声。

    报警必须同时满足"相对偏差超阈值"和"绝对差够大"，否则每条短请求都告警，
    真正的口径分裂反而被淹没（告警疲劳）。偏差值本身仍然记录，只是不报警。
    """
    samples = (
        TokenSample(source=TokenSource.ENGINE, in_tokens=19),
        TokenSample(source=TokenSource.HEURISTIC, in_tokens=16),
    )
    result = reconcile(samples)
    assert "TOKEN_DRIFT" not in [c for c, _ in result.anomalies]
    assert result.usage.drift_pct == pytest.approx(3 / 19, abs=1e-3)


def test_reconcile_drift_fires_on_real_gap():
    samples = (
        TokenSample(source=TokenSource.ENGINE, in_tokens=301),
        TokenSample(source=TokenSource.HEURISTIC, in_tokens=90),
    )
    result = reconcile(samples)
    detail = dict(result.anomalies)["TOKEN_DRIFT"]
    assert detail["abs_diff"] == 211


def test_compat_is_never_chosen():
    """P14：兼容层与原生不一致 ⇒ 永不采信，哪怕它是唯一来源。"""
    result = reconcile((TokenSample(source=TokenSource.COMPAT, in_tokens=20),))
    assert result.usage.source is not TokenSource.COMPAT
    assert "NO_USAGE_SOURCE" in [c for c, _ in result.anomalies]
    assert result.usage.in_tokens is None, "宁可空着也不填一个不可信的数字"


def test_compat_deviation_is_noted():
    samples = (
        TokenSample(source=TokenSource.ENGINE, in_tokens=22),
        TokenSample(source=TokenSource.COMPAT, in_tokens=20),
    )
    result = reconcile(samples)
    assert any("compat" in n for n in result.notes)


def test_reconcile_without_any_sample():
    result = reconcile(())
    assert result.usage.in_tokens is None
    assert "NO_USAGE_SOURCE" in [c for c, _ in result.anomalies]


# ── prefill_mode（P11）────────────────────────────────────────────
def test_prefill_mode_uses_measured_numbers():
    assert prefill_mode(CACHED_PROMPT_TOKENS, COLD_PROMPT_EVAL_MS) == "cold"
    assert prefill_mode(CACHED_PROMPT_TOKENS, WARM_PROMPT_EVAL_MS) == "warm"


def test_prefill_mode_unknown_without_data():
    assert prefill_mode(None, 100.0) == "unknown"
    assert prefill_mode(644, None) == "unknown"
    assert prefill_mode(0, 10.0) == "unknown"


def test_prefill_mode_threshold_configurable():
    assert prefill_mode(644, 200.0, warm_threshold_ms_per_token=0.5) == "warm"
    assert prefill_mode(644, 200.0, warm_threshold_ms_per_token=0.1) == "cold"


def test_latency_summary_separates_cold_and_warm():
    cold = _gen(
        in_tokens=644, out_tokens=10, text="x",
        latency=EngineLatency(prompt_eval_ns=384_500_000, eval_ns=200_000_000, load_ns=900_000_000),
    )
    warm = _gen(
        in_tokens=644, out_tokens=10, text="x",
        latency=EngineLatency(prompt_eval_ns=83_400_000, eval_ns=200_000_000, load_ns=1_000_000),
    )
    cold_summary = latency_summary(cold)
    warm_summary = latency_summary(warm)
    assert cold_summary["prefill_mode"] == "cold" and warm_summary["prefill_mode"] == "warm"
    assert cold_summary["cold_load"] is True and warm_summary["cold_load"] is False
    assert cold_summary["prefill_tps"] < warm_summary["prefill_tps"]
    assert cold_summary["prefill_tps"] == pytest.approx(1675, abs=10), "冷：644/0.3845s"
    assert warm_summary["prefill_tps"] == pytest.approx(7722, abs=50), "热：644/0.0834s（虚高 4.65×）"


# ── parts 归因 ────────────────────────────────────────────────────
def test_input_segments_split_system_tools_messages():
    req = _req(system="你是助手", tools=2, user="北京天气")
    parts = [s.part for s in input_segments(req)]
    assert parts == ["system", "tool_defs", "msg:1"], f"实际 {parts}"


def test_attribute_residual_equals_engine_count():
    """Σ分段 + template_ctl = 引擎计数（P9 决定的归因口径）。"""
    req = _req(system="你是助手", tools=1)
    count = lambda text: len(text)  # noqa: E731 - 测试用确定性计数
    segments_total = sum(len(s.text) for s in input_segments(req))
    parts, report = attribute(req, count_fn=count, engine_in=segments_total + 37, gen_text="晴")
    totals = part_totals(parts)
    assert totals["template_ctl"] == 37
    assert report.input_segments_tokens == segments_total
    assert report.complete is True
    assert totals["tool_defs"] > 0, "工具定义开销必须可单独查出（DESIGN §6.2）"
    assert totals["output"] == 1


def test_attribute_without_engine_count_has_no_residual():
    parts, report = attribute(_req(), count_fn=len)
    assert report.residual_raw is None and report.template_ctl_tokens == 0
    assert "template_ctl" not in part_totals(parts)


def test_attribute_flags_overestimate_instead_of_hiding_it():
    """分段和超过引擎计数 ⇒ count_fn 高估或引擎截断，必须显式标记而不是静默 clamp。"""
    req = _req(user="x" * 100)
    parts, report = attribute(req, count_fn=lambda t: len(t) * 3, engine_in=50)
    assert report.clamped is True and report.complete is False
    assert report.residual_raw == 50 - 300
    assert part_totals(parts)["template_ctl"] == 0


def test_attribute_records_bytes():
    parts, _ = attribute(_req(user="中文"), count_fn=len)
    msg = next(p for p in parts if p.part == "msg:0")
    assert msg.bytes == len("中文".encode()) == 6


def test_token_part_is_core_type():
    assert isinstance(TokenPart(part="x", tokens=1), TokenPart)
