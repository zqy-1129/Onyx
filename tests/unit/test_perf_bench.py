"""采集与聚合（S36）。

不发真请求：假 gateway 造出**引擎形状**的结果（`EngineLatency` + 引擎回报的 token 数），
然后交给生产口径 `latency_summary()` 算 —— 这样测的是"perf 会不会篡改或误读这些数"，
而不是"perf 自己算得对不对"（那是 measurement 层的测试）。

刻意盯住三种失败：把"没测到"算成 0、把被引擎裁短的正文当成一次真测量、
以及把并发下的合计吞吐与单请求吞吐混成同一个数。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from onyx.core.clock import FakeClock
from onyx.core.types import (
    EngineLatency,
    Generation,
    Status,
    TokenSample,
    TokenSource,
    TracePurpose,
)
from onyx.llm.measurement.reconciler import latency_summary
from onyx.perf.bench import collect
from onyx.perf.corpus import cjk_count
from onyx.perf.report import fmt_spread
from onyx.perf.spec import BenchPlan

PROMPT_MS_PER_CHAR = 0.1         # 0.1 ms/汉字 ⇒ 低于 reconciler 的 0.30 阈值 ⇒ 判成 warm
DECODE_MS_PER_TOKEN = 50.0       # 每 token 50ms ⇒ decode_tps 恒为 20.0


class FakeGateway:
    """按请求给出可预期的引擎数字。`in_factor` 用来模拟"引擎把正文裁短了"。"""

    def __init__(self, *, ns: bool = True, in_factor: float = 1.0,
                 fail: bool = False, report_usage: bool = True,
                 load_ms: float = 0.0, ttft_ms: float | None = 12.5,
                 prompt_ms_per_char: float = PROMPT_MS_PER_CHAR) -> None:
        self.ns = ns
        self.in_factor = in_factor
        self.fail = fail
        self.report_usage = report_usage
        self.load_ms = load_ms
        self.ttft_ms = ttft_ms
        self.prompt_ms_per_char = prompt_ms_per_char
        self.seen: list[tuple[object, object]] = []
        self.calls = 0

    def generate(self, req, *, purpose=None):
        self.calls += 1
        self.seen.append((req, purpose))
        prompt = req.messages[-1].content
        out = req.params.max_tokens or 0
        declared = cjk_count(prompt)
        in_tokens = max(1, int(declared * self.in_factor)) if self.report_usage else None
        latency = None
        if self.ns:
            latency = EngineLatency(
                total_ns=int((declared * self.prompt_ms_per_char + out * DECODE_MS_PER_TOKEN
                              + self.load_ms) * 1e6),
                load_ns=int(self.load_ms * 1e6) if self.load_ms else None,
                prompt_eval_ns=int(declared * self.prompt_ms_per_char * 1e6),
                eval_ns=int(out * DECODE_MS_PER_TOKEN * 1e6),
            )
        usage = ()
        if self.report_usage:
            usage = (TokenSample(source=TokenSource.ENGINE, in_tokens=in_tokens,
                                 out_tokens=out),)
        generation = Generation(
            text="输" * out, model=req.model,
            status=Status.ERROR if self.fail else Status.OK,
            error="引擎 500" if self.fail else "",
            wall_ms=out * DECODE_MS_PER_TOKEN, ttft_ms=None if self.fail else self.ttft_ms,
            latency=latency, usage=usage,
        )
        return SimpleNamespace(
            trace_id=f"trace-{self.calls}", generation=generation,
            latency=latency_summary(generation),
        )


def _plan(**kw) -> BenchPlan:
    defaults = dict(model="qwen3.5:9b", prompt_chars=(600,), target_tokens=(64,),
                    concurrency=(1,), repeat=2, budget_s=600.0)
    return BenchPlan(**{**defaults, **kw})


def _collect(gateway, plan=None, **kw):
    plan = plan or _plan()
    info = {"provider_id": "ollama-local", "version": "0.35.1", "quantization": "Q4_K_M",
            "app_version": "0.8.0", "git_rev": "abc1234"}
    clock = kw.pop("clock", FakeClock())
    return collect(gateway, plan, clock=clock, engine_info=info, **kw)


# ── 请求形状 ──────────────────────────────────────────────────────
def test_every_request_carries_bench_purpose_and_the_planned_params():
    gateway = FakeGateway()
    _collect(gateway, _plan(keep_alive="5m", num_ctx=8192))
    assert gateway.calls == 2
    req, purpose = gateway.seen[0]
    assert purpose is TracePurpose.BENCH, "基线要能在 trace 页被单独筛出来，purpose 是唯一抓手"
    assert req.stream is True and req.keep_alive == "5m"
    assert req.params.max_tokens == 64 and req.params.num_ctx == 8192
    assert req.params.temperature == 0.0, "基线要可复现，温度默认钉 0"
    assert req.context.extra["perf_cell"] == "600c/64t/x1/warm", "trace 要能反查它是哪一格"


def test_no_stream_plan_is_honoured():
    gateway = FakeGateway()
    _collect(gateway, _plan(stream=False))
    assert gateway.seen[0][0].stream is False


# ── 数字来源 ──────────────────────────────────────────────────────
def test_decode_throughput_comes_from_the_engine_numbers():
    outcome = _collect(FakeGateway())
    metrics = outcome.cells[0].metrics
    assert metrics["decode_tps"]["median"] == pytest.approx(20.0)
    assert metrics["decode_tps"]["n"] == 2
    # prefill 按冷热分列，不合并：合并出来的 P50 既不代表冷启动也不代表稳态
    assert metrics["prefill_tps_warm"]["median"] == pytest.approx(10_000.0)
    assert metrics["prefill_tps_cold"] is None, "没测到的列是 None，不是 0"
    assert metrics["n_prefill_mode_unknown"] == 0


def test_slow_prefill_is_labeled_cold_not_blended_in():
    """0.30 ms/汉字 是 reconciler 的冷热线。超过它就必须落进 cold 那一列。"""
    metrics = _collect(FakeGateway(prompt_ms_per_char=0.9)).cells[0].metrics
    assert metrics["prefill_tps_cold"]["median"] == pytest.approx(1 / 0.0009, rel=1e-6)
    assert metrics["prefill_tps_warm"] is None


def test_concurrency_reports_two_throughputs_separately():
    """单请求吞吐与整批合计是两个问题：并发下它们必然分叉，只报一个就糊掉了 batching。"""
    serial = _collect(FakeGateway(), _plan(concurrency=(1,))).cells[0].metrics
    parallel = _collect(FakeGateway(), _plan(concurrency=(2,))).cells[0].metrics
    assert serial["decode_tps"]["median"] == pytest.approx(parallel["decode_tps"]["median"])
    assert parallel["aggregate_tps"]["median"] == pytest.approx(
        serial["aggregate_tps"]["median"] * 2)
    assert parallel["n_requests"] == 4 and parallel["n_measured"] == 4


def test_aggregate_is_dropped_when_the_engine_does_not_report_tokens():
    """少一路 out_tokens 也照样算得出合计——那个数会被读成"同条件"，所以整批判为未知。"""
    metrics = _collect(FakeGateway(report_usage=False)).cells[0].metrics
    assert "note" in metrics["aggregate_tps"] and "median" not in metrics["aggregate_tps"]
    assert metrics["decode_tps"] is None


# ── 三种"不许悄悄通过" ────────────────────────────────────────────
def test_truncated_cell_is_skipped_not_scored():
    """引擎回报的输入 token 低于正文的汉字下限 ⇒ 正文被裁 ⇒ 这一格不作数。

    这就是 S32 那条负控制的形状：自称 8k 实际 2k 的格子，数字看起来完全合理。
    """
    outcome = _collect(FakeGateway(in_factor=0.01))
    cell = outcome.cells[0]
    assert cell.status == "skipped", "被裁的正文不能进吞吐聚合"
    assert "in_tokens" in cell.reason and "汉字下限" in cell.reason
    assert "--num-ctx" in cell.reason, "要给出修法，而不只是宣布失败"
    assert cell.metrics["n_measured"] == 0
    assert cell.metrics["n_truncated"] == 2
    assert outcome.status == "partial"


def test_budget_leftovers_are_recorded_as_not_measured():
    """预算用完：剩下的格子必须留下一行"没测到"，而不是消失。"""
    plan = _plan(prompt_chars=(600, 1200, 2400), target_tokens=(64, 256), repeat=1,
                 budget_s=0.12)
    # FakeClock 每次读前进 40ms ⇒ 跑不了几格就会越过预算线，且越线位置完全确定
    outcome = _collect(FakeGateway(), plan, clock=FakeClock(step_ns=40_000_000))
    assert outcome.status == "partial"
    measured = [item for item in outcome.cells if item.status == "measured"]
    missing = outcome.unmeasured
    assert measured and missing, "预算要真的切在中间，否则这条测试什么都没测"
    assert all("预算" in item.reason for item in missing)
    assert sum(len(item.samples) for item in missing) == 0, "没发的格子一条样本都不能有"


def test_cold_start_is_only_claimed_when_the_model_was_really_unloaded():
    gateway = FakeGateway(load_ms=1500.0)
    without = _collect(gateway, _plan(cold=True))
    assert without.cells[0].status == "skipped" and "卸载接口" in without.cells[0].reason

    def boom(_model: str) -> None:
        raise RuntimeError("通道不支持")

    broken = _collect(gateway, _plan(cold=True), unload=boom)
    assert broken.cells[0].status == "skipped" and "通道不支持" in broken.cells[0].reason

    ok = _collect(FakeGateway(load_ms=1500.0), _plan(cold=True), unload=lambda _m: None)
    cold = ok.cells[0]
    assert cold.cell.phase == "cold" and cold.status == "measured"
    assert cold.metrics["load_ms"]["median"] == pytest.approx(1500.0), "载入时间单列，不混进 prefill"


def test_all_requests_failing_is_an_error_not_an_empty_cell():
    outcome = _collect(FakeGateway(fail=True))
    cell = outcome.cells[0]
    assert cell.status == "error" and "引擎 500" in cell.reason
    assert outcome.status == "error"
    assert cell.metrics["n_error"] == 2 and cell.metrics["n_measured"] == 0


def test_ttft_comes_from_the_request_side_because_the_observer_still_drops_it():
    """观测层从来没写 `TraceState.ttft_ms`（真机：库里 2318 条 usage 的 ttft_ms 全是 NULL），
    所以 latency 字典里那一路永远是 None。基线取的是 `generation.ttft_ms` ——原始值，出处更近。

    这条测试同时是那个观测缺陷的**哨兵**：哪天观测层修好了，两路都有值，
    这里仍然过；哪天有人把这一路也改回"只信字典"，红的是它。
    """
    gateway = FakeGateway(ttft_ms=None)
    outcome = _collect(gateway)
    assert outcome.cells[0].metrics["ttft_ms"] is None, "两路都没有才是 None，不是 0"

    class StateBlindGateway(FakeGateway):
        """provider 算出了 TTFT，而聚合视图里没有——就是今天真实的样子。"""

        def generate(self, req, *, purpose=None):
            result = super().generate(req, purpose=purpose)
            return SimpleNamespace(**{**result.__dict__, "latency": {**result.latency,
                                                                    "ttft_ms": None}})

    outcome = _collect(StateBlindGateway(ttft_ms=8.25))
    assert outcome.cells[0].metrics["ttft_ms"]["median"] == pytest.approx(8.25)


def test_partial_failures_stay_countable():
    """一部分失败仍然给出剩下那些的数，但失败数必须露在外面。"""
    flaky = FakeGateway()
    original = flaky.generate
    counter = {"n": 0}

    def mixed(req, *, purpose=None):
        counter["n"] += 1
        if counter["n"] == 1:
            return SimpleNamespace(trace_id="t0",
                                   generation=Generation(status=Status.ERROR, error="偶发 503"),
                                   latency=latency_summary(Generation(status=Status.ERROR)))
        return original(req, purpose=purpose)

    flaky.generate = mixed
    outcome = _collect(flaky)
    cell = outcome.cells[0]
    assert cell.status == "measured" and cell.metrics["n_error"] == 1
    assert "1 发失败" in cell.reason


# ── 出处与指纹 ────────────────────────────────────────────────────
def test_no_engine_timing_degrades_to_wall_only_and_unknown_throughput():
    """兼容通道没有纳秒分段 ⇒ 吞吐列是「—」，而 TTFT 仍然可以是真的。"""
    outcome = _collect(FakeGateway(ns=False))
    assert outcome.conditions["timing_source"] == "wall_only"
    cell = outcome.cells[0]
    assert cell.metrics["decode_tps"] is None
    assert cell.metrics["ttft_ms"]["median"] == pytest.approx(12.5)
    assert fmt_spread(cell.metrics["decode_tps"]) == "—"


def test_same_conditions_same_fingerprint_different_ones_do_not():
    a = _collect(FakeGateway())
    b = _collect(FakeGateway())
    assert a.env_hash == b.env_hash
    c = _collect(FakeGateway(), _plan(model="qwen3:8b"))
    assert c.env_hash != a.env_hash


def test_unrecognizable_engine_is_marked_not_comparable():
    """报不出版本 ⇒ comparable=False。compare 会拒绝，而不是拿着可能张冠李戴的哈希给结论。"""
    gateway = FakeGateway()
    outcome = collect(gateway, _plan(), clock=FakeClock(),
                      engine_info={"provider_id": "openai-compat", "version": ""})
    assert outcome.comparable is False
    assert outcome.conditions["engine_version"] == ""


def test_progress_callback_sees_the_final_request_count():
    seen: list[tuple[int, int, str]] = []
    _collect(FakeGateway(), _plan(prompt_chars=(600, 1200)),
             on_progress=lambda done, total, key: seen.append((done, total, key)))
    assert seen and seen[-1][0] == seen[-1][1] == 4
