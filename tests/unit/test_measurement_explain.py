"""`token explain` 背后的判定（S38）。

这个模块存在的意义是把"分段闭合吗"从人眼加法变成一个判定，
所以它的测试必须包含**不闭合时会不会响**——否则它只是个装饰。

真机第一次跑就抓到了不闭合的实例（未标定模型的启发式归因比引擎计数多约 31%），
所以这里的数字不是编的形状，是把那次观察固化成回归保护。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from onyx.llm.measurement.explain import UNIMPLEMENTED, explain
from onyx.llm.measurement.fidelity import FITTED_MIN_SAMPLES
from onyx.llm.measurement.reconciler import DEFAULT_DRIFT_THRESHOLD


def _alt(source: str, in_tokens: int | None, out_tokens: int | None, *, ok: bool = True,
         note: str = ""):
    return SimpleNamespace(source=source, in_tokens=in_tokens, out_tokens=out_tokens,
                           thinking_tokens=None, cached_tokens=None, ok=ok, note=note)


def _bundle(usage, alts=(), parts=()):
    return SimpleNamespace(usage=usage, alts=tuple(alts), parts=tuple(parts))


def _usage(source="engine", confidence="high", in_tokens=1000, out_tokens=50,
           drift_pct=None, trace_id="T1"):
    return SimpleNamespace(trace_id=trace_id, source=source, confidence=confidence,
                           in_tokens=in_tokens, out_tokens=out_tokens, drift_pct=drift_pct)


def _part(part: str, tokens: int):
    return SimpleNamespace(part=part, tokens=tokens)


# ── 阶梯状态 ──────────────────────────────────────────────────────
def test_chosen_tier_is_marked_and_unimplemented_tiers_are_named():
    result = explain(_bundle(_usage(), [_alt("heuristic", 1300, 40)]))
    by_source = {row.source: row for row in result.tiers}
    assert by_source["engine"].status == "采信"
    for source in UNIMPLEMENTED:
        assert by_source[source].status == "未实现", f"{source} 不许被当成「可用但没数」"
    assert by_source["heuristic"].status == "有数（未采信）"


def test_compat_is_labeled_cross_check_and_never_chosen():
    """P14：`/v1` 与原生通道对同一份 prompt 计数不同。它必须被标成交叉验证。"""
    result = explain(_bundle(_usage(source="engine"), [_alt("compat", 1016, 66)]))
    compat = next(r for r in result.tiers if r.source == "compat")
    assert compat.status == "交叉验证" and "P14" in compat.note


def test_fitted_tier_reports_how_many_samples_are_still_missing():
    """标定状态属于模型，但它决定 fitted 档能不能用——所以必须在这一行里看得见。"""
    result = explain(_bundle(_usage(source="heuristic")), fitted_n=5, model="qwen3.5:9b")
    fitted = next(r for r in result.tiers if r.source == "fitted")
    assert fitted.status == "样本不足"
    assert f"{5}/{FITTED_MIN_SAMPLES}" in fitted.note
    assert any("onyx calibrate" in line for line in result.advice), "要给出路，不只是宣布不够"


def test_a_broken_source_says_why_instead_of_looking_like_zero():
    """坏掉的档要说"为什么没数"。留空 + 一句 note 才能和"这一档没参与"区分开。"""
    result = explain(_bundle(_usage(), [_alt("heuristic", None, None, ok=False,
                                             note="tokenizer 未装")]))
    row = next(r for r in result.tiers if r.source == "heuristic")
    assert row.status == "没报数" and row.in_tokens is None
    assert "tokenizer 未装" in row.note


# ── 差值 ─────────────────────────────────────────────────────────
def test_deltas_are_signed_against_the_chosen_value():
    result = explain(_bundle(_usage(in_tokens=1000, out_tokens=50), [_alt("heuristic", 1310, 44)]))
    delta = next(d for d in result.deltas if d["source"] == "heuristic")
    assert delta["in_delta"] == 310 and delta["out_delta"] == -6
    assert delta["in_pct"] == pytest.approx(31.0)


# ── 闭合判定（这一步的核心）──────────────────────────────────────
def test_closure_closes_when_parts_sum_equals_engine_count():
    bundle = _bundle(_usage(in_tokens=1000), [],
                     [_part("system", 100), _part("messages", 800),
                      _part("template_ctl", 100), _part("output", 50)])
    result = explain(bundle)
    assert result.closure["checked"] and result.closure["closed"] is True
    assert result.clean, "闭合 + 无漂移 ⇒ 干净"


def test_closure_shouts_when_attribution_drifts_apart():
    """注入缺陷自检：把某个 part 加大 ⇒ 必须不闭合、报出差值、退出码依据翻转。

    真机第一次跑 `token explain` 就是这个形状：未标定模型的启发式归因比引擎计数多约 31%
    （库里 `ATTRIBUTION_CLAMPED` 也报了同一件事）。那时只有 `traces show` 的表标题在
    宣称"Σ分段 = 引擎计数"，没有任何一处做判定。
    """
    bundle = _bundle(_usage(in_tokens=1000), [],
                     [_part("system", 100), _part("messages", 1100), _part("template_ctl", 100)])
    result = explain(bundle)
    assert result.closure["closed"] is False
    assert result.closure["delta"] == 300
    assert any("不闭合" in line for line in result.advice)
    assert result.clean is False


def test_missing_attribution_is_unknown_not_closed():
    """没有分段就说"判不了"。写成闭合等于把"没测"说成"通过"。"""
    result = explain(_bundle(_usage()))
    assert result.closure["checked"] is False and result.closure["closed"] is None
    assert result.clean, "未知不算脏，但也不算过——输出里会显式写「未判定」"


def test_drift_above_the_shared_threshold_makes_it_unclean():
    result = explain(_bundle(_usage(drift_pct=DEFAULT_DRIFT_THRESHOLD + 0.05),
                             [_alt("heuristic", 1300, 50)]))
    assert result.clean is False
    assert any("阈值" in line for line in result.advice)


def test_the_thresholds_reported_are_the_same_constants_the_reconciler_uses():
    """CLI/看板/异常判据必须共用一个阈值；这里断言的是"同一个对象"，不是同一个字面量。"""
    result = explain(_bundle(_usage()))
    assert result.as_dict()["thresholds"]["drift_pct"] is DEFAULT_DRIFT_THRESHOLD
    assert result.as_dict()["thresholds"]["fitted_min_samples"] == FITTED_MIN_SAMPLES
