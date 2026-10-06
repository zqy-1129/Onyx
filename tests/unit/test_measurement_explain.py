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
    # S39 之后"不闭合"这三个字在 CLI 那一行（`_closure_line` ⇒ `✗ 不闭合｜…`，由
    # test_cli_token_report.py 守着）；这一层负责的是**说出原因**，不是复述判定。
    assert result.advice and any("归因" in line for line in result.advice)
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


# ── S39：归因档位与 clamp（"不闭合"的唯一成因）─────────────────────
_CLAMP = {"count_source": "heuristic", "clamped": True, "residual_raw": -300,
          "input_segments_tokens": 1300, "template_ctl_tokens": 0, "has_template": False}

_UNCALI = _bundle(_usage(in_tokens=1000), [],
                  [_part("system", 100), _part("messages", 1200), _part("template_ctl", 0)])


def test_a_clamped_row_names_the_tier_and_carries_the_residual():
    """真机形状（bench `01M47VS1…`）：分段按 heuristic 数 ⇒ 621 vs 475 ⇒ 残差 -146 被 clamp。"""
    result = explain(_UNCALI, attribution=_CLAMP, fitted_n=0, model="qwen3.5:9b")
    assert result.closure["count_source"] == "heuristic"
    assert result.closure["clamped"] is True and result.closure["residual_raw"] == -300
    advice = "\n".join(result.advice)
    assert "clamp" in advice and "启发式" in advice
    assert "onyx calibrate --model qwen3.5:9b" in advice, "未标定 ⇒ 给出那一条命令，不是空喊缺陷"
    assert "相对占比" in advice, "要说明这一条的分段现在能用来比什么、不能用来当什么"


def test_a_model_that_is_already_calibrated_is_not_told_to_calibrate_again():
    """档位是**当时**的事实，标定状态是**现在**的——两者不一致时必须说清，否则给的是错建议。

    这条对应真机上"标定之后重看老 trace"的那一次：模型档案 n=40，而老行仍写着 heuristic。
    """
    result = explain(_UNCALI, attribution=_CLAMP, fitted_n=FITTED_MIN_SAMPLES, model="qwen3.5:9b")
    advice = "\n".join(result.advice)
    assert "calibrate --model" not in advice, "已经标定过了，再叫人生跑一次标定是把人支走"
    assert "标定前跑的" in advice and "重跑" in advice
    assert "不回填" in advice, "历史行不重算——那等于伪造当时的测量"


def test_unclosed_but_not_clamped_is_reported_as_inconsistent_records():
    """`template_ctl` 就是残差，没被 clamp 时闭合**由构造保证**。

    所以"不等且没 clamp"只能是记录被第三方改过——把它写成"计数器高估"就是编一个不存在的解释。
    """
    result = explain(_UNCALI, attribution={**_CLAMP, "clamped": False, "residual_raw": 10},
                     fitted_n=FITTED_MIN_SAMPLES, model="m")
    advice = "\n".join(result.advice)
    assert "不自洽" in advice and "clamp" in advice
    assert "calibrate" not in advice, "这种情况让人去标定是错的方向"


def test_a_legacy_row_without_attribution_admits_it_cannot_explain():
    """没记档位就说"说不出原因"。硬编一个"高估"是把猜测写成结论。"""
    result = explain(_UNCALI, attribution=None, fitted_n=0, model="m")
    assert result.attribution["recorded"] is False
    assert result.closure["count_source"] is None and result.closure["clamped"] is None
    assert "没记归因档位" in "\n".join(result.advice)


def test_closure_closes_by_construction_when_template_ctl_is_the_residual():
    """未 clamp 时 `Σ非output分段 + template_ctl == 引擎计数` 是恒等式 ⇒ 闭合不该被当成"验证通过"。

    它只说明两边是同一次计算；真正有意义的是档位（fitted/tokenizer 才谈得上归因可信）。
    """
    result = explain(_bundle(_usage(in_tokens=1000), [],
                             [_part("system", 100), _part("messages", 800),
                              _part("template_ctl", 100)]),
                     attribution={"count_source": "fitted", "clamped": False, "residual_raw": 100})
    assert result.closure["closed"] is True and result.closure["count_source"] == "fitted"
    assert result.attribution["recorded"] is True


def test_a_calibrated_tier_that_still_clamps_points_at_the_engine_not_at_calibration():
    """标定过了还被 clamp ⇒ 该查引擎有没有裁正文，而不是再标一次（第三条分支）。"""
    result = explain(_UNCALI, attribution={**_CLAMP, "count_source": "fitted"},
                     fitted_n=FITTED_MIN_SAMPLES, model="qwen3.5:9b")
    advice = "\n".join(result.advice)
    assert "已标定" in advice and "裁过正文" in advice
    assert "calibrate --model" not in advice


def test_a_trace_without_a_chosen_count_cannot_be_delta_ed_or_judged():
    """采信总数缺失（引擎没报）⇒ 差值表与闭合判定都必须**空着并说明**，不是 0。"""
    result = explain(_bundle(_usage(in_tokens=None), [_alt("heuristic", 621, 24)],
                             [_part("msg:0", 621)]), attribution=_CLAMP)
    assert result.deltas == ()
    assert result.closure["checked"] is False and result.closure["closed"] is None
    assert "判不了" in result.closure["note"] and result.closure["sum"] == 621
    assert result.clean, "判不了不算脏——但它也不能被读成通过"


def test_no_advice_line_is_ever_a_single_character():
    """`out.extend(_closure_advice(...))` 配的是**元组**：分支返回裸字符串时
    extend 会把它按字符展开，终端上就成了"一个字一行"（S39 第一次真机跑就撞到了）。

    这条断言不测文案内容，只测形状——所以它对以后新增的任何分支都有效。
    """
    for attribution in (None, _CLAMP, {**_CLAMP, "clamped": False}):
        for fitted_n in (0, FITTED_MIN_SAMPLES):
            result = explain(_UNCALI, attribution=attribution, fitted_n=fitted_n, model="qwen3.5:9b")
            assert isinstance(result.attribution, dict)
            for line in result.advice:
                assert len(line) > 12, f"被展开成了字符：{line!r}"
                assert "\n" not in line
