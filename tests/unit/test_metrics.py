"""S13 验收：指标层。全部用**手算好的小表格**断言。

刻意不用宽容差：浮点容差会把"逻辑写错了"和"浮点误差"混为一谈，
而指标层的逻辑错误恰恰是最容易被容差掩盖的那类（例如把 F1=None 当成 0 参与宏平均，
差值往往只有零点几，一个 `abs=0.5` 的容差就把它吞了）。
每个期望值旁边都写着它是怎么算出来的。
"""

from __future__ import annotations

import pytest

from onyx.eval.metrics import (
    CI,
    LOW_CONFIDENCE_N,
    _percentile,
    accuracy,
    balanced_accuracy,
    bootstrap_ci,
    confusion,
    hit_at_k,
    macro,
    macro_f1,
    macro_f1_ci,
    mean_ci,
    pass_at_k,
    pass_hat_k,
    per_class_prf1,
    prf1,
    rate,
    stability_gap,
    summarize_pairs,
    top_confusions,
)

#: 4 条样本：A→A, A→A, A→B, B→B
PAIRS = [("A", "A"), ("A", "A"), ("A", "B"), ("B", "B")]


# ── prf1：零除必须是 None 而不是 0 ────────────────────────────────
def test_prf1_hand_computed():
    # tp=2 fp=0 fn=1 → P=2/2=1  R=2/3  F1=2·1·(2/3)/(1+2/3)=4/5
    item = prf1(2, 0, 1)
    assert item.precision == 1.0
    assert item.recall == pytest.approx(2 / 3)
    assert item.f1 == pytest.approx(0.8)
    assert (item.tp, item.fp, item.fn, item.support) == (2, 0, 1, 3)


def test_prf1_undefined_and_zero_are_two_different_things():
    """`未定义` 与 `0 分` 必须分开，而且分界只有一条：P 或 R 自己算不算得出来。

    - `prf1(0,0,0)`：这个类一条都没考到 ⇒ 未定义 ⇒ None，且 `macro` 必须跳过它。
      填 0 会让没考到的类拖低宏平均，看起来像模型能力差。
    - `prf1(0,1,1)`：考到了，而且**全错** ⇒ P=0、R=0 都是算得出来的数 ⇒ F1 必须是 0.0。
      以前这里也返回 None，于是 `macro` 把这个类整个跳过——错得最彻底的类不参与平均，
      模型越差 macro_f1 反而越高。
    """
    empty = prf1(0, 0, 0)
    assert empty.precision is None and empty.recall is None and empty.f1 is None
    assert empty.is_defined is False

    all_wrong = prf1(0, 1, 1)
    assert all_wrong.precision == 0.0 and all_wrong.recall == 0.0
    assert all_wrong.f1 == 0.0, "P=R=0 是「全错」这个事实，不是「没考到」"
    assert all_wrong.is_defined is True, "必须进宏平均，否则最差的类会被静默剔除"

    perfect = prf1(3, 0, 0)
    assert (perfect.precision, perfect.recall, perfect.f1) == (1.0, 1.0, 1.0)


def test_macro_f1_does_not_drop_a_totally_wrong_class():
    """一个全错的类必须把宏平均拉下来，而不是从分母里消失。

    两个类整个互换（A→B、B→A）：每个类都被预测过、也都被漏掉过，
    所以 P=0 与 R=0 **都算得出来**，F1 就是 0。
    按旧口径（P+R=0 ⇒ 未定义）这里会得到"一个类都定义不出来"⇒ macro_f1=None，
    界面上显示「—」（不知道），而真相是"全错"——两者差一个数量级，且方向相反。
    """
    pairs = [("A", "B"), ("B", "A")]
    per_class = per_class_prf1(pairs)
    assert per_class["A"].f1 == 0.0 and per_class["B"].f1 == 0.0
    assert macro_f1(pairs) == 0.0


def test_prf1_support_defaults_to_tp_plus_fn():
    assert prf1(2, 5, 3).support == 5
    assert prf1(2, 5, 3, support=99).support == 99, "显式 support 优先（用于带权宏平均）"


# ── 混淆矩阵 ──────────────────────────────────────────────────────
def test_confusion_matrix_hand_computed():
    matrix = confusion(PAIRS)
    assert matrix == {"A": {"A": 2, "B": 1}, "B": {"A": 0, "B": 1}}


def test_confusion_includes_hallucinated_labels():
    """模型吐出一个标签集里没有的标签时，它必须能在矩阵里被看见。

    只按"期望标签"建行的话，越界标签会被静默丢掉——那正是越界标签率要抓的东西。
    """
    matrix = confusion([("A", "X")])
    assert set(matrix) == {"A", "X"}
    assert matrix["A"] == {"A": 0, "X": 1}
    assert matrix["X"] == {"A": 0, "X": 0}


def test_confusion_on_empty_input():
    assert confusion([]) == {}


# ── 逐类 P/R/F1 与宏平均 ──────────────────────────────────────────
def test_per_class_prf1_hand_computed():
    per_class = per_class_prf1(PAIRS)
    # A: tp=2 fn=1(A→B) fp=0(B→A) → P=1 R=2/3 F1=0.8
    assert per_class["A"].f1 == pytest.approx(0.8)
    assert per_class["A"].precision == 1.0
    # B: tp=1 fn=0 fp=1(A→B) → P=1/2 R=1 F1=2·0.5·1/1.5=2/3
    assert per_class["B"].f1 == pytest.approx(2 / 3)
    assert per_class["B"].recall == 1.0


def test_macro_f1_hand_computed():
    # (0.8 + 2/3) / 2 = 0.7333…
    assert macro_f1(PAIRS) == pytest.approx((0.8 + 2 / 3) / 2)


def test_macro_skips_undefined_classes_instead_of_counting_them_as_zero():
    """这是指标层最容易犯、也最难发现的错。

    A→A, A→B, B→B, X→A（X 是幻觉标签）：
      A: tp=1 fn=1 fp=1 → P=R=F1=1/2
      B: tp=1 fn=0 fp=1 → P=1/2 R=1 F1=2/3
      X: tp=0 fn=1 fp=0 → P 未定义 ⇒ F1 未定义
    宏平均只对 A、B 求：(1/2 + 2/3)/2 = 0.5833…
    若把 X 当成 0：(1/2 + 2/3 + 0)/3 = 0.3889 —— 差 0.19，模型凭空"变差"了。
    """
    pairs = [("A", "A"), ("A", "B"), ("B", "B"), ("X", "A")]
    per_class = per_class_prf1(pairs)
    assert per_class["X"].f1 is None
    assert per_class["A"].f1 == pytest.approx(0.5)
    assert per_class["B"].f1 == pytest.approx(2 / 3)
    assert macro_f1(pairs) == pytest.approx((0.5 + 2 / 3) / 2)
    assert macro_f1(pairs) != pytest.approx((0.5 + 2 / 3 + 0.0) / 3)


def test_macro_f1_is_none_when_no_class_is_defined():
    """全部答成幻觉标签 ⇒ 一个类都定义不出来 ⇒ 未定义，不是 0 分。"""
    assert macro_f1([("A", "B")]) is None
    assert macro_f1([]) is None


def test_macro_accepts_raw_floats_and_skips_none():
    assert macro([1.0, 0.5]) == pytest.approx(0.75)
    assert macro([1.0, None, 0.5]) == pytest.approx(0.75)
    assert macro([None, None]) is None
    assert macro([]) is None


def test_accuracy_and_balanced_accuracy_hand_computed():
    assert accuracy(PAIRS) == pytest.approx(0.75)  # 3/4
    # 各类召回率平均：(2/3 + 1)/2 = 5/6
    assert balanced_accuracy(PAIRS) == pytest.approx(5 / 6)
    assert accuracy([]) is None
    assert balanced_accuracy([]) is None


def test_balanced_accuracy_exposes_class_imbalance_that_accuracy_hides():
    """99 条 A 全对 + 1 条 B 答错：accuracy 0.99，balanced accuracy 0.5。

    只看 accuracy 会以为模型很强；它其实完全没学会 B。
    """
    pairs = [("A", "A")] * 99 + [("B", "A")]
    assert accuracy(pairs) == pytest.approx(0.99)
    assert balanced_accuracy(pairs) == pytest.approx((1.0 + 0.0) / 2)


def test_rate_returns_none_on_empty_denominator():
    assert rate(3, 4) == pytest.approx(0.75)
    assert rate(0, 0) is None


# ── hit@k / pass^k / pass@k ───────────────────────────────────────
def test_hit_at_k():
    ranked = ["a", "b", "c"]
    assert hit_at_k("a", ranked, 1) is True
    assert hit_at_k("b", ranked, 1) is False
    assert hit_at_k("b", ranked, 2) is True
    assert hit_at_k("c", ranked, 3) is True
    assert hit_at_k("a", ranked, 0) is False, "k=0 什么都不看"
    assert hit_at_k("z", ranked, 9) is False


def test_pass_hat_k_and_pass_at_k_hand_computed():
    """pass^k 全对 / pass@k 至少一次对。两者的差就是稳定性缺口。"""
    samples = [
        [True, True, True],      # 稳定对
        [True, False, True],     # 时对时错
        [False, False, False],   # 稳定错
        [True, True, True],      # 稳定对
    ]
    assert pass_hat_k(samples) == pytest.approx(2 / 4)  # 只有第 1、4 组全对
    assert pass_at_k(samples) == pytest.approx(3 / 4)   # 前 3 组里至少一次对
    assert stability_gap(samples) == pytest.approx(0.25)


def test_pass_metrics_on_empty_input():
    assert pass_hat_k([]) is None
    assert pass_at_k([]) is None
    assert stability_gap([]) is None
    # 某个 case 一次都没采到（被 skip 了）时不参与分母
    assert pass_hat_k([[True], []]) == pytest.approx(1.0)


def test_pass_hat_k_never_exceeds_pass_at_k():
    samples = [[True, False], [True, True], [False, False]]
    assert pass_hat_k(samples) <= pass_at_k(samples)


# ── bootstrap CI ──────────────────────────────────────────────────
def test_percentile_hand_computed():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert _percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    # q=0.5 → position=1.5 → 2·0.5 + 3·0.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)
    assert _percentile([7.0], 0.5) == 7.0
    with pytest.raises(ValueError):
        _percentile([], 0.5)


def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic():
    pairs = PAIRS * 25  # n=100
    first = macro_f1_ci(pairs, iterations=400, seed=42)
    second = macro_f1_ci(pairs, iterations=400, seed=42)
    assert first.low == second.low and first.high == second.high, "同种子必须可复现"
    assert first.point == pytest.approx(macro_f1(pairs))
    assert first.low <= first.point <= first.high
    assert first.n == 100 and first.iterations == 400
    assert first.method == "bootstrap"


def test_ci_widens_as_n_shrinks_at_the_same_point_estimate():
    """这条证明区间**真的在算**，不是返回一个常数。

    两个数据集的点估计完全相同（accuracy 都是 0.5、macro_f1 都是 2/3），
    只有样本量差 50 倍。若实现是假的，两个区间会一样宽。
    """
    small = [("A", "A"), ("A", "B")]
    large = small * 50
    assert accuracy(small) == accuracy(large) == pytest.approx(0.5)
    assert macro_f1(small) == macro_f1(large) == pytest.approx(2 / 3)

    narrow = macro_f1_ci(large, iterations=600, seed=7)
    wide = macro_f1_ci(small, iterations=600, seed=7)
    narrow_width = narrow.high - narrow.low
    wide_width = wide.high - wide.low
    assert narrow_width < wide_width, f"{narrow} vs {wide}"
    # n=2 时重采样只有 4 种下标组合，上界直接顶到 1.0：完全无法排除"其实全对"
    assert wide_width > 0.3, f"n=2 的区间本该很宽，实际 {wide}"
    assert wide.high == pytest.approx(1.0)
    assert narrow_width < wide_width / 1.5, "n=100 的区间应显著窄于 n=2"


def test_ci_refuses_to_claim_certainty_from_one_sample():
    """n<=1 时重采样永远得到同一个样本，区间宽度为 0 —— 那是假的确定性。"""
    result = macro_f1_ci([("A", "A")], iterations=200, seed=1)
    assert result.n == 1
    assert result.low is None and result.high is None
    assert result.point is not None, "点估计仍然可以给"

    empty = macro_f1_ci([], iterations=200, seed=1)
    assert empty.point is None and empty.n == 0


def test_bootstrap_resamples_cases_not_precomputed_scores():
    """统计量必须每个 bootstrap 样本重算。

    对 macro_f1 这种非线性统计量，"把算好的分数重排求均值"会得到一个
    看起来合理但错误的区间——所以这里检查 statistic 确实被按重采样下标调用。
    """
    seen: list[tuple[int, ...]] = []

    def statistic(indices: list[int]) -> float:
        seen.append(tuple(indices))
        return float(len(set(indices)))

    result = bootstrap_ci(6, statistic, iterations=50, seed=3)
    assert result.iterations == 50
    assert len(seen) == 51, "1 次点估计 + 50 次重采样"
    assert seen[0] == (0, 1, 2, 3, 4, 5), "点估计必须用全部下标"
    assert any(len(set(item)) < 6 for item in seen[1:]), "重采样必须出现重复下标"
    assert all(max(item) < 6 for item in seen)


def test_bootstrap_skips_none_statistics():
    """某些重采样样本里统计量未定义（例如全被采成同一类）时必须跳过，不许当 0。"""
    calls = {"n": 0}

    def statistic(indices: list[int]) -> float | None:
        calls["n"] += 1
        return None if len(set(indices)) == 1 else 1.0

    result = bootstrap_ci(3, statistic, iterations=40, seed=5)
    assert calls["n"] == 41
    assert result.point is not None
    if result.low is not None:
        assert result.low == pytest.approx(1.0)


def test_mean_ci_ignores_none_and_reports_the_used_n():
    result = mean_ci([1.0, None, 0.0, 1.0, 0.0], iterations=300, seed=11)
    assert result.n == 4, "None 不参与，n 报的是真正参与计算的条数"
    assert result.point == pytest.approx(0.5)
    assert result.low <= 0.5 <= result.high


def test_low_confidence_flag_follows_the_documented_threshold():
    assert LOW_CONFIDENCE_N == 100
    assert CI(low=0.1, high=0.9, point=0.5, n=99).low_confidence is True
    assert CI(low=0.1, high=0.9, point=0.5, n=100).low_confidence is False


def test_ci_format_renders_unknown_as_a_dash_never_as_zero():
    """UI_DESIGN R2：未知显示「—」，不显示 0。"""
    assert CI(low=None, high=None, point=None, n=0).format() == "—"
    assert CI(low=None, high=None, point=0.5, n=1).format() == "0.500"
    text = CI(low=0.4, high=0.6, point=0.5, n=200).format()
    assert text == "0.500 [95% CI 0.400–0.600] (n=200)"
    assert "低样本" in CI(low=0.0, high=1.0, point=0.5, n=4).format()


# ── 混淆对与汇总 ──────────────────────────────────────────────────
def test_top_confusions_names_the_actionable_pair():
    """只看 macro_f1 不知道该改什么；"混淆集中在 A↔B"才是可行动的结论。"""
    pairs = [("A", "B"), ("A", "B"), ("B", "A"), ("A", "A"), ("C", "C")]
    assert top_confusions(pairs) == [("A", "B", 2), ("B", "A", 1)]
    assert top_confusions(pairs, limit=1) == [("A", "B", 2)]
    assert top_confusions([("A", "A")]) == []


def test_summarize_pairs_shape():
    report = summarize_pairs(PAIRS, iterations=200, seed=1)
    assert report["n"] == 4
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert report["balanced_accuracy"] == pytest.approx(5 / 6)
    assert report["labels"] == ["A", "B"]
    assert report["confusion"] == {"A": {"A": 2, "B": 1}, "B": {"A": 0, "B": 1}}
    assert set(report["per_class"]) == {"A", "B"}
    assert isinstance(report["macro_f1_ci"], CI)
    assert report["low_confidence"] is True, "n=4 必须标低置信"


def test_summarize_pairs_flags_low_confidence_only_below_the_threshold():
    assert summarize_pairs(PAIRS * 25, iterations=50)["low_confidence"] is False


# ── JSON 序列化 ───────────────────────────────────────────────────
def test_ci_survives_json_round_trip():
    """`CI` 走 `json.dumps(default=str)` 会变成 `"CI(low=0.97, …)"` 这样一个字符串：
    写进去不报错，读出来既取不到下界也取不到上界，置信区间就静默消失了。"""
    import json as _json

    from onyx.eval.metrics import jsonable
    from onyx.store.codec import dumps, loads_dict

    ci = macro_f1_ci(PAIRS * 25, iterations=200, seed=3)
    report = summarize_pairs(PAIRS * 25, iterations=200, seed=3)

    # 落库路径（codec.dumps 用 default=str）
    restored = loads_dict(dumps(jsonable(report)))
    assert isinstance(restored["macro_f1_ci"], dict), "CI 必须落成对象，不是字符串"
    assert restored["macro_f1_ci"]["low"] == pytest.approx(ci.low)
    assert restored["macro_f1_ci"]["high"] == pytest.approx(ci.high)
    assert restored["macro_f1_ci"]["low_confidence"] is ci.low_confidence

    # --json 输出路径
    text = _json.dumps(jsonable(report), ensure_ascii=False, default=str)
    assert "CI(low=" not in text, "dataclass 的 repr 泄漏进了 JSON"
    assert _json.loads(text)["macro_f1_ci"]["point"] == pytest.approx(ci.point)


def test_jsonable_converts_nested_containers_and_enums():
    from onyx.eval.metrics import jsonable
    from onyx.eval.task import Verdict

    payload = {
        "ci": CI(low=0.1, high=0.9, point=0.5, n=10),
        "verdict": Verdict.OUT_OF_LABEL,
        "labels": {"转账", "查余额"},
        "nested": [{"verdict": Verdict.CORRECT, "tuple": (1, 2)}],
        "none": None,
        "number": 1.5,
    }
    out = jsonable(payload)
    assert out["ci"] == {"low": 0.1, "high": 0.9, "point": 0.5, "iterations": 0,
                         "n": 10, "method": "bootstrap", "low_confidence": True}
    assert out["verdict"] == "out_of_label"
    assert sorted(out["labels"]) == ["查余额", "转账"]
    assert out["nested"] == [{"verdict": "correct", "tuple": [1, 2]}]
    assert out["none"] is None and out["number"] == 1.5

    import json as _json

    assert _json.loads(_json.dumps(out, ensure_ascii=False)) == out


def test_zero_width_ci_on_a_small_perfect_sample_is_flagged_not_trusted():
    """20 条全对时每次重采样仍然全对，区间就是 [1.0, 1.0]。

    这是**退化的 bootstrap**，不是"置信度 100%"。所以 low_confidence 必须为 True，
    UI 也不许把零宽区间渲染成"没有不确定性"。
    """
    perfect = [("A", "A")] * 20
    ci = macro_f1_ci(perfect, iterations=300, seed=5)
    assert ci.point == pytest.approx(1.0)
    assert ci.low == pytest.approx(1.0) and ci.high == pytest.approx(1.0)
    assert ci.high - ci.low == pytest.approx(0.0)
    assert ci.low_confidence is True, "零宽 + 小样本必须标低置信"
    assert "⚠低样本" in ci.format()

    # 同样的满分在 200 条上就该给出一个非退化区间
    wide = macro_f1_ci([("A", "A")] * 199 + [("A", "B")], iterations=300, seed=5)
    assert wide.high - wide.low > 0
    assert wide.low_confidence is False
