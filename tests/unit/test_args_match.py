"""S14 验收：类型感知的参数比对。

计划要求"手算用例表（日期/数值容差/枚举/集合各一）"。这里每一类都单独成节，
并且**同时断言判定结果与判定方式（kind）**：只断言 True 的话，
"靠容差蒙对"和"本来就相等"就分不开了，而这个区别决定了分数有多可信。
"""

from __future__ import annotations

from datetime import date

import pytest

from onyx.eval.graders.args_match import (
    ArgMatch,
    FieldMatch,
    match_args,
    normalize_date,
    summarize,
)


def _kinds(result: ArgMatch) -> dict[str, str]:
    return {item.field: item.kind for item in result.fields}


# ── 日期归一 ──────────────────────────────────────────────────────
@pytest.mark.parametrize("left,right", [
    ("2026-10-03", "2026/10/3"),
    ("2026-10-03", "2026.10.03"),
    ("2026-10-03", "20261003"),
    ("2026-10-03", "2026年10月3日"),
    ("2026-10-03", " 2026-10-03 "),
    ("2026-10", "2026-10-01"),
    ("2026年10月", "2026-10-01"),
    ("2026", "2026-01-01"),
])
def test_date_forms_normalize_to_the_same_value(left, right):
    assert normalize_date(left) == normalize_date(right), f"{left} vs {right}"


def test_normalize_date_hand_computed():
    assert normalize_date("2026-10-03") == date(2026, 10, 3)
    assert normalize_date("2026年3月5日") == date(2026, 3, 5)
    assert normalize_date("20261231") == date(2026, 12, 31)
    # 归一不了返回 None，不是某个默认日期——默认值会把"没识别出来"伪装成"识别成了今天"
    assert normalize_date("下周三") is None
    assert normalize_date("2026-13-45") is None
    assert normalize_date("") is None
    assert normalize_date(None) is None
    assert normalize_date(20261003) is None, "整数不是日期字符串，不该被当成 YYYYMMDD"


def test_date_match_reports_its_kind():
    result = match_args({"date": "2026-10-03"}, {"date": "2026年10月3日"})
    assert result.ok
    assert _kinds(result) == {"date": "date_normalized"}
    assert "2026-10-03" in result.fields[0].detail


def test_different_dates_do_not_match():
    result = match_args({"date": "2026-10-03"}, {"date": "2026-10-04"})
    assert not result.ok
    assert _kinds(result) == {"date": "value_mismatch"}


# ── 数值容差 ──────────────────────────────────────────────────────
def test_numeric_tolerance_hand_computed():
    exact = match_args({"amount": 3.14}, {"amount": 3.14})
    assert exact.ok and exact.fields[0].kind == "identical"

    within = match_args({"amount": 3.14}, {"amount": 3.1400000001})
    assert within.ok
    # 值不完全相等但在容差内 ⇒ 必须标 numeric_tolerance，不能冒充 identical
    assert within.fields[0].kind == "numeric_tolerance"

    loose = match_args({"amount": 100}, {"amount": 100.5}, abs_tol=1.0)
    assert loose.ok and loose.fields[0].kind == "numeric_tolerance"

    tight = match_args({"amount": 100}, {"amount": 100.5}, abs_tol=0.1)
    assert not tight.ok and tight.fields[0].kind == "value_mismatch"

    relative = match_args({"amount": 1000}, {"amount": 1010}, rel_tol=0.02)
    assert relative.ok and relative.fields[0].kind == "numeric_tolerance"


def test_numeric_string_is_accepted_and_labelled():
    """一边是数字一边是数字字符串：语义相同，但这是放宽，必须标出来。"""
    result = match_args({"amount": 500}, {"amount": "500"})
    assert result.ok
    assert result.fields[0].kind == "numeric_tolerance"
    assert "numeric_tolerance" in result.relaxed_kinds


def test_currency_words_are_not_silently_stripped():
    """`"500元"` 与 `500` **不**判等。

    剥掉单位词是一条滑坡：`元` 可以不改变量级，但 `万`/`千` 会（500万 ≠ 500），
    而一旦开始剥就得分清哪些能剥哪些不能——这个判断不该藏在比对器里。
    要接受带单位的写法，应该在工具 schema 里声明成 string 并写明格式，
    让模型和执行器看到同一套规则。
    """
    result = match_args({"amount": 500}, {"amount": "500元"})
    assert not result.ok
    assert result.fields[0].kind == "value_mismatch"
    assert "数值不等" in result.fields[0].detail


def test_bool_is_never_treated_as_a_number():
    """Python 里 True == 1，先判数值就会把 True 当成 1 放过。"""
    assert not match_args({"flag": True}, {"flag": 1}).ok
    assert not match_args({"flag": 1}, {"flag": True}).ok
    assert match_args({"flag": True}, {"flag": True}).ok
    assert _kinds(match_args({"flag": True}, {"flag": 1})) == {"flag": "type_mismatch"}


# ── 枚举 ──────────────────────────────────────────────────────────
ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        "city": {"type": "string"},
    },
}


def test_enum_match_is_case_insensitive_and_labelled():
    result = match_args({"unit": "celsius"}, {"unit": "Celsius"}, schema=ENUM_SCHEMA)
    assert result.ok and result.fields[0].kind == "enum"


def test_enum_violation_is_reported_with_the_allowed_set():
    result = match_args({"unit": "celsius"}, {"unit": "kelvin"}, schema=ENUM_SCHEMA)
    assert not result.ok
    assert result.fields[0].kind == "value_mismatch"
    assert "枚举越界" in result.fields[0].detail


def test_a_bad_expectation_in_an_enum_field_is_surfaced():
    """期望值本身不在 enum 里 ⇒ 用例写错了。这必须显式报出来，
    否则"模型答对了"与"用例是坏的"会混成同一个结果。"""
    result = match_args({"unit": "kelvin"}, {"unit": "kelvin"}, schema=ENUM_SCHEMA)
    assert not result.ok
    assert "不在 schema 的 enum" in result.fields[0].detail


# ── 集合语义 ──────────────────────────────────────────────────────
UNORDERED_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "uniqueItems": True, "items": {"type": "string"}},
        "steps": {"type": "array", "items": {"type": "string"}},
    },
}


def test_unique_items_array_uses_set_semantics():
    result = match_args({"tags": ["a", "b"]}, {"tags": ["b", "a"]}, schema=UNORDERED_SCHEMA)
    assert result.ok and result.fields[0].kind == "set_equal"

    bad = match_args({"tags": ["a", "b"]}, {"tags": ["a", "c"]}, schema=UNORDERED_SCHEMA)
    assert not bad.ok
    assert "缺 ['b'] 多 ['c']" in bad.fields[0].detail


def test_ordered_array_keeps_order_semantics():
    """没有 uniqueItems 的数组顺序有意义（例如多步操作的步骤列表），不能放宽。"""
    result = match_args({"steps": ["开", "关"]}, {"steps": ["关", "开"]}, schema=UNORDERED_SCHEMA)
    assert not result.ok
    assert "第 0 项" in result.fields[0].detail


def test_explicit_unordered_fields_override_the_schema():
    result = match_args({"tags": ["a", "b"]}, {"tags": ["b", "a"]}, unordered_fields=["tags"])
    assert result.ok and result.fields[0].kind == "set_equal"


def test_unordered_array_reports_what_is_missing_not_just_the_length():
    """集合语义下"长度不同"不是重点，**缺了哪个**才是可行动的信息。"""
    result = match_args({"tags": ["a", "b", "c"]}, {"tags": ["a", "b"]},
                        schema=UNORDERED_SCHEMA)
    assert not result.ok
    assert "缺 ['c']" in result.fields[0].detail


def test_ordered_array_reports_the_length_difference():
    result = match_args({"steps": ["a", "b", "c"]}, {"steps": ["a", "b"]},
                        schema=UNORDERED_SCHEMA)
    assert not result.ok
    assert "长度不同：3 vs 2" in result.fields[0].detail


# ── 缺字段 / 多字段 / 嵌套 ────────────────────────────────────────
def test_missing_and_unexpected_are_distinct_kinds():
    result = match_args({"city": "北京", "unit": "c"}, {"city": "北京", "extra": 1})
    assert _kinds(result) == {"city": "identical", "unit": "missing", "extra": "unexpected"}
    assert result.missing == ("unit",)
    assert result.unexpected == ("extra",)
    assert result.ok is False, "缺字段不算完全匹配"
    assert result.subset_ok is False


def test_extra_fields_pass_by_default_but_fail_under_exact():
    """模型多给一个字段通常比少给一个更有用，所以默认只要求期望字段都在且值对。"""
    result = match_args({"city": "北京"}, {"city": "北京", "unit": "c"})
    assert result.subset_ok is True
    assert result.ok is True, "非严格档下多余字段不算错"

    strict = match_args({"city": "北京"}, {"city": "北京", "unit": "c"}, exact=True)
    assert strict.ok is False
    assert strict.unexpected == ("unit",)


def test_nested_object_is_compared_recursively():
    result = match_args(
        {"where": {"city": "北京", "zip": "100000"}},
        {"where": {"city": "北京", "zip": "100000", "extra": 1}},
    )
    assert not result.ok, "嵌套默认按严格档比：多一个字段就是不一致"
    assert "嵌套不匹配" in result.fields[0].detail

    same = match_args({"where": {"city": "北京"}}, {"where": {"city": "北京"}})
    assert same.ok and same.fields[0].kind == "identical"


def test_type_mismatch_between_object_and_scalar():
    result = match_args({"where": {"city": "北京"}}, {"where": "北京"})
    assert not result.ok and result.fields[0].kind == "type_mismatch"


# ── 字符串 ────────────────────────────────────────────────────────
def test_presentation_differences_are_normalized_not_forgiven():
    for actual in ("北京", " 北京 ", "「北京」", '"北京"'):
        result = match_args({"city": "北京"}, {"city": actual})
        assert result.ok, actual
    assert _kinds(match_args({"city": "北京"}, {"city": " 北京 "})) == {"city": "normalized"}


def test_similar_but_different_entities_do_not_match():
    """`北京` 与 `北京市` 差一个字，但模糊匹配会给 0.86 的高分。

    默认**不开**模糊：地名/人名/ID 这类字段差一点就是另一个实体，
    放宽等于把错误答案判成对的。要用必须显式列进 fuzzy_fields。
    """
    result = match_args({"city": "北京"}, {"city": "北京市"})
    assert not result.ok and result.fields[0].kind == "value_mismatch"

    fuzzy = match_args({"city": "北京"}, {"city": "北京市"}, fuzzy_fields=["city"],
                       fuzz_threshold=0.8)
    assert fuzzy.ok and fuzzy.fields[0].kind == "fuzzy"
    assert "模糊匹配" in fuzzy.fields[0].detail


def test_fuzzy_below_threshold_still_fails():
    result = match_args({"q": "如何重置密码"}, {"q": "怎么查余额"},
                        fuzzy_fields=["q"], fuzz_threshold=0.9)
    assert not result.ok
    assert result.fields[0].kind == "value_mismatch"


def test_null_handling():
    assert match_args({"a": None}, {"a": None}).ok
    assert not match_args({"a": None}, {"a": 1}).ok
    assert not match_args({"a": 1}, {"a": None}).ok


def test_empty_expectations():
    """无参工具：期望 {} 而模型给了 {} ⇒ 匹配，score 为 None（没有字段可比）。"""
    result = match_args({}, {})
    assert result.ok and result.subset_ok
    assert result.score is None
    assert result.fields == ()


def test_score_is_the_matched_share_of_expected_fields():
    result = match_args({"a": 1, "b": 2, "c": 3}, {"a": 1, "b": 9, "extra": 0})
    # 期望 3 个字段：a 对、b 错、c 缺 ⇒ 1/3
    assert result.score == pytest.approx(1 / 3)
    assert result.matched == ("a",)
    assert result.missing == ("c",) and result.unexpected == ("extra",)


# ── 汇总 ──────────────────────────────────────────────────────────
def test_summarize_reports_the_relaxed_share():
    """靠归一化/容差/模糊通过的字段占比。

    这个数越高，说明分数越依赖比对器的宽容度而不是模型能力——
    必须与 exact_rate 一起看，只报后者会把"我们放宽了"藏起来。
    """
    matches = [
        match_args({"city": "北京"}, {"city": "北京"}),                    # identical
        match_args({"city": "北京"}, {"city": " 北京 "}),                  # normalized
        match_args({"amount": 500}, {"amount": "500"}),                    # numeric_tolerance
        match_args({"date": "2026-10-03"}, {"date": "2026年10月3日"}),      # date_normalized
        match_args({"city": "北京"}, {"city": "上海"}),                    # 错
    ]
    report = summarize(matches)
    assert report["n"] == 5
    # exact 的含义是"字面就一致"：只有第 1 条算
    assert report["exact_rate"] == pytest.approx(1 / 5)
    # semantic 的含义是"语义上对"：靠归一化/容差/日期放宽挣来的也算
    assert report["semantic_rate"] == pytest.approx(4 / 5)
    assert report["subset_rate"] == pytest.approx(4 / 5)
    assert report["field_total"] == 5 and report["field_matched"] == 4
    assert report["relaxed_fields"] == 3, "normalized + numeric_tolerance + date_normalized"
    assert report["relaxed_share"] == pytest.approx(3 / 4)
    assert report["kinds"]["identical"] == 1
    assert report["kinds"]["value_mismatch"] == 1
    # 三个口径必须能对上：字段级 matched - relaxed = 字面就一致的那一个
    assert report["field_matched"] - report["relaxed_fields"] == 1


def test_summarize_on_empty_input():
    report = summarize([])
    assert report["n"] == 0
    assert report["exact_rate"] is None and report["field_rate"] is None
    assert report["relaxed_share"] is None


def test_field_match_and_arg_match_are_value_objects():
    item = FieldMatch("city", True, "identical", "北京", "北京")
    assert item.field == "city" and item.ok and item.kind == "identical"
    assert ArgMatch(fields=(item,)).matched == ("city",)
    assert ArgMatch(fields=(item,)).relaxed_kinds == ()
