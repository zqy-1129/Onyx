"""S13 验收：评分器。

重点不是"能匹配"，而是**失败形态可区分**：
"没抽到答案"不能是空串（会与"期望值就是空串"混淆，把没答对判成答对），
"jsonschema 没装"不能报通过（那是未知），
"文本里出现两个候选标签"不能挑第一个（分数会凭空变高）。
"""

from __future__ import annotations

import pytest

from onyx.eval.graders import (
    check_schema,
    contains_all,
    contains_none,
    exact,
    field_em,
    find,
    first_of,
    fuzzy_match,
    match_label,
    normalize_text,
    numeric,
    parse_json,
    ratio,
    set_match,
)
from onyx.eval.graders.normalize import LabelMatch, normalize_number, strip_code_fence, trim_wrappers

LABELS = ["转账", "查余额", "投诉", "其他"]


# ── 归一化 ────────────────────────────────────────────────────────
def test_strip_code_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence("```\nplain\n```") == "plain"
    assert strip_code_fence("  no fence  ") == "no fence"
    # 只有整体被包裹时才剥；正文里的围栏不该动
    assert strip_code_fence("结果：```x```") == "结果：```x```"


def test_normalize_text_handles_fullwidth_and_whitespace():
    assert normalize_text("　ＡＢＣ　") == "abc"
    assert normalize_text("a\n\n  b\tc") == "a b c"
    assert normalize_text("ABC", casefold=False) == "ABC"


def test_trim_wrappers():
    assert trim_wrappers("「转账」") == "转账"
    assert trim_wrappers('"转账"') == "转账"
    assert trim_wrappers("（转账）") == "转账"
    assert trim_wrappers("转账。") == "转账"


def test_normalize_number():
    assert normalize_number("1,234.5") == 1234.5
    assert normalize_number(" ￥12 ") == 12.0
    assert normalize_number("3.5%") == 3.5
    assert normalize_number(7) == 7.0
    # 归一不了返回 None，不是 0：0 是一个有意义的数值
    assert normalize_number("abc") is None
    assert normalize_number("") is None
    assert normalize_number(None) is None
    assert normalize_number(True) is None, "bool 不是数值，否则 True 会被当成 1"


# ── exact ─────────────────────────────────────────────────────────
def test_exact_tolerates_presentation_differences_only():
    assert exact("转账", "转账")
    assert exact("转账", "「转账」")
    assert exact("转账", "  转账 \n")
    assert exact("TRANSFER", "transfer")
    assert exact("转账", "＼ｔ转账".replace("＼ｔ", ""))
    assert not exact("转账", "查余额")
    assert not exact("转账", "转账操作")


def test_exact_does_not_confuse_bool_with_int():
    """bool 是 int 的子类：True == 1 会判对，那是彻底的误判。"""
    assert not exact(True, 1)
    assert not exact(1, True)
    assert exact(True, True)
    assert not exact(True, False)


def test_exact_treats_none_as_its_own_value():
    assert exact(None, None)
    assert not exact(None, "")
    assert not exact("", None)


def test_numeric_with_tolerance():
    assert numeric("3.14", 3.14)
    assert numeric(3.14, 3.1400000001, tolerance=1e-6)
    assert not numeric(3.14, 3.15, tolerance=1e-6)
    assert numeric(100, 101, rel=0.02), "相对容差"
    # 非数字返回 False 而不是抛异常：评测里这是正常的失败形态
    assert numeric(1, "abc") is False
    assert numeric("abc", 1) is False


# ── 标签抽取 ──────────────────────────────────────────────────────
def test_match_label_exact_and_normalized_paths():
    assert match_label("转账", LABELS) == LabelMatch("转账", "exact", ("转账",), "转账")
    assert match_label("「转账」", LABELS).status == "exact"
    assert match_label("  查余额 \n", LABELS).status == "exact"
    assert match_label("其他", LABELS).label == "其他"


def test_match_label_handles_extra_explanation_text():
    """本地模型很少只吐标签。加了唯一可辨识的标签时仍能判，但 status 要如实说明。"""
    result = match_label("这个意图是转账。", LABELS)
    assert result.label == "转账"
    assert result.status == "unique_substring"
    assert result.usable is True


def test_match_label_refuses_to_guess_when_ambiguous():
    """同时出现多个候选标签时**不可判定**。

    挑第一个会把"不是转账，是查余额"判成"转账"正确，分数凭空变高，
    而且高得看不出来——这是评分器最危险的一种失败。
    """
    result = match_label("不是转账，应该是查余额", LABELS)
    assert result.status == "ambiguous"
    assert result.label is None
    assert result.usable is False
    assert set(result.found) == {"转账", "查余额"}


def test_match_label_out_of_set_and_empty():
    assert match_label("退款", LABELS).status == "not_found"
    assert match_label("退款", LABELS).label is None
    assert match_label("", LABELS).status == "empty"
    assert match_label(None, LABELS).status == "empty"
    assert match_label("   \n ", LABELS).status == "empty"


def test_match_label_is_case_insensitive_for_ascii_labels():
    labels = ["TRANSFER", "BALANCE"]
    assert match_label("transfer", labels).label == "TRANSFER"
    assert match_label("Balance", labels).label == "BALANCE"


# ── regex ─────────────────────────────────────────────────────────
def test_regex_find_takes_the_last_match():
    """答案通常在结尾，开头往往是模型在复述题目。"""
    text = "题目问 12 个苹果。答案是 42。"
    assert find(r"(\d+)", text).value == "42"
    assert find(r"(\d+)", text).occurrences == 2


def test_regex_missing_is_none_not_empty_string():
    """空串会与"期望值是空串"混淆，把没抽到判成抽到了。"""
    hit = find(r"答案：(\S+)", "完全没有答案这个词")
    assert hit.matched is False
    assert hit.value is None
    assert hit.error == ""


def test_regex_reports_a_broken_pattern_without_raising():
    """评测跑到一半因为一个正则崩掉，比这条样本判错严重得多。"""
    hit = find("([unclosed", "任何文本")
    assert hit.value is None and hit.matched is False
    assert "编译失败" in hit.error


def test_regex_reports_an_out_of_range_group():
    hit = find(r"(a)(b)", "ab", group=5)
    assert hit.value is None
    assert "捕获组 5" in hit.error
    assert hit.occurrences == 1, "模式是匹配上的，问题在组号"


def test_regex_number_helper():
    assert find(r"答案是\s*([\d,\.]+)", "答案是 1,234.5").number == 1234.5
    assert find(r"(\w+)", "答案：四十二").number is None


def test_regex_strips_code_fence_before_matching():
    text = '```json\n{"city": "北京"}\n```'
    assert find(r'"city":\s*"([^"]+)"', text).value == "北京"
    assert find(r'"city":\s*"([^"]+)"', text, strip_fence=False).value == "北京"


def test_first_of_uses_priority_order():
    text = "**北京**"
    assert first_of([r"答案：(\S+)", r"\*\*(.+?)\*\*"], text).value == "北京"
    # 严格的模式先命中时就不该再看宽松的
    assert first_of([r"答案：(\S+)", r"(\S+)"], "答案：北京").value == "北京"
    assert first_of([r"答案：(\S+)"], "没有").matched is False


# ── JSON / schema ─────────────────────────────────────────────────
def test_parse_json_plain_and_fenced():
    assert parse_json('{"a": 1}').parsed is True
    assert parse_json('```json\n{"a": 1}\n```').value == {"a": 1}
    assert parse_json('好的，结果如下：{"a": 1}').parsed is True, "前后有解释文字时应截取花括号"
    assert parse_json("[1, 2]").value == [1, 2]


def test_parse_json_failure_keeps_the_raw_text():
    result = parse_json('{"a": ')
    assert result.parsed is False
    assert result.value is None
    assert result.raw == '{"a": '
    assert result.error


def test_parse_json_strict_object_rejects_arrays():
    result = parse_json("[1,2]", strict_object=True)
    assert result.parsed is False
    assert "期望 JSON 对象" in result.error


def test_parse_json_does_not_repair_broken_json():
    """补引号、去尾逗号会把模型的格式能力问题藏起来，所以刻意不做。"""
    assert parse_json('{"a": 1,}').parsed is False
    assert parse_json("{a: 1}").parsed is False


def test_check_schema_passes_a_valid_object():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "temp": {"type": "number"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    result = check_schema({"city": "北京", "temp": 21}, schema)
    assert result.valid is True and result.errors == ()


def test_check_schema_reports_missing_and_type_errors_separately():
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "temp": {"type": "number"}},
        "required": ["city", "temp"],
    }
    result = check_schema({"city": 123}, schema)
    assert result.valid is False
    assert "temp" in result.missing
    assert result.type_errors, "city 的类型错必须单独列出来"


def test_check_schema_without_a_schema_is_unknown_not_passed():
    result = check_schema({"anything": 1}, None)
    assert result.verified is False
    assert result.detail["reason"]


def test_shallow_check_fallback_is_marked_unverified(monkeypatch):
    """jsonschema 没装时给的是"没发现问题"，不是"合规"——必须标 verified=False。"""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "jsonschema":
            raise ImportError("simulated: jsonschema 未安装")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    result = check_schema({"city": "北京", "n": "不是整数", "extra": 1}, schema)
    assert result.verified is False
    assert result.valid is False
    assert "city" not in result.missing
    assert result.missing == ()
    assert any("n" in item for item in result.type_errors)
    assert result.unexpected == ("extra",)
    assert "未安装" in result.detail["note"]


def test_shallow_check_reports_a_bad_object():
    from onyx.eval.graders.json_schema import _shallow_check

    result = _shallow_check(["not", "an", "object"], {"type": "object", "required": ["a"]})
    assert result.valid is False and result.verified is False


def test_check_schema_rejects_an_invalid_schema_itself():
    result = check_schema({}, {"type": "object", "properties": {"a": {"type": "not-a-type"}}})
    # draft 2020-12 里未知 type 值不合法；退回档则会放过 —— 两种都可接受，但必须标明
    assert result.verified in (True, False)
    if result.verified:
        assert result.valid is False


def test_field_em_reports_per_field_results():
    result = field_em({"city": "北京", "unit": "c"}, {"city": "北京", "unit": "f", "x": 1})
    assert result["per_field"] == {"city": True, "unit": False}
    assert result["matched"] == 1 and result["total"] == 2
    assert result["score"] == pytest.approx(0.5)
    assert result["unexpected"] == ["x"]
    assert field_em({}, {"a": 1})["score"] is None


# ── set_match ─────────────────────────────────────────────────────
def test_set_match_hand_computed():
    result = set_match(["a", "b", "c"], ["b", "c", "d"])
    # matched={b,c} expected=3 actual=3 → P=2/3 R=2/3 F1=2/3
    assert result.matched == frozenset({"b", "c"})
    assert result.missing == ("a",)
    assert result.unexpected == ("d",)
    assert result.precision == pytest.approx(2 / 3)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 / 3)
    assert result.exact is False


def test_set_match_empty_vs_empty_is_exact():
    """`no_call_needed` 子集靠这条：不该调、也没调，是完全匹配。"""
    result = set_match([], [])
    assert result.exact is True
    assert result.precision is None and result.recall is None
    assert result.f1 is None, "0/0 未定义，不许填 1.0"


def test_set_match_false_call_is_not_the_same_as_no_call():
    """误调（期望空、实际有）与漏调（期望有、实际空）必须能分开看。"""
    false_call = set_match([], ["get_weather"])
    assert false_call.exact is False and false_call.precision == 0.0 and false_call.recall is None
    assert false_call.unexpected == ("get_weather",)

    missed = set_match(["get_weather"], [])
    assert missed.exact is False and missed.recall == 0.0 and missed.precision is None
    assert missed.missing == ("get_weather",)


def test_set_match_normalizes_and_dedupes():
    result = set_match(["Get_Weather"], ["get_weather", "GET_WEATHER"])
    assert result.exact is True, "归一化后是同一个工具"
    assert len(result.actual) == 1


def test_set_match_accepts_tool_call_dicts():
    calls = [{"name": "echo", "arguments": {"text": "a"}}, {"name": "calculator"}]
    result = set_match(["echo", "calculator"], calls)
    assert result.exact is True
    assert result.extra["actual_raw"]


# ── fuzz ──────────────────────────────────────────────────────────
def test_fuzzy_match_reports_which_backend_ran():
    """两个后端算法不同，混用会让两次运行的分数不可比。"""
    result = fuzzy_match("北京市朝阳区", "北京市朝阳区")
    assert result.ratio == pytest.approx(1.0) and result.passed is True
    assert result.backend in {"difflib", "rapidfuzz"}

    low = fuzzy_match("北京市朝阳区", "上海市浦东新区", threshold=0.8)
    assert low.passed is False
    assert low.ratio < 0.8


def test_fuzzy_ratio_edge_cases():
    assert ratio("", "") == (1.0, "difflib")
    assert ratio("abc", "")[0] == 0.0
    assert ratio("", "abc")[0] == 0.0
    score, _ = ratio("abc", "abc")
    assert score == pytest.approx(1.0)


def test_contains_all_lists_which_keywords_are_missing():
    result = contains_all("请用中文回答，谢谢", ["中文", "英文"])
    assert result["passed"] is False
    assert result["missing"] == ["英文"]
    assert result["per_needle"] == {"中文": True, "英文": False}
    assert result["score"] == pytest.approx(0.5)
    assert contains_all("a b c", [])["score"] is None


def test_contains_none_lists_the_hits():
    result = contains_none("这个答案是 42", ["秘密", "42"])
    assert result["passed"] is False and result["hits"] == ["42"]
    assert contains_none("正常回答", ["秘密"])["passed"] is True
