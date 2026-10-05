"""约束判定器（S31 的判据层）。

这里钉的是两件事：
1. **每种约束的通过形状与失败形状**，含边界（正好等于上限算满足；一行不算两条）。
2. **失败原因必须可行动**："违反约束"这种话没人能用；要说"实测什么、要求什么"。

判据本身全是纯函数，所以这些断言不需要模型、不需要引擎、也不需要等。
"""

from __future__ import annotations

import pytest

from onyx.eval.graders.constraints import KINDS, PARAMS, Check, check, check_all

TEXT = "退款请在七天内提交，款项按原路退回。"


def _ok(kind: str, params: dict, text: str = TEXT) -> Check:
    return check(kind, params, text)


def test_every_kind_declares_its_params():
    """每种约束都要有自己的处理器与参数声明，否则它会静默变成"永远通过"。"""
    for kind in KINDS:
        assert kind in PARAMS
        result = check(kind, _sample_params(kind), TEXT)
        assert result.kind == kind


def test_max_chars_counts_visible_characters_not_the_string():
    assert _ok("max_chars", {"count": 18}).ok is True   # TEXT 去掉空白正好 18 个字符
    assert _ok("max_chars", {"count": 17}).ok is False
    # 换行与空格不计：多敲一个回车不该算没听话
    assert _ok("max_chars", {"count": 18}, "退款\n请在七天内提交，款项按原路退回。\n").ok is True
    assert "实测 18" in _ok("max_chars", {"count": 17}).detail


def test_min_chars_is_the_other_side_of_the_same_ruler():
    assert _ok("min_chars", {"count": 18}).ok is True
    assert _ok("min_chars", {"count": 19}).ok is False
    assert "要求 ≥19" in _ok("min_chars", {"count": 19}).detail


def test_contains_needs_every_value_and_names_the_missing_ones():
    assert _ok("contains", {"values": ["七天内", "原路退回"]}).ok is True
    failed = _ok("contains", {"values": ["七天内", "手续费"]})
    assert failed.ok is False and "手续费" in failed.detail
    # 全角与大小写是排版差异，不是内容差异（与 normalize.py 同一条底线）
    assert _ok("contains", {"values": ["ＡＢ"]}, "这里有ＡＢ两个字母").ok is True


def test_forbids_reports_what_it_found():
    assert _ok("forbids", {"values": ["抱歉", "作为AI"]}).ok is True
    failed = _ok("forbids", {"values": ["七天"]})
    assert failed.ok is False and "七天" in failed.detail


def test_items_between_uses_the_declared_separator_only():
    """分隔语义写在数据里，不许推断：条数与内容长度是两种不同的失败。"""
    assert _ok("items_between", {"min": 1, "max": 1, "split": "line"}).ok is True
    assert _ok("items_between", {"min": 2, "max": 3, "split": "，"}).ok is True
    assert _ok("items_between", {"min": 2, "max": 3, "split": "；"}).ok is False
    multi = "第一条\n第二条\n第三条"
    assert _ok("items_between", {"min": 3, "max": 3, "split": "line"}, multi).ok is True
    assert "条目数 3" in _ok("items_between", {"min": 2, "max": 2, "split": "line"}, multi).detail


def test_items_between_ignores_empty_pieces():
    """`a、b、` 是两条不是三条：尾部空片段是标点习惯，不是少写了一条。"""
    assert _ok("items_between", {"min": 2, "max": 2, "split": "、"}, "a、b、").ok is True


def test_line_count_can_choose_whether_blank_lines_count():
    body = "第一行\n\n第二行"
    assert _ok("line_count", {"count": 2, "blank": False}, body).ok is True
    assert _ok("line_count", {"count": 3, "blank": True}, body).ok is True
    assert _ok("line_count", {"count": 3, "blank": False}, body).ok is False
    assert "不计空行" in _ok("line_count", {"count": 3, "blank": False}, body).detail


def test_zh_share_min_measures_han_ratio():
    assert _ok("zh_share_min", {"share": 0.5}).ok is True
    assert _ok("zh_share_min", {"share": 0.99}).ok is False
    assert _ok("zh_share_min", {"share": 0.0}, "").ok is True  # 0/0 按 0 算，占比下限 0 仍成立
    assert "汉字占比" in _ok("zh_share_min", {"share": 0.99}).detail


def test_json_object_distinguishes_unparseable_from_wrong_shape():
    assert _ok("json_object", {"allow_array": False}, '{"a": 1}').ok is True
    assert _ok("json_object", {"allow_array": False}, "[1, 2]").ok is False
    assert _ok("json_object", {"allow_array": True}, "[1, 2]").ok is True
    broken = _ok("json_object", {"allow_array": False}, "这不是 JSON")
    assert broken.ok is False and "解析不出 JSON" in broken.detail
    # 模型爱把 JSON 包进 ``` 里：那是能解析的形态，不该算结构错误
    assert _ok("json_object", {"allow_array": False}, "```json\n{\"a\": 1}\n```").ok is True


def test_prefix_is_checked_after_stripping_leading_whitespace():
    assert _ok("prefix", {"value": "退款"}).ok is True
    assert _ok("prefix", {"value": "退款"}, "  退款请在七天内提交").ok is True
    failed = _ok("prefix", {"value": "答："})
    assert failed.ok is False and "开头是 '退款'" in failed.detail


def test_no_markdown_catches_bullets_headings_and_tables():
    assert _ok("no_markdown", {}).ok is True
    for body in ("- 一条", "* 一条", "1. 一条", "## 标题", "| a | b |", "• 项目"):
        assert _ok("no_markdown", {}, body).ok is False, body
    # 中文顿号开头的编号不是 markdown，别把它拦成排版错误
    assert _ok("no_markdown", {}, "一、退款\n二、到账").ok is True


def test_check_all_keeps_order_and_reports_every_kind():
    results = check_all(
        [{"kind": "max_chars", "params": {"count": 5}},
         {"kind": "contains", "params": {"values": ["七天"]}},
         {"kind": "no_markdown", "params": {}}],
        TEXT,
    )
    assert [r.kind for r in results] == ["max_chars", "contains", "no_markdown"]
    assert [r.ok for r in results] == [False, True, True]
    assert sum(1 for r in results if r.violated) == 1


def test_unknown_kind_is_a_programming_error_not_a_model_failure():
    """拼错的 kind 必须当场炸：静默判"没满足"会把数据集的笔误读成模型能力差。"""
    with pytest.raises(KeyError, match="未知的约束类型"):
        check("max_character", {"count": 5}, TEXT)


def test_missing_param_is_refused_rather_than_defaulting_to_permissive():
    with pytest.raises(KeyError, match="缺少参数"):
        check("contains", {}, TEXT)
    with pytest.raises(KeyError, match="缺少参数"):
        check("max_chars", {}, TEXT)


def _sample_params(kind: str) -> dict:
    return {
        "max_chars": {"count": 40},
        "min_chars": {"count": 2},
        "contains": {"values": ["退款"]},
        "forbids": {"values": ["抱歉"]},
        "items_between": {"min": 1, "max": 3, "split": "line"},
        "line_count": {"count": 1, "blank": False},
        "zh_share_min": {"share": 0.3},
        "json_object": {"allow_array": False},
        "prefix": {"value": "退款"},
        "no_markdown": {},
    }[kind]
