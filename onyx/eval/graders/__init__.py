"""评分器集合。

分工必须清楚，否则"分数"会变成一个含义不明的数：
- `exact`    归一化后精确匹配（标签、枚举、短答案）
- `numeric`  数值容差匹配
- `set_match` 集合 P/R/F1（并行工具调用、多标签）
- `regex`    从自由文本里抽答案（抽不到就是 None，不猜）
- `json_schema` 解析 / schema 合规 / 字段级 EM，三层分开报
- `fuzz`     **只**用于自由文本；标签与数值走它会掩盖真实错误
"""

from __future__ import annotations

from onyx.eval.graders.exact import exact, numeric
from onyx.eval.graders.fuzz import contains_all, contains_none, fuzzy_match, ratio
from onyx.eval.graders.json_schema import (
    JsonCheck,
    SchemaCheck,
    check_schema,
    field_em,
    parse_json,
)
from onyx.eval.graders.normalize import LabelMatch, match_label, normalize_text
from onyx.eval.graders.regex import RegexHit, find, first_of
from onyx.eval.graders.set_match import SetMatch, set_match

__all__ = [
    "JsonCheck",
    "LabelMatch",
    "RegexHit",
    "SchemaCheck",
    "SetMatch",
    "check_schema",
    "contains_all",
    "contains_none",
    "exact",
    "field_em",
    "find",
    "first_of",
    "fuzzy_match",
    "match_label",
    "normalize_text",
    "numeric",
    "parse_json",
    "ratio",
    "set_match",
]
