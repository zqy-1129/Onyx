"""类型感知的参数比对。

工具调用的参数对不对，不能用 `==` 判：
- `{"amount": 500}` 与 `{"amount": "500元"}` 是同一个意思
- `{"date": "2026-10-03"}` 与 `{"date": "2026/10/3"}` 是同一个意思
- `{"tags": ["a","b"]}` 与 `{"tags": ["b","a"]}` 通常也是同一个意思
- `{"city": "北京"}` 与 `{"city": "北京市"}` 不是同一个意思（差一个字就是另一个实体）

所以比对必须**按类型走不同规则**，而且每一条判定都要留下 `kind`：
"日期归一后相等"与"数值在容差内"与"模糊匹配过关"的可信度完全不同，
混成一个 True 就没法回答"这个分数有多少是靠放宽规则挣来的"。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from onyx.eval.graders.exact import numeric
from onyx.eval.graders.normalize import normalize_text, trim_wrappers

#: 数值默认容差。绝对 + 相对二选一满足即可
DEFAULT_ABS_TOL = 1e-9
DEFAULT_REL_TOL = 0.0
#: 模糊匹配只对显式列入 `fuzzy_fields` 的字段生效——默认不开
DEFAULT_FUZZ_THRESHOLD = 0.9

_DATE_PATTERNS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%Y%m%d",
    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m", "%Y",
)
_CN_DATE = re.compile(r"^\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日?\s*$")
_CN_MONTH = re.compile(r"^\s*(\d{4})\s*年\s*(\d{1,2})\s*月\s*$")

#: 靠"放宽规则"达成的匹配方式。两处统计（`ArgMatch.relaxed_kinds` 与 `summarize`）
#: 必须用同一份名单，否则 exact_rate 与 relaxed_share 会互相矛盾。
#: 注意 `enum` 不在里面：枚举的大小写差异算命中同一条规则，不是放宽。
RELAXED_KINDS = frozenset({
    "normalized", "numeric_tolerance", "date_normalized", "set_equal", "fuzzy",
})


@dataclass(frozen=True, slots=True)
class FieldMatch:
    field: str
    ok: bool
    #: 判定方式：identical / normalized / numeric_tolerance / date_normalized /
    #: enum / set_equal / fuzzy / type_mismatch / value_mismatch / missing / unexpected
    kind: str
    expected: Any = None
    actual: Any = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ArgMatch:
    fields: tuple[FieldMatch, ...] = ()
    exact: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def matched(self) -> tuple[str, ...]:
        return tuple(f.field for f in self.fields if f.ok)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(f.field for f in self.fields if f.kind == "missing")

    @property
    def unexpected(self) -> tuple[str, ...]:
        return tuple(f.field for f in self.fields if f.kind == "unexpected")

    @property
    def mismatched(self) -> dict[str, dict[str, Any]]:
        return {
            f.field: {"expected": f.expected, "actual": f.actual, "kind": f.kind,
                      "detail": f.detail}
            for f in self.fields
            if not f.ok and f.kind not in {"missing", "unexpected"}
        }

    @property
    def ok(self) -> bool:
        """完全匹配：不缺、不错、（严格档下）不多。"""
        bad = [f for f in self.fields if not f.ok]
        if self.exact:
            return not bad
        return not [f for f in bad if f.kind != "unexpected"]

    @property
    def subset_ok(self) -> bool:
        """期望的字段都在且值对；模型多给字段不算错。"""
        return not [f for f in self.fields if not f.ok and f.kind != "unexpected"]

    @property
    def strict_ok(self) -> bool:
        """完全匹配，且**没有靠任何放宽规则**（归一化/容差/日期/集合/模糊）挣来的。

        `ok` 回答"语义上对不对"，`strict_ok` 回答"字面上就一模一样吗"。
        两者的差就是 `relaxed_share`：只报 `ok` 会把"我们放宽了规则"藏起来，
        于是分数变高看起来像模型变强了。
        """
        return self.ok and not self.relaxed_kinds

    @property
    def score(self) -> float | None:
        expected = [f for f in self.fields if f.kind != "unexpected"]
        if not expected:
            return None
        return sum(1 for f in expected if f.ok) / len(expected)

    @property
    def relaxed_kinds(self) -> tuple[str, ...]:
        """靠"放宽规则"挣来的那些匹配。

        必须能被看见：`numeric_tolerance` 与 `fuzzy` 通过的比例越高，
        这个分数就越依赖比对器的宽容度而不是模型的能力。
        """
        relaxed = {"normalized", "numeric_tolerance", "date_normalized", "set_equal", "fuzzy"}
        return tuple(f.kind for f in self.fields if f.ok and f.kind in relaxed)


def match_args(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
    *,
    schema: Mapping[str, Any] | None = None,
    exact: bool = False,
    abs_tol: float = DEFAULT_ABS_TOL,
    rel_tol: float = DEFAULT_REL_TOL,
    fuzzy_fields: Sequence[str] = (),
    fuzz_threshold: float = DEFAULT_FUZZ_THRESHOLD,
    unordered_fields: Sequence[str] = (),
) -> ArgMatch:
    """逐字段比对。`schema` 用来拿字段的声明类型与 enum（有则更准，没有也能跑）。"""
    expected = dict(expected or {})
    actual = dict(actual or {})
    properties = (schema or {}).get("properties") or {}
    fuzzy = set(fuzzy_fields)
    unordered = set(unordered_fields)

    out: list[FieldMatch] = []
    for name in list(expected) + [k for k in actual if k not in expected]:
        prop = properties.get(name) if isinstance(properties.get(name), Mapping) else {}
        if name not in actual:
            out.append(FieldMatch(name, False, "missing", expected=expected[name]))
            continue
        if name not in expected:
            out.append(FieldMatch(name, False, "unexpected", actual=actual[name]))
            continue
        out.append(_match_field(
            name, expected[name], actual[name], prop,
            abs_tol=abs_tol, rel_tol=rel_tol,
            fuzzy=name in fuzzy, fuzz_threshold=fuzz_threshold,
            unordered=name in unordered or _looks_unordered(prop),
        ))
    return ArgMatch(fields=tuple(out), exact=exact)


def _looks_unordered(prop: Mapping[str, Any]) -> bool:
    """schema 里声明为 array + uniqueItems 的字段按集合语义比。

    顺序有意义的数组（例如多步操作的步骤列表）不该走这条，所以只在
    `uniqueItems` 明确为真时才启用——宁可漏放宽，不可错放宽。
    """
    return prop.get("type") == "array" and prop.get("uniqueItems") is True


def _match_field(
    name: str,
    expected: Any,
    actual: Any,
    prop: Mapping[str, Any],
    *,
    abs_tol: float,
    rel_tol: float,
    fuzzy: bool,
    fuzz_threshold: float,
    unordered: bool,
) -> FieldMatch:
    if expected is None or actual is None:
        ok = expected is None and actual is None
        return FieldMatch(name, ok, "identical" if ok else "value_mismatch", expected, actual,
                          "" if ok else "一边是 null")

    # bool 必须在数值之前判：Python 里 True == 1，先判数值会把 True 当成 1 放过
    if isinstance(expected, bool) or isinstance(actual, bool):
        ok = isinstance(actual, bool) and expected is actual
        return FieldMatch(name, ok, "identical" if ok else "type_mismatch", expected, actual,
                          "" if ok else f"期望 bool，实际 {type(actual).__name__}")

    declared = prop.get("type")
    enum = prop.get("enum")

    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping):
            return FieldMatch(name, False, "type_mismatch", expected, actual,
                              f"期望对象，实际 {type(actual).__name__}")
        nested = match_args(expected, actual, exact=True, abs_tol=abs_tol, rel_tol=rel_tol)
        return FieldMatch(name, nested.ok, "identical" if nested.ok else "value_mismatch",
                          expected, actual,
                          "" if nested.ok else f"嵌套不匹配: {nested.mismatched or nested.missing}")

    if isinstance(expected, list | tuple):
        if not isinstance(actual, list | tuple):
            return FieldMatch(name, False, "type_mismatch", expected, actual,
                              f"期望数组，实际 {type(actual).__name__}")
        if unordered:
            left = {normalize_text(_scalar(item)) for item in expected}
            right = {normalize_text(_scalar(item)) for item in actual}
            ok = left == right
            return FieldMatch(name, ok, "set_equal" if ok else "value_mismatch",
                              list(expected), list(actual),
                              "" if ok else f"集合不等：缺 {sorted(left - right)} 多 {sorted(right - left)}")
        if len(expected) != len(actual):
            return FieldMatch(name, False, "value_mismatch", list(expected), list(actual),
                              f"长度不同：{len(expected)} vs {len(actual)}")
        for index, (want, got) in enumerate(zip(expected, actual, strict=False)):
            item = _match_field(f"{name}[{index}]", want, got, {}, abs_tol=abs_tol,
                                rel_tol=rel_tol, fuzzy=fuzzy, fuzz_threshold=fuzz_threshold,
                                unordered=False)
            if not item.ok:
                return FieldMatch(name, False, item.kind, list(expected), list(actual),
                                  f"第 {index} 项: {item.detail or item.kind}")
        return FieldMatch(name, True, "identical", list(expected), list(actual))

    if enum:
        left, right = normalize_text(expected), normalize_text(actual)
        allowed = {normalize_text(str(item)): str(item) for item in enum}
        if left not in allowed:
            # 期望值本身不在 enum 里，说明用例写错了——这必须显式报出来
            return FieldMatch(name, False, "value_mismatch", expected, actual,
                              f"期望值 {expected!r} 不在 schema 的 enum {list(enum)} 内")
        ok = right == left
        return FieldMatch(name, ok, "enum" if ok else "value_mismatch", expected, actual,
                          "" if ok else f"枚举越界：{actual!r} 应为 {allowed[left]!r}")

    if isinstance(expected, int | float) or declared in {"number", "integer"}:
        if numeric(expected, actual, tolerance=abs_tol, rel=rel_tol):
            kind = "identical" if expected == actual else "numeric_tolerance"
            return FieldMatch(name, True, kind, expected, actual)
        return FieldMatch(name, False, "value_mismatch", expected, actual,
                          f"数值不等（容差 abs={abs_tol} rel={rel_tol}）")

    left, right = str(expected), str(actual)
    if left == right:
        return FieldMatch(name, True, "identical", expected, actual)
    if normalize_text(left) == normalize_text(right) or \
            normalize_text(trim_wrappers(left)) == normalize_text(trim_wrappers(right)):
        return FieldMatch(name, True, "normalized", expected, actual, "仅排版差异")

    left_date, right_date = normalize_date(left), normalize_date(right)
    if left_date is not None and left_date == right_date:
        return FieldMatch(name, True, "date_normalized", expected, actual,
                          f"归一到 {left_date.isoformat()}")

    if numeric(left, right, tolerance=abs_tol, rel=rel_tol):
        return FieldMatch(name, True, "numeric_tolerance", expected, actual,
                          "一边是数字一边是数字字符串")

    if fuzzy:
        from onyx.eval.graders.fuzz import fuzzy_match

        result = fuzzy_match(left, right, threshold=fuzz_threshold)
        return FieldMatch(name, result.passed, "fuzzy" if result.passed else "value_mismatch",
                          expected, actual,
                          f"模糊匹配 {result.ratio:.3f} / 阈值 {fuzz_threshold}（后端 {result.backend}）")

    return FieldMatch(name, False, "value_mismatch", expected, actual, "值不同")


def _scalar(item: Any) -> str:
    return item if isinstance(item, str) else str(item)


def normalize_date(value: Any) -> date | None:
    """把常见日期写法归一成 `date`。归一不了返回 None（**不是**某个默认日期）。

    支持的形态：`2026-10-03` / `2026/10/3` / `2026.10.03` / `20261003` /
    `2026年10月3日` / `2026-10` / `2026年10月` / `2026`。
    只到月或只到年时归一到该月/年的 1 号，这样 `2026-10` 与 `2026-10-01` 相等——
    这是个**有意的放宽**，所以判定 kind 会标成 `date_normalized` 让人看见。
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None

    match = _CN_DATE.match(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = _CN_MONTH.match(text)
    if match:
        return _safe_date(int(match.group(1)), int(match.group(2)), 1)

    cleaned = normalize_text(text, casefold=True)
    for pattern in _DATE_PATTERNS:
        try:
            return datetime.strptime(cleaned, pattern).date()
        except ValueError:
            continue
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def summarize(matches: Sequence[ArgMatch]) -> dict[str, Any]:
    """一批参数比对的汇总。

    `relaxed_share` 是这里最重要的一个数：靠归一化/容差/模糊匹配通过的字段占比。
    它越高，说明这个分数越依赖比对器的宽容度——必须与 args_exact 一起看。
    """
    usable = [m for m in matches if m.fields]
    total_fields = sum(len([f for f in m.fields if f.kind != "unexpected"]) for m in usable)
    ok_fields = sum(len(m.matched) for m in usable)
    kinds: dict[str, int] = {}
    for item in usable:
        for match in item.fields:
            kinds[match.kind] = kinds.get(match.kind, 0) + 1
    relaxed = sum(
        1 for item in usable for f in item.fields
        if f.ok and f.kind in RELAXED_KINDS
    )
    return {
        "n": len(matches),
        # exact 用 strict_ok：`exact_rate` 的含义必须是"字面就一致"，
        # 否则它与 relaxed_share 会互相重叠，两个数加起来说不清
        "exact_rate": _rate(sum(1 for m in matches if m.strict_ok), len(matches)),
        "subset_rate": _rate(sum(1 for m in matches if m.subset_ok), len(matches)),
        "semantic_rate": _rate(sum(1 for m in matches if m.ok), len(matches)),
        "field_total": total_fields,
        "field_matched": ok_fields,
        "field_rate": _rate(ok_fields, total_fields),
        "relaxed_fields": relaxed,
        "relaxed_share": _rate(relaxed, ok_fields),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
