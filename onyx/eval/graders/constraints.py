"""约束判定器（S31 指令遵循任务的判据层）。

每条约束都是一个**纯函数**：`check(kind, params, text) -> Check(ok, detail)`。
两条纪律：

1. **判据必须代码可判**。凡需要人读才能判断的"是否遵循"（语气是否礼貌、表达是否自然）
   都不在这里——那是评审不是测量，混进分数就再也说不清 0.8 与 0.9 差在哪。
2. **失败必须可行动**。`detail` 说的是"实测什么、要求什么"（例如"条目数 6，要求 2–3"），
   而不是"违反约束"。一个只报"没满足"的判据，等于把 20 种不同的病压成同一个症状。

归一化只处理无语义差异的形态（全半角、空白、大小写、代码围栏），与 `normalize.py` 同一条底线：
放宽排版，不放宽内容。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from onyx.eval.graders.json_schema import parse_json
from onyx.eval.graders.normalize import normalize_text

#: 每种约束需要哪些参数。声明出来是为了**考卷自检**：
#: 参数写漏的约束会静默变成"永远通过"，那比没有这条约束更坏。
PARAMS: dict[str, tuple[str, ...]] = {
    "max_chars": ("count",),
    "min_chars": ("count",),
    "contains": ("values",),
    "forbids": ("values",),
    "items_between": ("min", "max", "split"),
    "line_count": ("count", "blank"),
    "zh_share_min": ("share",),
    "json_object": ("allow_array",),
    "prefix": ("value",),
    "no_markdown": (),
}

KINDS: tuple[str, ...] = tuple(PARAMS)

_HAN = re.compile(r"[㐀-䶿一-鿿]")
_LIST_LINE = re.compile(r"^\s*(?:[-*•·]|\d+[.、)])\s*")
_HEADING_LINE = re.compile(r"^\s*#{1,6}\s*\S")
_TABLE_LINE = re.compile(r"^\s*\|.*\|")


@dataclass(frozen=True, slots=True)
class Check:
    """一次约束判定。`detail` 永远写"实测 vs 要求"。"""

    kind: str
    ok: bool
    detail: str = ""
    measured: float | int | str | None = None

    @property
    def violated(self) -> bool:
        return not self.ok

    def as_dict(self) -> dict[str, Any]:
        """落库用的形状：每条约束的判定与原因都要能回看，
        否则界面上只有一个 0.67 而没人知道是哪条约束没满足。"""
        return {"kind": self.kind, "ok": self.ok, "detail": self.detail}


def check(kind: str, params: Mapping[str, Any], text: str) -> Check:
    """判定一条约束。未知 kind 是**编程错误**而不是模型的失败，所以直接抛。"""
    handler = _HANDLERS.get(kind)
    if handler is None:
        raise KeyError(f"未知的约束类型 {kind!r}；可选: {sorted(KINDS)}")
    missing = [name for name in PARAMS[kind] if name not in params]
    if missing:
        raise KeyError(f"约束 {kind!r} 缺少参数 {missing}")
    return handler(params, text)


def check_all(constraints: Sequence[Mapping[str, Any]], text: str) -> list[Check]:
    return [check(str(c["kind"]), c.get("params") or {}, text) for c in constraints]


def _max_chars(params: Mapping[str, Any], text: str) -> Check:
    limit = int(params["count"])
    body = _visible(text)
    return Check("max_chars", len(body) <= limit,
                 f"实测 {len(body)} 个字符（不含空白），要求 ≤{limit}", len(body))


def _min_chars(params: Mapping[str, Any], text: str) -> Check:
    floor = int(params["count"])
    body = _visible(text)
    return Check("min_chars", len(body) >= floor,
                 f"实测 {len(body)} 个字符（不含空白），要求 ≥{floor}", len(body))


def _contains(params: Mapping[str, Any], text: str) -> Check:
    wanted = [str(v) for v in params["values"]]
    hay = normalize_text(text)
    missing = [v for v in wanted if normalize_text(v) not in hay]
    return Check("contains", not missing,
                 "全部要求词都出现" if not missing else f"缺少必须出现的词：{missing}",
                 len(wanted) - len(missing))


def _forbids(params: Mapping[str, Any], text: str) -> Check:
    banned = [str(v) for v in params["values"]]
    hay = normalize_text(text)
    found = [v for v in banned if normalize_text(v) in hay]
    return Check("forbids", not found,
                 "没有出现禁用词" if not found else f"出现了禁止的词：{found}", len(found))


def _items_between(params: Mapping[str, Any], text: str) -> Check:
    low, high = int(params["min"]), int(params["max"])
    items = _items(text, str(params["split"]))
    return Check("items_between", low <= len(items) <= high,
                 f"条目数 {len(items)}，要求 {low}–{high}（分隔方式 {params['split']}）", len(items))


def _line_count(params: Mapping[str, Any], text: str) -> Check:
    want = int(params["count"])
    blank = bool(params["blank"])
    lines = [line for line in text.strip().splitlines() if blank or line.strip()]
    return Check("line_count", len(lines) == want,
                 f"行数 {len(lines)}，要求 {want}（{'计入' if blank else '不计'}空行）", len(lines))


def _zh_share_min(params: Mapping[str, Any], text: str) -> Check:
    floor = float(params["share"])
    visible = _visible(text)
    share = len(_HAN.findall(visible)) / len(visible) if visible else 0.0
    return Check("zh_share_min", share >= floor,
                 f"汉字占比 {share:.2f}，要求 ≥{floor:.2f}", round(share, 4))


def _json_object(params: Mapping[str, Any], text: str) -> Check:
    allow_array = bool(params["allow_array"])
    result = parse_json(text, strict_object=not allow_array)
    if not result.parsed:
        return Check("json_object", False, f"解析不出 JSON：{result.error[:80]}", "unparseable")
    if allow_array:
        return Check("json_object", True, "JSON 数组或对象", "ok")
    return Check("json_object", result.as_object is not None,
                 "JSON 顶层不是对象" if result.as_object is None else "JSON 对象", "ok")


def _prefix(params: Mapping[str, Any], text: str) -> Check:
    want = str(params["value"])
    head = text.strip()[: len(want)]
    return Check("prefix", head == want, f"开头是 {head!r}，要求以 {want!r} 开头", head)


def _no_markdown(params: Mapping[str, Any], text: str) -> Check:
    hits = [
        line.strip()[:24]
        for line in text.splitlines()
        if _LIST_LINE.match(line) or _HEADING_LINE.match(line) or _TABLE_LINE.match(line)
    ]
    return Check("no_markdown", not hits,
                 "没有 markdown 列表/标题/表格" if not hits else f"出现了 markdown 排版：{hits[:3]}",
                 len(hits))


_HANDLERS = {
    "max_chars": _max_chars,
    "min_chars": _min_chars,
    "contains": _contains,
    "forbids": _forbids,
    "items_between": _items_between,
    "line_count": _line_count,
    "zh_share_min": _zh_share_min,
    "json_object": _json_object,
    "prefix": _prefix,
    "no_markdown": _no_markdown,
}


def _visible(text: str) -> str:
    """去掉所有空白后的正文。中文字数按这个算最接近人的直觉，
    而且不会把"模型多敲了两个换行"算成没听话。"""
    return re.sub(r"\s+", "", text or "")


def _items(text: str, split: str) -> list[str]:
    """按声明的方式数条目。分隔语义写死在数据里，**不许推断**：
    "模型给了 3 条"与"模型给了 1 行长句"是两种不同的失败，混起来就说不清该改什么。"""
    body = (text or "").strip()
    if not body:
        return []
    if split == "line":
        parts = body.splitlines()
    elif split == "sentence":
        parts = re.split(r"[。！？!?；;]", body)
    else:
        parts = re.split(re.escape(split), body)
    return [part.strip() for part in parts if part.strip()]
