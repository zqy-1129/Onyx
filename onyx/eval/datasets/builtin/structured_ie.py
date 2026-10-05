"""结构化抽取数据集生成器（`structured_ie.jsonl` 的来源）。

和 `intent_zh` 一样是**生成器**而不是一份来历不明的 JSONL：
谁抽出了什么字段、槽位怎么轮换、难例从哪来，都要能被读出来与重跑。

四条刻意的设计（每条都对应用过去踩过的坑）
1. **有"无可抽取"的负样本**（tag `none`）。没有它们，模型把"今天天气不错"编成一个
   `{"person": ""}` 也得满分——而凭空造字段恰恰是抽取任务最贵的失败。
2. **期望值是可机械判定的形态**。日期一律 ISO（`2026-03-05`），金额一律数值。
   表面写法可以千变万化（"两万三"、"下周三"），但**判据不许需要人读**。
   相对日期由 `ANCHOR` 算出（`_offset` / `_next_week_day`）而不是手填：这一版原稿里
   "后天"就填错了天，而填错的期望值会让模型答对也得 0 分。
3. **难例单独打 tag**（`hard`）：中文数字、相对日期（锚定日在 `meta.anchor` 里，所以可复现）、
   多主体句子里挑主要人物。这些单独跑才看得出模型到底卡在哪一层。
4. **v1 刻意不做数组字段**。列表字段的比较语义是集合而不是精确匹配（要用 `set_match`），
   和标量混在一个 grader 里会让"顺序不同"被判成"内容错"——那是测排版不是测能力。
   要加数组时先加比较器，再改这里。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

#: 相对日期的锚定日。写死而不是"今天"：数据集必须可复现，
#: 否则同一份 seed 明天重跑会换一批期望值，历史分数就对不上了。
#: 2026-03-01 是周日，所以"下周三"落在 03-04（下一个 ISO 周的周三）
ANCHOR = "2026-03-01"
_ANCHOR_DAY = date.fromisoformat(ANCHOR)


def _offset(days: int) -> str:
    """锚定日 ± N 天。相对日期的期望值一律**算出来**，不手填。"""
    return (_ANCHOR_DAY + timedelta(days=days)).isoformat()


def _next_week_day(iso_weekday: int) -> str:
    """下一个 ISO 周（周一是首日）里指定星期几的那天。0=周一。"""
    return _offset(7 - _ANCHOR_DAY.weekday() + iso_weekday)


PERSONS = ("张伟", "李娜", "王芳", "陈杰", "赵敏", "周立")
ORGS = ("支付宝", "招商银行", "国家电网", "美团", "腾讯客服")
PLACES = ("北京", "上海", "广州", "深圳", "杭州", "成都")
EVENTS = ("退款", "开户", "投诉", "预约维修", "注销账户")
#: (期望数值, 表面写法)。表面写法故意混着阿拉伯数字、逗号分组与中文数字
AMOUNTS: tuple[tuple[float, str], ...] = (
    (500.0, "500元"), (12800.0, "12,800元"), (88.5, "88块5"),
    (2300.0, "两千三"), (15000.0, "一万五"), (660.0, "六百六十元"),
    (1.25, "一块二毛五"), (9800.0, "九千八百块"),
)
#: (期望 ISO, 表面写法)。相对日期是难例：值由 `ANCHOR` 算出，
#: 手填的话没人能复核（原稿里「后天」被写成 03-02，锚定日 03-01 的后天其实是 03-03）
DATES: tuple[tuple[str, str], ...] = (
    ("2026-03-05", "2026年3月5日"),
    ("2026-03-04", "3月4号"),
    ("2026-02-28", "二月二十八日"),
    (_offset(2), "后天"),
    (_next_week_day(2), "下周三"),
    (_offset(7), "一周后"),
)

#: 模板：文本写法 + 槽位 → 期望字段的映射（槽位名 → 期望字段名）
TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "text": "{person}在{place}的{org}申请了{event}。",
        "slots": ("person", "place", "org", "event"),
        "expect": {"person": "person", "place": "place", "org": "org", "event": "event"},
    },
    {
        "text": "{person}于{date}向{org}支付了{amount_text}。",
        "slots": ("person", "date", "org", "amount_text"),
        "expect": {"person": "person", "date": "date", "org": "org", "amount": "amount"},
    },
    {
        "text": "麻烦帮{person}查{date}在{place}的那笔{amount_text}支出。",
        "slots": ("person", "date", "place", "amount_text"),
        "expect": {"person": "person", "date": "date", "place": "place", "amount": "amount"},
    },
    {
        "text": "{org}的客服说{person}的{event}已经处理完，费用是{amount_text}。",
        "slots": ("org", "person", "event", "amount_text"),
        "expect": {"org": "org", "person": "person", "event": "event", "amount": "amount"},
    },
    {
        "text": "{person}说人在{place}，{date}要办{event}。",
        "slots": ("person", "place", "date", "event"),
        "expect": {"person": "person", "place": "place", "date": "date", "event": "event"},
    },
)

#: 人工难例：(文本, 期望字段, 说明)
HARD_CASES: tuple[tuple[str, dict[str, Any], str], ...] = (
    ("张伟说转账两万三给李娜，用的是招商银行的卡。",
     {"person": "张伟", "amount": 23000.0, "org": "招商银行"},
     "中文数字 + 句中还有第二个人名，主要人物是发起动作的那个"),
    ("赵敏昨天在美团下了一单，后天再退，金额八十九块九。",
     {"person": "赵敏", "org": "美团", "date": _offset(-1), "amount": 89.9},
     "两个相对时间，取第一个（昨天）作为交易日期"),
    ("杭州的李娜说她在 2026 年 3 月 4 号那天丢了钱包，要投诉。",
     {"person": "李娜", "place": "杭州", "date": "2026-03-04", "event": "投诉"},
     "地点在人名前面，且日期是带空格的自然写法"),
    ("客服告诉我，国家电网的开户费是一千五百元，我上个月就在深圳办过了。",
     {"org": "国家电网", "event": "开户", "amount": 1500.0, "place": "深圳"},
     "没有人名；“我”不该被抽成 person"),
)

#: 负样本：句子里没有任何可抽取的字段。期望是**空对象**
NONE_CASES: tuple[str, ...] = (
    "今天天气不错，适合出去走走。",
    "你觉得这部电影怎么样？",
    "不好意思，我刚从外面回来。",
    "这件事我再想想，明天答复你。",
    "谢谢，辛苦了。",
)

FIELD_TYPES: dict[str, str] = {
    "person": "string", "org": "string", "place": "string",
    "event": "string", "amount": "number", "date": "string",
}

#: 取值是封闭集合的字段：提示词必须把词表给出来，否则"事件抽错"测的是猜词而不是抽取
#: （与 intent 任务必须声明标签集同一条理由）。词表与 `EVENTS` 同源，
#: 所以期望值一定落在集合内——否则考卷自己就不合规
VALUE_VOCAB: dict[str, tuple[str, ...]] = {"event": EVENTS}


def schema_for(keys: Sequence[str]) -> dict[str, Any]:
    """某个 case 的 schema：只允许约定字段，且要求的字段必须都在。

    `required` 是逐条不同的（每条要抽的字段不一样），所以它是**每次调用现造**的
    小对象；共享一份 schema 会让"少抽一个字段"变成合规。
    """
    return {
        "type": "object",
        "properties": {
            key: ({"type": "number"} if FIELD_TYPES[key] == "number"
                  else {"type": "string", "minLength": 1})
            for key in keys
        },
        "required": sorted(keys),
        "additionalProperties": False,
    }


def build_cases(*, seed: int = 20261003) -> list[dict[str, Any]]:
    """生成全部样本。同 seed 必然同结果。"""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []

    for template_index, template in enumerate(TEMPLATES):
        slot_names = template["slots"]
        for combo in _rotations(slot_names, seed + template_index):
            expect = _expected(template, combo)
            if expect is None:
                continue
            text = str(template["text"]).format(**combo)
            cases.append({
                "input": {"text": text},
                "expect": {"fields": expect, "keys": sorted(expect)},
                "kind": "single",
                "tags": ["template", *sorted(expect)],
                "meta": {"template_index": template_index, "anchor": ANCHOR,
                         "slots": {k: combo[k] for k in slot_names}},
            })

    for index, (text, expect, note) in enumerate(HARD_CASES):
        cases.append({
            "input": {"text": text},
            "expect": {"fields": expect, "keys": sorted(expect)},
            "kind": "single",
            "tags": ["hard", *sorted(expect)],
            "meta": {"hard_index": index, "note": note, "anchor": ANCHOR},
        })

    for index, text in enumerate(NONE_CASES):
        cases.append({
            "input": {"text": text},
            "expect": {"fields": {}, "keys": []},
            "kind": "none",
            "tags": ["none"],
            "meta": {"none_index": index},
        })

    for case in cases:
        case["id"] = _case_id(case["input"]["text"], case["expect"]["fields"])
    rng.shuffle(cases)
    for ord_index, case in enumerate(cases):
        case["ord"] = ord_index
    return cases


def _rotations(slot_names: Sequence[str], seed: int) -> Sequence[dict[str, Any]]:
    """每个槽位独立轮换一遍再对齐：宽度取最长的那个，保证每个值都出场过。"""
    pool = {name: _values_for(name) for name in slot_names}
    width = max(len(values) for values in pool.values())
    rng = random.Random(seed)
    orders = {name: rng.sample(range(len(values)), len(values)) for name, values in pool.items()}
    out: list[dict[str, Any]] = []
    for index in range(width):
        out.append({
            name: values[orders[name][index % len(values)]]
            for name, values in pool.items()
        })
    return out


def _values_for(slot: str) -> Sequence[Any]:
    if slot == "amount_text":
        return [text for _value, text in AMOUNTS]
    if slot == "date":
        return [text for _iso, text in DATES]
    return {"person": PERSONS, "org": ORGS, "place": PLACES, "event": EVENTS}[slot]


def _expected(template: dict[str, Any], combo: dict[str, Any]) -> dict[str, Any] | None:
    """槽位值 → 期望字段。日期与金额要把表面写法换成可判定的规范值。"""
    out: dict[str, Any] = {}
    for field, slot in template["expect"].items():
        if slot == "amount":
            pair = next(((value, text) for value, text in AMOUNTS if text == combo["amount_text"]),
                        None)
            if pair is None:
                return None
            out["amount"] = pair[0]
        elif slot == "date":
            iso = next((value for value, text in DATES if text == combo["date"]), None)
            if iso is None:
                return None
            out["date"] = iso
        else:
            out[field] = combo[slot]
    return out


def _case_id(text: str, fields: dict[str, Any]) -> str:
    """稳定 id 由**内容**算出。重跑生成器不能让 case_id 漂移，否则历史 grade 对不上。"""
    payload = json.dumps({"t": text, "f": fields}, ensure_ascii=False, sort_keys=True)
    return f"sie-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def to_jsonl(cases: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases)


def stats(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """数据集画像。字段分布必须可见：只有 person 抽得准、amount 全错时，
    单一总分说不清到底哪里坏了。
    """
    from collections import Counter

    fields = Counter(key for case in cases for key in case["expect"]["keys"])
    return {
        "n": len(cases),
        "fields": dict(sorted(fields.items(), key=lambda kv: -kv[1])),
        "none": sum(1 for case in cases if case["kind"] == "none"),
        "hard": sum(1 for case in cases if "hard" in case["tags"]),
        "unique_texts": len({case["input"]["text"] for case in cases}),
    }


def write_jsonl(path: Path | str, *, seed: int = 20261003) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_jsonl(build_cases(seed=seed)) + "\n", encoding="utf-8", newline="\n")
    return target


BUILTIN_PATH = Path(__file__).resolve().parent / "structured_ie.jsonl"

if __name__ == "__main__":  # pragma: no cover
    print(f"{write_jsonl(BUILTIN_PATH)} → {stats(build_cases())}")
