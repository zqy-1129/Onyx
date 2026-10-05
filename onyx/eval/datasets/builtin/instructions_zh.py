"""指令遵循数据集生成器（`instructions_zh.jsonl` 的来源）。

这个数据集的核心设计只有一句话：**约束由参考回答派生，而不是先写数字再指望有人能满足。**

每条 case 都带一个 `meta.reference` —— 一句必然满足该 case 全部约束的参考文本。
约束参数（长度上限、条目数、汉字占比、必须出现的词）从它算出来并留一点余量，
于是"这道题存在解"变成一条可断言的事实
（`tests/unit/test_instructions_zh_dataset.py`），而不是一句好话。
为什么值得这么做：先写"≤20 字 / 必须含『风险』『收益』 / 分 3 行"这种数字，
很容易写出一道**没人能同时满足**的题；现场只会怀疑模型，而真正坏掉的是考卷。

另外三条：
1. **提示语由约束渲染出来**。凡是考模型的，必须在指令文本里说得出——
   不写"表达要自然"这种只有人能给分的判据（那是评审不是测量）。
2. **约束数 2–4 条**，且难例（tag `hard`）是"故意紧张但仍有解"的组合。
3. **每种约束都要有足够分母**：某类约束只考了 2 条时，它的满足率说明不了任何事，
   所以 `by_kind` 带着 n 一起出（与 S30 的 `per_field` 同一条理由）。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from onyx.eval.graders.constraints import check_all

#: 话题池：`ask` 是让学生做什么，`words` 是可以拿来当"必须包含"的词
#: （它们必然出现在参考回答里，所以约束不会出现"指令没说要含 X 却考 X"），
#: `lines` 是分条形态时的参考内容，`object` 是 JSON 形态时的参考内容。
TOPICS: tuple[dict[str, Any], ...] = (
    {
        "id": "refund", "ask": "说明退款政策",
        "words": ("七天内", "原路退回", "手续费"),
        "lines": ("请在七天内提交申请", "款项按原路退回", "逾期需付手续费"),
        "object": {"window": "七天内", "channel": "原路退回", "fee": True},
    },
    {
        "id": "meeting", "ask": "安排一场会议",
        "words": ("周三", "会议室", "十五分钟"),
        "lines": ("周三下午两点开会", "地点在三楼会议室", "每人限时十五分钟"),
        "object": {"day": "周三", "room": "会议室", "limit": "十五分钟"},
    },
    {
        "id": "battery", "ask": "介绍一块电池的保养",
        "words": ("不要", "高温", "满电"),
        "lines": ("不要长期满电存放", "避开高温环境", "每月放一次到三成"),
        "object": {"avoid": "高温", "charge": "满电", "monthly": True},
    },
    {
        "id": "recipe", "ask": "写一段番茄炒蛋的做法",
        "words": ("鸡蛋", "番茄", "小火"),
        "lines": ("鸡蛋打散加少许盐", "番茄切块炒出汁", "小火回锅翻匀"),
        "object": {"eggs": 3, "tomato": 2, "heat": "小火"},
    },
    {
        "id": "privacy", "ask": "解释一句隐私声明",
        "words": ("不会", "出售", "脱敏"),
        "lines": ("我们不会出售你的数据", "统计前先脱敏处理", "随时可导出并删除"),
        "object": {"sell": False, "mask": "脱敏", "export": True},
    },
    {
        "id": "delivery", "ask": "告知一次配送进度",
        "words": ("今天", "驿站", "取件码"),
        "lines": ("包裹今天送达", "放在小区驿站", "取件码是 4-4-8821"),
        "object": {"day": "今天", "place": "驿站", "code": "4-4-8821"},
    },
)

#: 禁止出现的词：参考回答里必然没有它们（生成时会校验），所以约束是可满足的
BANNED = ("抱歉", "作为AI", "我无法", "免责声明", "首先")

#: 分条形态。`split` 与 `constraints.py` 里的 `_items()` 同源
SPLITS = ("line", "；", "、", "sentence")

#: 每种约束至少要考几条。低于这个数，`by_kind` 里那一格的"满足率"没有意义，
#: 而"加了新约束类型却没人出题"也会在这里被抓到（测试读的是这个常量）。
KIND_COVERAGE = 3

#: 单条样本的约束上限。题越复杂越说不清"到底哪条把它难住了"，
#: 而且 6 条以上必然有冗余（`min_chars` 与 `max_chars` 同时在场时，掉分往往来自字数而不是内容）
MAX_CONSTRAINTS = 5

#: 超上限时先丢谁：字数是一对冗余约束，`prefix` 的信息量也低于内容约束
_DROP_ORDER = ("min_chars", "prefix", "forbids")


def _trim(constraints: list[dict[str, Any]], limit: int, rng: random.Random) -> None:
    """就地裁掉超出上限的约束。裁谁要写死顺序，不能让随机数决定——
    否则同一个 seed 会因为多抽了一条而丢掉不同的约束，两次分数就不是同一张考卷。"""
    while len(constraints) > limit:
        victim = next((kind for kind in _DROP_ORDER
                       if any(item["kind"] == kind for item in constraints)), None)
        if victim is None:
            victim = str(constraints[-1]["kind"])
        for index, item in enumerate(constraints):
            if item["kind"] == victim:
                del constraints[index]
                break


#: 每个话题 × 每种形态出几个变体。少了某类约束就没有分母（21 条时 `line_count` 只有 1 条），
#: 那种"满足率 100%"与"没考到"几乎同样没有意义。
VARIANTS = 2


def build_cases(*, seed: int = 20261005) -> list[dict[str, Any]]:
    """生成全部样本。同 seed 必然同结果。"""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []

    for topic in TOPICS:
        for form in ("plain", "lines", "json"):
            for _variant in range(VARIANTS):
                cases.append(_make(topic, form=form, rng=rng))

    for hard in HARD:
        cases.append(_make_hard(hard))

    for case in cases:
        case["id"] = _case_id(case["input"]["text"], case["expect"]["constraints"])
    rng.shuffle(cases)
    for ord_index, case in enumerate(cases):
        case["ord"] = ord_index
    return cases


def _make(topic: dict[str, Any], *, form: str, rng: random.Random) -> dict[str, Any]:
    """按形态造一条样本：**先有参考回答，再从它派生约束与提示语**。

    每条约束都问参考回答要实测值（长度、条目数、汉字占比、能用的必含词），
    所以"这道题有解"是构造出来的事实，而不是事后祈祷。
    """
    split = rng.choice(SPLITS) if form == "lines" else "line"
    reference = _reference(topic, form, split)
    constraints: list[dict[str, Any]] = []

    length = len(_visible(reference))
    constraints.append({"kind": "max_chars",
                        "params": {"count": length + rng.choice((6, 10, 16))}})
    if rng.random() < 0.5:
        constraints.append({"kind": "min_chars", "params": {"count": max(4, length - 6)}})

    # 必含词只能取"参考回答里真的有"的那些，否则约束当场就无解
    available = [word for word in topic["words"] if word in reference]
    if available:
        wanted = rng.sample(available, k=min(len(available), rng.choice((1, 2))))
        constraints.append({"kind": "contains", "params": {"values": sorted(wanted)}})

    banned = [word for word in BANNED if word not in reference]
    if banned and rng.random() < 0.5:
        constraints.append({"kind": "forbids", "params": {"values": [rng.choice(banned)]}})

    if form == "lines":
        count = len(topic["lines"])
        if split == "line":
            # 按换行分条时，"几条"就是"几行"：用 line_count，别再叠一条同义的
            # items_between——两条考同一件事会让 micro 口径被重复计权
            constraints.append({"kind": "line_count", "params": {"count": count, "blank": False}})
        else:
            constraints.append({"kind": "items_between",
                                "params": {"min": count, "max": count, "split": split}})
        constraints.append({"kind": "no_markdown", "params": {}})
    elif form == "json":
        constraints.append({"kind": "json_object", "params": {"allow_array": False}})
    else:
        share = _zh_share(reference)
        constraints.append({"kind": "zh_share_min",
                            "params": {"share": round(max(0.5, share - 0.05), 2)}})

    # JSON 形态不再叠 prefix：它的前两个字符必然是 `{"`，那是一条永远满足的假约束
    if form != "json" and rng.random() < 0.4:
        constraints.append({"kind": "prefix", "params": {"value": reference.strip()[:2]}})

    # 约束数封顶：题越复杂越说不清"到底哪条把它难住了"，而且 6 条以上里必然有冗余
    # （`min_chars` 与 `max_chars` 同时出现时，掉分往往来自字数而不是内容）
    _trim(constraints, MAX_CONSTRAINTS, rng)
    rng.shuffle(constraints)
    tags = [form, topic["id"]] + ([f"split-{split}"] if form == "lines" else [])
    return _case(topic, constraints, reference, tags,
                 note="约束参数由参考回答实测派生（长度/条目数/可用词），所以这道题必然有解")


def _make_hard(hard: dict[str, Any]) -> dict[str, Any]:
    """人工难例：约束彼此紧张但参考回答确实满足它们（由测试逐条核对）。"""
    return _case(hard["topic"], hard["constraints"], hard["reference"],
                 ["hard", hard["topic"]["id"], *hard.get("extra_tags", ())], note=hard["note"])


def _reference(topic: dict[str, Any], form: str, split: str) -> str:
    """参考回答。

    分条形态按**声明的那个分隔符**拼，所以 `items_between` 的条数与它同源；
    刻意不含 markdown 列表符，`no_markdown` 于是对它必然成立。
    """
    lines = list(topic["lines"])
    if form == "lines":
        if split == "line":
            return "\n".join(lines)
        if split == "sentence":
            return "".join(f"{line}。" for line in lines)
        return split.join(lines) + "。"
    if form == "json":
        return json.dumps(topic["object"], ensure_ascii=False)
    return f"{lines[0]}，{lines[1]}。"


def _case(topic: dict[str, Any], constraints: list[dict[str, Any]], reference: str,
          tags: Sequence[str], *, note: str) -> dict[str, Any]:
    return {
        "input": {"text": render_instruction(topic["ask"], constraints)},
        "expect": {"constraints": constraints},
        "kind": "instruction",
        "tags": list(dict.fromkeys(tags)),
        "meta": {"reference": reference, "topic": topic["id"], "note": note},
    }


def render_instruction(ask: str, constraints: Sequence[dict[str, Any]]) -> str:
    """把约束渲染回中文提示语——**凡是考模型的都必须说得出**。

    这句话是本数据集的地基：如果约束在提示语里没有对应的说法，
    那它考的不是"听话"而是"猜话"，与 S30 里"封闭词表没给模型"是同一种错。
    """
    phrases = [phrase_for(c["kind"], c.get("params") or {}) for c in constraints]
    return f"请{ask}。要求：{'；'.join(phrases)}。"


def phrase_for(kind: str, params: dict[str, Any]) -> str:
    if kind == "max_chars":
        return f"全文不超过 {params['count']} 个字（不含空格换行）"
    if kind == "min_chars":
        return f"全文不少于 {params['count']} 个字"
    if kind == "contains":
        return "必须包含" + "、".join(f"「{v}」" for v in params["values"])
    if kind == "forbids":
        return "不要出现" + "、".join(f"「{v}」" for v in params["values"])
    if kind == "items_between":
        low, high = params["min"], params["max"]
        span = f"{low} 条" if low == high else f"{low}–{high} 条"
        unit = "行" if params["split"] == "line" else "条"
        return f"分 {span}来说，{unit}之间用{_split_name(params['split'])}分开"
    if kind == "line_count":
        return f"正好 {params['count']} 行"
    if kind == "zh_share_min":
        return f"用中文回答（汉字占比至少 {params['share']:.0%}）"
    if kind == "json_object":
        return "只输出一个 JSON 对象，不要任何解释"
    if kind == "prefix":
        return f"以「{params['value']}」开头"
    if kind == "no_markdown":
        return "不要用 markdown 的列表符号、标题井号或表格"
    raise KeyError(f"约束 {kind!r} 还没有对应的提示语写法；不能默默漏掉——那会让它变成猜话")


def _split_name(split: str) -> str:
    return {"line": "换行", "；": "分号", "、": "顿号", "sentence": "句号"}.get(split, split)


#: 人工难例：紧张但仍可满足的组合。`reference` 由测试逐条核对每条约束
HARD: tuple[dict[str, Any], ...] = (
    {
        "topic": TOPICS[0],
        "reference": "退款七天内原路退回。",
        "constraints": [
            {"kind": "max_chars", "params": {"count": 14}},
            {"kind": "contains", "params": {"values": ("七天内", "原路退回")}},
            {"kind": "forbids", "params": {"values": ("抱歉",)}},
        ],
        "note": "上限只有 14 个字，两个必含词就占掉 9 个：考的是压缩而不是抄写",
    },
    {
        "topic": TOPICS[1],
        "reference": "周三两点开会；地点在会议室；每人十五分钟。",
        "constraints": [
            {"kind": "items_between", "params": {"min": 3, "max": 3, "split": "；"}},
            {"kind": "contains", "params": {"values": ("周三", "会议室", "十五分钟")}},
            {"kind": "no_markdown", "params": {}},
        ],
        "note": "三个必含词 + 恰好三条 + 不许列表符：条数与内容互相挤",
    },
    {
        "topic": TOPICS[4],
        "reference": json.dumps({"sell": False, "mask": "脱敏", "export": True},
                                ensure_ascii=False),
        "constraints": [
            {"kind": "json_object", "params": {"allow_array": False}},
            {"kind": "contains", "params": {"values": ("脱敏",)}},
            {"kind": "max_chars", "params": {"count": 46}},
        ],
        "note": "JSON 形态还要含中文词：布尔值写成 false 而不是「不会」，容易漏掉必含词",
    },
)


def _visible(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _zh_share(text: str) -> float:
    visible = _visible(text)
    return len(re.findall(r"[㐀-䶿一-鿿]", visible)) / len(visible) if visible else 0.0


def _case_id(instruction: str, constraints: list[dict[str, Any]]) -> str:
    """稳定 id 由**内容**算出：改了约束就是换了一道题，历史 grade 不该继续挂在这条 id 上。"""
    payload = json.dumps({"i": instruction, "c": constraints}, ensure_ascii=False, sort_keys=True)
    return f"ins-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def stats(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """数据集画像：每种约束的条数必须可见，否则"某类约束总违反"会被总分埋掉。"""
    from collections import Counter

    kinds = Counter(str(c["kind"]) for case in cases for c in case["expect"]["constraints"])
    return {
        "n": len(cases),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "constraints": sum(kinds.values()),
        "mean_constraints": round(sum(kinds.values()) / len(cases), 2) if cases else 0,
        "hard": sum(1 for case in cases if "hard" in case["tags"]),
        "unique_instructions": len({case["input"]["text"] for case in cases}),
    }


def to_jsonl(cases: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases)


def write_jsonl(path: Path | str, *, seed: int = 20261005) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_jsonl(build_cases(seed=seed)) + "\n", encoding="utf-8", newline="\n")
    return target


def unsatisfiable(cases: Sequence[dict[str, Any]] | None = None) -> list[tuple[str, list[str]]]:
    """考卷自检：参考回答满足不了自己的约束 ⇒ 这道题**无解**（或约束写坏了）。

    返回 [(case_id, 失败原因)]。生成器与随包文件都过一遍，
    所以手工改紧一个数字时，这条会当场指出来是哪道题、哪一条约束。
    """
    out: list[tuple[str, list[str]]] = []
    for case in cases if cases is not None else build_cases():
        reference = str((case.get("meta") or {}).get("reference") or "")
        failures = [
            f"{result.kind}: {result.detail}"
            for result in check_all(case["expect"]["constraints"], reference)
            if result.violated
        ]
        if failures:
            out.append((case["id"], failures))
    return out


BUILTIN_PATH = Path(__file__).resolve().parent / "instructions_zh.jsonl"

if __name__ == "__main__":  # pragma: no cover
    print(f"{write_jsonl(BUILTIN_PATH)} → {stats(build_cases())}")
