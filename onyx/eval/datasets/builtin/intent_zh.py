"""中文意图识别数据集生成器（`intent_zh.jsonl` 的来源）。

为什么是生成器而不是直接提交一份 JSONL：
一份来历不明的 JSONL 无法审计——不知道标签是怎么定的、有没有重复、
难例是不是被无意漏掉。生成器把"这份数据是怎么来的"变成可读的代码，
换 seed 就能重出一份，扩充也只是加模板。

四条刻意的设计：
1. **标签集含 `其他`**：没有兜底类时，模型面对越界输入只能硬塞进四个类之一，
   于是"越界标签率"永远测不出来，而它恰恰是幻觉的主要信号。
2. **难例单独打 tag**：`hard` 子集可以单独跑，看模型在边界上的表现，
   而不是被大量简单样本稀释成一个好看的平均分。
3. **同一模板配不同槽位**：只测"换个说法还认不认得"，
   避免数据集变成"记住 200 个固定句子"。
4. **每条都有 `ord`**：`--limit 20` 取的是前 20 条，所以顺序必须稳定，
   否则两次跑的子集不同，分数差异无法解释。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

LABELS: tuple[str, ...] = ("转账", "查余额", "投诉", "其他")

#: 槽位值。放在模板之前：模板在**模块导入时**就要引用它们
AMOUNTS = ("500元", "2000块", "一万", "300", "88.5元", "五千", "1500", "两万三")
NAMES = ("张伟", "李娜", "王芳", "我妈", "房东", "小李", "老周", "我妹")
ACCOUNTS = ("储蓄卡", "信用卡", "活期账户", "工资卡", "定期账户", "二类户")
TARGETS = ("客服", "柜员", "这个APP", "电话催收", "营业厅", "在线客服", "催收部门")
TOPICS = ("秋天", "故乡", "AI", "猫", "远方", "童年")
CITIES = ("北京", "上海", "广州", "深圳", "武汉")
CITIES2 = ("成都", "杭州", "西安", "南京", "重庆")
APPS = ("美团", "饿了么", "这个软件", "手机银行", "小程序")
PHRASES = ("你好", "谢谢", "请问洗手间在哪", "我迷路了", "这个多少钱")

#: 每个标签的模板。`{slot}` 会被槽位值替换。
TEMPLATES: dict[str, tuple[tuple[str, dict[str, Sequence[str]]], ...]] = {
    "转账": (
        ("帮我把{amount}转给{name}", {"amount": AMOUNTS, "name": NAMES}),
        ("我想给{name}转{amount}", {"amount": AMOUNTS, "name": NAMES}),
        ("从我的账户转{amount}到{name}的卡上", {"amount": AMOUNTS, "name": NAMES}),
        ("{name}急用钱，赶紧转{amount}过去", {"amount": AMOUNTS, "name": NAMES}),
        ("转账{amount}给{name}", {"amount": AMOUNTS, "name": NAMES}),
        ("给{name}打{amount}过去，备注{note}", {"name": NAMES, "amount": AMOUNTS,
                                              "note": ("房租", "还款", "货款", "生活费")}),
        ("我要往{name}的账户汇款{amount}", {"amount": AMOUNTS, "name": NAMES}),
        ("麻烦把上个月的{amount}转给{name}", {"amount": AMOUNTS, "name": NAMES}),
    ),
    "查余额": (
        ("我的账户还有多少钱", {}),
        ("帮我查一下余额", {}),
        ("卡里还剩{amount}吗", {"amount": AMOUNTS}),
        ("看看我{account}的可用余额", {"account": ACCOUNTS}),
        ("查询{name}账户的余额", {"account": ACCOUNTS, "name": NAMES}),
        ("我想知道现在账上有多少", {}),
        ("余额是多少", {}),
        ("帮我看看{account}里还有没有钱", {"account": ACCOUNTS}),
        ("账户里现在有多少钱", {}),
        ("我这张{account}还有余额吗", {"account": ACCOUNTS}),
        ("查一下我名下所有账户的余额", {}),
    ),
    "投诉": (
        ("我要投诉{target}", {"target": TARGETS}),
        ("{target}态度太差了，我要反映一下", {"target": TARGETS}),
        ("怎么投诉{target}？", {"target": TARGETS}),
        ("你们{target}把我多扣了{amount}，必须给个说法", {"target": TARGETS, "amount": AMOUNTS}),
        ("我要举报{target}的违规行为", {"target": TARGETS}),
        ("对{target}非常不满意，找谁申诉", {"target": TARGETS}),
        ("{target}的问题拖了三天没人管，我要投诉", {"target": TARGETS}),
        ("给我转人工，我要投诉{target}", {"target": TARGETS}),
    ),
    "其他": (
        ("今天天气怎么样", {}),
        ("帮我写一首关于{topic}的诗", {"topic": TOPICS}),
        ("{city}到{city2}的高铁要多久", {"city": CITIES, "city2": CITIES2}),
        ("推荐几部好看的电影", {}),
        ("怎么用{app}点外卖", {"app": APPS}),
        ("翻译成英文：{phrase}", {"phrase": PHRASES}),
        ("{amount}等于多少美元", {"amount": AMOUNTS}),
        ("给我讲个笑话", {}),
        ("{city}今天限行吗", {"city": CITIES}),
        ("帮我订一张去{city2}的机票", {"city2": CITIES2}),
        ("这道数学题怎么做", {}),
    ),
}

#: 人工补充的难例：这些是**真实会出错**的边界形态，模板生成不出来
HARD_CASES: tuple[tuple[str, str], ...] = (
    # 一句话里同时提到两个意图 —— 标签取主诉求
    ("我想查下余额，顺便把500转给张伟", "转账"),
    ("先看看还剩多少钱，够的话就转2000给李娜", "转账"),
    ("查余额", "查余额"),
    # 口语化、省略主语
    ("还有钱没", "查余额"),
    ("转过去", "转账"),
    ("这什么破玩意儿", "投诉"),
    # 含否定，容易被"投诉"关键词带跑
    ("我不是要投诉，只是想问下余额", "查余额"),
    ("不用转了，我查一下余额就行", "查余额"),
    # 含"投诉"字样但其实是问流程 —— 仍属投诉意图
    ("投诉的入口在哪里", "投诉"),
    # 越界但很像金融场景 —— 正确答案是「其他」
    ("帮我算一下房贷利率", "其他"),
    ("股票今天涨了吗", "其他"),
    ("怎么开通网上银行", "其他"),
    # 空泛输入
    ("在吗", "其他"),
    ("你好", "其他"),
    # 多意图，且第二个才是主诉求
    ("我投诉过客服了，现在你帮我查下余额", "查余额"),
    ("余额不用查了，直接转3000给王芳", "转账"),
    # 情绪强烈但诉求明确
    ("气死我了！转个账转了三次都失败！", "转账"),
    ("你们再这样我就投诉到银保监会", "投诉"),
    # 疑问句式包裹的指令
    ("能不能帮我看看余额？", "查余额"),
    ("可以转账吗？给小李转500", "转账"),
    # 与银行无关但含金融词
    ("比特币现在多少钱一个", "其他"),
    ("帮我写一份关于转账手续费的调研报告", "其他"),
    ("贷款需要哪些材料", "其他"),
    # 极短输入
    ("余额", "查余额"),
    ("转账", "转账"),
    ("投诉", "投诉"),
    # 中英混杂
    ("check一下我的balance", "查余额"),
    ("transfer 500 to 张伟", "转账"),
    # 上下文缺失，无法判定 → 兜底类
    ("那个怎么办", "其他"),
    ("帮我处理一下", "其他"),
    # 长句里意图在结尾
    ("我昨天去了趟营业厅，排队两小时，办完之后发现卡里钱不对，我要投诉", "投诉"),
    ("我妈说她没收到钱，你帮我看看是不是转账失败了，再转一次500给她", "转账"),
)


def _slots_combinations(slots: dict[str, Sequence[str]], seed: int) -> Iterator[dict[str, str]]:
    """按模板生成若干槽位组合。

    不做笛卡尔积（会爆炸到几万条且高度重复），而是**按索引配对**再随机打散：
    每个模板产出 `max(len(槽位值))` 条，既覆盖了不同说法，又不产生近似重复。
    """
    if not slots:
        yield {}
        return
    keys = sorted(slots)
    width = max(len(slots[key]) for key in keys)
    rng = random.Random(seed)
    orders = {key: rng.sample(range(len(slots[key])), len(slots[key])) for key in keys}
    for index in range(width):
        yield {key: slots[key][orders[key][index % len(orders[key])]] for key in keys}


def build_cases(*, seed: int = 20261003) -> list[dict[str, Any]]:
    """生成全部样本。同 seed 必然同结果——数据集不可复现，评测就不可复现。"""
    cases: list[dict[str, Any]] = []
    for label_index, (label, templates) in enumerate(TEMPLATES.items()):
        for template_index, (template, slots) in enumerate(templates):
            for combo_index, values in enumerate(
                _slots_combinations(dict(slots), seed + label_index * 100 + template_index)
            ):
                text = template.format(**values) if values else template
                cases.append({
                    "input": {"instruction": text},
                    "expect": {"label": label},
                    "kind": "single",
                    "tags": ["template", label],
                    "meta": {"template": template, "label_index": label_index,
                             "template_index": template_index, "combo_index": combo_index},
                })
    for index, (text, label) in enumerate(HARD_CASES):
        cases.append({
            "input": {"instruction": text},
            "expect": {"label": label},
            "kind": "single",
            "tags": ["hard", label],
            "meta": {"hard_index": index},
        })

    # 稳定 id：由内容算出，所以重跑生成器不会让 case_id 变化，历史 grade 仍能对上
    for case in cases:
        case["id"] = _case_id(case["input"]["instruction"], case["expect"]["label"])
    # 打散顺序但保持可复现：`--limit 20` 取到的必须是各类混合，不能全是「转账」
    random.Random(seed).shuffle(cases)
    for ord_index, case in enumerate(cases):
        case["ord"] = ord_index
    return cases


def _case_id(instruction: str, label: str) -> str:
    digest = hashlib.sha256(f"{instruction}|{label}".encode()).hexdigest()[:16]
    return f"izh-{digest}"


def to_jsonl(cases: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases)


def stats(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """数据集画像。类别分布必须能被看见——极度不均衡时 accuracy 会骗人。"""
    from collections import Counter

    labels = Counter(case["expect"]["label"] for case in cases)
    return {
        "n": len(cases),
        "labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        "hard": sum(1 for case in cases if "hard" in case["tags"]),
        "unique_instructions": len({case["input"]["instruction"] for case in cases}),
    }


def write_jsonl(path: Path | str, *, seed: int = 20261003) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases(seed=seed)
    target.write_text(to_jsonl(cases) + "\n", encoding="utf-8", newline="\n")
    return target


BUILTIN_PATH = Path(__file__).resolve().parent / "intent_zh.jsonl"

if __name__ == "__main__":  # pragma: no cover
    written = write_jsonl(BUILTIN_PATH)
    print(f"{written} → {stats(build_cases())}")
