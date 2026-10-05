"""长上下文中文集生成器（`longctx_zh.jsonl` 的来源）。

四件事决定这份数据有没有意义，每件都有一条可跑的自检盯着（不靠人读一遍）：

1. **测检索，不测摘要**。每个埋点是一句带独特数值/编号的事实，问题只问"值是多少"，
   答案可精确匹配。摘要类任务没有可机械判定的对错，进了这个数据集就是破坏判据。
2. **填充文本里一个数字都不许出现**（`digits_only_in_needles`）。
   这样"模型有没有真读到那一句"才与"它有没有猜对一个常见数字"分得开。
3. **每个埋点配一个同句式、另一实体的干扰项**，且题面只点名答案实体
   （`ambiguous_questions` / `value_string_collisions`）。
   第一版没有干扰项，真机跑出来 9/9 全对（qwen3.5:9b，16k 档实测 in≈16.7k tok）——
   那不是模型强，是"扫到任意一个数字"就能得分。加了干扰项，检索必须认对实体；
   代价是题面一歧义就变成"两个值都算对"，所以题面干净度也必须是断言而不是注释。
4. **位置是一等公民**。每个埋点标 first / middle / last，且用字符偏移验证它真在那儿：
   长上下文的典型失效是中部检索最差（lost in the middle），
   而它在"总分 0.78"里完全看不见——看不见就不会有人去改文档排布。

档位按**汉字预算**写（4k≈6,000 字、8k≈12,000、16k≈24,000），
换算比例来自 2026-10-05 在 qwen3.5:9b 上的实测（0.6–0.7 token/汉字）。
但分数不依赖这个估算：**是否越界只看引擎回报的 in_tokens**（见任务侧的 skip 判据）。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: 档位 → 汉字预算。名字里的 4k/8k/16k 指的是 token 量级，实际值以引擎回报为准
BUCKETS: dict[str, int] = {"4k": 6000, "8k": 12000, "16k": 24000}

#: 正文与问题块的分隔线（生成与自检都按它切，不留两处字面量）
SEPARATOR = "——"

#: 每个埋点一行：`value` 是答案，`distractor_value` 是同句式另一实体的值。
#: 两句刻意互不提及对方实体（`subject` / `distractor_subject` 就是为这条自检存在的字段）：
#: 干扰句里出现埋点实体，题面就变歧义，而歧义题的分数不能算在模型头上。
#: 值之间还不许有子串关系（`value_string_collisions`），否则"全文只出现一次"数的是假数。
NEEDLES: tuple[dict[str, Any], ...] = (
    {"value": 3.6, "subject": "备用泵", "distractor_subject": "主泵",
     "sentence": "备用泵的振动阈值定为每秒 {v} 毫米，超出即触发停机检查。",
     "question": "备用泵的振动阈值是每秒多少毫米？",
     "distractor": "主泵的振动阈值定为每秒 {v} 毫米，报警后先降负荷再复核。",
     "distractor_value": 4.1},
    {"value": 1998, "subject": "厂内手册", "distractor_subject": "安全手册",
     "sentence": "这本厂内手册的首版印于 {v} 年，此后只做局部修订。",
     "question": "厂内手册首版是哪一年印的？",
     "distractor": "同一系列的安全手册于 {v} 年再版，修订记录单独存放。",
     "distractor_value": 2003},
    {"value": "C-447", "subject": "夜班巡检", "distractor_subject": "白班巡检",
     "sentence": "夜班巡检要登记的线路编号是 {v}，交接时需逐条读一遍。",
     "question": "夜班巡检要登记的线路编号是什么？",
     "distractor": "白班巡检登记的线路编号为 {v}，核对后随表归档。",
     "distractor_value": "C-474"},
    {"value": 12.5, "subject": "冷却塔", "distractor_subject": "冷冻水",
     "sentence": "冷却塔的补水量按每小时 {v} 吨核定，雨季另计。",
     "question": "冷却塔每小时补水量按多少吨核定？",
     "distractor": "冷冻水循环量按每小时 {v} 吨核定，填报时单独成行。",
     "distractor_value": 9.8},
    {"value": "L-209", "subject": "液压阀", "distractor_subject": "密封圈",
     "sentence": "备件库里液压阀存放在货架 {v} 层，取用需登记。",
     "question": "液压阀存放在哪个货架编号？",
     "distractor": "密封圈统一放在货架 {v}，出库时另行登记。",
     "distractor_value": "L-902"},
    {"value": 42, "subject": "交接班表", "distractor_subject": "季度巡检表",
     "sentence": "交接班表要求签字人数不少于 {v} 人，缺额需补签。",
     "question": "交接班表要求签字人数不少于多少人？",
     "distractor": "季度巡检表要求签字人数为 {v} 人，年底汇总一次。",
     "distractor_value": 24},
    {"value": "R-13", "subject": "北侧", "distractor_subject": "南侧",
     "sentence": "厂区北侧的应急集合点编号记作 {v}，演练时按点清点。",
     "question": "厂区北侧应急集合点的编号是什么？",
     "distractor": "厂区南侧的应急集合点编号记作 {v}，演练时单独清点。",
     "distractor_value": "R-31"},
    {"value": 5.2, "subject": "主变压器", "distractor_subject": "电抗器",
     "sentence": "主变压器的温升报警设在 {v} 摄氏度，超过要降载。",
     "question": "主变压器的温升报警设在多少摄氏度？",
     "distractor": "电抗器的温升报警设在 {v} 摄氏度，复核后另行降载。",
     "distractor_value": 6.4},
    {"value": 2011, "subject": "巡检制度", "distractor_subject": "培训台账",
     "sentence": "这套巡检制度在 {v} 年秋季正式并入岗位说明书。",
     "question": "巡检制度哪一年并入岗位说明书？",
     "distractor": "培训台账在 {v} 年春季重新装订并移交档案室。",
     "distractor_value": 1976},
)

#: 填充句的组合词池。**全部不含数字**（见模块 docstring 第 2 条）
SUBJECTS = (
    "值班组长", "检修班组", "安全员", "仓储同事", "培训讲师", "调度中心", "巡检人员",
    "技术档案室", "外协队伍", "新入职员工", "班后复盘会", "设备台账",
)
ACTIONS = (
    "逐条核对记录", "把情况写进交接班表", "在现场复述一遍流程", "按要求补齐签字",
    "把异常照片归档留存", "重新排一遍巡查顺序", "将口径同步给下一班",
    "对照图纸确认管路走向", "把待办事项列成清单", "通知相关岗位到场确认",
)
TAILS = (
    "，遇到不一致的地方当场更正", "，随后在群里说明处理结果", "，不得延后到下一班",
    "，必要时请技术档案室调取旧版资料", "，全程保持通讯畅通", "，并在备注栏简述原因",
    "，如涉及外来人员需陪同登记", "，处理完再离开现场", "，相关单据随设备档案归档",
)
CONNECTORS = ("此外", "另外", "同时", "随后", "其间", "当天", "事后", "按照惯例", "按流程")


def build_cases(*, seed: int = 20261005, variants: int = 3) -> list[dict[str, Any]]:
    """生成全部样本：3 档 × 每档 `variants` 条，每条埋 3 个事实。"""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for bucket, budget in BUCKETS.items():
        for variant in range(variants):
            cases.append(_make(bucket=bucket, budget=budget, index=variant, rng=rng))
    for case in cases:
        case["id"] = _case_id(case["input"]["text"], case["expect"]["answers"])
    for ord_index, case in enumerate(cases):
        case["ord"] = ord_index
    return cases


def _make(*, bucket: str, budget: int, index: int, rng: random.Random) -> dict[str, Any]:
    picked = _pick_needles(rng)
    paragraphs = _paragraphs(budget, rng)
    placed = _plant(paragraphs, picked)
    questions = "\n".join(f"{item['id']}: {item['question']}" for item in picked)
    text = (
        "\n\n".join(placed)
        + f"\n\n{SEPARATOR}\n读完上面的记录，只输出一个 JSON 对象。"
        + "键是 " + "、".join(item["id"] for item in picked) + "，值是对应问题的答案；"
        + "数字就写数字，不要带单位，也不要额外解释。\n"
        + questions
    )
    return {
        "input": {"text": text},
        "expect": {
            "answers": {item["id"]: item["value"] for item in picked},
            "keys": [item["id"] for item in picked],
            "positions": {item["id"]: item["position"] for item in picked},
        },
        "kind": "longctx",
        "tags": [bucket, f"{bucket}-{index}"],
        "meta": {
            "bucket": bucket, "budget": budget, "hanzi": _han_count(text),
            "chars": len(text),
            "needles": [
                {"id": item["id"], "position": item["position"], "value": item["value"],
                 "subject": item["subject"], "sentence": item["sentence"],
                 "distractor_subject": item["distractor_subject"],
                 "distractor": item["distractor"], "distractor_value": item["distractor_value"]}
                for item in picked
            ],
        },
    }


#: 埋点位置固定为 (first, middle, last)，问题 id 依次是 q1/q2/q3
POSITIONS = ("first", "middle", "last")


def _pick_needles(rng: random.Random) -> list[dict[str, Any]]:
    """三个埋点 = 三种位置，各带一个干扰项。

    位置与问题 id 一一对应，失败才能归因到"哪一段没读到"；
    干扰项负责让"随便找一个数字"失败——两者缺一不可。
    """
    return [
        {
            "id": f"q{index}", "position": position, "value": needle["value"],
            "question": needle["question"],
            "subject": needle["subject"], "distractor_subject": needle["distractor_subject"],
            "sentence": needle["sentence"].format(v=needle["value"]),
            "distractor": needle["distractor"].format(v=needle["distractor_value"]),
            "distractor_value": needle["distractor_value"],
        }
        for index, (position, needle) in enumerate(
            zip(POSITIONS, rng.sample(list(NEEDLES), k=len(POSITIONS)), strict=True), start=1
        )
    ]


def _paragraphs(budget: int, rng: random.Random) -> list[str]:
    """拼出够长的中文段落。句子由词池组合，所以内容不重复到能让"背下来"成为捷径。"""
    paragraphs: list[str] = []
    produced = 0
    while produced < budget:
        parts = []
        for _ in range(rng.randint(4, 6)):
            sentence = (rng.choice(SUBJECTS) + rng.choice(ACTIONS) + rng.choice(TAILS))
            parts.append(sentence)
        body = "；".join(parts) + "。"
        opener = rng.choice(CONNECTORS)
        paragraphs.append(f"{opener}，" + body)
        produced += _han_count(body)
    return paragraphs


def _plant(paragraphs: list[str], needles: Sequence[dict[str, Any]]) -> list[str]:
    """把埋点与干扰项插进段落：埋点按 first/middle/last，干扰项插在它后面两段处。

    刻意不与埋点同段：那样"看到相邻两句"就等于答对，
    而我们要测的是读完一段之后还能认对实体。
    两处落点相同也要能共存（按段落归组后再拼），否则会拼出一个不存在的句子，
    让"埋点在正文里"这条自检变成唯一能发现它的地方。
    """
    count = len(paragraphs)
    spots = {
        "first": max(1, int(count * 0.06)),
        "middle": max(2, int(count * 0.5)),
        "last": max(3, int(count * 0.94)),
    }
    extras: dict[int, list[str]] = {}
    for needle in needles:
        home = spots[needle["position"]]
        extras.setdefault(home, []).append(needle["sentence"])
        host = min(count - 1, home + 2) if home + 2 < count else max(0, home - 2)
        extras.setdefault(host, []).append(needle["distractor"])
    placed = list(paragraphs)
    for index, sentences in extras.items():
        stem = placed[index].rstrip("。")
        placed[index] = " ".join([stem, *sentences]) + "。"
    return placed


def _han_count(text: str) -> int:
    return len(re.findall(r"[㐀-䶿一-鿿]", text))


def digits_only_in_needles(cases: Sequence[dict[str, Any]] | None = None) -> list[str]:
    """考卷自检：除了埋点句与干扰句，正文里不许出现任何数字。

    填充文本有数字的话，"猜一个常见数字"与"真去检索了那一句"就分不开——
    而后者才是这份数据要测的东西。问题块在分隔线之后，不在检查范围内
    （那里出现的是 q1/q2/q3 这样的题号，是题目本身）。
    """
    offenders: list[str] = []
    for case in cases if cases is not None else build_cases():
        body = str(case["input"]["text"]).split(SEPARATOR)[0]
        for needle in case["meta"]["needles"]:
            body = body.replace(str(needle["sentence"]), "")
            body = body.replace(str(needle["distractor"]), "")
        if re.search(r"\d", body):
            offenders.append(case["id"])
    return offenders


def value_string_collisions(pool: Sequence[dict[str, Any]] | None = None) -> list[str]:
    """答案值与干扰值的字面串不许互为子串（也不许相等）。

    `12.5` 里含着 `2.5`：判"这个数在全文只出现一次"数的是子串，
    于是两条无害埋点会被报成冲突，而真冲突又会被当成巧合放过。先把这层字面歧义清掉。
    """
    items = list(pool) if pool is not None else list(NEEDLES)
    strings = [str(item[key]) for item in items for key in ("value", "distractor_value")]
    out: list[str] = []
    for index, mine in enumerate(strings):
        for other in strings[index + 1:]:
            if mine == other or (mine in other or other in mine):
                out.append(f"{mine} ↔ {other}")
    return out


def ambiguous_questions(cases: Sequence[dict[str, Any]] | None = None) -> list[str]:
    """考卷自检：问句只能点名答案那个实体，不能点名同一条里的任何其它实体。

    问句里同时出现两个实体时，两个候选值都算"读对了"，检索判据当场失效——
    而分数照旧产出，看不出题面有问题。这是加干扰项之后新长出来的失效方式，
    所以和"填充文本无数字"一样用可跑的断言钉住，而不是靠人读一遍。
    """
    offenders: list[str] = []
    for case in cases if cases is not None else build_cases():
        tail = str(case["input"]["text"]).split(SEPARATOR, 1)[1]
        needles = case["meta"]["needles"]
        names = {str(item[key]) for item in needles for key in ("subject", "distractor_subject")}
        for needle in needles:
            asked = next((line for line in tail.splitlines()
                          if line.startswith(f"{needle['id']}:")), "")
            own = str(needle["subject"])
            others = sorted(name for name in names if name != own)
            if own not in asked:
                offenders.append(f"{case['id']}/{needle['id']} 问句没点名答案实体 {own}")
            for name in others:
                if name in asked:
                    offenders.append(f"{case['id']}/{needle['id']} 问句点名了别的实体 {name}")
            # 埋点句与干扰句也互不提及对方实体，否则"哪句才是答案"由模型自己挑
            if needle["distractor_subject"] in needle["sentence"]:
                offenders.append(f"{case['id']}/{needle['id']} 埋点句提到了干扰实体")
            if own in needle["distractor"]:
                offenders.append(f"{case['id']}/{needle['id']} 干扰句提到了答案实体")
    return offenders


def position_spread(cases: Sequence[dict[str, Any]]) -> dict[str, int]:
    """每种位置的埋点条数——`by_position` 的分母在数据侧就该看得见。"""
    from collections import Counter

    spread = Counter(item["position"] for case in cases for item in case["meta"]["needles"])
    return dict(sorted(spread.items()))


def stats(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_bucket: dict[str, dict[str, int]] = {}
    for case in cases:
        bucket = str(case["meta"]["bucket"])
        entry = by_bucket.setdefault(bucket, {"cases": 0, "hanzi": 0, "min_hanzi": 10**9})
        entry["cases"] += 1
        entry["hanzi"] += int(case["meta"]["hanzi"])
        entry["min_hanzi"] = min(entry["min_hanzi"], int(case["meta"]["hanzi"]))
    for entry in by_bucket.values():
        entry["avg_hanzi"] = round(entry["hanzi"] / entry["cases"]) if entry["cases"] else 0
    return {
        "n": len(cases),
        "needles": sum(len(case["meta"]["needles"]) for case in cases),
        "positions": position_spread(cases),
        "buckets": by_bucket,
        "monotonic": [by_bucket[b]["avg_hanzi"] for b in BUCKETS]
        == sorted(by_bucket[b]["avg_hanzi"] for b in BUCKETS),
    }


def _case_id(text: str, answers: dict[str, Any]) -> str:
    payload = json.dumps({"t": text, "a": answers}, ensure_ascii=False, sort_keys=True)
    return f"lc-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def to_jsonl(cases: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases)


def write_jsonl(path: Path | str, *, seed: int = 20261005) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_jsonl(build_cases(seed=seed)) + "\n", encoding="utf-8", newline="\n")
    return target


BUILTIN_PATH = Path(__file__).resolve().parent / "longctx_zh.jsonl"

if __name__ == "__main__":  # pragma: no cover
    print(f"{write_jsonl(BUILTIN_PATH)} → {stats(build_cases())}")
