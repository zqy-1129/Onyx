"""中文同义/反义对的向量检索数据集（`embeddings_zh.jsonl` 的来源，S33）。

关系是**派生出来的，不是人标的**——这样每条 gold 都能被机械检查：

    query   = "{主体}{谓语[0]}{客体}"      评审会推迟到下午三点
    gold    = "{主体}{谓语[1]}{客体}"      评审会延后至下午三点   ← 同义，换了词
    anti    = "{主体}{否定}{谓语[0]}{客体}" 评审会没有推迟到下午三点 ← 反义，几乎同词面
    其余    = 别的 frame 的 query          ← 无关，实体不重叠

四件事必须先量清楚，否则分数会撒谎（2026-10-06 在 qwen3-embedding:0.6b 上实测）：

1. **不许用相似度阈值当判据**：同义对 cos 0.749–0.909，反义对 0.649–0.796，**区间重叠**——
   "离得近"完全可能是"意思相反"。所以判据是排序（gold 在候选池里排第几）。
2. 反义项与 query **共享几乎全部词面**，所以"词面重叠就靠前"的模型会把它排第一。
   这一位（`anti_first_rate`）才是 embedding 模型的真短板，比 recall@1 更值得看。
3. gold 与 query **不许字面相同**：一样就变成字符串匹配而不是检索，
   `same_string_pairs()` 把这条钉成断言。
4. 候选池大小固定、每条 case 一个请求：分母与成本都要能对上。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: 否定标记（anti 里必须出现、query/gold 里必须不出现）。逐 frame 核对，不猜。
NEGATORS: tuple[str, ...] = ("不", "未", "没有", "禁止")

#: 每个 frame 是一组可机械展开的同义/反义三元组。三句都用模板拼出来：
#: `"{subject}" + 谓语 + "{object}"`——谓语有三个写法，命题只有一个。
#: - `p_query` / `p_para`：**同一命题的两种写法**（这才是同义 gold）；
#: - `p_anti`：**带否定的写法**，与 query 几乎同词面而极性相反；
#:   它刻意不靠"机械加否定前缀"生成（那样会产出"无需需要"这种句子，
#:   而怪句子会把测量从"语义"带偏成"通顺度"）。
#: - `negator`：`p_anti` 里那个否定标记，自检逐条核对它只出现在 anti 里。
FRAMES: tuple[dict[str, Any], ...] = (
    {"id": "mtg-delay", "topic": "会议", "subject": "评审会", "object": "下午三点",
     "p_query": "推迟到", "p_para": "延后至", "p_anti": "没有推迟到", "negator": "没有"},
    {"id": "mtg-reschedule", "topic": "会议", "subject": "例会", "object": "本周五",
     "p_query": "改期到", "p_para": "顺延至", "p_anti": "没有改期到", "negator": "没有"},
    {"id": "ins-need", "topic": "巡检", "subject": "主泵", "object": "每周巡检",
     "p_query": "需要", "p_para": "须安排", "p_anti": "不需要", "negator": "不"},
    {"id": "ins-pass", "topic": "巡检", "subject": "冷却塔", "object": "合格标准",
     "p_query": "达到", "p_para": "满足", "p_anti": "未达到", "negator": "未"},
    {"id": "fin-approve", "topic": "报销", "subject": "这张单据", "object": "财务复核",
     "p_query": "通过", "p_para": "获批", "p_anti": "未通过", "negator": "未"},
    {"id": "fin-limit", "topic": "报销", "subject": "差旅额度", "object": "上限",
     "p_query": "超出", "p_para": "突破", "p_anti": "未超出", "negator": "未"},
    {"id": "whs-stock", "topic": "仓储", "subject": "液压阀", "object": "安全库存",
     "p_query": "低于", "p_para": "少于", "p_anti": "不低于", "negator": "不"},
    {"id": "whs-move", "topic": "仓储", "subject": "密封圈", "object": "B 号货架",
     "p_query": "转入", "p_para": "移至", "p_anti": "没有转入", "negator": "没有"},
    {"id": "sft-wear", "topic": "安全", "subject": "外协队伍", "object": "防护装备",
     "p_query": "必须佩戴", "p_para": "须按规定戴", "p_anti": "禁止佩戴", "negator": "禁止"},
    {"id": "sft-entry", "topic": "安全", "subject": "无关人员", "object": "配电间",
     "p_query": "进入了", "p_para": "闯入过", "p_anti": "没有进入", "negator": "没有"},
    {"id": "doc-file", "topic": "归档", "subject": "这批图纸", "object": "档案室",
     "p_query": "已送交", "p_para": "已移交", "p_anti": "未送交", "negator": "未"},
    {"id": "doc-sign", "topic": "归档", "subject": "验收单", "object": "班组长签字",
     "p_query": "缺少", "p_para": "尚无", "p_anti": "并不缺少", "negator": "不"},
)

#: 每条 case 的候选池里放几条无关句（连同 gold 与 anti ⇒ 池大小 = UNRELATED + 2）
UNRELATED = 2

#: 判"实体是否被包含"用的最小长度：单字不算实体，避免"上/下"这类误命中
_MIN_ENTITY = 2


def query_of(frame: dict[str, Any]) -> str:
    return f"{frame['subject']}{frame['p_query']}{frame['object']}"


def paraphrase_of(frame: dict[str, Any]) -> str:
    return f"{frame['subject']}{frame['p_para']}{frame['object']}"


def antonym_of(frame: dict[str, Any]) -> str:
    return f"{frame['subject']}{frame['p_anti']}{frame['object']}"


def build_cases(*, seed: int = 20261006) -> list[dict[str, Any]]:
    """每个 frame 一条 case：query + [gold, anti, 无关×N]，池内顺序由 seed 决定。"""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    for frame in FRAMES:
        others = [other for other in FRAMES if other["id"] != frame["id"]]
        unrelated = [query_of(other) for other in rng.sample(others, k=UNRELATED)]
        pool = [
            {"text": paraphrase_of(frame), "relation": "paraphrase"},
            {"text": antonym_of(frame), "relation": "antonym"},
            *[{"text": text, "relation": "unrelated"} for text in unrelated],
        ]
        rng.shuffle(pool)
        gold = next(i for i, item in enumerate(pool) if item["relation"] == "paraphrase") + 1
        anti = next(i for i, item in enumerate(pool) if item["relation"] == "antonym") + 1
        cases.append({
            "id": "",  # 下面按内容算
            "input": {"query": query_of(frame), "candidates": [item["text"] for item in pool]},
            "expect": {"gold": gold, "antonym": anti},
            "kind": "embed",
            "tags": [str(frame["topic"]), f"pool-{len(pool)}"],
            "meta": {
                "frame": str(frame["id"]), "topic": str(frame["topic"]),
                "entities": [str(frame["subject"]), str(frame["object"])],
                "p_query": str(frame["p_query"]), "p_para": str(frame["p_para"]),
                "p_anti": str(frame["p_anti"]), "negator": str(frame["negator"]),
                "pool_size": len(pool),
                "relations": [item["relation"] for item in pool],
                "texts": [item["text"] for item in pool],
            },
        })
    for index, case in enumerate(cases):
        case["id"] = _case_id(case["input"], case["expect"])
        case["ord"] = index
    return cases


def same_string_pairs(cases: Sequence[dict[str, Any]] | None = None) -> list[str]:
    """考卷自检：gold 不许与 query 字面相同。

    一样的话"检索"就退化成字符串比较——任何模型都能拿满分，而这正是 S32 学到的
    "分数太好也是缺陷信号"。返回违规的 case id。
    """
    out: list[str] = []
    for case in cases if cases is not None else build_cases():
        pool = case["meta"]["texts"]
        gold_index = int(case["expect"]["gold"]) - 1
        if str(pool[gold_index]) == str(case["input"]["query"]):
            out.append(str(case["id"]))
    return out


def broken_relations(cases: Sequence[dict[str, Any]] | None = None) -> list[str]:
    """考卷自检：三种关系的构造承诺逐条核对。

    - gold/anti 都必须含全部实体（否则比的是话题而不是命题）；
    - 否定标记只能出现在 anti 里（query 带否定 ⇒ anti 不是反义而是同义）；
    - gold 必须用另一个谓语写法（没换词就退化成字符串匹配，见 `same_string_pairs`）；
    - 无关项不许与本 frame 共享全部实体（否则它偷着做了 gold 的活）。
    """
    out: list[str] = []
    for case in cases if cases is not None else build_cases():
        meta = case["meta"]
        entities = [str(e) for e in meta["entities"]]
        query = str(case["input"]["query"])
        negator = str(meta["negator"])
        lemma = str(meta["p_query"])
        synonym = str(meta["p_para"])
        for index, (text, relation) in enumerate(
            zip(meta["texts"], meta["relations"], strict=True), start=1
        ):
            missing = [e for e in entities if len(e) >= _MIN_ENTITY and e not in str(text)]
            if relation in ("paraphrase", "antonym") and missing:
                out.append(f"{case['id']}#{index} {relation} 丢了实体 {missing}")
            if relation == "unrelated" and not missing:
                out.append(f"{case['id']}#{index} 无关项却含全部实体，等于送分")
            if negator not in NEGATORS:
                out.append(f"{case['id']} 声明的否定标记 {negator} 不在已知清单里，自检会空转")
            has_neg = negator in str(text)
            if relation == "antonym" and not has_neg:
                out.append(f"{case['id']}#{index} 反义句里没有否定标记 {negator}")
            if relation in ("paraphrase", "query") and has_neg:
                out.append(f"{case['id']}#{index} 同义句不该带否定 {negator}")
            if relation == "paraphrase" and (synonym not in str(text) or lemma in str(text)):
                out.append(f"{case['id']}#{index} 同义句没换谓语（{lemma}/{synonym}）")
        if negator in query:
            out.append(f"{case['id']} query 自带否定 {negator}，反义项就不是反义了")
        if synonym in query:
            out.append(f"{case['id']} query 与 gold 用了同一个谓语 {synonym}")
    return out


def gold_is_unique(cases: Sequence[dict[str, Any]] | None = None) -> list[str]:
    """每个候选池恰好一个 paraphrase：两个就是题面歧义，零个就是无解题。"""
    out: list[str] = []
    for case in cases if cases is not None else build_cases():
        relations = list(case["meta"]["relations"])
        if relations.count("paraphrase") != 1 or relations.count("antonym") != 1:
            out.append(str(case["id"]))
    return out


def stats(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_topic: dict[str, int] = {}
    for case in cases:
        by_topic[str(case["meta"]["topic"])] = by_topic.get(str(case["meta"]["topic"]), 0) + 1
    pools = {int(case["meta"]["pool_size"]) for case in cases}
    return {
        "n": len(cases),
        "frames": len({str(case["meta"]["frame"]) for case in cases}),
        "topics": dict(sorted(by_topic.items())),
        "pool_sizes": sorted(pools),
        "uniform_pool": len(pools) == 1,
        "inputs_per_case": (max(pools) + 1) if pools else 0,
    }


def _case_id(payload: dict[str, Any], expect: dict[str, Any]) -> str:
    blob = json.dumps({"p": payload, "e": expect}, ensure_ascii=False, sort_keys=True)
    return f"em-{hashlib.sha256(blob.encode()).hexdigest()[:16]}"


def to_jsonl(cases: Sequence[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(case, ensure_ascii=False, sort_keys=True) for case in cases)


def write_jsonl(path: Path | str, *, seed: int = 20261006) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_jsonl(build_cases(seed=seed)) + "\n", encoding="utf-8", newline="\n")
    return target


BUILTIN_PATH = Path(__file__).resolve().parent / "embeddings_zh.jsonl"

if __name__ == "__main__":  # pragma: no cover
    print(f"{write_jsonl(BUILTIN_PATH)} → {stats(build_cases())}")
