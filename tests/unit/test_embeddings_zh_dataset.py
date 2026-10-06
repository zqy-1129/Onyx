"""S33 向量数据集的验收（`embeddings_zh`）。

这份考卷的失效方式与 S31/S32 同族：**看起来永远是对的**——三句话都通顺、
gold 也确实在池子里。所以要把它写在注释里的承诺变成可跑的断言：

1. 同义句必须**换了谓语写法**（没换就是字符串匹配，任何模型都能满分）；
2. 反义句必须**只靠否定翻转极性**（换了话题就考不到"词面几乎全同"这个真短板）；
3. 无关项必须**不共享实体**（否则它在偷着做 gold 的活）；
4. 每条自检都配**注入缺陷**测试：不响的自检等于没有自检。
"""

from __future__ import annotations

import json

from onyx.eval.datasets.builtin.embeddings_zh import (
    BUILTIN_PATH,
    FRAMES,
    NEGATORS,
    UNRELATED,
    broken_relations,
    build_cases,
    gold_is_unique,
    same_string_pairs,
    stats,
    to_jsonl,
)
from onyx.eval.datasets.loader import load_builtin


def test_same_seed_reproduces_and_seed_changes_order():
    first, second = build_cases(), build_cases()
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    assert build_cases(seed=11) != build_cases(seed=12), "seed 不生效的话 revision 就是假的"


def test_ids_unique_and_ord_contiguous():
    cases = build_cases()
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids)
    assert sorted(case["ord"] for case in cases) == list(range(len(cases)))
    assert all(str(case["id"]).startswith("em-") for case in cases)


def test_every_construction_promise_holds():
    """三条自检对真实数据都必须说"没问题"——它们说的问题一个都不该存在。"""
    cases = build_cases()
    assert broken_relations(cases) == []
    assert same_string_pairs(cases) == []
    assert gold_is_unique(cases) == []


def test_one_case_per_frame_and_uniform_pool():
    cases = build_cases()
    image = stats(cases)
    assert image["n"] == len(cases) == len(FRAMES)
    assert image["frames"] == len(FRAMES), "有 frame 没出题"
    assert image["uniform_pool"] is True and image["pool_sizes"] == [UNRELATED + 2]
    assert image["inputs_per_case"] == UNRELATED + 3, "向量请求 = query + 候选池"
    for topic, count in image["topics"].items():
        assert count >= 2, f"话题 {topic} 只有 {count} 条，它的分母撑不起一个数"


def test_query_and_candidates_match_the_declared_relations():
    for case in build_cases():
        meta = case["meta"]
        assert case["input"]["query"] == f"{meta['entities'][0]}{meta['p_query']}{meta['entities'][1]}"
        assert list(meta["texts"]) == list(case["input"]["candidates"])
        gold = int(case["expect"]["gold"]) - 1
        anti = int(case["expect"]["antonym"]) - 1
        assert meta["relations"][gold] == "paraphrase"
        assert meta["relations"][anti] == "antonym"
        assert case["input"]["candidates"][gold] == (
            f"{meta['entities'][0]}{meta['p_para']}{meta['entities'][1]}")
        assert case["input"]["candidates"][anti] == (
            f"{meta['entities'][0]}{meta['p_anti']}{meta['entities'][1]}")


def test_gold_is_never_at_a_fixed_position():
    """池子被 shuffle 过：gold 永远排第 1 位的话，模型不看内容也能满分。"""
    positions = {int(case["expect"]["gold"]) for case in build_cases()}
    assert len(positions) > 1, f"gold 总在位置 {positions}"


# ── 注入缺陷：自检不许是摆设 ───────────────────────────────────────
def test_copy_paste_gold_is_caught_by_the_string_check():
    cases = build_cases()
    case = cases[0]
    gold = int(case["expect"]["gold"]) - 1
    case["meta"]["texts"][gold] = case["input"]["query"]
    case["input"]["candidates"][gold] = case["input"]["query"]
    assert same_string_pairs([case]) == [case["id"]], "同义句抄原句没被查出来"
    assert broken_relations([case]), "同义句没换谓语，broken_relations 也该一起响"


def test_missing_negation_is_caught_as_a_broken_antonym():
    cases = build_cases()
    case = cases[0]
    anti = int(case["expect"]["antonym"]) - 1
    # 把否定词抹掉 ⇒ 反义项变成另一条同义句，池子里就有两个 gold
    case["meta"]["texts"][anti] = case["input"]["query"]
    case["input"]["candidates"][anti] = case["input"]["query"]
    offenders = broken_relations([case])
    assert offenders and "否定" in " ".join(offenders), f"没查出反义句丢失否定: {offenders}"


def test_unrelated_item_that_shares_entities_is_caught():
    cases = build_cases()
    case = cases[0]
    slots = [i for i, r in enumerate(case["meta"]["relations"]) if r == "unrelated"]
    case["meta"]["texts"][slots[0]] = case["input"]["query"] + "（同一话题）"
    offenders = broken_relations([case])
    assert offenders and "送分" in " ".join(offenders), f"无关项含全部实体却没被点名: {offenders}"


def test_duplicated_gold_is_caught_by_uniqueness():
    cases = build_cases()
    case = cases[0]
    case["meta"]["relations"] = ["paraphrase", *case["meta"]["relations"][1:]]
    assert gold_is_unique([case]) == [case["id"]]


def test_declared_negators_are_a_closed_list():
    """否定标记必须在已知清单里：写错一个字，"anti 含否定"这条自检就会静默空转。"""
    assert {str(frame["negator"]) for frame in FRAMES} <= set(NEGATORS)


# ── 装载与来历 ─────────────────────────────────────────────────────
def test_shipped_jsonl_matches_the_generator():
    assert BUILTIN_PATH.read_text(encoding="utf-8") == to_jsonl(build_cases()) + "\n"


def test_load_builtin_carries_the_exam_provenance():
    dataset = load_builtin("embeddings_zh")
    assert dataset.id == "embeddings_zh-v1"
    assert dataset.upstream == "builtin:embeddings_zh"
    assert dataset.revision == "seed=20261006+frames=12+pool=4"
    assert dataset.loader == "builtin.embeddings_zh"
    assert dataset.splits()["default"] == len(dataset.cases) == len(FRAMES)


def test_topic_subsets_and_truncation_stay_nonempty():
    dataset = load_builtin("embeddings_zh")
    for topic in {str(case["meta"]["topic"]) for case in build_cases()}:
        assert dataset.select(split=topic), f"{topic} 子集是空的"
    assert len(dataset.select(limit=2)) == 2
