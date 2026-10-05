"""指令集生成器的验收（S31）。

这份考卷的地基是两句话，两条都有对应的测试：

1. **每道题都必须有解**——参考回答满足不了自己的约束，就是无解的题，
   现场只会怀疑模型（`unsatisfiable()` 把这件事变成一条可跑的断言）。
2. **凡是考模型的都必须说得出**——提示语由约束渲染回来，
   约束与提示语不同源就是"猜话"，与 S30 里"封闭词表没给模型"是同一种错。

其余是考卷卫生：可复现、id 稳定、每种约束有分母、子集与截断非空、随包 JSONL 与生成器一致。
"""

from __future__ import annotations

import json

import pytest

from onyx.eval.datasets.builtin.instructions_zh import (
    BANNED,
    BUILTIN_PATH,
    HARD,
    KIND_COVERAGE,
    TOPICS,
    VARIANTS,
    build_cases,
    phrase_for,
    render_instruction,
    stats,
    to_jsonl,
    unsatisfiable,
)
from onyx.eval.datasets.loader import load_builtin
from onyx.eval.graders.constraints import KINDS


def test_same_seed_reproduces_the_same_cases():
    first, second = build_cases(), build_cases()
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    assert build_cases(seed=1) != build_cases(seed=2), "seed 不生效的话 revision 就是假的"


def test_every_case_is_satisfiable_by_its_own_reference():
    """**这条是整个数据集的地基**：无解的题不叫难题，叫坏题。"""
    bad = unsatisfiable(build_cases())
    assert not bad, f"这些题的参考回答满足不了自己的约束: {bad[:3]}"


def test_the_satisfiability_detector_actually_fires():
    """注入一条缺陷：把某题的字数上限收到参考回答之下，检测必须点名它。

    "考卷有解"这条断言只有**能被人为破坏**才算在守着什么；
    否则我们只是在读一个永远为空的列表。
    """
    cases = build_cases()
    victim = cases[0]
    ceiling = next(item for item in victim["expect"]["constraints"]
                   if item["kind"] == "max_chars")
    ceiling["params"]["count"] = 1  # 一道没人能满足的题
    bad = unsatisfiable(cases)
    assert [case_id for case_id, _reasons in bad] == [victim["id"]], bad
    assert "实测" in bad[0][1][0], "失败原因要说清实测多少、要求多少"


def test_a_constraint_kind_without_a_phrase_is_refused():
    """新加约束类型却忘了写提示语 ⇒ 当场炸，而不是悄悄考一道没说清要求的题。"""
    with pytest.raises(KeyError, match="还没有对应的提示语"):
        phrase_for("must_be_polite", {})


def test_instruction_text_states_every_constraint():
    """逐条把约束渲染回提示语，要求它原样出现在指令文本里。

    约束在文本里没有对应说法，模型就只能猜——那测的不是遵循能力。
    """
    for case in build_cases():
        text = case["input"]["text"]
        for item in case["expect"]["constraints"]:
            phrase = phrase_for(item["kind"], item.get("params") or {})
            assert phrase in text, f"{case['id']} 的约束「{item['kind']}」没写进指令：{text}"


def test_reference_and_constraints_agree_on_the_rendered_instruction():
    """指令文本 = `render_instruction(话题, 约束)`，不留第二份写法。"""
    for case in build_cases():
        topic = next(t for t in TOPICS if t["id"] == case["meta"]["topic"])
        assert case["input"]["text"] == render_instruction(
            topic["ask"], case["expect"]["constraints"]
        )


def test_ids_unique_and_ord_contiguous():
    cases = build_cases()
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids)
    assert sorted(case["ord"] for case in cases) == list(range(len(cases)))
    assert all(case["id"].startswith("ins-") for case in cases)


def test_length_limits_are_never_self_contradictory():
    """同一条题里 min ≤ max 是底线；反了就是无解，而上面那条测试也会抓到它。"""
    offenders = []
    for case in build_cases():
        bounds = {c["kind"]: c["params"] for c in case["expect"]["constraints"]}
        if ("min_chars" in bounds and "max_chars" in bounds
                and bounds["min_chars"]["count"] > bounds["max_chars"]["count"]):
            offenders.append(case["id"])
    assert not offenders


def test_every_constraint_kind_has_a_denominator():
    """某类约束只考 1 条时它的"满足率"没有意义，所以每种都要有下限。

    下限写在数据模块里（`KIND_COVERAGE`），这样"加了新约束类型但没出题"也会被抓到。
    """
    image = stats(build_cases())
    for kind in KINDS:
        assert image["kinds"].get(kind, 0) >= KIND_COVERAGE, (
            f"{kind} 只考了 {image['kinds'].get(kind, 0)} 条，少于下限 {KIND_COVERAGE}"
        )
    for case in build_cases():
        assert 2 <= len(case["expect"]["constraints"]) <= 5, case["id"]


def test_banned_words_really_are_absent_from_references():
    """禁词池里的词若出现在某条参考回答里，那条题就自相矛盾了。"""
    for case in build_cases():
        reference = str(case["meta"]["reference"])
        for item in case["expect"]["constraints"]:
            if item["kind"] == "forbids":
                for word in item["params"]["values"]:
                    assert word in BANNED or word not in reference
                    assert word not in reference, f"{case['id']} 的参考回答里有禁词 {word}"


def test_subsets_and_truncation_stay_nonempty():
    dataset = load_builtin("instructions_zh")
    assert dataset.splits()["hard"] == len(HARD)
    for split in ("default", "hard", "plain", "lines", "json"):
        assert dataset.select(split=split), f"{split} 子集是空的"
    assert len(dataset.select(limit=3)) == 3
    assert {c["expect"]["constraints"][0]["kind"] for c in dataset.select(limit=3)}


def test_variants_multiply_the_cases_not_the_topics():
    image = stats(build_cases())
    assert image["n"] == len(TOPICS) * 3 * VARIANTS + len(HARD)
    assert image["unique_instructions"] == image["n"], "同题不同约束会撞 id，指令也必须互不相同"


def test_shipped_jsonl_matches_the_generator():
    assert BUILTIN_PATH.read_text(encoding="utf-8") == to_jsonl(build_cases()) + "\n"


def test_load_builtin_carries_the_exam_provenance():
    dataset = load_builtin("instructions_zh")
    assert dataset.id == "instructions_zh-v1"
    assert dataset.upstream == "builtin:instructions_zh"
    assert dataset.revision == "seed=20261005"
    assert dataset.loader == "builtin.instructions_zh"
    assert dataset.license == "generated-in-repo"
