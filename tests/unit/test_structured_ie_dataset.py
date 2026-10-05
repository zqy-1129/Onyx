"""`structured_ie` 生成器的验收（S30）。

数据集就是**考卷**，考卷错了分数没有意义。所以这里测的全是"考卷自洽"：
期望值过不了自己的 schema、负样本其实有可抽字段、相对日期与锚定日不一致——
这三种在真实评测里都表现为"模型明明对了却拿 0 分"，而现场只会怀疑模型。
（原稿里就有一条相对日期填错了天，是 `test_relative_dates_match_the_anchor` 抓出来的。）

生成器而非 JSONL 的理由写在数据模块的 docstring 里：槽位怎么轮换必须能被读出来与重跑。
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from onyx.eval.datasets.builtin.structured_ie import (
    ANCHOR,
    BUILTIN_PATH,
    DATES,
    EVENTS,
    FIELD_TYPES,
    HARD_CASES,
    NONE_CASES,
    ORGS,
    PERSONS,
    PLACES,
    VALUE_VOCAB,
    build_cases,
    schema_for,
    stats,
    to_jsonl,
)
from onyx.eval.datasets.loader import load_builtin
from onyx.eval.graders.json_schema import check_schema
from onyx.eval.tasks import build_task


def test_same_seed_reproduces_the_same_cases():
    """同 seed 同结果：revision 只写 `seed=` 就靠这条撑着。"""
    first, second = build_cases(), build_cases()
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    assert [case["id"] for case in first] == [case["id"] for case in second]
    assert build_cases(seed=1) != build_cases(seed=2), "seed 不生效的话 revision 就是假的"


def test_ids_unique_and_ord_contiguous():
    cases = build_cases()
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids), "case id 重复会让 grade 互相覆盖"
    assert sorted(case["ord"] for case in cases) == list(range(len(cases)))
    assert all(case["id"].startswith("sie-") for case in cases)


def test_every_expected_object_passes_its_own_schema():
    """**这条是考卷的自检**：期望值自己都不合规，模型答对了也拿不到分。"""
    bad = []
    for case in build_cases():
        keys = list(case["expect"]["keys"])
        if not keys:
            continue
        check = check_schema(dict(case["expect"]["fields"]), schema_for(keys))
        if not check.valid:
            bad.append((case["id"], check.errors))
    assert not bad, f"期望值过不了自己的 schema: {bad[:3]}"


def test_keys_are_exactly_the_field_names():
    """`keys` 是 schema 的 required，也是提示词里"必须包含的字段"，三处必须同源。"""
    for case in build_cases():
        fields = case["expect"]["fields"]
        assert sorted(case["expect"]["keys"]) == sorted(fields)
        assert set(fields) <= set(FIELD_TYPES), f"{case['id']} 用了没声明类型的字段"


def test_schema_is_per_case_and_closed():
    """`required` 逐条不同且 `additionalProperties: False`：共享 schema 会让少抽变成合规。"""
    schema = schema_for(["person", "amount"])
    assert schema["required"] == ["amount", "person"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["amount"]["type"] == "number"
    assert schema["properties"]["person"]["type"] == "string"
    # 空字符串不算抽到了：否则模型用 "" 占位就能混过 required
    assert check_schema({"person": "", "amount": 1}, schema).valid is False
    assert check_schema({"person": "张伟"}, schema).valid is False


def test_negative_cases_really_have_nothing_to_extract():
    """负样本必须真的没有可抽字段——否则它测的是别的东西。"""
    negatives = [case for case in build_cases() if case["kind"] == "none"]
    assert len(negatives) == len(NONE_CASES) and negatives
    names = set(PERSONS) | set(ORGS) | set(PLACES) | set(EVENTS)
    for case in negatives:
        assert case["expect"] == {"fields": {}, "keys": []}
        text = case["input"]["text"]
        assert not any(name in text for name in names), f"{text} 里其实有可抽字段"


@pytest.mark.parametrize(("surface", "expected"), [
    ("后天", "2026-03-03"),
    ("下周三", "2026-03-04"),
    ("一周后", "2026-03-08"),
])
def test_relative_surfaces_agree_with_the_anchor(surface, expected):
    """相对日期的期望值必须等于锚定日推出来的那天。

    填错的期望值比少一条样本坏得多：模型答对了也得 0 分，
    而 `date` 这一列的低分会被读成"模型不懂中文时间表达"。
    这里按字面重算一遍（不共用生成器里的 `_offset`），因为原稿就填错过一天。
    """
    assert date.fromisoformat(ANCHOR).strftime("%A") == "Sunday", (
        "锚定日的星期变了，下面这几条字面值要跟着重算"
    )
    surfaces = {text: iso for iso, text in DATES}
    assert surface in surfaces, f"数据集里已经没有带「{surface}」的样本了"
    assert surfaces[surface] == expected


def test_sentences_with_one_time_expression_use_that_day():
    """句子里只有一个时间说法时，期望日期就是那一天。"""
    surfaces = {text: iso for iso, text in DATES}
    checked = 0
    for case in build_cases():
        if "hard" in case["tags"]:
            continue  # 难例刻意塞了两个时间说法（取第一个），那条规则单独测
        text = case["input"]["text"]
        hit = [surface for surface in surfaces if surface in text]
        if len(hit) != 1 or "date" not in case["expect"]["fields"]:
            continue
        checked += 1
        assert case["expect"]["fields"]["date"] == surfaces[hit[0]], (
            f"{text} 只提到「{hit[0]}」，期望值却不是那天"
        )
    assert checked >= 3, f"只有 {checked} 条样本被这条自检覆盖，日期列基本没测"


def test_amounts_and_dates_are_judgable_values():
    """期望值是数值与 ISO，不是表面写法：判据不许需要人读。"""
    amounts = {
        case["expect"]["fields"]["amount"]
        for case in build_cases() if "amount" in case["expect"]["fields"]
    }
    assert amounts and all(isinstance(value, float) for value in amounts)
    dates = {
        case["expect"]["fields"]["date"]
        for case in build_cases() if "date" in case["expect"]["fields"]
    }
    assert dates and all(len(value) == 10 and value.count("-") == 2 for value in dates)


def test_chinese_numeral_surfaces_map_to_numbers():
    """中文数字是这任务最难的一档，期望值必须是数值而不是"两万三"这样的写法。"""
    cases = build_cases()
    assert any(
        "两千三" in case["input"]["text"] and case["expect"]["fields"]["amount"] == 2300.0
        for case in cases
    )
    by_text = {case["input"]["text"]: case for case in cases}
    hard = by_text["张伟说转账两万三给李娜，用的是招商银行的卡。"]
    assert hard["expect"]["fields"]["amount"] == 23000.0


def test_hard_cases_are_tagged_and_carry_their_note():
    cases = build_cases()
    hard = [case for case in cases if "hard" in case["tags"]]
    assert len(hard) == len(HARD_CASES)
    by_text = {case["input"]["text"]: case for case in cases}
    for text, expect, note in HARD_CASES:
        case = by_text[text]
        assert case["expect"]["fields"] == expect
        assert case["meta"]["note"] == note, "难例为什么难要写在数据里，而不是只留在注释里"
    assert any("person" not in case["expect"]["fields"] for case in hard), (
        "难例里必须有一条不含人名（「我」不该被抽成 person）"
    )
    both = by_text["赵敏昨天在美团下了一单，后天再退，金额八十九块九。"]
    assert both["expect"]["fields"]["date"] == (
        date.fromisoformat(ANCHOR) - timedelta(days=1)
    ).isoformat(), "句中有两个相对时间时取第一个（昨天），不是取最后一个"


def test_event_values_stay_inside_the_declared_vocabulary():
    """词表是提示词的一部分：期望值落在词表外，模型照词表输出反而被判错。"""
    vocab = set(VALUE_VOCAB["event"])
    offenders = [
        (case["id"], fields["event"])
        for case in build_cases()
        for fields in [case["expect"]["fields"]]
        if "event" in fields and fields["event"] not in vocab
    ]
    assert not offenders, f"期望值不在词表里: {offenders[:3]}"
    # 词表必须真的出现在提示词里，否则它只是数据模块里的一段死代码
    task = build_task("structured_extraction", model="mock/x")
    prompt = task.build(next(c for c in task.load() if "event" in c.expect["fields"])).messages[0].content
    assert all(value in prompt for value in vocab), "提示词没把 event 的取值给出来"


def test_prompt_declares_the_normalization_the_grader_demands():
    """期望值要求 ISO 日期与纯数值，提示词就必须这么说。

    真机第一次跑：`date` 的字段级 EM 是 0.000（15 条全错），
    因为模型照原句抄了「3 月 4 号」而我们要求 `2026-03-04`——
    那时旧提示词写的恰恰是"字段值必须来自原句，不要改写"。
    考卷没说清答题格式，分数却长得像能力问题。
    """
    task = build_task("structured_extraction", model="mock/x")
    prompt = task.build(next(c for c in task.load() if "date" in c.expect["fields"])).messages[0].content
    assert "YYYY-MM-DD" in prompt and ANCHOR in prompt
    assert "只输出数值" in prompt and "千分位" in prompt
    assert "不要改写" in prompt


def test_every_field_has_enough_denominator():
    """字段分布可见：只有 person 抽得准时，单一总分说不清哪里坏了。"""
    cases = build_cases()
    image = stats(cases)
    assert image["n"] == len(cases)
    assert image["unique_texts"] == image["n"], "同句不同期望会让两个任务共用一个 case id"
    assert image["none"] == len(NONE_CASES)
    for name in FIELD_TYPES:
        assert image["fields"].get(name, 0) >= 20, f"{name} 的样本太少，per_field 没有分母"


def test_subsets_and_truncation_stay_nonempty():
    """DoD：`--limit` 截断与子集选择都必须非空，否则跑出来的是空考卷。"""
    dataset = load_builtin("structured_ie")
    assert dataset.splits()["none"] == len(NONE_CASES)
    for split in ("default", "hard", "none", "amount", "date", "person"):
        assert dataset.select(split=split), f"{split} 子集是空的"
    head = dataset.select(limit=6)
    assert len(head) == 6
    assert {field for case in head for field in case["expect"]["keys"]}, "前 6 条全无字段可抽"


def test_shipped_jsonl_matches_the_generator():
    """随包分发的 JSONL 是生成器的镜像：它漂了必须被发现，而不是各留一份口径。"""
    assert BUILTIN_PATH.read_text(encoding="utf-8") == to_jsonl(build_cases()) + "\n"


def test_load_builtin_carries_the_exam_provenance():
    dataset = load_builtin("structured_ie")
    assert dataset.id == "structured_ie-v1"
    assert dataset.upstream == "builtin:structured_ie"
    assert dataset.revision == "seed=20261003+anchor=2026-03-01"
    assert dataset.loader == "builtin.structured_ie"
    assert dataset.license == "generated-in-repo"
