"""长上下文数据集的验收（S32）。

这份考卷的失效方式很特别：**它看起来永远是对的**——文档很长、问题很短、答案就在里面。
所以要逐条钉死几件只有把文本摊开才会发现的事：

1. 埋点句子必须真的在正文里（不在就是无解题）；
2. 答案值必须在全文里唯一（填充文本一个数字都不许有，否则"猜个常见数字"与"检索到了"分不开）；
3. **标签写 first/middle/last 不够，要用字符偏移验证它真的在那个位置**
   （`_plant` 漂了而 tag 没变，是这种数据最容易长出来的 bug，而它会让"中部塌陷"的结论作废）；
4. 三档必须单调变长，否则"16k 掉分"可能只是文本反而更短。
"""

from __future__ import annotations

import json

from onyx.eval.datasets.builtin.longctx_zh import (
    BUCKETS,
    BUILTIN_PATH,
    NEEDLES,
    POSITIONS,
    SEPARATOR,
    ambiguous_questions,
    build_cases,
    digits_only_in_needles,
    stats,
    to_jsonl,
    value_string_collisions,
)
from onyx.eval.datasets.loader import load_builtin


def test_same_seed_reproduces_the_same_cases():
    first, second = build_cases(), build_cases()
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True
    )
    assert build_cases(seed=7) != build_cases(seed=8), "seed 不生效的话 revision 就是假的"


def test_ids_unique_and_ord_contiguous():
    cases = build_cases()
    ids = [case["id"] for case in cases]
    assert len(set(ids)) == len(ids)
    assert sorted(case["ord"] for case in cases) == list(range(len(cases)))
    assert all(case["id"].startswith("lc-") for case in cases)


def test_every_needle_sentence_is_actually_in_the_document():
    """埋点句子不在正文里，这道题就**无解**——而表面看起来完全正常。"""
    missing = []
    for case in build_cases():
        body = case["input"]["text"].split(SEPARATOR)[0]
        for needle in case["meta"]["needles"]:
            if needle["sentence"] not in body:
                missing.append((case["id"], needle["id"]))
    assert not missing, f"埋点句子不在正文里: {missing}"


def test_answer_value_appears_exactly_once_in_the_whole_prompt():
    """答案唯一出现：否则"找到了"与"看到任何一个相同数字都算对"分不开。"""
    offenders = []
    for case in build_cases():
        text = case["input"]["text"]
        for needle in case["meta"]["needles"]:
            occurrences = text.count(str(needle["value"]))
            if occurrences != 1:
                offenders.append((case["id"], needle["id"], needle["value"], occurrences))
    assert not offenders, f"答案值出现次数不是 1: {offenders}"


def test_filler_text_contains_no_digits_at_all():
    """填充句里不许有数字（连"三点"这种都不行，判据用的是 \\d）。"""
    assert digits_only_in_needles(build_cases()) == []


def test_positions_are_where_the_tags_say_they_are():
    """**按字符偏移验证位置**，而不是只信 tag。

    `_plant` 一旦漂了（比如把 middle 算到 0.9 处），tag 仍会写着 middle，
    而"中部检索塌陷"这个结论就整条作废——这是长上下文数据特有的、最贵的一种错。
    """
    for case in build_cases():
        body = case["input"]["text"].split(SEPARATOR)[0]
        for needle in case["meta"]["needles"]:
            offset = body.index(needle["sentence"]) / len(body)
            position = needle["position"]
            if position == "first":
                assert offset < 0.2, f"{case['id']} 标 first 却出现在 {offset:.2f}"
            elif position == "middle":
                assert 0.3 <= offset <= 0.7, f"{case['id']} 标 middle 却在 {offset:.2f}"
            else:
                assert offset > 0.8, f"{case['id']} 标 last 却在 {offset:.2f}"


def test_distractors_are_present_and_in_a_different_paragraph():
    """每个埋点都配一个干扰项：同句式、不同实体、不同值，且**不在同一段**。

    没有干扰项时这份数据会悄悄退化成"数字扫描"——真机上确实发生过：
    第一版 9/9 全对，因为只要找到任意一个数字就能得分。
    同段的干扰项也不行：相邻两句等于把答案并排摆出来，那是句式匹配而不是检索。
    """
    for case in build_cases():
        body = case["input"]["text"].split(SEPARATOR)[0]
        paragraphs = body.split("\n\n")
        for needle in case["meta"]["needles"]:
            assert needle["distractor"] in body, f"{case['id']} 的干扰项没进正文"
            assert needle["distractor_value"] != needle["value"]
            home = next(i for i, p in enumerate(paragraphs) if needle["sentence"] in p)
            rival = next(i for i, p in enumerate(paragraphs) if needle["distractor"] in p)
            assert home != rival, f"{case['id']} 的干扰项与埋点同段"
            gap = abs(body.index(needle["sentence"]) - body.index(needle["distractor"]))
            assert gap > 120, f"{case['id']} 两句相隔 {gap} 字，太近等于送分"


def test_value_strings_never_contain_each_other():
    """值与值之间不许有子串关系，否则"全文只出现一次"这条自检数的就不是它想数的东西。

    真发生过：埋点 12.5 与干扰值 2.5 同篇共存，`text.count("2.5")` 数到 2——
    报出来像数据坏了，其实是判据在说谎。
    """
    assert value_string_collisions() == []
    pool = [{"value": 12.5, "distractor_value": 2.5}]
    assert value_string_collisions(pool), "子串冲突没被查出来，上面那条断言就是假的"


def test_distractor_value_also_appears_exactly_once():
    offenders = []
    for case in build_cases():
        text = case["input"]["text"]
        for needle in case["meta"]["needles"]:
            if text.count(str(needle["distractor_value"])) != 1:
                offenders.append((case["id"], needle["id"], needle["distractor_value"]))
    assert not offenders, f"干扰值出现次数不是 1: {offenders}"


def test_question_names_the_entity_the_answer_belongs_to():
    """题面自检：问句只点名答案实体，埋点句与干扰句互不提及对方实体。

    这是加干扰项之后**新长出来**的失效方式：问句里同时出现两个实体时，两个候选值都算
    "读对了"，检索判据当场失效，而分数照旧产出、看不出题面有问题。
    所以判据跑在 `subject` / `distractor_subject` 上，不是只检查"两句不一样"。
    """
    assert ambiguous_questions(build_cases()) == []
    for case in build_cases():
        names = {str(item[key]) for item in case["meta"]["needles"]
                 for key in ("subject", "distractor_subject")}
        assert len(names) == 2 * len(case["meta"]["needles"]), f"{case['id']} 有重名实体"
        for needle in case["meta"]["needles"]:
            assert needle["subject"] in needle["sentence"]
            assert needle["distractor_subject"] in needle["distractor"]
            assert needle["subject"] != needle["distractor_subject"]


def test_ambiguity_checker_reports_an_injected_ambiguous_question():
    """注入缺陷自检：把问句改成只点名另一个实体，`ambiguous_questions` 必须报出来。

    没有这一条，"题面不歧义"就只是写在注释里的愿望——检查器自己也会漂。
    """
    case = build_cases()[0]
    needle = case["meta"]["needles"][0]
    rival = case["meta"]["needles"][1]["subject"]
    head, tail = case["input"]["text"].split(SEPARATOR, 1)
    lines = tail.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith(f"{needle['id']}:"))
    lines[index] = f"{needle['id']}: {rival}是哪一项？"
    case["input"]["text"] = f"{head}{SEPARATOR}" + "\n".join(lines)
    assert ambiguous_questions([case]), "改坏了题面却检查不出来，说明这条自检是假的"


def test_digits_checker_reports_an_injected_number():
    """注入缺陷自检：往正文塞一个数字，`digits_only_in_needles` 必须抓到那一条。"""
    case = build_cases()[0]
    head, tail = case["input"]["text"].split(SEPARATOR, 1)
    case["input"]["text"] = f"{head}大约在 1990 年前后完成.{SEPARATOR}{tail}"
    assert digits_only_in_needles([case]) == [case["id"]]


def test_ambiguity_checker_reports_cross_mentioning_sentences():
    """注入缺陷自检：埋点句提到干扰实体、或干扰句提到答案实体，两条都要报。

    这是"两个候选值都算对"的另一条来路——不在题面里点名，而在句子里串门。
    """
    def _broken(field: str) -> list[str]:
        case = build_cases()[0]
        needle = case["meta"]["needles"][0]
        other = needle["distractor_subject"] if field == "sentence" else needle["subject"]
        needle[field] = f"{needle[field]}（与{other}同批登记）"
        return ambiguous_questions([case])

    assert _broken("sentence"), "埋点句串门没被查出来"
    assert _broken("distractor"), "干扰句串门没被查出来"


def test_write_jsonl_is_idempotent_with_the_shipped_file(tmp_path):
    """重新生成必须得到同一份字节：否则"数据文件与生成器同源"这条断言只在测试里成立。"""
    from onyx.eval.datasets.builtin.longctx_zh import write_jsonl

    target = write_jsonl(tmp_path / "regen.jsonl")
    assert target.read_text(encoding="utf-8") == BUILTIN_PATH.read_text(encoding="utf-8")


def test_each_case_has_one_needle_per_position():
    for case in build_cases():
        slots = [needle["position"] for needle in case["meta"]["needles"]]
        assert sorted(slots) == sorted(POSITIONS), f"{case['id']} 的位置覆盖不完整"
        assert len(set(slots)) == len(slots)


def test_buckets_are_monotonically_longer():
    """三档必须真的越来越长，否则"16k 掉分"可能只是文本更短。"""
    image = stats(build_cases())
    averages = [image["buckets"][bucket]["avg_hanzi"] for bucket in BUCKETS]
    assert image["monotonic"] is True
    assert averages == sorted(averages) and len(set(averages)) == 3
    for bucket, budget in BUCKETS.items():
        entry = image["buckets"][bucket]
        assert entry["cases"] >= 2, f"{bucket} 档只有 {entry['cases']} 条，没有分母"
        assert entry["min_hanzi"] >= budget * 0.9, f"{bucket} 档比预算短，token 量对不上档位名"


def test_answers_are_judgable_values_not_prose():
    """答案只能是数值或短编号：需要人读才知道对不对的判据不进这个数据集。"""
    for case in build_cases():
        for needle in case["meta"]["needles"]:
            value = needle["value"]
            assert isinstance(value, int | float | str)
            if isinstance(value, str):
                assert len(value) <= 8 and " " not in value, f"字符串答案太长：{value!r}"
            assert case["expect"]["answers"][needle["id"]] == value


def test_answer_types_cover_both_numeric_and_code():
    """数值与编号都得有：`field_em` 对两者走的是两套口径（按数值比 / 按字符串精确比），
    只考一种等于另一条代码路径从没被验收过。"""
    types = {type(needle["value"]) for needle in NEEDLES}
    assert str in types and types & {int, float}
    assert sum(isinstance(item["value"], str) for item in NEEDLES) >= 2
    assert all(isinstance(item["distractor_value"], item["value"].__class__) for item in NEEDLES), (
        "干扰值与答案值必须同型，否则'认错实体'会被判成'类型写错'"
    )


def test_subsets_and_truncation_stay_nonempty():
    dataset = load_builtin("longctx_zh")
    for bucket in BUCKETS:
        assert dataset.select(split=bucket), f"{bucket} 子集是空的"
    assert len(dataset.select(limit=1)) == 1
    assert dataset.splits()["default"] == len(dataset.cases) == len(build_cases())


def test_shipped_jsonl_matches_the_generator():
    assert BUILTIN_PATH.read_text(encoding="utf-8") == to_jsonl(build_cases()) + "\n"


def test_load_builtin_carries_the_exam_provenance():
    dataset = load_builtin("longctx_zh")
    assert dataset.id == "longctx_zh-v1"
    assert dataset.upstream == "builtin:longctx_zh"
    assert dataset.revision == "seed=20261005+budgets=6000,12000,24000+distractors=yes"
    assert dataset.loader == "builtin.longctx_zh"


def test_prompt_asks_for_an_answer_shape_the_grader_can_check():
    """问题块必须把输出形状说清（JSON、键名、不要单位）。

    否则"输出没结构"会被算成"没检索到"——那是考卷没说清，不是模型读不到。
    """
    for case in build_cases():
        tail = case["input"]["text"].split(SEPARATOR)[1]
        assert "JSON" in tail and "不要带单位" in tail
        for key in case["expect"]["keys"]:
            assert f"{key}:" in tail, f"{case['id']} 的问题块缺 {key}"
