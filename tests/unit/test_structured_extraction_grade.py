"""结构化抽取任务的形态表（S30）。

这里钉的是**"哪种输出形态判成什么"**，一共两层：

1. 三层正交判定：能不能解析 / 结构合不合规 / 内容准不准。
   每种形态都必须只落在它自己那一层——把"多抽了一个字段"判成"内容错"，
   修的方向就成了换模型，而真正该改的是字段约束与提示词。
2. 聚合的分母：负样本不许进主分数，引擎故障不许进能力分母，
   格式不听话不许伪装成能力差。

跨任务同源的三条（声明==产出、CI 跟着分数、主分数不占稳定性位）在
`tests/contract/test_task_contract.py`，这里不重复。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from onyx.core.types import Generation, Status
from onyx.eval.datasets.loader import Dataset
from onyx.eval.task import Grade, Verdict
from onyx.eval.tasks.structured_extraction import StructuredExtraction

MODEL = "mock/sie"
FIELDS: dict[str, Any] = {
    "person": "张伟", "date": "2026-03-05", "org": "支付宝", "amount": 500.0,
}
TEXT = "张伟于2026年3月5日向支付宝支付了500元。"


def _cases() -> list[dict[str, Any]]:
    return [
        {
            "id": "pos-1", "ord": 0, "kind": "single", "input": {"text": TEXT},
            "expect": {"fields": dict(FIELDS), "keys": sorted(FIELDS)},
            "tags": ["template"], "meta": {},
        },
        {
            "id": "pos-2", "ord": 1, "kind": "single",
            "input": {"text": "李娜在美团消费了88块5。"},
            "expect": {"fields": {"person": "李娜", "amount": 88.5}, "keys": ["amount", "person"]},
            "tags": ["template"], "meta": {},
        },
        {
            "id": "pos-3", "ord": 3, "kind": "single",
            "input": {"text": "陈杰在杭州预约维修。"},
            "expect": {"fields": {"person": "陈杰", "place": "杭州", "event": "预约维修"},
                       "keys": ["event", "person", "place"]},
            "tags": ["template"], "meta": {},
        },
        {
            "id": "neg-1", "ord": 2, "kind": "none",
            "input": {"text": "今天天气不错，适合出去走走。"},
            "expect": {"fields": {}, "keys": []}, "tags": ["none"], "meta": {},
        },
    ]


def _task(cases: Sequence[dict[str, Any]] | None = None) -> StructuredExtraction:
    dataset = Dataset(id="sie-tiny-v1", cases=tuple(cases if cases is not None else _cases()),
                      upstream="test", revision="r1", loader="test")
    return StructuredExtraction(dataset, model=MODEL)


def _case(case_id: str):
    return next(c for c in _task().load() if c.id == case_id)


def _gen(text: str, **kw: Any) -> Generation:
    return Generation(text=text, model=MODEL, status=kw.pop("status", Status.OK), **kw)


def _dump(fields: dict[str, Any]) -> str:
    return json.dumps(fields, ensure_ascii=False)


# ── 形态表 ─────────────────────────────────────────────────────────
def test_clean_object_is_correct_and_format_valid():
    grade = _task().grade(_case("pos-1"), _gen(_dump(FIELDS)))
    assert grade.verdict is Verdict.CORRECT and grade.passed is True
    assert grade.score == 1.0 and grade.invalid_format is False
    assert grade.metrics["json_valid"] and grade.metrics["schema_valid"]
    assert grade.metrics["field_total"] == len(FIELDS)


def test_code_fence_is_rescued_by_the_parser_but_costs_the_format_dimension():
    """能解析 ≠ 干净：围栏会被 `parse_json` 救回来，但那是没听话的形态（DESIGN §9.4）。"""
    grade = _task().grade(_case("pos-1"), _gen(f"```json\n{_dump(FIELDS)}\n```"))
    assert grade.verdict is Verdict.CORRECT and grade.invalid_format is True
    assert grade.metrics["json_valid"] and grade.metrics["schema_valid"]


def test_leading_prose_is_rescued_but_marked_dirty():
    grade = _task().grade(_case("pos-1"), _gen("好的，结果如下：" + _dump(FIELDS)))
    assert grade.verdict is Verdict.CORRECT and grade.invalid_format is True


def test_wrong_value_is_a_content_error_not_a_structure_error():
    """值错才是能力问题：结构必须仍然算合规，否则修方向就全错了。"""
    wrong = dict(FIELDS, org="招商银行")
    grade = _task().grade(_case("pos-1"), _gen(_dump(wrong)))
    assert grade.verdict is Verdict.PARTIAL and grade.passed is False
    assert grade.metrics["schema_valid"] is True and grade.metrics["json_valid"] is True
    assert grade.metrics["per_field_ok"]["org"] is False
    assert grade.metrics["per_field_ok"]["person"] is True
    assert grade.score == pytest.approx(3 / 4)


def test_all_values_wrong_is_wrong_not_partial():
    grade = _task().grade(_case("pos-2"), _gen(_dump({"person": "王芳", "amount": 12.0})))
    assert grade.verdict is Verdict.WRONG and grade.score == 0.0
    assert grade.metrics["schema_valid"] is True


def test_missing_field_is_a_structure_error_with_the_name_in_the_reason():
    """少抽一个字段必须点名是哪个字段：只报"不合规"没法行动。"""
    grade = _task().grade(_case("pos-1"), _gen(_dump({"person": "张伟", "date": "2026-03-05"})))
    assert grade.verdict is Verdict.WRONG
    assert grade.metrics["schema_valid"] is False and grade.metrics["json_valid"] is True
    assert {"amount", "org"} <= set(grade.metrics["missing"])


def test_extra_field_is_rejected_by_the_closed_schema():
    grade = _task().grade(_case("pos-1"), _gen(_dump(dict(FIELDS, phone="13800000000"))))
    assert grade.verdict is Verdict.WRONG and grade.metrics["schema_valid"] is False
    assert "phone" in grade.error


def test_string_amount_is_a_type_error_not_a_value_error():
    """"500元" 是抄表面写法：字段约束没吃进去，与"值抽错"是两种病。"""
    grade = _task().grade(_case("pos-1"), _gen(_dump(dict(FIELDS, amount="500元"))))
    assert grade.metrics["json_valid"] is True and grade.metrics["schema_valid"] is False
    assert "amount" in grade.error


def test_unparseable_output_stops_at_the_first_layer():
    """解析不出来就到此为止：再往下每一层都无意义，硬算会把格式问题记成能力问题。"""
    grade = _task().grade(_case("pos-1"), _gen("张伟 支付宝 500 元 三月五号"))
    assert grade.verdict is Verdict.INVALID_FORMAT and grade.invalid_format is True
    assert grade.metrics["json_valid"] is False and "schema_valid" in grade.metrics
    assert grade.score == 0.0


def test_empty_body_names_the_thinking_budget_when_it_was_eaten():
    """正文为空但产出了推理内容 ⇒ 是预算被 thinking 吃光（P12），与"模型没输出"修法不同。"""
    task = _task()
    plain = task.grade(_case("pos-1"), _gen(""))
    eaten = task.grade(_case("pos-1"), _gen("", thinking="让我先想想这句话的结构"))
    assert plain.verdict is Verdict.INVALID_FORMAT
    assert "thinking" not in plain.error, "没被 thinking 吃光时不该提它"
    assert "thinking" in eaten.error and eaten.metrics["thinking_chars"] > 0


def test_refusal_is_its_own_verdict():
    grade = _task().grade(_case("pos-1"), _gen("抱歉，我无法处理这句话。"))
    assert grade.verdict is Verdict.REFUSED and grade.invalid_format is True
    assert grade.metrics["json_valid"] is False


def test_engine_failure_is_not_a_model_answer():
    grade = _task().grade(_case("pos-1"), _gen("", status=Status.ERROR, error="connection reset"))
    assert grade.verdict is Verdict.ERROR and grade.passed is None
    assert not grade.attributable and grade.error == "connection reset"


def test_negative_case_wants_an_empty_object():
    grade = _task().grade(_case("neg-1"), _gen("{}"))
    assert grade.verdict is Verdict.CORRECT and grade.score == 1.0
    assert grade.metrics["field_total"] == 0 and grade.metrics["hallucinated"] == []
    # 期望/预测要留着给界面：负样本那两行显示「—」会让人以为这条没判
    assert grade.metrics["expected"] == {} and grade.metrics["predicted"] == {}


def test_inventing_a_field_on_a_negative_case_is_named():
    """凭空造字段是抽取任务里最贵的失败：它会顺着管道流进数据库。"""
    grade = _task().grade(_case("neg-1"), _gen(_dump({"person": ""})))
    assert grade.verdict is Verdict.WRONG and grade.score == 0.0
    assert grade.metrics["hallucinated"] == ["person"]
    assert "person" in grade.error


def test_negative_case_with_unparseable_output_still_counts_as_format_failure():
    grade = _task().grade(_case("neg-1"), _gen("这句话里没有信息"))
    assert grade.verdict in (Verdict.INVALID_FORMAT, Verdict.REFUSED)
    assert grade.invalid_format is True


# ── 聚合的分母 ─────────────────────────────────────────────────────
def _grades(*grades: Grade) -> list[Grade]:
    return list(grades)


def test_negative_samples_stay_out_of_the_headline_denominator():
    """主分数只数"确实有字段可抽"的样本：混进负样本等于奖励"什么都不抽"。"""
    task = _task()
    case = _case("neg-1")
    grades = _grades(task.grade(case, _gen("{}")))
    aggregate = task.aggregate(grades)
    assert aggregate["n_total"] == 1 and aggregate["n_judged"] == 0
    assert aggregate["score"] is None, "只有负样本时主分数必须是「没考到」，不是 1.0"
    assert aggregate["none_total"] == 1 and aggregate["none_correct_rate"] == 1.0
    assert aggregate["exact_object_rate"] == 1.0


def test_structural_failures_are_out_of_score_but_inside_exact_object_rate():
    """两个正确率分母不同：差值就是格式与结构层造成的损失。"""
    task = _task()
    pos, neg = _case("pos-1"), _case("neg-1")
    grades = _grades(
        task.grade(pos, _gen(_dump(FIELDS))),
        task.grade(pos, _gen(_dump({"person": "张伟"}))),
        task.grade(pos, _gen("不是 JSON")),
        task.grade(neg, _gen("{}")),
    )
    aggregate = task.aggregate(grades)
    assert aggregate["n_attributable"] == 4
    assert aggregate["n_judged"] == 1
    assert aggregate["score"] == 1.0
    assert aggregate["exact_object_rate"] == pytest.approx(2 / 4)
    assert aggregate["json_valid_rate"] == pytest.approx(3 / 4)
    assert aggregate["schema_valid_rate"] == pytest.approx(2 / 4)


def test_score_ci_is_the_same_statistic_as_score():
    task = _task()
    pos = _case("pos-1")
    grades = _grades(
        task.grade(pos, _gen(_dump(FIELDS))),
        task.grade(pos, _gen(_dump(dict(pos.expect["fields"], org="别的")))),
    )
    aggregate = task.aggregate(grades, seed=11)
    ci = aggregate["score_ci"]
    assert ci["point"] == aggregate["score"] == pytest.approx(0.5)
    assert ci["n"] == aggregate["n_judged"] == 2
    assert ci["low_confidence"] is True


def test_no_schema_check_ran_means_schema_verified_is_unknown():
    """一条都没走到 schema 检查时，`schema_verified` 必须是 None 而不是 True。

    "没校验过"与"校验过且通过"在界面上都只有一位布尔，
    把它们混起来等于把 jsonschema 缺席或格式全坏伪装成合规（DESIGN §9.4 / 未知≠0）。
    """
    task = _task()
    case = _case("pos-1")
    aggregate = task.aggregate(_grades(
        task.grade(case, _gen("这不是 JSON")),
        task.grade(case, _gen("抱歉，我无法处理。")),
    ))
    assert aggregate["json_valid_rate"] == 0.0
    assert aggregate["schema_verified"] is None, "没做过 schema 校验，不许报「真校验过」"


def test_engine_errors_leave_the_capability_denominator():
    task = _task()
    pos = _case("pos-1")
    aggregate = task.aggregate(_grades(
        task.grade(pos, _gen(_dump(FIELDS))),
        task.grade(pos, _gen("", status=Status.ERROR, error="boom")),
    ))
    assert aggregate["n_total"] == 2 and aggregate["n_attributable"] == 1
    assert aggregate["verdicts"][Verdict.ERROR.value] == 1
    assert aggregate["score"] == 1.0


def test_per_field_reports_a_denominator_and_stays_unknown_when_untested():
    """没考到的字段必须是 None：填 0 会让"没测"看起来像"这项能力差"。"""
    task = _task()
    aggregate = task.aggregate(_grades(task.grade(_case("pos-2"), _gen(_dump(
        {"person": "李娜", "amount": 88.5})))))
    per_field = aggregate["per_field"]
    assert per_field["person"] == {"em": 1.0, "n": 1}
    assert per_field["date"]["em"] is None and per_field["date"]["n"] == 0


def test_hallucinated_fields_are_counted_across_the_whole_run():
    task = _task()
    aggregate = task.aggregate(_grades(
        task.grade(_case("neg-1"), _gen(_dump({"person": "张三", "org": "某公司"}))),
        task.grade(_case("pos-2"), _gen(_dump({"person": "李娜", "amount": 88.5}))),
    ))
    assert aggregate["hallucinated_fields"] == 2
    assert aggregate["none_correct_rate"] == 0.0


def test_low_confidence_follows_case_count_not_grade_count():
    """k=4 时按 grade 条数判会虚高 4 倍，把小样本说成够样本。"""
    task = _task()
    case = _case("pos-1")
    repeated = _grades(*[task.grade(case, _gen(_dump(FIELDS))) for _ in range(4)])
    aggregate = task.aggregate(repeated)
    assert aggregate["n_total"] == 4 and aggregate["k"] == 4
    assert aggregate["low_confidence"] is True, "只有 1 个 case，4 条 grade 不算样本量"
    assert aggregate["pass_hat_k"] == aggregate["pass_at_k"] == 1.0
    assert aggregate["stability_gap"] == 0.0


def test_system_prompt_declares_the_required_fields_of_this_case():
    """`required` 逐条不同：提示词必须说清这条要抽哪些字段。"""
    task = _task()
    pos = task.build(_case("pos-1")).messages[0].content
    neg = task.build(_case("neg-1")).messages[0].content
    for name in FIELDS:
        assert name in pos
    assert "空对象" in neg
    assert neg.split("必须包含的字段")[1].split("。")[0] == "：（无，输出空对象即可）"
    assert task.build(_case("pos-1")).thinking is False


def test_json_number_style_is_not_an_extraction_error():
    """`500` 与 `500.0` 是同一笔钱：按字符串比会把 JSON 写法算成抽错。

    真机第一跑 amount 的字段级 EM 只有 0.211（19 条），
    逐条看下去大部分掉分的正是写成整数的 500 / 1500。
    """
    task = _task()
    grade = task.grade(_case("pos-1"), _gen(_dump(dict(FIELDS, amount=500))))
    assert grade.verdict is Verdict.CORRECT and grade.score == 1.0


def test_a_string_amount_is_still_wrong():
    """放宽只针对写法：`"500元"` 单位没去掉，是真的没抽对（schema 那一层先拦下）。"""
    grade = _task().grade(_case("pos-1"), _gen(_dump(dict(FIELDS, amount="500元"))))
    assert grade.metrics["schema_valid"] is False and "amount" in grade.error


def test_made_up_event_word_is_marked_out_of_vocabulary():
    """造词与"选错了另一个类别"分开计：修法一个是提示词与词表说明，一个是模型分不清。"""
    case = _case("pos-3")
    fields = dict(case.expect["fields"], event="修电脑")
    grade = _task().grade(case, _gen(_dump(fields)))
    assert grade.out_of_set is True
    assert grade.metrics["off_vocabulary"] == ["event"]
    assert grade.verdict is not Verdict.CORRECT

    aggregate = _task().aggregate([grade])
    assert aggregate["off_vocabulary_rate"] == 1.0
    assert aggregate["schema_valid_rate"] == 1.0, "结构与词表是两个维度，不许挤在一起"


def test_in_vocab_event_is_not_marked_out_of_set():
    case = _case("pos-3")
    grade = _task().grade(case, _gen(_dump(case.expect["fields"])))
    assert grade.out_of_set is False and grade.verdict is Verdict.CORRECT
    assert _task().aggregate([grade])["off_vocabulary_rate"] == 0.0
