"""指令遵循任务的形态表与聚合口径（S31）。

两层都要钉住：

1. **形态表**：同一条题在"照做 / 半做 / 没做 / 什么都不写 / 拒答 / 引擎挂了"
   六种产出下各自判成什么。其中最要紧的是两条反直觉的顺序：
   空正文**不给任何约束记分**（否则"什么都不写"能靠 `max_chars` 蹭分），
   而全部约束都满足的回答**不许被拒答启发式改判**（否则测量被改写成了故事）。
2. **三个口径会分叉**：`score`（样本等权）/ `micro_rate`（约束等权）/
   `all_satisfied_rate`（最严）。手算一例钉住它们的差，是为了让"只引用最好看的那个数"
   这件事在 review 时一眼可见。

跨任务同源的三条在 `tests/contract/test_task_contract.py`，这里不重复。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from onyx.core.types import Generation, Status
from onyx.eval.datasets.loader import Dataset
from onyx.eval.task import Verdict
from onyx.eval.tasks.instruction_following import InstructionFollowing

MODEL = "mock/ins"
REFUND = "退款请在七天内提交，款项按原路退回。"


def _cases() -> list[dict[str, Any]]:
    """两条小题：A 有四条约束，B 只有两条——口径分叉就来自这里。"""
    return [
        {
            "id": "A", "ord": 0, "kind": "instruction",
            "input": {"text": "请说明退款政策。"},
            "expect": {"constraints": [
                {"kind": "max_chars", "params": {"count": 40}},
                {"kind": "contains", "params": {"values": ["七天内"]}},
                {"kind": "forbids", "params": {"values": ["抱歉"]}},
                {"kind": "no_markdown", "params": {}},
            ]},
            "tags": ["plain"], "meta": {"reference": REFUND},
        },
        {
            "id": "B", "ord": 1, "kind": "instruction",
            "input": {"text": "请安排一场会议。"},
            "expect": {"constraints": [
                {"kind": "min_chars", "params": {"count": 200}},
                {"kind": "contains", "params": {"values": ["周三"]}},
            ]},
            "tags": ["plain"], "meta": {"reference": "周三下午两点在三楼会议室开会，请提前十分钟到。"},
        },
    ]


def _task(cases: Sequence[dict[str, Any]] | None = None) -> InstructionFollowing:
    dataset = Dataset(id="ins-tiny-v1", cases=tuple(cases if cases is not None else _cases()),
                      upstream="test", revision="r1", loader="test")
    return InstructionFollowing(dataset, model=MODEL)


def _case(case_id: str):
    return next(c for c in _task().load() if c.id == case_id)


def _gen(text: str, **kw: Any) -> Generation:
    return Generation(text=text, model=MODEL, status=kw.pop("status", Status.OK), **kw)


# ── 形态表 ─────────────────────────────────────────────────────────
def test_full_compliance_is_correct_with_every_constraint_measured():
    case = _case("A")
    grade = _task().grade(case, _gen(REFUND))
    assert grade.verdict is Verdict.CORRECT and grade.passed is True
    assert grade.score == 1.0 and grade.invalid_format is False
    assert grade.metrics["constraint_satisfied"] == grade.metrics["constraint_total"] == 4
    assert grade.metrics["violated"] == []


def test_partial_compliance_reports_which_constraints_failed_and_why():
    case = _case("A")
    grade = _task().grade(case, _gen("- 退款要七天内处理\n- 款项原路退回"))
    assert grade.verdict is Verdict.PARTIAL and grade.passed is False
    assert grade.metrics["constraint_satisfied"] == 3
    assert grade.metrics["constraint_total"] == 4
    assert any("markdown" in line for line in grade.metrics["violated"]), grade.metrics["violated"]


def test_no_compliance_is_wrong_not_partial():
    """四条约束全违反才是 WRONG；满足一条就是 PARTIAL，两者修法不同。

    这里刻意写得很长（而不是短到像一句"抱歉我无法"）：短会被判成拒答，
    而这道题要测的是"答了但什么都没照做"。
    """
    case = _case("A")
    text = (
        "- 抱歉，今天天气不错，适合出去走走；"
        "另外我还要多写几句完全无关的话，好让字数上限也被一起打破，"
        "顺便再补一条关于会议安排的说明，虽然这道题问的并不是会议。"
    )
    grade = _task().grade(case, _gen(text))
    assert grade.metrics.get("refused") is False
    assert grade.metrics["constraint_satisfied"] == 0
    assert grade.verdict is Verdict.WRONG and grade.score == 0.0
    assert len(grade.metrics["violated"]) == 4


def test_silence_gets_no_credit_from_the_easy_constraints():
    """空正文必须 0 分：逐条判的话 `max_chars` 与 `forbids` 会对"没有产出"判通过。"""
    case = _case("A")
    grade = _task().grade(case, _gen(""))
    assert grade.verdict is Verdict.INVALID_FORMAT and grade.score == 0.0
    assert grade.metrics["constraint_satisfied"] == 0
    assert grade.metrics["constraint_total"] == 4, "约束条数仍是这道题的分母，不许消失"
    assert grade.metrics["empty"] is True
    assert grade.invalid_format is True
    assert "thinking" not in grade.error


def test_silence_caused_by_the_thinking_budget_names_it():
    case = _case("A")
    grade = _task().grade(case, _gen("", thinking="我先想想它的字数要求"))
    assert "thinking" in grade.error and grade.metrics["empty"] is True


def test_refusal_is_labelled_but_keeps_the_measured_value():
    """拒答单独成判定，但分数按产出实测：清零是"为了故事好看"而改测量。"""
    case = _case("A")
    grade = _task().grade(case, _gen("抱歉，我无法回答这个问题。"))
    assert grade.verdict is Verdict.REFUSED
    assert grade.metrics["refused"] is True
    # 这句短话确实满足"≤40 字"和"不含 markdown"，但没含必含词、又出现了禁词
    assert grade.metrics["constraint_satisfied"] == 2
    assert grade.score == pytest.approx(0.5)


def test_an_answer_that_follows_every_rule_is_never_relabelled_as_refusal():
    """全对的答案不许被拒答启发式改判——启发式不能推翻实测。

    这条用的是一道**没有把「抱歉」列进禁词**的题：如果题目自己禁了这个词，
    那"含抱歉"就是货真价实的违反，判 WRONG 才对（见上一条测试）。
    """
    case = {
        "id": "C", "ord": 0, "kind": "instruction", "input": {"text": "请说明退款政策。"},
        "expect": {"constraints": [
            {"kind": "max_chars", "params": {"count": 40}},
            {"kind": "contains", "params": {"values": ["七天内"]}},
            {"kind": "no_markdown", "params": {}},
        ]},
        "tags": ["plain"], "meta": {},
    }
    dataset = Dataset(id="ins-c-v1", cases=(case,), upstream="test", revision="r1")
    task = InstructionFollowing(dataset, model=MODEL)
    loaded = next(task.load())
    grade = task.grade(loaded, _gen("抱歉打扰：退款请在七天内提交，款项按原路退回。"))
    assert grade.verdict is Verdict.CORRECT and grade.metrics.get("refused") is False
    assert grade.score == 1.0


def test_a_long_answer_mentioning_regret_is_not_a_refusal():
    """长答案里的"抱歉"是内容。闸门是长度，不是关键词存在与否。"""
    case = _case("B")
    text = (
        "周三下午两点在三楼会议室开会，请提前十分钟到场并且带上上季度的全部材料，"
        "另外抱歉打扰一下，本次会议还需要每位同事准备五分钟发言，发言内容请围绕风险与收益展开。"
    )
    grade = _task().grade(case, _gen(text))
    assert grade.metrics.get("refused") is False
    assert grade.verdict is not Verdict.REFUSED
    assert grade.metrics["constraint_satisfied"] >= 1, "内容确实照做了，不该一个都不算"


def test_code_fence_costs_the_format_dimension_only():
    case = _case("A")
    grade = _task().grade(case, _gen(f"```\n{REFUND}\n```"))
    assert grade.invalid_format is True
    assert grade.metrics["constraint_satisfied"] == 4, "内容仍然逐条满足，围栏只算没听话"


def test_engine_failure_is_not_an_answer():
    case = _case("A")
    grade = _task().grade(case, _gen("", status=Status.ERROR, error="connection reset"))
    assert grade.verdict is Verdict.ERROR and grade.passed is None
    assert not grade.attributable and grade.error == "connection reset"


# ── 三个口径会分叉 ─────────────────────────────────────────────────
def test_three_denominators_disagree_by_construction():
    """A 全对（4/4），B 只满足一条（1/2）：三个数分别是 0.75 / 5/6 / 0.5。

    只引用其中任何一个都会讲出不同的故事，所以三个都在指标里。
    """
    a, b = _case("A"), _case("B")
    task = _task()
    grades = [task.grade(a, _gen(REFUND)), task.grade(b, _gen("周三开会，另附一段很长的话。"))]
    assert [g.metrics["constraint_satisfied"] for g in grades] == [4, 1]
    aggregate = task.aggregate(grades)
    assert aggregate["score"] == pytest.approx(0.75)
    assert aggregate["micro_rate"] == pytest.approx(5 / 6)
    assert aggregate["all_satisfied_rate"] == pytest.approx(0.5)
    assert aggregate["constraint_total"] == 6 and aggregate["constraint_satisfied"] == 5
    assert aggregate["mean_constraints"] == pytest.approx(3.0)


def test_score_ci_is_the_same_statistic_as_score():
    a, b = _case("A"), _case("B")
    task = _task()
    grades = [task.grade(a, _gen(REFUND)), task.grade(b, _gen("随便写点东西"))]
    aggregate = task.aggregate(grades, seed=11)
    ci = aggregate["score_ci"]
    assert ci["point"] == pytest.approx(aggregate["score"])
    assert ci["n"] == aggregate["n_judged"] == 2
    assert ci["low_confidence"] is True


def test_by_kind_carries_its_own_denominator_and_omits_untested_kinds():
    """没考到的约束类型**不出现**，而不是出现一个 0：后者会让人以为模型不会用表格。"""
    a = _case("A")
    aggregate = _task().aggregate([_task().grade(a, _gen(REFUND))])
    assert set(aggregate["by_kind"]) == {"max_chars", "contains", "forbids", "no_markdown"}
    assert aggregate["by_kind"]["max_chars"] == {"satisfied": 1, "n": 1, "rate": 1.0}


def test_silence_and_refusal_show_up_in_their_own_columns():
    a = _case("A")
    task = _task()
    aggregate = task.aggregate([
        task.grade(a, _gen("")),
        task.grade(a, _gen("抱歉，我无法回答这个问题。")),
    ])
    assert aggregate["empty_outputs"] == 1
    assert aggregate["n_judged"] == 2, "没写不等于没考：这两条都在分母里"
    assert aggregate["score"] == pytest.approx((0.0 + 0.5) / 2)
    assert aggregate["refusal_rate"] == pytest.approx(0.5)
    # 空正文算"格式不合法"（没有任何产出），拒答不算：它是一句正常的话，只是没干活。
    # 两者各有各的列，挤在同一列里就分不开了。
    assert aggregate["invalid_format_rate"] == pytest.approx(0.5)
    assert aggregate["verdicts"][Verdict.INVALID_FORMAT.value] == 1
    assert aggregate["verdicts"][Verdict.REFUSED.value] == 1


def test_an_all_engine_outage_run_has_no_score_rather_than_zero():
    """全是故障时不许报 `score 0.0`：那是把"根本没测到"报成"测到 0 分"。"""
    a = _case("A")
    grade = _task().grade(a, _gen("", status=Status.ERROR, error="boom"))
    aggregate = _task().aggregate([grade, grade])
    assert aggregate["n_total"] == 2 and aggregate["n_attributable"] == 0
    assert aggregate["n_judged"] == 0
    assert aggregate["score"] is None and aggregate["micro_rate"] is None
    assert aggregate["all_satisfied_rate"] is None
    assert aggregate["by_kind"] == {}
    assert aggregate["verdicts"][Verdict.ERROR.value] == 2


def test_empty_aggregate_declares_every_metric_it_promises():
    """空输入也要产齐全部指标——看板按声明建列，缺列就是"这项一直未知"。"""
    task = _task()
    empty = task.aggregate([])
    assert set(empty) == set(task.metric_names)
    assert empty["mean_constraints"] is None and empty["constraint_total"] == 0


def test_stability_metrics_use_case_count_not_grade_count():
    a = _case("A")
    task = _task()
    good = task.grade(a, _gen(REFUND))
    bad = task.grade(a, _gen("- 没包含必含词\n- 而且出现了抱歉这样的禁词\n- 第三行"))
    aggregate = task.aggregate([good, bad, good, bad])
    assert aggregate["k"] == 4
    assert aggregate["pass_hat_k"] == 0.0 and aggregate["pass_at_k"] == 1.0
    assert aggregate["stability_gap"] == pytest.approx(1.0)
    assert aggregate["low_confidence"] is True


def test_prompt_is_neutral_and_does_not_contradict_any_constraint():
    """系统提示不能与约束打架（S30 的教训：提示词与判据必须同源）。"""
    task = _task()
    request = task.build(_case("A"))
    system = request.messages[0].content
    assert "不要解释" in system and "不要复述" in system
    assert request.thinking is False and request.params.max_tokens >= 200
    assert request.model == MODEL


def test_instruction_text_is_passed_through_verbatim():
    case = _case("A")
    assert _task().build(case).messages[1].content == "请说明退款政策。"


def test_aggregate_survives_json_round_trip():
    a, b = _case("A"), _case("B")
    task = _task()
    aggregate = task.aggregate([task.grade(a, _gen(REFUND)), task.grade(b, _gen("周三"))])
    assert json.loads(json.dumps(aggregate, ensure_ascii=False))["score_ci"]["n"] == 2
