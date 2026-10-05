"""长上下文任务的形态表与聚合（S32）。

要紧的是**四种失败必须长成四种不同的样子**：
读不到（WRONG/PARTIAL）、认错实体（`needle_confused`）、没产出（INVALID_FORMAT）、
窗口不够（SKIPPED + 原因）。
把它们混成一个"0 分"，下一步就会被拿去做错的决定——
越界的那条尤其危险：分数会指导人换模型，而该改的其实是 `--num-ctx`。

`by_position` / `by_bucket` 是这任务存在的理由：总分看不出"中部塌陷"，也看不出"过 8k 就掉"。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from onyx.core.types import (
    Confidence,
    FinishReason,
    Generation,
    Status,
    TokenSample,
    TokenSource,
)
from onyx.eval.datasets.loader import Dataset, load_builtin
from onyx.eval.task import Grade, Verdict
from onyx.eval.tasks.long_context import LongContext

MODEL = "mock/lc"
ANSWERS: dict[str, Any] = {"q1": "L-209", "q2": 2011, "q3": 12.5}
POSITIONS = {"q1": "first", "q2": "middle", "q3": "last"}
#: 每个埋点的干扰值（数据集 meta 里带的就是这两个字段）：认错实体要能被单独量出来
DISTRACTORS: dict[str, Any] = {"q1": "L-902", "q2": 1976, "q3": 15.2}


def _cases(extra: str = "") -> list[dict[str, Any]]:
    needles = [
        {"id": key, "position": POSITIONS[key], "value": ANSWERS[key],
         "distractor_value": DISTRACTORS[key]}
        for key in ("q1", "q2", "q3")
    ]
    return [
        {
            "id": f"lc-a{extra}", "ord": 0, "kind": "longctx",
            "input": {"text": "很长的正文……" + extra},
            "expect": {"answers": dict(ANSWERS), "keys": ["q1", "q2", "q3"],
                       "positions": dict(POSITIONS)},
            "tags": ["4k"], "meta": {"bucket": "4k", "needles": needles},
        },
        {
            "id": f"lc-b{extra}", "ord": 1, "kind": "longctx",
            "input": {"text": "另一篇长文……" + extra},
            "expect": {"answers": dict(ANSWERS), "keys": ["q1", "q2", "q3"],
                       "positions": dict(POSITIONS)},
            "tags": ["16k"], "meta": {"bucket": "16k", "needles": needles},
        },
    ]


def _task(**kw: Any) -> LongContext:
    dataset = Dataset(id="lc-tiny-v1", cases=tuple(_cases()), upstream="test", revision="r1")
    return LongContext(dataset, model=MODEL, **kw)


def _grade(task: LongContext, case_id: str, text: str, **kw: Any) -> Grade:
    case = next(c for c in task.load() if c.id == case_id)
    status = kw.pop("status", Status.OK)
    in_tokens = kw.pop("in_tokens", 1500)
    usage = kw.pop("usage", None)
    if usage is None and in_tokens is not None:
        usage = (TokenSample(source=TokenSource.ENGINE, in_tokens=in_tokens,
                             confidence=Confidence.HIGH),)
    return task.grade(case, Generation(
        text=text, model=MODEL, status=status, usage=usage or (), **kw
    ))


def _dump(answers: dict[str, Any]) -> str:
    return json.dumps(answers, ensure_ascii=False)


# ── 形态表 ─────────────────────────────────────────────────────────
def test_all_three_needles_found_is_correct():
    grade = _grade(_task(), "lc-a", _dump(ANSWERS))
    assert grade.verdict is Verdict.CORRECT and grade.passed is True
    assert grade.score == 1.0 and grade.invalid_format is False
    assert grade.metrics["needle_matched"] == grade.metrics["needle_total"] == 3
    assert grade.metrics["per_needle_ok"] == {"q1": True, "q2": True, "q3": True}


def test_partial_retrieval_is_partial_not_wrong():
    """找到 1/3 与一个都没找到修法不同：前者是位置/注意力，后者是没读或格式坏。"""
    grade = _grade(_task(), "lc-a", _dump({"q2": 2011}))
    assert grade.verdict is Verdict.PARTIAL
    assert grade.score == pytest.approx(1 / 3)
    # 没答的那两题也各有判定：漏答与答错在 by_position 里都要算进分母
    assert grade.metrics["per_needle_ok"] == {"q1": False, "q2": True, "q3": False}
    assert sorted(grade.metrics["missing"]) == ["q1", "q3"]


def test_numeric_answer_tolerates_json_number_style():
    grade = _grade(_task(), "lc-a", _dump({"q1": "L-209", "q2": 2011.0, "q3": 12.5}))
    assert grade.metrics["needle_matched"] == 3


def test_wrong_value_is_a_miss_not_a_format_problem():
    grade = _grade(_task(), "lc-a", _dump({"q1": "L-290", "q2": 2011, "q3": 12.5}))
    assert grade.metrics["per_needle_ok"]["q1"] is False
    assert grade.metrics["needle_matched"] == 2
    assert grade.invalid_format is False


def test_invented_extra_key_blocks_a_correct_verdict():
    """多答一个键就不算"完全照做"：下游按固定 schema 读时会炸。"""
    grade = _grade(_task(), "lc-a", _dump(dict(ANSWERS, q4="X-1")))
    assert grade.verdict is Verdict.PARTIAL and grade.passed is False
    assert grade.metrics["unexpected"] == ["q4"]


def test_unparseable_output_stops_at_the_format_layer():
    grade = _grade(_task(), "lc-a", "q1 是 L-209，q2 是 2011，q3 大约 12.5")
    assert grade.verdict is Verdict.INVALID_FORMAT and grade.invalid_format is True
    assert grade.metrics["json_valid"] is False
    assert grade.metrics["needle_total"] == 3 and grade.metrics["needle_matched"] == 0


def test_code_fence_costs_format_but_not_content():
    grade = _grade(_task(), "lc-a", f"```json\n{_dump(ANSWERS)}\n```")
    assert grade.verdict is Verdict.CORRECT and grade.invalid_format is True


def test_empty_body_names_the_thinking_budget_when_it_was_eaten():
    task = _task()
    plain = _grade(task, "lc-a", "", in_tokens=None)
    eaten = _grade(task, "lc-a", "", thinking="让我先把三段读一遍")
    assert plain.verdict is Verdict.INVALID_FORMAT and "thinking" not in plain.error
    assert "thinking" in eaten.error


def test_refusal_is_its_own_verdict():
    grade = _grade(_task(), "lc-a", "抱歉，文中没有提到这些数值。")
    assert grade.verdict is Verdict.REFUSED and grade.invalid_format is True


def test_answer_clipped_by_max_tokens_says_so():
    """被预算切断要说"提高 max_tokens"，不能让人以为模型读不到。"""
    grade = _grade(_task(), "lc-a", '{"q1": "L-209", "q2"', finish_reason=FinishReason.LENGTH)
    assert grade.metrics["clipped"] is True
    assert "max_tokens" in grade.error


def test_engine_failure_stays_out_of_the_capability_denominator():
    grade = _grade(_task(), "lc-a", "", status=Status.ERROR, error="connection reset",
                   in_tokens=None)
    assert grade.verdict is Verdict.ERROR and not grade.attributable


# ── 越界：不记分 ───────────────────────────────────────────────────
def test_engine_number_below_the_document_floor_is_read_as_truncation():
    """真机形状：`--num-ctx 4096` 跑 16k 档，引擎回报 2050 tok——**比窗口还小**。

    "in_tokens ≥ 窗口"这条判据在这种情况下永远不响，于是三条 16k 样本被判成 partial、
    `score 0.000`，而那一次跑出的 `per_needle_ok` 是 `{q1:F, q2:F, q3:T}`：
    只有结尾的埋点活下来，正是"开头被切掉"的形状。判据必须拿正文自己的下限去比。
    """
    raw = _cases()[0]
    raw["input"] = {"text": "记录" * 8400}  # 16,800 汉字 ⇒ 下限 8,400 tok
    dataset = Dataset(id="lc-long-v1", cases=(raw,), upstream="test", revision="r1")
    task = LongContext(dataset, model=MODEL, num_ctx=4096)

    grade = _grade(task, "lc-a", _dump(ANSWERS), in_tokens=2050)
    assert grade.verdict is Verdict.SKIPPED and grade.passed is None
    assert grade.metrics["truncated"] is True
    assert grade.metrics["min_prompt_tokens"] == 8400
    assert grade.metrics["shrink"] == pytest.approx(2050 / 8400, abs=1e-4)
    assert "2050" in grade.error and "8400" in grade.error and "num-ctx" in grade.error
    assert task.aggregate([grade])["score"] is None, "被切的样本不许进分子也不许进分母"
    assert task.aggregate([grade])["n_truncated"] == 1

    # 同一篇正文，窗口够大时数字就正常：判据不是"越长越 skip"
    healthy = _grade(LongContext(dataset, model=MODEL, num_ctx=20480), "lc-a", _dump(ANSWERS),
                     in_tokens=16755)
    assert healthy.verdict is Verdict.CORRECT and not healthy.metrics.get("truncated")


def test_input_over_the_window_is_skipped_with_an_actionable_reason():
    grade = _grade(_task(num_ctx=2048), "lc-a", _dump(ANSWERS), in_tokens=3000)
    assert grade.verdict is Verdict.SKIPPED and grade.passed is None
    assert grade.metrics["truncated"] is True
    assert "num-ctx" in grade.error and "3000" in grade.error and "2048" in grade.error


def test_truncated_cases_leave_the_score_denominator_but_stay_countable():
    """`n_truncated` 必须存在：否则"16k 全错"与"16k 根本没测"在数字上同形。"""
    task = _task(num_ctx=2048)
    aggregate = task.aggregate([
        _grade(task, "lc-a", _dump(ANSWERS), in_tokens=3000),
        _grade(task, "lc-b", _dump(ANSWERS), in_tokens=3000),
    ])
    assert aggregate["n_total"] == 2 and aggregate["n_attributable"] == 0
    assert aggregate["n_truncated"] == 2
    assert aggregate["score"] is None, "全部越界时不许报 0 分"
    assert aggregate["verdicts"][Verdict.SKIPPED.value] == 2
    # 占用率把越界那条也算进来：它正是"该调窗口"的那个证据
    assert aggregate["max_ctx_util"] == pytest.approx(3000 / 2048, abs=1e-3)


def test_no_engine_usage_means_no_truncation_claim():
    """引擎没回报 in_tokens 就不做越界判断——用估算冒充测量会凭空造出"被截断"的结论。"""
    grade = _grade(_task(num_ctx=2048), "lc-a", _dump(ANSWERS), in_tokens=None)
    assert grade.verdict is Verdict.CORRECT
    assert grade.metrics["in_tokens"] is None and grade.metrics["ctx_util"] is None
    aggregate = _task(num_ctx=2048).aggregate([grade])
    assert aggregate["n_truncated"] == 0 and aggregate["reported_in_tokens"] == 0
    assert aggregate["mean_in_tokens"] is None


def test_heuristic_token_count_is_not_treated_as_engine_number():
    """启发式估算的 30,000 tok 不算引擎数字：不许凭它判越界，也不许进 in_tokens 字段。

    保真阶梯上"数字没有出处就不许上看板"这一条，在长上下文里的后果最具体——
    拿估算判越界会批量把可判样本变成 skip，而 skip 是"没测"，不是"测出来说不行"。
    """
    estimates = (
        TokenSample(source=TokenSource.HEURISTIC, in_tokens=30000, confidence=Confidence.LOW),
        TokenSample(source=TokenSource.ENGINE, ok=False, in_tokens=30000,
                    confidence=Confidence.HIGH),
    )
    for sample in estimates:
        grade = _grade(_task(num_ctx=2048), "lc-a", _dump(ANSWERS), usage=(sample,))
        assert grade.verdict is Verdict.CORRECT, f"{sample.source} 被当成了引擎数字"
        assert grade.metrics["in_tokens"] is None


# ── 聚合的分桶 ─────────────────────────────────────────────────────
def test_by_position_and_by_bucket_carry_their_own_denominators():
    task = _task()
    grades = [
        _grade(task, "lc-a", _dump({"q1": "L-209"})),
        _grade(task, "lc-b", _dump({"q2": 2011, "q3": 12.5})),
    ]
    aggregate = task.aggregate(grades)
    by_position = aggregate["by_position"]
    assert {slot: entry["n"] for slot, entry in by_position.items()} == {
        "first": 2, "middle": 2, "last": 2
    }
    assert by_position["first"] == {"matched": 1, "confused": 0, "n": 2, "rate": 0.5}
    # 总分 0.5 里藏着两种完全不同的失败：漏答也算进它所属位置的分母，分母才不会骗人
    assert by_position["middle"] == {"matched": 1, "confused": 0, "n": 2, "rate": 0.5}
    assert by_position["last"] == {"matched": 1, "confused": 0, "n": 2, "rate": 0.5}
    assert list(by_position) == ["first", "middle", "last"], "位置顺序是界面上的读序"
    assert set(aggregate["by_bucket"]) == {"4k", "16k"}
    assert aggregate["by_bucket"]["4k"]["all_correct_rate"] == 0.0
    assert aggregate["by_bucket"]["4k"]["needle_rate"] == pytest.approx(1 / 3)
    assert aggregate["by_bucket"]["16k"]["needle_rate"] == pytest.approx(2 / 3)


def test_answering_the_distractor_is_counted_as_confusion():
    """答成干扰值 = 读到了那一段却认错了实体，与"没读到"是两种病，要分开记。

    第一版数据没有干扰项，真机 9/9 全对（qwen3.5:9b）——那说明"扫到任意一个数字"就能得分。
    这一条是干扰项存在的唯一理由：它必须能被骗出来，也必须被单独量出来。
    """
    task = _task()
    grade = _grade(task, "lc-a", _dump({"q1": "L-902", "q2": 2011, "q3": 12.5}))
    assert grade.verdict is Verdict.PARTIAL
    assert grade.metrics["per_needle_ok"] == {"q1": False, "q2": True, "q3": True}
    assert grade.metrics["needle_confused"] == 1
    assert grade.metrics["per_needle_confused"] == {"q1": True}
    aggregate = task.aggregate([grade])
    assert aggregate["needle_confused"] == 1
    # 分母是"答错的那些"（3−2=1），不是全部埋点：这一位读作"错的全是认错实体"
    assert aggregate["confusion_rate"] == pytest.approx(1.0)


def test_a_random_wrong_number_is_missed_not_confused():
    """凭空编一个数字不该算成"认错实体"：那会把问题归到文档排布上，而病在检索。"""
    task = _task()
    grade = _grade(task, "lc-a", _dump({"q1": "X-777", "q2": 2011, "q3": 12.5}))
    assert grade.metrics["per_needle_confused"] == {"q1": False}
    assert grade.metrics["needle_confused"] == 0
    aggregate = task.aggregate([grade])
    assert aggregate["needle_confused"] == 0
    assert aggregate["confusion_rate"] == pytest.approx(0.0)


def test_missing_answer_is_not_counted_as_confusion():
    """漏答不是"认错实体"：键根本没出现时混淆位必须是 0，否则归因会指向文档排布。"""
    task = _task()
    grade = _grade(task, "lc-a", _dump({"q2": 2011, "q3": 12.5}))
    assert grade.metrics["needle_confused"] == 0
    assert grade.metrics["per_needle_confused"] == {"q1": False}
    # 有 1 个埋点没拿到，而那 1 个不是干扰值 ⇒ 混淆占比是 0 而不是「—」
    assert task.aggregate([grade])["confusion_rate"] == pytest.approx(0.0)


def test_all_correct_leaves_confusion_rate_undefined():
    """全对时"错的那部分"是空集，占比必须写「—」而不是 0——0 会被读成"没混淆"。"""
    task = _task()
    aggregate = task.aggregate([_grade(task, "lc-a", _dump(ANSWERS))])
    assert aggregate["needle_confused"] == 0 and aggregate["confusion_rate"] is None


def test_confusion_is_located_in_the_position_it_happened_at():
    """`by_position` 里的 `confused` 让"中部塌陷"能进一步归因：是认错实体还是没读到。"""
    task = _task()
    aggregate = task.aggregate([
        _grade(task, "lc-a", _dump({"q1": "L-209", "q2": 1976, "q3": 15.2})),
        _grade(task, "lc-b", _dump({"q1": "X-1", "q2": 2011, "q3": 12.5})),
    ])
    by_position = aggregate["by_position"]
    assert by_position["first"] == {"matched": 1, "confused": 0, "n": 2, "rate": 0.5}
    assert by_position["middle"] == {"matched": 1, "confused": 1, "n": 2, "rate": 0.5}
    assert by_position["last"] == {"matched": 1, "confused": 1, "n": 2, "rate": 0.5}


def test_dataset_without_distractors_records_no_confusion():
    """外部数据没带干扰项时不猜：混淆表是空的，占比写「—」而不是 0。

    "0" 会被读成"没有认错实体这回事"，而真实情况是"没这个信息"——
    这是「未知 ≠ 0 分」在这一层的形态。
    """
    raw = dict(_cases()[0])
    raw["meta"] = {"bucket": "4k"}
    dataset = Dataset(id="lc-plain-v1", cases=(raw,), upstream="test", revision="r1")
    task = LongContext(dataset, model=MODEL)
    grade = _grade(task, "lc-a", _dump({"q1": "L-902", "q2": 2011, "q3": 12.5}))
    assert grade.metrics["per_needle_confused"] == {}
    assert grade.metrics["confusable_misses"] == 0
    aggregate = task.aggregate([grade])
    assert aggregate["needle_confused"] == 0 and aggregate["confusion_rate"] is None
    assert aggregate["by_position"]["first"]["confused"] == 0


def test_score_and_needle_rate_are_different_numbers_on_purpose():
    """主分数要"三个全找到"，needle_rate 是逐埋点命中率：差值就是"只找到一个"的那批题。"""
    task = _task()
    aggregate = task.aggregate([
        _grade(task, "lc-a", _dump({"q1": "L-209"})),
        _grade(task, "lc-b", _dump({"q1": "L-209", "q2": 2011, "q3": 12.5})),
    ])
    assert aggregate["score"] == pytest.approx(0.5)
    assert aggregate["needle_rate"] == pytest.approx(4 / 6)
    assert aggregate["score_ci"]["point"] == pytest.approx(aggregate["score"])
    assert aggregate["score_ci"]["n"] == aggregate["n_judged"] == 2


def test_declared_metrics_match_both_empty_and_real_aggregates():
    task = _task()
    declared = set(task.metric_names)
    assert set(task.aggregate([])) == declared
    assert set(task.aggregate(_grades_for(task))) == declared


def _grades_for(task: LongContext) -> list[Grade]:
    return [_grade(task, "lc-a", _dump(ANSWERS)), _grade(task, "lc-b", "不是 JSON")]


def test_window_and_build_params_are_visible():
    """窗口要出现在请求里：否则"这次跑在多大的窗上"只能靠猜。"""
    task = _task(num_ctx=8192)
    request = task.build(next(iter(task.load())))
    assert request.params.num_ctx == 8192
    assert request.thinking is False and request.params.max_tokens == 64
    assert len(request.messages) == 1
    assert task.aggregate([])["window_tokens"] == 8192


def test_real_dataset_loads_three_buckets():
    task = LongContext(load_builtin("longctx_zh"), model=MODEL)
    cases = list(task.load())
    assert len(cases) == 9
    assert {case.meta["bucket"] for case in cases} == {"4k", "8k", "16k"}
    for case in cases:
        assert list(case.expect["keys"]) == ["q1", "q2", "q3"]
        assert case.expect["positions"]["q1"] == "first"
    assert list(task.load(split="16k", limit=1))
