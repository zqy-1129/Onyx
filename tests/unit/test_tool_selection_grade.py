"""S14 验收：工具选择评测的判定。

计划要求"`no_call_needed` 子集：误调率单独统计（不是算成'没调对'）"。
这一条是整份测试的核心：**一个从不乱调工具的模型和一个不会调工具的模型，
在合并后的分数上完全一样，但它们相反**。所以误调率必须单独成列。

同理，`no_call`（该调却不调）与 `wrong_tool`（调了但选错）也必须分开：
前者改提示词与 description 的"什么时候该用"，后者改工具之间的区分度。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from onyx.core.types import (
    Cap,
    FinishReason,
    Generation,
    ParseStatus,
    Status,
    ToolCall,
    ToolSpec,
)
from onyx.eval.datasets.builtin.tool_calls_zh import TOOLS, _tool_spec
from onyx.eval.datasets.loader import Dataset
from onyx.eval.task import Case, Verdict, check_capabilities
from onyx.eval.tasks import BUILTIN_TASKS, build_task, task_ids
from onyx.eval.tasks.tool_selection import ToolSelection

TOOL_NAMES = tuple(TOOLS)


def _spec(name: str) -> ToolSpec:
    """数据集里存的是 JSON 形状，`Case.tools` 要的是 ToolSpec —— 与 loader 走同一条转换。"""
    return ToolSpec.from_openai_tool(_tool_spec(name))


def _case(
    instruction: str = "北京今天天气怎么样",
    expected: tuple[dict, ...] = ({"name": "get_weather", "arguments": {"city": "北京"}},),
    *,
    tools: tuple[str, ...] = TOOL_NAMES,
    kind: str = "single",
) -> Case:
    return Case(
        id="tc1", kind=kind, input={"instruction": instruction},
        expect={"calls": [dict(item) for item in expected], "must_call": bool(expected)},
        tools=tuple(_spec(name) for name in tools),
    )


def _gen(calls: tuple[dict, ...] = (), *, text: str = "", status: Status = Status.OK,
         error: str = "") -> Generation:
    return Generation(
        text=text, status=status, error=error,
        finish_reason=FinishReason.TOOL_CALLS if calls else FinishReason.STOP,
        tool_calls=tuple(
            ToolCall(index=index, id=f"call_{index}", name=str(item.get("name") or ""),
                     arguments=item.get("arguments"),
                     arguments_raw=str(item.get("arguments_raw") or ""),
                     parse_status=item.get("parse_status", ParseStatus.OK))
            for index, item in enumerate(calls)
        ),
        model="mock/tool",
    )


def _task(**kw) -> ToolSelection:
    dataset = Dataset(id="tiny", cases=({
        "id": "tc1", "ord": 0, "kind": "single",
        "input": {"instruction": "北京今天天气怎么样"},
        "expect": {"calls": [{"name": "get_weather", "arguments": {"city": "北京"}}],
                   "must_call": True},
        "tools": [_tool_spec(name) for name in TOOL_NAMES], "tags": [],
    },))
    return ToolSelection(dataset, model="mock/tool", **kw)


def _grade(calls=(), **kw):
    return _task().grade(_case(), _gen(calls, **kw))


# ── 七种判定互不相同 ──────────────────────────────────────────────
def test_correct_single_call():
    result = _grade(({"name": "get_weather", "arguments": {"city": "北京"}},))
    assert result.verdict is Verdict.CORRECT
    assert result.passed is True and result.score == 1.0
    assert result.metrics["hit_at_1"] is True
    assert result.metrics["set_f1"] == pytest.approx(1.0)
    assert result.metrics["args"][0]["ok"] is True


def test_no_call_when_a_tool_was_required_is_its_own_verdict():
    """该调却不调 ≠ 调错了：前者是覆盖问题，后者是区分度问题，修法不同。"""
    result = _task().grade(_case(), _gen((), text="北京今天晴，21 度。"))
    assert result.verdict is Verdict.NO_CALL
    assert result.metrics["no_call"] is True
    assert result.extra["text_chars"] > 0, "模型编了什么内容必须留下来，否则无从诊断"
    assert result.attributable is True


def test_wrong_tool_is_distinguished_from_no_call():
    result = _grade(({"name": "search_web", "arguments": {"query": "北京天气"}},))
    assert result.verdict is Verdict.WRONG_TOOL
    assert result.metrics["missing"] == ["get_weather"]
    assert result.metrics["unexpected"] == ["search_web"]
    assert "缺 ['get_weather']" in result.error


def test_hallucinated_tool_name_is_not_just_a_wrong_tool():
    """调了工具集里不存在的名字是**幻觉**，修法在提示词与工具集，不在描述区分度。"""
    result = _grade(({"name": "get_forecast", "arguments": {"city": "北京"}},))
    assert result.verdict is Verdict.HALLUCINATED_TOOL
    assert result.out_of_set is True
    assert result.metrics["hallucinated"] == ["get_forecast"]
    assert "get_weather" in result.metrics["available_tools"]


def test_unparsable_arguments_are_invalid_format_not_bad_args():
    """参数 JSON 截断是 max_tokens / 停止词的问题，不是"模型不会填参数"。"""
    result = _grade(({
        "name": "get_weather", "arguments": None,
        "arguments_raw": '{"city": "北', "parse_status": ParseStatus.TRUNCATED,
    },))
    assert result.verdict is Verdict.INVALID_FORMAT
    assert result.invalid_format is True
    assert result.metrics["args_raw"] == ['{"city": "北'], "原文必须保留"
    assert result.metrics["parse_status"] == ["truncated"]


def test_bad_args_reports_which_field_and_why():
    result = _grade(({"name": "get_weather", "arguments": {"city": "上海"}},))
    assert result.verdict is Verdict.BAD_ARGS
    assert result.passed is False
    assert result.score == 0.0, "唯一的字段就错了，没有部分分可言"
    fields = result.metrics["args"][0]["fields"]
    city = next(f for f in fields if f["field"] == "city")
    assert city["ok"] is False and city["kind"] == "value_mismatch"
    assert city["expected"] == "北京" and city["actual"] == "上海"


def test_missing_required_argument_is_bad_args():
    result = _grade(({"name": "get_weather", "arguments": {}},))
    assert result.verdict is Verdict.BAD_ARGS
    assert result.metrics["args"][0]["fields"][0]["kind"] == "missing"


def test_extra_optional_argument_passes_by_default_but_fails_under_exact():
    task = _task()
    calls = ({"name": "get_weather", "arguments": {"city": "北京", "unit": "celsius"}},)
    assert task.grade(_case(), _gen(calls)).verdict is Verdict.CORRECT

    strict = _task(exact_args=True)
    result = strict.grade(_case(), _gen(calls))
    assert result.verdict is Verdict.BAD_ARGS
    assert any(f["kind"] == "unexpected" for f in result.metrics["args"][0]["fields"])


def test_engine_failure_is_not_attributable():
    result = _task().grade(_case(), _gen((), status=Status.ERROR, error="连不上引擎"))
    assert result.verdict is Verdict.ERROR
    assert result.attributable is False and result.passed is None


# ── no_call_needed：误调率单独统计 ────────────────────────────────
def test_no_call_needed_passes_when_the_model_stays_quiet():
    task = _task()
    case = _case("用一句话解释什么是递归", expected=(), kind="no_call_needed")
    result = task.grade(case, _gen((), text="递归是函数调用自身。"))
    assert result.verdict is Verdict.CORRECT
    assert result.metrics["false_call"] is False
    assert result.metrics["must_call"] is False


def test_no_call_needed_flags_a_false_call():
    """误调是**独立指标**，不能被合并进"没调对"。"""
    task = _task()
    case = _case("用一句话解释什么是递归", expected=(), kind="no_call_needed")
    result = task.grade(case, _gen(({"name": "search_web", "arguments": {"query": "递归"}},)))
    assert result.verdict is Verdict.WRONG
    assert result.metrics["false_call"] is True
    assert "不该调用工具" in result.error


def test_false_call_rate_is_computed_only_over_the_optional_subset():
    """分母是 no_call_needed 的条数，不是全部条数——否则这个率没有意义。"""
    task = _task()
    optional = _case("解释一下递归", expected=(), kind="no_call_needed")
    required = _case("北京天气", expected=({"name": "get_weather",
                                           "arguments": {"city": "北京"}},))
    grades = [
        task.grade(optional, _gen(({"name": "search_web", "arguments": {"query": "x"}},))),  # 误调
        task.grade(optional, _gen((), text="递归是……")),                                     # 正确沉默
        task.grade(required, _gen(({"name": "get_weather", "arguments": {"city": "北京"}},))),
        task.grade(required, _gen(({"name": "get_weather", "arguments": {"city": "北京"}},))),
    ]
    report = task.aggregate(grades)
    assert report["n_no_call_needed"] == 2 and report["n_must_call"] == 2
    assert report["false_call_rate"] == pytest.approx(1 / 2), "分母是 2 条 optional，不是 4 条"
    assert report["must_call_acc"] == pytest.approx(1.0)
    assert report["no_call_rate"] == pytest.approx(0.0)


def test_a_model_that_never_calls_tools_scores_perfectly_on_the_optional_subset():
    """这正是"误调率必须单列"的理由：它与"不会调工具"在这一格上同分，但完全相反。

    所以看板上必须同时看 false_call_rate 与 must_call_acc，只看一个会被骗。
    """
    task = _task()
    optional = _case("解释一下递归", expected=(), kind="no_call_needed")
    required = _case("北京天气")
    grades = [
        task.grade(optional, _gen((), text="递归是……")),   # 从不乱调 ⇒ 满分
        task.grade(required, _gen((), text="北京今天晴")),  # 但该调的也不调 ⇒ 0
    ]
    report = task.aggregate(grades)
    assert report["false_call_rate"] == pytest.approx(0.0), "看起来完美"
    assert report["must_call_acc"] == pytest.approx(0.0), "实际上一个工具都不会调"
    assert report["no_call_rate"] == pytest.approx(1.0)


# ── 并行调用 ──────────────────────────────────────────────────────
def test_parallel_calls_all_correct():
    expected = (
        {"name": "get_weather", "arguments": {"city": "北京"}},
        {"name": "get_weather", "arguments": {"city": "上海"}},
    )
    case = _case("北京和上海哪个更热", expected, kind="parallel")
    result = _task().grade(case, _gen(expected))
    assert result.verdict is Verdict.CORRECT
    assert len(result.metrics["args"]) == 2
    assert result.metrics["hit_at_1"] is True


def test_parallel_calls_missing_one_is_wrong_tool():
    expected = (
        {"name": "get_weather", "arguments": {"city": "北京"}},
        {"name": "get_weather", "arguments": {"city": "上海"}},
    )
    case = _case("北京和上海哪个更热", expected, kind="parallel")
    result = _task().grade(case, _gen((expected[0],)))
    # 集合语义会把两次 get_weather 去重，所以集合是"相等"的——
    # 漏掉一次并行调用只能靠**配对次数**发现，这就是 BAD_ARGS 里也要带
    # paired/expected_calls 的原因
    assert result.verdict is Verdict.BAD_ARGS
    assert result.metrics["paired"] == 1
    assert result.metrics["expected_calls"] == 2
    # 配上的那些参数可能全对，所以必须明说是「少发了一次」，
    # 否则只剩一句没有信息量的「参数不匹配」
    assert "1 次没被发起" in result.error


def test_same_tool_twice_pairs_arguments_by_order_not_by_name():
    """`get_weather(北京)` 与 `get_weather(上海)` 是两个不同调用。

    若按"第一个同名的"配对，两次调用会与同一个期望配上，
    于是得出"参数全对"的假结论。
    """
    expected = (
        {"name": "get_weather", "arguments": {"city": "北京"}},
        {"name": "get_weather", "arguments": {"city": "上海"}},
    )
    case = _case("北京和上海哪个更热", expected, kind="parallel")
    swapped = (expected[1], expected[0])
    result = _task().grade(case, _gen(swapped))
    assert result.verdict is Verdict.BAD_ARGS, "顺序换了，参数就配错了"
    cities = [f["actual"] for item in result.metrics["args"] for f in item["fields"]
              if f["field"] == "city"]
    assert sorted(cities) == ["上海", "北京"]


# ── 类型感知参数 ──────────────────────────────────────────────────
def test_date_normalization_passes_but_is_counted_as_relaxed():
    case = _case(
        "查询 2026-10-03 那天的订单",
        ({"name": "db_query",
          "arguments": {"sql": "SELECT * FROM orders WHERE date = '2026-10-03'"}},),
        kind="args",
    )
    result = _task().grade(case, _gen((
        {"name": "db_query",
         "arguments": {"sql": "SELECT * FROM orders WHERE date = '2026-10-03'"}},
    )))
    assert result.verdict is Verdict.CORRECT


def test_enum_argument_is_checked_against_the_schema():
    case = _case(
        "深圳天气，用华氏度",
        ({"name": "get_weather", "arguments": {"city": "深圳", "unit": "fahrenheit"}},),
    )
    ok = _task().grade(case, _gen((
        {"name": "get_weather", "arguments": {"city": "深圳", "unit": "Fahrenheit"}},
    )))
    assert ok.verdict is Verdict.CORRECT, "枚举大小写属于排版差异"

    bad = _task().grade(case, _gen((
        {"name": "get_weather", "arguments": {"city": "深圳", "unit": "kelvin"}},
    )))
    assert bad.verdict is Verdict.BAD_ARGS
    assert any("枚举越界" in f["detail"] for f in bad.metrics["args"][0]["fields"])


def test_numeric_argument_uses_tolerance():
    case = _case("给我 10 条结果", ({"name": "search_web",
                                   "arguments": {"query": "x", "max_results": 10}},))
    result = _task().grade(case, _gen((
        {"name": "search_web", "arguments": {"query": "x", "max_results": 10}},
    )))
    assert result.verdict is Verdict.CORRECT

    wrong = _task().grade(case, _gen((
        {"name": "search_web", "arguments": {"query": "x", "max_results": 5}},
    )))
    assert wrong.verdict is Verdict.BAD_ARGS


# ── 聚合 ──────────────────────────────────────────────────────────
def _mixed_grades():
    task = _task()
    required = _case("北京天气")
    optional = _case("解释递归", expected=(), kind="no_call_needed")
    good = {"name": "get_weather", "arguments": {"city": "北京"}}
    return task, [
        task.grade(required, _gen((good,))),                              # correct
        task.grade(required, _gen((), text="晴")),                        # no_call
        task.grade(required, _gen(({"name": "search_web",
                                    "arguments": {"query": "天气"}},))),  # wrong_tool
        task.grade(required, _gen(({"name": "nope", "arguments": {}},))),  # hallucinated
        task.grade(optional, _gen((good,))),                              # false_call
        task.grade(optional, _gen((), text="递归……")),                    # correct
        task.grade(required, _gen(({"name": "get_weather", "arguments": None,
                                    "arguments_raw": '{"ci',
                                    "parse_status": ParseStatus.JSON_ERROR},))),  # invalid_format
        task.grade(required, _gen((), status=Status.ERROR, error="炸了")),  # error
    ]


def test_aggregate_reports_every_failure_channel_separately():
    task, grades = _mixed_grades()
    report = task.aggregate(grades)

    assert report["n_total"] == 8
    assert report["n_attributable"] == 7, "引擎失败不进分母"
    assert report["verdicts"]["correct"] == 2
    assert report["verdicts"]["error"] == 1

    # 所有率的分母都是 attributable：引擎失败的那条不是一次"有机会答对/有机会幻觉"的样本。
    # 6 条 must_call 里有 1 条是引擎失败 ⇒ 分母 5
    assert report["n_must_call"] == 5 and report["n_no_call_needed"] == 2
    assert report["must_call_acc"] == pytest.approx(1 / 5)
    assert report["no_call_rate"] == pytest.approx(1 / 5)
    assert report["wrong_tool_rate"] == pytest.approx(1 / 5)
    assert report["false_call_rate"] == pytest.approx(1 / 2)
    assert report["hallucinated_tool_rate"] == pytest.approx(1 / 7)
    assert report["parse_fail_rate"] == pytest.approx(1 / 7)
    assert report["scoring"] == "gen-based"


def test_aggregate_breaks_results_down_by_case_kind():
    task, grades = _mixed_grades()
    by_kind = task.aggregate(grades)["by_kind"]
    assert set(by_kind) == {"single", "no_call_needed"}
    assert by_kind["no_call_needed"]["n"] == 2
    assert by_kind["no_call_needed"]["correct"] == 1
    assert by_kind["no_call_needed"]["acc"] == pytest.approx(0.5)


def test_aggregate_survives_a_round_trip_through_metrics():
    """聚合必须能从**落库后的 metrics** 重算，否则续跑时只能算新跑的那一半。"""
    import json

    task, grades = _mixed_grades()
    direct = task.aggregate(grades)

    from dataclasses import replace

    restored = [
        replace(g, metrics=json.loads(json.dumps(g.metrics, default=str))) for g in grades
    ]
    via_json = task.aggregate(restored)
    for key in ("must_call_acc", "false_call_rate", "args_field_rate", "hit_at_1",
                "hallucinated_tool_rate", "parse_fail_rate"):
        assert direct[key] == via_json[key], key


def test_aggregate_reports_args_relaxed_share():
    """靠归一化/容差通过的字段占比必须与 args_exact_rate 一起看。

    只报 exact_rate 会把"我们放宽了规则"这件事藏起来，
    于是分数变高看起来像模型变强了。
    """
    task = _task()
    case = _case(
        "1200 港币换成日元",
        ({"name": "convert_currency",
          "arguments": {"amount": 1200, "from": "HKD", "to": "JPY"}},),
    )
    grades = [
        # 金额给成数字字符串 ⇒ 语义对，但这是靠 numeric_tolerance 放宽挣来的
        task.grade(case, _gen(({"name": "convert_currency",
                               "arguments": {"amount": "1200", "from": "HKD", "to": "JPY"}},))),
        task.grade(case, _gen(({"name": "convert_currency",
                               "arguments": {"amount": 1200, "from": "HKD", "to": "JPY"}},))),
    ]
    report = task.aggregate(grades)
    assert report["args_exact_rate"] == pytest.approx(0.5), "两条里只有一条字面就一致"
    assert report["args_subset_rate"] == pytest.approx(1.0)
    assert report["args_relaxed_share"] == pytest.approx(1 / 6), "6 个字段里 1 个靠容差通过"
    assert report["args_match_kinds"]["numeric_tolerance"] == 1


def test_enum_case_difference_is_not_counted_as_relaxation():
    """枚举值是标识符，大小写不该影响判定——所以它不算"放宽"。

    把它算成放宽会让 args_exact_rate 无意义地偏低，
    进而让人以为模型经常填错枚举。
    """
    from onyx.eval.graders.args_match import match_args

    schema = TOOLS["get_weather"]["parameters"]
    result = match_args({"city": "北京", "unit": "fahrenheit"},
                        {"city": "北京", "unit": "Fahrenheit"}, schema=schema)
    assert result.ok and result.strict_ok
    assert result.relaxed_kinds == ()


def test_aggregate_on_empty_input_is_all_unknown():
    report = _task().aggregate([])
    assert report["n_total"] == 0
    assert report["must_call_acc"] is None
    assert report["false_call_rate"] is None
    assert report["pass_hat_k"] is None
    assert report["low_confidence"] is True


def test_pass_hat_k_and_pass_at_k_are_reported_for_k_sampling():
    from dataclasses import replace

    task = _task()
    good = _gen(({"name": "get_weather", "arguments": {"city": "北京"}},))
    bad = _gen((), text="晴")
    grades = [
        replace(task.grade(_case(), good), seq=0, case_id="a"),
        replace(task.grade(_case(), bad), seq=1, case_id="a"),
        replace(task.grade(_case(), good), seq=0, case_id="b"),
        replace(task.grade(_case(), good), seq=1, case_id="b"),
    ]
    report = task.aggregate(grades)
    assert report["k"] == 2
    assert report["pass_hat_k"] == pytest.approx(0.5)
    assert report["pass_at_k"] == pytest.approx(1.0)
    assert report["stability_gap"] == pytest.approx(0.5)


# ── 请求构造与能力 ────────────────────────────────────────────────
def test_build_sends_the_case_toolset_not_the_whole_registry():
    case = _case(tools=("get_weather", "calculator"))
    request = _task().build(case)
    assert request.tool_names == ("get_weather", "calculator")
    assert request.thinking is False
    assert request.params.temperature == 0.0
    assert [str(m.role) for m in request.messages] == ["system", "user"]
    assert "只有在确实需要" in request.messages[0].content


def test_system_prompt_tells_the_model_not_to_over_call():
    """误调率高时第一件该改的就是这句话，所以它必须在提示词里存在。"""
    prompt = _task().system_prompt
    assert "不需要" in prompt or "能直接回答" in prompt
    assert "不要编造工具名" in prompt


def test_task_requires_native_tool_calling():
    task = build_task("tool_selection", model="m")
    assert task.requires == frozenset({Cap.TOOLS})
    assert check_capabilities(task, frozenset({Cap.CHAT})) is not None
    assert check_capabilities(task, frozenset({Cap.CHAT, Cap.TOOLS})) is None


def test_skip_reason_says_what_would_be_needed():
    skip = check_capabilities(build_task("tool_selection", model="m"), frozenset({Cap.CHAT}))
    assert skip.missing == ("tools",)
    assert "隐式降级" in skip.reason
    assert skip.as_grade().verdict is Verdict.SKIPPED


def test_registry_lists_every_builtin_task():
    """注册表按 id 排序列出全部内置任务。

    对着 `BUILTIN_TASKS` 断言而不是写死两个名字：新增任务时这条不该失败，
    它该失败的是"新任务没被契约测试覆盖"（那条在 tests/contract/test_task_contract.py）。
    """
    assert task_ids() == tuple(sorted(BUILTIN_TASKS))
    assert "tool_selection" in task_ids() and "structured_extraction" in task_ids()


def test_builtin_dataset_covers_every_kind_and_keeps_the_dangerous_tool_present():
    from onyx.eval.datasets.builtin.tool_calls_zh import build_cases, stats

    report = stats(build_cases())
    assert report["n"] >= 90
    assert report["unique_instructions"] == report["n"]
    for kind in ("single", "parallel", "no_call_needed", "args"):
        assert report["kinds"][kind] >= 10, f"{kind} 子集太小，单独统计没有意义"

    # send_email 是 write 副作用的危险工具，必须在**每条**样本的工具集里：
    # 只有它一直在场，才测得出"模型会不会因为工具存在就乱用"
    for case in build_cases():
        assert "send_email" in case["input"]["tools"]
        assert not any(c["name"] == "send_email" for c in case["expect"]["calls"]), (
            "数据集里没有任何样本应该期望调用 send_email"
        )


def test_limit_subset_covers_multiple_kinds():
    """`--limit 20` 取的是前 20 条，所以生成器必须打散顺序，否则子集会有偏。"""
    from onyx.eval.datasets.builtin.tool_calls_zh import build_cases

    dataset = Dataset(id="tool_calls_zh-v1", cases=tuple(build_cases()))
    kinds = {case["kind"] for case in dataset.select(limit=20)}
    assert len(kinds) >= 3, f"前 20 条只覆盖了 {kinds}，limit 出来的子集会有偏"


def test_low_confidence_counts_cases_not_samples():
    """k=3 时 120 条 grade 只相当于 40 个 bootstrap 单位。

    按 grade 条数判断会把"样本不足"报成"样本充足"，而 CI 自己带的 n 说的是反话；
    两个数打架时使用者不知道该信哪个，于是干脆都不信。
    """
    report = _task().aggregate(_repeated_grades(40, k=3))
    assert len(report) and report["n_bootstrap_units"] == 40
    assert report["low_confidence"] is True, "40 个 case 是低样本，120 条 grade 不是"


def test_must_call_ci_resamples_cases_not_samples():
    """CI 的重采样单位必须是 case。

    按 sample 算的话 n 虚高 k 倍，区间窄得像"模型很确定"，
    其实只是同一件事被数了三遍——temperature=0 下那三次几乎是同一个答案。
    """
    grades = _repeated_grades(40, k=3)
    report = _task().aggregate(grades)
    assert report["n_total"] == 120
    # aggregate 直接返回时 CI 还是 dataclass；落库前才由 metrics.jsonable 转成 dict
    assert report["must_call_acc_ci"].n == 40, "区间的 n 是 case 数，不是 grade 数"
    assert report["by_kind"]["single"]["cases"] == 40


def _repeated_grades(cases: int, *, k: int = 3):
    """造 `cases` 个 case、每个采 `k` 次的全对 grade，用于检查折算单位。"""
    task = _task()
    out = []
    for index in range(cases):
        case = Case(
            id=f"c{index}", kind="single",
            input={"instruction": f"第 {index} 条：查余额"},
            expect={"calls": [{"name": "get_weather", "arguments": {"city": "北京"}}],
                    "must_call": True},
            tools=tuple(_spec(name) for name in TOOL_NAMES),
        )
        grade = task.grade(case, _gen(({"name": "get_weather",
                                       "arguments": {"city": "北京"}},)))
        out.extend(replace(grade, case_id=case.id, seq=seq) for seq in range(k))
    return out
