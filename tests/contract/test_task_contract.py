"""任务契约：内置与插件任务都必须守住的三条同源（DESIGN §9.1 / §9.4，ROADMAP G5）。

这三条各对应一种"不会报错、只会让数字说谎"的漂移：

1. **声明的指标 == 产出的指标**。`metric_names` 是看板建列的依据；声明了却产不出，
   那一列永远是「—」，而「—」与"这项 0 分"在界面上长得一样（UI_DESIGN R2）。
   反向也要成立：产出了却没声明，就是"跑完才知道有哪些指标"。
   断言跑在**空输入与有样本两种聚合**上——只在有样本时才出现的键同样是漂移。
2. **区间必须跟着它那个数**。每个 `X_ci` 都要有同名的 `X`，且点估计相等、
   分母不超过参与聚合的样本数。区间描述的是另一个分母时，它比不报区间更坏。
3. **主分数必须是这个任务真的产出的指标，且不许是稳定性指标**。
   `pass_hat_k` 说的是"稳不稳"，让它占主分数的位置等于把"对不对"整个藏起来。

这里是**跨任务**的断言，所以新增任务不必改本文件也会被覆盖到；
某个任务的形态表（各种输出形状各自判成什么）在 `tests/unit/test_*_grade.py`。
"""

from __future__ import annotations

import json
from itertools import islice
from typing import Any

import pytest

from onyx.core.types import Cap, Generation, GenerationRequest, Status, TracePurpose
from onyx.eval.metrics import LOW_CONFIDENCE_N, jsonable
from onyx.eval.task import Verdict, check_capabilities, headline_of
from onyx.eval.tasks import BUILTIN_TASKS, build_task, specs

#: 只描述稳定性的键：它们可以存在，但不许占用主分数的位置
STABILITY_KEYS = frozenset({"pass_hat_k", "pass_at_k", "stability_gap"})

TASK_IDS = sorted(specs())


def _task(task_id: str) -> Any:
    return build_task(task_id, model="mock/contract")


def _cases(task: Any, *, limit: int = 4) -> list[Any]:
    return list(islice(task.load(), limit))


def _shapes(task: Any) -> list[Any]:
    """几种通用输出形态：空正文 / 空对象 / 一个整数 / 引擎失败。

    刻意不给"正确答案"：契约测的是指标集合与不变式，不是分数高低。
    """
    grades: list[Any] = []
    for case in _cases(task):
        for text, status, error in (
            ("", Status.OK, ""),
            ("{}", Status.OK, ""),
            ("42", Status.OK, ""),
            ("", Status.ERROR, "engine down"),
        ):
            grades.append(task.grade(
                case,
                Generation(text=text, model="mock/contract", status=status, error=error),
            ))
    return grades


# ── 1. 声明与产出同源 ──────────────────────────────────────────────
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_declared_metrics_are_exactly_produced(task_id):
    """空聚合与有样本两种聚合，键集合都必须等于声明。"""
    task = _task(task_id)
    declared = set(task.metric_names)
    assert len(declared) == len(task.metric_names), f"{task_id} 的 metric_names 有重复项"

    for label, aggregate in (("空聚合", task.aggregate([])), ("有样本", task.aggregate(_shapes(task)))):
        assert set(aggregate) == declared, (
            f"{task_id} 的{label}键集合与声明不一致：多了 {sorted(set(aggregate) - declared)}，"
            f"少了 {sorted(declared - set(aggregate))}"
            "（只在某类样本存在时才产出的指标，看板没法提前建列）"
        )


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_aggregate_is_storable(task_id):
    """聚合要落库并出现在 `--json` 里：非 JSON 类型会让 CI 变成一个字符串。"""
    task = _task(task_id)
    aggregate = task.aggregate(_shapes(task))
    assert all(isinstance(key, str) for key in aggregate)
    json.dumps(jsonable(aggregate), ensure_ascii=False)


# ── 2. 区间跟着分数 ────────────────────────────────────────────────
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_every_interval_accompanies_its_own_score(task_id):
    task = _task(task_id)
    grades = _shapes(task)
    aggregate = task.aggregate(grades)
    for key, value in aggregate.items():
        if not (key.endswith("_ci") and isinstance(value, dict)):
            continue
        base = key[: -len("_ci")]
        assert base in aggregate, f"{task_id} 报了 {key} 却没有 {base}：区间没有主语"
        assert value["n"] <= len(grades), (
            f"{task_id} 的 {key} 分母 {value['n']} 超过参与聚合的 {len(grades)} 条，"
            "区间在描述一个不存在的样本集"
        )
        assert value["low_confidence"] == (value["n"] < LOW_CONFIDENCE_N)
        point, score = value["point"], aggregate[base]
        if point is None or score is None:
            # 一边算不出来时另一边也不许有数，否则"有分数没区间"是假的
            assert point is None and score is None, (
                f"{task_id}：{base}={score!r} 与 {key}.point={point!r} 一空一有"
            )
            continue
        assert abs(point - float(score)) < 1e-9, (
            f"{task_id} 的 {base}={score} 与 {key}.point={point} 不是同一个统计量"
        )


# ── 3. 主分数的位置 ────────────────────────────────────────────────
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_headline_is_a_produced_metric_and_not_stability(task_id):
    task = _task(task_id)
    aggregate = task.aggregate(_shapes(task))
    picked = headline_of(aggregate)
    assert picked is not None, f"{task_id} 没有主分数候选，列表页与矩阵都会显示「—」"
    key, _value = picked
    assert key in aggregate and key in set(task.metric_names)
    assert key not in STABILITY_KEYS, (
        f"{task_id} 的主分数落到了稳定性指标 {key} 上"
    )


# ── 4. build 只产出请求，不发调用 ──────────────────────────────────
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_build_returns_a_request_with_thinking_off(task_id):
    """`build` 返回 `GenerationRequest` 而不是分数：评测不许建立第二条调用路径。

    `thinking=False` 是 P12 的实测结论——thinking 计入 eval_count 且会吃光小预算，
    正文于是变空并被误判成"格式非法"。这个开关必须由请求自己关掉，不能靠默认值。
    """
    task = _task(task_id)
    for case in _cases(task):
        request = task.build(case)
        assert isinstance(request, GenerationRequest)
        assert request.model == "mock/contract"
        assert request.context.purpose is TracePurpose.EVAL
        assert request.thinking is False, f"{task_id} 没有显式关掉 thinking"


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_capability_gap_skips_with_a_reason(task_id):
    """能力不满足必须 skip 并写清原因，禁止隐式降级（DESIGN §9.1）。"""
    task = _task(task_id)
    assert all(isinstance(cap, Cap) for cap in task.requires), (
        f"{task_id} 的 requires 里有非 Cap 的值"
    )
    skip = check_capabilities(task, [])
    if not task.requires:
        assert skip is None
        return
    assert skip is not None and skip.reason
    assert set(skip.missing) == {str(cap) for cap in task.requires}
    grade = skip.as_grade()
    assert grade.verdict is Verdict.SKIPPED and grade.passed is None


# ── 5. 数据形状与环境故障 ──────────────────────────────────────────
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_default_dataset_loads_and_survives_truncation(task_id):
    """DoD 要求 `--limit` 截断时子集仍非空：截断规则必须是"前 N 条且各类混合"。"""
    task = _task(task_id)
    cases = list(task.load())
    assert cases, f"{task_id} 的默认数据集一条样本都没有"
    ids = [case.id for case in cases]
    assert len(set(ids)) == len(ids), f"{task_id} 的 case id 重复，grade 会互相覆盖"
    assert all(case.input for case in cases), f"{task_id} 有空 input 的样本"
    assert list(task.load(limit=1)), f"{task_id} 截断到 1 条就空了"
    assert all(case.dataset_id == task.dataset.id for case in cases)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_an_engine_outage_leaves_no_score_rather_than_zero(task_id):
    """引擎全挂时分数必须是「算不出来」，不是 0 分。

    「没考到」与「考了且全错」在界面与报表上必须可区分，否则一次网络故障
    会伪装成一次模型退化，并进入 compare 的历史。
    """
    task = _task(task_id)
    case = _cases(task, limit=1)[0]
    failed = task.grade(
        case, Generation(text="", model="mock/contract", status=Status.ERROR, error="engine down")
    )
    assert failed.verdict is Verdict.ERROR and not failed.attributable

    aggregate = task.aggregate([failed])
    key, value = headline_of(aggregate)
    assert value is None, f"{task_id} 在只有环境故障时把主分数 {key} 报成了 {value!r}"
    if "n_attributable" in aggregate:
        assert aggregate["n_attributable"] == 0
    if "verdicts" in aggregate:
        assert aggregate["verdicts"].get(Verdict.ERROR.value) == 1


def test_the_contract_covers_every_registered_task():
    """参数化清单不许悄悄变空：`specs()` 坏了的话上面每一条都会"绿"。"""
    assert set(BUILTIN_TASKS) <= set(TASK_IDS)
    assert sorted(specs()) == TASK_IDS
