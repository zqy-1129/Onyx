"""配对对比的验收。

这一步的全部价值在于**它不是"比较两个平均值"**，所以测试必须能把"均值陷阱"钉住：
构造一组均值几乎不动、但 8 条变好 8 条变坏的样本，如果实现只看均值，
这里会得到"基本持平"，而真相是"在两类任务上方向相反"。

其余测试守的是可比性：数据集/版本/k/温度/seed 任一不同，差值就不该被当成能力差异，
所以它们必须出现在 warnings 里，而不是被静默比较。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from onyx.eval.compare import CompareError, compare_runs, per_case
from onyx.store.db import Database
from onyx.store.records import CaseRecord, DatasetRecord, GradeRecord, RunRecord
from onyx.store.repos import EvalRepo

BASE, TARGET = "run-base", "run-target"


def _grade(run_id: str, case_id: str, score: float, *, seq: int = 0, passed: Any = "auto",
           verdict: str = "correct", trace: str | None = None, kind: str = "single",
           instruction: str = "") -> GradeRecord:
    """`passed="auto"` 按分数推；显式传 None 表示"这一条没判定"（skip / 引擎失败）。

    默认值必须是哨兵而不是 None，否则测试里根本表达不出"未判定"这一档。
    """
    resolved = (score >= 0.5) if passed == "auto" else passed
    return GradeRecord(
        id=f"{run_id}-{case_id}-{seq}", eval_run_id=run_id, case_id=case_id, seq=seq,
        score=score, verdict=verdict, passed=resolved,
        trace_id=trace or f"tr-{run_id}-{case_id}-{seq}", graded_at="2026-10-03T00:00:00+00:00",
        metrics={"kind": kind, "instruction": instruction},
    )


def _db(tmp_path) -> Database:
    db = Database(tmp_path / "t.sqlite")
    repo = EvalRepo(db)
    for dataset_id, revision in (("ds-v1", "r1"), ("ds-v2", "r2")):
        repo.upsert_dataset(DatasetRecord(id=dataset_id, imported_at="2026-10-03",
                                          upstream="test", revision=revision))
    return db


def _make_run(db: Database, run_id: str, *, model: str, dataset_id: str = "ds-v1",
              task_id: str = "intent_classification", k: int = 1,
              params: dict | None = None, seed: int | None = 7,
              status: str = "done") -> None:
    """建 run 行。已存在就跳过：同一个测试里会多次配对同一对 run。"""
    if db.query_one("SELECT id FROM eval_run WHERE id=?", (run_id,)) is not None:
        return
    from onyx.store.records import TaskRecord

    repo = EvalRepo(db)
    if db.query_one("SELECT id FROM eval_task WHERE id=?", (task_id,)) is None:
        repo.upsert_task(TaskRecord(id=task_id, name=task_id, dataset_id="ds-v1",
                                    metrics=["macro_f1"]))
    repo.insert_run(RunRecord(
        id=run_id, task_id=task_id, model_id=model, started_at="2026-10-03T00:00:00+00:00",
        status=status, config={"k": k, "split": "default"},
        params_snapshot=params or {"temperature": 0.0, "max_tokens": 32}, seed=seed,
        app_version="0.1.0", dataset_id=dataset_id,
        dataset_revision="r1" if dataset_id == "ds-v1" else "r2",
    ))


@pytest.fixture
def env(tmp_path):
    db = _db(tmp_path)
    yield db, EvalRepo(db)
    db.close()


def _pair(env, base_grades, target_grades, *, model_a="A", model_b="B", **kw):
    db, repo = env
    _make_run(db, BASE, model=model_a)
    _make_run(db, TARGET, model=model_b)
    case_ids = sorted({g.case_id for g in list(base_grades) + list(target_grades)})
    repo.upsert_cases([CaseRecord(id=cid, dataset_id="ds-v1", ord=index,
                                  input={"instruction": cid}, expect={})
                       for index, cid in enumerate(case_ids)])
    repo.upsert_grades(list(base_grades))
    repo.upsert_grades(list(target_grades))
    return compare_runs(repo, BASE, TARGET, **kw)


# ── 均值陷阱 ──────────────────────────────────────────────────────
def test_mean_delta_can_be_zero_while_eight_cases_get_worse(env):
    """实现只看均值就会在这里露馅：均值差 0，但有 8 条实打实变差了。

    "基本持平"是一个错误的结论——正确结论是"在 8 类上变好、在另外 8 类上变差"，
    该看的是劣化清单里那些 case 有什么共同点。
    """
    base = [_grade(BASE, f"c{i}", 1.0) for i in range(8)] + \
           [_grade(BASE, f"w{i}", 0.0) for i in range(8)]
    target = [_grade(TARGET, f"c{i}", 0.0) for i in range(8)] + \
             [_grade(TARGET, f"w{i}", 1.0) for i in range(8)]

    result = _pair(env, base, target)

    assert result.mean_delta == pytest.approx(0.0), "均值不动"
    assert (result.improved, result.regressed, result.unchanged) == (8, 8, 0)
    assert len(result.regressions()) == 8
    assert all(item.delta < 0 for item in result.regressions())


def test_delta_direction_is_target_minus_base(env):
    """方向搞反是这类工具最致命的 bug：它会直接把回归报成提升。"""
    result = _pair(env, [_grade(BASE, "c1", 0.2)], [_grade(TARGET, "c1", 1.0)])
    assert result.paired[0].delta == pytest.approx(0.8)
    assert result.paired[0].direction == "improved"
    assert result.mean_delta == pytest.approx(0.8)

    worse = _pair(env, [_grade(BASE, "c1", 1.0)], [_grade(TARGET, "c1", 0.25)])
    assert worse.paired[0].delta == pytest.approx(-0.75)
    assert worse.regressed == 1 and worse.improved == 0


def test_regressions_are_ordered_worst_first(env):
    result = _pair(env, [_grade(BASE, f"c{i}", 1.0) for i in range(4)],
                   [_grade(TARGET, "c0", 0.75), _grade(TARGET, "c1", 0.0),
                    _grade(TARGET, "c2", 0.5), _grade(TARGET, "c3", 1.0)])
    order = [item.case_id for item in result.regressions()]
    assert order == ["c1", "c2", "c0"], "越差的越该排在前面，清单要能直接当工单用"
    assert result.unchanged == 1, "c3 两边都是 1.0"


# ── 配对与覆盖率 ──────────────────────────────────────────────────
def test_only_shared_cases_are_compared_and_the_rest_are_counted(env):
    """两个 run 的题不一样多时，必须说清结论只覆盖交集。

    静默按交集算会产出"n=40 的结论"，而读者以为考的是同一份 60 题的卷子。
    """
    base = [_grade(BASE, f"c{i}", 1.0) for i in range(6)]
    target = [_grade(TARGET, f"c{i}", 1.0) for i in range(4)] + \
             [_grade(TARGET, f"x{i}", 0.0) for i in range(2)]

    result = _pair(env, base, target)

    assert result.n_paired == 4
    assert result.only_base == ("c4", "c5")
    assert result.only_target == ("x0", "x1")
    assert result.coverage == pytest.approx(4 / 8)
    assert any("只有 4/8 个 case 被两边同时考到" in w for w in result.warnings)


def test_no_overlap_is_refused_not_reported_as_a_tie(env):
    with pytest.raises(CompareError, match="没有任何共同的 case"):
        _pair(env, [_grade(BASE, "c1", 1.0)], [_grade(TARGET, "z9", 0.0)])


def test_missing_run_is_reported_by_id(env):
    db, repo = env
    _make_run(db, BASE, model="A")
    with pytest.raises(CompareError, match="找不到 run 'nope'"):
        compare_runs(repo, BASE, "nope")


def test_different_task_is_refused(env):
    db, repo = env
    _make_run(db, BASE, model="A", task_id="intent_classification")
    _make_run(db, TARGET, model="B", task_id="tool_selection")
    repo.upsert_cases([CaseRecord(id="c1", dataset_id="ds-v1", ord=0,
                                  input={"instruction": "c1"}, expect={})])
    repo.upsert_grade(_grade(BASE, "c1", 1.0))
    repo.upsert_grade(_grade(TARGET, "c1", 0.0))
    with pytest.raises(CompareError, match="任务不同"):
        compare_runs(repo, BASE, TARGET)


# ── k>1 的折算 ────────────────────────────────────────────────────
def test_per_case_averages_scores_but_requires_all_samples_to_pass(env):
    """pass^k 的口径不能被均值偷偷替换。

    3 次里过了 2 次：能力估计算 0.67，但"可靠可用"是否——把 2/3 当成过，
    pass^k 就退化成 pass@k，而这两个指标的意义正好相反。
    """
    rows = [
        _grade(BASE, "c1", 1.0, seq=0, passed=True),
        _grade(BASE, "c1", 1.0, seq=1, passed=True),
        _grade(BASE, "c1", 0.0, seq=2, passed=False, verdict="wrong"),
    ]
    folded = per_case(rows)["c1"]
    assert folded["score"] == pytest.approx(2 / 3)
    assert folded["passed"] is False
    assert folded["n_samples"] == 3
    assert folded["verdict"] == "mixed", "同一 case 三种判定要显式标出来，不能挑一条代表全部"


def test_unjudged_sample_makes_pass_k_unknown_not_false(env):
    """有一边没判定（skip / 引擎失败）时，pass^k 是"不知道"。

    当成 False 会把基础设施故障记成模型能力问题——这正是 M4 里反复防的那类错位。
    """
    folded = per_case([_grade(BASE, "c1", 0.0, passed=True),
                       _grade(BASE, "c2", 0.0, passed=None)])
    assert folded["c1"]["passed"] is True
    assert folded["c2"]["passed"] is None


def test_flips_count_the_binary_dimension_separately_from_scores(env):
    """分数均值会被"部分分"抹平，翻转数不会。

    三条 case：一条从 1.0 掉到 0.0（会→不会），两条从 0.0 升到 0.5（不会→会）。
    均值正好是 0 ⇒ "总体没变"；而真相是"3 条题的答案换了方向"，
    其中掉下去那条是要立刻处理的回归。两个口径必须同时可见。
    """
    result = _pair(env,
                   [_grade(BASE, "down", 1.0), _grade(BASE, "u1", 0.0), _grade(BASE, "u2", 0.0)],
                   [_grade(TARGET, "down", 0.0), _grade(TARGET, "u1", 0.5),
                    _grade(TARGET, "u2", 0.5)])
    assert result.mean_delta == pytest.approx(0.0), "均值不动"
    assert (result.improved, result.regressed) == (2, 1)
    assert result.flips == {"up": 2, "down": 1, "net": 1}
    assert [item.case_id for item in result.regressions()] == ["down"]


def test_eps_suppresses_noise_below_the_threshold(env):
    base = [_grade(BASE, "c1", 0.5), _grade(BASE, "c2", 0.5)]
    target = [_grade(TARGET, "c1", 0.52), _grade(TARGET, "c2", 0.48)]
    loose = _pair(env, base, target, eps=0.05)
    strict = _pair(env, base, target, eps=0.0)
    assert (loose.improved, loose.regressed, loose.unchanged) == (0, 0, 2)
    assert (strict.improved, strict.regressed) == (1, 1), "不设阈值时 0.02 也算变化"


# ── 配对 bootstrap ────────────────────────────────────────────────
def test_identical_runs_give_a_zero_width_ci_and_no_change(env):
    grades = [_grade(BASE, f"c{i}", 1.0 if i % 2 else 0.0) for i in range(20)]
    mirrored = [_grade(TARGET, g.case_id, g.score) for g in grades]
    result = _pair(env, grades, mirrored)
    assert result.improved == 0 and result.regressed == 0
    assert result.delta_ci.low == 0.0 and result.delta_ci.high == 0.0


def test_ci_width_reacts_to_case_level_variance_not_just_n(env):
    """均值相同、离散度不同 ⇒ 区间必须不同宽。

    返回一个与数据无关的常数区间（或者干脆不重采样）会在这里暴露。
    """
    wide_run = _pair(env,
                     [_grade(BASE, f"c{i}", 0.5) for i in range(20)],
                     [_grade(TARGET, f"c{i}", 0.5 + (0.3 if i % 2 else -0.3)) for i in range(20)])
    narrow_run = _pair(env,
                       [_grade(BASE, f"c{i}", 0.5) for i in range(20)],
                       [_grade(TARGET, f"c{i}", 0.5 + (0.02 if i % 2 else -0.02))
                        for i in range(20)])
    assert wide_run.mean_delta == pytest.approx(narrow_run.mean_delta)
    wide = wide_run.delta_ci.high - wide_run.delta_ci.low
    narrow = narrow_run.delta_ci.high - narrow_run.delta_ci.low
    assert wide > narrow * 3, f"离散度差了 15 倍，区间宽度却没跟上：{wide} vs {narrow}"


def test_ci_is_stable_across_calls_because_the_seed_is_fixed(env):
    grades = [_grade(BASE, f"c{i}", 1.0) for i in range(10)]
    target = [_grade(TARGET, f"c{i}", 0.0 if i < 3 else 1.0) for i in range(10)]
    first = _pair(env, grades, target, seed=11)
    second = _pair(env, grades, target, seed=11)
    assert first.delta_ci.as_dict() == second.delta_ci.as_dict(), \
        "区间自己会抖动的话就没法用它做回归判断"
    assert first.delta_ci.n == 10 and first.regressed == 3


# ── 可比性警告 ────────────────────────────────────────────────────
def test_dataset_mismatch_is_warned_loudly(env):
    """换考卷之后的"提升"是最容易被当成模型进步的假信号。"""
    db, repo = env
    _make_run(db, BASE, model="A", dataset_id="ds-v1")
    _make_run(db, TARGET, model="B", dataset_id="ds-v2")
    repo.upsert_cases([CaseRecord(id="c1", dataset_id="ds-v1", ord=0,
                                  input={"instruction": "c1"}, expect={"label": "转账"})])
    repo.upsert_grade(_grade(BASE, "c1", 0.0))
    repo.upsert_grade(_grade(TARGET, "c1", 1.0))
    result = compare_runs(repo, BASE, TARGET)
    assert any("数据集不同" in w for w in result.warnings), result.warnings


def test_temperature_difference_is_warned(env):
    result = _pair(env, [_grade(BASE, "c1", 0.0)], [_grade(TARGET, "c1", 1.0)])
    # 默认构造的两次运行参数一致 ⇒ 不该有温度警告
    assert not any("生成参数不同" in w for w in result.warnings)

    db, repo = env
    _make_run(db, "run-hot", model="A", params={"temperature": 0.7, "max_tokens": 32})
    repo.upsert_grade(_grade("run-hot", "c1", 0.0))
    hot = compare_runs(repo, "run-hot", TARGET)
    assert any("生成参数不同" in w and "temperature" in w for w in hot.warnings), hot.warnings


def test_small_sample_is_flagged_in_warnings_and_flag(env):
    result = _pair(env, [_grade(BASE, "c1", 0.0)], [_grade(TARGET, "c1", 1.0)])
    assert result.low_confidence is True
    assert any("配对样本只有 1 条" in w for w in result.warnings)


def test_k_mismatch_is_warned(env):
    db, repo = env
    _make_run(db, BASE, model="A", k=1)
    _make_run(db, TARGET, model="B", k=3)
    repo.upsert_cases([CaseRecord(id="c1", dataset_id="ds-v1", ord=0,
                                  input={"instruction": "c1"}, expect={"label": "转账"})])
    repo.upsert_grade(_grade(BASE, "c1", 0.0))
    repo.upsert_grade(_grade(TARGET, "c1", 1.0))
    result = compare_runs(repo, BASE, TARGET)
    assert any("采样次数 k 不同" in w for w in result.warnings)


# ── 出口形状 ──────────────────────────────────────────────────────
def test_unjudged_samples_are_excluded_from_the_flip_counts(env):
    """pass^k 不知道 ⇔ 不该计入翻转。

    把"引擎挂了所以没判定"算成"从不会变成会"，会把基础设施故障报成模型进步。
    """
    result = _pair(env,
                   [_grade(BASE, "c1", 0.0, verdict="error", passed=None),
                    _grade(BASE, "c2", 0.0, passed=False)],
                   [_grade(TARGET, "c1", 1.0), _grade(TARGET, "c2", 1.0)])
    assert result.flips == {"up": 1, "down": 0, "net": 1}, "c1 两边都没判定，跳过"
    assert result.paired[0].passed_base is None


def test_as_dict_is_json_safe_with_a_real_ci_dict(env):
    """CI 必须是 dict。`json.dumps(default=str)` 会把它压成 "CI(low=…)"，
    写进库不报错、读出来取不到上下界——区间静默消失。"""
    result = _pair(env, [_grade(BASE, f"c{i}", 0.0) for i in range(4)],
                   [_grade(TARGET, f"c{i}", 1.0 if i % 2 else 0.5) for i in range(4)])
    payload = json.loads(json.dumps(result.as_dict(), ensure_ascii=False))
    assert isinstance(payload["delta_ci"], dict)
    assert payload["delta_ci"]["low"] is not None, "4 个配对样本该有区间了"
    assert payload["delta_ci"]["n"] == 4
    assert payload["cases"][0]["case_id"] == "c0"
    assert payload["base"]["model_id"] == "A" and payload["target"]["model_id"] == "B"


def test_a_single_paired_case_gets_no_interval(env):
    """只有 1 个配对样本时不许给区间——那是假的确定性。

    重采样一个样本永远得到它自己，宽度恒为 0。metrics 层已经把 n<=1 拦成
    low/high=None，对比这条路必须沿用同一个口径，否则 UI 上会出现
    "1 条样本 + 一个看起来极其确定的区间"。
    """
    result = _pair(env, [_grade(BASE, "c1", 0.0)], [_grade(TARGET, "c1", 1.0)])
    assert result.delta_ci.low is None and result.delta_ci.high is None
    assert result.delta_ci.point == pytest.approx(1.0), "点估计还是要给的"
    assert result.low_confidence is True


def test_each_paired_case_carries_both_trace_ids_for_drill_down(env):
    """对比页点一条劣化 case，要能同时打开两个模型的那两条 trace 并排看。"""
    result = _pair(env, [_grade(BASE, "c1", 1.0, trace="trA"),
                         _grade(BASE, "c2", 1.0, trace="trA2")],
                   [_grade(TARGET, "c1", 0.0, trace="trB")])
    item = next(c for c in result.paired if c.case_id == "c1")
    assert (item.trace_base, item.trace_target) == ("trA", "trB")
    assert result.coverage == pytest.approx(1 / 2), "c2 只有 base 考了"
