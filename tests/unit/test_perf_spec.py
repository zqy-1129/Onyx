"""网格与参数校验（S36）。

这里测的全是"要测什么"，不发一个请求。之所以单独成文件：网格一错，
基线不会报错、只会给出一个形状完全正常的错数字（比如把并发当成重复次数）。
"""

from __future__ import annotations

import pytest

from onyx.perf.spec import (
    MAX_TARGET_TOKENS,
    BenchPlan,
    PerfSpecError,
    parse_int_list,
)


def _plan(**kw) -> BenchPlan:
    return BenchPlan(model="m", **kw)


def test_default_grid_is_small_on_purpose():
    """默认网格刻意小：一次跑 20 分钟的命令一个月只跑一次，而那不构成"改版有没有变慢"。"""
    plan = _plan()
    assert plan.n_requests <= 24, f"默认网格超过 {plan.n_requests} 发，没人会常跑"


def test_grid_expands_cold_first_then_sorted_dimensions():
    plan = _plan(prompt_chars=(2400, 600), target_tokens=(256, 64), concurrency=(2, 1), cold=True)
    cells = plan.cells()
    assert cells[0].phase == "cold" and cells[0].concurrency == 1
    warm = [c for c in cells if c.phase == "warm"]
    assert [(c.prompt_chars, c.target_tokens, c.concurrency) for c in warm] == [
        (600, 64, 1), (600, 64, 2), (600, 256, 1), (600, 256, 2),
        (2400, 64, 1), (2400, 64, 2), (2400, 256, 1), (2400, 256, 2),
    ], "展开顺序必须可复现：预算用完时「欠哪几格」取决于跑到过哪几格"


def test_duplicate_options_collapse():
    """`1,1,2` 里的第二个 1 不是「再测一遍」——那是 --repeat 的活。留着会让同一格出现两次。"""
    plan = _plan(concurrency=(2, 1, 1))
    keys = [c.key for c in plan.cells()]
    assert len(keys) == len(set(keys)), f"格子不能重复：{keys}"
    # 默认两档长度 × 两档生成长度 = 4 格并发 1、4 格并发 2（`1,1` 没有变成 8 格并发 1）
    assert sum(1 for k in keys if "/x1/" in k) == 4


def test_n_requests_is_concurrency_times_repeat():
    plan = _plan(prompt_chars=(600,), target_tokens=(64,), concurrency=(3,), repeat=4)
    assert plan.n_requests == 12


def test_grid_signature_changes_with_each_dimension():
    base = _plan().grid_signature()
    assert _plan().grid_signature() == base
    assert _plan(repeat=3).grid_signature() != base
    assert _plan(cold=True).grid_signature() != base
    assert _plan(prompt_chars=(1200,)).grid_signature() != base


def test_warn_before_run_says_the_request_count():
    """跑之前要报"多少发、大概多久"：意外耗时是"这条命令没人再跑"的头号原因。"""
    line = _plan(prompt_chars=(600,), target_tokens=(64,), concurrency=(1,), repeat=2) \
        .warn_before_run()
    assert "2 发" in line and "预算" in line


@pytest.mark.parametrize("kw, fragment", [
    ({"model": "  "}, "模型名"),
    ({"prompt_chars": ()}, "不能为空"),
    ({"prompt_chars": (0,)}, "≥ 1"),
    ({"target_tokens": (0,)}, "≥ 1"),
    ({"concurrency": (0,)}, "≥ 1"),
    ({"prompt_chars": (8,)}, "个汉字"),
    ({"target_tokens": (MAX_TARGET_TOKENS + 1,)}, "最大"),
    ({"repeat": 0}, "--repeat"),
    ({"budget_s": 0}, "budget_s"),
    ({"keep_alive": ""}, "keep_alive"),
    ({"num_ctx": 0}, "--num-ctx"),
])
def test_bad_grids_are_rejected_with_a_fix(kw, fragment):
    with pytest.raises(PerfSpecError) as exc:
        BenchPlan(**{"model": "m", **kw})
    assert fragment in str(exc.value), f"报错要能被拿去修：{exc.value}"


@pytest.mark.parametrize("raw, option, fragment", [
    ("", "--concurrency", "不能为空"),
    ("a,2", "--concurrency", "不是整数"),
    ("0", "--concurrency", "≥ 1"),
    ("1,-2", "--concurrency", "≥ 1"),
    ("99", "--target-tokens", "上限"),
])
def test_parse_int_list_errors_name_the_flag(raw, option, fragment):
    """报错里带 flag 名：否则人会把整条命令重打一遍，而不是去看那个参数。"""
    with pytest.raises(PerfSpecError, match=option):
        parse_int_list(raw, option=option, hi=64)


def test_parse_int_list_accepts_spaces_and_order():
    assert parse_int_list(" 4, 2 ,1 ", option="--x") == (4, 2, 1)
