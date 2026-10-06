"""口径收敛（S36 第一步）：分位数与汉字下限都只有一份实现。

这一步不改任何行为，改的是"有几处实现"。理由在项目里反复验证过：
两处算同一个量，早晚会分叉，而分叉的表现永远是"两个页面给出不同的数，各自都能自证没错"。
所以这里的断言全部围绕**同一性**：换了实现位置，值必须一模一样。
"""

from __future__ import annotations

import pytest

from onyx.core.types import Generation, Status
from onyx.eval.metrics import _percentile as metrics_percentile
from onyx.llm.measurement.heuristic import MIN_TOKENS_PER_HANZI, min_prompt_tokens, split_cjk
from onyx.llm.measurement.stats import median, percentile, spread, spread_or_none


def test_stats_percentile_matches_the_eval_definition():
    """`eval/metrics._percentile` 现在只是转发。转发写错的话 CI 里的 CI 区间会整体漂移。"""
    values = [1.0, 2.0, 3.0, 4.0, 10.0]
    for q in (0.0, 0.025, 0.5, 0.95, 0.975, 1.0):
        assert metrics_percentile(values, q) == pytest.approx(percentile(values, q))


def test_single_value_is_itself_not_a_zero():
    assert percentile([7.5], 0.95) == 7.5
    assert median([4.0, 1.0, 9.0]) == 4.0


def test_empty_sequence_raises_instead_of_returning_zero():
    """空集抛异常而不是回 0：0 会被读成"延迟是 0ms"，而事实是"一条都没测到"。"""
    with pytest.raises(ValueError, match="空序列"):
        percentile([], 0.5)
    with pytest.raises(ValueError, match="没测到"):
        spread([])
    assert spread_or_none([]) is None, "要表达「没测到」请走 spread_or_none"


def test_spread_reports_its_own_n():
    got = spread([10.0, 20.0, 30.0, 40.0, 50.0])
    assert got["n"] == 5
    assert got["median"] == 30.0
    assert (got["min"], got["max"]) == (10.0, 50.0)
    # 线性插值下的 p95：落在 40 与 50 之间的 0.95 位
    assert got["p95"] == pytest.approx(48.0)


def test_min_prompt_tokens_is_a_floor_not_an_estimate():
    """下限只由汉字推：用途是识别"引擎把正文裁了"，不是拿来当 token 数。"""
    text = "汉" * 100 + "abc"
    assert min_prompt_tokens(text) == int(100 * MIN_TOKENS_PER_HANZI)
    assert min_prompt_tokens("") == 0, "推不出来 ⇒ 0 ⇒ 调用方不该做任何截断判断"
    assert min_prompt_tokens(text, tokens_per_hanzi=1.0) == 100


def test_long_context_uses_the_same_floor():
    """两处判据必须是同一个数：评测说没裁而基线说裁了，没人能解释哪边对。"""
    from onyx.eval.tasks.long_context import MIN_TOKENS_PER_HANZI as TASK_TOKENS_PER_HANZI
    from onyx.eval.tasks.long_context import _min_prompt_tokens

    text = "仓库东侧的货架按编号排列" * 30
    assert TASK_TOKENS_PER_HANZI == MIN_TOKENS_PER_HANZI
    assert _min_prompt_tokens(text) == min_prompt_tokens(text)
    cjk, other = split_cjk(text)
    assert cjk == len(text) and other == 0, "纯汉字正文不该有「其他字符」"


def test_generation_helpers_still_import_cleanly():
    """转发实现容易漏依赖（`metrics.py` 里 `math` 变成未使用就是信号）。

    这里不测行为，只测"模块还能不能 import"——那是一处改坏了会立刻显形的地方。
    """
    assert Generation(text="", status=Status.OK).wall_ms is None
