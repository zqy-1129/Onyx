"""可比性契约（S36）。

`perf compare` 的全部意义就一句话：**条件不同的两条基线不许相减**。
这句话现在的实现是"字段哈希相同"，而哈希相同的字段清单是可以被悄悄改小的——
把某个字段从 `FINGERPRINT_FIELDS` 里删掉，所有对比立刻"更可比"了，
而表现是"这次改版好像变快了"。所以这里断言的是清单与条件形状之间的**结构关系**，
不是某个具体的哈希值。

三条：
1. 条件里出现的字段，除了明确排除的两个，都必须在指纹清单里（新增字段会被这条顶回来）。
2. 清单里任何一个字段变，哈希必须变（少一个就意味着有一类改动逃过了检查）。
3. `app_version` / `git_rev` 变，哈希**不能**变——换代码正是基线要对比的对象。
"""

from __future__ import annotations

import pytest

from onyx.perf.bench import (
    FINGERPRINT_FIELDS,
    IDENTITY_FIELDS,
    collect,
    conditions_of,
    diff_conditions,
    fingerprint,
    fingerprint_ok,
)
from onyx.perf.spec import BenchPlan

EXCLUDED_BY_DESIGN = {"app_version", "git_rev"}


def _conditions(**plan_over) -> dict:
    base = dict(model="qwen3.5:9b", prompt_chars=(600,), target_tokens=(64,),
                concurrency=(1,), repeat=1)
    plan = BenchPlan(**{**base, **plan_over})
    return conditions_of(plan, gateway=object(), device="rtx4060ti",
                         engine_info={"provider_id": "ollama-local",
                                      "version": "0.35.1",
                                      "quantization": "Q4_K_M",
                                      "app_version": "0.8.0", "git_rev": "abc1234"},
                         cells=[])


def test_every_condition_field_is_either_in_the_fingerprint_or_explicitly_excluded():
    """新增一个条件字段而不决定它进不进指纹 ⇒ 这条红。

    这是本文件存在的理由：清单是被人"顺手少写一个字段"改小的，不是被人删掉的。
    """
    produced = set(_conditions())
    assert produced - set(FINGERPRINT_FIELDS) == EXCLUDED_BY_DESIGN, (
        f"条件字段与指纹清单对不上：多出来的是 {produced - set(FINGERPRINT_FIELDS)}，"
        f"排除名单只有 {sorted(EXCLUDED_BY_DESIGN)}")


@pytest.mark.parametrize("field", FINGERPRINT_FIELDS)
def test_each_fingerprint_field_actually_changes_the_hash(field):
    base = _conditions()
    changed = dict(base)
    value = base[field]
    if isinstance(value, bool):
        changed[field] = not value
    elif isinstance(value, int):
        changed[field] = value + 1
    elif isinstance(value, str):
        changed[field] = (value or "") + "|x"
    else:                                    # None：给了值就是变了
        changed[field] = 4096
    assert changed[field] != value, f"{field} 没法被改动，这条测试什么都没测"
    assert fingerprint(changed) != fingerprint(base), f"{field} 逃出了指纹"


def test_version_and_commit_do_not_break_comparability_on_purpose():
    base = _conditions()
    for field in EXCLUDED_BY_DESIGN:
        assert base[field], f"{field} 为空的话这条断言就是空的（看着过其实没测）"
        changed = {**base, field: "different"}
        assert fingerprint(changed) == fingerprint(base)
        assert diff_conditions(base, changed) == [], "代码版本不该出现在条件差异里"


def test_diff_names_exactly_the_fields_that_changed():
    base = _conditions()
    changed = {**base, "model": "qwen3:8b", "num_ctx": 32768}
    diffs = diff_conditions(base, changed)
    assert {item["field"] for item in diffs} == {"model", "num_ctx"}
    row = next(item for item in diffs if item["field"] == "model")
    assert row["a"] == "qwen3.5:9b" and row["b"] == "qwen3:8b", "要说清哪边是什么"


def test_identity_fields_gate_comparability():
    """认不出 provider / 引擎版本 / 模型 ⇒ comparable=False：这时哈希可能相同而实验不同。"""
    for field in IDENTITY_FIELDS:
        blank = {**_conditions(), field: ""}
        assert fingerprint_ok(blank) is False, f"{field} 未知还宣称可比"
    assert fingerprint_ok(_conditions()) is True
    # None 是"已知为不限"，不是"不知道"：混同的话每条兼容通道都会被误判
    assert fingerprint_ok({**_conditions(), "num_ctx": None, "seed": None}) is True


def test_grid_change_makes_the_two_runs_incomparable_end_to_end():
    """整条链的检查：改一个网格维度 ⇒ collect() 出来的 env_hash 不同，且差异只报 grid。"""
    from types import SimpleNamespace

    from onyx.core.clock import FakeClock
    from onyx.core.types import Generation, Status, TokenSample, TokenSource
    from onyx.llm.measurement.reconciler import latency_summary

    class Gateway:
        def generate(self, req, *, purpose=None):
            gen = Generation(text="输出", model=req.model, status=Status.OK, wall_ms=300.0,
                             usage=(TokenSample(source=TokenSource.ENGINE, in_tokens=600,
                                                 out_tokens=64),))
            return SimpleNamespace(trace_id="t", generation=gen, latency=latency_summary(gen))

    info = {"provider_id": "ollama-local", "version": "0.35.1", "quantization": ""}
    a = collect(Gateway(), BenchPlan(model="m", prompt_chars=(600,), target_tokens=(64,),
                                     repeat=1), clock=FakeClock(), engine_info=info)
    b = collect(Gateway(), BenchPlan(model="m", prompt_chars=(1200,), target_tokens=(64,),
                                     repeat=1), clock=FakeClock(), engine_info=info)
    assert a.env_hash != b.env_hash
    assert [item["field"] for item in diff_conditions(a.conditions, b.conditions)] == ["grid"]
