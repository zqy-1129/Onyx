"""语料的自洽（S36）。

基线的 x 轴就是这份语料的长度。它一错，每一格的横坐标都错，而数字仍然自洽——
所以这里不测"能不能生成"，测的是**生成出来的东西是不是我声明的那个东西**。
每条自检都配了"注入缺陷时会响"的证据（见最后两条）。
"""

from __future__ import annotations

import pytest

from onyx.perf.corpus import (
    INSTRUCTION,
    adjacent_repeats,
    cjk_count,
    colliding_prefixes,
    exact_length_failures,
    nondeterministic,
    prompt_for,
)
from onyx.perf.spec import MIN_PROMPT_CHARS

DEFAULT_GRID = [600, 2400, 9600]


@pytest.mark.parametrize("chars", [MIN_PROMPT_CHARS, 600, 2400, 9600])
def test_prompt_has_exactly_the_declared_hanzi_count(chars):
    """声明的是**汉字数**，不是自称的 token 数：tokenizer 是引擎的属性，不是我们的。"""
    assert cjk_count(prompt_for(chars)) == chars


def test_no_length_failures_on_the_default_grid():
    assert exact_length_failures(DEFAULT_GRID) == []


def test_prompts_are_deterministic():
    """同一档两次构造一字不差，否则"这两次跑的是同一份考卷"没有依据。"""
    assert nondeterministic(DEFAULT_GRID) == []
    assert prompt_for(600) == prompt_for(600)


def test_different_lengths_do_not_share_a_head():
    """档位之间不共享开头，否则跑完短档再跑长档会命中 KV 缓存，prefill 吞吐被凭空抬高。"""
    assert colliding_prefixes(DEFAULT_GRID) == []


def test_instruction_appears_once_at_the_end():
    text = prompt_for(600)
    assert text.endswith(INSTRUCTION) and text.count(INSTRUCTION) == 1


def test_no_two_adjacent_lines_repeat_the_same_body():
    """连着两句一样会让模型进入"复读"形态，那测的就不是生成长度而是复读速度。"""
    for chars in DEFAULT_GRID:
        assert adjacent_repeats(chars) == 0, f"{chars} 档出现了相邻重复"


def test_body_lines_are_numbered():
    text = prompt_for(600)
    assert text.startswith("档 600：第 1 条，") and "第 2 条，" in text


def test_too_short_a_prompt_cannot_hold_the_instruction():
    with pytest.raises(ValueError, match="装得下指令"):
        prompt_for(len(INSTRUCTION))


# ── 注入缺陷自检：三条检查都要能响 ────────────────────────────────
# 「守一件事」的断言必须回答"它坏掉时会红吗"。这里不改生产代码，而是把语料生成器
# 换成已知会犯错的版本（模块内的检查函数都是按名字取 `prompt_for`，所以会被一起换掉），
# 验证**检查本身**还活着。
def test_length_check_shouts_when_the_corpus_is_off_by_a_few(monkeypatch):
    import onyx.perf.corpus as corpus

    real = corpus.prompt_for

    def off_by_two(chars: int) -> str:
        return real(chars) + "额外"

    monkeypatch.setattr(corpus, "prompt_for", off_by_two)
    assert exact_length_failures([600]) == ["600：实际 602 个汉字"]


def test_prefix_check_shouts_when_the_labels_collide(monkeypatch):
    """两档开头相同 ⇒ 跑完短档再跑长档会命中 KV 缓存。检查必须点名那一对。"""
    import onyx.perf.corpus as corpus

    def same_head(chars: int) -> str:
        return "仓库东侧的货架按编号排列，每一层都有独立的标签条" + "余" * chars

    monkeypatch.setattr(corpus, "prompt_for", same_head)
    assert colliding_prefixes([600, 1200]) == [(600, 1200)]


def test_determinism_check_shouts_when_the_source_varies(monkeypatch):
    import onyx.perf.corpus as corpus

    real = corpus.prompt_for
    counter = {"n": 0}

    def drifting(chars: int) -> str:
        counter["n"] += 1
        return real(chars) + str(counter["n"])

    monkeypatch.setattr(corpus, "prompt_for", drifting)
    assert nondeterministic([600]) == [600]
