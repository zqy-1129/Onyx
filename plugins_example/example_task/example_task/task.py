"""数汉字任务：`onyx.tasks` 扩展点的样板实现。

它刻意只做三件事，把外部任务必须守住的三条纪律演示全：

1. **能力位**：`requires = {CHAT}`。不满足时 runner 会整任务 skip 并落一条带原因的
   记录——"没分数"与"跑了但 0 分"必须可区分（DESIGN §9.1）。
2. **`metric_names` 与 `aggregate` 同源**：声明了却产不出的指标，在界面上和
   "这项能力 0 分"长得一模一样（UI_DESIGN R2）。内核的
   `tests/unit/test_tool_selection_grade.py::test_declared_metric_names_are_actually_produced`
   对内置任务断言的就是这条；契约测试对插件跑同一条。
3. **内容与格式正交**：`verdict` 只管答得对不对，`invalid_format` 只管听不听话
   （DESIGN §9.4）。本地小模型这两类错误极常见，混成一个正确率就会把格式问题
   误读成能力问题。
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Any

from onyx.core.types import (
    Cap,
    Generation,
    GenerationRequest,
    GenParams,
    Message,
    Role,
    Status,
    TraceContext,
    TracePurpose,
)
from onyx.eval.datasets.loader import Dataset
from onyx.eval.task import Case, Grade, TaskSpec, Verdict

SYSTEM_PROMPT = "数一数句子里汉字的个数。**只输出一个整数**，不要输出解释、标点或任何其它文字。"

#: 语料。期望值由 `_han()` 按定义算出，不手填——样板自己写错答案就是最坏的示范。
TEXTS: tuple[str, ...] = (
    "今天天气不错",
    "会议室在三楼，请带好工牌",
    "OK，我们 12 点开会再讨论细节",
    "这个模型的上下文长度为 4096 tokens",
    "他说：「好的，我明白了」",
    "hello world 没有汉字",
    "一二三四五六七八九十",
    "Onyx 看板把 token 与延迟都记录下来",
)


def _han(text: str) -> int:
    """按码位区间数汉字（这就是本任务的"真值定义"）。"""
    return sum(1 for ch in text if "一" <= ch <= "鿿")


def dataset() -> Dataset:
    """插件自带数据集：直接构造 `Dataset`，不必先进库。

    `revision` 参与可比性判定：改了语料不改 revision，两次分数的 diff 就没有意义。
    """
    cases = [
        {
            "id": f"charcount-{index:02d}",
            "ord": index,
            "kind": "single",
            "input": {"text": text},
            "expect": {"answer": _han(text)},
            "tags": ("no_hanzi",) if _han(text) == 0 else (),
        }
        for index, text in enumerate(TEXTS, start=1)
    ]
    return Dataset(
        id="example_char_count-v1",
        cases=tuple(cases),
        upstream="self-generated",
        revision="v1",
        license="MIT",
        loader="example_task.dataset",
        notes="扩展点样板：期望值由汉字码位定义算出",
    )


class CharCount:
    id = "example_char_count"
    name = "数汉字（扩展点样板）"
    requires: frozenset[Cap] = frozenset({Cap.CHAT})
    metric_names = ("accuracy", "format_valid_rate", "n_judged")

    def __init__(
        self,
        dataset: Dataset,
        *,
        model: str,
        max_tokens: int = 16,
        temperature: float = 0.0,
        split: str = "default",
        **_extra: Any,
    ) -> None:
        self.dataset = dataset
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.split = split

    # ── 契约实现 ──────────────────────────────────────────────────
    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]:
        for raw in self.dataset.select(split=split or self.split, limit=limit):
            yield Case(
                id=str(raw["id"]), input=dict(raw.get("input") or {}),
                expect=dict(raw.get("expect") or {}), dataset_id=self.dataset.id,
                ord=int(raw.get("ord") or 0), kind=str(raw.get("kind") or "single"),
                tags=tuple(raw.get("tags") or ()),
            )

    def build(self, case: Case) -> GenerationRequest:
        return GenerationRequest(
            model=self.model,
            messages=(
                Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                Message(role=Role.USER, content=str(case.input.get("text") or "")),
            ),
            params=GenParams(max_tokens=self.max_tokens, temperature=self.temperature),
            # thinking 必须显式关：P12 实测 thinking 会吃光小预算，正文变空，
            # 于是"格式非法"的分数掉了一个不该掉的原因
            thinking=False,
            context=TraceContext(purpose=TracePurpose.EVAL),
        )

    def grade(self, case: Case, sample: Generation) -> Grade:
        expected = case.expect.get("answer")
        if not isinstance(expected, int):
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                invalid_format=True, error="样本缺少 expect.answer，无法评分",
            )
        if sample.status is not Status.OK:
            # 引擎失败是环境问题，不该进模型能力分母
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.ERROR, passed=None,
                error=sample.error or f"status={sample.status}",
            )

        text = sample.text or ""
        found = re.findall(r"-?\d+", text)
        if not found:
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True, error=f"输出里没有整数：{text[:80]!r}",
                metrics={"expected": expected, "text": text[:200]},
            )
        if len(found) > 1:
            # 多个候选数字 ⇒ 不可判定。挑一个（哪怕挑第一个）会让分数凭空变高
            return Grade(
                case_id=case.id, score=0.0, verdict=Verdict.INVALID_FORMAT, passed=False,
                invalid_format=True,
                error=f"输出里出现多个数字 {found}，不可判定",
                metrics={"expected": expected, "found": found, "text": text[:200]},
            )
        got = int(found[0])
        clean = text.strip() == found[0]
        correct = got == expected
        return Grade(
            case_id=case.id,
            score=1.0 if correct else 0.0,
            verdict=Verdict.CORRECT if correct else Verdict.WRONG,
            passed=correct,
            # 内容对了但带前后缀：仍可判，但格式合法率要扣——两个维度各记各的
            invalid_format=not clean,
            metrics={"expected": expected, "got": got, "clean": clean},
        )

    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]:
        judged = [g for g in grades if g.attributable]
        n = len(judged)
        return {
            "accuracy": (sum(g.score for g in judged) / n) if n else None,
            "format_valid_rate": (
                sum(1 for g in judged if not g.invalid_format) / n if n else None
            ),
            "n_judged": n,
        }


#: entry point 指向这个对象：任务类 + 它自带的默认数据集
spec = TaskSpec(CharCount, dataset)
