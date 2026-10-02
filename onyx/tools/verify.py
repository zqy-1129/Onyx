"""模型侧 fire-and-verify：一条指令 → 期望的调用 → 实际发生的调用。

这一层存在的唯一理由是**把"工具调用得分低"拆开**。一个笼统的失败率无法行动，
而下面这几种的修法完全不同：

| verdict | 含义 | 该改什么 |
|---|---|---|
| `NO_CALL` | 根本没调工具，直接编了答案 | 系统提示词 / 工具的 description（"什么时候该用"） |
| `WRONG_TOOL` | 调了，但选错工具 | 工具之间的描述区分度；名字太像就改名 |
| `BAD_ARGS` | 工具对，参数错（缺字段/类型错/枚举越界/JSON 截断） | 参数 description、required、max_tokens |
| `TOOL_FAILED` | 调用完全正确，是工具自己执行失败 | **工具**，不是模型——先跑 `onyx tools contract` |
| `LOOP_BROKEN` | 反复调同一个 (工具, 参数) 被熔断 | 工具返回值没给模型新信息，或提示词 |
| `ERROR` | 引擎调用失败 | 先跑 `onyx doctor` |

把 `TOOL_FAILED` 从"模型不会用工具"里分出来是这一层最重要的判断：
混在一起时，工具坏了会被记成模型能力差，于是你去调提示词——方向完全错。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from onyx.core.types import GenerationRequest, Message
from onyx.tools.args import diff_args
from onyx.tools.loop import StopReason, ToolLoop


class Verdict(StrEnum):
    PASS = "pass"
    NO_CALL = "no_call"
    WRONG_TOOL = "wrong_tool"
    BAD_ARGS = "bad_args"
    TOOL_FAILED = "tool_failed"
    LOOP_BROKEN = "loop_broken"
    ERROR = "error"


#: 期望的参数匹配口径：默认只要求"期望的字段都在且值对"，模型多给字段不算错。
#: 多给一个 `date` 往往比少给一个更有用；要严格就用 `exact=True`。
@dataclass(frozen=True, slots=True)
class Expectation:
    instruction: str
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    exact: bool = False
    case_id: str = ""
    #: 备注：失败时原样带进报告，便于人工判断是模型的问题还是用例写错了
    note: str = ""

    @classmethod
    def from_example(cls, example: Mapping[str, Any], **kw: Any) -> Expectation:
        """从 `ToolDef.examples` 的一条样本构造期望。

        定义里的 examples 本来就是为 fire-and-verify 准备的；审计规则 NO_EXAMPLE
        催的就是它。两处共用同一份数据，才不会各写一套然后悄悄漂移。
        """
        expect = example.get("expect") or {}
        return cls(
            instruction=str(example.get("instruction") or ""),
            tool=str(expect.get("name") or ""),
            arguments=dict(expect.get("arguments") or {}),
            **kw,
        )


@dataclass(frozen=True, slots=True)
class VerifyResult:
    verdict: Verdict
    expectation: Expectation
    #: 实际调过的工具名（按发生顺序，含重复）——WRONG_TOOL 时这就是证据
    called: tuple[str, ...] = ()
    actual_args: dict[str, Any] | None = None
    args_diff: dict[str, Any] = field(default_factory=dict)
    steps: int = 0
    stop_reason: str = ""
    detail: str = ""
    #: 最终消息序列。**必须带出来**：真机上"上下文错位"这类问题只能靠看这条序列诊断，
    #: 而它在 loop 返回后就没了。带 N 个 tool_calls 的 assistant 消息后面必须紧跟
    #: 恰好 N 条 role=tool 消息，少一条之后每次请求都错位，而引擎通常不报错。
    messages: tuple[Message, ...] = ()
    codes: tuple[str, ...] = ()
    #: 执行侧信息（是否 mocked、error_kind），PASS 也要留着：
    #: 用桩跑出来的 PASS 和真跑出来的 PASS 不是一回事
    executed_by: str = ""
    mocked: bool | None = None

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.PASS


def verify_case(
    loop: ToolLoop,
    expectation: Expectation,
    request: GenerationRequest,
) -> VerifyResult:
    """跑一条用例并判定。`request` 里通常只带那条指令，工具由 `loop` 提供。

    判定顺序就是归因优先级：**先看这一轮有没有正常收敛**，再看调没调、调得对不对。
    顺序反了会把"循环熔断"记成 PASS——因为第一次调用的参数往往是对的，
    而模型是在之后的步骤里开始打转的。
    """
    result = loop.run(request)
    calls = [call for step in result.steps for call in step.calls]
    called = tuple(call.name for call in calls)

    base: dict[str, Any] = {
        "expectation": expectation, "called": called, "steps": len(result.steps),
        "stop_reason": str(result.stop_reason), "codes": tuple(sorted(result.codes())),
        "messages": result.messages,
    }

    if result.stop_reason is StopReason.ERROR:
        return VerifyResult(verdict=Verdict.ERROR, detail=result.error or "引擎调用失败", **base)
    if result.stop_reason is StopReason.TOOL_LOOP:
        return VerifyResult(
            verdict=Verdict.LOOP_BROKEN,
            actual_args=calls[-1].args if calls else None,
            detail=(
                f"循环在第 {len(result.steps)} 步熔断（同一工具+参数反复出现）；"
                "工具返回值大概没给模型新信息，或提示词没让它换策略"
            ),
            **base,
        )
    if not calls:
        orphan = "ORPHAN_TOOL_CALL" in base["codes"]
        return VerifyResult(
            verdict=Verdict.NO_CALL,
            detail=(
                "模型声称要调工具（finish_reason=tool_calls）却没给出任何可执行的调用"
                if orphan else
                f"模型没有发起任何工具调用，直接给了正文（{len(result.text)} 字符）"
            ) + f"；finish_reason={result.steps[-1].finish_reason if result.steps else '—'}",
            **base,
        )

    match = next((c for c in calls if c.name == expectation.tool), None)
    if match is None:
        return VerifyResult(
            verdict=Verdict.WRONG_TOOL,
            detail=f"期望 {expectation.tool}，实际调了 {list(dict.fromkeys(called))}",
            **base,
        )

    if match.args is None:
        # 参数根本没解析出来。这是模型的输出问题，但修法与"填错字段"不同：
        # 截断要加 max_tokens，格式错要看模板与停止词
        return VerifyResult(
            verdict=Verdict.BAD_ARGS, actual_args=None,
            args_diff={"parse_status": match.parse_status, "args_raw": match.args_raw[:300]},
            detail=f"参数未能解析（parse_status={match.parse_status}）；原文已保留在 args_raw",
            **base,
        )

    diff = diff_args(expectation.arguments, match.args)
    passed = diff["ok"] if expectation.exact else diff["subset_ok"]
    if not passed:
        return VerifyResult(
            verdict=Verdict.BAD_ARGS, actual_args=match.args, args_diff=diff,
            detail=_describe_diff(diff, exact=expectation.exact), **base,
        )

    # 参数对了，接下来看执行：失败是**工具**的问题，不能记在模型头上
    outcome = match.result
    if outcome is not None and not outcome.ok and outcome.error_kind != "skipped":
        return VerifyResult(
            verdict=Verdict.TOOL_FAILED, actual_args=match.args, args_diff=diff,
            executed_by="client", mocked=outcome.mocked,
            detail=(
                f"调用正确但执行失败：{outcome.error_kind} — {outcome.error[:200]}"
                "（先跑 onyx tools contract 确认是工具坏了还是配置问题）"
            ),
            **base,
        )
    return VerifyResult(
        verdict=Verdict.PASS, actual_args=match.args, args_diff=diff,
        executed_by="client", mocked=None if outcome is None else outcome.mocked,
        detail="mocked（用桩）" if outcome is not None and outcome.mocked else "真实执行",
        **base,
    )


def verify_many(
    factory: Callable[[], ToolLoop],
    cases: Iterable[tuple[Expectation, GenerationRequest]],
) -> list[VerifyResult]:
    """逐条跑。**每条都新建一个 loop**：熔断计数与预算是状态，跨用例复用会互相污染。"""
    return [verify_case(factory(), expectation, request) for expectation, request in cases]


def summarize(results: Sequence[VerifyResult]) -> dict[str, Any]:
    counts = {str(v): 0 for v in Verdict}
    for item in results:
        counts[str(item.verdict)] += 1
    total = len(results)
    # 通过率的分母只算"能判定模型对错"的用例：TOOL_FAILED 是工具的缺陷，
    # ERROR 是环境问题，把它们算进模型得分就等于让模型替工具背锅
    attributable = total - counts[str(Verdict.TOOL_FAILED)] - counts[str(Verdict.ERROR)]
    return {
        "total": total,
        "counts": counts,
        "passed": counts[str(Verdict.PASS)],
        "attributable": attributable,
        "pass_rate": round(counts[str(Verdict.PASS)] / attributable, 4) if attributable else None,
        "mocked_passes": sum(1 for r in results if r.verdict is Verdict.PASS and r.mocked),
    }


def _describe_diff(diff: Mapping[str, Any], *, exact: bool) -> str:
    parts: list[str] = []
    if diff.get("missing"):
        parts.append(f"缺字段 {diff['missing']}")
    if diff.get("mismatched"):
        shown = {k: v for k, v in list(diff["mismatched"].items())[:3]}
        parts.append(f"值不对 {shown}")
    if exact and diff.get("unexpected"):
        parts.append(f"多余字段 {diff['unexpected']}（exact=True）")
    return "；".join(parts) or "参数不匹配"
