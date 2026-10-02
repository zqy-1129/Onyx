"""客户端工具循环（DESIGN §8.3）。

Ollama 只返回 `tool_calls`，不会替你执行工具，所以循环必须在本地实现。
而本地小模型上，**异常形态才是常态**：幻觉工具名、`finish_reason=tool_calls`
却没有 tool_calls、参数 JSON 被 max_tokens 截断、同一调用反复出现。
这一层的价值就是把这些形态各自归到一个可行动的码上，而不是混成"失败了"。

不变式（有单测盯着）：**一条带 N 个 tool_calls 的 assistant 消息，后面必须紧跟
恰好 N 条 role=tool 消息**。少一条，之后每一次请求的上下文都永久错位——
而引擎通常不报错，只是开始答非所问，属于最难归因的那类故障。
所以即使中途熔断，剩下的调用也要补占位结果。

异常码分工（避免与 tool visitor 重复上报）：
- visitor 从 `parse_status` / `result_status` 自己推得出 TRUNCATED / MALFORMED / TOOL_ERROR，
  本层**不重复发**；
- `UNKNOWN_TOOL` 只有本层判得出来（visitor 不知道注册表里有什么），由本层发；
- `BUDGET_EXCEEDED` 发生在两步之间，那时已经没有打开的 trace，只能记在 `LoopResult` 上。
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from onyx.core.clock import SYSTEM_CLOCK, Clock
from onyx.core.errors import OnyxError
from onyx.core.ids import new_trace_id
from onyx.core.types import (
    Generation,
    GenerationRequest,
    Message,
    Role,
    TraceContext,
    tool_call_fingerprint,
)
from onyx.llm.gateway import Gateway, TraceEmitter
from onyx.tools.executor import ToolCtx
from onyx.tools.executors import executor_for
from onyx.tools.spec import ToolDef, ToolResult


class StopReason(StrEnum):
    """循环为什么停。**每一种都对应不同的修法**，混成一个"结束"就没法行动。"""

    FINAL = "final"                    # 模型给出了最终回答
    MAX_STEPS = "max_steps"            # 步数上限：任务太复杂或模型在打转
    TOKEN_BUDGET = "token_budget"      # 累计 token 超预算
    WALL_BUDGET = "wall_budget"        # 墙钟超预算
    TOOL_LOOP = "tool_loop"            # 同一 (工具, 参数) 重复，已熔断
    ORPHAN_TOOL_CALL = "orphan"        # 要求调用工具但没给出可调用的内容
    ERROR = "error"                    # 引擎调用失败


@dataclass(frozen=True, slots=True)
class LoopBudget:
    max_steps: int = 6
    #: 累计 in+out token 上限。注意是**各步之和**——那才是真实付出的成本，
    #: 不是最后一步的 prompt 长度（对话在长，每步都要重付一次历史）
    max_total_tokens: int | None = None
    max_wall_ms: float | None = None
    #: 同一 (工具, 参数指纹) 出现超过这个次数即熔断
    repeat_threshold: int = 2


DEFAULT_BUDGET = LoopBudget()


@dataclass(frozen=True, slots=True)
class CallOutcome:
    """单次工具调用的结构化结果。

    `verdict` 与 `result.error_kind` 是两层：verdict 说明**循环为什么这么处理**
    （没执行 / 执行了），error_kind 说明执行失败的种类。缺了 verdict，
    "unknown_tool" 这种根本没进执行器的情况就只能伪装成执行失败。
    """

    name: str
    call_id: str
    #: 与 GENERATION_END 里的 step 对齐（= ToolCall.index + 1），visitor 靠它配对
    ordinal: int
    args: dict[str, Any] | None
    args_raw: str
    parse_status: str
    verdict: str  # executed | unknown_tool | parse_error | loop_break
    result: ToolResult | None = None
    backfilled: bool = False

    def to_tool_message(self) -> Message:
        """补进上下文的 tool 消息。失败也要给结构化内容——模型看到错误才可能改口。"""
        if self.result is None:
            payload: dict[str, Any] = {
                "ok": False,
                "error_kind": self.verdict,
                "error": _VERDICT_TEXT.get(self.verdict, self.verdict),
            }
            if self.verdict == "parse_error" and self.args_raw:
                # 把原文回给模型：截断的 JSON 让它自己看见，比一句"参数错"更可能纠正
                payload["args_raw"] = self.args_raw[:500]
        else:
            payload = {
                "ok": self.result.ok,
                "output": self.result.output if self.result.ok else None,
                "error": self.result.error or None,
                "error_kind": self.result.error_kind or None,
                "mocked": self.result.mocked,
            }
        return Message(
            role=Role.TOOL,
            content=json.dumps(payload, ensure_ascii=False, default=str),
            name=self.name,
            tool_call_id=self.call_id or None,
            extra={"verdict": self.verdict, "backfilled": self.backfilled},
        )


_VERDICT_TEXT = {
    "unknown_tool": "该工具不在注册表里，未被执行",
    "parse_error": "参数不是合法 JSON，未被执行",
    "loop_break": "检测到重复调用已熔断，本次未执行",
}


@dataclass(frozen=True, slots=True)
class StepRecord:
    step: int
    trace_id: str
    text: str
    finish_reason: str
    wanted_tool_call: bool
    in_tokens: int | None
    out_tokens: int | None
    usage_source: str
    usage_confidence: str
    calls: tuple[CallOutcome, ...] = ()
    anomalies: tuple[tuple[str, str, dict[str, Any]], ...] = ()
    latency: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LoopResult:
    text: str
    stop_reason: StopReason
    root_trace_id: str
    steps: tuple[StepRecord, ...] = ()
    messages: tuple[Message, ...] = ()
    #: 循环级异常：发生在两步之间，那时没有打开的 trace 可以落
    anomalies: tuple[tuple[str, str, dict[str, Any]], ...] = ()
    in_tokens: int | None = None
    out_tokens: int | None = None
    #: 各步计数出处不一致时是 `mixed`。**必须报出来**：把 engine 档与 heuristic 档
    #: 加在一起得到一个数，却不说明其中一半是估的，比不报更糟（原则 3：测量必须带出处）
    usage_source: str = ""
    #: 各步里最低的置信度
    usage_confidence: str = ""
    error: str = ""

    @property
    def tool_calls_total(self) -> int:
        return sum(len(step.calls) for step in self.steps)

    @property
    def executed(self) -> int:
        return sum(1 for s in self.steps for c in s.calls if c.verdict == "executed")

    def all_anomalies(self) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        """步骤级 + 循环级。看板与评测只该读这一个口径，别自己拼。"""
        return tuple(a for s in self.steps for a in s.anomalies) + self.anomalies

    def codes(self) -> set[str]:
        return {code for code, _, _ in self.all_anomalies()}


ExecutorFactory = Callable[[ToolDef], Any]


class ToolLoop:
    def __init__(
        self,
        gateway: Gateway,
        tools: Iterable[ToolDef] = (),
        *,
        budget: LoopBudget = DEFAULT_BUDGET,
        ctx: ToolCtx | None = None,
        stop_words: tuple[str, ...] = (),
        executor_factory: ExecutorFactory | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._gateway = gateway
        self._tools: dict[str, ToolDef] = {d.name: d for d in tools}
        self._budget = budget
        self._ctx = ctx or ToolCtx()
        # 停止词必须由模型的 tool_format 探针结果驱动，不要在这里硬编码：
        # 文本模板类模型需要 <tool_response> 之类，原生 tool_calls 头的模型加了反而截断正文
        self._stop_words = tuple(stop_words)
        self._executor_factory = executor_factory or (lambda d: executor_for(d))
        self._clock = clock

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    # ── 主循环 ────────────────────────────────────────────────────
    def run(
        self,
        request: GenerationRequest,
        *,
        purpose: Any | None = None,
        context: TraceContext | None = None,
    ) -> LoopResult:
        budget = self._budget
        root = new_trace_id()
        base_ctx = context or request.context
        messages: list[Message] = list(request.messages)
        steps: list[StepRecord] = []
        loop_anomalies: list[tuple[str, str, dict[str, Any]]] = []
        seen: dict[str, int] = {}
        started_ns = self._clock.monotonic_ns()
        stop = StopReason.FINAL
        error = ""
        text = ""
        in_total = out_total = 0
        missing_usage = False
        sources: set[str] = set()
        worst_confidence = ""

        for step in range(1, budget.max_steps + 1):
            # 预算检查看的是"已测到的部分和"：有步骤缺计数时这个门槛会偏松，
            # 但绝不允许为了凑预算把缺失当成 0 之外的别的数
            exhausted = self._budget_exhausted(started_ns, in_total + out_total, loop_anomalies)
            if exhausted is not None:
                stop = exhausted
                break

            outcomes: list[CallOutcome] = []
            req = request.with_messages(messages)
            if self._stop_words:
                req = req.with_params(stop=self._stop_words)

            def hook(
                gen: Generation, emitter: TraceEmitter,
                _outcomes: list[CallOutcome] = outcomes, _seen: dict[str, int] = seen,
            ) -> None:
                self._execute(gen, emitter, _outcomes, _seen)

            try:
                result = self._gateway.generate(
                    req, purpose=purpose, trace_id=new_trace_id(),
                    context=_link(base_ctx, root), before_trace_end=hook,
                )
            except OnyxError as exc:
                stop = StopReason.ERROR
                error = f"{type(exc).__name__}: {exc}"[:500]
                loop_anomalies.append((
                    "PROVIDER_ERROR", "error",
                    {"step": step, "error": error, "hint": "先跑 onyx doctor 确认服务与模型状态"},
                ))
                break

            gen = result.generation
            usage = result.usage
            if usage is None or usage.in_tokens is None or usage.out_tokens is None:
                # 有一步没测到，总和就标未知——不拿 0 顶上，也不拿"部分和"冒充总量。
                # 每步的值仍在 StepRecord 里，看板可以显示"3/4 步有计数"
                missing_usage = True
            else:
                in_total += usage.in_tokens
                out_total += usage.out_tokens
                sources.add(str(usage.source))
                worst_confidence = _worse(worst_confidence, str(usage.confidence))
            text = gen.text or text
            messages.append(Message(
                role=Role.ASSISTANT, content=gen.text, thinking=gen.thinking,
                tool_calls=gen.tool_calls,
            ))
            steps.append(StepRecord(
                step=step, trace_id=result.trace_id, text=gen.text,
                finish_reason=str(gen.finish_reason), wanted_tool_call=gen.wants_tool_call,
                in_tokens=usage.in_tokens if usage else None,
                out_tokens=usage.out_tokens if usage else None,
                usage_source=str(usage.source) if usage else "",
                usage_confidence=str(usage.confidence) if usage else "",
                calls=tuple(outcomes), anomalies=tuple(result.anomalies),
                latency=result.latency,
            ))

            if not gen.wants_tool_call:
                stop = StopReason.FINAL
                break
            if not gen.tool_calls:
                stop = StopReason.ORPHAN_TOOL_CALL
                break

            # 不变式：每个 tool_call 都必须有一条对应的 tool 消息，包括被熔断的那些
            messages.extend(outcome.to_tool_message() for outcome in outcomes)
            if any(o.verdict == "loop_break" for o in outcomes):
                stop = StopReason.TOOL_LOOP
                break
        else:
            stop = StopReason.MAX_STEPS
            loop_anomalies.append((
                "BUDGET_EXCEEDED", "warn",
                {"reason": "max_steps", "max_steps": budget.max_steps,
                 "hint": "提高 --max-steps，或检查模型是否在反复试同一个调用"},
            ))

        return LoopResult(
            text=text, stop_reason=stop, root_trace_id=root, steps=tuple(steps),
            messages=tuple(messages), anomalies=tuple(loop_anomalies),
            in_tokens=None if missing_usage else in_total,
            out_tokens=None if missing_usage else out_total,
            usage_source=("mixed" if len(sources) > 1 else next(iter(sources), "")),
            usage_confidence=worst_confidence,
            error=error,
        )

    # ── 单步内的工具执行（跑在该步 trace 的窗口内）────────────────
    def _execute(
        self,
        gen: Generation,
        emitter: TraceEmitter,
        outcomes: list[CallOutcome],
        seen: dict[str, int],
    ) -> None:
        if gen.wants_tool_call and not gen.tool_calls:
            emitter.anomaly("ORPHAN_TOOL_CALL", {
                "reason": "finish_reason 要求执行工具，但 tool_calls 为空",
                "finish_reason": str(gen.finish_reason),
                "hint": "本地模型高频形态；检查 max_tokens 是否被 thinking 吃光（P12）",
            })
            return

        broken = False
        for call in gen.tool_calls:
            ordinal = call.index + 1  # 必须与 GENERATION_END 里的 step 一致，否则 visitor 配不上对
            outcome = CallOutcome(
                name=call.name, call_id=call.id, ordinal=ordinal,
                args=call.arguments, args_raw=call.arguments_raw,
                parse_status=str(call.parse_status), verdict="executed",
            )
            outcomes.append(outcome)

            if broken:
                # 已熔断：剩下的调用不执行，但仍要补占位，否则上下文永久错位
                outcomes[-1] = _replace_verdict(outcome, "loop_break", backfilled=True)
                continue

            # 熔断检查排在最前：它是总闸。模型反复输出同一段截断 JSON、
            # 或反复幻觉同一个工具名，都是同一种病——先停下，再谈别的归因。
            # 指纹对解析失败的调用用 args_raw 原文，所以"两次截在同一处"算重复，
            # 而"两次截在不同处"不算。
            fingerprint = tool_call_fingerprint(call.name, call.arguments, call.arguments_raw)
            count = seen.get(fingerprint, 0) + 1
            seen[fingerprint] = count
            if count > self._budget.repeat_threshold:
                emitter.anomaly("TOOL_LOOP", {
                    "tool": call.name, "fingerprint": fingerprint, "count": count,
                    "threshold": self._budget.repeat_threshold,
                    "parse_status": str(call.parse_status),
                    "hint": "熔断；本地小模型常见，检查提示词与工具返回值是否给了模型新信息",
                })
                outcomes[-1] = _replace_verdict(outcome, "loop_break")
                broken = True
                continue

            definition = self._tools.get(call.name)
            if definition is None:
                emitter.anomaly("UNKNOWN_TOOL", {
                    "tool": call.name, "available": sorted(self._tools),
                    "hint": "幻觉工具名；检查工具描述与命名，必要时在提示词里限定可用工具",
                })
                outcomes[-1] = _replace_verdict(outcome, "unknown_tool")
                continue

            if call.arguments is None:
                # 参数没解析出来就别猜着执行：拿 {} 去调会得到一个误导性的 arg_error，
                # 看起来像"模型不会填参数"，其实是"参数根本没送达"
                outcomes[-1] = _replace_verdict(outcome, "parse_error")
                continue

            result = self._run_one(definition, call.name, call.arguments, emitter, ordinal)
            outcomes[-1] = _replace_verdict(outcome, "executed", result=result)

    def _run_one(
        self,
        definition: ToolDef,
        name: str,
        args: dict[str, Any],
        emitter: TraceEmitter,
        ordinal: int,
    ) -> ToolResult:
        # TRUNCATED/MALFORMED/TOOL_ERROR 由 tool visitor 从 parse_status / result_status
        # 自己推，这里不重复发，否则同一个失败会被计两次
        emitter.emit("tool_exec_start", {
            "name": name, "step": ordinal, "executed_by": "client",
            "tool_id": definition.name, "tool_def_hash": definition.hash,
            "args_ref": emitter.blobs.put_json(args),
        })
        ctx = ToolCtx(
            trace_id=emitter.trace_id, deadline_ms=self._ctx.deadline_ms,
            dry_run=self._ctx.dry_run, mock_policy=self._ctx.mock_policy,
            fixtures=self._ctx.fixtures, replay=self._ctx.replay, policy=self._ctx.policy,
            now=self._ctx.now, extra=dict(self._ctx.extra),
        )
        started_ns = self._clock.monotonic_ns()
        result = self._executor_factory(definition).call(name, args, ctx)
        latency_ms = round((self._clock.monotonic_ns() - started_ns) / 1e6, 3)
        emitter.emit("tool_exec_end", {
            "name": name, "step": ordinal, "status": result.status,
            "latency_ms": latency_ms, "result_bytes": result.bytes,
            "mocked": result.mocked, "executed_by": "client",
            "error": result.error,
            # 原始结果落 blob，事件里只放引用：一个大响应不该把事件流撑爆
            "result_ref": emitter.blobs.put_json({
                "ok": result.ok, "output": result.output, "error": result.error,
                "error_kind": result.error_kind, "mocked": result.mocked,
                "extra": result.extra,
            }),
        })
        return result

    def _budget_exhausted(
        self, started_ns: int, tokens: int, sink: list[tuple[str, str, dict[str, Any]]]
    ) -> StopReason | None:
        budget = self._budget
        elapsed_ms = (self._clock.monotonic_ns() - started_ns) / 1e6
        if budget.max_wall_ms and elapsed_ms > budget.max_wall_ms:
            sink.append(("BUDGET_EXCEEDED", "warn", {
                "reason": "wall_ms", "elapsed_ms": round(elapsed_ms, 1),
                "limit_ms": budget.max_wall_ms,
            }))
            return StopReason.WALL_BUDGET
        if budget.max_total_tokens and tokens >= budget.max_total_tokens:
            sink.append(("BUDGET_EXCEEDED", "warn", {
                "reason": "tokens", "tokens": tokens, "limit": budget.max_total_tokens,
                "hint": "这是各步之和，即真实付出的成本；对话在长，每步都要重付一次历史",
            }))
            return StopReason.TOKEN_BUDGET
        return None


#: 置信度排序：数值越大越不可信。跨步求和时取**最差**的那一档，
#: 因为一个总和的可信度不可能高于它最弱的那一项
_CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}


def _worse(current: str, incoming: str) -> str:
    if not current:
        return incoming
    return incoming if _CONFIDENCE_RANK.get(incoming, 9) > _CONFIDENCE_RANK.get(current, 9) else current


def _link(ctx: TraceContext, root: str) -> TraceContext:
    """把每一步挂到同一个 root 上：多步循环在库里必须能按 root_id 聚合出来。

    用 `root_trace_id` 而不是 `parent_trace_id`：root 是循环自己造的分组键，
    **没有对应的 trace 行**。塞进 parent_trace_id 会撞外键约束，
    结果是整条记录写不进去，而表面上只是日志里一行警告。
    """
    return dataclasses.replace(ctx, root_trace_id=ctx.root_trace_id or root)


def _replace_verdict(
    outcome: CallOutcome, verdict: str, *, result: ToolResult | None = None,
    backfilled: bool = False,
) -> CallOutcome:
    return dataclasses.replace(outcome, verdict=verdict, result=result, backfilled=backfilled)
