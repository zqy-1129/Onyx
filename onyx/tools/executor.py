"""执行器契约与统一调用管线。

`guarded_call` 是所有执行器的**唯一入口**：参数校验 → 沙箱策略 → deadline → 错误归一。
这样"非法参数必须变成 ToolArgError"这类保证只需要在一处成立，
而不是每个执行器各自实现一遍（各自实现必然漏）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from onyx.core.errors import (
    ToolArgError,
    ToolError,
    ToolRuntime,
    ToolSandboxDenied,
    ToolSkipped,
    ToolTimeout,
    ToolUnknown,
)
from onyx.tools.args import validate_args
from onyx.tools.sandbox import SandboxPolicy, enforce, run_with_deadline
from onyx.tools.spec import ToolDef, ToolResult


class MockPolicy(StrEnum):
    """评测期默认不碰真实副作用（DESIGN §8.4）。"""

    LIVE = "live"          # 真跑
    FIXTURE = "fixture"    # 用 case 里预置的返回值桩
    REPLAY = "replay"      # 用录制好的历史结果
    DENY = "deny"          # 一律拒绝（纯观测模式）


@dataclass(frozen=True, slots=True)
class ToolCtx:
    trace_id: str = ""
    deadline_ms: int | None = None
    dry_run: bool = False
    mock_policy: MockPolicy = MockPolicy.FIXTURE
    #: 工具名 → 预置返回值（FIXTURE 模式下使用）
    fixtures: Mapping[str, Any] = field(default_factory=dict)
    #: 工具名 → 录制结果（REPLAY 模式下使用）
    replay: Mapping[str, Any] = field(default_factory=dict)
    policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    now: Callable[[], float] = time.monotonic
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolExecutor(Protocol):
    kind: str

    def spec(self) -> ToolDef: ...
    def call(self, name: str, args: dict[str, Any], ctx: ToolCtx) -> ToolResult: ...


#: 异常 → 失败种类。**顺序即优先级**（子类在前）。
#: 这张表必须集中在一处：契约断言比对的是种类名，若同一种错误在不同抛出点
#: 得到不同 kind，"工具调不对"就无法归因到模型还是工具。
#:
#: 注意 `unknown_tool` **不在表里**：它专指路由错（请求的工具名与执行器绑定的定义不符），
#: 由执行器自己产出。`ToolUnknown` 从 `resolve_impl` / 签名不匹配抛出时属于
#: **定义与实现脱节**，必须落到 `error` 档——否则评测会把它记成"模型选错了工具"，
#: 而真正的修法是改工具定义，不是改提示词。
_KIND_BY_ERROR: tuple[tuple[type[ToolError], str], ...] = (
    (ToolSandboxDenied, "rejected"),
    (ToolArgError, "arg_error"),
    (ToolTimeout, "timeout"),
    (ToolSkipped, "skipped"),
)


def kind_for_error(exc: BaseException) -> str:
    for error_type, kind in _KIND_BY_ERROR:
        if isinstance(exc, error_type):
            return kind
    return "error"


def guarded_call(
    definition: ToolDef,
    args: Any,
    ctx: ToolCtx,
    run: Callable[[dict[str, Any]], Any],
    *,
    name: str | None = None,
) -> ToolResult:
    """统一管线。`run` 是执行器真正的实现，只接受已校验的参数 dict。

    顺序是有意义的，两处都不能换：
    1. **参数校验排在 mock 短路之前**——否则评测期一旦给了 fixture，畸形参数就会被
       静默吞掉，而"模型有没有把参数写对"恰恰是评测要测的东西。
    2. **沙箱策略也排在 mock 短路之前**——否则 mock 模式能绕过策略。
       "这个调用会被允许吗"必须在评测里照样有答案，而不是被桩掩盖。
    """
    tool_name = name or definition.name
    started = ctx.now()

    try:
        clean = validate_args(definition.parameters, args)
    except ToolArgError as exc:
        # 参数非法是**模型的问题**，不是工具坏了：必须与运行时错误区分开
        return _failure(exc, "arg_error", started, ctx)

    try:
        enforce(definition, clean, _policy_for(ctx, definition))
    except ToolSandboxDenied as exc:
        return _failure(exc, "rejected", started, ctx)

    if ctx.mock_policy is MockPolicy.DENY:
        return ToolResult(
            ok=False, error="mock_policy=deny：纯观测模式不执行工具",
            error_kind="skipped", mocked=True,
            extra={"latency_ms": round((ctx.now() - started) * 1000, 3), "args": clean},
        )
    if ctx.mock_policy is MockPolicy.FIXTURE:
        if tool_name in ctx.fixtures:
            return _result(ctx.fixtures[tool_name], mocked=True, started=started, ctx=ctx)
        # 没有桩就**不许**退回去真跑：FIXTURE 的含义是"用桩"，静默真跑会让
        # "评测可复现"变成一句口号（DESIGN §8.4），而且现象是数据漂移而非报错
        return _failure(ToolSkipped(
            f"mock_policy=fixture 但没有为 {tool_name} 预置返回值，拒绝真跑",
            detail={"kind": "missing_fixture", "tool": tool_name,
                    "available": sorted(ctx.fixtures)},
        ), "skipped", started, ctx)
    if ctx.mock_policy is MockPolicy.REPLAY:
        if tool_name in ctx.replay:
            return _result(ctx.replay[tool_name], mocked=True, started=started, ctx=ctx)
        return _failure(ToolSkipped(
            f"mock_policy=replay 但没有 {tool_name} 的录制结果，拒绝真跑",
            detail={"kind": "missing_replay", "tool": tool_name,
                    "available": sorted(ctx.replay)},
        ), "skipped", started, ctx)

    timeout = ctx.deadline_ms or definition.timeout_ms or ctx.policy.default_timeout_ms
    try:
        payload = run_with_deadline(lambda: run(clean), timeout, what=f"tool {tool_name}")
    except ToolError as exc:
        # 实现内部抛出的 ToolArgError / ToolSandboxDenied / ToolSkipped 归到与前置检查
        # **同一个 kind**：calculator 的 AST 白名单拒绝、python_fn 的 impl_ref 白名单拒绝
        # 都走这条，不许各自发明新种类。
        return _failure(exc, kind_for_error(exc), started, ctx)
    except Exception as exc:  # noqa: BLE001 - 实现抛的任何异常都必须变成结构化结果
        return _failure(ToolRuntime(f"{type(exc).__name__}: {exc}"[:500]), "error", started, ctx)

    return _result(payload, mocked=False, started=started, ctx=ctx)


def _policy_for(ctx: ToolCtx, definition: ToolDef) -> SandboxPolicy:
    """dry_run 是**调用级**意图，优先级高于策略里的静态设置。"""
    if ctx.dry_run and not ctx.policy.dry_run:
        return SandboxPolicy(
            allowed_side_effects=ctx.policy.allowed_side_effects,
            require_approval=ctx.policy.require_approval,
            dry_run=True,
            default_timeout_ms=ctx.policy.default_timeout_ms,
            allowed_impl_prefixes=ctx.policy.allowed_impl_prefixes,
            approver=ctx.policy.approver,
            extra=ctx.policy.extra,
        )
    return ctx.policy


def _result(payload: Any, *, mocked: bool, started: float, ctx: ToolCtx) -> ToolResult:
    raw = payload if isinstance(payload, str | bytes) else json.dumps(
        payload, ensure_ascii=False, default=str
    )
    size = len(raw.encode("utf-8")) if isinstance(raw, str) else len(raw)
    return ToolResult(
        ok=True, output=payload, mocked=mocked, bytes=size,
        extra={"latency_ms": round((ctx.now() - started) * 1000, 3)},
    )


def _failure(exc: ToolError, kind: str, started: float, ctx: ToolCtx) -> ToolResult:
    return ToolResult(
        ok=False, error=exc.message, error_kind=kind,
        # ToolSkipped 是"按策略没跑"，语义上属于 mock 结果而不是真实失败
        mocked=isinstance(exc, ToolSkipped),
        extra={"latency_ms": round((ctx.now() - started) * 1000, 3), "detail": exc.detail},
    )


def resolve_impl(impl_ref: str) -> Callable[..., Any]:
    """按 `pkg.mod:attr` 解析实现。**调用前必须过 `check_impl_ref` 白名单**。"""
    module_name, _, attr = impl_ref.partition(":")
    if not module_name or not attr:
        raise ToolUnknown(f"impl_ref 格式必须是 pkg.mod:attr，实际 {impl_ref!r}")
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ToolUnknown(f"无法导入 {module_name}: {exc}", detail={"impl_ref": impl_ref}) from exc
    target = getattr(module, attr, None)
    if target is None or not callable(target):
        raise ToolUnknown(f"{module_name} 中没有可调用对象 {attr}", detail={"impl_ref": impl_ref})
    return target
