"""沙箱：副作用策略、审批、dry-run、超时。

本地工具会真的动你的机器（写文件、发网络请求、执行命令），而调用参数来自**模型输出**。
所以默认策略是最严的：只允许 `read`，其余一律拒绝，除非显式放开或经审批。

超时的诚实说明：Python 无法杀死线程，所以 deadline 到点后我们**放弃等待并标记超时**，
底层线程可能仍在跑。这对 read 类工具无害；对 write/exec 类工具，
唯一可靠的隔离是子进程或容器——策略里因此默认拒绝这两类。
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from onyx.core.errors import ToolSandboxDenied, ToolTimeout
from onyx.tools.spec import SideEffect, ToolDef

#: 允许的 impl_ref 前缀。`python_fn` 执行器会按 `pkg.mod:fn` 动态导入，
#: 没有白名单就等于让模型输出决定进程加载什么代码。
DEFAULT_ALLOWED_IMPL_PREFIXES: tuple[str, ...] = ("onyx.tools.builtin.",)

DEFAULT_TIMEOUT_MS = 10_000

_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="onyx-tool"
)


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    allowed_side_effects: frozenset[SideEffect] = frozenset({SideEffect.READ})
    require_approval: frozenset[SideEffect] = frozenset(
        {SideEffect.WRITE, SideEffect.NETWORK, SideEffect.EXEC}
    )
    dry_run: bool = False
    default_timeout_ms: int = DEFAULT_TIMEOUT_MS
    allowed_impl_prefixes: tuple[str, ...] = DEFAULT_ALLOWED_IMPL_PREFIXES
    #: 审批回调：(工具名, 参数) → 是否放行。None 表示无人可批 ⇒ 直接拒绝
    approver: Callable[[str, dict[str, Any]], bool] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


#: 契约测试/本地实验用：放开网络与写，但仍要求审批为假（不弹交互）
PERMISSIVE_POLICY = SandboxPolicy(
    allowed_side_effects=frozenset(SideEffect), require_approval=frozenset(), dry_run=False
)


def enforce(definition: ToolDef, args: dict[str, Any], policy: SandboxPolicy) -> None:
    """执行前的策略检查。不通过就抛 ToolSandboxDenied，副作用不许发生。"""
    effect = definition.side_effect
    if effect not in policy.allowed_side_effects:
        raise ToolSandboxDenied(
            f"工具 {definition.name} 的副作用 {effect} 不在允许范围内 "
            f"{sorted(str(s) for s in policy.allowed_side_effects)}",
            detail={"tool": definition.name, "side_effect": str(effect),
                    "allowed": sorted(str(s) for s in policy.allowed_side_effects)},
        )
    if policy.dry_run and effect is not SideEffect.READ:
        raise ToolSandboxDenied(
            f"dry-run 模式下只允许 read 类工具，{definition.name} 是 {effect}",
            detail={"tool": definition.name, "side_effect": str(effect), "dry_run": True},
        )
    if effect in policy.require_approval:
        approver = policy.approver
        approved = bool(approver(definition.name, args)) if approver else False
        if not approved:
            raise ToolSandboxDenied(
                f"工具 {definition.name}（{effect}）需要审批，未获批准",
                detail={"tool": definition.name, "side_effect": str(effect), "needs_approval": True},
            )


def check_impl_ref(impl_ref: str, policy: SandboxPolicy) -> None:
    """动态导入前的白名单检查。

    这条不能省：`python_fn` 执行器按字符串导入模块，若前缀不受限，
    一份被篡改的工具定义就能让进程加载任意代码。
    """
    if not impl_ref:
        raise ToolSandboxDenied("impl_ref 为空", detail={"kind": "empty_impl_ref"})
    if not any(impl_ref.startswith(prefix) for prefix in policy.allowed_impl_prefixes):
        raise ToolSandboxDenied(
            f"impl_ref {impl_ref!r} 不在白名单内",
            detail={
                "impl_ref": impl_ref,
                "allowed_prefixes": list(policy.allowed_impl_prefixes),
                "hint": "把可信的实现包加入 SandboxPolicy.allowed_impl_prefixes",
            },
        )


def run_with_deadline[T](
    fn: Callable[[], T], timeout_ms: int | None, *, what: str = "tool"
) -> T:
    """在 deadline 内执行；超时抛 ToolTimeout。

    注意：超时后底层线程不会被杀死（Python 限制），只是不再等待。
    因此 write/exec 类工具默认被策略拒绝，而不是依赖这里的超时来兜底。
    """
    if not timeout_ms or timeout_ms <= 0:
        return fn()
    future = _executor.submit(fn)
    try:
        return future.result(timeout=timeout_ms / 1000)
    except concurrent.futures.TimeoutError as exc:
        raise ToolTimeout(
            f"{what} 超过 {timeout_ms}ms 未完成（已放弃等待，底层线程可能仍在运行）",
            detail={"timeout_ms": timeout_ms, "what": what},
        ) from exc
