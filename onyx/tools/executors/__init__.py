"""L4 执行器。每个执行器只负责"怎么调"，
参数校验/沙箱/超时/错误归一全部由 `executor.guarded_call` 统一处理。

`executor_for` 是唯一的分发点：新增一种执行器只需要登记（内置改这里，
外部插件走 entry points `onyx.tool_executors`），调用方（CLI、评测、循环）不用改。

插件形状：` Callable[[ToolDef], ToolExecutor]`（通常是执行器类本身）。
配置从 `ToolDef` 里带（`impl_ref` / `extra`），不靠注册时的全局参数——
否则"同一个工具在不同 case 里打到不同后端"就没有可追溯的解释了。
"""

from __future__ import annotations

import importlib
from typing import Any

from onyx.core.errors import ToolUnknown
from onyx.discovery import GROUP_TOOL_EXECUTORS, discover
from onyx.tools.executor import ToolExecutor
from onyx.tools.spec import ToolDef, ToolKind

from .mock_replay import MockReplayExecutor
from .python_fn import PythonFnExecutor

__all__ = [
    "EXECUTOR_KINDS",
    "PENDING_KINDS",
    "MockReplayExecutor",
    "PythonFnExecutor",
    "executor_for",
    "executor_kinds",
]

#: 已实现的执行器种类 → 构造器。
#: http 刻意**不在**这里静态导入：httpx 属于 `runtime` extra，
#: 而 `onyx.tools` 其余部分在零三方依赖下也必须能 import（与 core 的可移植性锚点一致）。
_BUILDERS: dict[str, Any] = {
    str(ToolKind.PYTHON_FN): PythonFnExecutor,
    str(ToolKind.FIXTURE): MockReplayExecutor,
}

#: 需要惰性导入的构造器（`模块:属性`）。
#: http 要 httpx；mcp 虽然只用 stdlib，但它的客户端会 spawn 子进程，
#: 让 `import onyx.tools` 顺手带上这套东西没有道理。
_LAZY: dict[str, str] = {
    str(ToolKind.HTTP): "onyx.tools.executors.http:HttpExecutor",
    str(ToolKind.MCP): "onyx.tools.executors.mcp:McpExecutor",
}

#: 尚未实现的种类与它们计划落地的里程碑（不许静默降级成 python_fn）
_PENDING: dict[str, str] = {
    str(ToolKind.OLLAMA_BUILTIN): "S16+（引擎内建工具，需先跑 P21 探针：模板里工具到底怎么渲染）",
}

#: 对外暴露：CLI 的契约矩阵从这里读"还剩谁没实现"，写死在 CLI 里就会在实现完成后
#: 继续宣称"未实现"——那句报错看起来永远合理，没人会去核对
PENDING_KINDS: dict[str, str] = dict(_PENDING)

#: 内建种类（不含插件）——契约测试与审计用它，值是稳定的
EXECUTOR_KINDS: tuple[str, ...] = tuple(sorted({*_BUILDERS, *_LAZY}))


def _plugin_builders() -> dict[str, Any]:
    return discover(GROUP_TOOL_EXECUTORS, {})


def executor_kinds() -> tuple[str, ...]:
    """内建 + 插件注册的执行器种类。"""
    return tuple(sorted({*EXECUTOR_KINDS, *_plugin_builders()}))


def executor_for(
    definition: ToolDef,
    *,
    kind: str | None = None,
    responses: dict[str, Any] | None = None,
    default: Any = None,
    transport: Any | None = None,
) -> ToolExecutor:
    """按定义（或显式指定的 kind）构造执行器。

    `kind="fixture"` 可以把任何定义强制走 mock 通道——评测期就是这么用的：
    定义照旧发给模型，执行换成桩，从而做到零真实副作用。
    `transport` 只给 http 执行器用，注入后可以在离线测试里断言零真实网络。
    """
    wanted = kind or str(definition.kind)
    if wanted == str(ToolKind.FIXTURE):
        # mock 通道优先于一切，也优先于插件：评测的零副作用保证不能被外部实现劫走。
        # 注意 http 定义被强制成 fixture 后就**不会**发请求，transport 也就不再有意义。
        return MockReplayExecutor(definition, responses, default=default)
    # 其余种类：插件优先（同名覆盖是 DESIGN §13 留的"就地替换实现"口子，
    # 覆盖关系由 `onyx plugins` 显式列出，不悄悄发生）。
    plugin = _plugin_builders().get(wanted)
    if plugin is not None:
        return plugin(definition)
    if wanted in _LAZY:
        module_name, _, attr = _LAZY[wanted].partition(":")
        lazy_builder: Any = getattr(importlib.import_module(module_name), attr)
        # 只有拿到注入对象时才传 `transport`：mcp 执行器没有 httpx transport 这个概念，
        # 强行给它加一个"接收但忽略"的参数，就等于让"我注入了替身"这件事变成幻觉
        kwargs = {"transport": transport} if transport is not None else {}
        return lazy_builder(definition, **kwargs)
    builder = _BUILDERS.get(wanted)
    if builder is None:
        pending = _PENDING.get(wanted)
        raise ToolUnknown(
            f"执行器 {wanted!r} 尚未实现" + (f"，计划在 {pending}" if pending else ""),
            detail={"kind": wanted, "tool": definition.name,
                    "available": list(executor_kinds()), "planned": pending or ""},
        )
    return builder(definition)
