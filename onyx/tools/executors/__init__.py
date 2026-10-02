"""L4 执行器。每个执行器只负责"怎么调"，
参数校验/沙箱/超时/错误归一全部由 `executor.guarded_call` 统一处理。

`executor_for` 是唯一的分发点：新增一种执行器只需要在这里登记，
调用方（CLI、评测、循环）不用改。
"""

from __future__ import annotations

import importlib
from typing import Any

from onyx.core.errors import ToolUnknown
from onyx.tools.executor import ToolExecutor
from onyx.tools.spec import ToolDef, ToolKind

from .mock_replay import MockReplayExecutor
from .python_fn import PythonFnExecutor

__all__ = [
    "EXECUTOR_KINDS",
    "MockReplayExecutor",
    "PythonFnExecutor",
    "executor_for",
]

#: 已实现的执行器种类 → 构造器。
#: http 刻意**不在**这里静态导入：httpx 属于 `runtime` extra，
#: 而 `onyx.tools` 其余部分在零三方依赖下也必须能 import（与 core 的可移植性锚点一致）。
_BUILDERS: dict[str, Any] = {
    str(ToolKind.PYTHON_FN): PythonFnExecutor,
    str(ToolKind.FIXTURE): MockReplayExecutor,
}

#: 需要惰性导入的构造器（`模块:属性`）
_LAZY: dict[str, str] = {str(ToolKind.HTTP): "onyx.tools.executors.http:HttpExecutor"}

#: 尚未实现的种类与它们计划落地的里程碑（不许静默降级成 python_fn）
_PENDING: dict[str, str] = {
    str(ToolKind.MCP): "S16（MCP 执行器）",
    str(ToolKind.OLLAMA_BUILTIN): "S16（引擎内建工具，需先跑 P21 探针）",
}

EXECUTOR_KINDS: tuple[str, ...] = tuple(sorted({*_BUILDERS, *_LAZY}))


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
        # mock 通道优先于一切：评测期定义照旧发给模型，执行换成桩。
        # 注意 http 定义被强制成 fixture 后就**不会**发请求，transport 也就不再有意义。
        return MockReplayExecutor(definition, responses, default=default)
    if wanted in _LAZY:
        module_name, _, attr = _LAZY[wanted].partition(":")
        lazy_builder: Any = getattr(importlib.import_module(module_name), attr)
        return lazy_builder(definition, transport=transport)
    builder = _BUILDERS.get(wanted)
    if builder is None:
        pending = _PENDING.get(wanted)
        raise ToolUnknown(
            f"执行器 {wanted!r} 尚未实现" + (f"，计划在 {pending}" if pending else ""),
            detail={"kind": wanted, "tool": definition.name,
                    "available": list(EXECUTOR_KINDS), "planned": pending or ""},
        )
    return builder(definition)
