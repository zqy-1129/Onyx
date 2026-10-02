"""python_fn 执行器：按 `pkg.mod:fn` 调用进程内实现。

安全边界（三道）：
1. `check_impl_ref` 白名单——只有 `SandboxPolicy.allowed_impl_prefixes` 内的模块可导入。
   没有这道闸，一份被篡改的工具定义就能让进程加载任意代码。
2. `extra.constants` 注入的可信参数不许出现在 schema 里，否则模型能覆盖它。
3. 实现自身负责输入净化（见 builtin/calculator.py 的 AST 白名单、
   builtin/fs_read.py 的路径穿越检查）。
"""

from __future__ import annotations

from typing import Any

from onyx.core.errors import ToolUnknown
from onyx.tools.executor import ToolCtx, guarded_call, resolve_impl
from onyx.tools.sandbox import SandboxPolicy, check_impl_ref
from onyx.tools.spec import ToolDef, ToolKind, ToolResult


class PythonFnExecutor:
    kind = ToolKind.PYTHON_FN

    def __init__(self, definition: ToolDef) -> None:
        self._definition = definition

    def spec(self) -> ToolDef:
        return self._definition

    def call(self, name: str, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        definition = self._definition
        if name != definition.name:
            # 名字对不上说明路由错了；必须与"参数错"区分开，否则归因会指向模型
            return ToolResult(
                ok=False, error_kind="unknown_tool",
                error=f"执行器绑定的是 {definition.name!r}，收到 {name!r}",
            )
        policy = ctx.policy
        return guarded_call(
            definition, args, ctx, lambda clean: self._run(clean, policy), name=name
        )

    def _run(self, clean_args: dict[str, Any], policy: SandboxPolicy) -> Any:
        definition = self._definition
        # 白名单必须按**调用方的策略**校验，取默认策略等于把这道闸废掉
        check_impl_ref(definition.impl_ref, policy)
        target = resolve_impl(definition.impl_ref)
        kwargs = _with_constants(definition, clean_args)
        try:
            return target(**kwargs)
        except TypeError as exc:
            # 签名不匹配是**定义与实现脱节**，不是模型的错，也不是运行时故障。
            # 它落到 error 档（而非 unknown_tool），这样评测不会把它算进"模型选错工具"。
            raise ToolUnknown(
                f"实现签名与 schema 不匹配: {exc}",
                detail={"kind": "signature_mismatch", "impl_ref": definition.impl_ref,
                        "args": sorted(kwargs)},
            ) from exc


def _with_constants(definition: ToolDef, clean_args: dict[str, Any]) -> dict[str, Any]:
    """注入 `extra.constants`：定义级、模型不可覆盖的可信参数。

    `fs_read` 的允许根目录就是这么传的。检查的是 **schema 声明了什么**，
    而不是"模型这次传了什么"：只要 root 出现在 properties 里，这个定义就是漏的，
    跟某一次调用有没有被利用无关。等到运行时才发现就已经晚了。
    """
    constants = definition.extra.get("constants")
    if not constants:
        return clean_args
    if not isinstance(constants, dict):
        raise ToolUnknown(
            f"{definition.name} 的 extra.constants 必须是对象",
            detail={"kind": "bad_constants"},
        )
    properties = (definition.parameters or {}).get("properties") or {}
    exposed = sorted(set(constants) & set(properties))
    if exposed:
        raise ToolUnknown(
            f"{definition.name} 的 schema 暴露了受信任常量 {exposed}，模型可以覆盖它",
            detail={"kind": "constant_exposed", "fields": exposed,
                    "fix": "把这些字段从 parameters.properties 里删掉，只留在 extra.constants"},
        )
    return {**clean_args, **constants}
