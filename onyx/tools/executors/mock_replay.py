"""mock_replay 执行器：永不触达真实实现。

它的价值是提供一个**可证明零真实副作用**的参照：评测默认走这一档（DESIGN §8.4），
测试可以断言"真实实现在评测期间一次都没被调用"。没有这个断言，
"评测可复现"就只是一句口号——真实搜索结果/天气会让模型误差和数据漂移混为一谈。

它仍然走 `guarded_call` 全管线（参数校验、沙箱策略都在），只是把最后一步
"真的去调实现"换成"没有桩就报 skipped"。这样评测期照样能测出
"模型给的参数合不合法""这个调用会不会被沙箱拒"，而不是被 mock 一并掩盖。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from onyx.tools.executor import MockPolicy, ToolCtx, guarded_call
from onyx.tools.spec import ToolDef, ToolKind, ToolResult


class MockReplayExecutor:
    kind = ToolKind.FIXTURE
    #: 隔离证据类型：本执行器结构上不可能触达实现（从不解析 impl_ref），
    #: 所以这个计数恒为 0 是**结构性质**而非实测；实测证据见测试里的 monkeypatch canary
    ISOLATION_PROOF = "structural"

    def __init__(
        self,
        definition: ToolDef,
        responses: dict[str, Any] | None = None,
        *,
        default: Any = None,
    ) -> None:
        self._definition = definition
        self._responses = dict(responses or {})
        self._default = default
        #: 调用计数：契约测试断言"真实实现零调用"用
        self.calls: list[dict[str, Any]] = []

    @property
    def real_calls(self) -> int:
        return 0

    def spec(self) -> ToolDef:
        return self._definition

    def call(self, name: str, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        definition = self._definition
        if name != definition.name:
            return ToolResult(
                ok=False, error_kind="unknown_tool",
                error=f"mock 绑定的是 {definition.name!r}，收到 {name!r}",
            )
        self.calls.append({"name": name, "args": args, "trace_id": ctx.trace_id})
        return guarded_call(definition, args, self._force_mock(ctx, name), self._never, name=name)

    def _force_mock(self, ctx: ToolCtx, name: str) -> ToolCtx:
        """把 LIVE 降级为 FIXTURE，并合并执行器自带的桩。

        调用级 `ctx.fixtures` 优先级更高：契约测试与评测用例都是按调用注入桩的。
        """
        stubs: dict[str, Any] = {}
        if name in self._responses:
            stubs[name] = self._responses[name]
        elif self._default is not None:
            stubs[name] = self._default
        stubs.update(ctx.fixtures)
        return replace(
            ctx,
            mock_policy=(
                ctx.mock_policy if ctx.mock_policy is not MockPolicy.LIVE else MockPolicy.FIXTURE
            ),
            fixtures=stubs,
        )

    def _never(self, _clean_args: dict[str, Any]) -> Any:
        # guarded_call 在 FIXTURE 下缺桩会直接返回 skipped，所以这里不该被走到。
        # 留着是因为"永不触达真实实现"是这个执行器唯一的立身之本，
        # 一旦管线顺序被改动，它必须炸出来而不是悄悄真跑。
        raise AssertionError("mock_replay 触达了真实实现：guarded_call 的短路顺序被破坏了")
