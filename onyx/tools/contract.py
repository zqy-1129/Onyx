"""契约测试运行器：同一套断言跑在所有执行器上。

这是"可替换"的实际含义——不是文档里写着可替换，而是有一组测试逼着它可替换。
断言覆盖的是**失败的种类**而不是"有没有失败"：参数错(arg_error)、被拒(rejected)、
超时(timeout)、路由错(unknown_tool)、跳过(skipped)、实现崩(error) 必须互不相同，
否则"工具调不对"永远无法归因到模型还是工具。

豁免（exemptions）是显式的：某个执行器结构上无法满足某条断言时，
必须写明原因并且**记为 not_applicable，不计入通过**。
静默跳过等于让最关键的保证消失。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from onyx.tools.executor import MockPolicy, ToolCtx, ToolExecutor
from onyx.tools.sandbox import PERMISSIVE_POLICY, SandboxPolicy
from onyx.tools.spec import SideEffect, ToolDef, ToolKind, ToolResult

ExecutorFactory = Callable[[ToolDef], ToolExecutor]

#: 合成"慢工具"与"写工具"定义的构造器：(名字, 角色) → ToolDef。
#: 每种执行器需要自己的版本——http 执行器要的是 URL 而不是 impl_ref，
#: 但**断言本身不变**，这正是"同一套契约跑在所有执行器上"的含义。
SynthFactory = Callable[[str, str], ToolDef]

DEFAULT_SLOW_IMPL = "onyx.tools.builtin.echo:slow_echo"
DEFAULT_WRITE_IMPL = "onyx.tools.builtin.echo:echo"

_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string", "description": "契约测试用的文本参数"}},
    "required": ["text"],
}


def text_schema() -> dict[str, Any]:
    """合成定义共用的参数 schema。

    给各执行器的 `synth` 用的——http 版住在 `executors/http.py`，
    那个模块必须自己 import httpx，所以不能反过来依赖本模块的私有常量。
    """
    return dict(_TEXT_SCHEMA)

#: 契约断言清单（CLI 与看板共用这份名字，便于对比不同执行器的覆盖情况）
CONTRACT_NAMES: tuple[str, ...] = (
    "valid_args_ok",
    "missing_required_is_arg_error",
    "wrong_type_is_arg_error",
    "unknown_tool_is_distinguished",
    "timeout_is_reported",
    "write_denied_without_approval",
    "read_is_idempotent",
    "mock_policy_makes_no_real_call",
)


@dataclass(frozen=True, slots=True)
class ContractResult:
    name: str
    passed: bool
    detail: str = ""
    applicable: bool = True


def run_contracts(
    factory: ExecutorFactory,
    sample: ToolDef,
    *,
    valid_args: dict[str, Any],
    fixtures: Mapping[str, Any] | None = None,
    exemptions: Mapping[str, str] | None = None,
    synth: SynthFactory | None = None,
) -> list[ContractResult]:
    """对一个执行器实现跑全部契约断言。

    `sample` 必须是一个 read 类、参数含 required 字段的工具。
    `fixtures` 给 mock 类执行器用：真实执行器在 LIVE 策略下会忽略它。
    超时与副作用拒绝两条断言需要合成定义，由 `synth` 按执行器类型生成
    （python_fn 给 impl_ref，http 给 URL），断言逻辑本身与执行器无关。
    """
    unknown = set(exemptions or {}) - set(CONTRACT_NAMES)
    if unknown:
        # 拼错的豁免名会静默失效，让一条真实失败被当成"已豁免"
        raise ValueError(f"未知的契约断言名: {sorted(unknown)}；可选 {list(CONTRACT_NAMES)}")

    synth = synth or python_synth()
    out: list[ContractResult] = []
    stubs = dict(fixtures or {})
    ctx = ToolCtx(
        mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY, fixtures=stubs
    )

    # 1) 合法参数必须成功
    result = factory(sample).call(sample.name, dict(valid_args), ctx)
    out.append(ContractResult(
        "valid_args_ok", result.ok,
        result.error or f"output={_short(result.output)} mocked={result.mocked}",
    ))

    # 2) 缺必填 → arg_error（不是崩、不是 200）
    required = list(sample.parameters.get("required") or [])
    key = _tamperable_key(sample, valid_args)
    if required:
        partial = {k: v for k, v in valid_args.items() if k != required[0]}
        out.append(_kind_check(
            factory(sample).call(sample.name, partial, ctx),
            "missing_required_is_arg_error", "arg_error",
        ))
    else:
        out.append(ContractResult(
            "missing_required_is_arg_error", False, "sample 没有 required 字段，无法验证",
            applicable=False,
        ))

    # 3) 类型错 → arg_error
    if key is not None:
        wrong = dict(valid_args)
        wrong[key] = {"not": "a scalar"}
        out.append(_kind_check(
            factory(sample).call(sample.name, wrong, ctx),
            "wrong_type_is_arg_error", "arg_error",
        ))
    else:
        out.append(ContractResult(
            "wrong_type_is_arg_error", False, "无可篡改的参数", applicable=False
        ))

    # 4) 名字对不上 → unknown_tool，必须与参数错区分
    out.append(_kind_check(
        factory(sample).call("definitely_not_" + sample.name, dict(valid_args), ctx),
        "unknown_tool_is_distinguished", "unknown_tool",
    ))

    # 5) 超时 → timeout（用真实的慢实现，不靠 mock 假装）
    slow_def = synth("contract_slow", "slow")
    slow_ctx = replace(ctx, deadline_ms=1)
    out.append(_kind_check(
        factory(slow_def).call(slow_def.name, {"text": "x", "delay_ms": 200}, slow_ctx),
        "timeout_is_reported", "timeout",
    ))

    # 6) write 副作用未审批 → rejected，且副作用不得发生
    write_def = synth("contract_write", "write")
    strict_ctx = replace(ctx, policy=SandboxPolicy(), fixtures={})
    out.append(_kind_check(
        factory(write_def).call(write_def.name, {"text": "x"}, strict_ctx),
        "write_denied_without_approval", "rejected",
    ))

    # 7) read 类工具两次调用结果一致（幂等）
    if "non-deterministic" in sample.tags:
        # 对非确定性工具这条断言是抛硬币：两次调用落在同一秒就会"通过"。
        # 报一个靠运气的 ✓ 比报 n/a 危险得多，所以直接判为不适用。
        out.append(ContractResult(
            "read_is_idempotent", False,
            f"{sample.name} 标记为 non-deterministic，幂等断言无意义", applicable=False,
        ))
    else:
        first = factory(sample).call(sample.name, dict(valid_args), ctx)
        second = factory(sample).call(sample.name, dict(valid_args), ctx)
        out.append(ContractResult(
            "read_is_idempotent",
            first.ok and second.ok and first.output == second.output,
            f"first={_short(first.output)} second={_short(second.output)}",
        ))

    # 8) mock 策略下真实实现零调用
    out.append(_check_mock_isolation(factory, sample, valid_args))

    return _apply_exemptions(out, exemptions or {})


def python_synth(
    *, slow_impl: str = DEFAULT_SLOW_IMPL, write_impl: str = DEFAULT_WRITE_IMPL
) -> SynthFactory:
    """python_fn 执行器用的合成定义：慢工具靠 `slow_echo` 真睡，写工具靠 side_effect 触发拒绝。"""

    def synth(name: str, role: str) -> ToolDef:
        if role == "slow":
            return ToolDef(
                name=name,
                description="契约测试用的慢工具，验证 deadline 真的生效而不是被忽略",
                parameters=dict(_TEXT_SCHEMA), kind=ToolKind.PYTHON_FN,
                impl_ref=slow_impl, side_effect=SideEffect.READ,
            )
        if role == "write":
            return ToolDef(
                name=name,
                description="契约测试用的写操作工具，验证沙箱默认拒绝未审批的写",
                parameters=dict(_TEXT_SCHEMA), kind=ToolKind.PYTHON_FN,
                impl_ref=write_impl, side_effect=SideEffect.WRITE,
            )
        raise ValueError(f"未知的合成角色: {role!r}（可选 slow / write）")

    return synth


def _kind_check(result: ToolResult, name: str, expected: str) -> ContractResult:
    return ContractResult(
        name,
        (not result.ok) and result.error_kind == expected,
        f"kind={result.error_kind or '—'}（期望 {expected}）error={result.error[:120]}",
    )


def _tamperable_key(sample: ToolDef, valid_args: Mapping[str, Any]) -> str | None:
    """挑一个"塞进对象就一定非法"的字段。

    标量字段塞 dict 必然类型错；object/array 字段塞 dict 反而可能合法，
    那样第 3 条断言就变成了对 schema 的偶然测试。
    """
    properties = sample.parameters.get("properties") or {}
    ordered = [name for name in (sample.parameters.get("required") or []) if name in valid_args]
    ordered += [name for name in valid_args if name not in ordered]
    for name in ordered:
        prop = properties.get(name)
        declared = prop.get("type") if isinstance(prop, Mapping) else None
        if isinstance(declared, list | tuple):
            declared = declared[0] if declared else None
        if declared not in {"object", "array"}:
            return name
    return ordered[0] if ordered else None


def sample_args(definition: ToolDef) -> dict[str, Any]:
    """为任意定义造一份合法参数：优先用 `examples[0]`，否则按 required 逐字段造占位值。

    没有 examples 的工具本来就过不了 NO_EXAMPLE 审计，但契约测试不该因此跑不起来——
    否则"审计有 info 级问题"会连带让执行层完全无法验证。
    """
    for example in definition.examples:
        expect = example.get("expect") if isinstance(example, Mapping) else None
        if isinstance(expect, Mapping) and expect.get("name") == definition.name:
            args = expect.get("arguments")
            if isinstance(args, Mapping):
                return dict(args)
    schema = definition.parameters or {}
    properties = schema.get("properties") or {}
    required = schema.get("required") or list(properties)
    return {
        name: _placeholder(properties.get(name) or {}, name)
        for name in required
        if isinstance(name, str)
    }


def _placeholder(prop: Mapping[str, Any], name: str) -> Any:
    enum = prop.get("enum")
    if isinstance(enum, list | tuple) and enum:
        return enum[0]
    declared = prop.get("type")
    if isinstance(declared, list | tuple):
        declared = declared[0] if declared else "string"
    return {
        "string": f"sample-{name}",
        "integer": 1,
        "number": 1.0,
        "boolean": False,
        "array": [],
        "object": {},
        "null": None,
    }.get(str(declared or "string"), f"sample-{name}")


def _check_mock_isolation(
    factory: ExecutorFactory, sample: ToolDef, valid_args: dict[str, Any]
) -> ContractResult:
    """FIXTURE/DENY 策略下必须完全不触达真实实现，且每个结果都标 `mocked=True`。

    证据分两种，`ISOLATION_PROOF` 会把是哪一种写进 detail：
    - `counter`：执行器自己数真实请求（http 执行器），断言计数为 0 是**实测**；
    - `structural`：执行器结构上不可能触达实现（mock_replay 从不解析 impl_ref），
      这条断言只是把该结构性质显式化，真正的实测证据是
      `tests/unit/test_tools_executors.py` 里的 monkeypatch canary。

    不提供 `real_calls` 的执行器标 `applicable=False`，**不能算通过**。
    """
    executor = factory(sample)
    if getattr(executor, "real_calls", None) is None:
        return ContractResult(
            "mock_policy_makes_no_real_call", False,
            f"{type(executor).__name__} 不提供 real_calls 计数，无法证明零真实调用",
            applicable=False,
        )
    proof = getattr(executor, "ISOLATION_PROOF", "counter")
    stub_ctx = ToolCtx(
        mock_policy=MockPolicy.FIXTURE, policy=PERMISSIVE_POLICY,
        fixtures={sample.name: {"mocked": True}},
    )
    stubbed = executor.call(sample.name, dict(valid_args), stub_ctx)
    bare = executor.call(sample.name, dict(valid_args), replace(stub_ctx, fixtures={}))
    denied = executor.call(
        sample.name, dict(valid_args), replace(stub_ctx, mock_policy=MockPolicy.DENY)
    )
    all_mocked = stubbed.mocked and bare.mocked and denied.mocked
    ok = (
        stubbed.ok and stubbed.output == {"mocked": True}
        and not bare.ok and bare.error_kind == "skipped"
        and not denied.ok and denied.error_kind == "skipped"
        and all_mocked and executor.real_calls == 0
    )
    return ContractResult(
        "mock_policy_makes_no_real_call", ok,
        f"stub_ok={stubbed.ok} bare_kind={bare.error_kind} denied_kind={denied.error_kind} "
        f"all_mocked={all_mocked} real_calls={executor.real_calls}（proof={proof}）",
    )


def _apply_exemptions(
    results: list[ContractResult], exemptions: Mapping[str, str]
) -> list[ContractResult]:
    if not exemptions:
        return results
    return [
        replace(item, passed=False, applicable=False, detail=exemptions[item.name])
        if item.name in exemptions
        else item
        for item in results
    ]


def summarize(results: list[ContractResult]) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "not_applicable": 0}
    for item in results:
        if not item.applicable:
            counts["not_applicable"] += 1
        elif item.passed:
            counts["passed"] += 1
        else:
            counts["failed"] += 1
    return counts


def _short(value: Any, limit: int = 120) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "…"
