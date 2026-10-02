"""S11 验收：执行器 + 沙箱 + 契约测试运行器。

重点不是"能跑通"，而是**失败的种类互不相同**：
arg_error / rejected / timeout / unknown_tool / skipped / error 六类必须可区分，
否则"工具调不对"永远无法归因到模型还是工具。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from onyx.core.errors import ToolSandboxDenied, ToolTimeout, ToolUnknown
from onyx.llm.measurement.heuristic import estimate_tokens
from onyx.tools.args import diff_args, validate_args
from onyx.tools.builtin.calculator import calculate
from onyx.tools.builtin.defs import BUILTIN_DEFS, CALCULATOR, CONTRACT_SAMPLE, ECHO, TIME_NOW
from onyx.tools.contract import CONTRACT_NAMES, run_contracts, sample_args, summarize
from onyx.tools.executor import MockPolicy, ToolCtx, kind_for_error
from onyx.tools.executors import EXECUTOR_KINDS, MockReplayExecutor, PythonFnExecutor, executor_for
from onyx.tools.registry import defs_from_payload
from onyx.tools.sandbox import (
    PERMISSIVE_POLICY,
    SandboxPolicy,
    check_impl_ref,
    enforce,
    run_with_deadline,
)
from onyx.tools.spec import Severity, SideEffect, ToolDef, ToolKind, audit

SAMPLE_ARGS = {"text": "onyx", "times": 1}
MOCK_STUBS = {"echo": {"echo": "onyx", "times": 1, "chars": 4}}
#: mock 执行器结构上不触达真实实现，deadline 无从生效——必须显式写明，不许静默跳过
MOCK_EXEMPTIONS = {
    "timeout_is_reported": (
        "mock 执行器不导入也不调用真实实现，deadline 无从生效；"
        "这条保证由 PythonFnExecutor 上的同名断言覆盖"
    )
}


def _live_ctx(**kw) -> ToolCtx:
    return ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY, **kw)


def _kinds(results) -> dict[str, str]:
    return {item.name: item.detail for item in results}


# ── 契约断言：两个执行器跑同一套 ──────────────────────────────────
def test_python_fn_executor_passes_every_applicable_contract():
    results = run_contracts(PythonFnExecutor, CONTRACT_SAMPLE, valid_args=SAMPLE_ARGS)
    counts = summarize(results)
    assert counts["failed"] == 0, _kinds(results)
    assert counts["passed"] == len(CONTRACT_NAMES) - 1
    # 唯一不适用的一条必须写明原因，不能是空字符串
    exempted = [item for item in results if not item.applicable]
    assert [item.name for item in exempted] == ["mock_policy_makes_no_real_call"]
    assert "无法证明零真实调用" in exempted[0].detail


def test_mock_replay_executor_passes_every_applicable_contract():
    results = run_contracts(
        MockReplayExecutor, CONTRACT_SAMPLE,
        valid_args=SAMPLE_ARGS, fixtures=MOCK_STUBS, exemptions=MOCK_EXEMPTIONS,
    )
    counts = summarize(results)
    assert counts["failed"] == 0, _kinds(results)
    assert counts == {"passed": 7, "failed": 0, "not_applicable": 1}


def test_unknown_exemption_name_is_rejected():
    """拼错的豁免名会静默失效，把一条真实失败伪装成"已豁免"。"""
    with pytest.raises(ValueError, match="未知的契约断言名"):
        run_contracts(
            MockReplayExecutor, CONTRACT_SAMPLE, valid_args=SAMPLE_ARGS,
            fixtures=MOCK_STUBS, exemptions={"timeout_is_reportd": "typo"},
        )


def test_exempted_assertion_never_counts_as_passed():
    results = run_contracts(
        MockReplayExecutor, CONTRACT_SAMPLE,
        valid_args=SAMPLE_ARGS, fixtures=MOCK_STUBS, exemptions=MOCK_EXEMPTIONS,
    )
    timeout = next(item for item in results if item.name == "timeout_is_reported")
    assert not timeout.passed and not timeout.applicable
    # 豁免必须留下**为什么**，否则矩阵里的 n/a 就是一句空话
    assert timeout.detail == MOCK_EXEMPTIONS["timeout_is_reported"]


# ── 失败种类互不相同 ──────────────────────────────────────────────
SLOW = ToolDef(
    name="contract_slow", description="故意慢的工具，用来触发真实的 deadline",
    parameters={"type": "object", "properties": {"text": {"type": "string", "description": "文本"}},
                "required": ["text"]},
    kind=ToolKind.PYTHON_FN, impl_ref="onyx.tools.builtin.echo:slow_echo",
    side_effect=SideEffect.READ,
)
WRITE = ToolDef(
    name="contract_write", description="故意带写副作用的工具，用来触发沙箱拒绝",
    parameters={"type": "object", "properties": {"text": {"type": "string", "description": "内容"}},
                "required": ["text"]},
    kind=ToolKind.PYTHON_FN, impl_ref="onyx.tools.builtin.echo:echo",
    side_effect=SideEffect.WRITE,
)
CRASHY = ToolDef(
    name="contract_boom", description="实现自己会崩，用来验证 error 这一档确实存在",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
    kind=ToolKind.PYTHON_FN, impl_ref="onyx.tools.builtin.echo:boom",
    side_effect=SideEffect.READ,
)
STRICT = ToolCtx(mock_policy=MockPolicy.LIVE, policy=SandboxPolicy())


def test_failure_kinds_are_pairwise_distinguished():
    """六类失败必须互不相同，否则"工具调不对"无法归因到模型还是工具。"""
    observed = {
        "arg_error": PythonFnExecutor(ECHO).call("echo", {}, _live_ctx()).error_kind,
        "unknown_tool": PythonFnExecutor(ECHO).call(
            "nope", dict(SAMPLE_ARGS), _live_ctx()).error_kind,
        "timeout": PythonFnExecutor(SLOW).call(
            SLOW.name, {"text": "x", "delay_ms": 200}, _live_ctx(deadline_ms=1)).error_kind,
        "rejected": PythonFnExecutor(WRITE).call(WRITE.name, {"text": "x"}, STRICT).error_kind,
        "skipped": PythonFnExecutor(ECHO).call(
            "echo", dict(SAMPLE_ARGS),
            ToolCtx(mock_policy=MockPolicy.DENY, policy=PERMISSIVE_POLICY)).error_kind,
        "error": PythonFnExecutor(CRASHY).call(CRASHY.name, {}, _live_ctx()).error_kind,
    }
    assert observed == {name: name for name in observed}, observed
    assert len(set(observed.values())) == 6


def test_successful_call_has_no_error_kind():
    result = PythonFnExecutor(ECHO).call("echo", dict(SAMPLE_ARGS), _live_ctx())
    assert result.ok and result.error_kind == "" and result.status == "ok"


def test_kind_for_error_maps_by_subclass():
    assert kind_for_error(ToolSandboxDenied("x")) == "rejected"
    assert kind_for_error(ToolTimeout("x")) == "timeout"
    # ToolUnknown 不映射成 unknown_tool：它来自定义与实现脱节，不是模型选错工具
    assert kind_for_error(ToolUnknown("x")) == "error"
    assert kind_for_error(ValueError("x")) == "error"


def test_impl_signature_drift_is_a_tool_defect_not_a_routing_error():
    """schema 声明的字段实现并不接受。若记成 unknown_tool，
    评测会把它算进"模型选错工具"，而真正的修法是改定义。"""
    drifted = ToolDef(
        name="drifted", description="schema 与实现签名脱节的样本",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，实现并不接受它"}},
            "required": ["city"],
        },
        kind=ToolKind.PYTHON_FN, impl_ref="onyx.tools.builtin.echo:echo",
        side_effect=SideEffect.READ,
    )
    result = PythonFnExecutor(drifted).call("drifted", {"city": "北京"}, _live_ctx())
    assert not result.ok
    assert result.error_kind == "error"
    assert result.extra["detail"]["kind"] == "signature_mismatch"


def test_unresolvable_impl_ref_is_a_tool_defect():
    definition = ToolDef(
        name="ghost", description="impl_ref 指向不存在的实现",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        kind=ToolKind.PYTHON_FN, impl_ref="onyx.tools.builtin.echo:not_there",
        side_effect=SideEffect.READ,
    )
    result = PythonFnExecutor(definition).call("ghost", {}, _live_ctx())
    assert not result.ok and result.error_kind == "error"
    assert "not_there" in result.error


# ── 沙箱 ──────────────────────────────────────────────────────────
def test_default_policy_denies_write_network_and_exec():
    definition = ToolDef(name="w", description="x" * 30, side_effect=SideEffect.WRITE)
    for effect in (SideEffect.WRITE, SideEffect.NETWORK, SideEffect.EXEC):
        with pytest.raises(ToolSandboxDenied):
            enforce(ToolDef(name="w", description="x" * 30, side_effect=effect),
                    {}, SandboxPolicy())
    enforce(definition, {}, PERMISSIVE_POLICY)  # 放开后不抛


def test_approver_can_let_a_write_through():
    seen: list[tuple[str, dict]] = []
    policy = SandboxPolicy(
        allowed_side_effects=frozenset(SideEffect),
        require_approval=frozenset({SideEffect.WRITE}),
        approver=lambda name, args: seen.append((name, args)) or True,
    )
    definition = ToolDef(name="w", description="x" * 30, side_effect=SideEffect.WRITE,
                         impl_ref="onyx.tools.builtin.echo:echo")
    result = PythonFnExecutor(definition).call("w", {"text": "hi"},
                                               ToolCtx(mock_policy=MockPolicy.LIVE, policy=policy))
    assert result.ok and seen == [("w", {"text": "hi"})]


def test_dry_run_blocks_everything_but_read():
    definition = ToolDef(name="w", description="x" * 30, side_effect=SideEffect.WRITE,
                         impl_ref="onyx.tools.builtin.echo:echo")
    permissive = ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY)
    assert PythonFnExecutor(definition).call("w", {"text": "x"}, permissive).ok
    # dry_run 是**调用级**意图，即使策略本身放开了也必须拦住
    dry = ToolCtx(mock_policy=MockPolicy.LIVE, policy=PERMISSIVE_POLICY, dry_run=True)
    result = PythonFnExecutor(definition).call("w", {"text": "x"}, dry)
    assert not result.ok and result.error_kind == "rejected"
    assert result.extra["detail"]["dry_run"] is True


def test_impl_ref_whitelist_is_checked_against_the_callers_policy():
    """白名单必须按调用方策略校验，取默认策略等于把这道闸废掉。"""
    definition = ToolDef(name="evil", description="x" * 30, impl_ref="os:system")
    result = PythonFnExecutor(definition).call("evil", {}, _live_ctx())
    assert not result.ok
    # 落在 rejected 而不是 error：这是策略决定，不是实现故障
    assert result.error_kind == "rejected"
    assert "白名单" in result.error

    with pytest.raises(ToolSandboxDenied):
        check_impl_ref("os:system", SandboxPolicy())
    check_impl_ref("onyx.tools.builtin.echo:echo", SandboxPolicy())


def test_run_with_deadline_reports_timeout_and_leaks_the_thread_honestly():
    import time

    finished = []

    def slow() -> str:
        time.sleep(0.3)
        finished.append(True)
        return "late"

    with pytest.raises(ToolTimeout):
        run_with_deadline(slow, 10, what="test")
    assert run_with_deadline(lambda: "fast", None) == "fast"
    assert run_with_deadline(lambda: "fast", 0) == "fast"


# ── mock 隔离的强证明 ─────────────────────────────────────────────
def test_mock_executor_never_touches_the_real_implementation(monkeypatch):
    """canary：把真实实现换成计数器。mock 通道计数必须是 0，
    真实通道必须 >0——否则这条断言是空的（canary 根本没接上）。"""
    import importlib

    calls: list[dict] = []

    def canary(text: str = "", times: int = 1):
        calls.append({"text": text, "times": times})
        return {"echo": text, "times": times, "chars": len(text)}

    # 注意：`onyx.tools.builtin.echo` 这个属性名被 __init__ 里的函数导出遮蔽了，
    # 所以必须走 import_module 拿到模块对象本身，否则 monkeypatch 会打到函数上
    echo_module = importlib.import_module("onyx.tools.builtin.echo")
    monkeypatch.setattr(echo_module, "echo", canary)

    real = PythonFnExecutor(ECHO)
    assert real.call("echo", dict(SAMPLE_ARGS), _live_ctx()).ok
    assert len(calls) == 1, "canary 没接上，下面对 mock 的断言毫无意义"

    calls.clear()
    mock = MockReplayExecutor(ECHO, MOCK_STUBS)
    for ctx in (
        _live_ctx(),                                                    # LIVE 也被降级
        ToolCtx(mock_policy=MockPolicy.FIXTURE, policy=PERMISSIVE_POLICY),
        ToolCtx(mock_policy=MockPolicy.DENY, policy=PERMISSIVE_POLICY),
        ToolCtx(mock_policy=MockPolicy.REPLAY, policy=PERMISSIVE_POLICY,
                replay={"echo": {"replayed": True}}),
    ):
        mock.call("echo", dict(SAMPLE_ARGS), ctx)
    assert calls == [], f"mock 执行器触达了真实实现: {calls}"
    assert len(mock.calls) == 4 and all(entry["name"] == "echo" for entry in mock.calls)


def test_mock_results_are_always_labelled_mocked():
    """看板上 mocked 徽标依赖这个字段；漏标就等于把桩数据当真实测量展示。"""
    mock = MockReplayExecutor(ECHO, MOCK_STUBS)
    for ctx in (
        _live_ctx(),
        ToolCtx(mock_policy=MockPolicy.FIXTURE, policy=PERMISSIVE_POLICY, fixtures=MOCK_STUBS),
        ToolCtx(mock_policy=MockPolicy.DENY, policy=PERMISSIVE_POLICY),
    ):
        result = mock.call("echo", dict(SAMPLE_ARGS), ctx)
        assert result.mocked, f"{ctx.mock_policy} 下结果未标 mocked"
        assert result.status == "mocked"


def test_fixture_mode_still_validates_arguments():
    """给了桩不等于放过畸形参数——"模型有没有把参数写对"正是评测要测的。"""
    mock = MockReplayExecutor(ECHO, MOCK_STUBS)
    result = mock.call("echo", {}, _live_ctx())
    assert not result.ok and result.error_kind == "arg_error"
    assert result.extra["detail"]["kind"] == "missing_required"


def test_replay_policy_uses_recorded_result():
    mock = MockReplayExecutor(ECHO)
    ctx = ToolCtx(mock_policy=MockPolicy.REPLAY, policy=PERMISSIVE_POLICY,
                  replay={"echo": {"echo": "onyx", "times": 1, "chars": 4}})
    result = mock.call("echo", dict(SAMPLE_ARGS), ctx)
    assert result.ok and result.mocked and result.output["chars"] == 4


# ── 计算器：AST 白名单，绝不用 eval ───────────────────────────────
def test_calculator_computes_whitelisted_arithmetic():
    assert calculate("2+3*4")["value"] == 14
    assert calculate("(12+8)*3")["value"] == 60
    assert calculate("2**10")["value"] == 1024
    assert calculate("7%3")["value"] == 1
    assert calculate("-5+2")["value"] == -3
    assert calculate("sqrt(2)")["value"] == pytest.approx(1.41421356, rel=1e-6)
    assert calculate("round(0.1+0.2, 2)")["value"] == 0.3


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "(1).__class__.__bases__",
    "open('/etc/passwd').read()",
    "[x for x in range(10)]",
    "lambda: 1",
    "{'a': 1}",
    "'abc'",
    "os",
    "pi.__class__",
    "calculate('1')",
])
def test_calculator_rejects_everything_outside_the_whitelist(expr):
    """这些都必须变成 ToolArgError（→ arg_error），既不能崩也不能执行。"""
    from onyx.core.errors import ToolArgError

    with pytest.raises(ToolArgError):
        calculate(expr)


def test_calculator_rce_attempt_is_an_arg_error_not_a_crash():
    result = PythonFnExecutor(CALCULATOR).call(
        "calculator", {"expr": "__import__('os').system('echo pwned')"}, _live_ctx()
    )
    assert not result.ok and result.error_kind == "arg_error"
    assert "白名单" in result.error
    # detail 里必须留下原文与分类，否则评测无法聚合"模型写的表达式哪里不合法"
    assert result.extra["detail"]["kind"] == "unsafe_expression"
    assert "__import__" in result.extra["detail"]["expr"]


def test_calculator_resource_limits():
    from onyx.core.errors import ToolArgError

    with pytest.raises(ToolArgError, match="指数过大"):
        calculate("2**99999999")
    with pytest.raises(ToolArgError, match="过长"):
        calculate("1+" * 600 + "1")
    with pytest.raises(ToolArgError, match="除零"):
        calculate("1/0")
    with pytest.raises(ToolArgError, match="为空"):
        calculate("   ")


# ── 参数校验与差异报告 ────────────────────────────────────────────
def test_validate_args_reports_fine_grained_kinds():
    """本地小模型产出的畸形参数五花八门，只报"参数错"等于没报。"""
    from onyx.core.errors import ToolArgError

    cases = [
        (TIME_NOW.parameters, {"fmt": "nope"}, "enum_violation"),
        (TIME_NOW.parameters, {"tz_offset_hours": True}, "type_mismatch"),
        (TIME_NOW.parameters, {"tz_offset_hours": None}, "null_not_allowed"),
        (TIME_NOW.parameters, {"fmt": "iso", "bogus": 1}, "unexpected_field"),
        (ECHO.parameters, {"text": "a", "bogus": 1}, "unexpected_field"),
        (ECHO.parameters, {}, "missing_required"),
        (ECHO.parameters, "not-an-object", "not_an_object"),
        (ECHO.parameters, {"text": 1}, "type_mismatch"),
    ]
    for schema, args, expected in cases:
        with pytest.raises(ToolArgError) as exc:
            validate_args(schema, args)
        assert exc.value.detail["kind"] == expected, f"{args} 应报 {expected}"

    assert validate_args(TIME_NOW.parameters, {"fmt": "iso"}) == {"fmt": "iso"}
    assert validate_args(None, {"a": 1}) == {"a": 1}
    assert validate_args({}, None) == {}


def test_diff_args_separates_missing_from_unexpected():
    report = diff_args({"city": "北京", "unit": "c"}, {"city": "上海", "extra": 1})
    assert report == {
        "ok": False, "missing": ["unit"], "unexpected": ["extra"],
        "mismatched": {"city": {"expected": "北京", "actual": "上海"}}, "subset_ok": False,
    }
    assert diff_args({"city": "北京"}, {"city": "北京", "extra": 1})["subset_ok"] is True


# ── 分发与内置定义 ────────────────────────────────────────────────
def test_executor_for_dispatches_and_refuses_unimplemented_kinds():
    assert isinstance(executor_for(ECHO), PythonFnExecutor)
    assert isinstance(executor_for(ECHO, kind="fixture"), MockReplayExecutor)
    assert EXECUTOR_KINDS == ("fixture", "http", "python_fn")
    for kind in ("mcp", "ollama_builtin"):
        definition = ToolDef(name="x", description="y" * 30, kind=ToolKind(kind))
        with pytest.raises(ToolUnknown) as exc:
            executor_for(definition)
        # 不许静默降级成 python_fn：那等于让模型输出决定进程加载什么
        assert "S16" in exc.value.message


@pytest.mark.parametrize("definition", BUILTIN_DEFS)
def test_builtin_definitions_are_audit_clean(definition):
    """内置定义同时是「工具该怎么写」的参照样本，所以必须零发现。

    用真实的 heuristic 计数器而不是 `len`：`len` 对中文 JSON 会把字符数当 token 数，
    凭空触发 DESCRIPTION_BUDGET，把"结构有没有缺陷"这个真正要测的东西盖掉。
    """
    findings = audit(definition, count_fn=estimate_tokens)
    assert findings == [], [(f.rule, f.message) for f in findings]


def test_builtin_definitions_stay_under_the_context_budget():
    """P17：这笔开销每次请求都要付，所以必须有个明确的上限测试盯着。"""
    from onyx.tools.spec import DESCRIPTION_TOKEN_BUDGET, openai_json

    for definition in BUILTIN_DEFS:
        tokens = estimate_tokens(openai_json(definition))
        assert tokens <= DESCRIPTION_TOKEN_BUDGET, (
            f"{definition.name} 注入 {tokens} token > 预算 {DESCRIPTION_TOKEN_BUDGET}"
        )


def test_untagged_side_effect_is_an_audit_error():
    [definition] = defs_from_payload([{"name": "t", "description": "x" * 30,
                                       "parameters": {"type": "object", "properties": {}}}])
    rules = {f.rule: f for f in audit(definition, count_fn=len)}
    assert "SIDE_EFFECT_UNTAGGED" in rules
    assert rules["SIDE_EFFECT_UNTAGGED"].severity is Severity.ERROR

    [tagged] = defs_from_payload([{"name": "t", "description": "x" * 30, "side_effect": "read",
                                   "parameters": {"type": "object", "properties": {}}}])
    assert "SIDE_EFFECT_UNTAGGED" not in {f.rule for f in audit(tagged, count_fn=len)}


def test_guarded_call_records_latency_and_size():
    result = PythonFnExecutor(ECHO).call("echo", {"text": "abc"}, _live_ctx())
    assert result.ok and result.bytes > 0 and not result.mocked
    assert result.extra["latency_ms"] >= 0
    assert result.status == "ok"


def test_time_now_is_flagged_non_deterministic():
    """它不能当契约 sample——read_is_idempotent 对它就是抛硬币。"""
    assert "non-deterministic" in TIME_NOW.tags
    assert "deterministic" in ECHO.tags and "non-deterministic" not in ECHO.tags


def test_idempotence_is_not_applicable_for_non_deterministic_samples():
    """两次调用落在同一秒就会"通过"——报一个靠运气的 ✓ 比报 n/a 危险得多。"""
    results = run_contracts(PythonFnExecutor, TIME_NOW, valid_args={"fmt": "unix"})
    item = next(r for r in results if r.name == "read_is_idempotent")
    assert not item.applicable and not item.passed
    assert "non-deterministic" in item.detail
    assert summarize(results)["failed"] == 0


def test_sample_args_prefers_the_first_matching_example():
    assert sample_args(ECHO) == {"text": "hello", "times": 1}
    assert sample_args(CALCULATOR) == {"expr": "(12+8)*3"}
    assert sample_args(TIME_NOW) == {"tz_offset_hours": 8, "fmt": "time"}


def test_sample_args_falls_back_to_schema_placeholders():
    """没有 examples 的工具本来就过不了 NO_EXAMPLE 审计，
    但契约测试不该因此完全跑不起来。"""
    bare = ToolDef(
        name="bare", description="没有 examples 的定义，用来验证占位参数构造",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名"},
                "limit": {"type": "integer", "description": "行数上限"},
                "unit": {"type": "string", "description": "单位", "enum": ["c", "f"]},
            },
            "required": ["city", "limit", "unit"],
        },
    )
    assert sample_args(bare) == {"city": "sample-city", "limit": 1, "unit": "c"}


def test_contract_runner_uses_the_same_pipeline_for_both_executors():
    """两个执行器都不许自己实现校验/沙箱——否则契约断言只在其中一个上成立。"""
    import inspect

    from onyx.tools.executors import mock_replay, python_fn

    for module in (python_fn, mock_replay):
        source = inspect.getsource(module)
        assert "guarded_call(" in source, f"{module.__name__} 绕过了统一管线"
        assert "validate_args(" not in source, f"{module.__name__} 自己实现了一遍参数校验"
        assert "enforce(" not in source, f"{module.__name__} 自己实现了一遍沙箱检查"


# ── 注入缺陷：契约运行器必须有牙 ──────────────────────────────────
class DeadlineIgnoringExecutor(PythonFnExecutor):
    """把 deadline 抹掉——最常见的"超时配置写了但没人看"缺陷。"""

    def call(self, name, args, ctx):
        return super().call(name, args, replace(ctx, deadline_ms=None))


class UnguardedExecutor(PythonFnExecutor):
    """绕过 guarded_call 直接返回成功——"每个执行器各自实现一遍"必然长这样。"""

    def call(self, name, args, ctx):
        from onyx.tools.spec import ToolResult

        return ToolResult(ok=True, output={"echo": args})


def test_contract_runner_detects_an_injected_timeout_defect():
    results = run_contracts(DeadlineIgnoringExecutor, SLOW, valid_args={"text": "x"})
    by_name = {item.name: item for item in results}
    assert not by_name["timeout_is_reported"].passed
    # 必须是**真失败**而不是被豁免：豁免会把缺陷伪装成"不适用"
    assert by_name["timeout_is_reported"].applicable
    assert summarize(results)["failed"] == 1


def test_contract_runner_detects_a_missing_guard_defect():
    """绕过统一管线的执行器会同时丢掉参数校验与沙箱——三条断言一起红。"""
    results = run_contracts(UnguardedExecutor, CONTRACT_SAMPLE, valid_args=SAMPLE_ARGS)
    failed = {item.name for item in results if item.applicable and not item.passed}
    assert {
        "missing_required_is_arg_error",
        "wrong_type_is_arg_error",
        "write_denied_without_approval",
    } <= failed, failed
    assert summarize(results)["failed"] >= 3
