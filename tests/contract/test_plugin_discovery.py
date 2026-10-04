"""扩展点发现契约：五个已接线的注册表 + 失败可见性。

这里测的不是"能不能 import 到一个类"，而是 DESIGN §13 那三条语义在**每个 group
上是否一致**：坏插件隔离、失败必须可见、同名覆盖（fixture 通道除外——
它是评测零副作用的保证，插件抢不走）。

外部包用伪造的 entry point 指进真实源码（`plugins_example/` 已挂在 sys.path 上），
所以这些断言在没有网络、没有额外安装步骤的 CI 里也成立。
"""

from __future__ import annotations

import pytest
from example_provider.provider import EchoProvider
from example_task.task import CharCount

from onyx.core.types import Cap
from onyx.discovery import (
    GROUP_OBSERVERS,
    GROUP_PROVIDERS,
    GROUP_SINKS,
    GROUP_TASKS,
    GROUP_TOOL_EXECUTORS,
    GROUPS,
    clear_cache,
    failures,
    plugin_names,
    report,
    reset_failures,
)
from onyx.eval.tasks import BUILTIN_TASKS, build_task, specs, task_ids
from onyx.llm.registry import available_kinds, build_provider
from onyx.obs.engine import ObserverEngine
from onyx.obs.visitors import builtin_visitors, default_visitors
from onyx.store.sinks import build_event_sink, sink_names
from onyx.tools.executor import ToolCtx
from onyx.tools.executors import EXECUTOR_KINDS, executor_for, executor_kinds
from onyx.tools.spec import SideEffect, ToolDef, ToolKind

# ── 供伪造 entry point 指向的"外部实现"（就在本测试模块里，模拟第三方包）───


class MemoryEventSink:
    """`onyx.sinks` 插件形状：`builder(**options) -> EventSink`。"""

    name = "memory"

    def __init__(self) -> None:
        self.events: list[object] = []
        self.flushed = 0
        self.closed = 0

    def emit(self, event) -> None:
        self.events.append(event)

    def flush(self, timeout: float = 1.0) -> None:
        self.flushed += 1

    def close(self) -> None:
        self.closed += 1


def _memory_sink(**_options: object) -> MemoryEventSink:
    return MemoryEventSink()


class ShoutExecutor:
    """`onyx.tool_executors` 插件形状：`builder(definition) -> ToolExecutor`。"""

    kind = "shout"

    def __init__(self, definition: ToolDef) -> None:
        self.definition = definition

    def spec(self) -> ToolDef:
        return self.definition

    def call(self, name: str, args: dict[str, object], ctx: ToolCtx) -> object:
        from onyx.tools.spec import ToolResult

        return ToolResult(ok=True, output=str(args.get("text", "")).upper())


class HijackExecutor(ShoutExecutor):
    """故意注册成 `fixture`：它必须抢不走评测的零副作用通道。"""

    kind = "fixture"


class CountingVisitor:
    name = "counting"

    def __init__(self) -> None:
        self.seen = 0

    def on(self, event, state) -> None:
        self.seen += 1

    def finalize(self, state) -> None: ...


class AnonymousVisitor:
    """没有 name：装配时必须补成 `plugin:<注册名>`，否则报错无法归因。"""

    def on(self, event, state) -> None: ...

    def finalize(self, state) -> None: ...


class NotAVisitor:
    def nope(self) -> None: ...


class ThrowingVisitor:
    """模块级定义：entry point 按 `module:attr` 解析，函数内的类是取不到的。"""

    name = "throwing"

    def on(self, event, state) -> None:
        raise RuntimeError("插件炸了")

    def finalize(self, state) -> None: ...


def _echo_builder(**kwargs: object) -> EchoProvider:
    """provider 注册表要的是 `(**kwargs) -> LlmProvider`，类本身就行；
    这里包一层顺便吸收内核传的 base_url 等参数。"""
    return EchoProvider(**{k: v for k, v in kwargs.items() if k in {"id", "base_url"}})


# ── 发现语义 ───────────────────────────────────────────────────────
def test_all_six_groups_are_named_in_the_public_contract():
    assert set(GROUPS) == {
        "onyx.providers", "onyx.tasks", "onyx.graders",
        "onyx.sinks", "onyx.tool_executors", "onyx.observers",
    }


def test_bad_plugin_is_isolated_but_stays_visible(entry_points_env):
    """一个坏插件不该让内置实现消失，但它坏掉这件事必须能被查到。"""
    entry_points_env.install(
        GROUP_TASKS, "broken=no_such_module_anywhere:Thing",
        "example_char_count=example_task.task:spec",
    )
    ids = task_ids()
    assert "intent_classification" in ids and "example_char_count" in ids
    assert "broken" not in ids

    recorded = failures()
    assert [f.name for f in recorded] == ["broken"]
    assert "ModuleNotFoundError" in recorded[0].error
    # 同一个坏插件被反复问到时，attempts 累加——"坏了三次"和"坏了一次"都要能看见
    task_ids()
    assert failures()[0].attempts == 2


def test_wrong_shape_plugin_reports_what_it_expected(entry_points_env):
    """加载成功但形状不对：报错必须说清期望，而不是消失在日志里。"""
    entry_points_env.install(GROUP_TASKS, "oops=example_task.task:_han")
    assert "oops" not in task_ids()
    error = failures()[0].error
    assert "EvalTask" in error and "TaskSpec" in error


def test_plugin_overrides_builtin_and_report_shows_it(entry_points_env):
    entry_points_env.install(GROUP_TASKS, "intent_classification=example_task.task:CharCount")
    assert specs()["intent_classification"].factory is CharCount

    rep = report(GROUP_TASKS, BUILTIN_TASKS)
    assert rep.overrides == ("intent_classification",)
    assert "intent_classification" in rep.builtin and "intent_classification" in rep.plugin


def test_plugin_names_and_entries_do_not_load_modules(entry_points_env):
    entry_points_env.install(GROUP_PROVIDERS, "echo=example_provider.provider:EchoProvider")
    assert plugin_names(GROUP_PROVIDERS) == ("echo",)
    rep = report(GROUP_PROVIDERS, {"ollama", "mock"})
    assert rep.plugin == ("echo",)
    # report 只做声明层面的对照：诊断命令不该为了"列一下"触发所有插件 import
    assert failures() == ()


def test_discovery_cache_is_explicit_and_clearable(entry_points_env):
    entry_points_env.install(GROUP_SINKS, "mem=test_plugin_discovery:_memory_sink")
    assert "mem" in sink_names()
    entry_points_env.table[GROUP_SINKS] = ()  # 直接改表，绕过 install() 的 clear
    assert "mem" in sink_names(), "缓存生效：装好的插件不会在一次进程里凭空消失"
    clear_cache()
    assert "mem" not in sink_names()


def test_reset_failures_clears_the_ledger(entry_points_env):
    entry_points_env.install(GROUP_SINKS, "broken=no_such_module:Thing")
    sink_names()
    assert failures()
    reset_failures()
    assert failures() == ()


# ── providers ──────────────────────────────────────────────────────
def test_external_provider_is_buildable_through_the_registry(entry_points_env):
    entry_points_env.install(GROUP_PROVIDERS, "echo=example_provider.provider:EchoProvider")
    assert "echo" in available_kinds()
    provider = build_provider("echo", id="echo-1")
    assert provider.id == "echo-1"
    assert provider.capabilities() == frozenset({Cap.CHAT})
    assert provider.list_models()[0].name in [c.name for c in provider.list_models()]


def test_registry_builder_kwargs_are_forwarded(entry_points_env):
    entry_points_env.install(GROUP_PROVIDERS, "echo=test_plugin_discovery:_echo_builder")
    provider = build_provider("echo", id="e2", base_url="echo://x")
    assert isinstance(provider, EchoProvider)
    assert (provider.id, provider.base_url) == ("e2", "echo://x")


def test_unknown_provider_kind_lists_options_including_plugins(entry_points_env):
    entry_points_env.install(GROUP_PROVIDERS, "echo=example_provider.provider:EchoProvider")
    with pytest.raises(KeyError) as exc:
        build_provider("nope")
    assert "echo" in str(exc.value) and "mock" in str(exc.value)


# ── tasks：外部任务真的能构造出来 ──────────────────────────────────
def test_external_task_loads_and_grades(entry_points_env):
    entry_points_env.install(GROUP_TASKS, "example_char_count=example_task.task:spec")
    task = build_task("example_char_count", model="mock/x")
    assert isinstance(task, CharCount)

    cases = list(task.load())
    assert len(cases) == 8
    assert cases[0].dataset_id == "example_char_count-v1"

    request = task.build(cases[0])
    assert request.model == "mock/x" and request.thinking is False

    from onyx.core.types import Generation

    grade = task.grade(cases[0], Generation(text="6", model="mock/x"))
    assert grade.verdict.value == "correct" and not grade.invalid_format

    dirty = task.grade(cases[0], Generation(text="答案是 6 个", model="mock/x"))
    # 内容对、格式不听：两个维度必须分开（DESIGN §9.4）
    assert dirty.verdict.value == "correct" and dirty.invalid_format is True

    aggregate = task.aggregate([grade, dirty])
    assert set(aggregate) == set(task.metric_names), "声明的指标与产出的指标同源"


def test_external_task_without_default_dataset_asks_for_one(entry_points_env):
    """只交任务类（没有 TaskSpec.dataset）→ 必须显式 --dataset，不许猜一份数据。"""
    entry_points_env.install(GROUP_TASKS, "example_char_count=example_task.task:CharCount")
    with pytest.raises(KeyError) as exc:
        build_task("example_char_count", model="mock/x")
    assert "--dataset" in str(exc.value)


def test_registration_name_must_equal_task_id(entry_points_env):
    """注册名与 `EvalTask.id` 不一致时，库里记的和用户查的是两个任务。

    这条裂缝一旦发生就没人会看见：分数照样产出，只是落在另一个任务名下。
    """
    entry_points_env.install(GROUP_TASKS, "renamed=example_task.task:spec")
    with pytest.raises(KeyError) as exc:
        build_task("renamed", model="mock/x")
    assert "example_char_count" in str(exc.value)


# ── sinks ──────────────────────────────────────────────────────────
def test_external_sink_is_built_by_name(entry_points_env):
    entry_points_env.install(GROUP_SINKS, "mem=test_plugin_discovery:_memory_sink")
    sink = build_event_sink("mem")
    assert sink.name == "memory"
    sink.emit(object())
    assert len(sink.events) == 1


def test_builtin_jsonl_sink_still_works(entry_points_env, tmp_path):
    sink = build_event_sink("jsonl", data_dir=tmp_path)
    assert sink.path == tmp_path / "events.ndjson"
    with pytest.raises(KeyError) as exc:
        build_event_sink("nope")
    assert "jsonl" in str(exc.value)


def test_missing_sink_points_at_the_broken_plugin(entry_points_env):
    """`--sink langfuse` 而插件坏了：报错要指到坏插件，而不是只说"未知 sink"。

    用 `langfuse` 而不是 `otlp` 举例是有意的：坏插件与内建同名时会**退回内建**
    （内建仍可用），那种情况下"未知 sink"反而是正确回答。
    """
    entry_points_env.install(GROUP_SINKS, "langfuse=no_such_pkg:X")
    with pytest.raises(KeyError) as exc:
        build_event_sink("langfuse")
    assert "langfuse" in str(exc.value) and "加载失败" in str(exc.value)


def test_broken_plugin_shares_name_with_builtin_falls_back_to_builtin(entry_points_env):
    """同名坏插件：跳过它、内建照常工作，而失败记录仍然可见。

    坏插件不该把内建一起拖死；但它坏过这件事不许消失（`failures()`）。
    """
    entry_points_env.install(GROUP_SINKS, "otlp=no_such_pkg:X")
    sink = build_event_sink("otlp", endpoint="http://collector.test:4318")
    assert sink.name == "otlp"
    assert [f.name for f in failures() if f.group == GROUP_SINKS] == ["otlp"]


# ── tool executors ─────────────────────────────────────────────────
def _def(kind: str) -> ToolDef:
    return ToolDef(
        name="shout", description="把文本变大写", kind=kind,
        side_effect=SideEffect.READ, parameters={"type": "object", "properties": {}},
    )


def test_plugin_kind_is_representable_without_touching_the_enum(entry_points_env):
    """`ToolKind` 是内建集合，枚举不能长出新成员。

    如果定义层不允许 "shout" 这种外部种类，`onyx.tool_executors` 就只是
    "覆盖内建种类"的口子，而不是扩展点——而且加一个执行器就得改内核。
    """
    assert _def("shout").kind == "shout"
    assert _def("python_fn").kind is ToolKind.PYTHON_FN, "内建种类仍归一成枚举，口径不变"
    with pytest.raises(ValueError, match="slug"):
        _def("Bad Kind")


def test_external_executor_kind_is_dispatched(entry_points_env):
    entry_points_env.install(GROUP_TOOL_EXECUTORS, "shout=test_plugin_discovery:ShoutExecutor")
    assert "shout" in executor_kinds()
    assert "shout" not in EXECUTOR_KINDS, "EXECUTOR_KINDS 是内建契约，不随插件变"

    executor = executor_for(_def("shout"))
    assert executor.kind == "shout"
    result = executor.call("shout", {"text": "hi"}, ToolCtx())
    assert result.ok and result.output == "HI"


def test_plugins_cannot_hijack_the_fixture_channel(entry_points_env):
    """评测的零副作用保证：`kind="fixture"` 永远走桩，插件注册同名也不能抢。

    否则一个外部执行器就能在评测期真的发请求/写文件，而 case 里写着 fixture 返回值，
    分数看起来一切正常。
    """
    entry_points_env.install(GROUP_TOOL_EXECUTORS, "fixture=test_plugin_discovery:HijackExecutor")
    executor = executor_for(_def("python_fn"), kind="fixture", responses={}, default=None)
    assert type(executor).__name__ == "MockReplayExecutor"


def test_import_boundary_accepts_a_kind_only_the_plugin_knows(entry_points_env):
    """定义导入处也必须认插件种类，否则 `onyx.tool_executors` 只能覆盖内建种类。

    反过来，没装插件时导入必须失败——拼错的 kind 不该躺进库里等某次调用才炸。
    """
    from onyx.tools.registry import defs_from_payload

    payload = [{
        "name": "shout", "description": "把文本变大写", "side_effect": "read",
        "kind": "shout", "parameters": {"type": "object", "properties": {}},
    }]
    with pytest.raises(ValueError, match="kind 非法"):
        defs_from_payload(payload)

    entry_points_env.install(GROUP_TOOL_EXECUTORS, "shout=test_plugin_discovery:ShoutExecutor")
    defs = defs_from_payload(payload)
    assert str(defs[0].kind) == "shout"


# ── observers ──────────────────────────────────────────────────────
def test_external_visitor_is_appended_after_builtins(entry_points_env):
    entry_points_env.install(
        GROUP_OBSERVERS, "counting=test_plugin_discovery:CountingVisitor"
    )
    visitors = default_visitors()
    assert [v.name for v in visitors][-1] == "counting"
    assert [v.name for v in visitors][: len(builtin_visitors())] == [
        v.name for v in builtin_visitors()
    ], "内置顺序是契约，插件不许插队"


def test_anonymous_plugin_visitor_gets_a_attributable_name(entry_points_env):
    entry_points_env.install(GROUP_OBSERVERS, "quiet=test_plugin_discovery:AnonymousVisitor")
    visitor = default_visitors()[-1]
    assert visitor.name == "plugin:quiet"


def test_duplicate_visitor_names_are_split_up(entry_points_env):
    """两个插件都用 `name="dup"` 时，`observer_errors` 会把它们的错误挤进同一个键。

    第二个改名为 `plugin:<注册名>`：注册名是唯一的，归因也就必须唯一。
    """
    entry_points_env.install(
        GROUP_OBSERVERS,
        "first=test_plugin_discovery:CountingVisitor",
        "second=test_plugin_discovery:CountingVisitor",
    )
    names = [v.name for v in default_visitors()]
    assert names[-2:] == ["counting", "plugin:second"]


def test_broken_visitor_does_not_disable_builtins(entry_points_env):
    entry_points_env.install(GROUP_OBSERVERS, "junk=test_plugin_discovery:NotAVisitor")
    visitors = default_visitors()
    assert len(visitors) == len(builtin_visitors())
    assert "EventVisitor" in failures()[0].error


def test_observer_engine_survives_a_plugin_visitor_that_throws(entry_points_env):
    """插件 visitor 抛错必须被引擎隔离，并且记进 observer_errors（可见，不是静默）。"""
    from onyx.core.event import EventType, make_event
    from onyx.store.sinks import NullRecordSink

    entry_points_env.install(GROUP_OBSERVERS, "boom=test_plugin_discovery:ThrowingVisitor")
    engine = ObserverEngine(record_sink=NullRecordSink())
    engine.handle(make_event(
        EventType.TRACE_START, "t-1",
        {"kind": "chat", "purpose": "ops", "provider_id": "p", "model": "m"},
    ))
    engine.handle(make_event(EventType.USAGE_ENGINE, "t-1"))
    engine.handle(make_event(EventType.TRACE_END, "t-1", {"status": "ok", "wall_ms": 1.0}))
    assert engine.observer_errors.get("throwing")
