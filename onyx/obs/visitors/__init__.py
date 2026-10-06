"""Visitor 基类与注册。

`on()` 处理单个事件，`finalize()` 在 TRACE_END 时做跨事件推导。
**finalize 按注册顺序执行**，所以依赖别人产出的 visitor（如 anomaly 依赖 token 的对账结果）
必须排在后面——这个顺序是契约的一部分，见 `default_visitors()`。

外部观测器走 entry points `onyx.observers`，值可以是 `BaseVisitor` 子类或实例。
插件 visitor **一律排在内置之后**：允许插队到 anomaly 前面，就等于让
"顺序即契约"变成运行期才决定的事，而它是静态可检查的这一条属性正是它的价值。
"""

from __future__ import annotations

from onyx.core.event import TraceEvent
from onyx.discovery import GROUP_OBSERVERS, discover, record_failure
from onyx.obs.state import TraceState


class BaseVisitor:
    name: str = "base"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        """消费一个事件。默认不做事。"""

    def finalize(self, state: TraceState) -> None:
        """TRACE_END 时的跨事件推导。默认不做事。"""


def builtin_visitors() -> tuple[BaseVisitor, ...]:
    """内建观测器。**顺序即依赖**：token 先对账 → cost 用采信值算成本 →
    anomaly 最后做跨切面判定。

    `timing` 排在 cost 之前：它不读任何人的产出（只看 `FIRST_TOKEN` 事件），
    而 `latency_summary()` 在 TRACE_END 时会读它写的 `ttft_ms`——
    把不依赖人的放前面，将来真有依赖时也不用回头改顺序。
    """
    from onyx.obs.visitors.anomaly import AnomalyVisitor
    from onyx.obs.visitors.cost import CostVisitor
    from onyx.obs.visitors.gpu import GpuVisitor
    from onyx.obs.visitors.timing import TimingVisitor
    from onyx.obs.visitors.token import TokenVisitor
    from onyx.obs.visitors.tool import ToolVisitor

    return (TokenVisitor(), ToolVisitor(), GpuVisitor(), TimingVisitor(),
            CostVisitor(), AnomalyVisitor())


def plugin_visitors() -> tuple[BaseVisitor, ...]:
    """`onyx.observers` 插件实例。构造失败或形状不对的跳过并记账。

    名字必须唯一：`ObserverEngine.observer_errors` 按 `visitor.name` 归因，
    两个同名 visitor 会把错误挤进同一个键，而"谁的计数"正是需要知道的那件事。
    """
    taken = {v.name for v in builtin_visitors()}
    out: list[BaseVisitor] = []
    for name, obj in discover(GROUP_OBSERVERS, {}).items():
        try:
            visitor = obj() if isinstance(obj, type) else obj
        except Exception as exc:  # noqa: BLE001 - 坏插件隔离
            record_failure(GROUP_OBSERVERS, name, f"构造失败 {type(exc).__name__}: {exc}")
            continue
        if not (callable(getattr(visitor, "on", None))
                and callable(getattr(visitor, "finalize", None))):
            record_failure(
                GROUP_OBSERVERS, name,
                "不是 EventVisitor：需要 on(event, state) 与 finalize(state) 两个方法",
            )
            continue
        current = str(getattr(visitor, "name", "") or "")
        if not current or current == "base" or current in taken:
            try:
                visitor.name = f"plugin:{name}"  # type: ignore[attr-defined]
                current = f"plugin:{name}"
            except (AttributeError, TypeError):
                current = current or f"plugin:{name}"
        taken.add(current)
        out.append(visitor)
    return tuple(out)


def default_visitors() -> tuple[BaseVisitor, ...]:
    return (*builtin_visitors(), *plugin_visitors())


__all__ = ["BaseVisitor", "builtin_visitors", "default_visitors", "plugin_visitors"]
