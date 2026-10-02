"""Visitor 基类与注册。

`on()` 处理单个事件，`finalize()` 在 TRACE_END 时做跨事件推导。
**finalize 按注册顺序执行**，所以依赖别人产出的 visitor（如 anomaly 依赖 token 的对账结果）
必须排在后面——这个顺序是契约的一部分，见 `default_visitors()`。
"""

from __future__ import annotations

from onyx.core.event import TraceEvent
from onyx.obs.state import TraceState


class BaseVisitor:
    name: str = "base"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        """消费一个事件。默认不做事。"""

    def finalize(self, state: TraceState) -> None:
        """TRACE_END 时的跨事件推导。默认不做事。"""


def default_visitors() -> tuple[BaseVisitor, ...]:
    from onyx.obs.visitors.anomaly import AnomalyVisitor
    from onyx.obs.visitors.cost import CostVisitor
    from onyx.obs.visitors.gpu import GpuVisitor
    from onyx.obs.visitors.token import TokenVisitor
    from onyx.obs.visitors.tool import ToolVisitor

    # 顺序即依赖：token 先对账 → cost 用采信值算成本 → anomaly 最后做跨切面判定
    return (TokenVisitor(), ToolVisitor(), GpuVisitor(), CostVisitor(), AnomalyVisitor())


__all__ = ["BaseVisitor", "default_visitors"]
