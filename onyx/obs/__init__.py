"""L3 观测层：把 gateway 产出的事件流折叠成可查询的记录。

分层职责（DESIGN §10）：
- gateway **只**产出事件，不知道 SQLite 长什么样；
- visitor **只**消费事件、产出记录，不发请求；
- engine 负责状态累加、异常隔离与落盘。

加一个新指标 = 加一个 visitor，不改 gateway。
"""

from __future__ import annotations

from .engine import ObserverEngine
from .state import ToolCallDraft, TraceState
from .visitors import BaseVisitor, default_visitors

__all__ = [
    "BaseVisitor",
    "ObserverEngine",
    "ToolCallDraft",
    "TraceState",
    "default_visitors",
]
