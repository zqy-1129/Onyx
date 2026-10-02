"""L4 工具子系统：定义、注册、执行、契约测试、客户端循环。

三层刻意分开（DESIGN §8.1）：
1. **契约层**（无模型）：定义是否合法可诊断、schema 的上下文开销
2. **执行层**（无模型）：真实调用与边界（超时/非法参数/幂等/沙箱）
3. **模型层**（有模型）：指令 → 是否选对工具、参数是否正确

分开的价值：工具调用得分低时先跑 1/2 层，立刻判定是模型的问题还是工具的问题。
"""

from __future__ import annotations

from .spec import (
    SideEffect,
    ToolDef,
    ToolKind,
    ToolResult,
    audit,
    content_hash,
    to_openai_tool,
)

__all__ = [
    "SideEffect",
    "ToolDef",
    "ToolKind",
    "ToolResult",
    "audit",
    "content_hash",
    "to_openai_tool",
]
