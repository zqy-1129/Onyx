"""L6 API 层：只读优先的 REST + SSE。

约定：
- **写操作只有 admin 与 playground**，且 admin 需 `confirm=1`（误删模型的代价太高）；
- 所有数字都带 `source` / `confidence`，前端必须渲染出处徽标——没有出处的数字等于撒谎；
- 冷/热分列，不提供合并视图（PROBES P11）。
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
