"""L3 探针：把"引擎到底怎么算的"变成可复现的实验结论。

为什么探针是一等公民而不是调试脚本：token 口径、缓存语义、thinking 计数这些东西
**猜错的代价是看板长期显示错误数字且没人发现**。所以每个结论都必须带
证据（trace 级原始值）、引擎版本、模型名与日期，并且能被重跑。
"""

from __future__ import annotations

from . import checks as _checks  # noqa: F401 - 导入即注册探针
from .runner import ProbeContext, ProbeReport, ProbeSuite, registered_probes, render_markdown, run_suite

__all__ = [
    "ProbeContext",
    "ProbeReport",
    "ProbeSuite",
    "registered_probes",
    "render_markdown",
    "run_suite",
]
