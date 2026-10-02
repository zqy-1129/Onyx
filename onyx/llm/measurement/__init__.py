"""L1 计量：token 保真阶梯、分段归因、多来源对账、模型标定。

所有数字都必须带 `source` 与 `confidence`（DESIGN 原则 4）。
本包**不做 IO**：它只消费 `GenerationRequest` 与 `Generation`，因此可完全离线单测。
"""

from __future__ import annotations

from .calibrate import CalibrationResult, CalibrationSample, fit_ratio
from .fidelity import (
    CompatCounter,
    Counter,
    CounterContext,
    EngineCounter,
    FittedCounter,
    HeuristicCounter,
    default_counters,
    text_counter,
)
from .heuristic import estimate_tokens
from .parts import AttributionReport, Segment, attribute, input_segments, part_totals
from .reconciler import ReconcileResult, latency_summary, prefill_mode, reconcile

__all__ = [
    "AttributionReport",
    "CalibrationResult",
    "CalibrationSample",
    "CompatCounter",
    "Counter",
    "CounterContext",
    "EngineCounter",
    "FittedCounter",
    "HeuristicCounter",
    "ReconcileResult",
    "Segment",
    "attribute",
    "default_counters",
    "estimate_tokens",
    "fit_ratio",
    "input_segments",
    "latency_summary",
    "part_totals",
    "prefill_mode",
    "reconcile",
    "text_counter",
]
