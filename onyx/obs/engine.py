"""观测引擎：事件 → 状态 → 记录。

三条硬规则：
1. **异常隔离**：任何 visitor 抛错都被吞掉并记 `OBSERVER_ERROR`，绝不影响请求主链路。
2. **顺序即契约**：`finalize` 按注册顺序执行，依赖别人产出的 visitor 必须排后面。
3. **有界状态**：TRACE_END 缺失（进程崩溃/取消）时按容量淘汰最老状态，
   宁可丢一条观测也不让看板进程 OOM。
"""

from __future__ import annotations

import logging
from collections import OrderedDict

from onyx.core.event import EventType, TraceEvent
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor, default_visitors
from onyx.store.sinks import NullRecordSink, RecordSink

log = logging.getLogger("onyx.obs")

DEFAULT_MAX_STATES = 4096


class ObserverEngine:
    def __init__(
        self,
        *,
        record_sink: RecordSink | None = None,
        visitors: tuple[BaseVisitor, ...] | None = None,
        max_states: int = DEFAULT_MAX_STATES,
    ) -> None:
        self.visitors: tuple[BaseVisitor, ...] = visitors if visitors is not None else default_visitors()
        self.sink: RecordSink = record_sink or NullRecordSink()
        self._states: OrderedDict[str, TraceState] = OrderedDict()
        self._max_states = max_states
        self.events = 0
        self.traces = 0
        self.observer_errors: dict[str, int] = {}
        self.evicted = 0

    # ── 事件入口 ──────────────────────────────────────────────────
    def handle(self, event: TraceEvent) -> TraceState | None:
        """消费一个事件。若该事件结束了 trace，返回已落盘的最终状态，否则返回 None。"""
        self.events += 1
        state = self._state_for(event)
        if event.type is EventType.TRACE_START:
            self._apply_trace_start(event, state)
        for visitor in self.visitors:
            try:
                visitor.on(event, state)
            except Exception as exc:  # noqa: BLE001 - 规则 1：观测失败不许影响请求
                self._isolate(visitor, state, exc)
        if event.type is EventType.TRACE_END:
            self._apply_trace_end(event, state)
            return self.finish(state.trace_id)
        return None

    def finish(self, trace_id: str) -> TraceState | None:
        state = self._states.pop(trace_id, None)
        if state is None:
            return None
        for visitor in self.visitors:
            try:
                visitor.finalize(state)
            except Exception as exc:  # noqa: BLE001
                self._isolate(visitor, state, exc)
        self._write(state)
        self.traces += 1
        return state

    # ── 内部 ──────────────────────────────────────────────────────
    def _state_for(self, event: TraceEvent) -> TraceState:
        state = self._states.get(event.trace_id)
        if state is None:
            state = TraceState(trace_id=event.trace_id, started_at=event.wall_iso)
            self._states[event.trace_id] = state
            self._evict_if_needed()
        return state

    def _evict_if_needed(self) -> None:
        while len(self._states) > self._max_states:
            stale_id, stale = self._states.popitem(last=False)
            self.evicted += 1
            log.warning("观测状态超限，淘汰最老的 trace: %s", stale_id)
            stale.add_anomaly("OBSERVER_ERROR", {"reason": "state evicted before TRACE_END"})
            self._write(stale)

    def _isolate(self, visitor: BaseVisitor, state: TraceState, exc: Exception) -> None:
        self.observer_errors[visitor.name] = self.observer_errors.get(visitor.name, 0) + 1
        state.observer_errors += 1
        state.add_anomaly("OBSERVER_ERROR", {
            "visitor": visitor.name, "error": f"{type(exc).__name__}: {exc}"[:300],
        })
        log.warning("visitor %s 失败（已隔离）: %s", visitor.name, exc, exc_info=True)

    def _apply_trace_start(self, event: TraceEvent, state: TraceState) -> None:
        payload = event.payload
        state.started_at = event.wall_iso
        state.kind = str(payload.get("kind") or state.kind)
        state.purpose = str(payload.get("purpose") or state.purpose)
        state.provider_id = payload.get("provider_id")
        state.model_name = payload.get("model")
        state.model_id = payload.get("model_id")
        state.params = dict(payload.get("params") or {})
        state.messages_ref = payload.get("messages_ref")
        state.tools_ref = payload.get("tools_ref")
        state.raw_request_ref = payload.get("raw_request_ref")
        state.rendered_prompt_ref = payload.get("rendered_prompt_ref")
        state.keep_alive = payload.get("keep_alive")
        state.contract_version = int(payload.get("contract_version") or state.contract_version)
        context = payload.get("context") or {}
        state.eval_run_id = context.get("eval_run_id")
        state.case_id = context.get("case_id")
        state.sample_seq = context.get("sample_seq")
        state.parent_id = context.get("parent_trace_id") or payload.get("trace_parent")
        state.root_id = context.get("root_trace_id")

    def _apply_trace_end(self, event: TraceEvent, state: TraceState) -> None:
        payload = event.payload
        state.finished_at = event.wall_iso
        state.status = str(payload.get("status") or "ok")
        state.wall_ms = payload.get("wall_ms")
        state.error = str(payload.get("error") or "")
        state.finish_reason = payload.get("finish_reason")
        state.output_ref = payload.get("output_ref")
        state.raw_response_ref = payload.get("raw_response_ref")
        state.text_chars = int(payload.get("text_chars") or 0)
        state.thinking_chars = int(payload.get("thinking_chars") or 0)
        if payload.get("gpu"):
            state.gpu = {**state.gpu, **payload["gpu"]}

    def _write(self, state: TraceState) -> None:
        sink = self.sink
        sink.write_trace(state.to_trace_record())
        usage = state.to_usage_record()
        if usage is not None:
            sink.write_usage(usage, alts=state.to_alt_records(), parts=state.to_part_records())
        for record in state.to_tool_call_records():
            sink.write_tool_call(record)
        for record in state.to_anomaly_records():
            sink.write_anomaly(record)
        sink.flush()

    def stats(self) -> dict[str, object]:
        return {
            "events": self.events,
            "traces": self.traces,
            "pending_states": len(self._states),
            "evicted": self.evicted,
            "observer_errors": dict(self.observer_errors),
        }
