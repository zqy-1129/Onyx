"""tool visitor：工具调用归因。

核心价值是**把失败分成不同种类**——"模型不会调"（没发起调用）、"调错工具"、
"参数格式坏"（TRUNCATED vs MALFORMED，修法完全不同）、"工具本身坏了"（result_status）。
混成一个"失败率"就没法行动了。
"""

from __future__ import annotations

from onyx.core.event import EventType, TraceEvent
from onyx.core.types import tool_call_fingerprint
from onyx.obs.state import ToolCallDraft, TraceState
from onyx.obs.visitors import BaseVisitor


class ToolVisitor(BaseVisitor):
    name = "tool"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        if event.type is EventType.GENERATION_END:
            self._absorb_calls(event, state)
        elif event.type is EventType.TOOL_EXEC_START:
            self._attach(event, state, started=True)
        elif event.type is EventType.TOOL_EXEC_END:
            self._attach(event, state, started=False)

    def _absorb_calls(self, event: TraceEvent, state: TraceState) -> None:
        for index, raw in enumerate(event.payload.get("tool_calls") or []):
            state.tool_calls.append(ToolCallDraft(
                step=int(raw.get("step", index + 1)),
                name=str(raw.get("name") or ""),
                call_id=str(raw.get("call_id") or ""),
                args=raw.get("args") if isinstance(raw.get("args"), dict) else None,
                args_raw=str(raw.get("args_raw") or ""),
                parse_status=str(raw.get("parse_status") or "ok"),
                parse_source=str(raw.get("parse_source") or ""),
                extra={"unknown_tool": bool(raw.get("unknown_tool"))},
            ))

    def _attach(self, event: TraceEvent, state: TraceState, *, started: bool) -> None:
        payload = event.payload
        step = payload.get("step")
        name = payload.get("name")
        draft = next(
            (d for d in reversed(state.tool_calls)
             if (step is None or d.step == step) and (name is None or d.name == name)),
            None,
        )
        if draft is None:
            state.add_anomaly("ORPHAN_TOOL_CALL", {
                "reason": "收到执行事件但没有对应的调用记录", "name": name, "step": step,
            })
            return
        if started:
            draft.started_at = event.wall_iso
            draft.executed_by = str(payload.get("executed_by") or "client")
            draft.tool_id = payload.get("tool_id")
            draft.tool_def_hash = payload.get("tool_def_hash")
        else:
            draft.result_status = str(payload.get("status") or "")
            draft.latency_ms = payload.get("latency_ms")
            draft.result_ref = payload.get("result_ref")
            draft.result_bytes = payload.get("result_bytes")
            draft.executed_by = draft.executed_by or str(payload.get("executed_by") or "")
            if payload.get("error"):
                draft.extra["error"] = str(payload["error"])[:500]

    def finalize(self, state: TraceState) -> None:
        seen: dict[str, int] = {}
        for draft in state.tool_calls:
            if draft.parse_status == "truncated":
                state.add_anomaly("TRUNCATED_TOOL_JSON", {
                    "tool": draft.name, "args_raw": draft.args_raw[:200],
                    "hint": "多为 max_tokens 不足，属预算问题而非能力问题",
                })
            elif draft.parse_status in {"json_error", "not_parsable"}:
                state.add_anomaly("MALFORMED_TOOL_JSON", {
                    "tool": draft.name, "args_raw": draft.args_raw[:200],
                })
            elif draft.parse_status == "unknown_tool" or draft.extra.get("unknown_tool"):
                state.add_anomaly("UNKNOWN_TOOL", {"tool": draft.name})
            if draft.result_status in {"error", "timeout"}:
                state.add_anomaly("TOOL_ERROR", {
                    "tool": draft.name, "status": draft.result_status,
                    "error": draft.extra.get("error", ""),
                })
            fingerprint = tool_call_fingerprint(draft.name, draft.args, draft.args_raw)
            seen[fingerprint] = seen.get(fingerprint, 0) + 1

        repeated = {k: v for k, v in seen.items() if v > 1}
        if repeated:
            state.add_anomaly("TOOL_LOOP", {"repeated_calls": repeated})

        wants_tool = state.finish_reason == "tool_calls" or bool(state.tool_calls)
        if state.finish_reason == "tool_calls" and not state.tool_calls:
            state.add_anomaly("ORPHAN_TOOL_CALL", {
                "reason": "finish_reason=tool_calls 但没有可执行的调用",
                "hint": "必须补占位 tool 消息，否则后续上下文永久错位",
            })
        if wants_tool and state.tool_calls and not any(d.result_status for d in state.tool_calls):
            # 只生成不执行（例如评测的 fixture 模式或纯观测），记录事实而不是报警
            state.extra["tool_execution"] = "not_executed"
