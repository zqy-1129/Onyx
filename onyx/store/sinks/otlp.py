"""OTLP/HTTP 导出 sink（`onyx.sinks` 的第二个真实实现）。

它存在的两个理由：
1. **验证 Sink 抽象**：如果"换一条出口就要改内核"，那 `EventSink` 就是假的。
   这个实现只碰 `emit/flush/close` 三个方法，装配靠 `onyx.sinks` 注册表 +
   `--sink otlp`，`onyx/core/**`、`gateway.py`、`obs/**` 一行没动
   （由 `scripts/check_extension_boundary.py` 判定）。
2. 给"已经有一整套可观测栈"的人一条出路：Onyx 的 trace 可以被 Jaeger /
   Grafana Tempo / 任意 OTLP collector 收走，而 `onyx.trace_id` 作为属性保留，
   两边可以互相跳转。

三条刻意的取舍，都写在这里而不是藏在代码里：

- **编码用 JSON 而不是 protobuf**（OTLP 的 HTTP 编码器允许
  `Content-Type: application/json`）。代价是不能声称"任何 collector 都能吃"：
  少数只实现了 protobuf 编码的接收端会拒绝。所以把 `onyx.encoding=json` 导成资源属性，
  接收端看到就知道自己面对的是什么。
- **一个 trace 只导出结构 span**（根 span + 每次工具执行一个子 span），
  逐 token 的增量折成计数属性。OTLP 的 span 语义是"有始有终的动作"，
  把 3000 个 `TEXT_DELTA` 变成 3000 个 span 会把 collector 打满，
  而且那等于在 collector 里重建第二个 Onyx。
- **默认不导出工具参数值**（只导出键名与数量）。参数里经常有地址、身份、内部 ID；
  把它们原样发到外部可观测栈不是"导出 trace"，是数据出境。
  确实需要时显式 `include_args=True`。

时间戳用 `wall_iso` 换算的 unix epoch 纳秒，**不用 `ts_ns`**：后者是单调钟，
换台机器就没有意义，导出成一个 1970 年附近的数比报错更难发现。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import UTC, datetime
from typing import Any

import httpx

from onyx import __version__
from onyx.core.event import EventType, TraceEvent

DEFAULT_ENDPOINT_ENV = "OTEL_EXPORTER_OTLP_ENDPOINT"
DEFAULT_HEADERS_ENV = "OTEL_EXPORTER_OTLP_HEADERS"
DEFAULT_SERVICE_NAME = "onyx"
#: 单次 POST 的最大 span 数。collector 的默认请求体上限常在 4MB 附近，
#: 分批比"攒大了再一次性发、被 413 拒绝后无限重试"更可控
DEFAULT_BATCH_SPANS = 200
#: 队列上限：观测系统宁可丢样本也不能把主链路拖垮，但**丢多少必须可见**
DEFAULT_QUEUE_CAP = 4000

#: 这些事件构成 span；其余折进根 span 的属性
_SPAN_START = EventType.TOOL_EXEC_START
_SPAN_END = EventType.TOOL_EXEC_END

log = logging.getLogger("onyx.store.sinks.otlp")


def _sha_hex(text: str, bytes_len: int) -> str:
    """从 onyx 的 trace/span 名字派生固定长度 hex id。

    onyx 的 trace_id 是 26 字符 ULID，而 OTLP 要求 traceId 是 16 字节、spanId 是 8 字节。
    派生而不是截断：截断会让两个不同 trace 有概率撞 id，而撞了之后 collector 会把
    两条无关的调用链画成同一条。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: bytes_len * 2]


def _epoch_ns(wall_iso: str) -> int:
    """`wall_iso` → unix epoch 纳秒。解析不了就返回 0 并让调用方保留原串作为属性。"""
    try:
        dt = datetime.fromisoformat(wall_iso)
    except (TypeError, ValueError):
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1e9)


def _attr(key: str, value: Any) -> dict[str, Any] | None:
    """OTLP AnyValue。`None` 表示"这个属性不存在"，直接不导而不是导一个空值。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        # OTLP 的 JSON 编码规定 fixed64/int64 是**字符串**；发数字会被多数 collector 拒绝
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    if isinstance(value, (list, tuple)):
        values = [v for item in value if (v := _raw_item(item)) is not None]
        return {"key": key, "value": {"arrayValue": {"values": values}}}
    if isinstance(value, dict):
        # 必须是 JSON 而不是 `str(dict)`：Python 的 repr 用单引号，
        # 接收端 json.loads 会直接失败——而失败发生在 collector 里，我们看不见
        return {"key": key, "value": {"stringValue": json.dumps(
            value, ensure_ascii=False, sort_keys=True, default=str)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def _raw_item(value: Any) -> dict[str, Any] | None:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if value is None:
        return None
    return {"stringValue": str(value)}


def _attrs(pairs: dict[str, Any]) -> list[dict[str, Any]]:
    return [a for key, value in pairs.items() if (a := _attr(key, value)) is not None]


def _status(code: int, message: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"code": code}
    if message:
        out["message"] = message[:500]
    return out


class OtlpEventSink:
    """把事件流攒成 OTLP spans 并 POST 到 collector。

    `emit()` **不发消息**，只入队：主链路不该为一个外部服务付网络延迟。
    发送发生在 `flush()`（`Runtime.flush()` / 进程关闭 / 队列涨过批次阈值）。
    """

    name = "otlp"

    def __init__(
        self,
        endpoint: str = "",
        *,
        service_name: str = DEFAULT_SERVICE_NAME,
        headers: dict[str, str] | None = None,
        timeout: float = 5.0,
        batch_spans: int = DEFAULT_BATCH_SPANS,
        queue_cap: int = DEFAULT_QUEUE_CAP,
        include_args: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        endpoint = (endpoint or os.environ.get(DEFAULT_ENDPOINT_ENV, "")).rstrip("/")
        if not endpoint:
            # 没配端点却显式要求导出 ⇒ 报错，而不是"接上了但其实什么都没发"
            raise ValueError(
                f"otlp sink 需要端点：传 endpoint=<url> 或设置 {DEFAULT_ENDPOINT_ENV}"
                "（例：http://127.0.0.1:4318）。静默不导出比不接这个 sink 更糟——"
                "它会让人以为数据已经在 collector 里了"
            )
        #: 允许只给 host 不带协议时补全，也接受完整 /v1/traces
        self.endpoint = endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"
        self.service_name = service_name
        self.include_args = include_args
        self._headers = {
            "content-type": "application/json",
            **_parse_headers(os.environ.get(DEFAULT_HEADERS_ENV, "")),
            **(headers or {}),
        }
        self._client = httpx.Client(timeout=timeout, headers=self._headers, transport=transport)
        self._batch_spans = max(1, batch_spans)
        self._queue_cap = max(1, queue_cap)
        self._lock = threading.Lock()
        #: trace_id → 正在攒的 span 集合
        self._open: dict[str, dict[str, Any]] = {}
        #: 已完整、待发出去的 span
        self._ready: list[dict[str, Any]] = []
        self.exported = 0
        self.dropped = 0
        self.failures = 0
        self.requeued = 0
        self.events_seen = 0
        #: 落在"没有开着的 trace"上的事件数。它增长说明事件顺序或装配出了问题，
        #: 而不是网络问题——没这个计数，"接上了但一条都没导出去"就看不出来
        self.orphans = 0

    # ── EventSink 契约 ────────────────────────────────────────────
    def emit(self, event: TraceEvent) -> None:
        self.events_seen += 1
        kind = event.type
        with self._lock:
            if kind is EventType.TRACE_START:
                self._open[event.trace_id] = self._start_span(event)
            elif kind is EventType.TRACE_END:
                self._close_trace(event)
            elif kind is _SPAN_START:
                self._tool_start(event)
            elif kind is _SPAN_END:
                self._tool_end(event)
            else:
                self._fold(event)

    def flush(self, timeout: float = 1.0) -> None:
        """把已完成的 span 全部发出去（按批次循环）。

        失败时把**这一批原样放回队首**并计数，然后抛出：下一次 flush 还会再试。
        退回的是 span 本身而不是"记一笔欠账"——编码是纯函数，重试不需要伪造数据；
        而"记了欠账但数据没了"的重试是假的重试。
        """
        while True:
            with self._lock:
                batch = self._take_locked()
            if not batch:
                return
            payload = json.dumps(
                _resource_batches(batch, self.service_name), ensure_ascii=False
            ).encode()
            try:
                resp = self._client.post(self.endpoint, content=payload, headers=self._headers)
                if resp.status_code >= 400:
                    raise RuntimeError(f"collector 返回 {resp.status_code}: {resp.text[:200]}")
            except Exception as exc:
                # 隔离由 EventFanout 负责；这里先把数据退回队列并让故障继续冒出去
                with self._lock:
                    self._ready = batch + self._ready
                    self._trim_locked()
                    self.requeued += len(batch)
                self.failures += 1
                raise RuntimeError(
                    f"OTLP 导出失败：{len(batch)} 个 span 已退回队列"
                    f"（累计失败 {self.failures} 次），下一次 flush 会重试: {exc}"
                ) from exc
            self.exported += len(batch)

    def close(self) -> None:
        """尽力把剩余 span 发出去，但**不抛**。

        `close()` 通常站在 `finally` 里：它抛出会盖掉真正让进程出错的那条异常，
        而数据并没有丢（SQLite 才是权威存储），欠着的 span 仍能在 `stats` 里看到。
        """
        try:
            self.flush()
        except Exception as exc:  # noqa: BLE001 - 关闭路径只记录，不制造第二个异常
            log.warning("OTLP 关闭时仍有导出失败：%s（队列剩余 %d）",
                        exc, len(self._ready))
        finally:
            self._client.close()

    # ── 内部：攒 span ─────────────────────────────────────────────
    def _start_span(self, event: TraceEvent) -> dict[str, Any]:
        payload = event.payload
        model = str(payload.get("model") or "")
        purpose = str(payload.get("purpose") or "")
        return {
            "trace_id": event.trace_id,
            "span_id": _sha_hex(f"{event.trace_id}:root", 8),
            "name": f"{purpose or 'trace'} {model}".strip(),
            "start_ns": _epoch_ns(event.wall_iso),
            "end_ns": 0,
            "status": _status(0),
            "attributes": _attrs({
                "onyx.trace_id": event.trace_id,
                "onyx.purpose": purpose,
                "onyx.provider_id": payload.get("provider_id"),
                "onyx.kind": payload.get("kind"),
                "gen_ai.request.model": model or None,
                "onyx.contract_version": event.version,
            }),
            "_tools": {},
            "_folded": {},
        }

    def _fold(self, event: TraceEvent) -> None:
        """非结构事件折进根 span：只统计与取关键结论，不做逐事件 span。"""
        span = self._open.get(event.trace_id)
        if span is None:
            # 落在"没有开着的 trace"上的事件。它增长说明装配或事件顺序出了问题，
            # 而不是网络问题——没这个计数，"接上了但一条都没导出去"就看不出来
            self.orphans += 1
            return
        folded: dict[str, Any] = span["_folded"]
        key = str(event.type)
        folded[key] = folded.get(key, 0) + 1
        if event.type is EventType.RECONCILED:
            # 该事件在契约里存在（`PAYLOAD_REQUIRED`），但 gateway 目前不发它：
            # 采信结果是在 observer 内部算完直接落库的。所以正常链路上这条分支不会命中，
            # 而 sink 也不因此编造"采信来源"——只导出事件流里真的出现过的东西。
            span["attributes"].extend(_attrs({
                "onyx.usage.source": event.payload.get("chosen_source"),
                "onyx.usage.confidence": event.payload.get("confidence"),
            }))
        elif event.type in (EventType.USAGE_ENGINE, EventType.USAGE_COMPAT):
            tag = "engine" if event.type is EventType.USAGE_ENGINE else "compat"
            payload = event.payload
            span["attributes"].extend(_attrs({
                f"onyx.usage.{tag}.in_tokens": payload.get("in_tokens"),
                f"onyx.usage.{tag}.out_tokens": payload.get("out_tokens"),
                f"onyx.usage.{tag}.thinking_tokens": payload.get("thinking_tokens"),
                f"onyx.usage.{tag}.cached_tokens": payload.get("cached_tokens"),
                f"onyx.usage.{tag}.ok": payload.get("ok"),
            }))
        elif event.type is EventType.FIRST_TOKEN:
            span["attributes"].extend(_attrs({
                "onyx.ttft_ms": event.payload.get("ttft_ms"),
            }))
        elif event.type is EventType.ANOMALY:
            codes = folded.setdefault("_anomaly_codes", [])
            code = str(event.payload.get("code") or "")
            if code and code not in codes:
                codes.append(code)

    def _tool_start(self, event: TraceEvent) -> None:
        span = self._open.get(event.trace_id)
        if span is None:
            self.orphans += 1
            return
        name = str(event.payload.get("name") or "")
        key = f"{name}:{event.payload.get('step')}"
        span["_tools"][key] = {
            "span_id": _sha_hex(f"{event.trace_id}:{key}", 8),
            "name": f"tool {name}",
            "start_ns": _epoch_ns(event.wall_iso),
            "end_ns": 0,
            "status": _status(0),
            "attributes": _attrs({
                "onyx.tool.name": name,
                "onyx.tool.step": event.payload.get("step"),
                **self._arg_attrs(event.payload.get("args")),
            }),
        }

    def _arg_attrs(self, args: Any) -> dict[str, Any]:
        """默认只导键名与数量：参数值里常有地址、身份、内部 ID。"""
        if not isinstance(args, dict):
            return {}
        if self.include_args:
            return {"onyx.tool.args": args}
        return {"onyx.tool.arg_keys": sorted(str(k) for k in args),
                "onyx.tool.arg_count": len(args)}

    def _tool_end(self, event: TraceEvent) -> None:
        span = self._open.get(event.trace_id)
        if span is None:
            self.orphans += 1
            return
        name = str(event.payload.get("name") or "")
        key = f"{name}:{event.payload.get('step')}"
        tool = span["_tools"].pop(key, None)
        status = str(event.payload.get("status") or "")
        tool = tool or {
            # 没等到 START 也照样导出：漏一个 span 会被读成"这次没执行工具"
            "span_id": _sha_hex(f"{event.trace_id}:{key}", 8),
            "name": f"tool {name}",
            "start_ns": _epoch_ns(event.wall_iso),
            "status": _status(0),
            "attributes": [],
        }
        tool["end_ns"] = _epoch_ns(event.wall_iso)
        tool["status"] = _status(2 if status not in {"ok", "", "mocked"} else 1,
                                  "" if status in {"ok", "mocked"} else status)
        tool["attributes"] = list(tool["attributes"]) + _attrs({
            "onyx.tool.status": status,
            "onyx.tool.latency_ms": event.payload.get("latency_ms"),
            "onyx.tool.error_kind": event.payload.get("error_kind"),
        })
        span.setdefault("_child_spans", []).append(tool)

    def _close_trace(self, event: TraceEvent) -> None:
        span = self._open.pop(event.trace_id, None)
        if span is None:
            return
        folded: dict[str, Any] = span.pop("_folded")
        children = span.pop("_tools")
        span["end_ns"] = _epoch_ns(event.wall_iso)
        if not span["start_ns"]:
            span["start_ns"] = span["end_ns"]
        status = str(event.payload.get("status") or "")
        span["status"] = _status(2 if status not in {"ok", ""} else 1,
                                 "" if status == "ok" else status)
        span["attributes"] = list(span["attributes"]) + _attrs({
            "onyx.status": status,
            "onyx.wall_ms": event.payload.get("wall_ms"),
            "onyx.finish_reason": event.payload.get("finish_reason"),
            "onyx.error": event.payload.get("error") or None,
            "onyx.anomalies": folded.get("_anomaly_codes") or None,
            "onyx.events": dict(sorted(folded.items(), key=lambda kv: kv[0])),
        })
        children_list = span.pop("_child_spans", [])
        children_list.extend(children.values())  # 没配到 END 的工具也带出去，标为未正常结束
        span["children"] = children_list
        self._ready.append(span)
        self._trim_locked()

    def _take_locked(self) -> list[dict[str, Any]]:
        """取走一批完整 span（编码由调用方做，失败时这批能原样退回）。"""
        batch = self._ready[: self._batch_spans]
        self._ready = self._ready[self._batch_spans:]
        return batch

    def _trim_locked(self) -> None:
        """队列上限：丢最老的，但**丢多少必须可见**（与 sqlite sink 的 dropped 一致）。"""
        overflow = len(self._ready) - self._queue_cap
        if overflow > 0:
            self._ready = self._ready[overflow:]
            self.dropped += overflow

    @property
    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "endpoint": self.endpoint, "events": self.events_seen,
                "exported": self.exported, "pending": len(self._ready),
                "open_traces": len(self._open), "dropped": self.dropped,
                "failures": self.failures, "requeued": self.requeued,
                "orphans": self.orphans,
                "encoding": "json",
            }



def _resource_batches(spans: list[dict[str, Any]], service_name: str) -> dict[str, Any]:
    """OTLP JSON 的形状：resourceSpans → scopeSpans → spans。"""
    encoded: list[dict[str, Any]] = []
    for span in spans:
        encoded.append({
            "traceId": _sha_hex(f"{span['trace_id']}", 16),
            "spanId": span["span_id"],
            "name": span["name"],
            "startTimeUnixNano": str(span.get("start_ns") or 0),
            "endTimeUnixNano": str(span.get("end_ns") or 0),
            "kind": 1,  # SPAN_KIND_INTERNAL
            "status": span.get("status") or {"code": 0},
            "attributes": span.get("attributes") or [],
            "children": [
                {
                    "traceId": _sha_hex(f"{span['trace_id']}", 16),
                    "spanId": child["span_id"],
                    "parentSpanId": span["span_id"],
                    "name": child["name"],
                    "startTimeUnixNano": str(child.get("start_ns") or 0),
                    "endTimeUnixNano": str(child.get("end_ns") or child.get("start_ns") or 0),
                    "kind": 1,
                    "status": child.get("status") or {"code": 0},
                    "attributes": child.get("attributes") or [],
                }
                for child in span.get("children") or []
            ],
        })
    flat = _lift_children(encoded)
    return {
        "resourceSpans": [{
            "resource": {"attributes": _attrs({
                "service.name": service_name,
                "onyx.app_version": __version__,
                "onyx.exporter": "onyx.sinks.otlp",
                "onyx.encoding": "json",
            })},
            "scopeSpans": [{
                "scope": {"name": "onyx", "version": __version__},
                "spans": flat,
            }],
        }]
    }


def _lift_children(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OTLP 没有嵌套 span 这个概念：children 要摊平成同级的独立 span。"""
    out: list[dict[str, Any]] = []
    for span in spans:
        children = span.pop("children", [])
        out.append(span)
        out.extend(children)
    return out


def _parse_headers(raw: str) -> dict[str, str]:
    """`OTEL_EXPORTER_OTLP_HEADERS` 是 `k1=v1,k2=v2`（可为 base64 值）。"""
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip().lower()] = value.strip()
    return out


def build(**options: Any) -> OtlpEventSink:
    """`onyx.sinks` 的注册入口：`build_event_sink("otlp", **opts)`。"""
    return OtlpEventSink(
        endpoint=str(options.get("endpoint") or ""),
        service_name=str(options.get("service_name") or DEFAULT_SERVICE_NAME),
        headers=options.get("headers"),
        timeout=float(options.get("timeout") or 5.0),
        batch_spans=int(options.get("batch_spans") or DEFAULT_BATCH_SPANS),
        queue_cap=int(options.get("queue_cap") or DEFAULT_QUEUE_CAP),
        include_args=bool(options.get("include_args") or False),
        transport=options.get("transport"),
    )


__all__ = ["OtlpEventSink", "build"]
