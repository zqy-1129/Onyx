"""timing visitor：把 `FIRST_TOKEN` 接成 `ttft_ms` 与 `first_token_at`。

这个字段曾经**永远为空**：provider 算得出首字时刻、事件也真发出来、gateway 也转进了观测，
但没有任何 visitor 接它，于是 `TraceState.ttft_ms` 与 `trace.first_token_at` 从来没有写入方，
看板的 TTFT 一栏恒显示「—」——而文档把「—」解释成"兼容通道没有分段时序"，
把一个缺失的字段读成了引擎的限制（S36 查出，S37 结案）。

两条口径：
1. **只认测出来的 TTFT**。非流式请求的 `FIRST_TOKEN` 带 `proxy: prompt_eval_duration`
   （`llm/streaming.py` 那侧写明的代理值）——那种请求根本没有"首字时间"这个量
   （没有逐字交付），所以 `ttft_ms` 保持 None，只把代理出处记下来。
   把 prefill 时长存成延迟，会让人把"想清楚了"读成"开口很快"。
2. **只认第一次**。重放、多次首包、或工具循环里的后续请求都不能覆盖首个时刻；
   落库侧 `mark_first_token` 的 `WHERE first_token_at IS NULL` 是同一条语义的第二处守护，
   这里先在内存里幂等，避免同一事件被投两次时把时长改大。
"""

from __future__ import annotations

from onyx.core.event import EventType, TraceEvent
from onyx.obs.state import TraceState
from onyx.obs.visitors import BaseVisitor

PROXY_KEY = "ttft_source"


class TimingVisitor(BaseVisitor):
    name = "timing"

    def on(self, event: TraceEvent, state: TraceState) -> None:
        if event.type is not EventType.FIRST_TOKEN:
            return
        if state.ttft_ms is not None or state.first_token_at is not None:
            return  # 只认第一次
        payload = event.payload
        proxy = payload.get("proxy")
        if proxy:
            # 代理值不当测量：记出处，数字留空
            state.extra[PROXY_KEY] = f"proxy:{proxy}"
            return
        value = payload.get("ttft_ms")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            state.extra[PROXY_KEY] = "absent"
            return
        state.ttft_ms = round(float(value), 3)
        state.first_token_at = event.wall_iso
        state.extra[PROXY_KEY] = "measured"
