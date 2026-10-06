"""性能基线的网格与条件（S36）。

这一层只管"要测什么、在什么条件下测"，不碰引擎也不碰库——因为基线最容易坏的地方
恰恰是**条件**：换了模型、换了引擎版本、换了 ctx，数字会变但形状一模一样，
看的人不会知道自己在比两个不同的东西。所以条件在这里是一等数据，不是脚注。

三条不许漂的判断：
1. **默认网格刻意小**。基线要常被跑才会被比较；一次跑 20 分钟的命令一个月只跑一次，
   而一个月只跑一次的数字不构成"改版有没有变慢"的答案。
2. **冷启动只测一发**。每次冷测都要先 `unload`，代价是下一格重新载入；
   把它做成"每格都冷热两遍"会让默认预算直接爆掉，而那点信息用一行 `load_ms` 就能说明白。
3. **长度档按汉字声明，实际长度以引擎回报为准**。tokenizer 是引擎的属性不是我们的，
   写死"8k tokens"只会得到一个自称 8k 而实际被裁到 2k 的格子（S32 的那条负控制）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: 一发的保守耗时估计（秒），只用于"要不要先警告一句"，不进任何计算结果
EST_REQUEST_SECONDS = 4.0
#: 超过这个发数就先打印一句预计耗时：意外地跑 20 分钟会让人从此不再跑这条命令
LOUD_REQUEST_COUNT = 24

MAX_TARGET_TOKENS = 8192
#: 短于这个长度的格子量的不是模型吞吐（提示板开销与调度占比都盖过解码），
#: 而且必须高过语料那句指令的长度，否则那一档连指令都装不下。
MIN_PROMPT_CHARS = 64


class PerfSpecError(ValueError):
    """网格搭不起来（空档、并发数小于 1、预算不是正数……）。带修法，不给堆栈。"""


@dataclass(frozen=True, slots=True)
class Cell:
    """一个网格点：在这个长度、这个并发下的一批请求。"""

    prompt_chars: int
    target_tokens: int
    concurrency: int
    repeat: int
    phase: str = "warm"          # warm | cold（cold 全网格最多一个）

    @property
    def key(self) -> str:
        return f"{self.prompt_chars}c/{self.target_tokens}t/x{self.concurrency}/{self.phase}"

    @property
    def n_requests(self) -> int:
        return self.concurrency * self.repeat


@dataclass(frozen=True, slots=True)
class BenchPlan:
    """一次 perf 运行要做什么。`grid_signature()` 进条件指纹——
    换了网格就是换了实验，两次结果不许直接相减。"""

    model: str
    prompt_chars: tuple[int, ...] = (600, 2400)
    target_tokens: tuple[int, ...] = (64, 256)
    concurrency: tuple[int, ...] = (1,)
    repeat: int = 2
    cold: bool = False
    budget_s: float = 300.0
    keep_alive: str = "10m"
    stream: bool = True
    #: 传给引擎的上下文窗口。None = 不传（用引擎默认）；它进条件指纹，
    #: 因为"同一个 8k 档"在 ctx=4096 与 ctx=32768 上量的根本不是同一件事。
    num_ctx: int | None = None
    #: 生成参数快照：温度非 0 时同一格的两次采样会因采样噪声分叉，必须记下来
    params: dict[str, Any] = field(default_factory=lambda: {"temperature": 0.0})

    def __post_init__(self) -> None:
        if not (self.model or "").strip():
            raise PerfSpecError("perf 需要模型名（--model），空模型名跑出来的基线不属于任何模型")
        for name, values in (("prompt_chars", self.prompt_chars),
                             ("target_tokens", self.target_tokens),
                             ("concurrency", self.concurrency)):
            if not values:
                raise PerfSpecError(f"{name} 不能为空：没有档位就没有格子，也就没有基线")
            if any(int(v) < 1 for v in values):
                raise PerfSpecError(f"{name} 每一项都要 ≥ 1（收到 {list(values)}）")
        if min(self.prompt_chars) < MIN_PROMPT_CHARS:
            raise PerfSpecError(
                f"prompt_chars 最小 {MIN_PROMPT_CHARS} 个汉字：再短的格子量到的是调度开销而不是模型吞吐"
            )
        if max(self.target_tokens) > MAX_TARGET_TOKENS:
            raise PerfSpecError(
                f"target_tokens 最大 {MAX_TARGET_TOKENS}：更长的生成要分档跑，"
                "否则预算会被一格吃光而剩下的格子会被记成『没测到』"
            )
        if int(self.repeat) < 1:
            raise PerfSpecError(f"--repeat 要 ≥ 1（收到 {self.repeat}）")
        if self.num_ctx is not None and int(self.num_ctx) < 1:
            raise PerfSpecError("--num-ctx 要 ≥ 1，或者干脆不传（表示「用引擎默认」）；"
                                "0 会被引擎读成别的意思，不该拿来冒充「不限」")
        if float(self.budget_s) <= 0:
            raise PerfSpecError(f"budget_s 必须 > 0（收到 {self.budget_s}）；0 不是"
                                "「不限」，而是「一格都别跑」，那种意思要显式写")
        if not (self.keep_alive or "").strip():
            raise PerfSpecError("keep_alive 不能是空串：引擎会把它读成「立即卸载」，"
                                "于是每一格都是冷启动，而吞吐数字看起来完全正常")

    # ── 网格 ──────────────────────────────────────────────────────
    def cells(self) -> tuple[Cell, ...]:
        """展开顺序固定：cold 那一发排最前，warm 按 (长度, 生成长度, 并发) 升序。

        顺序必须可复现：预算用完时"还剩哪几格"取决于跑到过哪几格，
        两个同样参数、同样预算的运行必须欠同一批格子。
        三个维度都先去重再排序——`1,1,2` 里的第二个 1 不是"再测一遍"，
        那是 `--repeat` 的活；留着它会让同一个格子出现两次，预算判断也会偏。
        """
        warm = tuple(
            Cell(pc, tt, cc, self.repeat, "warm")
            for pc in sorted(set(self.prompt_chars))
            for tt in sorted(set(self.target_tokens))
            for cc in sorted(set(self.concurrency))
        )
        if not self.cold:
            return warm
        first = warm[0]
        cold_cell = Cell(first.prompt_chars, first.target_tokens, 1, 1, "cold")
        return (cold_cell, *warm)

    @property
    def n_requests(self) -> int:
        return sum(cell.n_requests for cell in self.cells())

    def grid_signature(self) -> str:
        """进指纹的网格表示（与 `cells()` 同一份展开，不另写一遍排序规则）。"""
        return "|".join(
            f"{c.prompt_chars}:{c.target_tokens}:{c.concurrency}:{c.repeat}:{c.phase}"
            for c in self.cells()
        )

    def warn_before_run(self) -> str:
        """跑之前要说出口的一句话。意外耗时是"这条命令没人再跑"的头号原因。"""
        estimate = self.n_requests * EST_REQUEST_SECONDS
        note = (f"这次要发 {self.n_requests} 发（{len(self.cells())} 格），"
                f"按每发约 {EST_REQUEST_SECONDS:.0f}s 粗估约 {estimate / 60:.1f} 分钟，"
                f"预算上限 {self.budget_s:.0f}s")
        if self.n_requests > LOUD_REQUEST_COUNT:
            note += "；超出预算的格子会被记成「没测到」而不是补 0"
        return note

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "prompt_chars": list(self.prompt_chars),
            "target_tokens": list(self.target_tokens),
            "concurrency": list(self.concurrency),
            "repeat": self.repeat,
            "cold": self.cold,
            "budget_s": self.budget_s,
            "keep_alive": self.keep_alive,
            "stream": self.stream,
            "num_ctx": self.num_ctx,
            "params": dict(self.params),
            "n_requests": self.n_requests,
            "n_cells": len(self.cells()),
        }


def parse_int_list(raw: str, *, option: str, hi: int | None = None) -> tuple[int, ...]:
    """`"1,2,4"` → `(1,2,4)`。报错要带上是哪个 flag 与正确形状，
    否则人只会把整条命令重打一遍而不是去读帮助。"""
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        raise PerfSpecError(f"{option} 不能为空，要的是逗号分隔的正整数（例如 {option} 1,2,4）")
    out: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise PerfSpecError(f"{option} 里 {part!r} 不是整数") from exc
        if value < 1:
            raise PerfSpecError(f"{option} 里 {part!r} 要 ≥ 1")
        if hi is not None and value > hi:
            raise PerfSpecError(f"{option} 里 {part!r} 超过上限 {hi}")
        out.append(value)
    return tuple(out)
