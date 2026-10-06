"""`onyx perf`：吞吐/延迟基线（S36）。

一句话分工：`spec` 说"测什么、在什么条件下测"，`corpus` 造确定长度的 prompt，
`bench` 打请求并把每条数字连出处一起收下来，`report` 让人类读与机器读同一份形状。

刻意**不是**一个评测任务：任务契约要求每个任务交出"主分数"，而延迟没有对错可言，
唯一能填进去的就是 `decode_tps` ——那正是"用稳定性指标顶掉正确性"的反面教材。
理由与网格设计写在 `docs/IMPLEMENTATION.md` 的 S36 一节。

这里不 import 任何会发网络的库：请求一律经 `onyx.llm.gateway` 这一唯一咽喉点，
所以基线里的每个数字都有一条真 trace 可以对点。
"""

from onyx.perf.bench import BenchOutcome, CellResult, Sample, collect, diff_conditions
from onyx.perf.corpus import cjk_count, colliding_prefixes, prompt_for
from onyx.perf.report import cell_line, run_lines, unmeasured_lines
from onyx.perf.spec import BenchPlan, Cell, PerfSpecError, parse_int_list

__all__ = [
    "BenchOutcome",
    "BenchPlan",
    "Cell",
    "CellResult",
    "PerfSpecError",
    "Sample",
    "cell_line",
    "cjk_count",
    "collect",
    "colliding_prefixes",
    "diff_conditions",
    "parse_int_list",
    "prompt_for",
    "run_lines",
    "unmeasured_lines",
]
