"""性能基线的语料：按**汉字数**声明长度，内容确定且互不为前缀（S36）。

为什么不写"8k tokens"而写"8192 个汉字"：tokenizer 是引擎的属性，不是我们的。
自称的 token 档位在换 tokenizer 后量的就不是同一个长度；更糟的是引擎会在超过 `num_ctx`
时**自己把正文裁掉**（S32 实测：16.8k tok 的正文被 Ollama 报成 `in_tokens=2050`），
于是"8k 档的吞吐"实际上是一行 2k 的数字，而它看起来完全合理。
所以这里只声明可核实的物理量（汉字数），实际长度取**引擎回报**的数字。

两条会被断言的性质：
- **确定性**：同一个长度两次生成一字不差，否则"这两次跑的是同一份考卷"没有依据。
- **互不为前缀**：如果短档是长档的前缀，跑完短档再跑长档会命中 KV 缓存，
  长档的 prefill 吞吐被凭空抬高。`prefill_mode()` 能测出冷热，但那已经是事后了——
  语料本身就不该埋这个。

这份语料与评测数据无关：它唯一的作用是**产生一个已知长度的 prompt**。
"""

from __future__ import annotations

from itertools import pairwise

from onyx.llm.measurement.heuristic import CJK_RE, split_cjk

#: 让模型继续往下写而不是回答问题。措辞要中性：任何"答案在正文里"的暗示都会改变生成形态。
INSTRUCTION = "请接着上面的内容继续往下写，每次一行，不要复述原文，也不要总结。"

#: 一段固定语料池。顺序写死（不依赖哈希、不依赖目录读取顺序），每条约 18–26 个汉字。
_FILLER: tuple[str, ...] = (
    "仓库东侧的货架按编号排列，每一层都有独立的标签条",
    "周会记录里提到备件周转天数偏长，需要重新核定安全库存",
    "三号产线的温控曲线在夜班时段出现两次小幅回落",
    "供应商交付清单与入库单存在一行数量不一致的情况",
    "质检抽样的批次号已经登记，复检结论明天上午给出",
    "巡检表要求逐台设备记录振动值与轴承温度",
    "系统升级后导出的报表缺少了一列运费，已在核对",
    "冷链车的温度记录仪需要每周导出一回数据存档",
    "盘盈盘亏的差异要落到具体经办人并由主管确认",
    "车间能耗分时统计显示午后两小时的用电峰值偏高",
    "工装夹具的借用登记本最近三页缺少归还时间",
    "客户投诉的工单已经转给售后，等待现场核实结果",
)


def cjk_count(text: str) -> int:
    """正文里的汉字数（本项目声明长度用的就是这个物理量）。"""
    cjk, _other = split_cjk(text)
    return cjk


def prompt_for(chars: int) -> str:
    """构造**汉字数恰好为 `chars`** 的 prompt（含最后那句指令）。

    长度是数着汉字切的，不是按字符数切：正文里有数字与半角点号，按字符切会得到一个
    "自称 600 实际 533"的档位——而 x 轴一旦是假的，每一格的吞吐都在说一件没发生过的事。
    这条正是被 `exact_length_failures()` 当场抓出来的。
    """
    need = chars - cjk_count(INSTRUCTION)
    if need < 1:
        raise ValueError(f"prompt 长度至少要 {cjk_count(INSTRUCTION) + 1} 个汉字才装得下指令"
                         f"（收到 {chars}）")
    label = f"档 {chars}："
    parts = [label]
    index = 1
    while cjk_count("".join(parts)) < need:
        parts.append(f"第 {index} 条，{_FILLER[(index - 1) % len(_FILLER)]}；")
        index += 1
    return _cut_at_cjk("".join(parts), need) + INSTRUCTION


def _cut_at_cjk(text: str, target: int) -> str:
    """截到第 `target` 个汉字为止（含它）。多出来的尾巴不要，非汉字跟着它前面的汉字走。"""
    count = 0
    for index, char in enumerate(text):
        if CJK_RE.match(char):
            count += 1
            if count == target:
                return text[:index + 1]
    return text


# ── 考卷自检：语料本身要能被测出自洽 ───────────────────────────────
def exact_length_failures(lengths: list[int]) -> list[str]:
    """汉字数不对的档。返回空列表才算过。"""
    out = []
    for chars in lengths:
        try:
            got = cjk_count(prompt_for(chars))
        except ValueError as exc:
            out.append(f"{chars}：{exc}")
            continue
        if got != chars:
            out.append(f"{chars}：实际 {got} 个汉字")
    return out


def nondeterministic(lengths: list[int]) -> list[int]:
    """同一档位两次构造不一致的情况——那意味着排序或哈希漏进了语料。"""
    return [chars for chars in lengths if prompt_for(chars) != prompt_for(chars)]


def colliding_prefixes(lengths: list[int], *, window: int = 32) -> list[tuple[int, int]]:
    """返回前 `window` 个字符相同的档位对——非空就说明档位之间会互相命中 KV 缓存。

    查"前缀相同"而不是"整段互为前缀"：短档永远不是长档的完整前缀（结尾指令不同），
    但 KV 缓存只认**开头那一段**。所以这条判据要盯住开头，否则它会永远绿。
    """
    ordered = sorted(lengths)
    heads = {chars: prompt_for(chars)[:window] for chars in ordered}
    out: list[tuple[int, int]] = []
    for i, short in enumerate(ordered):
        for long in ordered[i + 1:]:
            if heads[short] == heads[long]:
                out.append((short, long))
    return out


def adjacent_repeats(chars: int) -> int:
    """相邻两行的正文完全相同的次数。

    语料池会循环（2400 个汉字要一百多行，池子只有 12 条），这本身没问题；
    有问题的是**连着两句一样**——那会让模型进入"复读"形态，测出来的就不是生成长度而是复读速度。
    """
    lines = [part for part in prompt_for(chars).split("；") if part.strip()]
    bodies = [line.split("，", 1)[-1] for line in lines[1:]]     # 去掉档位前缀那一行
    return sum(1 for a, b in pairwise(bodies) if a and a == b)
