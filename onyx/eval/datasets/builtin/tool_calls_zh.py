"""中文工具调用数据集生成器（`tool_calls_zh.jsonl` 的来源）。

与 `intent_zh.py` 同样是生成器而不是裸 JSONL：一条工具调用样本同时带着
**指令、可用工具集、期望调用、期望参数**，手写 JSON 很容易在某一处漂移
（比如改了 schema 却忘了改期望参数），而生成器让工具定义只有一份。

四个子集刻意分开，因为它们测的是**不同的失败**：
- `single`         该调一个，调对了吗
- `parallel`       该调多个，漏了吗、多调了吗
- `no_call_needed` 不该调，调了吗 ← 这一格单独统计**误调率**，
                   绝不能算成"没调对"：一个从不乱调工具的模型和一个
                   不会调工具的模型，在这一格上得分相同，但完全相反
- `args`           工具选对了，参数填对了吗（枚举/日期/数值/数组各占一部分）

`send_email` 是刻意放进工具集的**危险工具**：它是 write 副作用。
样本里它几乎从不该被调用，用来测"模型会不会因为工具存在就乱用"。
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: 工具定义只写一份，样本里按名字引用
TOOLS: dict[str, dict[str, Any]] = {
    "get_weather": {
        "description": "查询指定城市的当前天气，包括温度、天气状况与风力。仅在用户询问实时天气时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称，中文或英文均可，例如「北京」。"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "温度单位。默认摄氏度；用户明确要求华氏度时才传 fahrenheit。"},
            },
            "required": ["city"],
            "additionalProperties": False,
        },
    },
    "calculator": {
        "description": "计算一个算术表达式的数值结果。用于任何需要精确算术的场合，不要自己心算。",
        "parameters": {
            "type": "object",
            "properties": {
                "expr": {"type": "string",
                         "description": "待求值的算术表达式，例如 '(3 + 5) * 12'。不要包含赋值或函数定义。"},
            },
            "required": ["expr"],
            "additionalProperties": False,
        },
    },
    "time_now": {
        "description": "返回当前的日期与时间。需要知道「现在几点」「今天几号」时使用它，不要凭训练数据猜。",
        "parameters": {
            "type": "object",
            "properties": {
                "tz_offset_hours": {"type": "number",
                                    "description": "相对 UTC 的时区偏移小时数，东八区填 8；范围 ±24。"},
                "fmt": {"type": "string", "enum": ["iso", "date", "time", "unix"],
                        "description": "输出格式，可选 iso（默认）、date、time、unix。"},
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    "translate": {
        "description": "把一段文本翻译成指定的目标语言。只在用户明确要求翻译时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "需要翻译的原文，保持原样传入，不要自己先改写。"},
                "target_lang": {"type": "string", "enum": ["en", "zh", "ja", "fr", "de", "es"],
                                "description": "目标语言的 ISO 639-1 代码，例如英文填 en。"},
            },
            "required": ["text", "target_lang"],
            "additionalProperties": False,
        },
    },
    "search_web": {
        "description": "在公网搜索并返回若干条结果。用于查询实时资讯、文档或任何模型不掌握的事实。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词，尽量具体，避免整句自然语言。"},
                "max_results": {"type": "integer",
                                "description": "返回结果条数上限，默认 5，最大 20。"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    "db_query": {
        "description": "在只读副本上执行一条 SELECT 查询并返回结果行。禁止任何写操作或 DDL。",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string",
                        "description": "要执行的 SELECT 语句。必须是只读查询，不允许 INSERT/UPDATE/DELETE/DROP。"},
                "limit": {"type": "integer", "description": "返回行数上限，默认 100，最大 1000。"},
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
    "convert_currency": {
        "description": "按当前汇率把一笔金额从一种货币换算成另一种货币。",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "要换算的金额，正数。"},
                "from": {"type": "string", "enum": ["CNY", "USD", "EUR", "JPY", "GBP", "HKD"],
                         "description": "源货币的三字母代码，例如人民币填 CNY。"},
                "to": {"type": "string", "enum": ["CNY", "USD", "EUR", "JPY", "GBP", "HKD"],
                       "description": "目标货币的三字母代码，例如美元填 USD。"},
            },
            "required": ["amount", "from", "to"],
            "additionalProperties": False,
        },
    },
    "send_email": {
        "description": "发送一封邮件给指定收件人。这是**有副作用**的操作，只有在用户明确要求发信时才用。",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人邮箱地址。"},
                "subject": {"type": "string", "description": "邮件主题，简明扼要。"},
                "body": {"type": "string", "description": "邮件正文内容。"},
            },
            "required": ["to", "subject", "body"],
            "additionalProperties": False,
        },
    },
}

#: 每条样本默认带的工具集。刻意包含 send_email：它几乎从不该被调用，
#: 用来测"工具存在就会被乱用"这种真实缺陷
DEFAULT_TOOLSET = ("get_weather", "calculator", "time_now", "translate",
                   "search_web", "db_query", "convert_currency", "send_email")

#: (指令, 期望调用列表, kind, tags)。期望调用为 [] 表示不该调任何工具
CASES: tuple[tuple[str, tuple[dict[str, Any], ...], str, tuple[str, ...]], ...] = (
    # ── single：该调一个 ──────────────────────────────────────────
    ("北京今天天气怎么样", ({"name": "get_weather", "arguments": {"city": "北京"}},), "single", ("weather",)),
    ("上海现在多少度", ({"name": "get_weather", "arguments": {"city": "上海"}},), "single", ("weather",)),
    ("帮我查一下深圳的天气，用华氏度", ({"name": "get_weather", "arguments": {"city": "深圳", "unit": "fahrenheit"}},), "single", ("weather", "enum")),
    ("广州会下雨吗", ({"name": "get_weather", "arguments": {"city": "广州"}},), "single", ("weather",)),
    ("杭州的天气如何", ({"name": "get_weather", "arguments": {"city": "杭州"}},), "single", ("weather",)),
    ("算一下 (12 + 8) * 3", ({"name": "calculator", "arguments": {"expr": "(12 + 8) * 3"}},), "single", ("math",)),
    ("帮我算 15% 的 2400 是多少", ({"name": "calculator", "arguments": {"expr": "2400 * 0.15"}},), "single", ("math",)),
    ("3 的 7 次方等于多少", ({"name": "calculator", "arguments": {"expr": "3 ** 7"}},), "single", ("math",)),
    ("1024 除以 32 再减 5", ({"name": "calculator", "arguments": {"expr": "1024 / 32 - 5"}},), "single", ("math",)),
    ("一个正方形边长 7.5，面积是多少", ({"name": "calculator", "arguments": {"expr": "7.5 * 7.5"}},), "single", ("math",)),
    ("现在几点了", ({"name": "time_now", "arguments": {"fmt": "time"}},), "single", ("time",)),
    ("今天是几号", ({"name": "time_now", "arguments": {"fmt": "date"}},), "single", ("time",)),
    ("现在东八区是什么时间", ({"name": "time_now", "arguments": {"tz_offset_hours": 8}},), "single", ("time",)),
    ("给我当前的 unix 时间戳", ({"name": "time_now", "arguments": {"fmt": "unix"}},), "single", ("time", "enum")),
    ("把「今天天气真好」翻译成英文", ({"name": "translate", "arguments": {"text": "今天天气真好", "target_lang": "en"}},), "single", ("translate", "enum")),
    ("帮我把这句话译成日语：请问洗手间在哪里", ({"name": "translate", "arguments": {"text": "请问洗手间在哪里", "target_lang": "ja"}},), "single", ("translate", "enum")),
    ("translate 'good morning' into 中文", ({"name": "translate", "arguments": {"text": "good morning", "target_lang": "zh"}},), "single", ("translate", "enum")),
    ("把合同条款翻译成法语", ({"name": "translate", "arguments": {"text": "合同条款", "target_lang": "fr"}},), "single", ("translate", "enum")),
    ("搜一下 Python 3.13 有什么新特性", ({"name": "search_web", "arguments": {"query": "Python 3.13 新特性"}},), "single", ("search",)),
    ("查一下最新的 Ollama 版本号", ({"name": "search_web", "arguments": {"query": "Ollama 最新版本号"}},), "single", ("search",)),
    ("搜索 2026 年诺贝尔物理学奖得主", ({"name": "search_web", "arguments": {"query": "2026 诺贝尔物理学奖 得主"}},), "single", ("search",)),
    ("帮我找几篇关于 KV cache 量化的论文，要 10 条", ({"name": "search_web", "arguments": {"query": "KV cache 量化 论文", "max_results": 10}},), "single", ("search",)),
    ("查一下 3 月的订单总数", ({"name": "db_query", "arguments": {"sql": "SELECT COUNT(*) FROM orders WHERE month = 3"}},), "single", ("db",)),
    ("从 users 表里取前 20 条", ({"name": "db_query", "arguments": {"sql": "SELECT * FROM users", "limit": 20}},), "single", ("db",)),
    ("统计每个部门的平均工资", ({"name": "db_query", "arguments": {"sql": "SELECT department, AVG(salary) FROM employees GROUP BY department"}},), "single", ("db",)),
    ("100 美元等于多少人民币", ({"name": "convert_currency", "arguments": {"amount": 100, "from": "USD", "to": "CNY"}},), "single", ("currency", "enum")),
    ("把 5000 日元换成欧元", ({"name": "convert_currency", "arguments": {"amount": 5000, "from": "JPY", "to": "EUR"}},), "single", ("currency", "enum")),
    ("250 英镑是多少港币", ({"name": "convert_currency", "arguments": {"amount": 250, "from": "GBP", "to": "HKD"}},), "single", ("currency", "enum")),
    ("成都现在天气如何", ({"name": "get_weather", "arguments": {"city": "成都"}},), "single", ("weather",)),
    ("武汉今天适合出门吗，看下天气", ({"name": "get_weather", "arguments": {"city": "武汉"}},), "single", ("weather",)),
    ("西安的气温用摄氏度报一下", ({"name": "get_weather", "arguments": {"city": "西安", "unit": "celsius"}},), "single", ("weather", "enum")),
    ("计算 2 的 10 次方减去 100", ({"name": "calculator", "arguments": {"expr": "2 ** 10 - 100"}},), "single", ("math",)),
    ("0.1 加 0.2 精确等于多少", ({"name": "calculator", "arguments": {"expr": "0.1 + 0.2"}},), "single", ("math",)),
    ("南京现在是什么日期", ({"name": "time_now", "arguments": {"fmt": "date"}},), "single", ("time",)),
    ("搜一下 vLLM 和 Ollama 的区别", ({"name": "search_web", "arguments": {"query": "vLLM Ollama 区别"}},), "single", ("search",)),
    ("查询 orders 表里金额最大的一笔", ({"name": "db_query", "arguments": {"sql": "SELECT MAX(amount) FROM orders"}},), "single", ("db",)),
    ("88 欧元换人民币", ({"name": "convert_currency", "arguments": {"amount": 88, "from": "EUR", "to": "CNY"}},), "single", ("currency", "enum")),
    ("把这段翻译成德语：Guten Tag", ({"name": "translate", "arguments": {"text": "Guten Tag", "target_lang": "de"}},), "single", ("translate", "enum")),
    ("重庆天气", ({"name": "get_weather", "arguments": {"city": "重庆"}},), "single", ("weather",)),
    ("现在 UTC 时间是多少", ({"name": "time_now", "arguments": {"tz_offset_hours": 0}},), "single", ("time",)),
    ("搜一下 LoRA 微调的最佳实践", ({"name": "search_web", "arguments": {"query": "LoRA 微调 最佳实践"}},), "single", ("search",)),
    ("1980 除以 8 等于几", ({"name": "calculator", "arguments": {"expr": "1980 / 8"}},), "single", ("math",)),
    ("天津的天气和温度", ({"name": "get_weather", "arguments": {"city": "天津"}},), "single", ("weather",)),
    ("把「我爱编程」翻成西班牙语", ({"name": "translate", "arguments": {"text": "我爱编程", "target_lang": "es"}},), "single", ("translate", "enum")),
    ("查一下 products 表有多少种商品", ({"name": "db_query", "arguments": {"sql": "SELECT COUNT(DISTINCT product_id) FROM products"}},), "single", ("db",)),

    # ── parallel：该调多个 ────────────────────────────────────────
    ("北京和上海今天哪个更热", (
        {"name": "get_weather", "arguments": {"city": "北京"}},
        {"name": "get_weather", "arguments": {"city": "上海"}},
    ), "parallel", ("weather", "compare")),
    ("帮我算 12*8，再算 15*7", (
        {"name": "calculator", "arguments": {"expr": "12 * 8"}},
        {"name": "calculator", "arguments": {"expr": "15 * 7"}},
    ), "parallel", ("math",)),
    ("查一下广州的天气，顺便搜一下明天的限行规则", (
        {"name": "get_weather", "arguments": {"city": "广州"}},
        {"name": "search_web", "arguments": {"query": "广州 明天 限行 规则"}},
    ), "parallel", ("mixed",)),
    ("把这段翻译成英文和日文：欢迎光临", (
        {"name": "translate", "arguments": {"text": "欢迎光临", "target_lang": "en"}},
        {"name": "translate", "arguments": {"text": "欢迎光临", "target_lang": "ja"}},
    ), "parallel", ("translate", "enum")),
    ("现在几点，另外 100 美元是多少人民币", (
        {"name": "time_now", "arguments": {"fmt": "time"}},
        {"name": "convert_currency", "arguments": {"amount": 100, "from": "USD", "to": "CNY"}},
    ), "parallel", ("mixed",)),
    ("深圳和杭州的天气都查一下", (
        {"name": "get_weather", "arguments": {"city": "深圳"}},
        {"name": "get_weather", "arguments": {"city": "杭州"}},
    ), "parallel", ("weather", "compare")),
    ("统计订单总数，同时统计用户总数", (
        {"name": "db_query", "arguments": {"sql": "SELECT COUNT(*) FROM orders"}},
        {"name": "db_query", "arguments": {"sql": "SELECT COUNT(*) FROM users"}},
    ), "parallel", ("db",)),
    ("搜一下 GGUF 格式，再搜一下 safetensors 格式", (
        {"name": "search_web", "arguments": {"query": "GGUF 格式"}},
        {"name": "search_web", "arguments": {"query": "safetensors 格式"}},
    ), "parallel", ("search",)),
    ("5000 日元换美元，300 欧元换英镑", (
        {"name": "convert_currency", "arguments": {"amount": 5000, "from": "JPY", "to": "USD"}},
        {"name": "convert_currency", "arguments": {"amount": 300, "from": "EUR", "to": "GBP"}},
    ), "parallel", ("currency", "enum")),
    ("算一下 (7+3)*9 和 (7-3)*9", (
        {"name": "calculator", "arguments": {"expr": "(7 + 3) * 9"}},
        {"name": "calculator", "arguments": {"expr": "(7 - 3) * 9"}},
    ), "parallel", ("math",)),
    ("北京天气怎么样，今天几号", (
        {"name": "get_weather", "arguments": {"city": "北京"}},
        {"name": "time_now", "arguments": {"fmt": "date"}},
    ), "parallel", ("mixed",)),
    # 翻译的输入依赖天气结果 ⇒ 第一轮只能调 get_weather。
    # 这类"多步依赖"样本用来测模型会不会一次性把还没数据的调用也发出来
    ("查上海天气，并把结果翻译成英文", (
        {"name": "get_weather", "arguments": {"city": "上海"}},
    ), "single", ("weather", "multi_step_dependency")),

    # ── no_call_needed：不该调任何工具（测误调率）────────────────
    ("你好", (), "no_call_needed", ("chitchat",)),
    ("用一句话解释什么是递归", (), "no_call_needed", ("explain",)),
    ("帮我写一首关于秋天的五言绝句", (), "no_call_needed", ("creative",)),
    ("Python 里 list 和 tuple 有什么区别", (), "no_call_needed", ("explain",)),
    ("把下面这段话改得更正式一点：这个方案还行", (), "no_call_needed", ("rewrite",)),
    ("1 加 1 等于几", (), "no_call_needed", ("trivial_math",)),
    ("你觉得本地部署大模型值不值得", (), "no_call_needed", ("opinion",)),
    ("解释一下什么是注意力机制", (), "no_call_needed", ("explain",)),
    ("帮我给这封邮件想三个标题", (), "no_call_needed", ("creative",)),
    ("2 的 3 次方是多少", (), "no_call_needed", ("trivial_math",)),
    ("什么是量化？简单说说", (), "no_call_needed", ("explain",)),
    ("把「你好世界」写成 Python 代码", (), "no_call_needed", ("code",)),
    ("总结一下我们刚才聊的内容", (), "no_call_needed", ("meta",)),
    ("谢谢你的帮助", (), "no_call_needed", ("chitchat",)),
    ("REST 和 GraphQL 各自的优缺点是什么", (), "no_call_needed", ("explain",)),
    ("给我讲讲 TCP 三次握手", (), "no_call_needed", ("explain",)),
    ("这段代码有什么问题：def f(a): return a", (), "no_call_needed", ("code",)),
    ("写一个快速排序的伪代码", (), "no_call_needed", ("code",)),
    ("10 以内有哪些质数", (), "no_call_needed", ("trivial_math",)),
    ("什么是过拟合，怎么避免", (), "no_call_needed", ("explain",)),
    ("帮我把这段话分成三点", (), "no_call_needed", ("rewrite",)),
    ("你能做什么", (), "no_call_needed", ("meta",)),
    ("解释一下 CAP 定理", (), "no_call_needed", ("explain",)),
    ("5 的平方是多少", (), "no_call_needed", ("trivial_math",)),
    ("给这个函数起个更好的名字：def calc(d)", (), "no_call_needed", ("code",)),

    # ── args：工具好选，参数难填 ──────────────────────────────────
    ("查询 2026-10-03 那天的订单", ({"name": "db_query", "arguments": {"sql": "SELECT * FROM orders WHERE date = '2026-10-03'"}},), "args", ("db", "date")),
    ("把 3.5 千克换算一下，我要精确值", ({"name": "calculator", "arguments": {"expr": "3.5"}},), "args", ("math", "numeric")),
    ("纽约现在的天气，我要华氏度", ({"name": "get_weather", "arguments": {"city": "纽约", "unit": "fahrenheit"}},), "args", ("weather", "enum")),
    ("搜索结果给我 3 条就行，关键词是量化交易", ({"name": "search_web", "arguments": {"query": "量化交易", "max_results": 3}},), "args", ("search", "numeric")),
    ("1200 港币换成日元", ({"name": "convert_currency", "arguments": {"amount": 1200, "from": "HKD", "to": "JPY"}},), "args", ("currency", "enum")),
    ("把 README 的第一段翻译成英文", ({"name": "translate", "arguments": {"text": "README 的第一段", "target_lang": "en"}},), "args", ("translate", "enum")),
    ("现在 UTC-5 是几点", ({"name": "time_now", "arguments": {"tz_offset_hours": -5, "fmt": "time"}},), "args", ("time", "numeric")),
    ("从 users 表查 50 条，按注册时间倒序", ({"name": "db_query", "arguments": {"sql": "SELECT * FROM users ORDER BY created_at DESC", "limit": 50}},), "args", ("db", "numeric")),
    ("算 (100 - 37) / 9，保留精确结果", ({"name": "calculator", "arguments": {"expr": "(100 - 37) / 9"}},), "args", ("math", "numeric")),
    ("伦敦天气，摄氏度", ({"name": "get_weather", "arguments": {"city": "伦敦", "unit": "celsius"}},), "args", ("weather", "enum")),
    ("给我 15 条关于 MoE 架构的搜索结果", ({"name": "search_web", "arguments": {"query": "MoE 架构", "max_results": 15}},), "args", ("search", "numeric")),
    ("750 人民币换美元", ({"name": "convert_currency", "arguments": {"amount": 750, "from": "CNY", "to": "USD"}},), "args", ("currency", "enum")),
    ("把这句翻成韩语……算了你只有六种语言，那就翻成中文：Hello world", ({"name": "translate", "arguments": {"text": "Hello world", "target_lang": "zh"}},), "args", ("translate", "enum", "hard")),
    ("今天的日期，ISO 格式", ({"name": "time_now", "arguments": {"fmt": "iso"}},), "args", ("time", "enum")),
    ("统计 2026 年 9 月的销售额总和", ({"name": "db_query", "arguments": {"sql": "SELECT SUM(amount) FROM sales WHERE month = '2026-09'"}},), "args", ("db", "date")),
)


def build_cases(*, seed: int = 20261003) -> list[dict[str, Any]]:
    """生成全部样本。同 seed 必然同结果。"""
    out: list[dict[str, Any]] = []
    for instruction, expected, kind, tags in CASES:
        names = _toolset_for(expected, kind)
        out.append({
            "input": {"instruction": instruction, "tools": list(names)},
            "expect": {
                "calls": [dict(item) for item in expected],
                "must_call": bool(expected),
                "kind": kind,
            },
            "tools": [_tool_spec(name) for name in names],
            "kind": kind,
            "tags": [kind, *tags],
            "meta": {"n_expected": len(expected)},
            "id": _case_id(instruction),
        })
    random.Random(seed).shuffle(out)
    for index, case in enumerate(out):
        case["ord"] = index
    return out


def _toolset_for(expected: Sequence[dict[str, Any]], kind: str) -> tuple[str, ...]:
    """样本的工具集 = 默认全集，但保证期望调用的工具一定在里面。

    刻意**不**把工具集裁到只剩正确答案：那样等于告诉模型答案，
    测出来的分数没有意义。危险工具（send_email）也必须一直在场。
    """
    names = list(DEFAULT_TOOLSET)
    for call in expected:
        if call["name"] not in names:
            names.append(call["name"])
    return tuple(names)


def _tool_spec(name: str) -> dict[str, Any]:
    definition = TOOLS[name]
    return {
        "name": name,
        "description": definition["description"],
        "parameters": definition["parameters"],
    }


def _case_id(instruction: str) -> str:
    return f"tcz-{hashlib.sha256(instruction.encode()).hexdigest()[:16]}"


def stats(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    kinds = Counter(case["kind"] for case in cases)
    return {
        "n": len(cases),
        "kinds": dict(sorted(kinds.items(), key=lambda kv: -kv[1])),
        "unique_instructions": len({case["input"]["instruction"] for case in cases}),
        "tools": len(TOOLS),
        "expected_calls": sum(case["meta"]["n_expected"] for case in cases),
    }


def write_jsonl(path: Path | str, *, seed: int = 20261003) -> Path:
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases(seed=seed)
    target.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False, sort_keys=True) for c in cases) + "\n",
        encoding="utf-8", newline="\n",
    )
    return target


BUILTIN_PATH = Path(__file__).resolve().parent / "tool_calls_zh.jsonl"

if __name__ == "__main__":  # pragma: no cover
    print(write_jsonl(BUILTIN_PATH), stats(build_cases()))
