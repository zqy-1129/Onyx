"""内置工具的**定义**（与实现分开放）。

实现是普通函数，定义才是发给模型的东西——两者的失误模式完全不同，
所以定义单独成模块，好让 CLI、契约测试、评测用例共享同一份，
而不是各写一遍然后在某个版本悄悄漂移。

这里的三个定义都刻意做到审计零发现：它们同时是「工具该怎么写」的参照样本。
"""

from __future__ import annotations

from onyx.tools.spec import SideEffect, ToolDef, ToolKind

ECHO = ToolDef(
    name="echo",
    description="原样回显给定的文本。用于工具调用链路的连通性自检，不访问任何外部资源。",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "需要回显的文本内容；超过 4000 字符的部分会被截断。",
            },
            "times": {
                "type": "integer",
                "description": "重复次数，有效范围 1 到 10，超出范围会被夹到最近的边界。",
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    kind=ToolKind.PYTHON_FN,
    side_effect=SideEffect.READ,
    impl_ref="onyx.tools.builtin.echo:echo",
    tags=("diagnostic", "deterministic"),
    owner="onyx",
    timeout_ms=2_000,
    doc="确定性且幂等，是契约断言 read_is_idempotent 的参照实现。",
    examples=(
        {
            "instruction": "把 hello 原样回显一次",
            "expect": {"name": "echo", "arguments": {"text": "hello", "times": 1}},
        },
    ),
)

CALCULATOR = ToolDef(
    name="calculator",
    description=(
        "计算一个算术表达式的数值结果。只支持四则运算、幂、取余和白名单内的数学函数，"
        "不会执行任何代码，也不能访问变量。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "description": (
                    "待求值的算术表达式，例如 '2+3*4' 或 'sqrt(2)*pi'。"
                    "不允许标识符、属性访问、下标或任何函数调用之外的语法。"
                ),
            },
        },
        "required": ["expr"],
        "additionalProperties": False,
    },
    kind=ToolKind.PYTHON_FN,
    side_effect=SideEffect.READ,
    impl_ref="onyx.tools.builtin.calculator:calculate",
    tags=("math", "deterministic"),
    owner="onyx",
    timeout_ms=2_000,
    doc="求值走 AST 白名单，绝不使用 eval()：参数来自模型输出，等价于不可信输入。",
    examples=(
        {
            "instruction": "算一下 (12 + 8) 乘以 3 等于多少",
            "expect": {"name": "calculator", "arguments": {"expr": "(12+8)*3"}},
        },
    ),
)

TIME_NOW = ToolDef(
    name="time_now",
    description=(
        "返回当前的日期与时间。当需要知道「现在几点」「今天几号」或做时间换算时使用它，"
        "不要凭训练数据猜测当前时间。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "tz_offset_hours": {
                "type": "number",
                "description": "相对 UTC 的时区偏移小时数，例如东八区填 8；取值范围 ±24。",
            },
            "fmt": {
                "type": "string",
                "description": "输出格式，可选 iso（默认）、date、time、unix 四种。",
                "enum": ["iso", "date", "time", "unix"],
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    kind=ToolKind.PYTHON_FN,
    side_effect=SideEffect.READ,
    impl_ref="onyx.tools.builtin.time_now:time_now",
    # non-deterministic：不能当契约测试的 sample，read_is_idempotent 必然失败
    tags=("time", "non-deterministic"),
    owner="onyx",
    timeout_ms=1_000,
    doc="非确定性工具的样本，用来验证 deterministic 标记与幂等断言的边界。",
    examples=(
        {
            "instruction": "现在东八区几点了",
            "expect": {"name": "time_now", "arguments": {"tz_offset_hours": 8, "fmt": "time"}},
        },
    ),
)

#: 全部内置定义。CLI `tools import --builtin` / 契约测试 / 评测用例共用这一份
BUILTIN_DEFS: tuple[ToolDef, ...] = (ECHO, CALCULATOR, TIME_NOW)

#: 契约测试的默认 sample：read 类、确定性、含 required 字段
CONTRACT_SAMPLE = ECHO
CONTRACT_SAMPLE_ARGS: dict[str, object] = {"text": "onyx", "times": 1}


def builtin_def(name: str) -> ToolDef | None:
    return next((item for item in BUILTIN_DEFS if item.name == name), None)


def builtin_names() -> tuple[str, ...]:
    return tuple(item.name for item in BUILTIN_DEFS)
