"""计算器工具：AST 白名单求值。

**绝不用 `eval()`**：参数来自模型输出，`eval("__import__('os').system(...)")` 就是 RCE。
这里只接受算术表达式节点，其余（属性访问、下标、名字、lambda、推导式…）一律拒绝。
"""

from __future__ import annotations

import ast
import math
from typing import Any

from onyx.core.errors import ToolArgError

#: 允许的二元/一元运算
_ALLOWED_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: pow,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Any] = {
    ast.USub: lambda a: -a,
    ast.UAdd: lambda a: +a,
}
#: 允许的函数与常量
_ALLOWED_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp, "floor": math.floor,
    "ceil": math.ceil, "pow": pow,
}
_ALLOWED_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}

#: 防止 2**999999999 这类表达式把 CPU/内存打满
MAX_EXPONENT = 1024
MAX_NODES = 200
MAX_EXPR_CHARS = 500


def calculate(expr: str) -> dict[str, Any]:
    if not isinstance(expr, str):
        raise ToolArgError(
            f"expr 必须是字符串，实际 {type(expr).__name__}",
            detail={"kind": "bad_expr", "actual": type(expr).__name__},
        )
    text = expr.strip()
    if not text:
        raise ToolArgError("expr 不能为空", detail={"kind": "bad_expr"})
    if len(text) > MAX_EXPR_CHARS:
        raise ToolArgError(
            f"表达式过长（{len(text)} > {MAX_EXPR_CHARS}）",
            detail={"kind": "bad_expr", "chars": len(text), "limit": MAX_EXPR_CHARS},
        )
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ToolArgError(
            f"表达式语法错误: {exc.msg}",
            detail={"kind": "bad_expr", "expr": text[:200]},
        ) from exc

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_NODES:
        raise ToolArgError(
            f"表达式过于复杂（{len(nodes)} 个节点 > {MAX_NODES}）",
            detail={"kind": "bad_expr", "nodes": len(nodes), "limit": MAX_NODES},
        )

    try:
        value = _eval(tree.body)
    except ToolArgError as exc:
        # 统一带上 kind 与原文：评测要把"模型写的表达式哪里不合法"拆开统计，
        # 只有一句中文描述没法聚合
        raise ToolArgError(
            exc.message,
            detail={"kind": "unsafe_expression", "expr": text[:200], **(exc.detail or {})},
        ) from exc
    if isinstance(value, float):
        # 统一收敛到有限精度，避免 0.1+0.2 这类浮点噪声在评测里造成假失败
        value = round(value, 12)
    return {"expr": text, "value": value}


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool | int | float):
            return node.value
        raise ToolArgError(f"不允许的字面量类型: {type(node.value).__name__}")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return _ALLOWED_CONSTS[node.id]
        raise ToolArgError(f"不允许的标识符: {node.id!r}")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ToolArgError(f"不允许的运算符: {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and _exponent_too_large(right):
            raise ToolArgError(f"指数过大（>{MAX_EXPONENT}），拒绝计算以防资源耗尽")
        _require_numeric(left, right)
        try:
            return op(left, right)
        except ZeroDivisionError as exc:
            raise ToolArgError("除零") from exc
        except (OverflowError, ValueError) as exc:
            raise ToolArgError(f"数值错误: {exc}") from exc
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolArgError(f"不允许的一元运算符: {type(node.op).__name__}")
        operand = _eval(node.operand)
        _require_numeric(operand)
        return op(operand)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ToolArgError("只允许调用白名单内的数学函数")
        if node.keywords:
            raise ToolArgError("不允许关键字参数")
        args = [_eval(a) for a in node.args]
        for arg in args:
            _require_numeric(arg)
        try:
            return _ALLOWED_FUNCS[node.func.id](*args)
        except (ValueError, OverflowError, ZeroDivisionError, TypeError) as exc:
            raise ToolArgError(f"函数调用失败: {exc}") from exc
    raise ToolArgError(f"不允许的表达式节点: {type(node).__name__}")


def _require_numeric(*values: Any) -> None:
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ToolArgError(f"只支持数值运算，收到 {type(value).__name__}")


def _exponent_too_large(exponent: Any) -> bool:
    return isinstance(exponent, int | float) and not isinstance(exponent, bool) \
        and abs(exponent) > MAX_EXPONENT
