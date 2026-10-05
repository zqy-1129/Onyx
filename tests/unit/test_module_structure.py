"""模块结构门禁：一个模块里不许出现同名的顶层定义。

这条规则是从两次真实事故长出来的：

1. `onyx/cli.py` 里曾有**两份 `_fmt`**，后一份静默遮蔽前一份。它不报错、不崩溃在写法上，
   只在"数据恰好是某种形状"时炸（`None` 提前返回「—」所以一直没炸），而 ruff 的 F811
   因为"前一份被使用过"也不报。
2. S31 新写的 `instruction_following.py` 又出现一次：两份 `_looks_like_refusal`，
   早期那份还写着 `len(_visible(text))`，而 `_visible` 返回的已经是整数——
   只要有人调用到被遮蔽的那一份就会 TypeError。全量测试是绿的，因为 Python 取后一份。

⇒ 同名遮蔽是**模块级**的问题，不该靠人记住"上次那个文件踩过"，所以这里对 `onyx/` 全量断言。
"""

from __future__ import annotations

import ast
from pathlib import Path

ONYX_ROOT = Path(__file__).resolve().parents[2] / "onyx"

#: `@overload` 签名天生就是同名重复，它们不是遮蔽而是分发
_ALLOWED_REPEATS = {"overload"}


def _top_level_defs(source: str) -> dict[str, list[int]]:
    tree = ast.parse(source)
    seen: dict[str, list[int]] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        decorators = {
            name for name in (_decorator_label(item) for item in node.decorator_list) if name
        }
        if decorators & _ALLOWED_REPEATS:
            continue
        seen.setdefault(node.name, []).append(node.lineno)
    return seen


def _decorator_label(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_no_module_shadows_a_top_level_definition():
    offenders: list[str] = []
    for path in sorted(ONYX_ROOT.rglob("*.py")):
        dupes = {
            name: lines
            for name, lines in _top_level_defs(path.read_text(encoding="utf-8")).items()
            if len(lines) > 1
        }
        if dupes:
            offenders.append(f"{path.relative_to(ONYX_ROOT.parent)}: {dupes}")
    assert not offenders, (
        "这些模块里有同名顶层定义，后一份会静默遮蔽前一份：\n" + "\n".join(offenders)
    )


def test_the_guard_itself_reacts_to_a_shadowed_name():
    """这条断言必须能被抓坏：注入一份同名定义，结构检查要立刻点名它。"""
    dupes = _top_level_defs("def _a() -> int: return 1\n\n\ndef _a() -> str: return 'x'\n")
    assert list(dupes) == ["_a"] and len(dupes["_a"]) == 2, dupes
    overloaded = _top_level_defs(
        "from typing import overload\n"
        "@overload\ndef _b(x: int) -> int: ...\n"
        "@overload\ndef _b(x: str) -> str: ...\n"
        "def _b(x): return x\n"
    )
    assert "overload" not in overloaded, "@overload 是分发，不是遮蔽"
    assert len(overloaded.get("_b", [])) == 1, "只该记下真正实现那一份"
