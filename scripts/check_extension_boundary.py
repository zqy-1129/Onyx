"""扩展点边界检查（M6 出口 DoD 的机器断言）。

计划的验收条件是：**接入新 provider 不许改 `onyx/core/**`、`onyx/llm/gateway.py`、
`onyx/obs/**`；新任务不许改 `onyx/eval/runner.py`**——而且"用脚本断言，不靠人 review"。
人 review 的失效方式很具体：接第二个 provider 时"顺手"在 gateway 里加一个
`if kind == "openai_compat"`，测试全绿、看板全对，抽象却已经死了。

用法
    uv run python scripts/check_extension_boundary.py                  # 检查 HEAD 这一个提交
    uv run python scripts/check_extension_boundary.py --base HEAD~3 --head HEAD
    uv run python scripts/check_extension_boundary.py --staged          # 提交前自查
    uv run python scripts/check_extension_boundary.py --files a.py b.py # 纯函数模式（测试用）

退出码 0 = 通过；1 = 违规（打印是哪个实现接入触发了哪条保护规则）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: 注册点：扩展机制本身。改它们 = "在接线"，不是"在为一个实现开小灶"，所以允许改。
REGISTRATION_POINTS = (
    "onyx/discovery.py",
    "onyx/llm/registry.py",
    "onyx/llm/providers/base.py",
    "onyx/llm/providers/__init__.py",
    "onyx/eval/task.py",
    "onyx/eval/tasks/__init__.py",
    "onyx/store/sinks/__init__.py",
    "onyx/store/sinks/base.py",
    "onyx/store/sinks/registry.py",
    "onyx/tools/executors/__init__.py",
    "onyx/obs/visitors/__init__.py",
)

#: 接入某个实现时**必须保持不变**的内核文件/目录 → 为什么 Protected
PROTECTED: tuple[tuple[str, str], ...] = (
    ("onyx/core/", "L0 领域层零三方依赖，是所有口径的地基；插件需要改它 = 抽象装不下这个实现"),
    ("onyx/llm/gateway.py", "唯一调用喉咙。为某个 provider 加 if 分支，就等于承认扩展点失败了"),
    ("onyx/eval/runner.py", "评测调度内核（锁/续跑/落库）。新任务需要改它 = 任务契约不够用"),
    ("onyx/obs/engine.py", "观测引擎的事件循环；新 provider/visitor 不该需要动它"),
    ("onyx/obs/visitors/token.py", "内建对账实现；换 provider 不该改变 token 口径"),
    ("onyx/obs/visitors/tool.py", "内建工具观测；换 provider 不该改变工具判定"),
    ("onyx/obs/visitors/cost.py", "内建成本模型；与具体引擎无关"),
    ("onyx/obs/visitors/anomaly.py", "内建异常规则；与具体引擎无关"),
    ("onyx/eval/metrics.py", "指标定义（CI/F1 口径）；新任务不该重定义什么叫 F1"),
)

#: 哪些路径算"新增一个实现"（而不是"改注册机制"）
IMPL_PREFIXES = (
    "onyx/llm/providers/",
    "onyx/eval/tasks/",
    "onyx/tools/executors/",
    "onyx/store/sinks/",
    "onyx/obs/visitors/",
    "plugins_example/",
    "onyx/plugins_example/",
)


def is_implementation(path: str) -> bool:
    """这个文件是"某个扩展点的一个实现"吗？"""
    normalized = path.replace("\\", "/")
    if normalized in REGISTRATION_POINTS:
        return False
    return normalized.startswith(IMPL_PREFIXES)


def violations(changed: list[str]) -> list[tuple[str, str, str]]:
    """返回 (触发的实现文件, 被改的内核文件, 原因)。"""
    impls = sorted({p for p in changed if is_implementation(p)})
    if not impls:
        return []
    out: list[tuple[str, str, str]] = []
    for prefix, reason in PROTECTED:
        hit = sorted(p for p in changed if p.replace("\\", "/").startswith(prefix))
        if hit:
            out.append((impls[0], hit[0], reason))
    return out


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=str(REPO), capture_output=True, text=True, encoding="utf-8"
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} 失败: {result.stderr.strip()}")
    return result.stdout


def changed_files(base: str | None, head: str | None, staged: bool) -> list[str]:
    if staged:
        out = _git("diff", "--cached", "--name-only")
    elif base and head:
        out = _git("diff", "--name-only", f"{base}..{head}")
    elif base:
        out = _git("diff", "--name-only", f"{base}..HEAD")
    else:
        out = _git("diff", "--name-only", "HEAD~1..HEAD")
    return [line.strip() for line in out.splitlines() if line.strip()]


def _force_utf8() -> None:
    """Windows 中文环境下 stdout 默认 gbk，`✓ ✗` 会直接 UnicodeEncodeError。
    与 `onyx.cli.main()` 同一处理——检查脚本自己崩掉等于没有检查。"""
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="起始 ref，默认 HEAD~1")
    parser.add_argument("--head", help="结束 ref")
    parser.add_argument("--staged", action="store_true", help="检查已暂存但未提交的变化")
    parser.add_argument("--files", nargs="*", help="直接给文件列表（跳过 git，测试用）")
    parser.add_argument("--verbose", action="store_true", help="打印检查了哪些文件")
    args = parser.parse_args(argv)

    changed = (
        [f.replace("\\", "/") for f in args.files]
        if args.files is not None
        else changed_files(args.base, args.head, args.staged)
    )
    impls = [p for p in changed if is_implementation(p)]
    found = violations(changed)

    if args.verbose:
        print(f"变更 {len(changed)} 个文件，其中实现文件 {len(impls)} 个: {impls}")
        print(f"保护规则 {len(PROTECTED)} 条，注册点 {len(REGISTRATION_POINTS)} 个")

    if not impls:
        print("✓ 没有新增扩展点实现，跳过（本次变更只是改内核或改测试）")
        return 0
    if found:
        print(f"✗ 抽象泄漏：接入实现时修改了内核，共 {len(found)} 处")
        for impl, protected, reason in found:
            print(f"  · 接入 {impl} 时改了 {protected}")
            print(f"    为什么不许：{reason}")
        print("  处理：把改动放进插件/实现文件里；若确实装不下，"
              "记 issue 并在 DESIGN §13 补契约，不要在 gateway 里加 if 分支。")
        return 1
    print(f"✓ 接入 {len(impls)} 个实现未触碰受保护的内核文件")
    return 0


if __name__ == "__main__":
    sys.exit(main())
