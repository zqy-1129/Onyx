"""扩展点发现（DESIGN §13）：六个 group 共用一套语义。

内置实现静态登记，外部包通过 entry points 注册：
    [project.entry-points."onyx.tasks"]
    my_task = "my_pkg.task:Task"

三条语义在每个 group 上都必须一致，所以只实现一次：

1. **坏插件隔离**：某个插件 import 失败只记一条 `LoadFailure` 并跳过，
   不许让看板起不来——一个坏插件不该拖垮所有能力。
2. **失败必须可见**：只隔离不报告，就等于把"插件没生效"伪装成"插件正常工作"。
   所以失败进台账，`onyx plugins` / `onyx doctor` 必须能列出来。
3. **同名覆盖**：插件覆盖内置实现（就地替换某段实现做实验是刻意留的口子），
   但覆盖关系要能被 `report()` 查出来，不能悄悄发生。

发现只走 `importlib.metadata`，**不扫描目录、不写死 if/else**。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

log = logging.getLogger("onyx.plugins")

#: DESIGN §13 的六组扩展点（组名是公开契约，外部包按它注册）
GROUP_PROVIDERS = "onyx.providers"
GROUP_TASKS = "onyx.tasks"
GROUP_GRADERS = "onyx.graders"
GROUP_SINKS = "onyx.sinks"
GROUP_TOOL_EXECUTORS = "onyx.tool_executors"
GROUP_OBSERVERS = "onyx.observers"

GROUPS: tuple[str, ...] = (
    GROUP_PROVIDERS,
    GROUP_TASKS,
    GROUP_GRADERS,
    GROUP_SINKS,
    GROUP_TOOL_EXECUTORS,
    GROUP_OBSERVERS,
)


@dataclass(slots=True)
class LoadFailure:
    """一个插件加载失败。`attempts` 记录它被尝试了几次（注册表会被反复查询）。"""

    group: str
    name: str
    error: str
    attempts: int = 1

    def __str__(self) -> str:
        again = f"（累计 {self.attempts} 次）" if self.attempts > 1 else ""
        return f"{self.group} / {self.name}: {self.error}{again}"


_FAILURES: dict[tuple[str, str], LoadFailure] = {}
_ENTRIES: dict[str, tuple[EntryPoint, ...]] = {}
_LOADED: dict[tuple[str, str], Any] = {}


def _record(group: str, name: str, text: str) -> None:
    key = (group, name)
    existing = _FAILURES.get(key)
    if existing is None:
        _FAILURES[key] = LoadFailure(group=group, name=name, error=text)
    else:
        existing.attempts += 1
    log.warning("插件不可用: %s / %s: %s", group, name, text)


def _record_failure(group: str, name: str, exc: BaseException) -> None:
    _record(group, name, f"{type(exc).__name__}: {exc}")


def record_failure(group: str, name: str, message: str) -> None:
    """登记一个"加载成功但形状不对"的插件。

    形状错误也必须进台账：一个 `onyx.tasks` 插件返回了不是 `EvalTask` 的东西，
    如果只 log 一行就消失，用户看到的是"任务列表里没有我的任务"，
    而不是"你的插件注册失败，原因是 X"。
    """
    _record(group, name, message)


def failures() -> tuple[LoadFailure, ...]:
    """当前所有加载失败，按 (group, name) 排序。"""
    return tuple(sorted(_FAILURES.values(), key=lambda f: (f.group, f.name)))


def reset_failures() -> None:
    """清空失败台账。只给测试用——台账是进程级的，测试之间必须互不污染。"""
    _FAILURES.clear()


def plugin_entries(group: str) -> tuple[EntryPoint, ...]:
    """该 group 下声明的外部插件（未加载）。发现机制本身出错时返回空而不是抛。

    结果按 group 缓存：`entry_points()` 每次约 6ms，而注册表会在热路径上被反复问
    （每次工具调用、每个 API 请求）。安装包不会在进程运行期间变化，所以缓存是安全的；
    测试改了 entry points 需要 `clear_cache()`。
    """
    cached = _ENTRIES.get(group)
    if cached is not None:
        return cached
    try:
        found = tuple(entry_points(group=group))
    except Exception as exc:  # noqa: BLE001 - 老版本 importlib.metadata 签名不同，退化即可
        log.warning("entry point 发现失败 (%s): %s", group, exc)
        found = ()
    _ENTRIES[group] = found
    return found


def clear_cache() -> None:
    """丢弃 entry point 缓存与已加载值。只给测试用（或装/卸插件后的显式刷新）。"""
    _ENTRIES.clear()
    _LOADED.clear()


def discover[T](group: str, builtin: Mapping[str, T]) -> dict[str, T]:
    """内置 + 插件。同名时插件覆盖内置。加载失败的插件被跳过并记账。

    加载成功的值按 (group, name) 缓存；失败**不**缓存，所以台账里的 attempts
    会随每次查询累加——"插件坏了"这件事必须一直看得见。
    """
    found: dict[str, Any] = dict(builtin)
    for entry in plugin_entries(group):
        key = (group, entry.name)
        if key in _LOADED:
            found[entry.name] = _LOADED[key]
            continue
        try:
            value = entry.load()
        except Exception as exc:  # noqa: BLE001 - 坏插件隔离：不许拖垮整个注册表
            _record_failure(group, entry.name, exc)
            continue
        _LOADED[key] = value
        found[entry.name] = value
    return found  # type: ignore[return-value]


def plugin_names(group: str) -> tuple[str, ...]:
    return tuple(sorted(e.name for e in plugin_entries(group)))


@dataclass(frozen=True, slots=True)
class GroupReport:
    """一个扩展点组的实际装配结果——`onyx plugins` 就展示这张表。"""

    group: str
    builtin: tuple[str, ...]
    plugin: tuple[str, ...]
    overrides: tuple[str, ...]
    failures: tuple[LoadFailure, ...]

    @property
    def total(self) -> int:
        return len(set(self.builtin) | set(self.plugin))


def report(group: str, builtin: Iterable[str]) -> GroupReport:
    """不加载插件，只报告"谁会注册进来、谁覆盖了内置、谁已经坏了"。

    刻意不调 `discover()`：诊断命令不应该为了列出信息而触发所有插件 import。
    """
    builtins = tuple(sorted(builtin))
    entries = tuple(sorted(e.name for e in plugin_entries(group)))
    return GroupReport(
        group=group,
        builtin=builtins,
        plugin=entries,
        overrides=tuple(sorted(set(entries) & set(builtins))),
        failures=tuple(f for f in failures() if f.group == group),
    )


__all__ = [
    "GROUPS",
    "GROUP_GRADERS",
    "GROUP_OBSERVERS",
    "GROUP_PROVIDERS",
    "GROUP_SINKS",
    "GROUP_TASKS",
    "GROUP_TOOL_EXECUTORS",
    "GroupReport",
    "LoadFailure",
    "clear_cache",
    "discover",
    "failures",
    "plugin_entries",
    "plugin_names",
    "record_failure",
    "report",
    "reset_failures",
]
