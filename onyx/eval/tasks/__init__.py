"""内置评测任务与它们的默认数据集，以及 `onyx.tasks` 扩展点的装配。

`build_task` 是唯一的构造入口：CLI、API、runner 都走它，
所以"某个任务用哪份数据、什么参数"只有一处定义，不会前后端各写一套。

外部任务通过 entry point 注册（DESIGN §13），不改内核、不改 runner：
    [project.entry-points."onyx.tasks"]
    my_task = "my_pkg.task:MyTask"          # 或 TaskSpec(factory=..., dataset=...)
插件加载失败/形状不对会被记进台账（`onyx plugins` 可见），并跳过——
一个坏插件不该让内置任务也列不出来。
"""

from __future__ import annotations

from typing import Any

from onyx.discovery import GROUP_TASKS, discover, record_failure
from onyx.eval.datasets.loader import Dataset, load_builtin
from onyx.eval.task import EvalTask, TaskSpec, coerce_task_spec
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.eval.tasks.tool_selection import ToolSelection

#: 内置任务：task id → TaskSpec
BUILTIN_TASKS: dict[str, TaskSpec] = {
    IntentClassification.id: TaskSpec(
        IntentClassification, lambda: load_builtin("intent_zh")
    ),
    ToolSelection.id: TaskSpec(ToolSelection, lambda: load_builtin("tool_calls_zh")),
}

#: 内置数据集的规范别名（两种写法必须落到同一个 id，理由见 `load_dataset`）
_BUILTIN_DATASETS = {
    "tool_calls_zh": "tool_calls_zh", "tool_calls_zh-v1": "tool_calls_zh",
    "intent_zh": "intent_zh", "intent_zh-v1": "intent_zh",
}


def specs() -> dict[str, TaskSpec]:
    """内置 + 插件任务。坏插件跳过并记账，形状不对的插件同样跳过并记账。"""
    found: dict[str, Any] = discover(GROUP_TASKS, BUILTIN_TASKS)
    out: dict[str, TaskSpec] = {}
    for name, obj in found.items():
        try:
            out[name] = obj if isinstance(obj, TaskSpec) else coerce_task_spec(name, obj)
        except (TypeError, ValueError) as exc:
            record_failure(GROUP_TASKS, name, str(exc))
    return out


def task_ids() -> tuple[str, ...]:
    return tuple(sorted(specs()))


def builtin_dataset_names() -> tuple[str, ...]:
    """内置数据集**可接受的写法**（含 `-v1` 别名）。

    给服务侧校验用：校验必须和 `load_dataset` 认同一样的写法，
    否则会出现"CLI 能跑、界面说未知数据集"这种两边口径分裂。
    """
    return tuple(sorted(_BUILTIN_DATASETS))


def build_task(task_id: str, *, model: str, dataset: Dataset | None = None, **kw: Any) -> EvalTask:
    """构造任务实例。

    未知 task_id 直接报错并列出可选项——静默退回某个默认任务的话，
    "我跑的是哪个任务"就变成一个需要去翻代码才能回答的问题。
    """
    spec = specs().get(task_id)
    if spec is None:
        raise KeyError(f"未知任务 {task_id!r}；可选: {task_ids()}")
    resolved = dataset if dataset is not None else spec.default_dataset(task_id)
    task = spec.factory(resolved, model=model, **kw)
    # 注册名与任务自报的 id 必须一致：库里记的是 task.id，CLI/API 用的是注册名。
    # 两者不一致时同一个插件会裂成两个任务历史，且没人会看见这个裂缝。
    if task.id != task_id:
        raise KeyError(
            f"任务注册名 {task_id!r} 与其 EvalTask.id={task.id!r} 不一致；"
            "分数会落在后者名下，而用户查的是前者。请改 entry point 名或任务 id"
        )
    return task


def load_dataset(dataset_id: str | None, *, task_id: str, db: Any = None) -> Dataset:
    """按 id 载入数据集。`None` 时取该任务的默认数据集。

    三种写法按固定顺序解析，**CLI 与界面必须走这一个函数**：
    内置别名 → `file:<路径>` → 库里已登记的 id。少一层就会出现"命令行能跑、
    界面说未知数据集"这种两边口径分裂。
    """
    if dataset_id is None:
        spec = specs().get(task_id)
        if spec is None:
            raise KeyError(f"未知任务 {task_id!r}；可选: {task_ids()}")
        return spec.default_dataset(task_id)
    builtin = _BUILTIN_DATASETS.get(dataset_id)
    if builtin is not None:
        # 两种写法都必须落到**同一个规范 id**：否则同一份数据会在库里裂成两行，
        # grade 的外键指向哪一行取决于调用方怎么拼名字，历史就断成两截了。
        # 要自定义 id 请用 `onyx eval import --id`，那是显式动作。
        return load_builtin(builtin)
    if dataset_id.startswith("file:"):
        # `file:路径` 形式：从 JSONL 直接载入，不必先 import 进库
        from onyx.eval.datasets.loader import load_jsonl

        return load_jsonl(dataset_id[len("file:"):])
    if db is not None:
        # 导入过的数据集在库里。这一步之前不存在：`eval import --id x` 之后
        # `--dataset x` 会报未知 id，而它的样本明明就在同一张 DB 里。
        from onyx.eval.datasets.loader import load_registered

        return load_registered(db, dataset_id)
    raise KeyError(
        f"未知数据集 {dataset_id!r}；内置可选: {sorted(set(_BUILTIN_DATASETS))}，"
        "或用 file:<路径> 直接读 JSONL，或先用 onyx eval import 导入"
        "（导入过的 id 需要连着数据库一起解析，CLI 会自己做）"
    )


__all__ = ["BUILTIN_TASKS", "build_task", "builtin_dataset_names", "load_dataset",
           "specs", "task_ids"]
