"""内置评测任务与它们的默认数据集。

`build_task` 是唯一的构造入口：CLI、API、runner 都走它，
所以"某个任务用哪份数据、什么参数"只有一处定义，不会前后端各写一套。
S16 会在这里接 entry points，让第三方任务不改内核就能注册。
"""

from __future__ import annotations

from typing import Any

from onyx.eval.datasets.loader import Dataset, load_builtin
from onyx.eval.task import EvalTask
from onyx.eval.tasks.intent_classification import IntentClassification

#: task id → (构造函数, 默认数据集载入器)
TASKS: dict[str, dict[str, Any]] = {
    IntentClassification.id: {"factory": IntentClassification, "dataset": load_builtin},
}


def task_ids() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


def build_task(task_id: str, *, model: str, dataset: Dataset | None = None, **kw: Any) -> EvalTask:
    """构造任务实例。

    未知 task_id 直接报错并列出可选项——静默退回某个默认任务的话，
    "我跑的是哪个任务"就变成一个需要去翻代码才能回答的问题。
    """
    entry = TASKS.get(task_id)
    if entry is None:
        raise KeyError(f"未知任务 {task_id!r}；可选: {task_ids()}")
    resolved = dataset if dataset is not None else entry["dataset"]()
    return entry["factory"](resolved, model=model, **kw)


def load_dataset(dataset_id: str | None, *, task_id: str) -> Dataset:
    """按 id 载入数据集。`None` 时取该任务的默认数据集。"""
    if dataset_id is None:
        entry = TASKS.get(task_id)
        if entry is None:
            raise KeyError(f"未知任务 {task_id!r}；可选: {task_ids()}")
        return entry["dataset"]()
    if dataset_id in {"intent_zh", "intent_zh-v1"}:
        # 两种写法都必须落到**同一个规范 id**：否则同一份数据会在库里裂成两行，
        # grade 的外键指向哪一行取决于调用方怎么拼名字，历史就断成两截了。
        # 要自定义 id 请用 `onyx eval import --id`，那是显式动作。
        return load_builtin("intent_zh")
    if dataset_id.startswith("file:"):
        # `file:路径` 形式：从 JSONL 直接载入，不必先 import 进库
        from onyx.eval.datasets.loader import load_jsonl

        return load_jsonl(dataset_id[len("file:"):])
    raise KeyError(
        f"未知数据集 {dataset_id!r}；内置可选: intent_zh，"
        "或用 file:<路径> 直接读 JSONL，或先用 onyx eval import 导入"
    )


__all__ = ["TASKS", "build_task", "load_dataset", "task_ids"]
