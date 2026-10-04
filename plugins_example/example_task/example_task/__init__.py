"""Onyx 扩展点样板：外部评测任务插件。

注册方式只有一条：在 `pyproject.toml` 里声明 entry point
    [project.entry-points."onyx.tasks"]
    example_char_count = "example_task.task:spec"

装好后 `onyx eval tasks` 就能看到它，`onyx eval run --task example_char_count
--provider mock` 就能跑通 —— 不改 `onyx/eval/runner.py`、不改内核
（这条由 `scripts/check_extension_boundary.py` 断言）。

这个任务本身不重要（数汉字数），重要的是它演示了三件必须守住的事：
`requires` 能力位、`metric_names` 与 `aggregate` 同源、内容/格式两个维度正交。
"""

from .task import CharCount, spec

__all__ = ["CharCount", "spec"]
