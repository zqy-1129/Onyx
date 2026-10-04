"""扩展点在 CLI 上的出口：`onyx plugins` / `onyx eval tasks` / `--sink`。

诊断命令的价值在于"装了但没生效"必须现形，所以这里盯三件事：
1. 坏插件让命令**退出码非 0**（一个静默忽略坏插件的体检命令是负资产）；
2. 列表里能区分内建与插件，且 `onyx.graders` 明说"未接线"而不是假装支持；
3. `--sink` 写错名字必须报错退出，不能退化成"用默认的跑下去"——
   用户会以为导出真的在发生。
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from onyx.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """数据目录进 tmp：`--sink jsonl` 默认写到 data_dir，测试不许污染仓库的 .data。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "machine"))
    # rich 按终端宽度截断列：诊断表必须能整串读出来才好断言
    monkeypatch.setenv("COLUMNS", "240")
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


def test_plugins_lists_every_group_and_flags_unwired_one():
    result = _run("plugins")
    assert result.exit_code == 0, result.output
    for group in ("onyx.providers", "onyx.tasks", "onyx.sinks",
                  "onyx.tool_executors", "onyx.observers", "onyx.graders"):
        assert group in result.output
    # 没接线的组必须自己说清楚，而不是显示一行空白让人以为"装了也不会生效"
    assert "未接线" in result.output
    assert "intent_classification" in result.output and "mock" in result.output


def test_broken_plugin_makes_plugins_exit_1(entry_points_env):
    entry_points_env.install("onyx.tasks", "broken=no_such_pkg_nowhere:Thing")
    result = _run("plugins")
    assert result.exit_code == 1
    assert "broken" in result.output and "ModuleNotFoundError" in result.output


def test_plugin_override_is_marked(entry_points_env):
    """同名覆盖必须看得见：悄悄替换内置实现会让人在错误的假设上调系统。"""
    entry_points_env.install("onyx.tasks", "intent_classification=example_task.task:CharCount")
    result = _run("plugins")
    assert result.exit_code == 0
    assert "↻内置" in result.output


def test_eval_tasks_shows_the_external_task_with_its_source(entry_points_env):
    entry_points_env.install("onyx.tasks", "example_char_count=example_task.task:spec")
    result = _run("eval", "tasks")
    assert result.exit_code == 0, result.output
    assert "example_char_count" in result.output
    assert "插件" in result.output
    assert "自带" in result.output, "「自带数据集」与「需 --dataset」是可用的差别，必须显示"


def test_eval_tasks_exits_1_when_a_task_plugin_is_broken(entry_points_env):
    entry_points_env.install("onyx.tasks", "oops=example_task.task:_han")
    result = _run("eval", "tasks")
    assert result.exit_code == 1
    assert "oops" in result.output


def test_unknown_sink_fails_loudly(isolated_data_dir):
    result = _run("chat", "你好", "--provider", "mock", "--model", "mock/echo", "--sink", "nope")
    assert result.exit_code == 2
    assert "未知 sink" in result.output and "jsonl" in result.output


def test_named_sink_receives_the_event_stream(isolated_data_dir):
    """`--sink jsonl` 必须真的把事件写出去——扩展点最怕"接上了但没流量"。"""
    result = _run("chat", "说点什么", "--provider", "mock", "--model", "mock/echo",
                  "--sink", "jsonl")
    assert result.exit_code == 0, result.output
    written = (isolated_data_dir / "events.ndjson").read_text(encoding="utf-8")
    assert "trace_start" in written, "事件流必须落到 sink，而不是被静默丢弃"
    assert "trace_end" in written, "TRACE_END 缺失会让看板上的 trace 永远停在进行中"
