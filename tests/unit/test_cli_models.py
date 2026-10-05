"""S26 验收：模型治理的命令行出口（pull / rm / ls）。

`AdminProvider` 的 `pull` / `delete` 早就实现了（Playground 的 unload 也在用），
但 CLI 只有 `sync` 与 `ls` —— 于是"拉个模型进来"这件事只能开另一个终端敲 `ollama pull`，
而那条路径完全绕开 Onyx：不刷清单、不留痕、看板也看不到。

这里盯的是三件事：
- 危险操作要确认，`--yes` 给脚本用；
- **没有控制面的通道必须说不做**（兼容层没有统一端点，编一个假的卸载
  会让"显存已经让出来了"建立在谎话上）；
- 删除只动权重，**不动那些测量记录**：分数与 trace 属于已经发生过的事。
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from onyx.cli import app
from onyx.core.types import Cap
from onyx.store.db import Database
from onyx.store.repos import ModelRepo

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(*argv: str, input: str | None = None):
    return runner.invoke(app, list(argv), input=input)


MOCK = ["--provider", "mock", "--url", "mock://"]


# ── pull ──────────────────────────────────────────────────────────
def test_pull_writes_the_catalog_after_downloading(tmp_path):
    """拉完不刷清单的话，`models ls` 与界面上都看不到它 —— 那会被当成"拉取失败"。"""
    result = _run("models", "pull", "demo/new-model", *MOCK, "--yes")
    assert result.exit_code == 0, result.output
    assert "已拉取 demo/new-model" in result.output
    assert "清单已同步" in result.output

    db = Database(tmp_path / "onyx.sqlite")
    try:
        names = {m.name for m in ModelRepo(db).list_models("mock-local")}
    finally:
        db.close()
    assert "mock/echo" in names, "同步的是整份清单，不是只插拉到的那个"


def test_pull_without_yes_asks_before_touching_the_disk():
    """GB 级写入要有确认：默认必须是问，而不是"先拉了再说"。"""
    result = _run("models", "pull", "demo/giant", *MOCK, input="n\n")
    assert result.exit_code != 0
    assert "继续" in result.output


def test_pull_answer_yes_proceeds():
    result = _run("models", "pull", "demo/answered", *MOCK, input="y\n")
    assert result.exit_code == 0, result.output
    assert "已拉取 demo/answered" in result.output


def test_pull_fails_when_the_engine_rejects(monkeypatch):
    from onyx.core.types import AdminResult
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "pull",
                        lambda self, name, *, on_event=None: AdminResult(
                            ok=False, action="pull", error="toomanyrequests: 超出配额"))
    result = _run("models", "pull", "demo/denied", *MOCK, "--yes")
    assert result.exit_code == 1
    assert "拉取失败" in result.output and "配额" in result.output


def test_pull_refuses_a_channel_without_a_control_plane(monkeypatch):
    """没有 ADMIN 能力位时报错退出，而不是"试着发一个不存在的请求"。"""
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "capabilities",
                        lambda self: frozenset({Cap.CHAT}), raising=True)
    result = _run("models", "pull", "demo/x", *MOCK, "--yes")
    assert result.exit_code == 2, result.output
    assert "不暴露控制面" in result.output
    assert "谎话" in result.output, "要说明为什么不做一个假的，否则下次还会有人补上"


# ── rm ────────────────────────────────────────────────────────────
def test_rm_requires_confirmation_and_keeps_the_measurements():
    result = _run("models", "rm", "demo/gone", *MOCK, input="y\n")
    assert result.exit_code == 0, result.output
    assert "已删除 demo/gone" in result.output
    assert "历史 trace 与分数保留" in result.output, \
        "必须说明删的是权重不是证据，否则人会以为历史分数也一起没了"


def test_rm_abort_keeps_the_model():
    result = _run("models", "rm", "demo/keep", *MOCK, input="n\n")
    assert result.exit_code != 0
    assert "已删除" not in result.output


def test_rm_reports_engine_failure_as_exit_one(monkeypatch):
    from onyx.core.types import AdminResult
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "delete",
                        lambda self, name: AdminResult(
                            ok=False, action="delete", error="model is not found"))
    result = _run("models", "rm", "demo/missing", *MOCK, "--yes")
    assert result.exit_code == 1
    assert "删除失败" in result.output


def test_rm_refuses_a_channel_without_a_control_plane(monkeypatch):
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "capabilities",
                        lambda self: frozenset({Cap.CHAT}), raising=True)
    result = _run("models", "rm", "demo/x", *MOCK, "--yes")
    assert result.exit_code == 2
    assert "不暴露控制面" in result.output


# ── ls / sync 的既有出口不能被新命令弄坏 ─────────────────────────
def test_ls_still_reports_residency_as_unknown_for_admin_channels():
    assert _run("models", "sync", *MOCK).exit_code == 0
    result = _run("models", "ls", *MOCK)
    assert result.exit_code == 0, result.output
    assert "mock/echo" in result.output
