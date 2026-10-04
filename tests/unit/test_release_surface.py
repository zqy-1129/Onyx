"""发行面的一致性：版本号、CHANGELOG、CI 门禁清单。

这三样东西的共同点是"错了不会立刻炸"：版本对不上、CHANGELOG 漏了一版、
CI 里被删掉一步——都只会在很久以后以"没人说得清当时跑的是哪套代码"的形式暴露。
所以把它们钉成断言，而不是写在文档里靠人自觉。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

import onyx

ROOT = Path(__file__).resolve().parents[2]


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


# ── 版本号只有一处来源 ─────────────────────────────────────────────
def test_version_is_defined_in_exactly_one_place():
    """`pyproject` 里再写一份版本号，迟早会和 `onyx.__version__` 说两套话。

    看板上显示的版本、`eval_run.app_version` 落的版本、装出来的包的版本必须是同一个数——
    配对回归的前提就是"这两次跑的是同一份代码"。
    """
    project = _pyproject()["project"]

    assert "version" not in project, "静态 version 会让两处来源各说各话"
    assert "version" in project.get("dynamic", []), "要 dynamic，由 hatchling 从代码里读"
    assert _pyproject()["tool"]["hatch"]["version"]["path"] == "onyx/__init__.py"


def test_version_number_is_pep440_and_major_is_deliberately_zero():
    assert re.fullmatch(r"\d+\.\d+\.\d+(rc\d+)?", onyx.__version__), onyx.__version__
    # 未发行前 MAJOR 停在 0：这是政策，不是忘了升
    assert onyx.__version__.startswith("0."), "对外发行之前不该出现 1.x"


def test_changelog_documents_the_current_version():
    """升版本必须同时写清"这版给用户带来了什么"，两者绑在一个提交里。"""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert f"## [{onyx.__version__}]" in text, f"CHANGELOG 里没有 v{onyx.__version__} 的条目"
    assert "### Added" in text and "### Fixed" in text
    assert "PATCH" in text, "版本策略要写下来，否则每次升哪一位都是临场决定"


# ── CI 必须真的跑那五道门 ──────────────────────────────────────────
@pytest.fixture(scope="module")
def workflow_text() -> str:
    path = ROOT / ".github" / "workflows" / "ci.yml"
    assert path.exists(), "CI 工作流不见了"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("command", [
    "ruff check",
    "lint-imports",
    "check_extension_boundary.py",
    "coverage run -m pytest",
    "coverage report",
    "tsc --noEmit",
    "vitest run",
    "vite build",
    "uv build",
    "uv tool install",
])
def test_ci_runs_every_documented_gate(workflow_text: str, command: str):
    """门禁被"先删了让 CI 绿"删掉时，这条测试要说出它当初为什么在。

    `fail_under` 在 pyproject 里，所以 `coverage report` 这一步就是覆盖率门禁本身；
    少一步等于少一道门，而 CI 变绿时没人会去数步数。
    """
    assert command in workflow_text, f"CI 里少了这一步：{command}"


def test_ci_does_not_claim_to_run_engine_dependent_tiers(workflow_text: str):
    """`-m live` / `-m probe` 需要真实引擎与那块 GPU，CI 里不该假装跑过。

    谎报的代价是"CI 绿 = 计量口径没问题"这个错误信念。
    """
    assert "pytest -m live" not in workflow_text
    assert "pytest -m probe" not in workflow_text
    assert "不在 CI 跑" in workflow_text, "要显式写出这两档为什么不在这里跑"


def test_makefile_exposes_the_coverage_gate():
    make = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "coverage:" in make
    assert "fail_under" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
