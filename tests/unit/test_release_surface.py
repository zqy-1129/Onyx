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
    "pytest -m e2e",
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


def test_ci_runs_e2e_as_its_own_step_not_by_accident(workflow_text: str):
    """`-m e2e` 必须显式跑：默认 addopts 把它 deselect 掉了。

    少了这一步，六页取数与 SSE 的回归保护就只存在于"有人在本地记得跑过"这件事上。
    """
    assert "pytest -m e2e" in workflow_text
    assert "deselect" in workflow_text, "要写明它为什么不会随离线套件顺带跑到（addopts 默认排除）"
    before = workflow_text.split("pytest -m e2e")[0][-500:]
    assert "mock" in before and "127.0.0.1" in before, \
        "e2e 那一步要写明它只用 mock 引擎且只绑回环，否则读 CI 的人会以为这里在打真引擎"


def test_ci_does_not_claim_to_run_engine_dependent_tiers(workflow_text: str):
    """`-m live` / `-m probe` 需要真实引擎与那块 GPU，CI 里不该假装跑过。

    谎报的代价是"CI 绿 = 计量口径没问题"这个错误信念。
    """
    assert "pytest -m live" not in workflow_text
    assert "pytest -m probe" not in workflow_text
    assert "不在 CI 跑" in workflow_text, "要显式写出这两档为什么不在这里跑"


def test_markdown_tables_have_one_line_per_row():
    """GFM 的表格一行必须一行写完：不以 `|` 收尾的那一行会把整行截断成半张表。

    这不是排版洁癖。S36 给 README 的 M12 行追加内容时把它撑成了 17 行，
    于是"进度表"从那一行开始不再渲染成表——而 markdown 的差异在 diff 里完全看不出来，
    预览里也只是"少了一行"，很容易被当成渲染器的怪事。
    """
    problems: list[str] = []
    for name in ("README.md", "docs/STATUS.md", "docs/ROADMAP.md", "docs/DESIGN.md",
                 "docs/IMPLEMENTATION.md", "CHANGELOG.md"):
        text = (ROOT / name).read_text(encoding="utf-8")
        inside_fence = False
        for number, line in enumerate(text.split("\n"), start=1):
            stripped = line.rstrip()
            if stripped.startswith("```"):
                inside_fence = not inside_fence   # 代码块里的 ASCII 表不受本条约束
                continue
            if inside_fence:
                continue
            if stripped.startswith("|") and not stripped.endswith("|"):
                problems.append(f"{name}:{number}: 这一行以 `|` 开头但不以 `|` 收尾")
    assert not problems, "表格行被换行拆断了：\n  " + "\n  ".join(problems[:8])


def test_the_equality_claim_is_only_made_with_its_condition():
    """「Σ分段 + template_ctl = 引擎计数」是**有条件**的：残差为负时 `template_ctl` 被 clamp 成 0
    （`llm/measurement/parts.py`，S39 取证：未标定的模型条条如此）。

    文档里可以自由讨论这句话，但**给用户看的那一行字符串**不许再说它是无条件等式——
    那会把人骗去查引擎，而真正该跑的是 `onyx calibrate`。所以这里只扫代码（py/ts/tsx），
    并要求同一行里出现 `clamp` 这个限定词。
    """
    phrase = "Σ分段 + template_ctl = 引擎计数"
    targets = [p for p in (ROOT / "onyx").rglob("*.py") if "__pycache__" not in p.parts]
    targets += [p for p in (ROOT / "onyx" / "web" / "src").rglob("*.ts")]
    targets += [p for p in (ROOT / "onyx" / "web" / "src").rglob("*.tsx")]
    problems = []
    for path in targets:
        for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
            if phrase in line and "clamp" not in line:
                problems.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:90]}")
    assert not problems, "这句等式必须带着「仅未 clamp 时成立」一起出现：\n  " + "\n  ".join(problems[:6])

    # 而且两端（终端与看板）必须真的都在说这句带限定的话——只删掉旧句子不算补上了条件
    assert "仅未 clamp 时成立" in (ROOT / "onyx" / "cli.py").read_text(encoding="utf-8")
    assert "仅未 clamp 时成立" in (ROOT / "onyx" / "web" / "src" / "format.ts").read_text(encoding="utf-8")


def test_makefile_exposes_the_coverage_gate():
    make = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "coverage:" in make
    assert "fail_under" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_coverage_floor_is_pinned_and_carries_its_own_reason():
    """地板值、它的口径、以及"为什么是这个数"必须同时被钉住。

    `fail_under` 的作用是**挡住退化**，不是让我们通过：80 配 90% 上下的实测意味着
    一次 -10 个点的真实退化会全绿通过，而"门禁在守"这件事看起来照样成立。
    S35 把它抬到 85（留 6 个点余量）——余量小到一次真实退化就会被挡住，
    又大到正常的增删代码不会逼人顺手把地板调回去。

    数值本身之外还要断言注释：一个没有理由的数字，下一个人只会把它当成
    "某人随手写的 80"，而改成 70 看起来是修 CI 而不是放水。
    """
    coverage = _pyproject()["tool"]["coverage"]
    assert coverage["run"]["branch"] is True, \
        "门禁必须数分支：只数语句会让「两岔只走过一岔」的文件看起来很高"
    assert coverage["report"]["fail_under"] == 85, \
        "地板被悄悄调回 80（或更低）就等于没有门禁——它是用来挡退化的"

    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    above = text.split("[tool.coverage.report]")[1].split("fail_under")[0]
    assert re.search(r"\d{4}-\d{2}-\d{2}", above), "要写明实测是哪天量的"
    assert "%" in above, "要写明当前实测值，否则没人算得出余量是几个点"
    assert "离线" in above, "必须写明用离线套件量：CI 机器上不一定有 Ollama"
