"""扩展点边界脚本的单测。

M6 的出口条件是"接入新实现不改内核"，而这条**必须由脚本断言**：
人 review 的失效方式很具体——测试全绿、看板全对，`gateway.py` 里却多了
一个 `if kind == ...`，抽象从此名存实亡。
所以脚本本身也得被测：它漏判（该报错却没报错）就等于这条 DoD 不存在。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_extension_boundary.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_extension_boundary", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mod = _module()


def test_the_script_itself_exists_and_is_documented():
    assert SCRIPT.exists()
    assert mod.__doc__ and "DoD" in mod.__doc__


def test_adding_a_provider_alone_is_clean():
    changed = [
        "onyx/llm/providers/openai_compat.py",
        "tests/contract/test_provider_contract.py",
        "docs/IMPLEMENTATION.md",
    ]
    assert mod.violations(changed) == []
    assert mod.main(["--files", *changed]) == 0


def test_adding_a_provider_that_touches_gateway_is_a_leak(capsys):
    changed = ["onyx/llm/providers/openai_compat.py", "onyx/llm/gateway.py"]
    assert mod.main(["--files", *changed]) == 1
    out = capsys.readouterr().out
    assert "gateway.py" in out and "if" in out, "报错要说清是哪条规则、为什么"


def test_adding_a_provider_that_touches_core_is_a_leak():
    changed = ["onyx/llm/providers/openai_compat.py", "onyx/core/types.py"]
    found = mod.violations(changed)
    assert found and found[0][1] == "onyx/core/types.py"


def test_adding_a_provider_that_touches_obs_internals_is_a_leak():
    """换引擎不该改变 token 口径：对账实现属于内核，不是给 provider 留的口子。"""
    changed = ["onyx/llm/providers/vllm.py", "onyx/obs/visitors/token.py"]
    assert mod.violations(changed)


def test_adding_a_task_that_touches_the_runner_is_a_leak():
    changed = ["onyx/eval/tasks/rag_faithfulness.py", "onyx/eval/runner.py"]
    found = mod.violations(changed)
    assert found and found[0][1] == "onyx/eval/runner.py"


def test_changing_the_registration_itself_is_allowed():
    """注册机制本身的变更（S16a 做的就是这件事）不算泄漏——它是"接线"，不是"开小灶"。

    这条豁免必须写死在脚本里而不是靠人判断，否则下一次接插件时
    "顺手改一下 registry.py"就变成了常规操作。
    """
    changed = [
        "onyx/discovery.py",
        "onyx/eval/tasks/__init__.py",
        "onyx/obs/visitors/__init__.py",
        "plugins_example/example_task/example_task/task.py",
    ]
    assert mod.violations(changed) == []


def test_external_plugin_directory_counts_as_an_implementation():
    changed = ["plugins_example/example_provider/example_provider/provider.py",
               "onyx/llm/gateway.py"]
    assert mod.violations(changed)


def test_pure_kernel_or_docs_changes_are_skipped(capsys):
    """没接实现时不该拿这套规则管内核自己的开发（否则第一天就被绕过）。"""
    changed = ["onyx/llm/gateway.py", "onyx/core/types.py", "README.md"]
    assert mod.main(["--files", *changed]) == 0
    assert "跳过" in capsys.readouterr().out


def test_metrics_and_engine_are_protected_too():
    """指标口径与事件循环也是内核：新任务改 F1 定义 = 历史分数全部作废。"""
    assert mod.violations(["onyx/eval/tasks/new.py", "onyx/eval/metrics.py"])
    assert mod.violations(["onyx/store/sinks/otlp.py", "onyx/obs/engine.py"])


def test_windows_separators_are_normalized():
    """Windows 上 `git diff --name-only` 通常是正斜杠，但手动传参可能是反斜杠。"""
    assert mod.is_implementation("onyx\\llm\\providers\\vllm.py")
    assert mod.violations(["onyx\\llm\\providers\\vllm.py", "onyx\\llm\\gateway.py"])
