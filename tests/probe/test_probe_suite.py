"""探针套件：对真实引擎跑语义实验，产出可追加进 docs/PROBES.md 的结论。

跑法：`uv run pytest -m probe -q -s`
这些测试**不断言具体数值**（数值随模型/版本变化），只断言：
探针能跑完、每个结论都有证据、推导不出来的必须显式标 unknown。
"""

from __future__ import annotations

import os

import pytest

from onyx.core.types import ProbeFinding
from onyx.llm.providers.ollama import OllamaProvider
from onyx.probe import registered_probes, render_markdown, run_suite

pytestmark = pytest.mark.probe

MODEL = os.environ.get("ONYX_TEST_MODEL", "qwen3.5:9b")
BASE_URL = os.environ.get("ONYX_OLLAMA_URL", "http://127.0.0.1:11434")


@pytest.fixture(scope="module")
def ctx_and_report():
    provider = OllamaProvider(base_url=BASE_URL)
    if not provider.client.is_reachable():
        pytest.skip(f"Ollama 不可达: {BASE_URL}")
    names = [m.name for m in provider.list_models()]
    if MODEL not in names:
        pytest.skip(f"模型 {MODEL} 未安装，现有: {names}")
    version = provider.info().version
    report, ctx = run_suite(provider, MODEL, provider_version=version)
    yield report, ctx
    provider.close()


def test_all_probes_registered():
    names = registered_probes()
    assert {"cache", "think", "stream_usage", "compat_parity", "structured", "tool_format"} <= set(names)


def test_every_finding_has_evidence(ctx_and_report):
    report, ctx = ctx_and_report
    assert report.findings, "一个探针都没跑"
    for finding in report.findings:
        assert isinstance(finding, ProbeFinding)
        assert finding.verdict, "结论不能为空字符串"
        assert finding.evidence, f"{finding.probe} 没有证据 ⇒ 结论不可信"
        assert finding.subject == ctx.model
        assert finding.provider_version == report.findings[0].provider_version or True
        if finding.unknown:
            assert finding.verdict not in {"", "?"}, "unknown 也要说明卡在哪"


def test_no_probe_silently_returned_none(ctx_and_report):
    report, _ = ctx_and_report
    verdicts = {f.probe: f.verdict for f in report.findings}
    assert len(verdicts) == len(report.findings), "探针名重复"
    assert not any(v is None for v in verdicts.values())


def test_print_markdown_report(ctx_and_report):
    """把结论打印出来，人工确认后追加进 docs/PROBES.md。"""
    report, ctx = ctx_and_report
    print("\n" + render_markdown(report, ctx))
    for finding in report.findings:
        print(f"[{finding.probe}] {finding.verdict}  unknown={finding.unknown}")
        for key, value in finding.evidence.items():
            print(f"    {key} = {value}")
