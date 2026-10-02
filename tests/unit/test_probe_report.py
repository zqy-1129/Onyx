"""能力矩阵渲染测试：三态必须在输出里可区分，且图例说明处置方式。"""

from __future__ import annotations

from onyx.core.types import ApiStyle, Cap, ProviderKind
from onyx.llm.caps import infer_caps
from onyx.probe.report import MatrixRow, render_console, render_markdown


def _rows() -> list[MatrixRow]:
    probed = infer_caps(
        engine_capabilities=("completion", "tools", "thinking"),
        probe_findings={"structured": "enforced", "tool_format": "native_head"},
        api_style=ApiStyle.NATIVE, provider_kind=ProviderKind.OLLAMA,
    )
    unprobed = infer_caps(
        engine_capabilities=("completion", "vision"),
        api_style=ApiStyle.NATIVE, provider_kind=ProviderKind.OLLAMA,
    )
    return [
        MatrixRow(name="qwen3.5:9b", caps=probed, parameter_size="9.7B", quantization="Q4_K_M",
                  size_gb=6.6, tool_format="native", ctx_train=262144, ctx_loaded=4096,
                  probed=True, usage_ratio=0.657, usage_ratio_n=40),
        MatrixRow(name="unprobed:1b", caps=unprobed, parameter_size="1.3B", quantization="Q4_K_M",
                  size_gb=1.0, tool_format="unknown", ctx_train=8192, probed=False),
    ]


def test_markdown_contains_three_states_and_legend():
    text = render_markdown(_rows(), provider_version="0.35.0", provider_id="ollama-local")
    assert "| 模型 |" in text and "qwen3.5:9b" in text
    assert "✓" in text and "✗" in text and "?" in text
    assert "未实测" in text and "onyx probe run" in text
    assert "0.35.0" in text
    assert "4096 / 262144" in text, "载入上下文与训练上下文必须并列显示（PROBES P3）"


def test_markdown_lists_unprobed_caps():
    text = render_markdown(_rows())
    assert "未实测的能力位" in text
    assert Cap.STRUCTURED_OUTPUT.value in text


def test_ratio_text_requires_enough_samples():
    rows = _rows()
    assert rows[0].ratio_text.startswith("0.657")
    assert rows[1].ratio_text == "未标定"
    thin = MatrixRow(name="x", caps=rows[0].caps, usage_ratio=0.6, usage_ratio_n=5)
    assert thin.ratio_text == "未标定", "n<30 的比值不可用，必须显示未标定"


def test_console_render_does_not_raise():
    from rich.console import Console

    console = Console(width=200, force_terminal=False)
    render_console(_rows(), provider_version="0.35.0", provider_id="ollama-local", console=console)


def _capture(width: int) -> str:
    import io

    from rich.console import Console

    buffer = io.StringIO()
    render_console(
        _rows(), provider_version="0.35.0", provider_id="ollama-local",
        console=Console(width=width, file=buffer, force_terminal=False, color_system=None),
    )
    return buffer.getvalue()


def test_narrow_terminal_switches_to_compact_layout():
    """80 列终端里 13 列矩阵会渲染成乱码——必须自适应，而不是让用户自己拉宽窗口。"""
    out = _capture(80)
    assert "qwen3.5:9b" in out, "模型名必须完整可见"
    assert "…" not in out, "紧凑布局不该出现截断省略号"
    assert "能力位" in out, "应拆成概况表 + 能力位表"
    assert "tool✓" in out and "tch✗" in out and "str?" in out, f"紧凑能力行: {out!r}"


def test_wide_terminal_uses_full_matrix():
    out = _capture(200)
    assert "tool_ch" in out and "stream" in out
    assert "4096 / 262144" in out


def test_compact_caps_keeps_three_states_distinguishable():
    from onyx.probe.report import compact_caps

    text = compact_caps(_rows()[0])
    assert "✓" in text and "✗" in text and "?" in text
