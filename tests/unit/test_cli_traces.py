"""`onyx traces show` / `onyx chat` 的格式化回归。

这里守的是一条已经发生过的真实故障：`onyx/cli.py` 里出现过**两份同名 `_fmt`**
（S3 的那份第二参数是 `suffix`，S13 加的那份是 `digits`）。后定义的把前者遮蔽，
`_fmt(ttft_ms, "ms")` 于是变成 `digits="ms"`，`traces show` 在**任何有数字的 trace**
上直接 ValueError；而值为 None 时提前返回「—」，所以只有真实数据才炸——
mock-only 的测试永远看不见，ruff 的 F811 也因为"前一份被使用过"不报。
那条结构断言现在搬到了 `tests/unit/test_module_structure.py`，对全仓库生效；
本文件只留这几条命令的行为回归。
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from onyx.cli import _fmt, app
from onyx.core.content import FileBlobStore
from onyx.core.types import GenerationRequest
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.records import TokenPartRecord, TraceRecord, UsageRecord
from onyx.store.sinks import SqliteRecordSink

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "machine"))
    # rich 会按终端宽度截断列；诊断输出必须能整串读出来才好断言
    monkeypatch.setenv("COLUMNS", "220")
    return tmp_path


def _seed_trace(tmp_path) -> str:
    """用真实 gateway 落一条**带数字延迟**的 trace（流式才有 ttft）。"""
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=1, idle_wait=0.005)
    gateway = Gateway(MockProvider(), observer=ObserverEngine(record_sink=sink),
                      blobs=FileBlobStore(tmp_path / "blobs"))
    result = gateway.generate(GenerationRequest.of("mock/echo", "北京天气怎么样", stream=True))
    sink.flush(2.0)
    sink.close()
    db.close()
    return result.trace_id


def test_traces_show_survives_numeric_latency(isolated_data_dir):
    trace_id = _seed_trace(isolated_data_dir)
    result = runner.invoke(app, ["traces", "show", trace_id])
    assert result.exit_code == 0, repr(result.exception)
    assert "延迟" in result.output
    assert "ms" in result.output


def test_fmt_accepts_suffix_and_digits():
    assert _fmt(12.3456, 1, "ms") == "12.3ms"
    assert _fmt(None, 1, "ms") == "—", "未知不许被带上单位"
    assert _fmt(0.9876) == "0.988"
    assert _fmt(0.0, 1, "ms") == "0.0ms", "0 是测量结果，不是未知"


# ── S39：分段归因的标题必须是条件句 ────────────────────────────────
def _seed_clamped(tmp_path) -> str:
    """直接种一条**被 clamp** 的行（真机 bench 的形状：分段按启发式数，比引擎高估）。

    走 gateway 造不出这一条——mock 的正文短到不会高估，而"高估"恰恰是要展示的形状。
    `template_ctl` 记 0 是 `parts.py` 在残差为负时的既定行为，种子必须跟它一致。
    """
    from onyx.store.repos import TraceRepo, UsageRepo

    db = Database(tmp_path / "onyx.sqlite")
    TraceRepo(db).upsert(TraceRecord(
        id="CLAMP1", kind="generation", purpose="bench", started_at="2026-10-05T10:00:00+00:00",
        status="ok", provider_id="ollama-local", model_id="ollama-local/qwen3.5:9b",
        model_name="qwen3.5:9b",
        extra={"attribution": {"count_source": "heuristic", "clamped": True, "residual_raw": -146,
                               "input_segments_tokens": 621, "template_ctl_tokens": 0}},
    ))
    UsageRepo(db).upsert(UsageRecord(trace_id="CLAMP1", source="engine", confidence="high",
                                     in_tokens=475, out_tokens=32))
    UsageRepo(db).replace_parts("CLAMP1", [
        TokenPartRecord(trace_id="CLAMP1", part="msg:0", tokens=621),
        TokenPartRecord(trace_id="CLAMP1", part="output", ord=1, tokens=32),
        TokenPartRecord(trace_id="CLAMP1", part="template_ctl", ord=2, tokens=0),
    ])
    db.close()
    return "CLAMP1"


def test_shows_the_parts_table_with_a_conditional_title(isolated_data_dir):
    """真机跑通的一条（gateway → USAGE_ATTRIBUTION → trace.extra.attribution）：
    标题必须是"仅未 clamp 时成立"，并带上这一条实际用的档位。"""
    trace_id = _seed_trace(isolated_data_dir)
    result = runner.invoke(app, ["traces", "show", trace_id])
    assert result.exit_code == 0, repr(result.exception)
    assert "仅未 clamp 时成立" in result.output, "无条件等式的标题会把人骗去查引擎"
    assert "分段按" in result.output


def test_a_clamped_row_says_the_parts_are_only_relative_shares(isolated_data_dir):
    trace_id = _seed_clamped(isolated_data_dir)
    result = runner.invoke(app, ["traces", "show", trace_id])
    assert result.exit_code == 0, repr(result.exception)
    out = result.output
    assert "本条已 clamp" in out and "残差 -146" in out
    assert "只能比相对占比" in out.replace("\n", "").replace(" ", ""), "要把能怎么用说清"
    assert "onyx token explain" in out, "给一条能继续走的路，不是只宣布坏了"
