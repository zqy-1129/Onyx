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
