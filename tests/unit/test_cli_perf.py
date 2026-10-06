"""`onyx perf` 的 CLI 侧（S36）。

这些测试同时是 IMPLEMENTATION 里那几条"自测命令"的可执行版本：命令跑不出来，
文档里的验收步骤就是假的。真引擎不在这里测（`-m live` 那一档测），
这里测的是**四条拒绝与三种读数**：拒 mock、拒坏网格、拒跨条件对比、
以及"引擎连不上"要留下一行 status=error 而不是无声退出。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from onyx.cli import app
from onyx.store.db import Database
from onyx.store.records import PerfCellRecord, PerfRunRecord
from onyx.store.repos import PerfRepo

runner = CliRunner()

BASE_FLAGS = ["--provider", "ollama", "--model", "qwen3.5:9b"]


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


def _db(tmp_path) -> Database:
    return Database(tmp_path / "onyx.sqlite")


def _seed(tmp_path, *, run_id: str, model: str, env_hash: str, status: str = "done") -> None:
    database = _db(tmp_path)
    repo = PerfRepo(database)
    repo.insert_run(PerfRunRecord(
        id=run_id, started_at="2026-10-06T00:00:00+00:00",
        finished_at="2026-10-06T00:00:30+00:00", status=status, model=model,
        provider_id="ollama-local", engine_version="0.35.1", device="rtx4060ti",
        keep_alive="10m", timing_source="engine_ns", env_hash=env_hash, comparable=True,
        conditions={"model": model, "engine_version": "0.35.1", "provider_id": "ollama-local",
                    "quantization": "Q4_K_M", "device": "rtx4060ti", "num_ctx": None,
                    "keep_alive": "10m", "stream": True, "timing_source": "engine_ns",
                    "temperature": 0.0, "seed": None, "grid": "600:64:1:2:warm",
                    "app_version": "0.8.0", "git_rev": "abc1234"},
        grid={"repeat": 2}, elapsed_s=30.0, n_requests=4,
    ))
    repo.insert_cells([PerfCellRecord(
        run_id=run_id, cell_key="600c/64t/x1/warm", phase="warm", prompt_chars=600,
        target_tokens=64, concurrency=1, repeat=2, status="measured",
        n_requests=2, n_measured=2,
        metrics={"n_requests": 2, "n_measured": 2, "n_error": 0, "n_truncated": 0,
                 "decode_tps": {"median": 20.0, "p95": 21.0, "n": 2, "min": 19.0, "max": 21.0},
                 "ttft_ms": {"median": 12.5, "p95": 13.0, "n": 2, "min": 12.0, "max": 13.0},
                 "aggregate_tps": None, "prefill_tps_cold": None},
        trace_ids=("t1", "t2"))])
    database.close()


# ── 四条拒绝 ──────────────────────────────────────────────────────
def test_run_refuses_mock_because_fake_latency_would_become_a_baseline(tmp_path):
    result = _run("perf", "run", "--provider", "mock", "--model", "m")
    assert result.exit_code == 2
    assert "假延迟" in result.output, "要说清为什么不做，否则下一次有人换个 flag 再试"
    assert "tests/unit/test_perf_" in result.output, "要给出路：测形状走测试"


def test_run_rejects_a_bad_grid_with_the_flag_name(tmp_path):
    result = _run("perf", "run", *BASE_FLAGS, "--concurrency", "0")
    assert result.exit_code == 2
    assert "--concurrency" in result.output


def test_ls_without_any_baseline_says_so(tmp_path):
    result = _run("perf", "ls")
    assert result.exit_code == 1
    assert "onyx perf run" in result.output, "空表要给入口，不能只留一句「没有」"


def test_show_with_an_unknown_id_is_a_usage_error(tmp_path):
    _seed(tmp_path, run_id="R1", model="qwen3.5:9b", env_hash="h1")
    result = _run("perf", "show", "nope")
    assert result.exit_code == 2
    assert "onyx perf ls" in result.output


# ── 引擎连不上：要留下一行 error，而不是无声退出 ──────────────────
def test_unreachable_engine_still_records_an_error_run(tmp_path):
    result = _run("perf", "run", *BASE_FLAGS, "--url", "http://127.0.0.1:1",
                  "--prompt-chars", "600", "--target-tokens", "64",
                  "--concurrency", "1", "--repeat", "1",
                  "--gpu-lock", str(tmp_path / "g.lock"))
    assert result.exit_code == 1, f"一条没跑成的基线不该以 0 退出：{result.output}"
    database = _db(tmp_path)
    runs = PerfRepo(database).list_runs()
    assert runs and runs[0].status == "error"
    cells = PerfRepo(database).cells(runs[0].id)
    assert cells[0].status == "error" and cells[0].reason, "失败形状要能被读出来"
    database.close()


# ── 读数与跨条件拒绝 ─────────────────────────────────────────────
def test_show_renders_conditions_and_cells(tmp_path):
    _seed(tmp_path, run_id="R1", model="qwen3.5:9b", env_hash="h1")
    result = _run("perf", "show", "R1")
    assert result.exit_code == 0, result.output
    assert "指纹" in result.output and "0.35.1" in result.output
    assert "600c/64t/x1/warm" in result.output and "✓" in result.output


def test_show_json_carries_the_conditions_and_every_cell(tmp_path):
    _seed(tmp_path, run_id="R1", model="qwen3.5:9b", env_hash="h1")
    payload = json.loads(_run("perf", "show", "R1", "--json").output)
    assert payload["run"]["env_hash"] == "h1"
    assert payload["cells"][0]["metrics"]["decode_tps"]["median"] == 20.0
    assert payload["cells"][0]["metrics"]["aggregate_tps"] is None, "「没测到」不能变成 0"
    # 这条基线引用的 trace 不在了（测试只写了基线行）——数字仍然成立，但要说清点不回去
    assert payload["traces_still_present"] == 0


def test_compare_refuses_across_different_conditions_and_names_the_field(tmp_path):
    _seed(tmp_path, run_id="R1", model="qwen3.5:9b", env_hash="h1")
    _seed(tmp_path, run_id="R2", model="qwen3:8b", env_hash="h2")
    result = _run("perf", "compare", "R1", "R2")
    assert result.exit_code == 1
    assert "不是同一个实验" in result.output
    assert "model" in result.output, "要逐字段列出来，而不是只甩一句「条件不同」"
    assert result.output.count("  · ") == 1, f"只有 model 变了，就该只列一行：{result.output}"


def test_compare_force_still_prints_the_caveat_then_the_delta(tmp_path):
    _seed(tmp_path, run_id="R1", model="qwen3.5:9b", env_hash="h1")
    _seed(tmp_path, run_id="R2", model="qwen3:8b", env_hash="h2")
    result = _run("perf", "compare", "R1", "R2", "--force")
    assert result.exit_code == 0, result.output
    assert "--force" in result.output
    assert "decode_tps" in result.output


def test_compare_between_identical_conditions_gives_signed_delta(tmp_path):
    from dataclasses import replace

    _seed(tmp_path, run_id="R1", model="qwen3.5:9b", env_hash="h1")
    _seed(tmp_path, run_id="R2", model="qwen3.5:9b", env_hash="h1")
    database = _db(tmp_path)
    repo = PerfRepo(database)
    row = repo.cells("R2")[0]
    repo.insert_cells([replace(row, metrics={
        **row.metrics,
        "decode_tps": {"median": 18.0, "p95": 19.0, "n": 2, "min": 17.0, "max": 19.0}})])
    database.close()

    result = _run("perf", "compare", "R1", "R2")
    assert result.exit_code == 0, result.output
    assert "-10.0%" in result.output, "20.0 → 18.0 必须是 -10%，方向不能反"
    assert "n<5" in result.output, "两发的样本要标低置信"


def test_help_lists_the_four_commands():
    result = _run("perf", "--help")
    for name in ("run", "ls", "show", "compare"):
        assert name in result.output
