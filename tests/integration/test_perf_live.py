"""真机基线（S36）：对着**真实运行的 Ollama** 跑最小网格。

这一档存在的理由和 `docs/PROBES.md` 一样：假 gateway 能证明聚合口径不出错，
但证明不了"引擎真的会回报这些数"。延迟与吞吐尤其危险——一个从来没在真机上出过数的字段，
第一次用的时候才发现引擎不回报，而界面上它一直显示「—」，没人知道是坏了还是本来就没有。

跑法：`uv run pytest -m live -q -s`（GPU 由 `tests/integration/conftest.py` 的 session 锁独占，
所以这里**不再自己拿锁**：并发跑评测会让所有延迟数字失真，那不是基准而是噪声）。
可用 `ONYX_TEST_MODEL` 换模型。
"""

from __future__ import annotations

import os

import pytest

from onyx.llm.providers.ollama import OllamaProvider

pytestmark = pytest.mark.live

MODEL = os.environ.get("ONYX_TEST_MODEL", "qwen3.5:9b")
BASE_URL = os.environ.get("ONYX_OLLAMA_URL", "http://127.0.0.1:11434")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    """一个临时库 + 真 provider。库用临时路径：真机基线不该写进开发者的 `.data`。"""
    from onyx.runtime import build_runtime
    from onyx.settings import load_settings

    probe = OllamaProvider(base_url=BASE_URL)
    if not probe.client.is_reachable():
        pytest.skip(f"Ollama 不可达: {BASE_URL}")
    names = [m.name for m in probe.list_models()]
    if MODEL not in names:
        pytest.skip(f"模型 {MODEL} 未安装，现有: {names}")
    probe.close()

    built = build_runtime(
        provider_kind="ollama", provider_id="ollama-local", base_url=BASE_URL,
        settings=load_settings(tmp_path_factory.mktemp("perfdata")),
        db_path=str(tmp_path_factory.mktemp("perfdb") / "live.sqlite"),
        event_log=False,
    )
    yield built
    built.close()


def test_real_engine_yields_engine_sourced_throughput(runtime):
    """最小的真机网格：一格 warm、一发请求，只要求"引擎给的数真的到了"。"""
    from onyx.perf.bench import collect
    from onyx.perf.spec import BenchPlan
    from onyx.store.repos import TraceRepo

    plan = BenchPlan(model=MODEL, prompt_chars=(600,), target_tokens=(32,),
                     concurrency=(1,), repeat=1, budget_s=240.0)
    outcome = collect(runtime.gateway, plan, device="testbench")
    cell = outcome.cells[0]
    assert cell.status == "measured", f"真机最小网格没跑成：{cell.reason}"
    metrics = cell.metrics
    assert metrics["n_measured"] == 1
    assert metrics["decode_tps"], "引擎的 eval_ns / out_tokens 有一个没回报，吞吐就不成立"
    assert metrics["ttft_ms"]["median"] > 0
    assert metrics["prompt_tokens"]["median"] >= 300, \
        "汉字下限 300 都没到 ⇒ 正文被裁，而这一格自称 600 字"

    # timing_source 必须由真机决定：这里如果是 wall_only，说明分段时序在这条通道上根本没拿到
    assert outcome.conditions["timing_source"] == "engine_ns", outcome.conditions
    assert outcome.comparable and outcome.env_hash

    trace_ids = [sample.trace_id for sample in cell.samples]
    assert all(TraceRepo(runtime.db).get(tid) is not None for tid in trace_ids), \
        "基线的每个数字都要能点回一条真 trace"
