"""基线落库（S36）。

`perf_run` / `perf_cell` 是这条命令唯一的事实源：CLI 的 `ls` / `show` / `compare` 都从这两张表读。
所以这里测的不是"能不能 insert"，而是**读回来还是不是同一个事实**——
尤其是 `None` 必须在 JSON 往返后还是 `None`（补 0 的话，一条"没测到"的格子
会在 compare 里变成"延迟 0ms"，那是全场最快的结果）。
"""

from __future__ import annotations

import pytest

from onyx.store.db import Database
from onyx.store.records import PerfCellRecord, PerfRunRecord
from onyx.store.repos import PerfRepo


@pytest.fixture
def db(tmp_path) -> Database:
    return Database(tmp_path / "perf.sqlite")


def _run(run_id: str = "R1", *, model: str = "qwen3.5:9b", env_hash: str = "h1",
         **overrides) -> PerfRunRecord:
    base = dict(
        id=run_id, started_at="2026-10-06T00:00:00+00:00",
        finished_at="2026-10-06T00:01:00+00:00",
        status="done", model=model, provider_id="ollama-local", engine_version="0.35.1",
        quantization="Q4_K_M", device="rtx4060ti", num_ctx=8192, keep_alive="10m", stream=True,
        timing_source="engine_ns", env_hash=env_hash, comparable=True,
        conditions={"model": model, "engine_version": "0.35.1"},
        grid={"repeat": 2}, elapsed_s=60.0, n_requests=8,
        app_version="0.8.0", git_rev="abc1234",
    )
    return PerfRunRecord(**{**base, **overrides})


def _cell(run_id: str = "R1", key: str = "600c/64t/x1/warm", **kw) -> PerfCellRecord:
    base = dict(run_id=run_id, cell_key=key, phase="warm", prompt_chars=600,
                target_tokens=64, concurrency=1, repeat=2, status="measured", reason="",
                n_requests=2, n_measured=2,
                metrics={"decode_tps": {"median": 20.0, "n": 2},
                         "prefill_tps_cold": None, "n_truncated": 0},
                trace_ids=("t1", "t2"))
    return PerfCellRecord(**{**base, **kw})


def test_run_and_cells_round_trip(db):
    repo = PerfRepo(db)
    repo.insert_run(_run())
    repo.insert_cells([_cell(), _cell(key="600c/64t/x2/warm")])

    got = repo.get_run("R1")
    assert got is not None and got.model == "qwen3.5:9b" and got.comparable
    assert got.num_ctx == 8192 and got.conditions["engine_version"] == "0.35.1"
    cells = repo.cells("R1")
    assert [c.cell_key for c in cells] == ["600c/64t/x1/warm", "600c/64t/x2/warm"]
    assert cells[0].metrics["decode_tps"]["median"] == 20.0
    assert cells[0].trace_ids == ("t1", "t2")


def test_unknown_stays_unknown_through_the_database(db):
    """`None` 往返后必须还是 `None`。补 0 会让"没测到"变成"最快的一格"。"""
    repo = PerfRepo(db)
    repo.insert_run(_run())
    repo.insert_cells([_cell()])
    cell = repo.cells("R1")[0]
    assert cell.metrics["prefill_tps_cold"] is None
    assert "prefill_tps_cold" in cell.metrics, "键要留着：区分「测了但没有」与「没这一列」"


def test_skipped_cells_are_persisted_with_their_reason(db):
    repo = PerfRepo(db)
    repo.insert_run(_run(status="partial"))
    repo.insert_cells([_cell(status="skipped", reason="预算 300s 用尽，这一格一条没发",
                            n_requests=0, n_measured=0, metrics={}, trace_ids=())])
    cell = repo.cells("R1")[0]
    assert cell.status == "skipped" and "预算" in cell.reason
    assert cell.metrics == {}


def test_insert_is_idempotent_on_id(db):
    """同一个 id 重写不产生第二行（跑挂之后补记录用的上）。"""
    repo = PerfRepo(db)
    repo.insert_run(_run(note="第一次"))
    repo.insert_run(_run(note="补写的备注"))
    repo.insert_cells([_cell(n_measured=1), _cell(n_measured=2)])
    assert len(repo.list_runs()) == 1
    assert repo.get_run("R1").note == "补写的备注"
    assert repo.cells("R1")[0].n_measured == 2


def test_find_by_env_excludes_the_run_itself(db):
    repo = PerfRepo(db)
    repo.insert_run(_run("R1"))
    repo.insert_run(_run("R2", env_hash="h1"))
    repo.insert_run(_run("R3", env_hash="other"))
    peers = repo.find_by_env("h1", before="R2")
    assert [item.id for item in peers] == ["R1"], "同条件历史要能找得到，自己不能算进去"


def test_list_runs_filters_and_orders(db):
    repo = PerfRepo(db)
    repo.insert_run(_run("R1"))
    repo.insert_run(_run("R2", model="qwen3:8b"))
    assert [item.id for item in repo.list_runs(model="qwen3:8b")] == ["R2"]
    assert len(repo.list_runs(limit=1)) == 1


def test_new_ids_are_unique_and_time_sortable(db):
    ids = [PerfRepo(db).new_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert ids == sorted(ids), "id 要按时间可排序：`ORDER BY id` 得走索引而不是靠 started_at"


def test_get_run_returns_none_for_an_unknown_id(db):
    assert PerfRepo(db).get_run("nope") is None


# ── 写路径：collect → rows → 库 → 读回 ────────────────────────────
def test_outcome_rows_survive_the_database(db):
    """这条是把采集器与库缝起来：任何一边改了字段形状，这里就会红。"""
    from types import SimpleNamespace

    from onyx.core.clock import FakeClock
    from onyx.core.types import Generation, Status, TokenSample, TokenSource
    from onyx.llm.measurement.reconciler import latency_summary
    from onyx.perf.bench import collect
    from onyx.perf.report import outcome_to_rows
    from onyx.perf.spec import BenchPlan

    class TinyGateway:
        def generate(self, req, *, purpose=None):
            gen = Generation(text="输出", model=req.model, status=Status.OK, wall_ms=320.0,
                             ttft_ms=9.5,
                             usage=(TokenSample(source=TokenSource.ENGINE, in_tokens=600,
                                                 out_tokens=64),))
            return SimpleNamespace(trace_id="t-1", generation=gen, latency=latency_summary(gen))

    plan = BenchPlan(model="qwen3.5:9b", prompt_chars=(600,), target_tokens=(64,),
                     concurrency=(1,), repeat=1)
    outcome = collect(TinyGateway(), plan, clock=FakeClock(),
                      engine_info={"provider_id": "ollama-local", "version": "0.35.1"})
    repo = PerfRepo(db)
    run, cells = outcome_to_rows(outcome, run_id=repo.new_id(),
                                 started_at="2026-10-06T00:00:00+00:00",
                                 finished_at="2026-10-06T00:00:30+00:00")
    repo.insert_run(run)
    repo.insert_cells(cells)

    back = repo.get_run(run.id)
    assert back.env_hash == outcome.env_hash == repo.get_run(run.id).env_hash
    assert back.grid["model"] == "qwen3.5:9b"
    row = repo.cells(run.id)[0]
    assert row.status == "measured" and row.metrics["ttft_ms"]["median"] == 9.5
    assert row.trace_ids == ("t-1",)
