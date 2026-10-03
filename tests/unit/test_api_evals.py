"""S14 验收：评测 API。

这个文件最重要的断言是 `grades` 里带 `trace_id`。
"每个分数都能点进一条真实 trace"是整个系统的立足点（DESIGN §15）——
API 不暴露 trace_id，前端就做不出下钻，这条主张就只写在文档里。

也断言 GPU 状态是**只读**的：看板轮询它不能把锁抢了。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.eval.datasets.builtin.intent_zh import build_cases
from onyx.eval.datasets.loader import Dataset
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.llm.providers.mock import MockScript
from onyx.runtime import build_runtime
from onyx.settings import load_settings
from onyx.store.repos import EvalRepo

SCRIPTS = {
    "mock/echo": MockScript(text="转账", in_tokens=120, out_tokens=4, done_reason="stop"),
}


def _dataset(n: int = 6) -> Dataset:
    cases = build_cases()[:n]
    return Dataset(id="intent_zh-v1", cases=tuple(cases), upstream="test", revision="r1")


@pytest.fixture
def env(tmp_path):
    """runtime + 一次真实跑完的评测，API 测的是"有数据时返回什么"。"""
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": tuple(SCRIPTS)},
    )
    dataset = _dataset()
    # 让每个样本的期望标签正好是脚本会输出的那个，才有 correct 可看
    cases = []
    for case in dataset.cases:
        case = dict(case)
        case["expect"] = {"label": "转账", "must_call": True}
        case["input"] = {"instruction": "帮我转 500 给张伟"}
        cases.append(case)
    dataset = Dataset(id="intent_zh-v1", cases=tuple(cases), upstream="test", revision="r1")

    task = IntentClassification(dataset, model="mock/echo")
    report = EvalRunner(runtime.gateway, EvalRepo(runtime.db), task, dataset=dataset).run(
        RunConfig(model="mock/echo", seed=7)
    )
    runtime.flush()
    # 锁路径指到 tmp：默认的机器级锁锁的是**真 GPU**，离线套件不该去抢它
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")) as client:
        yield client, runtime, report
    runtime.close()


# ── datasets / runs ───────────────────────────────────────────────
def test_datasets_are_listed_with_full_provenance(env):
    client, _, _ = env
    rows = client.get("/api/datasets").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "intent_zh-v1"
    assert row["n_cases"] == 6
    assert row["upstream"] == "test"
    assert row["revision"] == "r1", "换了 revision 就该看得出，否则两次分数不可比会被误当成可比"
    assert row["imported_at"]


def test_runs_are_listed_and_filterable(env):
    client, _, report = env
    rows = client.get("/api/runs").json()
    assert len(rows) == 1
    assert rows[0]["id"] == report.run_id
    assert rows[0]["status"] == "done"
    assert rows[0]["seed"] == 7
    assert rows[0]["app_version"]

    assert client.get("/api/runs", params={"task": "intent_classification"}).status_code == 200
    assert client.get("/api/runs", params={"task": "nope"}).json() == []


def test_run_detail_exposes_aggregate_and_cost(env):
    client, _, report = env
    payload = client.get(f"/api/runs/{report.run_id}").json()
    run = payload["run"]
    assert run["n_done"] == 6
    # 汇总必须原样可读：CI 是 dict（落库前已 jsonable），不是 dataclass 的 repr 字符串
    assert isinstance(run["aggregate"]["macro_f1_ci"], dict)
    assert "CI(low=" not in str(run["aggregate"])
    assert run["aggregate"]["scoring"] == "gen-based"
    assert run["cost"]["requests"] == 6
    assert run["params_snapshot"]["temperature"] == 0.0
    assert payload["verdict_counts"].get("correct") == 6


def test_unknown_run_is_404_not_500(env):
    client, _, _ = env
    assert client.get("/api/runs/nope").status_code == 404
    assert client.get("/api/runs/nope/grades").status_code == 404
    body = client.get("/api/runs/nope").json()
    assert "找不到 run" in str(body)
    assert "Traceback" not in str(body), "错误响应不许泄漏内部结构"


# ── grades：下钻的证据链 ──────────────────────────────────────────
def test_grades_carry_trace_ids_so_scores_can_drill_into_traces(env):
    client, _, report = env
    grades = client.get(f"/api/runs/{report.run_id}/grades").json()
    assert len(grades) == 6
    for grade in grades:
        assert grade["trace_id"], "分数点不进 trace，评测就退化成一个孤立的数字"
        assert grade["verdict"] == "correct"
        assert grade["passed"] is True
        assert grade["invalid_format"] is False
        # 前端要能按 trace_id 直接拿详情
        detail = client.get(f"/api/traces/{grade['trace_id']}")
        assert detail.status_code == 200, "grade 指向的 trace 必须真的可查"


def test_grades_can_be_filtered_by_verdict(env):
    client, _, report = env
    assert len(client.get(f"/api/runs/{report.run_id}/grades",
                          params={"verdict": "correct"}).json()) == 6
    assert client.get(f"/api/runs/{report.run_id}/grades",
                      params={"verdict": "wrong"}).json() == []


def test_grade_limit_is_enforced(env):
    client, _, report = env
    assert len(client.get(f"/api/runs/{report.run_id}/grades", params={"limit": 2}).json()) == 2
    assert client.get(f"/api/runs/{report.run_id}/grades", params={"limit": 0}).status_code == 422


# ── GPU 状态 ──────────────────────────────────────────────────────
def test_gpu_is_free_when_nothing_holds_it(env):
    client, _, _ = env
    assert client.get("/api/gpu").json() == {"busy": False, "owner": None, "progress": None,
                                            "eta_s": None, "holder_host": None}


def test_gpu_status_reports_owner_progress_and_eta(env):
    client, runtime, _ = env
    holder = _holder_for(client, runtime)
    assert holder["busy"] is True
    assert holder["owner"] == "eval:intent_classification@qwen3.5:9b"
    assert holder["progress"] == "30/240"
    assert holder["eta_s"] is not None and holder["eta_s"] > 0


def test_reading_gpu_status_does_not_acquire_the_lock(env):
    """看板会每秒轮询这个接口；轮询者不许把锁抢了或放了。"""
    from onyx.eval.gpu_lock import GpuLock

    client, _, _ = env
    lock_path = _lock_path_of(client)
    lock = GpuLock(lock_path, owner="other", stale_after_s=60)
    lock.acquire()
    lock.heartbeat(1, 10)
    try:
        for _ in range(3):
            assert client.get("/api/gpu").json()["busy"] is True
        assert lock.held is True, "只读接口动了锁的状态"
        assert lock.peek().owner == "other"
    finally:
        lock.release()
    assert client.get("/api/gpu").json()["busy"] is False


def _lock_path_of(client) -> object:
    """从 app 自己的 state 上取锁路径。

    硬编码的话，默认锁路径一改这些测试就会静默失去意义——
    它们会和被测对象锁不同的文件，于是"锁住了/没锁住"全都看不出来。
    """
    return client.app.state.onyx.gpu_lock.path


def _holder_for(client, runtime) -> dict:
    """让 API 观察到一把被别人持有的锁。"""
    from onyx.eval.gpu_lock import GpuLock

    lock = GpuLock(_lock_path_of(client),
                   owner="eval:intent_classification@qwen3.5:9b", stale_after_s=60)
    lock.acquire()
    lock.heartbeat(30, 240)
    try:
        return client.get("/api/gpu").json()
    finally:
        lock.release()
