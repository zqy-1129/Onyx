"""S23 验收：从界面发起评测的 HTTP 面。

一个端点形状要盯住两件事：
- **提交立刻返回**：跑评测的是服务进程里的 worker 线程，不是请求线程。
  请求如果等着结果，HTTP 超时会让人以为失败了而 GPU 还在跑。
- **进度与取消都要说清出处**：CLI 发起的运行没有内存快照，
  把它显示成"可以取消"是撒谎；说"取消不了"才是对的。

只读看板上这些 POST 必须 403（S21 的姿态在这里继续成立）。
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.eval.datasets.builtin.intent_zh import build_cases
from onyx.eval.datasets.loader import Dataset
from onyx.eval.gpu_lock import GpuLock
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.llm.providers.mock import MockScript
from onyx.runtime import build_runtime, sync_models
from onyx.settings import load_settings
from onyx.store.records import RunRecord
from onyx.store.repos import EvalRepo

MODEL = "mock/echo"
TASK = "intent_classification"
SCRIPTS = {MODEL: MockScript(text="转账", in_tokens=120, out_tokens=4, done_reason="stop")}


def _wait_progress(client, run_id: str, pred, what: str, timeout: float = 20.0) -> dict:
    """轮询进度直到某个条件成立；超时就带着最后看到的现场失败。

    异步任务的测试最难受的是"失败了但看不出卡在哪"，所以把最后一次响应留在断言消息里。
    """
    deadline = time.monotonic() + timeout
    last: dict | None = None
    while time.monotonic() < deadline:
        last = client.get(f"/api/runs/{run_id}/progress").json()
        if pred(last):
            return last
        if last.get("state") == "error":
            # 已经失败了就没必要等到超时：现场（error/holder/done）直接说出来才有用
            raise AssertionError(f"提前失败（等 {what}）：{last}")
        time.sleep(0.02)
    raise AssertionError(f"没等到 {what}，最后看到：{last}")


def _wait_state(client, run_id: str, expected: str, timeout: float = 20.0) -> dict:
    return _wait_progress(client, run_id, lambda p: p.get("state") == expected,
                          f"state={expected}", timeout)


@pytest.fixture
def app(tmp_path):
    """一个跑过**一次 CLI 评测**的看板：既有真实分数可看，也有一条别人进程留下的记录。"""
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": (MODEL,)},
    )
    cases = [dict(case, expect={"label": "转账"}) for case in build_cases()[:3]]
    dataset = Dataset(id="intent_zh-v1", cases=tuple(cases), upstream="test", revision="r1")
    task = IntentClassification(dataset, model=MODEL)
    cli_run = EvalRunner(runtime.gateway, EvalRepo(runtime.db), task, dataset=dataset).run(
        RunConfig(model=MODEL, seed=7)
    )
    sync_models(runtime)
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")) as client:
        yield client, runtime, cli_run.run_id
    runtime.close()


# ── 任务与数据集清单 ──────────────────────────────────────────────
def test_tasks_are_listed_from_the_registry(app):
    client, _, _ = app
    rows = client.get("/api/tasks").json()
    ids = {row["id"] for row in rows}
    assert TASK in ids, "界面必须能看到内建任务，否则只能靠猜任务名"
    intent = next(row for row in rows if row["id"] == TASK)
    assert intent["name"] and intent["default_dataset"] == "intent_zh-v1"
    # 这是**注册表默认数据集**的规模，不是某次运行用的那份：提交前界面就该知道有多少题
    assert intent["n_cases"] == intent["splits"]["default"] > 0
    assert intent["error"] == ""
    assert "macro_f1" in intent["metrics"], "跑之前就该知道会产出哪些指标"


def test_a_broken_task_degrades_to_one_row_not_an_empty_list(app, monkeypatch):
    """一个坏插件任务把 /api/tasks 打成 500，界面看起来就是"一个任务都没有"。

    这比少一列糟糕得多：真相（这个任务坏了）会被读成"没有任何评测可跑"。
    """
    client, _, _ = app
    from onyx.api.routes import evals as evals_module

    real_build = evals_module.build_task

    def broken(task_id, **kw):
        if task_id == TASK:
            raise KeyError(f"未知任务 {task_id!r}")
        return real_build(task_id, **kw)

    monkeypatch.setattr(evals_module, "build_task", broken)
    rows = client.get("/api/tasks").json()
    assert len(rows) >= 2, "内建有两个任务，坏一个之后至少还要列得出另一个"
    mine = next(row for row in rows if row["id"] == TASK)
    assert "KeyError" in mine["error"], "坏任务要带着坏在哪里出现，而不是被静默丢掉"
    assert all(other["error"] == "" for other in rows if other["id"] != TASK)


def test_datasets_say_which_ones_the_ui_can_run(app):
    client, _, _ = app
    rows = client.get("/api/datasets").json()
    assert rows and all(row["selectable"] for row in rows), "内建数据集在库里就是可选的"


# ── 提交 ──────────────────────────────────────────────────────────
def test_submit_returns_202_and_the_run_finishes_outside_the_request(app):
    client, runtime, _ = app
    started = time.monotonic()
    resp = client.post("/api/runs", json={"task": TASK, "model": MODEL, "limit": 2, "seed": 3})
    assert resp.status_code == 202, resp.text
    assert time.monotonic() - started < 0.5, "请求线程不该被占到评测跑完"
    body = resp.json()
    run_id = body["run_id"]
    assert body["state"] in ("queued", "running") and body["position"] >= 0
    assert run_id and run_id != "unknown"

    assert _wait_state(client, run_id, "done")["done"] == 2
    row = EvalRepo(runtime.db).get_run(run_id)
    assert row is not None and row.status == "done" and row.n_done == 2
    assert row.config["trigger"] == "api", "出处要落库，否则重启后分不清僵尸与真跑"
    assert row.seed == 3
    # 每个分数仍然点得进真实 trace（这条主张不因触发方式而变）
    grades = client.get(f"/api/runs/{run_id}/grades").json()
    assert len(grades) == 2 and all(grade["trace_id"] for grade in grades)


def test_submit_shows_up_in_the_queue(app):
    client, _, _ = app
    run_id = client.post("/api/runs", json={"task": TASK, "model": MODEL, "limit": 2}).json()["run_id"]
    queue = client.get("/api/queue").json()
    assert queue["max_pending"] >= 1
    assert run_id in {job["run_id"] for job in queue["jobs"]}
    _wait_state(client, run_id, "done")
    settled = next(job for job in client.get("/api/queue").json()["jobs"] if job["run_id"] == run_id)
    assert settled["state"] == "done" and settled["source"] == "service"


def test_invalid_submit_returns_422_with_the_options(app):
    client, _, _ = app
    resp = client.post("/api/runs", json={"task": "no_such_task", "model": MODEL})
    assert resp.status_code == 422, resp.text
    error = resp.json()["error"]
    assert error["code"] == "EVAL_ERROR"
    assert TASK in error["detail"]["available"], "报错要给出可选项，否则只能靠猜"
    assert client.get("/api/runs", params={"limit": 50}).json()[0]["config"]["trigger"] == "cli"


def test_submit_rejects_a_model_that_is_not_in_the_catalog(app):
    client, _, _ = app
    resp = client.post("/api/runs", json={"task": TASK, "model": "mock/typo"})
    assert resp.status_code == 422
    assert "不在已同步的清单里" in resp.json()["error"]["message"]


def test_submit_rejects_file_datasets_over_http(app):
    """`file:<路径>` 是 CLI 的本地便利；放进 HTTP 就是"请求体写什么就读什么"。"""
    client, _, _ = app
    resp = client.post("/api/runs", json={"task": TASK, "model": MODEL, "dataset": "file:/etc/passwd"})
    assert resp.status_code == 422
    assert "file:" in resp.json()["error"]["message"]


# ── 进度出处 ──────────────────────────────────────────────────────
def test_progress_for_a_cli_run_says_it_is_not_cancellable(app):
    client, _, cli_run_id = app
    body = client.get(f"/api/runs/{cli_run_id}/progress").json()
    assert body["source"] == "db" and body["state"] == "done"
    assert body["cancellable"] is False
    assert body["done"] == body["total"] == 3


def test_progress_of_an_unknown_run_is_404(app):
    client, _, _ = app
    assert client.get("/api/runs/00NOPE/progress").status_code == 404


# ── 续跑（S26）────────────────────────────────────────────────────
def test_resume_continues_the_interrupted_run_in_place(app, monkeypatch):
    """界面上点「续跑」必须接回同一条 run：另起 id 会把一次评测的历史劈成两半。"""
    client, runtime, _ = app
    original = runtime.provider.generate

    def slow(req, **kw):        # 给取消留出真实的时间窗（每条样本 ~60ms）
        time.sleep(0.06)
        return original(req, **kw)

    monkeypatch.setattr(runtime.provider, "generate", slow)
    run_id = client.post("/api/runs", json={"task": TASK, "model": MODEL, "limit": 4}).json()["run_id"]
    _wait_progress(client, run_id, lambda p: p["done"] >= 1, "至少跑完一条")
    assert client.post(f"/api/runs/{run_id}/cancel").json()["cancelled"] is True
    _wait_state(client, run_id, "cancelled")
    graded_before = len(client.get(f"/api/runs/{run_id}/grades").json())
    assert 1 <= graded_before < 4

    resp = client.post("/api/runs", json={
        "task": TASK, "model": MODEL, "limit": 4, "resume_run_id": run_id})
    assert resp.status_code == 202, resp.text
    assert resp.json()["run_id"] == run_id, "续跑必须是同一条 run"

    final = _wait_state(client, run_id, "done")
    assert final["total"] == 4
    body = client.get(f"/api/runs/{run_id}").json()["run"]
    assert body["status"] == "done" and body["n_done"] == 4
    assert body["aggregate"]["resumed"] is True
    assert body["aggregate"]["already_graded_before"] == graded_before
    assert len(client.get(f"/api/runs/{run_id}/grades").json()) == 4


def test_resume_rejects_a_finished_run(app):
    """已跑完的 run 续它只会重扫一遍已评过的 case —— 那是纯浪费的 GPU 时间。"""
    client, _, cli_run_id = app
    resp = client.post("/api/runs", json={
        "task": TASK, "model": MODEL, "limit": 3, "resume_run_id": cli_run_id})
    assert resp.status_code == 422, resp.text
    assert "已经跑完" in resp.json()["error"]["message"]


def test_resume_rejects_an_unknown_id_and_a_mismatched_task(app):
    client, _, cli_run_id = app
    bad = client.post("/api/runs", json={
        "task": TASK, "model": MODEL, "limit": 1, "resume_run_id": "never-existing"})
    assert bad.status_code == 422
    assert "找不到要续跑的 run" in bad.json()["error"]["message"]

    mismatch = client.post("/api/runs", json={
        "task": "tool_selection", "model": MODEL, "limit": 1, "resume_run_id": cli_run_id})
    assert mismatch.status_code == 422
    assert "任务与模型" in mismatch.json()["error"]["message"]


def test_cancel_a_foreign_running_run_is_409(app):
    """别的进程持有锁在跑的运行，本服务既没有它的取消开关，也不该动它的锁。"""
    client, runtime, cli_run_id = app
    repo = EvalRepo(runtime.db)
    repo.insert_run(RunRecord(
        id="foreign-run", task_id=repo.get_run(cli_run_id).task_id, model_id=MODEL,
        started_at="2026-10-04T00:00:00+00:00", status="running", config={"trigger": "cli"},
    ))
    resp = client.post("/api/runs/foreign-run/cancel")
    assert resp.status_code == 409, resp.text
    assert "不是本服务发起的" in resp.json()["detail"]
    assert repo.get_run("foreign-run").status == "running", "取消失败就不该改别人的状态"


def test_cancel_a_finished_run_is_idempotent(app):
    client, _, cli_run_id = app
    resp = client.post(f"/api/runs/{cli_run_id}/cancel")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["cancelled"] is False and body["state"] == "done"
    assert "终态" in body["message"]


def test_cancel_while_it_waits_for_the_gpu_lock(app, tmp_path):
    """锁被别人占着时提交：界面要能看到持有者，并且取消真的生效。"""
    client, runtime, _ = app
    holder = GpuLock(tmp_path / "gpu.lock", owner="cli-eval", poll_s=0.02, stale_after_s=60)
    holder.acquire()
    holder.heartbeat(1, 10)
    try:
        run_id = client.post("/api/runs", json={"task": TASK, "model": MODEL, "limit": 2}).json()["run_id"]
        _wait_progress(
            client, run_id,
            lambda p: p["state"] == "running" and p["holder"] == "cli-eval",
            "running 且看得到 GPU 持有者",
        )
        resp = client.post(f"/api/runs/{run_id}/cancel")
        assert resp.status_code == 200 and resp.json()["cancelled"] is True
        _wait_state(client, run_id, "cancelled")
        again = client.post(f"/api/runs/{run_id}/cancel")
        assert again.status_code == 200
        assert "幂等" in again.json()["message"], "重复取消要说不改任何东西，而不是演一遍成功"
    finally:
        holder.release()
    assert EvalRepo(runtime.db).get_run(run_id) is None, "一条样本都没跑，库里不该有这条 run"


def test_submitted_runs_share_the_apps_gpu_lock(app):
    """界面发起的评测必须进**同一把**机器级锁。

    锁路径一旦分叉，"看板发起的评测"就会与 Playground / CLI 并发跑，
    而现象不是报错是所有延迟数字失真——这正是这把锁存在的唯一理由。
    """
    client, _, _ = app
    state = client.app.state.onyx
    assert str(state.eval_service.lock_path) == str(state.gpu_lock.path)
    assert state.eval_service.stale_after_s == state.gpu_lock.stale_after_s
    assert client.get("/api/gpu").json()["busy"] is False, "评测跑完必须放锁"


# ── 只读与鉴权 ────────────────────────────────────────────────────
def test_read_only_dashboard_refuses_to_run_evals(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "ro.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": (MODEL,)},
    )
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock",
                               token="secret-token", read_only=True)) as client:
        client.headers.update({"Authorization": "Bearer secret-token"})
        assert client.get("/api/tasks").status_code == 200
        resp = client.post("/api/runs", json={"task": TASK, "model": MODEL, "limit": 1})
        assert resp.status_code == 403, resp.text
        assert resp.json()["error"]["code"] == "READ_ONLY"
        assert client.get("/api/runs", params={"limit": 5}).json() == [], "只读看板一条都没跑过"


def test_submit_requires_a_token(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "auth.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": (MODEL,)},
    )
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock",
                               token="secret-token")) as client:
        resp = client.post("/api/runs", json={"task": TASK, "model": MODEL, "limit": 1})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"
        assert EvalRepo(runtime.db).list_runs(limit=5) == []
        # 未授权的请求连入队都没发生：队列里必须一个任务都没有
        assert client.get("/api/queue", headers={"Authorization": "Bearer secret-token"}
                          ).json()["jobs"] == []
