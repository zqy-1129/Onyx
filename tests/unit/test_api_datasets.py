"""S24 验收：数据集导入的 HTTP 面，以及"导入完就能在界面跑起来"。

这一段最要紧的两个断言：
- 报错必须带**行号**。"格式错误"没法行动，"第 3 行不是合法 JSON"可以。
- 导入完的那个 id 必须能立刻用来发起评测。CLI 有 `import` 而解析器不认库，
  是这条闭环断过的地方。
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.api.routes import evals as evals_routes
from onyx.eval.datasets.loader import Dataset
from onyx.llm.providers.mock import MockScript
from onyx.runtime import build_runtime, sync_models
from onyx.settings import load_settings
from onyx.store.repos import EvalRepo

MODEL = "mock/echo"
TASK = "intent_classification"
SCRIPTS = {MODEL: MockScript(text="转账", in_tokens=120, out_tokens=4, done_reason="stop")}


def _jsonl(items: list[dict]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items)


def _cases(n: int = 3) -> list[dict]:
    return [
        {"id": f"c{i}", "input": {"instruction": f"帮我转 {i * 100} 元"},
         "expect": {"label": "转账"}, "tags": ["hard"] if i == 0 else []}
        for i in range(n)
    ]


@pytest.fixture
def client(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": (MODEL,)},
    )
    sync_models(runtime)
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")) as c:
        c.runtime = runtime  # 只是给测试一个读库的把手，不改 app 行为
        yield c
    runtime.close()


def _post(client, **kw) -> object:
    body = {"jsonl": _jsonl(_cases(3)), "name": "mini", "upstream": "unit-test", **kw}
    return client.post("/api/datasets", json=body)


# ── 导入 ──────────────────────────────────────────────────────────
def test_import_returns_provenance_and_shows_up_in_the_list(client):
    resp = _post(client, id="mini-v1", license="CC0-1.0")
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["id"] == "mini-v1" and body["n_cases"] == 3
    assert body["upstream"] == "unit-test" and body["license"] == "CC0-1.0"
    assert body["revision"].startswith("sha256:"), "没填 revision 也要有可比性判据"
    assert body["splits"]["default"] == 3 and body["splits"]["hard"] == 1
    assert body["replaced"] is False and body["warnings"] == []

    row = next(item for item in client.get("/api/datasets").json() if item["id"] == "mini-v1")
    assert row["selectable"] is True, "导入完就该能在界面选它跑评测"
    assert row["revision"] == body["revision"]

    record = EvalRepo(client.runtime.db).get_dataset("mini-v1")
    assert record is not None and record.n_cases == 3
    assert EvalRepo(client.runtime.db).count_cases("mini-v1") == 3


def test_broken_line_reports_its_number_not_just_invalid_jsonl(client):
    text = _jsonl(_cases(2)) + "\n{not json}\n"
    resp = client.post("/api/datasets", json={"jsonl": text, "name": "bad", "id": "bad-v1"})
    assert resp.status_code == 422, resp.text
    assert "第 3 行" in resp.json()["detail"]
    assert EvalRepo(client.runtime.db).get_dataset("bad-v1") is None, "坏文件不许留下半套数据"


@pytest.mark.parametrize("payload", ["", "   \n  \n"])
def test_empty_upload_is_refused(client, payload):
    resp = client.post("/api/datasets", json={"jsonl": payload, "name": "e", "id": "e-v1"})
    assert resp.status_code == 422
    assert EvalRepo(client.runtime.db).list_datasets() == []


def test_oversized_upload_is_refused_before_parsing(client, monkeypatch):
    """解析要占请求线程。不设上限就等于"任何人一个 POST 就能把看板钉住"。"""
    monkeypatch.setattr(evals_routes, "MAX_UPLOAD_BYTES", 64)
    resp = _post(client, id="big-v1")
    assert resp.status_code == 413, resp.text
    assert "上限" in resp.json()["detail"]


def test_case_ids_are_content_derived_and_unique_across_batches(client):
    """没写 id 的行按内容生成 id：重复导入同一份数据不许产生第二份样本。"""
    text = _jsonl([{"input": case["input"], "expect": case["expect"]} for case in _cases(2)])
    first = client.post("/api/datasets",
                        json={"jsonl": text, "name": "auto", "id": "auto-v1", "revision": "r1"})
    assert first.status_code == 201
    ids = [case.id for case in EvalRepo(client.runtime.db).list_cases("auto-v1")]
    assert len(set(ids)) == 2

    again = client.post("/api/datasets", json={
        "jsonl": text, "name": "auto", "id": "auto-v1", "revision": "r1", "allow_replace": True})
    assert again.status_code == 201
    assert EvalRepo(client.runtime.db).count_cases("auto-v1") == 2, "重导入不许把样本堆成 4 条"


# ── 覆盖必须显式 ──────────────────────────────────────────────────
def test_replacing_an_existing_id_needs_confirmation(client):
    assert _post(client, id="rep-v1", revision="r1").status_code == 201

    resp = _post(client, id="rep-v1", revision="r2")
    assert resp.status_code == 409, resp.text
    assert "allow_replace" in resp.json()["detail"]
    assert EvalRepo(client.runtime.db).get_dataset("rep-v1").revision == "r1", "没确认就不该动过"

    confirmed = _post(client, id="rep-v1", revision="r2", allow_replace=True)
    assert confirmed.status_code == 201
    body = confirmed.json()
    assert body["replaced"] is True
    assert any("不可比" in text for text in body["warnings"]), \
        "换了 revision 必须警告：历史分数指向的已经不是这份考卷了"
    assert EvalRepo(client.runtime.db).get_dataset("rep-v1").revision == "r2"


# ── 闭环：导入完就能跑 ────────────────────────────────────────────
def test_imported_dataset_can_start_a_run_from_the_api(client):
    assert _post(client, id="run-me-v1", revision="r1").status_code == 201

    resp = client.post("/api/runs", json={
        "task": TASK, "model": MODEL, "dataset": "run-me-v1", "seed": 5})
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["run_id"]

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = client.get(f"/api/runs/{run_id}/progress").json()
        if state["state"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert state["state"] == "done", state

    run = EvalRepo(client.runtime.db).get_run(run_id)
    assert run.n_cases == 3 and run.n_done == 3
    # 来历跟着结果走：矩阵与对比靠它回答"这两次是不是同一份考卷"
    assert run.dataset_id == "run-me-v1" and run.dataset_revision == "r1"
    grades = EvalRepo(client.runtime.db).list_grades(run_id)
    assert {g.case_id for g in grades} == {"c0", "c1", "c2"}


def test_unknown_dataset_in_a_run_request_lists_the_options(client):
    resp = client.post("/api/runs", json={"task": TASK, "model": MODEL, "dataset": "nope"})
    assert resp.status_code == 422
    detail = resp.json()["error"]["detail"]
    assert "intent_zh" in detail["available"], "内置与已导入的都要出现在可选项里"


def test_registered_dataset_without_cases_is_not_selectable(client):
    """dataset 行在而样本不在：选它跑评测只会得到一条 error，所以界面上就该灰掉。"""
    record, _ = Dataset(id="hollow-v1", cases=(), upstream="test", revision="r1").to_records()
    EvalRepo(client.runtime.db).upsert_dataset(record)

    row = next(item for item in client.get("/api/datasets").json() if item["id"] == "hollow-v1")
    assert row["selectable"] is False

    resp = client.post("/api/runs", json={"task": TASK, "model": MODEL, "dataset": "hollow-v1"})
    assert resp.status_code == 422, "校验也要跟 selectable 一致，不能让界面灰着而 API 放行"
