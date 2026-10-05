"""S26 验收：模型治理的 HTTP 出口（pull / rm）与鉴权姿态。

`AdminProvider` 的 pull/delete 实现存在很久了，出口只有 Playground 的 unload 一个。
这里盯三件事：
- 危险动作要 `confirm=1`，且**没确认时什么都不能做**；
- 没有控制面的通道报 501 并说明为什么不做假的（AttributeError 变 500 是最难读的错误）；
- 只读看板上这些 POST 一律 403 —— "共享看板"不等于"共享操作台"。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.core.types import AdminResult, Cap
from onyx.runtime import build_runtime
from onyx.settings import load_settings
from onyx.store.repos import ModelRepo

MODEL = "mock/echo"


def _post(client, path, **params):
    return client.post(path, params=params)


@pytest.fixture
def app(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
    )
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")) as client:
        client.runtime = runtime
        yield client
    runtime.close()


@pytest.fixture
def ro(tmp_path):
    """带 token 的只读看板（共享给同事看的那一档）。"""
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "ro.sqlite", event_log=False,
    )
    client = TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock",
                                   token="t0ken", read_only=True))
    client.headers.update({"Authorization": "Bearer t0ken"})
    with client:
        yield client
    runtime.close()


# ── pull ──────────────────────────────────────────────────────────
def test_pull_requires_confirmation(app):
    client = app
    resp = client.post("/api/admin/models/pull", params={"name": "demo/giant"})
    assert resp.status_code == 400
    assert "confirm=1" in resp.json()["detail"]
    # 没确认就什么都不能发生：不能"先开始拉，等确认完再说"
    assert ModelRepo(client.runtime.db).count() == 0


def test_pull_syncs_the_catalog_afterwards(app):
    client = app
    resp = client.post("/api/admin/models/pull", params={"name": "demo/new", "confirm": 1})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True and body["action"] == "pull"
    assert body["models_synced"] >= 1, "拉完不刷清单，界面就会说「没这个模型」"

    names = {m["name"] for m in client.get("/api/models").json()}
    assert MODEL in names
    assert ModelRepo(client.runtime.db).find_by_name("mock-local", MODEL) is not None


def test_pull_reports_engine_rejection_as_502(app, monkeypatch):
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "pull",
                        lambda self, name, *, on_event=None: AdminResult(
                            ok=False, action="pull", error="404: model not found"))
    resp = _post(app, "/api/admin/models/pull", name="demo/missing", confirm=1)
    assert resp.status_code == 502
    assert "not found" in resp.json()["detail"]


# ── rm ────────────────────────────────────────────────────────────
def test_rm_requires_confirmation_and_keeps_history(app):
    client = app
    assert client.post("/api/admin/models/rm", params={"name": MODEL}).status_code == 400

    resp = _post(client, "/api/admin/models/rm", name=MODEL, confirm=1)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True and body["action"] == "delete"
    assert "历史 trace 与分数保留" in body["note"], \
        "必须说清删的是权重不是证据，否则人会以为历史分数也没了"


def test_rm_reports_engine_failure(app, monkeypatch):
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "delete",
                        lambda self, name: AdminResult(
                            ok=False, action="delete", error="model is in use"))
    resp = _post(app, "/api/admin/models/rm", name=MODEL, confirm=1)
    assert resp.status_code == 502
    assert "in use" in resp.json()["detail"]


# ── 没有控制面的通道 ──────────────────────────────────────────────
@pytest.mark.parametrize("path", [
    "/api/admin/models/pull", "/api/admin/models/rm", "/api/admin/models/unload",
])
def test_channels_without_a_control_plane_say_why(app, monkeypatch, path):
    """报 501 + 原因，而不是 AttributeError 变 500。

    兼容层没有统一的这些端点；编一个假的 unload 会让"显存已经让出来了"
    这种关键判断建立在谎话上 —— 那正是本项目最不能接受的一类错误。
    """
    from onyx.llm.providers.mock import MockProvider

    monkeypatch.setattr(MockProvider, "capabilities", lambda self: frozenset({Cap.CHAT}))
    resp = _post(app, path, name=MODEL, confirm=1)
    assert resp.status_code == 501, resp.text
    detail = resp.json()["detail"]
    assert "不暴露控制面" in detail and "谎话" in detail


# ── 鉴权姿态 ─────────────────────────────────────────────────────
@pytest.mark.parametrize("path", [
    "/api/admin/models/pull", "/api/admin/models/rm", "/api/admin/models/unload",
])
def test_read_only_board_refuses_model_governance(ro, path):
    resp = ro.post(path, params={"name": MODEL, "confirm": 1})
    assert resp.status_code == 403, resp.text
    assert resp.json()["error"]["code"] == "READ_ONLY"


def test_model_governance_needs_a_token(tmp_path):
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "auth.sqlite", event_log=False,
    )
    client = TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock",
                                   token="t0ken"))
    with client:
        resp = client.post("/api/admin/models/pull", params={"name": "x/y", "confirm": 1})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "UNAUTHORIZED"
        assert ModelRepo(runtime.db).count() == 0, "未授权的请求不能把模型写进库"
    runtime.close()
