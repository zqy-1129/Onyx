"""S21：局域网共享时的那道闸（token + 只读）。

默认绑 127.0.0.1 从来不是鉴权，只是碰巧没人连得上。所以这里测的不是
"有没有 401"，而是三件容易做错的事：
1. 闸必须**覆盖所有数据出口**（含 /api/docs 与 /api/openapi.json——它们暴露能力面）；
2. SSE 走 query 参数（`EventSource` 设不了头），而其它请求走 header；
3. 没 token 时非回环绑定要**拒绝启动**，而不是"先起来再说"。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from onyx.api.app import create_app
from onyx.api.auth import install_auth, is_loopback
from onyx.cli import app
from onyx.llm.providers.mock import MockScript
from onyx.runtime import build_runtime

TOKEN = "s3cr3t-token-value"
runner = CliRunner()


@pytest.fixture
def runtime(tmp_path):
    built = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": {"mock/echo": MockScript(text="hi", in_tokens=3, out_tokens=2)},
                         "models": ("mock/echo",)},
    )
    yield built
    built.close()


def _client(runtime, tmp_path, **kw):
    app = create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock", **kw)
    return TestClient(app)


# ── 回环判定 ───────────────────────────────────────────────────────
@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.53"])
def test_loopback_hosts_recognised(host):
    assert is_loopback(host) is True


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "", "127.5.5.5.example"])
def test_non_loopback_hosts(host):
    """`0.0.0.0` 与 `::` 是"所有网卡"，不是"本机"——把它们当回环等于默认对全网开放。"""
    assert is_loopback(host) is False


def test_an_empty_token_refuses_to_pretend():
    """`install_auth(token="")` 会装出一道永远过不去的闸，看起来"有鉴权"其实全 401。"""
    from fastapi import FastAPI

    app = FastAPI()
    with pytest.raises(ValueError, match="非空 token"):
        install_auth(app, token="")


# ── 闸的覆盖面 ─────────────────────────────────────────────────────
@pytest.mark.parametrize("path", [
    "/api/fleet", "/api/models", "/api/traces", "/api/health",
    "/api/openapi.json", "/api/docs", "/api/stream",
])
def test_every_data_exit_is_gated(runtime, tmp_path, path):
    client = _client(runtime, tmp_path, token=TOKEN)

    assert client.get(path).status_code == 401, f"{path} 竟然不用 token"


def test_non_api_paths_are_not_gated(runtime, tmp_path):
    """SPA 外壳与静态资源不含数据，拦它们只会让页面白屏而保护不了任何东西。"""
    client = _client(runtime, tmp_path, token=TOKEN)

    assert client.get("/").status_code != 401


def test_header_token_works_and_wrong_one_does_not(runtime, tmp_path):
    client = _client(runtime, tmp_path, token=TOKEN)

    assert client.get("/api/fleet", headers={"authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert client.get("/api/fleet", headers={"authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/fleet", headers={"authorization": "Bearer"}).status_code == 401
    # 前缀大小写不敏感，但少了 Bearer 就是不给：不能"看起来像就放行"
    assert client.get("/api/fleet", headers={"authorization": TOKEN}).status_code == 401


def test_query_token_exists_because_eventsource_cannot_set_headers(runtime, tmp_path):
    client = _client(runtime, tmp_path, token=TOKEN)

    assert client.get(f"/api/fleet?token={TOKEN}").status_code == 200
    assert client.get("/api/fleet?token=wrong").status_code == 401


def test_401_body_is_actionable_and_leaks_nothing(runtime, tmp_path):
    client = _client(runtime, tmp_path, token=TOKEN)

    resp = client.get("/api/fleet")
    body = resp.json()

    assert body["error"]["code"] == "UNAUTHORIZED"
    assert "token" in body["error"]["detail"]["hint"]
    # 错误体里绝不能出现真 token 或堆栈——它是给外人看的第一份"说明书"
    assert TOKEN not in resp.text
    assert "Traceback" not in resp.text
    assert resp.headers.get("www-authenticate") == "Bearer"


# ── 只读模式 ───────────────────────────────────────────────────────
def test_read_only_blocks_writes_but_lets_the_dashboard_look(runtime, tmp_path):
    client = _client(runtime, tmp_path, token=TOKEN, read_only=True)
    headers = {"authorization": f"Bearer {TOKEN}"}

    assert client.get("/api/fleet", headers=headers).status_code == 200
    resp = client.post("/api/admin/models/unload?name=mock/echo&confirm=1", headers=headers)

    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "READ_ONLY"
    assert "--no-read-only" in resp.json()["error"]["detail"]["hint"]


def test_without_read_only_the_write_endpoint_is_reachable(runtime, tmp_path):
    """只读关掉后必须真的放行——否则"403 消失"这件事没人验证过，read_only 就成了永久挡路。"""
    client = _client(runtime, tmp_path, token=TOKEN, read_only=False)
    headers = {"authorization": f"Bearer {TOKEN}"}

    resp = client.post("/api/admin/models/unload?name=mock/echo&confirm=1", headers=headers)

    assert resp.status_code != 403


def test_no_token_configured_keeps_today_behaviour(runtime, tmp_path):
    client = _client(runtime, tmp_path)

    assert client.get("/api/fleet").status_code == 200


def test_the_gate_does_not_disturb_sse_handshake(runtime, tmp_path):
    """闸必须认得 SSE 只能走 query 参数，否则共享看板会"页面在、实时流没了"。

    这里刻意**不**建立长连接：`/api/stream` 是无终生成器，TestClient 读它会挂住；
    事件内容与 trace 关联由 test_api.py 的 SSE 用例覆盖。
    """
    client = _client(runtime, tmp_path, token=TOKEN)

    denied = client.get("/api/stream")
    assert denied.status_code == 401, "无 token 时流也要挡住"
    assert denied.json()["error"]["detail"]["hint"].startswith("带 Authorization")
    # 带 token 的 query 形式必须能通过闸（浏览器里 EventSource 只有这一条路）
    assert client.get(f"/api/fleet?token={TOKEN}").status_code == 200


# ── 启动姿态：非回环 + 无 token = 不起 ─────────────────────────────
@pytest.fixture
def serve_probe(monkeypatch):
    """拦住 uvicorn，记录 create_app 实际收到的 token / read_only。"""
    import uvicorn

    seen: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    import onyx.api.app as app_module

    real = app_module.create_app

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(app_module, "create_app", spy)
    return seen


def _serve(*argv: str):
    return runner.invoke(app, ["serve", *argv])


def test_non_loopback_without_token_refuses_to_start(tmp_path, monkeypatch, serve_probe):
    """这一条是整个 S21 的目的：把"不小心把看板开放给全网"变成起不来，而不是运行时才发现。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ONYX_SERVE_TOKEN", raising=False)

    result = _serve("--host", "0.0.0.0")

    assert result.exit_code == 2, result.output
    out = result.output + getattr(result, "stderr", "")
    assert "拒绝启动" in out
    assert "ONYX_SERVE_TOKEN" in out, "要给出可执行的下一步，不是只说不行"
    assert "--allow-insecure-local" in out
    assert serve_probe == {}, "拒绝启动时不该已经装好 app"


def test_non_loopback_with_token_starts_and_says_so(tmp_path, monkeypatch, serve_probe):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ONYX_SERVE_TOKEN", TOKEN)

    result = _serve("--host", "0.0.0.0", "--read-only")

    assert result.exit_code == 0, result.output
    assert serve_probe["token"] == TOKEN
    assert serve_probe["read_only"] is True
    assert "token 已启用" in result.output and "只读" in result.output


def test_explicit_flag_beats_env_token(tmp_path, monkeypatch, serve_probe):
    """优先级必须对 token 也成立——它是这一层最敏感的键。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ONYX_SERVE_TOKEN", "from-env")

    assert _serve("--token", "from-flag").exit_code == 0
    assert serve_probe["token"] == "from-flag"


def test_allow_insecure_local_is_a_one_time_declaration(tmp_path, monkeypatch, serve_probe):
    """明知非回环也要裸跑：必须每次显式说，所以这个开关**不放进配置文件**。"""
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ONYX_SERVE_TOKEN", raising=False)
    _write_insecure_config(tmp_path, monkeypatch)

    refused = _serve("--host", "192.168.1.10")
    assert refused.exit_code == 2, "配置文件里写了 allow_insecure 也不算数"

    allowed = _serve("--host", "192.168.1.10", "--allow-insecure-local")
    assert allowed.exit_code == 0, allowed.output
    assert serve_probe["token"] is None
    assert "警告" in allowed.output, "裸跑可以，但要每次提醒一句"


def _write_insecure_config(tmp_path, monkeypatch) -> None:
    path = tmp_path / "onyx.toml"
    path.write_text('[serve]\nallow_insecure_local = true\n', encoding="utf-8")
    monkeypatch.setenv("ONYX_CONFIG", str(path))


def test_config_show_never_prints_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ONYX_SERVE_TOKEN", TOKEN)

    result = runner.invoke(app, ["config", "show"])

    assert TOKEN not in result.output, "生效值报告不能把密钥打进终端（它会进日志与截图）"
    assert "serve.token" in result.output and "已设置" in result.output
