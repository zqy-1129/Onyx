"""e2e（S34）：SSE 在真 HTTP 连接上能读到帧，以及"断链会被发现"这件事本身有测试。

当年 SseBroker 只实现了 `publish()` 没实现 `emit()`，挂进 Fanout 后每个事件都失败，
而 Fanout 的异常隔离把它吞成一行 warning —— SSE 从上线起就是静默失效的，
四个页面的测试与构建全绿都没发现。现有单测补了协议一致性与 Fanout 联通，
但**从没在 HTTP 层读过一帧**：`test_api_auth.py` 里还特意绕开（无终生成器会把 TestClient 挂住）。

**为什么起真 uvicorn 而不是 TestClient**：实测 TestClient 的传输层会把流式响应缓冲住——
一次请求之后 broker `published=9`、订阅者 1 个，而客户端 `iter_lines()` 一行都读不到。
那是测试装置的限制不是应用的 bug，可如果 e2e 就这么写，它会以"读不到"为由永远红，
或者更糟：被人改成"只断言订阅成功"而悄悄失去守断链的能力。
所以这里起一个 127.0.0.1 上的真服务器 + 真 httpx 流式读取（不出机器、不碰网络）。
"""

from __future__ import annotations

import json
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from onyx.api.app import create_app
from onyx.llm.providers.mock import MockScript
from onyx.runtime import build_runtime

pytestmark = pytest.mark.e2e

MODEL = "mock/stream"
SCRIPTS = {MODEL: MockScript(text="好的", in_tokens=42, out_tokens=7, done_reason="stop")}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live(tmp_path):
    """一个真的 serve：只绑 127.0.0.1，随测试起停。"""
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        db_path=tmp_path / "sse.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": (MODEL,)},
    )
    app = create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock", sample_gpu=False)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="e2e-uvicorn", daemon=True)
    thread.start()
    deadline = time.monotonic() + 20.0
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "e2e 的假服务器没起来，后面的断言都不用看了"

    base = f"http://127.0.0.1:{port}"
    # 只带 base url、app 与 runtime：测试里不需要再传字符串或猜属性
    yield SimpleNamespace(base=base, app=app, runtime=runtime)
    server.should_exit = True
    thread.join(5.0)
    runtime.close()


def _open_stream(base: str, want: int = 3, timeout: float = 8.0) -> tuple[list[dict], str | None]:
    """读 SSE 前 `want` 个 data 帧，返回（帧，错误）。"""
    frames: list[dict] = []
    error: str | None = None
    with httpx.Client(timeout=timeout) as client, client.stream("GET", f"{base}/api/stream") as resp:
        if resp.status_code != 200:
            return [], f"HTTP {resp.status_code}"
        try:
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    frames.append(json.loads(line[6:]))
                    if len(frames) >= want:
                        break
        except Exception as exc:  # 读流坏了要变成断言信息，而不是让测试挂死
            error = f"{type(exc).__name__}: {exc}"
    return frames, error


def test_stream_delivers_real_events_over_http(live):
    """订阅 → 真发一次请求 → 流里必须出现这次请求的事件。"""
    assert any(getattr(s, "name", "") == "sse" for s in live.runtime.events.sinks), \
        "broker 没挂上事件总线时 SSE 会静默失效——先把这件事钉在断言里"

    frames: list[dict] = []
    done = threading.Event()
    trace_id: list[str] = []

    def _reader() -> None:
        got, err = _open_stream(live.base, want=4, timeout=10.0)
        frames.extend(got)
        if err:
            frames.append({"__error__": err})
        done.set()

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    time.sleep(0.5)  # 先让订阅建立：broker 只推给当时已存在的订阅者

    resp = httpx.post(f"{live.base}/api/playground/chat",
                      json={"model": MODEL, "prompt": "说一句"}, timeout=10.0)
    assert resp.status_code == 200, resp.text
    trace_id.append(resp.json()["trace_id"])
    assert done.wait(12.0), "读流的线程没结束：SSE 很可能是静默失效的"

    assert frames, "一帧都没读到（订阅或推送断了）"
    assert frames[0]["type"] == "hello", "订阅成功的第一帧必须是 hello"
    types = {frame.get("type") for frame in frames}
    assert "trace_start" in types, f"流里没有请求的开始帧，只有 {types}"
    assert trace_id[0] in {frame.get("trace_id") for frame in frames}, \
        "读到的事件不属于这次请求"


def test_a_severed_link_is_caught_not_silently_green(live):
    """把 broker 从事件总线上摘掉（就是当年那个故障的形状）⇒ 流里只剩 hello，没有事件。

    这条是上一条的**自检**：摘掉之后上一条必须会红。
    否则"我们有一条测试守着 SSE"这句话本身就是空的。
    """
    original = list(live.runtime.events.sinks)
    live.runtime.events.sinks[:] = [s for s in original if getattr(s, "name", "") != "sse"]
    try:
        frames: list[dict] = []
        done = threading.Event()

        def _reader() -> None:
            got, _err = _open_stream(live.base, want=2, timeout=3.0)
            frames.extend(got)
            done.set()

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        time.sleep(0.4)
        httpx.post(f"{live.base}/api/playground/chat",
                   json={"model": MODEL, "prompt": "说一句"}, timeout=10.0)
        assert done.wait(8.0)
        assert frames and frames[0]["type"] == "hello", "连订阅都没建立，这条自检就没在测任何东西"
        events = [f for f in frames if f.get("type") != "hello"]
        assert not events, "断链之后还收得到事件 ⇒ 上一条的断言已经不可信，去修那条"
    finally:
        live.runtime.events.sinks[:] = original
