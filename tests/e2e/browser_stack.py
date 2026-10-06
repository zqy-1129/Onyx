"""真浏览器那一档的进程编排（S40）。

**这里唯一的工作是"把两个进程起干净、收干净"**：一个后端（真 uvicorn，随机端口，线程内）
与一个前端（vite dev，另一个随机端口，`ONYX_API` 指向刚才那个后端）。
种数据的动作复用 `seed.py`——浏览器档与取数档测的必须是同一份形状。

四条不是洁癖的规矩：

1. **绝不复用"别人已经起好的服务"**。开发机上 8787/5173 常常正被人开着；真去连它们，
   测试就变成了"看我此刻的看板"，既不确定也测不到本次代码（`onyx serve` 没有热重载）。
   所以端口一律随机、vite 带 `--strictPort`，起不来就是错，不许回落到别人的端口。
2. **收尾按端口复核，不按工具的返回值**。本机实测过：停止工具报"成功"而 `node.exe`
   仍占着 5173（探活照样 200）。所以 kill 之后要确认那个端口不再有人听。
3. **杀整棵进程树**。vite 会派生 esbuild；只杀父进程会留下孤儿继续占端口。
4. **provider 一律 mock**。这一档不碰 GPU，也就不该去抢那把机器级锁（评测/Playground/live 互斥）。
"""

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from onyx.api.app import create_app
from onyx.obs.alerts.channels import FileChannel
from onyx.obs.alerts.rules import AlertRule
from onyx.runtime import build_runtime, sync_models
from tests.e2e.seed import (
    LONG_CJK_PROMPT,
    MODEL_A,
    MODEL_B,
    MODEL_NOCOUNT,
    MODEL_UNDERCOUNT,
    chat,
    fire_alerts,
    provider_kwargs,
    run_eval,
    seed_tools,
)

WEB_DIR = Path(__file__).resolve().parents[2] / "onyx" / "web"
VITE_BIN = WEB_DIR / "node_modules" / "vite" / "bin" / "vite.js"

#: 前端冷启动（dev server 按需转换）比后端慢得多。这些是**等待上限**，不是"睡这么久"
WEB_BOOT_S = 60.0
BACKEND_BOOT_S = 30.0


def _free_port() -> int:
    """要一个当下没人用的端口：绑 0 让系统挑，读完就放。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _port_listening(port: int) -> bool:
    with contextlib.suppress(OSError), socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            return True
    return False


def _wait_until(probe, *, timeout_s: float, what: str) -> None:
    """条件等待。慢机器上多等一会儿是对的，把超时写进 fail 的文案里更有用的是"是谁没就绪"。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if probe():
            return
        time.sleep(0.1)
    pytest.fail(f"{what} 在 {timeout_s:.0f}s 内没就绪——这是编排没起来，不是应用的 bug")


def _backdate(runtime, trace_id: str, *, hours: int) -> None:
    """把一条 trace 的整体时间往前挪，用来制造跨时间桶的样本。

    只改 `started_at` 会让 `finished_at` 落在它后面几个小时，页面上的时长变成一个假数——
    所以两个一起挪。`first_token_at` 清掉：这一发不是流式的话它本来就该是空，
    留着一个没挪的墙钟时刻比留空更容易骗人。
    """
    from dataclasses import replace
    from datetime import datetime, timedelta

    from onyx.store.repos import TraceRepo

    repo = TraceRepo(runtime.db)
    record = repo.get(trace_id)
    if record is None:
        return
    delta = timedelta(hours=hours)

    def _shift(value: str | None) -> str | None:
        return None if not value else (datetime.fromisoformat(value) - delta).isoformat()

    repo.upsert(replace(record, started_at=_shift(record.started_at),
                         finished_at=_shift(record.finished_at), first_token_at=None))
    runtime.flush()


def _kill_tree(pid: int) -> None:
    """杀整棵进程树。Windows 上 taskkill /T；POSIX 上子进程是同组（start_new_session）。"""
    if os.name == "nt":
        with contextlib.suppress(OSError):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        os.killpg(os.getpgid(pid), 15)


@pytest.fixture(scope="session")
def backend_url(tmp_path_factory):
    """真 uvicorn（线程内、随机端口、只绑回环），并把数据按生产路径种好。"""
    import uvicorn

    root = tmp_path_factory.mktemp("browser")
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        db_path=root / "onyx.sqlite", event_log=False,
        provider_kwargs=provider_kwargs(),
    )
    alert_path = root / "alerts" / "alerts.jsonl"
    app = create_app(
        runtime, gpu_lock_path=root / "gpu.lock", sample_gpu=False,
        alert_rule=AlertRule(poll_s=0.05), alert_channels=(FileChannel(alert_path),),
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, name="browser-uvicorn", daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    _wait_until(lambda: _health_ok(base), timeout_s=BACKEND_BOOT_S, what=f"后端 {base}")

    sync_models(runtime)
    seed_tools(runtime)
    with httpx.Client(base_url=base, timeout=20.0) as client:
        trace_ids = [chat(client, m) for m in (MODEL_A, MODEL_B, MODEL_NOCOUNT)]
        # 这一发是 clamp 形状（引擎少报、启发式数得多），S39 的那三段文案只有它能到达
        clamped_trace_id = chat(client, MODEL_UNDERCOUNT, prompt=LONG_CJK_PROMPT)
        run_a = run_eval(runtime, MODEL_A)
        run_b = run_eval(runtime, MODEL_B)
        fire_alerts(runtime, alert_path)

    # 把最早那一发整体挪到三小时前：不然整栈全落在同一个小时里，Ledger 的时序面板只会说
    # "样本不足，画不出趋势"，S39 那条"没测到的速率不许画成 0"就没有可看的地方。
    # 走 TraceRepo.upsert（生产路径，就是 sink 写这一行用的同一个方法），不手写 SQL。
    _backdate(runtime, trace_ids[0], hours=3)

    try:
        yield SimpleNamespace(base=base, port=port, runtime=runtime, root=root,
                              trace_ids=trace_ids, clamped_trace_id=clamped_trace_id,
                              run_a=run_a, run_b=run_b)
    finally:
        server.should_exit = True
        time.sleep(0.2)
        runtime.close()


def _health_ok(base: str) -> bool:
    with contextlib.suppress(httpx.HTTPError, OSError):
        return httpx.get(f"{base}/api/health", timeout=1.0).status_code == 200
    return False


def _serves_app(base: str) -> bool:
    """前端就绪 = 首页真的挂着这个应用。"""
    with contextlib.suppress(httpx.HTTPError, OSError):
        text = httpx.get(base, timeout=2.0).text
        return "<div id=\"root\"" in text or "onyx" in text.lower()
    return False


@pytest.fixture(scope="session")
def web_url(backend_url):
    """vite dev 指向**这个**后端（ONYX_API），端口随机且 strict——不许落到别人的 5173 上。"""
    node = shutil.which("node") or shutil.which("node.exe")
    if node is None:
        pytest.skip("这一档需要 node（起 vite dev server）")
    if not VITE_BIN.exists():
        pytest.skip(f"缺 {VITE_BIN}——先在 onyx/web 跑一次 npm install")

    port = _free_port()
    proc = subprocess.Popen(
        [node, str(VITE_BIN), "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
        cwd=str(WEB_DIR), env={**os.environ, "ONYX_API": backend_url.base},
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        start_new_session=(os.name != "nt"),
    )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_until(lambda: proc.poll() is None and _serves_app(base),
                    timeout_s=WEB_BOOT_S, what=f"前端 {base}")
        yield SimpleNamespace(base=base, port=port, api=backend_url.base)
    finally:
        _kill_tree(proc.pid)
        with contextlib.suppress(Exception):
            proc.wait(timeout=10)
        # 按端口复核：工具说"已停止"不算，端口不再听才算（本机真踩过它骗人）
        deadline = time.monotonic() + 10
        while _port_listening(port) and time.monotonic() < deadline:
            _kill_tree(proc.pid)
            time.sleep(0.3)
        assert not _port_listening(port), \
            f"vite 的某个子进程还占着 {port}——留着它，下一条测试会连到别人的服务"


@pytest.fixture(scope="session")
def browser_type():
    """优先吃系统里已有的浏览器（`channel="msedge"`，不下载），CI 上退回 playwright chromium。

    两个都起不来 ⇒ **skip 并说清要装什么**。"这台机器没浏览器"与"应用坏了"是两件事，
    混成一个红会让人去查错的方向。
    """
    sync = pytest.importorskip("playwright.sync_api", reason="这一档要 playwright（dev extra）")
    pw = sync.sync_playwright().start()
    launched = None
    problems: list[str] = []
    for channel in ("msedge", "chrome", None):
        try:
            launched = (pw.chromium.launch(channel=channel, headless=True) if channel
                        else pw.chromium.launch(headless=True))
            break
        except Exception as exc:
            problems.append(f"{channel or 'chromium'}: {str(exc).splitlines()[0][:120]}")
    if launched is None:
        pw.stop()
        pytest.skip("没有可用的浏览器（依次试过 msedge / chrome / playwright chromium）。"
                    + " | ".join(problems)
                    + "；CI 里对应的一步是 `playwright install chromium`")
    try:
        yield launched
    finally:
        launched.close()
        pw.stop()


@pytest.fixture
def page(browser_type, web_url):
    """每条测试一个全新 context：localStorage、滚动位置与 SSE 订阅都不许在测试之间串味。"""
    context = browser_type.new_context(viewport={"width": 1440, "height": 900})
    view = context.new_page()
    try:
        yield view
    finally:
        context.close()


def open_app(page, web_base: str, route: str, *, wait_for: str) -> None:
    """打开某个页面并等到"它真的渲染出东西了"。

    `wait_for` 挑的是**这个页面独有的文案**：导航条六页都在，等它只能证明应用起来了。
    """
    page.goto(f"{web_base}/#{route}", wait_until="domcontentloaded")
    page.get_by_text(wait_for).first.wait_for(state="visible", timeout=20_000)


def wait_loaded(page, *, timeout_ms: int = 20_000) -> None:
    """等到页面**不再处于加载态**。

    各 Panel 各自取数各自渲染：标题先出来、内容后到，所以"看到标题就断言没有骨架屏"
    测的是时序不是应用（真踩过：Tool Bench 那一瞬正好挂着 4 块骨架屏）。
    """
    with contextlib.suppress(Exception):        # 一块骨架屏都没有时 wait 直接抛，那就是通过
        page.wait_for_selector(".skeleton", state="detached", timeout=timeout_ms)
