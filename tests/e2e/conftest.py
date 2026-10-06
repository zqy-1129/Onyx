"""e2e（S34）：起真实 app + mock 引擎，种一份够六个页面取数的栈。

**为什么这一档存在**：S23–S29 的八处真实缺陷全部只能靠手工点浏览器发现，
而 `-m e2e` 的用例数一直是 0。这里补的是"页面取的那份数据"的回归保护：
跨端点的数字必须同源，grade 必须能落到真实 trace，SSE 必须真的能在 HTTP 层读到帧。

**它不覆盖什么也要写清**：这一档不起浏览器——渲染由 `-m browser`（S40）那一档管，
前端映射由 vitest 的纯函数测试钉住。把这条边界写死，是为了不让"e2e 全绿"
被读成"界面没问题"。

种数据的动作住在 `seed.py`：浏览器档必须共用它，否则两档测的不是同一份形状。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.obs.alerts.channels import FileChannel
from onyx.obs.alerts.rules import AlertRule
from onyx.runtime import build_runtime, sync_models
from tests.e2e.browser_stack import (  # noqa: F401  浏览器档（-m browser）的 fixture：在这里才被发现
    backend_url,
    browser_type,
    page,
    web_url,
)
from tests.e2e.seed import (
    MODEL_A,
    MODEL_B,
    MODEL_NOCOUNT,
    chat,
    fire_alerts,
    provider_kwargs,
    run_eval,
    seed_tools,
)


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """一颗种好的栈：3 次对话 + 2 条评测 run + 1 行告警触发历史。"""
    root = tmp_path_factory.mktemp("e2e")
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
    with TestClient(app) as client:
        sync_models(runtime)
        seed_tools(runtime)
        chat_ids = [chat(client, m) for m in (MODEL_A, MODEL_B, MODEL_NOCOUNT)]
        runs = [run_eval(runtime, m) for m in (MODEL_A, MODEL_B)]
        triggered = fire_alerts(runtime, alert_path)

        yield SimpleNamespace(
            client=client, runtime=runtime, root=root, alert_path=alert_path,
            chat_ids=chat_ids, run_a=runs[0], run_b=runs[1],
            n_chats=len(chat_ids), triggered=triggered,
        )
    runtime.close()
