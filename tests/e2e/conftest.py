"""e2e（S34）：起真实 app + mock 引擎，种一份够六个页面取数的栈。

**为什么这一档存在**：S23–S29 的八处真实缺陷全部只能靠手工点浏览器发现，
而 `-m e2e` 的用例数一直是 0。这里补的是"页面取的那份数据"的回归保护：
跨端点的数字必须同源，grade 必须能落到真实 trace，SSE 必须真的能在 HTTP 层读到帧。

**它不覆盖什么也要写清**：本仓库没有任何浏览器驱动，所以渲染本身不在这里测
（前端的映射逻辑由 vitest 的纯函数测试钉住）。把这条边界写死，是为了不让"e2e 全绿"
被读成"界面没问题"。

种数据只走生产路径（gateway / EvalRunner / AlertService），不手写 SQL 造形状——
手写出来的行恰恰测不到"生产路径会不会写错"这件事。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from onyx.api.app import create_app
from onyx.eval.datasets.builtin.intent_zh import build_cases
from onyx.eval.datasets.loader import Dataset
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.llm.providers.mock import MockScript
from onyx.obs.alerts.channels import FileChannel
from onyx.obs.alerts.rules import AlertRule
from onyx.obs.alerts.service import AlertService
from onyx.runtime import build_runtime, sync_models

#: 两个模型答不同的标签 ⇒ 配对比较里两个方向都有翻转（净改善与净劣化都要有样本）
MODEL_A = "mock/zh-转账"
MODEL_B = "mock/zh-投诉"
#: 这个模型故意不带引擎计数 ⇒ 触发 NO_ENGINE_COUNT（error 级），Fleet 顶部那一行才有东西
MODEL_NOCOUNT = "mock/nocount"

SCRIPTS = {
    MODEL_A: MockScript(text="转账", in_tokens=120, out_tokens=4, done_reason="stop"),
    MODEL_B: MockScript(text="投诉", in_tokens=130, out_tokens=5, done_reason="stop"),
    MODEL_NOCOUNT: MockScript(text="好的", report_usage=False),
}
MODELS = (MODEL_A, MODEL_B, MODEL_NOCOUNT)


def _dataset(n: int = 8) -> Dataset:
    """内建意图集的前 n 条（标签天然混合：前 8 条是 投诉×5 / 转账×3）。"""
    return Dataset(id="intent_zh-v1", cases=tuple(build_cases()[:n]),
                   upstream="builtin", revision="seed=20261003")


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    """一颗种好的栈：3 次对话 + 2 条评测 run + 1 行告警触发历史。"""
    root = tmp_path_factory.mktemp("e2e")
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        db_path=root / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": MODELS},
    )
    alert_path = root / "alerts" / "alerts.jsonl"
    app = create_app(
        runtime, gpu_lock_path=root / "gpu.lock", sample_gpu=False,
        alert_rule=AlertRule(poll_s=0.05), alert_channels=(FileChannel(alert_path),),
    )
    with TestClient(app) as client:
        sync_models(runtime)

        # 工具注册表：Tool Bench 页要有东西可算（开销面板与审计都读这张表）。
        # 走 registry.register 而不是手写 SQL：那才是 `onyx tools import` 用的同一条路。
        _seed_tools(runtime)

        chat_ids = [
            _chat(client, MODEL_A), _chat(client, MODEL_B), _chat(client, MODEL_NOCOUNT),
        ]

        runs = [_run_eval(client, runtime, model) for model in (MODEL_A, MODEL_B)]

        # 告警：走生产的那条轮询判定（tick 是公开入口，测试与 CLI 都靠它，不靠睡觉）。
        # app 自己的线程也在轮询，所以"谁先投"不确定——页面测试断言的是"至少有一行"，
        # 而不是"恰好一行"。
        service = AlertService(runtime.db, rule=AlertRule(),
                               channels=[FileChannel(alert_path)])
        triggered = service.tick()

        yield SimpleNamespace(
            client=client, runtime=runtime, root=root, alert_path=alert_path,
            chat_ids=chat_ids, run_a=runs[0], run_b=runs[1],
            n_chats=len(chat_ids), triggered=triggered,
        )
    runtime.close()


def _chat(client, model: str) -> str:
    resp = client.post("/api/playground/chat", json={"model": model, "prompt": "帮我查一下天气"})
    assert resp.status_code == 200, resp.text
    return resp.json()["trace_id"]


def _seed_tools(runtime) -> None:
    """注册两个工具，让 Tool Bench 的开销/审计面板有东西可算。"""
    from onyx.runtime import build_tool_registry
    from onyx.tools.spec import ToolDef

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"],
        "additionalProperties": False,
    }
    registry = build_tool_registry(runtime.db, provider_id=runtime.provider.id)
    registry.register(ToolDef(
        name="get_weather", description="查询指定城市的当前天气，返回温度与风向。",
        kind="python_fn", side_effect="read", impl_ref="onyx.tools.builtin.echo:echo",
        parameters=schema,
        examples=[{"instruction": "北京天气", "expect": {"name": "get_weather",
                                                       "arguments": {"city": "北京"}}}],
    ))
    registry.register(ToolDef(
        name="send_mail", description="向指定收件人发送邮件，正文由调用方给。",
        kind="python_fn", side_effect="write", impl_ref="onyx.tools.builtin.echo:echo",
        parameters=schema,
    ))
    runtime.flush()


def _run_eval(client, runtime, model: str) -> str:
    """用 EvalRunner 真跑一条（和被 `onyx eval run` 用的同一套对象）。"""
    dataset = _dataset()
    task = IntentClassification(dataset, model=model)
    report = EvalRunner(runtime.gateway, _repo(runtime), task, dataset=dataset).run(
        RunConfig(model=model, seed=7)
    )
    assert report.status == "done", report.status
    return report.run_id


def _repo(runtime):
    from onyx.store.repos import EvalRepo

    return EvalRepo(runtime.db)
