"""e2e 两档共用的种数据动作（S34 的取数档 + S40 的浏览器档）。

**为什么要抽这一份**：两档唯一的区别是"客户端怎么连上应用"（TestClient / 真 uvicorn + httpx / 真浏览器），
种进去的数据必须是同一份。留两份就会漂，而漂掉之后浏览器档测的就不再是那条我们知道会坏的路径。

**种数据只走生产路径**（gateway / EvalRunner / registry.register / AlertService.tick），
不手写 SQL 造形状——手写出来的行恰恰测不到"生产路径会不会写错"这件事。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from onyx.eval.datasets.builtin.intent_zh import build_cases
from onyx.eval.datasets.loader import Dataset
from onyx.eval.runner import EvalRunner, RunConfig
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.llm.providers.mock import MockScript
from onyx.obs.alerts.channels import FileChannel
from onyx.obs.alerts.rules import AlertRule
from onyx.obs.alerts.service import AlertService
from onyx.store.repos import EvalRepo

#: 两个模型答不同的标签 ⇒ 配对比较里两个方向都有翻转（净改善与净劣化都要有样本）
MODEL_A = "mock/zh-转账"
MODEL_B = "mock/zh-投诉"
#: 这个模型故意不带引擎计数 ⇒ 触发 NO_ENGINE_COUNT（error 级），Fleet 顶部那一行才有东西
MODEL_NOCOUNT = "mock/nocount"
#: 引擎只报 6 个 in-token，而 prompt 是一大段中文 ⇒ 启发式分段必然数得比引擎多
#: ⇒ `parts.py` 把残差 clamp 成 0。这是 S39 那些文案唯一的可达路径，别用真引擎造它
MODEL_UNDERCOUNT = "mock/undercount"

#: 一段足够长的中文，让启发式（≈1 tok/汉字）远超引擎被假装报出来的 6
LONG_CJK_PROMPT = "仓库东侧的货架按编号排列，" * 12

SCRIPTS = {
    MODEL_A: MockScript(text="转账", in_tokens=120, out_tokens=4, done_reason="stop"),
    MODEL_B: MockScript(text="投诉", in_tokens=130, out_tokens=5, done_reason="stop"),
    MODEL_NOCOUNT: MockScript(text="好的", report_usage=False),
    MODEL_UNDERCOUNT: MockScript(text="收到", in_tokens=6, out_tokens=2, done_reason="stop"),
}
MODELS = (MODEL_A, MODEL_B, MODEL_NOCOUNT, MODEL_UNDERCOUNT)


def dataset(n: int = 8) -> Dataset:
    """内建意图集的前 n 条（标签天然混合：前 8 条是 投诉×5 / 转账×3）。"""
    return Dataset(id="intent_zh-v1", cases=tuple(build_cases()[:n]),
                   upstream="builtin", revision="seed=20261003")


def provider_kwargs() -> dict[str, Any]:
    return {"scripts": SCRIPTS, "models": MODELS}


def seed_tools(runtime: Any) -> None:
    """注册两个工具，让 Tool Bench 的开销/审计面板有东西可算。

    走 registry.register 而不是手写 SQL：那才是 `onyx tools import` 用的同一条路。
    """
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


def chat(client: Any, model: str, *, prompt: str = "帮我查一下天气") -> str:
    """一发对话。`client` 只要支持 `.post(path, json=...)`——TestClient 与 httpx.Client 都是。"""
    resp = client.post("/api/playground/chat", json={"model": model, "prompt": prompt})
    assert resp.status_code == 200, resp.text
    return resp.json()["trace_id"]


def run_eval(runtime: Any, model: str) -> str:
    """用 EvalRunner 真跑一条（和被 `onyx eval run` 用的同一套对象）。"""
    ds = dataset()
    task = IntentClassification(ds, model=model)
    report = EvalRunner(runtime.gateway, EvalRepo(runtime.db), task, dataset=ds).run(
        RunConfig(model=model, seed=7)
    )
    assert report.status == "done", report.status
    return report.run_id


def fire_alerts(runtime: Any, alert_path: Path) -> int:
    """走生产的那条轮询判定（tick 是公开入口，测试与 CLI 都靠它，不靠睡觉）。

    应用自己的线程也可能同时在轮询，所以返回值只用于"至少触发过"，
    任何页面断言都不许依赖它等于某个具体数。
    """
    service = AlertService(runtime.db, rule=AlertRule(), channels=[FileChannel(alert_path)])
    return len(service.tick())
