"""S15 验收：矩阵与配对对比的 API。

这两个端点是 Eval UI 的全部数据来源，所以断言的重点不是"返回 200"，而是：
- 矩阵每格带**数据集来历**，跨数据集的网格必须自带警告（否则 UI 会画出一张不可比的表）；
- 对比返回的是**派生后的结论**（净改善/净劣化 + 配对 CI），前端不再自己算差值——
  两处各算一遍必然漂移，而漂移出来的差别看起来像"模型变了"；
- 每个劣化 case 带两条 trace_id，界面才能并排打开它们。
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

#: 两个模型在同一份数据上的表现：a 全对，b 后两条答错 ⇒ 恰好 2 条劣化
SCRIPTS = {
    "mock/a": MockScript(text="转账", in_tokens=100, out_tokens=4, done_reason="stop"),
    "mock/b": [
        MockScript(text="转账", in_tokens=100, out_tokens=4, done_reason="stop"),
        MockScript(text="转账", in_tokens=100, out_tokens=4, done_reason="stop"),
        MockScript(text="查余额", in_tokens=100, out_tokens=4, done_reason="stop"),
        MockScript(text="投诉", in_tokens=100, out_tokens=4, done_reason="stop"),
    ],
}


def _dataset(n: int = 4) -> Dataset:
    cases = []
    for index, raw in enumerate(build_cases()[:n]):
        case = dict(raw)
        case["id"] = f"izh-{index}"
        case["input"] = {"instruction": f"第 {index} 条：帮我转 500 给张伟"}
        case["expect"] = {"label": "转账", "must_call": True}
        case["tags"] = ["test"]
        case["ord"] = index
        cases.append(case)
    return Dataset(id="intent_zh-v1", cases=tuple(cases), upstream="test", revision="seed=1")


@pytest.fixture
def env(tmp_path):
    """同一个 runtime 上跑两次评测（两个模型、同一份数据），API 测的是"有对比可做时返回什么"。"""
    runtime = build_runtime(
        provider_kind="mock", provider_id="mock-local", base_url="mock://",
        settings=load_settings(tmp_path), db_path=tmp_path / "onyx.sqlite", event_log=False,
        provider_kwargs={"scripts": SCRIPTS, "models": tuple(SCRIPTS)},
    )
    dataset = _dataset()
    repo = EvalRepo(runtime.db)
    reports = []
    for model in ("mock/a", "mock/b"):
        task = IntentClassification(dataset, model=model)
        reports.append(EvalRunner(runtime.gateway, repo, task, dataset=dataset).run(
            RunConfig(model=model, seed=7)
        ))
    runtime.flush()
    # 锁路径指到 tmp：默认的机器级锁锁的是**真 GPU**，离线套件不该去抢它
    with TestClient(create_app(runtime, gpu_lock_path=tmp_path / "gpu.lock")) as client:
        yield client, reports
    runtime.close()


# ── 矩阵 ──────────────────────────────────────────────────────────
def test_matrix_returns_one_cell_per_model_and_task(env):
    client, _ = env
    payload = client.get("/api/matrix").json()
    assert sorted(payload["models"]) == ["mock/a", "mock/b"]
    assert payload["tasks"] == ["intent_classification"]
    assert len(payload["cells"]) == 2
    a = next(cell for cell in payload["cells"] if cell["model_id"] == "mock/a")
    assert a["metric"] == "macro_f1"
    assert a["value"] == pytest.approx(1.0)
    assert a["ci"]["low"] is not None, "CI 必须是结构化的，不能是 repr 字符串"
    assert "CI(low=" not in str(payload)


def test_matrix_carries_dataset_provenance(env):
    client, _ = env
    payload = client.get("/api/matrix").json()
    assert payload["provenance"] == ["intent_zh-v1@seed=1"], \
        "矩阵必须能回答「这几格是不是同一份数据」"
    assert payload["warnings"] == []


def test_matrix_can_be_filtered_by_task(env):
    client, _ = env
    assert len(client.get("/api/matrix", params={"task": "intent_classification"}).json()["cells"]) == 2
    assert client.get("/api/matrix", params={"task": "nope"}).json()["cells"] == []


def test_matrix_exposes_the_headline_denominator(env):
    """每格要带"主分数看到了多少样本"，UI 才能标出覆盖率。"""
    client, _ = env
    payload = client.get("/api/matrix").json()
    cell = payload["cells"][0]
    assert cell["n_total"] == 4
    assert cell["n_judged"] == 4
    assert cell["coverage"] == pytest.approx(1.0)
    assert cell["run_id"], "格子的存在是为了跳到那次运行"


# ── 配对对比 ──────────────────────────────────────────────────────
def test_compare_returns_paired_counts_and_ci(env):
    client, (good, bad) = env
    payload = client.get("/api/compare", params={"base": good.run_id, "target": bad.run_id}).json()
    assert payload["n_paired"] == 4
    assert payload["improved"] == 0 and payload["regressed"] == 2
    assert payload["mean_delta"] == pytest.approx(-0.5)
    assert payload["delta_ci"]["n"] == 4
    assert payload["delta_ci"]["low"] is not None
    assert payload["base"]["model_id"] == "mock/a"
    assert payload["target"]["model_id"] == "mock/b"
    assert payload["low_confidence"] is True, "4 个配对样本必须标低置信"


def test_compare_cases_carry_both_trace_ids(env):
    client, (good, bad) = env
    payload = client.get("/api/compare", params={"base": good.run_id, "target": bad.run_id}).json()
    worse = [case for case in payload["cases"] if case["delta"] < 0]
    assert len(worse) == 2
    for case in worse:
        assert case["trace_base"] and case["trace_target"]
        # 两条 trace 都必须真的可查——并排看是这一步的全部意义
        for trace_id in (case["trace_base"], case["trace_target"]):
            assert client.get(f"/api/traces/{trace_id}").status_code == 200
        assert case["instruction"].startswith("第"), "列表要能看出是哪道题"


def test_compare_can_skip_the_case_list_for_a_summary_view(env):
    client, (good, bad) = env
    payload = client.get("/api/compare", params={
        "base": good.run_id, "target": bad.run_id, "with_cases": "false",
    }).json()
    assert payload["cases"] == [] and payload["n_paired"] == 4


def test_direction_is_target_minus_base(env):
    client, (good, bad) = env
    forward = client.get("/api/compare", params={"base": good.run_id, "target": bad.run_id,
                                                 "with_cases": "false"}).json()
    backward = client.get("/api/compare", params={"base": bad.run_id, "target": good.run_id,
                                                  "with_cases": "false"}).json()
    assert forward["mean_delta"] == pytest.approx(-0.5)
    assert backward["mean_delta"] == pytest.approx(0.5)
    assert (backward["improved"], backward["regressed"]) == (2, 0)


def test_incomparable_requests_are_404_with_a_readable_reason(env):
    client, (good, bad) = env
    missing = client.get("/api/compare", params={"base": good.run_id, "target": "nope"})
    assert missing.status_code == 404
    assert "找不到 run" in str(missing.json())
    same = client.get("/api/compare", params={"base": good.run_id, "target": good.run_id,
                                             "with_cases": "false"})
    assert same.status_code == 200 and same.json()["regressed"] == 0, "自己比自己当然全不变"
    bad_eps = client.get("/api/compare", params={"base": good.run_id, "target": bad.run_id,
                                                 "eps": 5})
    assert bad_eps.status_code == 422, "eps 超出 [0,1] 要被拦下，而不是算出一个「全不变」"
