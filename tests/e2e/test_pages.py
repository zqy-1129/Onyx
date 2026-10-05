"""e2e（S34）：六个页面各取一次数，断言的是**同源**，不是"接口返回 200"。

为什么盯同源：这个产品的每个数字都会被拿去和另一个数字对照（Fleet 的窗口 vs 列表的总数、
Ledger 的汇总 vs 逐条求和、grade vs trace、开销面板 vs 注册表）。
两处各算一套时不会报错，只会给出两个都"看起来合理"的数——那是本项目最贵的一类 bug。
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


# ── 1. Fleet ──────────────────────────────────────────────────────
def test_fleet_numbers_are_the_same_numbers_the_other_pages_read(stack):
    client = stack.client
    fleet = client.get("/api/fleet").json()

    # 同一个窗口里，总览说的请求数必须就是 Traces 页能数出来的条数
    listed = client.get("/api/traces", params={"limit": 500}).json()
    assert fleet["window"]["traces"] == listed["total"], (
        "Fleet 说 19 条、Traces 页能列出 20 条，就一定有一边在按别的条件数")

    # token 也一样：Ledger 与 Fleet 是同一份 usage 的两个视图
    ledger = client.get("/api/usage/summary").json()
    assert fleet["window"]["in_tokens"] == ledger["in_tokens"]
    assert fleet["window"]["out_tokens"] == ledger["out_tokens"]

    # error 级异常那一行：计数与"最近一条能落到真实 trace"必须同时成立
    errors = fleet["error_anomalies"]
    assert errors["n"] >= 1 and errors["by_code"].get("NO_ENGINE_COUNT")
    assert client.get(f"/api/traces/{errors['latest_trace_id']}").status_code == 200

    # 顶部那一行的后半句：通知系统自己好不好
    assert fleet["alerts"]["enabled"] is True and fleet["alerts"]["channels"] == ["file"]
    assert fleet["alerts"]["thread_alive"] is True and fleet["alerts"]["last_error"] == ""


def test_fleet_anomaly_severities_come_from_the_backend(stack):
    """级别由后端给。前端写死 warn 会把 error 级显示成"提醒"。"""
    anomalies = stack.client.get("/api/fleet").json()["anomalies"]
    assert anomalies["NO_ENGINE_COUNT"]["severities"] == ["error"]
    warn_codes = [code for code, stat in anomalies.items() if "warn" in stat["severities"]]
    assert warn_codes, "种子里应当也有一条 warn，否则这条断言测不出级别在不在"


# ── 2. Traces ─────────────────────────────────────────────────────
def test_trace_list_and_detail_agree_on_every_row(stack):
    client = stack.client
    rows = client.get("/api/traces", params={"limit": 50}).json()["items"]
    assert rows, "列表空了的话这条测试什么都证明不了"
    for row in rows:
        head = client.get(f"/api/traces/{row['id']}").json()["trace"]
        assert head["id"] == row["id"]
        assert head["status"] == row["status"], "列表与详情状态不一致时，人只会信先看到的那个"
        assert head["in_tokens"] == row["in_tokens"] and head["out_tokens"] == row["out_tokens"], \
            "同一个数在两个视图里不一样，就等于有两个事实源"


# ── 3. Token Ledger ───────────────────────────────────────────────
def test_ledger_totals_equal_the_sum_of_their_own_timeseries(stack):
    ledger = stack.client.get("/api/usage/summary").json()
    buckets = ledger["timeseries"]
    assert buckets, "时间序列空了，页面就只剩一个无法核对的总数"
    assert sum(b["in_tokens"] + b["out_tokens"] for b in buckets) \
        == ledger["in_tokens"] + ledger["out_tokens"], (
        "分桶求和与总数对不上，说明两边查的不是同一批 trace")
    # by_source 数的是"每条 trace 采信自谁"，量纲是条数：它能加回到 traces，
    # 加不回 token 总数——把两种量纲混在一个面板上正是"看起来合理但错了"的那种
    assert sum(ledger["by_source"].values()) == ledger["traces"]
    assert sum(ledger["by_confidence"].values()) == ledger["traces"]


# ── 4. Tool Bench ─────────────────────────────────────────────────
def test_tool_cost_adds_up_and_reports_the_template_share_honestly(stack):
    client = stack.client
    cost = client.get("/api/tools/cost").json()
    assert [t["name"] for t in cost["tools"]] == ["get_weather", "send_mail"]
    assert cost["json_tokens"] == sum(t["tokens"] for t in cost["tools"])
    assert cost["effective_tokens"] == cost["json_tokens"] + cost["template_overhead_tokens"]
    # P17：没传模板开销时报「—」而不是 0%（0% 会被读成"模板不花钱"，实测恰恰相反）
    assert cost["template_share"] is None
    with_overhead = client.get("/api/tools/cost", params={"overhead": cost["json_tokens"]}).json()
    assert with_overhead["template_share"] == pytest.approx(0.5), \
        "传了模板开销还不占比，P17 的结论就白测了（大头是模板注入的说明文本）"
    assert with_overhead["effective_tokens"] == 2 * cost["json_tokens"]


def test_contract_matrix_never_calls_not_applicable_a_pass(stack):
    matrix = stack.client.get("/api/tools/matrix").json()
    cells = [(name, assertion, cell)
             for name, row in matrix["executors"].items()
             for assertion, cell in row.items()]
    assert cells
    for name, assertion, cell in cells:
        if cell["applicable"] is False:
            assert cell["passed"] is False, (
                f"{name}/{assertion}：不适用却记成通过，等于把缺能力写成质量保证")
    # 没实现的列进 pending、装不上的进 unavailable，两者都不许出现在矩阵的列里
    for pending in matrix["pending"]:
        assert pending not in matrix["executors"]
    for missing in matrix["unavailable"]:
        assert missing not in matrix["executors"]


# ── 5. 评测运行与 grade ───────────────────────────────────────────
def test_every_grade_points_at_a_real_trace(stack):
    """这是"每个分数都能点进一次真实请求"的机器化版本——改版时最容易悄悄断的一条。"""
    client = stack.client
    for run_id in (stack.run_a, stack.run_b):
        detail = client.get(f"/api/runs/{run_id}").json()["run"]
        grades = client.get(f"/api/runs/{run_id}/grades").json()
        assert len(grades) == detail["n_done"] == detail["n_cases"]
        for grade in grades:
            assert grade["trace_id"], f"{grade['case_id']} 没有 trace_id，这个分数无法复核"
            assert client.get(f"/api/traces/{grade['trace_id']}").status_code == 200, \
                "grade 指向一条不存在的 trace：分数与证据断线了"


def test_undefined_metric_stays_null_instead_of_becoming_zero(stack):
    """`转账` 这一类只有 3 个样本且全错 ⇒ macro 里它是"未定义"，不是 0 分。"""
    run = stack.client.get(f"/api/runs/{stack.run_a}").json()["run"]
    per_class = run["aggregate"]["per_class_f1"]
    assert None in per_class.values(), "未定义被填成 0 之后，模型越差分数越高"
    assert run["aggregate"]["low_confidence"] is True and run["aggregate"]["n_total"] == 8


# ── 6. 矩阵与回归对比 ─────────────────────────────────────────────
def test_matrix_cells_carry_their_denominator(stack):
    matrix = stack.client.get("/api/matrix").json()
    assert set(matrix["models"]) == {
        next(run for run in stack.client.get("/api/runs?limit=50").json()
             if run["id"] == cell["run_id"])["model_id"]
        for cell in matrix["cells"]
    }, "矩阵里的模型必须在运行列表里"
    for cell in matrix["cells"]:
        assert cell["n"], "格子没有分母就等于在说「这格很可信」"
        ci = cell["ci"] or {}
        assert ci.get("n") == cell["n"], "CI 的重采样单位必须与格子的分母同源"
        assert cell["run_id"], "每格取的是该组合最新一次 done，得能点回去"


def test_regression_list_entries_have_two_drillable_traces(stack):
    client = stack.client
    comparison = client.get("/api/compare", params={
        "base": stack.run_a, "target": stack.run_b, "iterations": 200}).json()
    worse = [case for case in comparison["cases"] if case["delta"] < 0]
    better = [case for case in comparison["cases"] if case["delta"] > 0]
    assert worse and better, "两个模型都答固定标签，种子必须造出双向翻转才有可比性"
    assert comparison["regressed"] == len(worse) and comparison["improved"] == len(better)
    for case in worse:
        for key in ("trace_base", "trace_target"):
            assert client.get(f"/api/traces/{case[key]}").status_code == 200, \
                f"{case['case_id']} 的 {key} 下钻不到：劣化清单只剩一句话"
