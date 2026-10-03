"""S13 验收：eval CLI。

这些测试同时是"自测命令"的可执行版本——IMPLEMENTATION.md 里写的
`onyx eval import/run/show/ls` 必须真的能跑出预期输出。

重点盯两条容易被忽略的口径：
1. 未知显示「—」，**绝不显示 0**（UI_DESIGN R2）。曾经出现过
   macro_f1=None 时列表页悄悄改显示 pass_hat_k 的 0.000，看起来像"模型得了 0 分"，
   而真相是"一条可判定样本都没有"——两者的修法完全相反。
2. 分数必须带指标名与口径，否则一个孤零零的数字无法解释。
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from onyx.cli import _call_brief, _fmt, _headline_score, app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """数据目录与机器级 GPU 锁都落在 tmp 里。

    锁默认是机器级的（这在真实环境里正是它该有的行为），但离线套件如果去抢那把真锁，
    "有人在跑评测时 pytest 挂住"就成了失败原因——而失败原因必须是代码，不是环境。
    """
    monkeypatch.setenv("ONYX_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "machine"))
    return tmp_path


def _run(*argv: str):
    return runner.invoke(app, list(argv))


MOCK_RUN = ["eval", "run", "--task", "intent_classification", "--model", "mock/echo",
            "--provider", "mock", "--limit", "8", "--seed", "42", "--quiet"]


def _import_builtin():
    result = _run("eval", "import", "--builtin", "intent_zh")
    assert result.exit_code == 0, result.output
    return result


# ── 显示口径 ──────────────────────────────────────────────────────
def test_fmt_shows_unknown_as_a_dash_never_as_zero():
    assert _fmt(None) == "—"
    assert _fmt(0) == "0.000", "0 是一个真实的值，必须显示成 0"
    assert _fmt(0.9913) == "0.991"
    assert _fmt(True) == "是" and _fmt(False) == "否"


def test_headline_score_names_the_metric_it_shows():
    text = _headline_score({"macro_f1": 0.812, "macro_f1_ci": {"low": 0.77, "high": 0.85},
                            "low_confidence": False})
    assert text == "macro_f1 0.812 [0.770–0.850]"


def test_headline_score_shows_a_dash_when_the_primary_metric_is_unknown():
    """这条是回归测试：曾经悄悄换成 pass_hat_k 的 0.000，把"未知"显示成"0 分"。"""
    aggregate = {"macro_f1": None, "accuracy": None, "pass_hat_k": 0.0, "n_total": 12}
    text = _headline_score(aggregate)
    assert text.startswith("macro_f1 —"), text
    assert "0.000" not in text


def test_headline_score_flags_low_sample_size():
    text = _headline_score({"macro_f1": 0.9, "low_confidence": True})
    assert "⚠低样本" in text


def test_headline_score_carries_the_ci_for_any_metric_that_has_one():
    """区间按「指标名 + _ci」找，不是 macro_f1 专属。

    写死 macro_f1 的话，工具任务换用 must_call_acc 之后列表页只剩一个孤零零的数，
    而"0.639 是 72 个 case 的 0.639"这件事就再也看不见了。
    """
    text = _headline_score({"must_call_acc": 0.6389,
                            "must_call_acc_ci": {"low": 0.5278, "high": 0.7361, "n": 72},
                            "low_confidence": True})
    assert text.startswith("must_call_acc 0.639 [0.528–0.736]"), text
    assert "⚠低样本" in text


def test_headline_score_without_a_ci_stays_a_plain_number():
    assert _headline_score({"pass_hat_k": 0.25}) == "pass_hat_k 0.250"


def test_headline_score_handles_skipped_and_empty():
    assert _headline_score({}) == "—"
    assert _headline_score({"skip": {"reason": "x"}}) == "skipped"


def test_headline_score_falls_back_to_a_named_alternative():
    """任务没有 macro_f1 时才轮到下一个指标，而且同样要带名字。"""
    assert _headline_score({"accuracy": 0.5}).startswith("accuracy 0.500")
    assert _headline_score({"pass_hat_k": 0.25}).startswith("pass_hat_k 0.250")


def test_call_brief_compacts_expected_calls_without_losing_the_distinction():
    """`eval show` 的期望/预测两列要能一眼看完。

    空列表必须显示成"（不调用）"而不是 `—`：前者是"期望模型不调工具"这个**结论**，
    后者是"这个 grade 没有期望值"，两者混在一起就看不出 no_call_needed 的判定。
    """
    assert _call_brief([{"name": "get_weather", "arguments": {"city": "北京"}}]) \
        == "get_weather(city=北京)"
    assert _call_brief([{"name": "a", "arguments": {}}, {"name": "b", "arguments": {}}]) == "a + b"
    assert _call_brief([]) == "（不调用）"
    assert _call_brief(None) == "—"
    assert _call_brief("转账") == "转账", "分类任务的期望是字符串，原样显示"


def test_show_prints_a_full_trace_id_for_drilling_in(tmp_path):
    """表里的 trace 只有 12 个字符（列宽限制），跳进去需要完整 id。

    只有截断列的话，"每个分数能跳到 trace"就成了一句做不到的承诺。
    """
    _import_builtin()
    run = _run(*MOCK_RUN, "--json")
    run_id = json.loads(run.output)["run_id"]
    shown = _run("eval", "show", run_id, "--limit", "2")
    assert "下钻: onyx traces show 0" in shown.output, shown.output


# ── import / ls ───────────────────────────────────────────────────
def test_import_builtin_reports_provenance_and_subsets():
    result = _import_builtin()
    assert "intent_zh-v1" in result.output
    assert "236 条" in result.output
    assert "builtin:intent_zh" in result.output, "来源必须报出来"
    assert "seed=20261003" in result.output, "revision 必须报出来，否则换版本后分数不可比"
    assert "hard" in result.output, "子集要能单独跑，所以得让人知道有哪些"


def test_import_requires_a_source():
    result = _run("eval", "import")
    assert result.exit_code == 2
    assert "--builtin" in result.output


def test_import_from_a_jsonl_file(tmp_path):
    path = tmp_path / "mine.jsonl"
    path.write_text(
        '{"input": {"instruction": "查余额"}, "expect": {"label": "查余额"}}\n'
        '{"input": {"instruction": "转钱给张伟"}, "expect": {"label": "转账"}}\n',
        encoding="utf-8",
    )
    result = _run("eval", "import", str(path), "--id", "mine-v1", "--upstream", "手工整理",
                  "--license", "internal")
    assert result.exit_code == 0, result.output
    assert "mine-v1" in result.output and "2 条" in result.output
    assert "手工整理" in result.output


def test_import_reports_the_offending_line_of_a_broken_file(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": {"instruction": "a"}}\n{坏了}\n', encoding="utf-8")
    result = _run("eval", "import", str(path))
    assert result.exit_code == 1
    assert "第 2 行" in result.output


def _bfcl_pair(tmp_path):
    """一份最小 BFCL 风格文件：一条单轮 + 一条多轮（应当被跳过并计数）。"""
    questions = tmp_path / "questions.jsonl"
    answers = tmp_path / "answers.jsonl"
    weather_tool = {
        "name": "get_weather", "description": "查询天气",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "城市"}},
                       "required": ["city"]},
    }
    questions.write_text("\n".join([
        json.dumps({"id": "q1", "question": [[{"role": "user", "content": "北京天气怎么样"}]],
                    "function": [weather_tool]}),
        json.dumps({"id": "q2", "question": [[{"role": "user", "content": "第一轮"}],
                                             [{"role": "user", "content": "第二轮"}],
                                             ]}),
    ]), encoding="utf-8")
    answers.write_text(json.dumps(
        {"id": "q1", "ground_truth": [[{"name": "get_weather",
                                        "arguments": {"city": "北京"}}]]}), encoding="utf-8")
    return questions, answers


def test_import_bfcl_needs_no_converter_step(tmp_path):
    """BFCL 必须能直接导入：要人先手写转换脚本，实际结果就是没人导，
    而"没有数据集"在报告里长得和"模型很差"一样。
    """
    questions, answers = _bfcl_pair(tmp_path)
    result = _run("eval", "import", str(questions), "--source", "bfcl",
                  "--answers", str(answers), "--subset", "ast", "--id", "bfcl-ast-v1")
    assert result.exit_code == 0, result.output
    assert "bfcl-ast-v1 · 1 条" in result.output
    assert "bfcl:ast" in result.output, "来源要精确到子集"
    assert "sha256:" in result.output, "没给 --revision 时用文件 hash 兜底，绝不能留空"
    assert "跳过 1 条多轮样本" in result.output, "丢样本必须当场看见"

    listed = _run("eval", "ls")
    assert "bfcl-ast-v1" in listed.output


def test_import_bfcl_is_also_inferred_from_answers(tmp_path):
    """只给 `--answers` 就足够判定源；让人多打一个 --source 只会变成漏打。"""
    questions, answers = _bfcl_pair(tmp_path)
    result = _run("eval", "import", str(questions), "--answers", str(answers))
    assert result.exit_code == 0, result.output
    assert "bfcl-v1" in result.output


def test_import_rejects_an_unknown_source(tmp_path):
    questions, _ = _bfcl_pair(tmp_path)
    result = _run("eval", "import", str(questions), "--source", "lmsys")
    assert result.exit_code == 1
    assert "不支持的数据源" in result.output and "bfcl" in result.output, "要列出可选值"


def test_import_rejects_builtin_combined_with_a_source(tmp_path):
    """参数组合错了必须报错，不能静默挑一个。

    静默忽略的话，使用者以为自己导入的是 BFCL，实际导入的是内置集，
    而两份数据的分数看起来都"正常"。
    """
    result = _run("eval", "import", "--builtin", "intent_zh", "--source", "bfcl")
    assert result.exit_code == 2
    assert "互斥" in result.output


def test_ls_lists_datasets_and_tasks():
    _import_builtin()
    result = _run("eval", "ls")
    assert result.exit_code == 0, result.output
    assert "intent_zh-v1" in result.output
    assert "intent_classification" in result.output


# ── run ───────────────────────────────────────────────────────────
def test_run_offline_completes_and_reports_a_run_id():
    _import_builtin()
    result = _run(*MOCK_RUN)
    assert result.exit_code == 0, result.output
    assert "intent_classification · mock/echo · n=8" in result.output
    assert "run_id" in result.output and "onyx eval show" in result.output
    # mock provider 的默认回复不是标签，所以全是越界；关键是**报成越界而不是 0 分**
    assert "out_of_label" in result.output


def test_run_separates_the_content_and_format_dimensions():
    """DESIGN §9.4：两个维度必须分开打印，混成一个正确率会把格式问题读成能力问题。"""
    _import_builtin()
    result = _run(*MOCK_RUN)
    assert "[内容]" in result.output and "[格式]" in result.output
    assert "format_valid" in result.output and "out_of_label" in result.output
    assert "gen-based" in result.output, "打分口径必须交代，否则会被拿去和 leaderboard 比"


def test_run_json_output_is_machine_readable():
    _import_builtin()
    result = _run(*MOCK_RUN, "--json")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "done"
    assert payload["n_cases"] == 8 and payload["n_done"] == 8
    assert payload["aggregate"]["verdicts"]["out_of_label"] == 8
    assert payload["aggregate"]["scoring"] == "gen-based"
    assert payload["cost"]["requests"] == 8
    assert payload["run_id"]


def test_run_reports_cost_including_unmeasured_requests():
    _import_builtin()
    payload = json.loads(_run(*MOCK_RUN, "--json").output)
    cost = payload["cost"]
    assert cost["requests"] == 8
    assert cost["in_tokens"] > 0 and cost["out_tokens"] > 0
    assert cost["in_tokens_unknown"] == 0


def test_run_rejects_an_unknown_task():
    result = _run("eval", "run", "--task", "nope", "--model", "m", "--provider", "mock")
    assert result.exit_code == 2
    assert "未知任务" in result.output


def test_run_rejects_an_unknown_dataset():
    _import_builtin()
    result = _run(*MOCK_RUN, "--dataset", "nope")
    assert result.exit_code == 2
    assert "未知数据集" in result.output


def test_hard_subset_can_be_run_on_its_own():
    """难例单独跑才有意义：混在 200 条简单样本里会被稀释成一个好看的平均分。"""
    _import_builtin()
    payload = json.loads(_run(*MOCK_RUN, "--split", "hard", "--json").output)
    assert payload["status"] == "done"
    assert payload["n_cases"] > 0


def test_k_sampling_multiplies_the_request_count():
    _import_builtin()
    payload = json.loads(_run(*MOCK_RUN, "--k", "2", "--json").output)
    assert payload["cost"]["requests"] == 16, "8 条 × 2 次采样"
    assert payload["aggregate"]["k"] == 2
    assert "pass_hat_k" in payload["aggregate"]


# ── show ──────────────────────────────────────────────────────────
def test_show_lists_grades_with_trace_ids():
    """DoD：任一分数都能跳到 trace。表格里必须直接看得见 trace_id。"""
    _import_builtin()
    run_id = json.loads(_run(*MOCK_RUN, "--json").output)["run_id"]
    result = _run("eval", "show", run_id, "--limit", "3")
    assert result.exit_code == 0, result.output
    assert "out_of_label" in result.output
    assert "trace" in result.output
    # trace 列必须是真实 id 而不是「—」，否则"分数能点进 trace"就只是文档里的说法
    # 只查表格 body 行：错误样例那段也以 case id 开头，但它没有 trace 列
    body_rows = [line for line in result.output.splitlines()
                 if "izh-" in line and line.lstrip().startswith("│")]
    assert body_rows, "表格里没有任何 grade 行"
    for row in body_rows:
        assert "01M" in row, f"trace 列是空的: {row}"


def test_show_json_includes_params_snapshot_and_costs():
    _import_builtin()
    run_id = json.loads(_run(*MOCK_RUN, "--json").output)["run_id"]
    payload = json.loads(_run("eval", "show", run_id, "--json").output)
    run = payload["run"]
    assert run["id"] == run_id
    # 参数快照必须落库：否则换了 temperature 之后的分数差异无法解释
    assert run["params_snapshot"]["temperature"] == 0.0
    assert run["params_snapshot"]["max_tokens"] == 32
    assert run["app_version"]
    assert run["aggregate"]["scoring"] == "gen-based"
    assert len(payload["grades"]) == 8
    assert all(g["trace_id"] for g in payload["grades"])


def test_show_filters_by_verdict():
    _import_builtin()
    run_id = json.loads(_run(*MOCK_RUN, "--json").output)["run_id"]
    payload = json.loads(_run("eval", "show", run_id, "--verdict", "out_of_label", "--json").output)
    assert len(payload["grades"]) == 8
    empty = json.loads(_run("eval", "show", run_id, "--verdict", "correct", "--json").output)
    assert empty["grades"] == []


def test_show_rejects_an_unknown_run():
    result = _run("eval", "show", "does-not-exist")
    assert result.exit_code == 2
    assert "找不到 run" in result.output


# ── tool_selection 也走同一套 CLI ─────────────────────────────────
def test_tool_selection_task_runs_offline_end_to_end():
    _run("eval", "import", "--builtin", "tool_calls_zh")
    result = _run("eval", "run", "--task", "tool_selection", "--model", "mock/echo",
                  "--provider", "mock", "--limit", "10", "--seed", "7", "--json", "--quiet")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["aggregate"]["scoring"] == "gen-based"
    # mock provider 不调工具 ⇒ 该调的样本全判 no_call，这个分布本身就是结论
    assert payload["aggregate"]["verdicts"].get("no_call", 0) > 0
    assert payload["aggregate"]["n_must_call"] > 0


def test_report_prints_only_the_metrics_this_task_produced():
    """回归测试：报告模板曾经硬编码 intent 的指标，
    于是 tool_selection 整行都是「—」，看着像评测坏了，其实是模板对不上口径。"""
    _run("eval", "import", "--builtin", "tool_calls_zh")
    tool = _run("eval", "run", "--task", "tool_selection", "--model", "mock/echo",
                "--provider", "mock", "--limit", "10", "--quiet")
    assert tool.exit_code == 0, tool.output
    assert "must_call_acc" in tool.output, "工具任务的头号指标必须出现"
    assert "false_call_rate" in tool.output
    assert "bal_acc" not in tool.output, "intent 专属指标不该出现在工具任务的报告里"

    _run("eval", "import", "--builtin", "intent_zh")
    intent = _run("eval", "run", "--task", "intent_classification", "--model", "mock/echo",
                  "--provider", "mock", "--limit", "10", "--quiet")
    assert "macro_f1" in intent.output
    assert "must_call_acc" not in intent.output, "反向同理：分类报告不该出现工具指标"


def test_report_prints_the_ci_for_whichever_metric_has_one():
    """CI 跟着指标走，不是 `macro_f1` 的专属装饰。

    只对 macro_f1 特判的话，`must_call_acc` 明明算出了 95% 区间，报告里却显示成
    一个孤零零的数——而 DESIGN §9.3 要求"聚合带 bootstrap 95% CI"，
    没有区间的单个分数无法判断是"测准了"还是"只测了 10 条"。
    """
    _run("eval", "import", "--builtin", "tool_calls_zh")
    result = _run("eval", "run", "--task", "tool_selection", "--model", "mock/echo",
                  "--provider", "mock", "--limit", "10", "--k", "2", "--quiet")
    assert result.exit_code == 0, result.output
    line = next(line for line in result.output.splitlines() if "must_call_acc" in line)
    assert "95% CI" in line, line
    assert "n=" in line, "区间必须带着它的重采样单位，否则看不出这是 10 个 case 的区间"


def test_eval_run_queues_on_the_gpu_lock_by_default(tmp_path):
    """锁路径可以指到临时目录，用它验证"被别人占着就不开跑"。"""
    from onyx.eval.gpu_lock import GpuLock

    _import_builtin()
    holder = GpuLock(tmp_path / "gpu.lock", owner="other-run", stale_after_s=60)
    holder.acquire()
    holder.heartbeat(1, 10)
    try:
        result = _run("eval", "run", "--task", "intent_classification", "--model", "mock/echo",
                      "--provider", "mock", "--limit", "2", "--quiet",
                      "--gpu-lock", str(tmp_path / "gpu.lock"), "--no-queue")
        assert result.exit_code == 3, result.output
        assert "other-run" in result.output
        assert "--no-queue" in result.output, "要告诉用户怎么改变排队行为"
    finally:
        holder.release()


def test_eval_run_uses_the_machine_lock_when_not_overridden():
    """不给 `--gpu-lock` 时，eval run 必须去拿**机器级**那把锁。

    默认路径一旦退回数据目录，两个实例就各锁各的文件，然后照样同时往显存里塞模型；
    而这个缺陷不会以报错呈现，只会让两边的数字一起失真。
    """
    from onyx.eval.gpu_lock import GpuLock, default_lock_path

    _import_builtin()
    holder = GpuLock(default_lock_path(), owner="other-instance", stale_after_s=60)
    holder.acquire()
    try:
        result = _run("eval", "run", "--task", "intent_classification", "--model", "mock/echo",
                      "--provider", "mock", "--limit", "2", "--quiet", "--no-queue")
        assert result.exit_code == 3, result.output
        assert "other-instance" in result.output, "必须说明谁在占，否则用户只会以为命令坏了"
    finally:
        holder.release()


def test_eval_show_exposes_grades_for_the_tool_task():
    _run("eval", "import", "--builtin", "tool_calls_zh")
    run = _run("eval", "run", "--task", "tool_selection", "--model", "mock/echo",
               "--provider", "mock", "--limit", "6", "--json", "--quiet")
    run_id = json.loads(run.output)["run_id"]
    payload = json.loads(_run("eval", "show", run_id, "--json").output)
    assert payload["run"]["task_id"] == "tool_selection"
    assert all(grade["trace_id"] for grade in payload["grades"])
