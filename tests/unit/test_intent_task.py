"""S13 验收：意图识别任务。

计划要求用 mock provider 的四类响应断言"四类得分与 verdict 各不相同"：
正确 / 错标签 / **越界标签（不在标签集）** / 附加解释文字。

第 1 类与第 4 类内容上都对，所以 verdict 相同——但 `invalid_format` 必须不同。
这正是 DESIGN §9.4 的核心：**格式合法率与内容正确率是两个正交维度**，
混成一个数就会把"模型爱加解释"误读成"模型分类能力差"，而前者改提示词就能修。
"""

from __future__ import annotations

import pytest

from onyx.core.content import FileBlobStore
from onyx.core.types import Cap, FinishReason, Generation, Status
from onyx.eval.datasets.loader import Dataset, DatasetError, load_builtin, load_jsonl
from onyx.eval.metrics import LOW_CONFIDENCE_N
from onyx.eval.task import Verdict, check_capabilities
from onyx.eval.tasks import build_task, load_dataset, task_ids
from onyx.eval.tasks.intent_classification import IntentClassification
from onyx.llm.gateway import Gateway
from onyx.llm.providers.mock import MockProvider, MockScript
from onyx.obs.engine import ObserverEngine
from onyx.store.db import Database
from onyx.store.repos import EvalRepo, TraceRepo
from onyx.store.sinks import SqliteRecordSink

LABELS = ("转账", "查余额", "投诉", "其他")


def _case(instruction: str = "帮我把300转给小李", label: str = "转账", **kw):
    from onyx.eval.task import Case

    return Case(id="c1", input={"instruction": instruction}, expect={"label": label}, **kw)


def _task(cases=None, **kw) -> IntentClassification:
    dataset = Dataset(
        id="tiny", cases=tuple(cases if cases is not None else [
            {"id": "c1", "input": {"instruction": "帮我把300转给小李"},
             "expect": {"label": "转账"}, "ord": 0, "tags": []},
        ]),
    )
    return IntentClassification(dataset, model="mock/classifier", **kw)


def _grade(text: str, *, expected: str = "转账", thinking: str = "",
           status: Status = Status.OK, error: str = ""):
    task = _task()
    sample = Generation(
        text=text, thinking=thinking, status=status, error=error,
        finish_reason=FinishReason.STOP, model="mock/classifier",
    )
    return task.grade(_case(label=expected), sample)


# ── 四类响应，四种可区分的结果 ────────────────────────────────────
def test_four_response_shapes_are_distinguishable():
    correct = _grade("转账")
    wrong = _grade("查余额")
    out_of_set = _grade("退款")
    verbose = _grade("这个意图是转账。")

    assert (correct.verdict, correct.invalid_format) == (Verdict.CORRECT, False)
    assert (wrong.verdict, wrong.invalid_format) == (Verdict.WRONG, False)
    assert (out_of_set.verdict, out_of_set.invalid_format) == (Verdict.OUT_OF_LABEL, True)
    assert (verbose.verdict, verbose.invalid_format) == (Verdict.CORRECT, True)

    # 四种 (verdict, invalid_format) 组合互不相同 —— 这就是"可区分"的含义
    shapes = {(g.verdict, g.invalid_format) for g in (correct, wrong, out_of_set, verbose)}
    assert len(shapes) == 4

    assert correct.score == 1.0 and wrong.score == 0.0
    assert correct.passed is True and wrong.passed is False
    assert verbose.score == 1.0, "内容对就算对，格式问题单独记"


def test_out_of_label_is_hallucination_not_a_wrong_choice():
    """标签集之外的输出必须单独成一类：修法在标签集与提示词，不在模型能力。"""
    result = _grade("退款申请")
    assert result.verdict is Verdict.OUT_OF_LABEL
    assert result.out_of_set is True
    assert "不在标签集" in result.error
    assert result.metrics["expected"] == "转账"
    assert "predicted" not in result.metrics, "越界标签不是预测，不该进混淆矩阵"


def test_ambiguous_output_is_not_guessable():
    """同时出现两个候选标签 ⇒ 不可判定。挑一个会让分数凭空变高。"""
    result = _grade("不是转账，应该是查余额")
    assert result.verdict is Verdict.INVALID_FORMAT
    assert result.invalid_format is True
    assert result.score == 0.0
    assert "不可判定" in result.error
    assert set(result.metrics["found"]) == {"转账", "查余额"}


def test_empty_output_distinguishes_thinking_from_silence():
    """空正文有两种成因，修法完全不同（P12：预算被 thinking 吃光）。"""
    silent = _grade("")
    assert silent.verdict is Verdict.INVALID_FORMAT
    assert silent.error == "正文为空"

    eaten = _grade("", thinking="让我分析一下这句话的意图……")
    assert eaten.verdict is Verdict.INVALID_FORMAT
    assert "thinking" in eaten.error and "P12" in eaten.error
    assert eaten.metrics["thinking_chars"] > 0


def test_refusal_is_its_own_verdict():
    """过度拒答是真实缺陷（DESIGN §9.2），不该混进"格式非法"。"""
    result = _grade("抱歉，我无法回答这个问题。")
    assert result.verdict is Verdict.REFUSED
    assert result.attributable is True


def test_engine_failure_is_not_attributable_to_the_model():
    """引擎挂了是环境问题。算进模型得分就等于让模型替基础设施背锅。"""
    result = _grade("", status=Status.ERROR, error="ProviderUnreachable: 连不上")
    assert result.verdict is Verdict.ERROR
    assert result.attributable is False
    assert result.passed is None, "未判定，不是判错"


def test_case_without_an_expected_label_is_a_data_error():
    result = _task().grade(_case(label=""), Generation(text="转账", status=Status.OK))
    assert result.verdict is Verdict.ERROR
    assert "缺少 expect.label" in result.error


def test_wrapped_and_padded_labels_still_count_as_clean_format():
    for text in ("「转账」", "  转账 \n", "转账。", '"转账"'):
        result = _grade(text)
        assert result.verdict is Verdict.CORRECT, text
        assert result.invalid_format is False, f"{text!r} 只应算排版差异，不该扣格式分"


def test_ascii_labels_are_case_insensitive():
    task = _task(labels=("TRANSFER", "BALANCE"), fallback_label="BALANCE")
    from onyx.eval.task import Case

    case = Case(id="c1", input={"instruction": "move money"}, expect={"label": "TRANSFER"})
    result = task.grade(case, Generation(text="transfer", status=Status.OK))
    assert result.verdict is Verdict.CORRECT


def test_fallback_label_must_be_in_the_label_set():
    """兜底标签不在标签集里，模型输出它就永远算越界，分数会莫名偏低。"""
    with pytest.raises(ValueError, match="必须在 labels"):
        _task(fallback_label="未知")


# ── 请求构造 ──────────────────────────────────────────────────────
def test_build_produces_a_two_message_request_with_thinking_off():
    task = _task()
    request = task.build(_case("帮我把300转给小李"))
    assert request.model == "mock/classifier"
    assert [str(m.role) for m in request.messages] == ["system", "user"]
    assert request.messages[1].content == "帮我把300转给小李"
    # thinking 必须显式关掉，否则小预算会被推理吃光，正文变空串被判成格式非法
    assert request.thinking is False
    assert request.params.temperature == 0.0
    assert request.params.max_tokens == 32
    assert request.tools == ()


def test_system_prompt_lists_the_labels_and_the_fallback():
    task = _task()
    prompt = task.build(_case()).messages[0].content
    for label in LABELS:
        assert label in prompt
    assert "只输出意图名称本身" in prompt
    assert "其他" in prompt


def test_build_does_not_send_anything():
    """`build` 只产请求，不发请求——这是"评测不建立第二条调用路径"的落点。"""
    task = _task()
    assert task.build(_case()).messages  # 纯数据，无 IO
    assert not hasattr(task, "generate")


# ── 聚合 ──────────────────────────────────────────────────────────
def _grades(specs):
    """specs: [(expected, output_text)] → grades"""
    task = _task()
    out = []
    for expected, text in specs:
        sample = Generation(text=text, status=Status.OK, finish_reason=FinishReason.STOP)
        grade = task.grade(_case(label=expected), sample)
        out.append(grade)
    return task, out


def test_aggregate_separates_content_from_format_dimensions():
    task, grades = _grades([
        ("转账", "转账"),        # 干净且对
        ("转账", "这个意图是转账。"),  # 对但格式脏
        ("查余额", "转账"),       # 干净但错
        ("投诉", "退款"),        # 越界
    ])
    report = task.aggregate(grades)

    assert report["n_total"] == 4
    # 内容维度只对"有预测标签"的 3 条算（越界那条没有 predicted）
    assert report["n_judged"] == 3
    assert report["format_valid_rate"] == pytest.approx(2 / 4), "只有前两条格式干净"
    assert report["invalid_format_rate"] == pytest.approx(2 / 4)
    assert report["out_of_label_rate"] == pytest.approx(1 / 4)
    assert report["accuracy"] == pytest.approx(2 / 3), "3 条可判定里对了 2 条"
    assert report["scoring"] == "gen-based"


def test_aggregate_reports_confusions_and_per_class_scores():
    task, grades = _grades([
        ("转账", "转账"), ("转账", "转账"), ("转账", "转账"),
        ("投诉", "投诉"), ("投诉", "其他"), ("投诉", "其他"),
        ("其他", "其他"), ("查余额", "查余额"),
    ])
    report = task.aggregate(grades)
    assert report["top_confusions"][0] == {"expected": "投诉", "actual": "其他", "count": 2}
    assert report["per_class_f1"]["转账"] == pytest.approx(1.0)
    assert report["per_class_f1"]["投诉"] < 1.0
    assert report["macro_f1"] is not None
    assert report["confusion"]["投诉"]["其他"] == 2


def test_aggregate_is_all_unknown_when_nothing_is_judgeable():
    """全部越界 ⇒ 一个预测标签都没有 ⇒ macro_f1 必须是 None，不是 0。"""
    task, grades = _grades([("转账", "退款"), ("投诉", "咨询")])
    report = task.aggregate(grades)
    assert report["n_judged"] == 0
    assert report["macro_f1"] is None
    assert report["accuracy"] is None
    assert report["out_of_label_rate"] == pytest.approx(1.0)


def test_aggregate_includes_ci_and_low_confidence_flag():
    task, grades = _grades([("转账", "转账"), ("查余额", "查余额"), ("投诉", "投诉")])
    report = task.aggregate(grades, seed=1)
    assert report["macro_f1_ci"].n == 3
    assert report["low_confidence"] is True, "n=3 远低于 100 的门槛"
    assert LOW_CONFIDENCE_N == 100


def test_aggregate_computes_pass_hat_k_from_repeated_samples():
    """k>1 时同一 case 有多条 grade，pass^k 要求**全对**。"""
    task = _task()
    from dataclasses import replace

    sample_ok = Generation(text="转账", status=Status.OK)
    sample_bad = Generation(text="查余额", status=Status.OK)
    grades = [
        replace(task.grade(_case(label="转账"), sample_ok), seq=0),
        replace(task.grade(_case(label="转账"), sample_bad), seq=1),   # 同 case 第 2 次错
        replace(task.grade(_case(label="转账"), sample_ok), seq=0, case_id="c2"),
        replace(task.grade(_case(label="转账"), sample_ok), seq=1, case_id="c2"),
    ]
    report = task.aggregate(grades)
    assert report["k"] == 2
    assert report["pass_hat_k"] == pytest.approx(0.5), "c1 有一次错 ⇒ 不算稳定通过"
    assert report["pass_at_k"] == pytest.approx(1.0), "两个 case 都至少对过一次"
    assert report["stability_gap"] == pytest.approx(0.5)


def test_verdict_counts_cover_every_grade():
    task, grades = _grades([("转账", "转账"), ("转账", "退款"), ("转账", "")])
    counts = task.aggregate(grades)["verdicts"]
    assert sum(counts.values()) == 3
    assert counts["correct"] == 1 and counts["out_of_label"] == 1 and counts["invalid_format"] == 1


# ── 数据集 ────────────────────────────────────────────────────────
def test_builtin_dataset_meets_the_size_and_balance_requirements():
    from onyx.eval.datasets.builtin.intent_zh import build_cases, stats

    report = stats(build_cases())
    assert report["n"] >= 200, "计划要求 ≥200 条"
    assert report["unique_instructions"] == report["n"], "不许有重复样本"
    assert report["hard"] >= 20, "难例太少的话 hard 子集没有统计意义"
    counts = list(report["labels"].values())
    assert min(counts) / max(counts) > 0.5, f"类别过于不均衡: {report['labels']}"
    assert set(report["labels"]) == set(LABELS)


def test_builtin_dataset_is_reproducible():
    """数据集不可复现，评测就不可复现。"""
    from onyx.eval.datasets.builtin.intent_zh import build_cases

    first = build_cases()
    second = build_cases()
    assert [c["id"] for c in first] == [c["id"] for c in second]
    assert [c["input"]["instruction"] for c in first] == [c["input"]["instruction"] for c in second]


def test_builtin_dataset_limit_takes_a_mixed_subset():
    """`--limit 20` 取前 20 条，所以顺序必须各类混合，否则子集全是一个类。"""
    dataset = load_builtin()
    subset = dataset.select(limit=20)
    labels = {case["expect"]["label"] for case in subset}
    assert len(labels) >= 3, f"前 20 条只覆盖了 {labels}，limit 出来的子集会有偏"


def test_hard_subset_is_selectable():
    dataset = load_builtin()
    hard = dataset.select(split="hard")
    assert len(hard) >= 20
    assert all("hard" in case["tags"] for case in hard)
    with pytest.raises(DatasetError, match="没有名为"):
        dataset.select(split="does_not_exist")


def test_case_ids_are_content_derived_and_stable():
    """id 由内容算出，所以重新导入不会产生新 id，历史 grade 仍能对上。"""
    dataset = load_builtin()
    ids = {case["id"] for case in dataset.cases}
    assert len(ids) == len(dataset.cases)
    assert all(i.startswith("izh-") for i in ids)


def test_load_jsonl_reports_the_offending_line(tmp_path):
    """坏行必须报出是第几行——"第 37 行坏了"才是可行动的信息。"""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"input": {"instruction": "a"}}\n{不是 JSON}\n', encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        load_jsonl(path)
    assert exc.value.line == 2


def test_load_jsonl_rejects_missing_input_and_duplicate_ids(tmp_path):
    no_input = tmp_path / "no_input.jsonl"
    no_input.write_text('{"expect": {"label": "转账"}}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="input"):
        load_jsonl(no_input)

    dup = tmp_path / "dup.jsonl"
    dup.write_text(
        '{"id": "x", "input": {"instruction": "a"}}\n{"id": "x", "input": {"instruction": "b"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="重复"):
        load_jsonl(dup)


def test_load_jsonl_derives_ids_and_revision_from_content(tmp_path):
    path = tmp_path / "good.jsonl"
    path.write_text('{"input": {"instruction": "查余额"}, "expect": {"label": "查余额"}}\n',
                    encoding="utf-8")
    dataset = load_jsonl(path)
    assert dataset.id == "good-v1"
    assert dataset.cases[0]["id"], "缺 id 时必须由内容补一个"
    assert dataset.revision.startswith("sha256:"), "revision 必须是内容 hash，否则改文件看不出来"
    # 同一份文件两次载入 revision 相同
    assert load_jsonl(path).revision == dataset.revision


def test_load_jsonl_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="没有任何样本"):
        load_jsonl(path)


def test_dataset_to_records_round_trips(tmp_path):
    dataset = load_builtin()
    record, cases = dataset.to_records()
    assert record.n_cases == len(dataset) == len(cases)
    assert record.splits["hard"] >= 20
    assert cases[0].dataset_id == dataset.id
    assert cases[0].expect and cases[0].input


# ── 任务注册与能力跳过 ────────────────────────────────────────────
def test_task_registry_builds_the_builtin_task():
    assert task_ids() == ("intent_classification",)
    task = build_task("intent_classification", model="m")
    assert isinstance(task, IntentClassification)
    assert task.model == "m"
    with pytest.raises(KeyError, match="未知任务"):
        build_task("nope", model="m")


def test_load_dataset_accepts_builtin_and_file_forms(tmp_path):
    assert load_dataset(None, task_id="intent_classification").id == "intent_zh-v1"
    assert load_dataset("intent_zh", task_id="intent_classification").id == "intent_zh-v1"
    path = tmp_path / "d.jsonl"
    path.write_text('{"input": {"instruction": "x"}, "expect": {"label": "其他"}}\n',
                    encoding="utf-8")
    assert len(load_dataset(f"file:{path}", task_id="intent_classification")) == 1
    with pytest.raises(KeyError, match="未知数据集"):
        load_dataset("nope", task_id="intent_classification")


def test_intent_task_requires_no_capabilities():
    """纯文本分类任何 provider 都能跑，所以永远不该被 skip。"""
    task = build_task("intent_classification", model="m")
    assert task.requires == frozenset()
    assert check_capabilities(task, frozenset()) is None


def test_capability_skip_carries_a_reason_and_is_not_silent():
    """禁止隐式降级：用提示词模拟工具调用得到的分数无法与原生支持比较，
    却看不出区别——那比没有分数更糟。"""

    class NeedsTools:
        id = "tool_selection"
        name = "工具选择"
        requires = frozenset({Cap.TOOLS, Cap.TOOL_CHOICE})
        metric_names = ("hit_at_1",)

    skip = check_capabilities(NeedsTools(), frozenset({Cap.CHAT}))
    assert skip is not None
    assert skip.missing == ("tool_choice", "tools")
    assert "不做隐式降级" in skip.reason

    grade = skip.as_grade()
    assert grade.verdict is Verdict.SKIPPED
    assert grade.passed is None, "跳过是未判定，不是判错"
    assert grade.attributable is False

    assert check_capabilities(NeedsTools(), frozenset({Cap.CHAT, Cap.TOOLS, Cap.TOOL_CHOICE})) is None


# ── 端到端：任务 + gateway + 落库 ─────────────────────────────────
@pytest.fixture
def env(tmp_path):
    db = Database(tmp_path / "onyx.sqlite")
    sink = SqliteRecordSink(db, batch_size=16, idle_wait=0.005)
    yield db, sink, ObserverEngine(record_sink=sink), FileBlobStore(tmp_path / "blobs")
    sink.close()
    db.close()


def test_grades_carry_a_real_trace_id(env):
    """DoD：每条 grade 都有 trace_id —— 分数必须能点进一条真实 trace。"""
    from onyx.eval.runner import EvalRunner, RunConfig

    db, sink, observer, blobs = env
    provider = MockProvider(
        scripts={"mock/classifier": [MockScript(text="转账", done_reason="stop")]},
        models=("mock/classifier",),
    )
    gateway = Gateway(provider, observer=observer, blobs=blobs)
    task = _task()
    runner = EvalRunner(gateway, EvalRepo(db), task, dataset=task.dataset)
    report = runner.run(RunConfig(model="mock/classifier"))

    sink.flush(5.0)
    assert report.status == "done"
    assert report.n_done == 1
    for grade in report.grades:
        assert grade.trace_id, "没有 trace_id 的分数不该被展示"
        assert TraceRepo(db).get(grade.trace_id) is not None, "trace_id 指向了一条不存在的 trace"
    assert report.grades[0].verdict is Verdict.CORRECT
