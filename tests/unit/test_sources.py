"""S14 验收：外部数据集导入器。

导入器的价值全在**来历可追溯**与**丢样本可察觉**这两点上：
- `revision` 绝不能为空，否则两天后没人能回答"这两次跑的是不是同一份数据"；
- 跳过的多轮样本必须计数并写进 notes，否则样本数悄悄变少，
  而 macro_f1 的分母变化在报告上完全看不出来；
- `no_call` 子集必须落成独立 kind，否则误调率会被算进"没调对"里。
"""

from __future__ import annotations

import json

import pytest

from onyx.eval.datasets.loader import DatasetError
from onyx.eval.datasets.sources import (
    SOURCES,
    describe,
    detect_no_call_subset,
    import_bfcl,
    supported,
)


def _write(path, lines: list[dict]) -> str:
    path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in lines),
                    encoding="utf-8")
    return str(path)


@pytest.fixture
def bfcl_files(tmp_path):
    """一份最小 BFCL 风格样本：单轮、多轮、无工具、irrelevance 各一。"""
    questions = tmp_path / "questions.jsonl"
    answers = tmp_path / "answers.jsonl"
    _write(questions, [
        {"id": "q1", "question": [[{"role": "user", "content": "北京天气怎么样"}]],
         "function": [{"name": "get_weather", "description": "查询天气",
                       "parameters": {"type": "object", "properties": {
                           "city": {"type": "string", "description": "城市名"}},
                           "required": ["city"]}}]},
        {"id": "q2", "question": [[{"role": "user", "content": "第一轮"}],
                                  [{"role": "user", "content": "第二轮"}]],
         "function": []},
        {"id": "q3", "question": [[{"role": "user", "content": "帮我算 3+5"}]],
         "function": {"calculator": {"description": "算表达式",
                                     "parameters": {"type": "object",
                                                    "properties": {
                                                        "expr": {"type": "string",
                                                                 "description": "表达式"}},
                                                    "required": ["expr"]}}}},
        {"id": "irrelevance-0", "question": [[{"role": "user", "content": "解释递归"}]],
         "function": [], "invable_tool_name": None},
    ])
    _write(answers, [
        {"id": "q1", "ground_truth": [[{"name": "get_weather",
                                        "arguments": {"city": "北京"}}]]},
        {"id": "q2", "ground_truth": []},
        {"id": "q3", "ground_truth": '[{"name": "calculator", "arguments": {"expr": "3+5"}}]'},
        {"id": "irrelevance-0", "ground_truth": []},
    ])
    return questions, answers


def test_bfcl_import_maps_single_turn_cases(bfcl_files):
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers, dataset_id="bfcl-ast-v1", subset="ast")
    assert dataset.id == "bfcl-ast-v1"
    assert len(dataset) == 3, "多轮的那条被跳过，剩 3 条"
    kinds = {case["input"]["instruction"]: case["kind"] for case in dataset.cases}
    assert kinds["北京天气怎么样"] == "single"
    assert kinds["解释递归"] == "no_call_needed"


def test_bfcl_import_preserves_expected_arguments(bfcl_files):
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers)
    by_instruction = {case["input"]["instruction"]: case for case in dataset.cases}
    weather = by_instruction["北京天气怎么样"]
    assert weather["expect"]["calls"] == [
        {"name": "get_weather", "arguments": {"city": "北京"}}
    ]
    assert weather["expect"]["must_call"] is True
    # 答案是 JSON 字符串的写法也要能解析（上游两种格式都有）
    calc = by_instruction["帮我算 3+5"]
    assert calc["expect"]["calls"] == [{"name": "calculator", "arguments": {"expr": "3+5"}}]


def test_irrelevance_subset_becomes_no_call_needed(bfcl_files):
    """误调率要单独统计，前提是这些样本在数据里就是**独立的一类**。"""
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers)
    irr = next(c for c in dataset.cases if c["input"]["instruction"] == "解释递归")
    assert irr["kind"] == "no_call_needed"
    assert irr["expect"]["calls"] == []
    assert irr["expect"]["must_call"] is False


def test_no_call_detection_covers_upstream_naming_variants():
    assert detect_no_call_subset("irrelevance_detection") is True
    assert detect_no_call_subset("no_call") is True
    assert detect_no_call_subset("ast") is False
    assert detect_no_call_subset("IRRELEVANCE") is True, "大小写不该影响归类"


def test_tools_from_dict_shaped_functions_are_normalised(bfcl_files):
    """BFCL 有的子集按函数名分桶，有的给数组。两种都得能用，但形状必须统一。"""
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers)
    by_instruction = {case["input"]["instruction"]: case for case in dataset.cases}
    calc = by_instruction["帮我算 3+5"]
    assert calc["tools"][0]["name"] == "calculator"
    assert calc["tools"][0]["parameters"]["required"] == ["expr"]


def test_skipped_multi_turn_cases_are_counted_not_silent(bfcl_files):
    """静默丢样本比报错危险：n 变小在报告上看不出来，而 macro_f1 的分母变了。

    另一半风险是"塌缩成空指令后被导入"——样本数看着对，内容却是坏的，
    所以两个方向都要断言。
    """
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers)
    assert "跳过 1 条多轮样本" in dataset.notes
    instructions = [case["input"]["instruction"] for case in dataset.cases]
    assert "第一轮" not in instructions and "第二轮" not in instructions
    assert all(item.strip() for item in instructions), "导入了空指令的样本"
    assert dataset.splits()["default"] == len(dataset.cases) == 3


def test_revision_defaults_to_file_hash(bfcl_files):
    """revision 绝不能为空：否则"这两次是不是同一份数据"就成了无法回答的问题。"""
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers)
    assert dataset.revision.startswith("sha256:")
    again = import_bfcl(questions, answers)
    assert again.revision == dataset.revision, "同一份文件两次导入必须得到同一个 revision"

    # 内容变了 revision 必须变
    _write(questions, [{"id": "q9", "question": [[{"role": "user", "content": "换个"}]],
                        "function": []}])
    assert import_bfcl(questions, answers).revision != dataset.revision


def test_explicit_revision_and_license_are_kept(bfcl_files):
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers, revision="git:abc123", license="ODC-BY-1.0")
    assert dataset.revision == "git:abc123"
    assert dataset.license == "ODC-BY-1.0"


def test_missing_license_is_flagged_not_blank(bfcl_files):
    """转载数据集没有许可证是个合规问题，必须显眼。"""
    questions, answers = bfcl_files
    dataset = import_bfcl(questions, answers)
    assert "许可证" in dataset.license


def test_answers_are_paired_by_id_not_by_order(bfcl_files):
    """两个文件顺序不一致很常见。按行号配对会把 A 的答案配到 B 的问题上，
    而错得看起来完全合理——所有分数都会偏低，且没人怀疑是数据对齐问题。"""
    questions, answers = bfcl_files
    original = [json.loads(line) for line in
                answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    _write(answers, list(reversed(original)))   # 顺序整个倒过来，id 不变
    dataset = import_bfcl(questions, answers)
    weather = next(c for c in dataset.cases
                   if c["input"]["instruction"] == "北京天气怎么样")
    assert weather["expect"]["calls"] == [
        {"name": "get_weather", "arguments": {"city": "北京"}}
    ], "答案必须按 id 配对，不能按行号"


def test_case_ids_are_content_derived_and_stable(bfcl_files):
    """id 由内容算出：重新导入不产生新 id，历史 grade 仍能对上。"""
    questions, answers = bfcl_files
    first = import_bfcl(questions, answers)
    second = import_bfcl(questions, answers)
    assert [c["id"] for c in first.cases] == [c["id"] for c in second.cases]
    assert len({c["id"] for c in first.cases}) == len(first.cases)


def test_missing_files_raise_instead_of_returning_empty(tmp_path):
    with pytest.raises(DatasetError, match="不存在"):
        import_bfcl(tmp_path / "nope.jsonl")
    (tmp_path / "bad.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(DatasetError, match="空的"):
        import_bfcl(tmp_path / "bad.jsonl")


def test_broken_line_reports_its_number(bfcl_files, tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "a", "question": []}\n{坏了\n', encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        import_bfcl(path)
    assert exc.value.line == 2


def test_question_without_tools_still_imports(tmp_path):
    """没有工具定义的样本仍然是合法的 no_call_needed，不该因为缺 function 就报错。"""
    path = tmp_path / "q.jsonl"
    _write(path, [{"id": "a", "question": [[{"role": "user", "content": "你好"}]]}])
    dataset = import_bfcl(path)
    assert dataset.cases[0]["kind"] == "no_call_needed"


def test_unsupported_source_is_rejected_with_the_option_list():
    assert SOURCES == ("bfcl", "jsonl")
    with pytest.raises(DatasetError, match="不支持的数据源"):
        supported(source="lmsys")
    supported(source="bfcl")  # 不抛


def test_upstream_defaults_to_source_and_subset(bfcl_files):
    """来源要精确到子集，否则两次跑不同子集会看起来是同一份数据。

    `--upstream` 给定时用给定的：镜像/转换后的文件需要记它真实的出处。
    """
    questions, answers = bfcl_files
    assert import_bfcl(questions, answers, subset="ast").upstream == "bfcl:ast"
    mirrored = import_bfcl(questions, answers, subset="ast", upstream="hf:mirror/ast")
    assert mirrored.upstream == "hf:mirror/ast"


def test_describe_shows_provenance_first(bfcl_files):
    questions, answers = bfcl_files
    text = describe(import_bfcl(questions, answers, subset="ast"))
    assert text.startswith("bfcl-v1 ·")
    assert "bfcl:ast" in text and "sha256:" in text
    assert "single" in text, "各 kind 的条数要能一眼看到"
