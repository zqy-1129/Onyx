"""S24 验收：数据集的"导入 → 读回 → 跑评测"闭环。

这一段的价值全在**同一份数据在两个入口是同一个东西**：
- 解析只有一份（`parse_jsonl`），CLI 与 API 对同一行坏数据必须给同一个行号；
- 落库只有一份（`register_dataset`），覆盖旧 revision 时两边说同一句警告；
- 读回必须有（`load_registered`），否则 `eval import --id x` 之后 `--dataset x`
  会报"未知数据集"，而样本明明就在同一张库里。

以及一条老规矩：**空集合不许冒充成功**。登记过却一条样本都没有的数据集，
跑起来会是 `status=done, n_total=0` 这种看起来完全正常的东西。
"""

from __future__ import annotations

import json

import pytest

from onyx.eval.datasets.loader import (
    Dataset,
    DatasetError,
    load_jsonl,
    load_registered,
    parse_jsonl,
    register_dataset,
)
from onyx.eval.tasks import load_dataset
from onyx.store.db import Database
from onyx.store.repos import EvalRepo


def _cases(n: int = 3) -> list[dict]:
    return [
        {"id": f"c{i}", "ord": i, "input": {"instruction": f"第 {i} 条"},
         "expect": {"label": "转账"}, "tags": ["hard"] if i == 0 else [], "kind": "single"}
        for i in range(n)
    ]


def _jsonl(items: list[dict]) -> str:
    return "\n".join(json.dumps(item, ensure_ascii=False) for item in items)


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "onyx.sqlite")
    yield database
    database.close()


# ── 读回 ──────────────────────────────────────────────────────────
def test_registered_dataset_reads_back_identical_cases(db):
    """导入 → 读回必须是同一份考卷：id 顺序、期望、tags、工具、fixture 一个都不能变味。"""
    text = _jsonl([
        {**case, "tools": [{"name": "get_weather"}], "fixture": {"get_weather": "晴"},
         "meta": {"src": "manual"}}
        for case in _cases(4)
    ])
    original = parse_jsonl(text, name="rt", dataset_id="rt-v1", upstream="manual", license="MIT")
    register_dataset(db, original)

    back = load_registered(db, "rt-v1")
    assert back.id == "rt-v1"
    assert back.upstream == "manual" and back.license == "MIT"
    assert back.revision == original.revision, "读回来的 revision 必须与导入时一致"
    assert [case["id"] for case in back.cases] == [case["id"] for case in original.cases]
    assert [case["ord"] for case in back.cases] == [case["ord"] for case in original.cases]
    assert back.cases[0]["tools"] == [{"name": "get_weather"}], "工具定义丢了就等于换了任务"
    assert back.cases[0]["fixture"] == {"get_weather": "晴"}, "返回值桩丢了评测就不可复现"
    assert back.cases[0]["meta"] == {"src": "manual"}
    assert back.splits()["hard"] == 1


def test_load_dataset_resolves_a_registered_id_with_a_db(db):
    """这一步以前是断的：`eval import --id x` 之后 `--dataset x` 报未知数据集。"""
    register_dataset(db, parse_jsonl(_jsonl(_cases(2)), name="mine", dataset_id="mine-v1"))
    dataset = load_dataset("mine-v1", task_id="intent_classification", db=db)
    assert dataset.id == "mine-v1" and len(dataset) == 2


def test_unknown_id_with_a_db_lists_what_is_registered(db):
    register_dataset(db, parse_jsonl(_jsonl(_cases(1)), name="a", dataset_id="present-v1"))
    with pytest.raises(DatasetError) as exc:
        load_dataset("absent", task_id="intent_classification", db=db)
    assert "未知数据集" in str(exc.value)
    assert "present-v1" in str(exc.value), "报错要列出库里到底有什么，否则只能靠猜"


def test_unknown_id_without_a_db_keeps_the_old_message():
    """没带 db 时不许偷偷"当成不存在"就报一个看不懂的错——它得说清怎么才认得出来。"""
    with pytest.raises(KeyError) as exc:
        load_dataset("whatever", task_id="intent_classification")
    assert "未知数据集" in str(exc.value)
    assert "import" in str(exc.value)


def test_registered_but_empty_dataset_is_refused_not_run_as_zero_cases(db):
    repo = EvalRepo(db)
    record, _cases_rows = Dataset(id="hollow-v1", cases=(), upstream="test", revision="r1").to_records()
    repo.upsert_dataset(record)  # 只有 dataset 行、没有样本：导入被中断或库被动过

    with pytest.raises(DatasetError, match="一条样本都没有"):
        load_registered(db, "hollow-v1")


# ── 解析只有一份 ──────────────────────────────────────────────────
def test_file_and_text_parsers_reject_the_same_line_the_same_way(tmp_path):
    """两个入口共用一个行校验器：否则"CLI 能跑、界面报错"会永远查不完。"""
    broken = _jsonl(_cases(2)) + '\n{"input": 3}\n'
    path = tmp_path / "broken.jsonl"
    path.write_text(broken, encoding="utf-8")

    with pytest.raises(DatasetError) as from_file:
        load_jsonl(path)
    with pytest.raises(DatasetError) as from_text:
        parse_jsonl(broken, name="broken")
    assert from_file.value.line == from_text.value.line == 3
    assert "input" in str(from_file.value) and "input" in str(from_text.value)


def test_text_upload_revision_is_the_content_hash_and_is_stable():
    text = _jsonl(_cases(2))
    first = parse_jsonl(text, name="up")
    second = parse_jsonl(text, name="up")
    assert first.revision.startswith("sha256:") and first.revision == second.revision
    assert first.upstream == "uploaded:up", "没填 upstream 也要说得清它是从哪来的"

    other = parse_jsonl(_jsonl(_cases(3)), name="up")
    assert other.revision != first.revision, "内容变了 revision 必须变，否则可比性判据是假的"


def test_explicit_revision_wins_over_the_content_hash():
    dataset = parse_jsonl(_jsonl(_cases(1)), name="up", revision="v3")
    assert dataset.revision == "v3"


# ── 覆盖必须说话 ──────────────────────────────────────────────────
def test_reimport_with_a_different_revision_warns(db):
    register_dataset(db, parse_jsonl(_jsonl(_cases(2)), name="w", dataset_id="w-v1", revision="r1"))

    warnings = register_dataset(db, parse_jsonl(_jsonl(_cases(3)), name="w",
                                                 dataset_id="w-v1", revision="r2"))

    assert any("revision" in text and "不可比" in text for text in warnings)
    assert any("条数从 2 变成 3" in text for text in warnings)
    assert EvalRepo(db).get_dataset("w-v1").revision == "r2", "覆盖本身是允许的，但必须留下话"


def test_identical_reimport_is_silent(db):
    """同一份数据重复导入不该刷警告——那会让人把所有警告都当成噪音。"""
    dataset = parse_jsonl(_jsonl(_cases(2)), name="s", dataset_id="s-v1", revision="r1")
    register_dataset(db, dataset)
    assert register_dataset(db, parse_jsonl(_jsonl(_cases(2)), name="s",
                                             dataset_id="s-v1", revision="r1")) == []
    assert EvalRepo(db).count_cases("s-v1") == 2, "重复导入不许产生第二份样本"


def test_first_import_has_nothing_to_warn_about(db):
    assert register_dataset(db, parse_jsonl(_jsonl(_cases(1)), name="n", dataset_id="n-v1")) == []
