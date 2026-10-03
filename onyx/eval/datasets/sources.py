"""外部数据集导入器。

只支持"从本地文件导入"，**不替你下载**。理由不是省事：

- 下载会在评测路径上引入网络依赖，于是"离线复现一次评测"变得做不到，
  而这正是本地评测相对云 API 的主要优势；
- 上游改版本会让分数静默不可比。所以下载这一步交给使用方，
  导入时**必须**记下 `--revision`，这样两次运行是不是同一份数据是可以回答的。

BFCL 的原始格式是"问题 + 函数定义"和"参考答案"两个文件分开，
且一条样本可能包含多轮。这里只处理**单轮**子集，多轮的直接跳过并计数——
静默丢掉一半样本比报错危险得多，所以条数会写进 `Dataset.notes` 和 `splits`。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from onyx.eval.datasets.loader import Dataset, DatasetError
from onyx.eval.graders.normalize import normalize_text

#: 支持的数据源标识。未知的一律报错并列出可选，不静默退回内置集
SOURCES = ("bfcl", "jsonl")

#: BFCL 里这些子集是"不该调用工具"的负样本，必须单独成 kind，
#: 否则误调率会被算进"没调对"里（DESIGN §9.2）
NO_CALL_SUBSTRINGS = ("no_call", "irrelevance", "irrelevance_detection")


def import_bfcl(
    questions: Path | str,
    answers: Path | str | None = None,
    *,
    dataset_id: str = "bfcl-v1",
    subset: str = "v1",
    upstream: str = "",
    revision: str = "",
    license: str = "",
) -> Dataset:
    """把 BFCL 风格的 `question`/`function` + answer 文件转成内部样本。

    `revision` 留空时用文件内容 hash 兜底——**绝不能留空**，
    否则两天后没人能回答"这两次跑的是不是同一份数据"。
    """
    qpath, apath = Path(questions), Path(answers) if answers else None
    if not qpath.exists():
        raise DatasetError(f"问题文件不存在: {qpath}", path=str(qpath))

    records = _read_jsonl(qpath, line_key="question")
    answers_by_id = _answers_by_id(apath) if apath and apath.exists() else {}

    cases: list[dict[str, Any]] = []
    skipped_multi_turn = 0
    for number, raw in records:
        payload = raw
        question = payload.get("question") or payload.get("user_query") or []
        funcs = payload.get("function") or payload.get("functions") or payload.get("tools") or []
        instance_id = str(payload.get("id") or payload.get("question_id") or f"bfcl-{number}")

        turns = _turns(question)
        if len(turns) != 1:
            # 多轮需要状态化地喂回工具结果，当前 runner 不建模对话历史，
            # 所以只能跳过——但必须计数并写进 notes，否则样本数悄悄变少
            skipped_multi_turn += 1
            continue
        instruction = turns[0]
        tools = _tool_specs(funcs)
        expected = _expected_calls(answers_by_id.get(instance_id), payload)
        no_call = _is_no_call(instance_id, subset, payload) or not expected

        cases.append({
            "id": _case_id(instance_id, instruction),
            "input": {"instruction": instruction, "tools": [t["name"] for t in tools]},
            "expect": {"calls": [] if no_call else expected,
                       "must_call": not no_call and bool(expected)},
            "tools": tools,
            "kind": "no_call_needed" if no_call else ("parallel" if len(expected) > 1 else "single"),
            "tags": ["upstream-bfcl", subset],
            "meta": {"upstream_id": instance_id, "raw": _shrink(payload)},
        })

    if not cases:
        raise DatasetError(
            f"{qpath} 里没有任何可导入的单轮样本"
            + (f"（跳过了 {skipped_multi_turn} 条多轮）" if skipped_multi_turn else ""),
            path=str(qpath),
        )
    for index, case in enumerate(cases):
        case["ord"] = index

    notes = (
        f"来自 {qpath.name}"
        + (f" + {apath.name}" if apath else "（无答案文件，按'不该调用'处理）")
        + (f"；跳过 {skipped_multi_turn} 条多轮样本" if skipped_multi_turn else "")
    )
    return Dataset(
        id=dataset_id, cases=tuple(cases), upstream=upstream or f"bfcl:{subset}",
        revision=revision or _file_hash(qpath, apath),
        license=license or "unspecified — 转载前必须补上游许可证",
        loader="bfcl", notes=notes,
    )


def detect_no_call_subset(name: str) -> bool:
    lowered = normalize_text(name)
    return any(marker in lowered for marker in NO_CALL_SUBSTRINGS)


def supported(*, source: str) -> None:
    if source not in SOURCES:
        raise DatasetError(
            f"不支持的数据源 {source!r}；可选: {list(SOURCES)}。"
            "新增一种需要显式写导入器，因为不同上游的字段形状差异很大，"
            "靠猜会把样本对齐错。", path=source,
        )


# ── 内部 ──────────────────────────────────────────────────────────
def _read_jsonl(path: Path, *, line_key: str = "") -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name} 第 {number} 行不是合法 JSON: {exc}",
                               line=number, path=str(path)) from exc
        if not isinstance(payload, dict):
            raise DatasetError(f"{path.name} 第 {number} 行不是对象", line=number, path=str(path))
        out.append((number, payload))
    if not out:
        raise DatasetError(f"{path} 是空的", path=str(path))
    return out


def _answers_by_id(path: Path) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for _, payload in _read_jsonl(path):
        if not isinstance(payload, dict):
            continue
        key = str(payload.get("id") or payload.get("question_id") or payload.get("questionKey") or "")
        if key:
            mapping[key] = payload
    return mapping


def _turns(question: Any) -> list[str]:
    """展开成"每轮一句用户指令"。返回长度 >1 就是多轮，调用方据此跳过并计数。

    长度必须真实反映轮数：这里曾经把多轮样本塌缩成一条空指令，于是它们
    不是被跳过而是**带着空输入被导入**，notes 里的跳过数也变成 0——
    样本数看着对，内容却是坏的。
    """
    if isinstance(question, str):
        return [question]
    if not isinstance(question, list) or not question:
        return []

    def user_text(turn: Any) -> str:
        if not isinstance(turn, list):
            return ""
        pieces = [str(item.get("content") or "") for item in turn
                  if isinstance(item, dict) and item.get("role") == "user"]
        return " ".join(piece for piece in pieces if piece)

    if isinstance(question[0], list):
        # BFCL 的标准形状：外层是轮，内层是该轮的消息列表
        return [user_text(turn) for turn in question]
    if isinstance(question[0], dict):
        # 已经拍平成单轮消息列表
        return [user_text(question)]
    return []


def _tool_specs(funcs: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if isinstance(funcs, dict):
        # 有的子集按函数名分桶：{"get_weather": {description, parameters}}
        for name, body in funcs.items():
            if isinstance(body, dict):
                out.append({"name": str(name), "description": str(body.get("description") or ""),
                            "parameters": body.get("parameters") or {}})
        return out
    for item in funcs or []:
        if not isinstance(item, dict):
            continue
        body = item.get("function") if isinstance(item.get("function"), dict) else item
        out.append({
            "name": str(body.get("name") or ""),
            "description": str(body.get("description") or ""),
            "parameters": body.get("parameters") or {},
        })
    return [item for item in out if item["name"]]


def _expected_calls(answer: Any, payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw = answer if answer is not None else payload.get("answer") or payload.get("ground_truth")
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = raw.get("ground_truth") or raw.get("calls") or raw.get("single_call") or []
        if isinstance(raw, str):
            raw = _parse_possible_json(raw)
    if isinstance(raw, str):
        raw = _parse_possible_json(raw)
    if isinstance(raw, dict):
        raw = [raw]
    out: list[dict[str, Any]] = []
    for group in raw if isinstance(raw, list) else []:
        if isinstance(group, dict):
            out.append({"name": str(group.get("name") or group.get("function") or ""),
                        "arguments": dict(group.get("arguments") or group.get("parameters") or {})})
        elif isinstance(group, list):
            for item in group:
                if isinstance(item, dict):
                    out.append({"name": str(item.get("name") or ""),
                                "arguments": dict(item.get("arguments") or {})})
    return [item for item in out if item["name"]]


def _parse_possible_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return []


def _is_no_call(instance_id: str, subset: str, payload: dict[str, Any]) -> bool:
    if detect_no_call_subset(subset) or detect_no_call_subset(instance_id):
        return True
    return payload.get("invable_tool_name") is None and str(
        payload.get("category") or payload.get("test_category") or ""
    ).lower() in {"irrelevance", "no_call", "no_tool_call"}


def _case_id(instance_id: str, instruction: str) -> str:
    digest = hashlib.sha256(f"{instance_id}|{instruction}".encode()).hexdigest()[:16]
    return f"bf-{digest}"


def _file_hash(*paths: Path | None) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path and path.exists():
            digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()[:16]}"


def _shrink(payload: dict[str, Any]) -> dict[str, Any]:
    """只留诊断需要的字段。把整份上游记录塞进 meta 会让 grade 表迅速膨胀。"""
    keep = ("id", "question_id", "category", "test_category", "invable_tool_name")
    return {key: payload[key] for key in keep if key in payload} | {"_trimmed": len(payload) > len(keep)}


def describe(dataset: Dataset) -> str:
    """一行摘要，给 CLI 打印。来历必须出现在第一行。"""
    counts: dict[str, int] = {}
    for case in dataset.cases:
        counts[str(case.get("kind") or "?")] = counts.get(str(case.get("kind") or "?"), 0) + 1
    return (f"{dataset.id} · {len(dataset)} 条 · 来源 {dataset.upstream} · "
            f"revision {dataset.revision} · kinds {counts}")
