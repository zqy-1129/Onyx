"""数据集载入与导入。

三条纪律：
1. **来历必须落库**（upstream / revision / license / loader）。
   换了数据集版本之后，两次评测的分数不可比；不记 revision 就永远发现不了这件事。
2. **坏行要报出是第几行**，不能静默跳过。静默跳过会让 n 悄悄变小，
   而 macro_f1 的分母变化在报告上完全看不出来。
3. 样本 id 由**内容**决定（生成器已算好，或这里按内容 hash 补），
   所以重新导入同一份数据不会产生新 id，历史 grade 仍能对上。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.store.records import CaseRecord, DatasetRecord


class DatasetError(ValueError):
    """数据集格式错误。带行号，因为"第 37 行坏了"才是可行动的信息。"""

    def __init__(self, message: str, *, line: int | None = None, path: str = "") -> None:
        super().__init__(message)
        self.line = line
        self.path = path


@dataclass(frozen=True, slots=True)
class Dataset:
    id: str
    cases: tuple[dict[str, Any], ...]
    upstream: str = ""
    revision: str = ""
    license: str = ""
    loader: str = ""
    notes: str = ""

    def __len__(self) -> int:
        return len(self.cases)

    def splits(self) -> dict[str, int]:
        """按 tag 划分子集条数。`hard` 这类子集要能单独跑。"""
        counts: dict[str, int] = {"default": len(self.cases)}
        for case in self.cases:
            for tag in case.get("tags") or ():
                counts[str(tag)] = counts.get(str(tag), 0) + 1
        return counts

    def select(
        self, *, split: str = "default", limit: int | None = None, offset: int = 0
    ) -> tuple[dict[str, Any], ...]:
        """取子集。

        `--limit` 取的是**前 N 条**，所以生成器必须保证顺序稳定且各类混合，
        否则两次跑的子集不同，分数差异无法解释（见 intent_zh.py 的 shuffle）。
        """
        if split and split != "default":
            pool = [c for c in self.cases if split in (c.get("tags") or ())]
            if not pool:
                raise DatasetError(f"数据集 {self.id} 没有名为 {split!r} 的子集")
        else:
            pool = list(self.cases)
        pool.sort(key=lambda c: int(c.get("ord") or 0))
        return tuple(pool[offset:] if limit is None else pool[offset : offset + limit])

    def to_records(self) -> tuple[DatasetRecord, list[CaseRecord]]:
        dataset = DatasetRecord(
            id=self.id, imported_at=utc_now_iso(), upstream=self.upstream,
            revision=self.revision, license=self.license, splits=self.splits(),
            n_cases=len(self.cases), loader=self.loader, notes=self.notes,
        )
        cases = [
            CaseRecord(
                id=str(case["id"]), dataset_id=self.id, ord=int(case.get("ord") or index),
                kind=str(case.get("kind") or "single"), input=dict(case.get("input") or {}),
                expect=dict(case.get("expect") or {}),
                tools=tuple(case.get("tools") or ()),
                fixture=dict(case.get("fixture") or {}), meta=dict(case.get("meta") or {}),
                tags=tuple(case.get("tags") or ()),
            )
            for index, case in enumerate(self.cases)
        ]
        return dataset, cases


def content_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


def load_jsonl(
    path: Path | str,
    *,
    dataset_id: str | None = None,
    upstream: str = "",
    revision: str = "",
    license: str = "",
    notes: str = "",
) -> Dataset:
    """读 JSONL。每行一个样本，必须含 `input`；`expect` 可空（latency_bench 这类任务）。"""
    target = Path(path)
    if not target.exists():
        raise DatasetError(f"数据集文件不存在: {target}", path=str(target))

    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(
        target.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"第 {line_number} 行不是合法 JSON: {exc}", line=line_number, path=str(target)
            ) from exc
        if not isinstance(case, dict):
            raise DatasetError(
                f"第 {line_number} 行不是对象: {type(case).__name__}",
                line=line_number, path=str(target),
            )
        if not isinstance(case.get("input"), dict) or not case["input"]:
            raise DatasetError(
                f"第 {line_number} 行缺少非空的 input 对象", line=line_number, path=str(target)
            )
        case.setdefault("expect", {})
        case.setdefault("kind", "single")
        case.setdefault("tags", [])
        case.setdefault("meta", {})
        case_id = str(case.get("id") or content_id(
            target.stem, case["input"], json.dumps(case["expect"], ensure_ascii=False, sort_keys=True)
        ))
        if case_id in seen_ids:
            raise DatasetError(
                f"第 {line_number} 行的 id {case_id!r} 重复；重复 id 会让 grade 互相覆盖",
                line=line_number, path=str(target),
            )
        seen_ids.add(case_id)
        case["id"] = case_id
        case.setdefault("ord", line_number - 1)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{target} 里没有任何样本", path=str(target))

    return Dataset(
        id=dataset_id or f"{target.stem}-v1",
        cases=tuple(cases),
        upstream=upstream or f"file:{target.name}",
        revision=revision or _file_revision(target),
        license=license, loader="jsonl", notes=notes,
    )


def load_builtin(name: str = "intent_zh", **kw: Any) -> Dataset:
    """载入随包分发的数据集。

    直接调生成器而不是读 JSONL：这样即使仓库里那份 .jsonl 被误删或改坏，
    评测仍然可复现，而且生成参数（seed）本身就是 revision 的一部分。

    `**kw` 里非 None 的项覆盖默认值——CLI 会用 `--upstream/--revision/--license`
    补真实来历，None 表示"调用方没指定"，不该把默认值冲掉。
    """
    entry = _BUILTIN.get(name)
    if entry is None:
        raise DatasetError(
            f"没有内置数据集 {name!r}；可选: {sorted(_BUILTIN)}", path=name
        )

    overrides = {key: value for key, value in kw.items() if value is not None}
    dataset_id = overrides.pop("dataset_id", None) or f"{name}-v1"
    fields: dict[str, Any] = {
        "id": dataset_id,
        "cases": tuple(entry["build"]()),
        "upstream": f"builtin:{name}",
        "revision": entry["revision"],
        "license": "generated-in-repo",
        "loader": f"builtin.{name}",
        "notes": entry["notes"],
    }
    fields.update(overrides)
    return Dataset(**fields)


def _build_intent_zh():
    from onyx.eval.datasets.builtin.intent_zh import build_cases

    return build_cases()


def _build_tool_calls_zh():
    from onyx.eval.datasets.builtin.tool_calls_zh import build_cases

    return build_cases()


#: 内置数据集登记表。加一个数据集只需要在这里加一行 + 一个生成器模块
_BUILTIN: dict[str, dict[str, Any]] = {
    "intent_zh": {
        "build": _build_intent_zh,
        "revision": "seed=20261003",
        "notes": "模板生成 + 人工补充难例；见 onyx/eval/datasets/builtin/intent_zh.py",
    },
    "tool_calls_zh": {
        "build": _build_tool_calls_zh,
        "revision": "seed=20261003",
        "notes": "手写工具调用样本，含 no_call_needed 子集；"
                 "见 onyx/eval/datasets/builtin/tool_calls_zh.py",
    },
}


def builtin_names() -> tuple[str, ...]:
    return tuple(sorted(_BUILTIN))


def _file_revision(path: Path) -> str:
    """文件内容 hash 作 revision：同一份文件两次导入必须得到同一个 revision。"""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"sha256:{digest}"


def iter_cases(dataset: Dataset, *, split: str = "default", limit: int | None = None) -> Iterator[dict[str, Any]]:
    yield from dataset.select(split=split, limit=limit)


def case_ids(cases: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(str(case["id"]) for case in cases)
