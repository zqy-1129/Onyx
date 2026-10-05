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
    """读 JSONL 文件。每行一个样本，必须含 `input`；`expect` 可空（latency_bench 这类任务）。"""
    target = Path(path)
    if not target.exists():
        raise DatasetError(f"数据集文件不存在: {target}", path=str(target))
    return parse_jsonl(
        target.read_text(encoding="utf-8"), name=target.stem,
        dataset_id=dataset_id, upstream=upstream or f"file:{target.name}",
        # 文件的 revision 用字节 hash：同一份文件两次导入必须得到同一个值，
        # 否则"换了数据版本"这件事在两次分数之间发现不了
        revision=revision or _file_revision(target), license=license, notes=notes,
    )


def parse_jsonl(
    text: str,
    *,
    name: str,
    dataset_id: str | None = None,
    upstream: str = "",
    revision: str = "",
    license: str = "",
    notes: str = "",
) -> Dataset:
    """把 JSONL **文本**解析成 Dataset（`load_jsonl` 与 HTTP 上传共用这一份解析）。

    分成两层是必须的：两个入口对"来历默认值"的算法不同（文件按字节 hash、
    上传按内容 hash），但**行的合法性判据必须完全一致**——否则同一份数据
    从 CLI 导入能跑、从界面导入报错（或反过来），而没人会想到是解析器有两份。
    """
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            case = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetError(
                f"第 {line_number} 行不是合法 JSON: {exc}", line=line_number, path=name
            ) from exc
        if not isinstance(case, dict):
            raise DatasetError(
                f"第 {line_number} 行不是对象: {type(case).__name__}",
                line=line_number, path=name,
            )
        if not isinstance(case.get("input"), dict) or not case["input"]:
            raise DatasetError(
                f"第 {line_number} 行缺少非空的 input 对象", line=line_number, path=name
            )
        case.setdefault("expect", {})
        case.setdefault("kind", "single")
        case.setdefault("tags", [])
        case.setdefault("meta", {})
        case_id = str(case.get("id") or content_id(
            name, case["input"], json.dumps(case["expect"], ensure_ascii=False, sort_keys=True)
        ))
        if case_id in seen_ids:
            raise DatasetError(
                f"第 {line_number} 行的 id {case_id!r} 重复；重复 id 会让 grade 互相覆盖",
                line=line_number, path=name,
            )
        seen_ids.add(case_id)
        case["id"] = case_id
        case.setdefault("ord", line_number - 1)
        cases.append(case)

    if not cases:
        raise DatasetError(f"{name or '上传内容'} 里没有任何样本", path=name)

    return Dataset(
        id=dataset_id or f"{name}-v1",
        cases=tuple(cases),
        upstream=upstream or (f"uploaded:{name}" if name else "uploaded"),
        # 没给 revision 就用内容 hash：同一段文本两次导入必须是同一个 revision，
        # 而"没填"绝不能变成空串（空 revision 等于放弃可比性判据）
        revision=revision or f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}",
        license=license, loader="jsonl", notes=notes,
    )


def register_dataset(db: Any, dataset: Dataset, *, repo: Any = None) -> list[str]:
    """把 Dataset 落库，并返回"这次导入会影响可比性"的警告。

    重复导入同一个 id 是覆盖（upsert），这本身没问题；有问题的是**覆盖了但没人知道**：
    历史 grade 仍然指向这个 id，而它现在指向另一份数据，于是"这两次分数可比吗"
    从"是"悄悄变成"不知道"。所以旧 revision / 旧条数与新值不一致时必须警告，
    CLI 打印它、界面显示它。
    """
    from onyx.store.repos import EvalRepo

    repo = repo or EvalRepo(db)
    warnings: list[str] = []
    existing = repo.get_dataset(dataset.id)
    record, cases = dataset.to_records()
    if existing is not None:
        if existing.revision and record.revision and existing.revision != record.revision:
            warnings.append(
                f"数据集 {dataset.id} 原先登记的 revision 是 {existing.revision!r}，"
                f"现在是 {record.revision!r}：指向它的历史分数与之后的分数不可比"
            )
        if existing.n_cases is not None and existing.n_cases != record.n_cases:
            warnings.append(
                f"条数从 {existing.n_cases} 变成 {record.n_cases}：同一个 id 换了规模，"
                "历史 run 的分母就对不上现在这份考卷了"
            )
    repo.upsert_dataset(record)
    repo.upsert_cases(cases)
    return warnings


def load_registered(db: Any, dataset_id: str) -> Dataset:
    """把**已在库里登记过**的数据集读回来（导入之后跑评测的第二步）。

    这一步以前不存在：`onyx eval import --id x` 之后 `--dataset x` 会报未知数据集，
    而库里明明有它的样本。读回来时必须带上来历字段，否则分数历史与考卷就断链了。
    """
    from onyx.store.repos import EvalRepo

    record = EvalRepo(db).get_dataset(dataset_id)
    if record is None:
        registered = sorted(item.id for item in EvalRepo(db).list_datasets())
        raise DatasetError(
            f"未知数据集 {dataset_id!r}；库里已登记的: {registered or '（一个都没有，先 onyx eval import）'}",
            path=dataset_id,
        )
    cases = tuple(
        {
            "id": case.id, "ord": case.ord, "kind": case.kind, "input": case.input,
            "expect": case.expect, "tools": list(case.tools), "fixture": case.fixture,
            "meta": case.meta, "tags": list(case.tags),
        }
        for case in EvalRepo(db).list_cases(dataset_id)
    )
    if not cases:
        # dataset 行在而样本不在：导入被中断或库被清过。返回空集会变成"0 条样本的评测"，
        # 那在界面上看起来完全正常（status=done、n_total=0），必须当场拒绝
        raise DatasetError(
            f"数据集 {dataset_id} 登记过但一条样本都没有；请重新导入", path=dataset_id
        )
    return Dataset(
        id=record.id, cases=cases, upstream=record.upstream, revision=record.revision,
        license=record.license, loader=f"{record.loader or 'registered'} (registered)",
        notes=record.notes,
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


def _build_structured_ie():
    from onyx.eval.datasets.builtin.structured_ie import build_cases

    return build_cases()


def _build_instructions_zh():
    from onyx.eval.datasets.builtin.instructions_zh import build_cases

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
    "structured_ie": {
        "build": _build_structured_ie,
        # revision 写的是**决定期望值的全部参数**（seed + 锚定日），而不是版本号：
        # 条数不变、内容变了的修正是最常见的"两个分数其实不可比"，
        # 光看 `seed=20261003` 会发现不了这件事。改生成器就必须改这一行。
        "revision": "seed=20261003+anchor=2026-03-01",
        "notes": "生成器：模板槽位轮换 + 人工难例（中文数字/相对日期/多主体）+ 无可抽取负样本；"
                 "相对日期的期望值由锚定日算出；见 onyx/eval/datasets/builtin/structured_ie.py",
    },
    "instructions_zh": {
        "build": _build_instructions_zh,
        "revision": "seed=20261005",
        "notes": "生成器：每条样本的约束由一句必然满足它的参考回答派生（所以题目一定有解），"
                 "提示语再由约束渲染出来（所以不会考没说过的话）；"
                 "见 onyx/eval/datasets/builtin/instructions_zh.py",
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
