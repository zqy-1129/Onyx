"""工具注册表：定义 CRUD、内容 hash 版本化、批量审计、上下文开销核算。

版本化规则：`hash`（name+description+parameters+kind+side_effect）变化 ⇒ version 递增。
旧 trace 通过 `tool_call.tool_def_hash` 仍能定位当时的定义——否则"改描述前后模型表现对比"
这个最常见的调优动作就没有可比性。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from onyx.core.clock import utc_now_iso
from onyx.core.ids import new_trace_id
from onyx.core.types import ToolSpec
from onyx.llm.measurement.heuristic import estimate_tokens
from onyx.store.records import ToolDefRecord
from onyx.store.repos.tool_repo import ToolRepo
from onyx.tools.spec import (
    AuditFinding,
    SideEffect,
    ToolDef,
    ToolKind,
    audit_many,
    cost_report,
    openai_json,
)

CountFn = Callable[[str], int]


def default_count_fn() -> CountFn:
    """未指定模型时的兜底计数（heuristic 档，标 low 置信）。"""
    fn: CountFn = lambda text: estimate_tokens(text)  # noqa: E731
    fn.source_name = "heuristic"  # type: ignore[attr-defined]
    return fn


class ToolRegistry:
    def __init__(self, repo: ToolRepo, *, count_fn: CountFn | None = None) -> None:
        self.repo = repo
        self.count_fn = count_fn or default_count_fn()

    # ── 注册 ──────────────────────────────────────────────────────
    def register(self, definition: ToolDef, *, tokens: int | None = None) -> ToolDefRecord:
        payload = openai_json(definition)
        digest = definition.hash
        existing = self.repo.find_by_name(definition.name)
        if existing is None:
            version = definition.version or "1"
            created = utc_now_iso()
        elif existing.hash == digest:
            # 内容没变：保留版本号，只刷新开销（换模型标定时 tokens 会变）
            version = existing.version
            created = existing.created_at
        else:
            version = str(_bump(existing.version))
            created = existing.created_at
        record = ToolDefRecord(
            id=existing.id if existing else new_trace_id(),
            name=definition.name, version=version, kind=str(definition.kind),
            schema_json=definition.parameters, description=definition.description,
            hash=digest, impl_ref=definition.impl_ref,
            tokens=self.count_fn(payload) if tokens is None else tokens,
            bytes=len(payload.encode("utf-8")), tags=definition.tags, owner=definition.owner,
            enabled=definition.enabled, side_effect=str(definition.side_effect),
            timeout_ms=definition.timeout_ms, doc=definition.doc,
            examples=tuple(definition.examples),
            extra={**definition.extra, "content_hash": digest},
            created_at=created,
        )
        self.repo.upsert_def(record)
        return record

    def register_many(self, definitions: Iterable[ToolDef]) -> list[ToolDefRecord]:
        return [self.register(d) for d in definitions]

    def refresh_costs(self) -> int:
        """换模型/换标定后重算所有工具的开销。不改版本号（内容没变）。"""
        count = 0
        for record in self.repo.list_defs():
            definition = self.to_def(record)
            self.register(definition)
            count += 1
        return count

    # ── 读取 ──────────────────────────────────────────────────────
    def get(self, name: str) -> ToolDef | None:
        record = self.repo.find_by_name(name)
        return self.to_def(record) if record else None

    def list(self, *, enabled_only: bool = False) -> list[ToolDef]:
        return [self.to_def(r) for r in self.repo.list_defs(enabled_only=enabled_only)]

    def specs(self, names: Sequence[str] | None = None) -> tuple[ToolSpec, ...]:
        """给 gateway 用的 ToolSpec 列表（只含启用的工具）。"""
        wanted = set(names) if names else None
        return tuple(
            definition.to_spec()
            for definition in self.list(enabled_only=True)
            if wanted is None or definition.name in wanted
        )

    @staticmethod
    def to_def(record: ToolDefRecord) -> ToolDef:
        return ToolDef(
            name=record.name, description=record.description, parameters=record.schema_json,
            kind=ToolKind(record.kind), side_effect=SideEffect(record.side_effect),
            impl_ref=record.impl_ref, version=record.version, tags=record.tags,
            owner=record.owner, enabled=record.enabled, timeout_ms=record.timeout_ms,
            doc=record.doc, examples=tuple(record.examples), extra=dict(record.extra),
        )

    # ── 审计与开销 ────────────────────────────────────────────────
    def audit_all(self, definitions: Iterable[ToolDef] | None = None) -> dict[str, list[AuditFinding]]:
        return audit_many(
            definitions if definitions is not None else self.list(), count_fn=self.count_fn
        )

    def cost(self, *, template_overhead: int = 0) -> dict[str, Any]:
        """工具库上下文开销。

        `template_overhead` 应传该模型实测的 `template_ctl` 残差（trace 归因里就有）。
        P17：一个 78 token 的 JSON 实吃约 291 token，其中 73% 是模板脚手架——
        只报 JSON 大小会把优化方向引到"精简描述"，而真正有效的是换模板/换模型。
        """
        return cost_report(
            self.list(enabled_only=True), count_fn=self.count_fn,
            template_overhead=template_overhead,
        )


def _bump(version: str) -> int:
    try:
        return int(version) + 1
    except ValueError:
        return 2


def defs_from_payload(payload: Any) -> list[ToolDef]:
    """从 YAML/JSON 载入的定义列表构造 ToolDef。

    输入是**外部数据**，所以每个字段都做显式转换与校验：
    非法的 kind/side_effect 必须报错并指名是哪个工具，而不是静默取默认值。
    """
    if isinstance(payload, dict):
        payload = payload.get("tools") or []
    if not isinstance(payload, list):
        raise ValueError("工具定义必须是列表，或 {'tools': [...]} 形式")

    out: list[ToolDef] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index} 项不是对象: {raw!r}")
        name = str(raw.get("name") or "")
        if not name:
            raise ValueError(f"第 {index} 项缺少 name")
        try:
            kind = ToolKind(str(raw.get("kind") or "python_fn"))
        except ValueError as exc:
            raise ValueError(f"工具 {name} 的 kind 非法: {raw.get('kind')!r}") from exc
        try:
            side_effect = SideEffect(str(raw.get("side_effect") or "read"))
        except ValueError as exc:
            raise ValueError(
                f"工具 {name} 的 side_effect 非法: {raw.get('side_effect')!r}"
                f"（可选: {[str(s) for s in SideEffect]}）"
            ) from exc
        out.append(ToolDef(
            name=name,
            description=str(raw.get("description") or ""),
            parameters=dict(raw.get("parameters") or {}),
            kind=kind, side_effect=side_effect,
            impl_ref=str(raw.get("impl_ref") or raw.get("impl") or ""),
            version=str(raw.get("version") or "1"),
            tags=tuple(raw.get("tags") or ()), owner=str(raw.get("owner") or ""),
            enabled=bool(raw.get("enabled", True)),
            timeout_ms=raw.get("timeout_ms"), doc=str(raw.get("doc") or ""),
            examples=tuple(raw.get("examples") or ()),
            extra={k: v for k, v in raw.items() if k not in {
                "name", "description", "parameters", "kind", "side_effect", "impl_ref", "impl",
                "version", "tags", "owner", "enabled", "timeout_ms", "doc", "examples"}},
        ))
    return out
