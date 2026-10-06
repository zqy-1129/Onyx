"""Tool Bench 的读端点（S25）：注册表 / 契约审计 / 上下文开销 / 契约矩阵 / 运行历史。

**全部与 CLI 同源**：注册表装配走 `runtime.build_tool_registry`，契约矩阵走
`tools.matrix.build_matrix`，审计与开销直接调 registry 的方法。看板自己算一套的话，
"工具库上下文开销"就会有两个版本——而它正是要拿去和 trace 归因的 `part=tool_defs`
对齐的那个量（P17：只报 JSON 大小会把优化方向引错）。

`GET /api/tools/matrix` 是 GET 但**不该轮询**：它会真跑一遍各执行器
（python_fn 走 AST 白名单、http 用 MockTransport、mcp 用假连接、
`mcp_stdio` 起一个真子进程走真管道，S35），全程零真实网络。
第五列要起进程，所以整张矩阵是**秒级**而不是几十毫秒——正因为如此它才只该由人点一下，
而不是每次打开页面就替人跑一次。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from onyx.api.deps import AppState, get_state
from onyx.core.errors import ToolUnknown
from onyx.store.repos import ToolRepo
from onyx.tools.matrix import build_matrix

router = APIRouter(prefix="/api/tools", tags=["tools"])

#: 一次矩阵请求最多允许跑几条断言之类的事——矩阵本身是固定 8 断言 × 5 列，
#: 这里的上限防的是"参数 JSON 里塞个巨型对象"
MAX_ARGS_CHARS = 8_000


class ToolDefView(BaseModel):
    name: str
    version: int
    kind: str
    side_effect: str
    #: None = 还没核算过（`refresh_costs` 没跑过），不是 0 token
    tokens: int | None = None
    bytes: int | None = None
    enabled: bool
    hash: str
    impl_ref: str | None = None
    timeout_ms: int | None = None
    description: str = ""
    tags: list[str] = []
    n_examples: int = 0


class FindingView(BaseModel):
    tool: str
    rule: str
    severity: str
    message: str
    fix: str = ""
    path: str = ""
    meaning: str = ""


class AuditView(BaseModel):
    findings: list[FindingView]
    counts: dict[str, int]
    #: 审计是"按当前注册表算出来的"，不是历史快照：界面要显示这句话，
    #: 否则人会把两次刷新看到的差异读成"工具坏了"
    computed_at: str = ""
    note: str = ""


class RunView(BaseModel):
    id: str
    tool_id: str | None
    tool_def_hash: str | None
    test_id: str | None
    trace_id: str | None
    status: str
    started_at: str
    latency_ms: float | None = None
    error: str | None = None
    deterministic: bool | None = None
    idempotent: bool | None = None
    #: 结果 payload 的引用（内容寻址）。没引就不显示，而不是显示空串
    output_ref: str | None = None


def _registry(state: AppState, model: str | None) -> Any:
    from onyx.runtime import build_tool_registry

    return build_tool_registry(
        state.runtime.db, provider_id=state.runtime.provider.id, model=model
    )


def _def_view(record: Any) -> ToolDefView:
    return ToolDefView(
        name=record.name, version=record.version, kind=record.kind,
        side_effect=record.side_effect, tokens=record.tokens, bytes=record.bytes,
        enabled=bool(record.enabled), hash=record.hash, impl_ref=record.impl_ref,
        timeout_ms=record.timeout_ms, description=record.description or "",
        tags=list(record.tags or []), n_examples=len(record.examples or ()),
    )


@router.get("", response_model=list[ToolDefView])
def list_tools(
    enabled_only: bool = Query(default=False),
    kind: str | None = Query(default=None),
    state: AppState = Depends(get_state),
) -> list[ToolDefView]:
    """注册表里的工具定义（含历史版本行）。"""
    records = ToolRepo(state.runtime.db).list_defs(enabled_only=enabled_only, kind=kind)
    return [_def_view(item) for item in records]


@router.get("/audit", response_model=AuditView)
def audit_tools(
    model: str | None = Query(default=None, description="用该模型的标定档位核算描述预算"),
    state: AppState = Depends(get_state),
) -> AuditView:
    """契约审计：逐条规则列出问题与修法。规则文案与 CLI 共用 `tools.spec.RULES`。"""
    from onyx.core.clock import utc_now_iso
    from onyx.tools.spec import RULES

    results = _registry(state, model).audit_all()
    findings = [
        FindingView(
            tool=name, rule=finding.rule, severity=str(finding.severity),
            message=finding.message, fix=finding.fix, path=finding.path,
            meaning=RULES.get(finding.rule, ""),
        )
        for name, items in results.items() for finding in items
    ]
    counts = {"error": 0, "warn": 0, "info": 0}
    for item in findings:
        counts[item.severity] = counts.get(item.severity, 0) + 1
    return AuditView(
        findings=findings, counts=counts, computed_at=utc_now_iso(),
        note="按当前注册表现算，不是历史快照",
    )


@router.get("/cost", response_model=dict)
def tool_cost(
    model: str | None = Query(default=None, help="按该模型的计数档位核算（推荐）"),
    overhead: int = Query(default=0, ge=0, description="模板脚手架开销，取 trace 归因的 template_ctl"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """工具库上下文开销：JSON 本身 + 模板脚手架。

    P17：只报 JSON 会把优化方向引到"精简描述"，而实测 73% 的开销来自模板注入的说明文本。
    所以 `template_overhead` 要单独传进来，`template_share` 要单独显示。
    """
    report = _registry(state, model).cost(template_overhead=overhead)
    report["model"] = model or ""
    report["hint"] = (
        "未指定模型 ⇒ heuristic 档，绝对值仅供比较；带上已标定的模型名可得标定后的数字"
        if report["count_source"] == "heuristic" else ""
    )
    return report


@router.get("/matrix", response_model=dict)
def contract_matrix(
    tool: str = Query(default="echo", description="python_fn/mock 两列用的样本工具"),
    args: str | None = Query(default=None, description="JSON 对象，覆盖自动构造的合法参数"),
    state: AppState = Depends(get_state),
) -> dict[str, Any]:
    """执行器契约矩阵。与 `onyx tools contract --json` 返回同一个形状。"""
    import json as _json

    registry = _registry(state, None)
    definition = registry.get(tool)
    # 出处要报对：注册表里覆盖的同名 echo 必须用它自己的 examples，
    # 否则矩阵测的是内置定义、界面说的是注册版本
    source = "registry"
    if definition is None:
        from onyx.tools.builtin.defs import builtin_def

        definition, source = builtin_def(tool), "builtin"
    given: dict[str, Any] | None = None
    if args:
        if len(args) > MAX_ARGS_CHARS:
            raise HTTPException(status_code=413, detail=f"args 太长（>{MAX_ARGS_CHARS} 字符）")
        try:
            given = _json.loads(args)
        except _json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"args 不是合法 JSON: {exc}") from None
        if not isinstance(given, dict):
            raise HTTPException(status_code=422, detail="args 必须是一个 JSON 对象")

    try:
        matrix = build_matrix(definition, source=source, valid_args=given, tool_label=tool)
    except ToolUnknown as exc:
        raise HTTPException(status_code=404, detail=exc.message) from None
    return matrix.as_dict()


@router.get("/runs", response_model=list[RunView])
def list_runs(
    tool_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    state: AppState = Depends(get_state),
) -> list[RunView]:
    """`tools run` / `tools fire` 的留痕。每条都带着它那次 trace_id，能点回真实请求。"""
    records = ToolRepo(state.runtime.db).list_runs(tool_id=tool_id, limit=limit)
    return [
        RunView(
            id=item.id, tool_id=item.tool_id, tool_def_hash=item.tool_def_hash,
            test_id=item.test_id, trace_id=item.trace_id, status=item.status,
            started_at=item.started_at, latency_ms=item.latency_ms, error=item.error,
            deterministic=item.deterministic, idempotent=item.idempotent,
            output_ref=item.output_ref,
        )
        for item in records
    ]
