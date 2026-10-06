"""MCP 执行器：`ToolKind.MCP` 的实现，走同一条 `guarded_call` 管线。

参数校验、沙箱策略、mock 短路、deadline、错误归一**都由 `executor.guarded_call` 做**，
这里只负责"怎么把一次调用打到 server 上"。这是本项目的一条硬规矩：
这些保证只需要在一处成立，各自实现一遍必然漏。

评测期默认 `mock_policy=fixture` ⇒ 永远不会起子进程：
定义照旧发给模型（测的就是"模型能不能选对这个工具、写对这个参数"），
执行换成桩。真起进程会让评测结果取决于某个第三方 server 当时的心情，
那种分数不可复现，也不该被报出来。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from onyx.core.errors import ToolTimeout, ToolUnknown
from onyx.tools.executor import ToolCtx, guarded_call
from onyx.tools.mcp import McpPool, load_servers_file, payload_from_mcp
from onyx.tools.sandbox import check_impl_ref
from onyx.tools.spec import ToolDef, ToolKind, ToolResult

__all__ = [
    "ENV_CONFIG",
    "McpExecutor",
    "config_path",
    "contract_target",
    "default_pool",
    "parse_impl_ref",
    "reset_pools",
    "stdio_contract_target",
]

ENV_CONFIG = "ONYX_MCP_CONFIG"

_POOLS: dict[str, McpPool] = {}
_POOL_LOCK = threading.Lock()


def config_path() -> Path:
    """MCP 配置文件位置：`ONYX_MCP_CONFIG` 优先，否则 `<data_dir>/mcp.json`。"""
    env = os.environ.get(ENV_CONFIG)
    if env:
        return Path(env)
    from onyx.settings import load_settings

    return load_settings().data_dir / "mcp.json"


def default_pool(*, path: str | os.PathLike[str] | None = None) -> McpPool:
    """按配置文件拿（并复用）一个连接池。按路径缓存：一个 server 一个子进程。"""
    resolved = str(path or config_path())
    with _POOL_LOCK:
        cached = _POOLS.get(resolved)
        if cached is not None:
            return cached
        if not Path(resolved).exists():
            raise ToolUnknown(
                f"没有 MCP 配置文件: {resolved}",
                detail={
                    "kind": "missing_mcp_config",
                    "env": ENV_CONFIG,
                    "hint": "写 {\"mcpServers\": {\"名字\": {\"command\": [\"python\", \"server.py\"]}}} "
                            "或用 --config / ONYX_MCP_CONFIG 指定",
                },
            )
        pool = McpPool(load_servers_file(resolved))
        _POOLS[resolved] = pool
        return pool


def reset_pools() -> None:
    """关掉并丢弃所有缓存的池（测试与进程退出用）。"""
    with _POOL_LOCK:
        for pool in _POOLS.values():
            pool.close()
        _POOLS.clear()


def parse_impl_ref(impl_ref: str) -> tuple[str, str]:
    """`mcp:<server>:<tool>` → (server, tool)。

    只接受这个形状：工具定义可以被导入/改写，如果这里允许任意 argv，
    一份被篡改的定义就等于让模型决定跑什么命令。
    """
    parts = (impl_ref or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "mcp" or not parts[1] or not parts[2]:
        raise ToolUnknown(
            f"MCP 的 impl_ref 必须是 'mcp:<server>:<tool>'，收到 {impl_ref!r}",
            detail={"kind": "bad_impl_ref", "impl_ref": impl_ref},
        )
    return parts[1], parts[2]


class McpExecutor:
    """把一次工具调用打到 MCP server 上。"""

    kind = ToolKind.MCP
    #: 隔离证据类型：`real_calls` 是真打到 server 上的次数，评测期必须是 0
    ISOLATION_PROOF = "counter"

    def __init__(
        self,
        definition: ToolDef,
        *,
        pool: McpPool | None = None,
        session: Any | None = None,
        config_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self.definition = definition
        self._pool = pool
        self._session = session
        self._config_path = config_path
        self.real_calls = 0
        self.started_servers = 0
        #: 最近一次真正用到的 `(server, session)`。隔离必须按**身份**丢弃：
        #: 超时是在被放弃的工作线程里发现的，等它来关时主线程可能已建好新会话。
        self._last: tuple[str, Any] | None = None

    def spec(self) -> ToolDef:
        return self.definition

    def call(self, name: str, args: dict[str, Any], ctx: ToolCtx) -> ToolResult:
        if name != self.definition.name:
            # 路由错：这个执行器绑的是哪份定义是确定的，名字对不上就不是它的活
            return ToolResult(
                ok=False, error_kind="unknown_tool",
                error=f"MCP 执行器绑定的是 {self.definition.name!r}，收到 {name!r}",
            )
        result = guarded_call(self.definition, args, ctx,
                              lambda clean: self._invoke(clean, ctx), name=name)
        if result.error_kind == "timeout":
            # 超时之后这条连接就不可以再用了：那个被放弃的线程还在读管道，会把下一笔
            # 调用的回答读走。在这里（而不是 `_invoke` 的 except 里）隔离才靠得住——
            # `run_with_deadline` 先于内部超时报错时，`_invoke` 那条路径根本不会执行。
            self._quarantine()
        return result

    # ── 实际调用 ──────────────────────────────────────────────────
    def _pool_or_default(self) -> McpPool:
        if self._pool is None:
            self._pool = default_pool(path=self._config_path)
        return self._pool

    def _invoke(self, clean: dict[str, Any], ctx: ToolCtx) -> Any:
        # impl_ref 白名单：工具定义是从文件导入、可被改写的，`mcp:<server>:<tool>`
        # 里的 server 必须来自人工审核过的配置。没有这道闸，一份坏定义就等于
        # 让模型决定跑哪个命令（与 python_fn 的 check_impl_ref 同一条理由）。
        check_impl_ref(self.definition.impl_ref, ctx.policy)
        server, tool = parse_impl_ref(self.definition.impl_ref)

        session: Any = self._session
        if session is None:
            pool = self._pool_or_default()
            before = len(pool.started)
            session = pool.session(server)
            self.started_servers += len(pool.started) - before
        else:
            # 注入的 session 不经池 ⇒ 永远没有子进程被起（离线契约测试就靠这条）
            session.started_processes = 0
        self._last = (server, session)

        budget = session.spec.timeout_s
        for candidate in (
            (ctx.deadline_ms / 1000) if ctx.deadline_ms else None,
            (self.definition.timeout_ms / 1000) if self.definition.timeout_ms else None,
        ):
            if candidate is not None:
                budget = min(budget, candidate)
        if budget <= 0:
            raise ToolTimeout(
                f"MCP 调用 {self.definition.name} 的 deadline 已经用尽",
                detail={"kind": "deadline_exhausted", "budget_s": budget},
            )

        self.real_calls += 1
        raw = session.call_tool(tool, clean, timeout=budget)
        return payload_from_mcp(raw)

    # ── 隔离 ──────────────────────────────────────────────────────
    def _quarantine(self) -> None:
        """超时的调用不许留下可复用的连接：当场丢掉那条会话（→ 子进程被回收）。

        不隔离的后果是**污染下一个调用**：Python 杀不掉线程，被放弃的那笔仍在读管道，
        它会把下一笔的回答读进一个没人看的地方。真子进程契约列第一次跑就是这样——
        报的是"weather 返回了空"，而罪在上一笔 slow 调用，最难归因的一类。
        """
        if self._session is not None or self._last is None:
            return          # 注入 session 的离线列没有池，也没什么可隔离
        server, session = self._last
        self._pool_or_default().discard(server, only_if_is=session)


# ── 离线契约样本 ─────────────────────────────────────────────────
class _ContractConnection:
    """假连接：按最后一次请求的 id 回答，慢工具永远不回答。

    契约矩阵必须能在**不起任何子进程、不碰任何真实 server**的前提下跑完，
    否则 `onyx tools contract` 会变成一条需要外部依赖的命令——那没人会经常跑。
    """

    def __init__(self, *, slow: bool = False) -> None:
        self.slow = slow
        self.sent: list[dict] = []
        self.closed = False

    def write(self, payload: dict) -> None:
        self.sent.append(payload)

    def read(self, timeout: float) -> dict | None:
        if self.slow:
            raise ToolTimeout("contract: server 不回答", detail={"kind": "contract_slow"})
        rid = self.sent[-1].get("id") if self.sent else None
        return {"jsonrpc": "2.0", "id": rid,
                "result": {"content": [{"type": "text", "text": "pong"}]}}

    def close(self) -> None:
        self.closed = True


def contract_target():
    """`onyx tools contract` 的 mcp 列：(样本定义, 工厂, 合法参数, 合成器)。"""
    from onyx.tools.contract import text_schema
    from onyx.tools.mcp import McpSession, ServerSpec
    from onyx.tools.spec import SideEffect

    spec = ServerSpec(name="contract", command=("unused",), timeout_s=5.0)

    def session_for(tool: str) -> McpSession:
        connection = _ContractConnection(slow=tool.endswith("slow"))
        session = McpSession(spec, transport=connection)
        session.protocol_version = "2025-06-18"
        return session

    def make_def(name: str, impl_tool: str, effect: SideEffect) -> ToolDef:
        return ToolDef(
            name=name, description="契约测试用的 MCP 工具，验证同一套断言在每个执行器上都成立",
            parameters=text_schema(), kind=ToolKind.MCP, side_effect=effect,
            impl_ref=f"mcp:contract:{impl_tool}",
            extra={"mcp_server": "contract", "mcp_tool": impl_tool},
        )

    sample = make_def("contract_mcp", "echo", SideEffect.READ)

    def factory(definition: ToolDef) -> McpExecutor:
        impl_tool = definition.impl_ref.rsplit(":", 1)[-1]
        return McpExecutor(definition, session=session_for(impl_tool))

    def synth(name: str, role: str) -> ToolDef:
        if role == "slow":
            return make_def(name, "slow", SideEffect.READ)
        if role == "write":
            return make_def(name, "write", SideEffect.WRITE)
        raise ValueError(f"未知的合成角色: {role!r}（可选 slow / write）")

    return sample, factory, {"text": "hello"}, synth


# ── 真子进程契约样本（S35：矩阵的第五列）───────────────────────────
def stdio_contract_target():
    """`onyx tools contract` 的 **mcp_stdio** 列：真子进程 + 真 OS 管道。

    与上面那个离线列的分工是刻意的：假连接测"我们对协议的理解"，
    真进程测**帧本身**——换行分隔、stdout 上的非 JSON 日志行、stderr 排空、进程回收。
    换 MCP 协议版本、换 server 实现、或 server 开始乱打 stdout，只有这一列会红；
    而那正是 G6 要的保护（"契约矩阵 4 列但 mcp 用假连接"在 ROADMAP 里被记成缺口）。

    返回 `(样本定义, 工厂, 合法参数, 合成器, teardown)`。
    样本与两个合成角色都来自**真发现**（`tools/list`），不是手写定义——
    手写的那份与 server 漂移时，矩阵会通过一个现实中不存在的契约。
    起不来时照常抛错，由 matrix 记成 `unavailable`（未知），不是"通过"。
    """
    import sys

    from onyx.tools import reference_mcp_server
    from onyx.tools.contract import text_schema
    from onyx.tools.mcp import McpPool, ServerSpec, tooldefs_from_mcp
    from onyx.tools.spec import SideEffect

    pool = McpPool({"reference": ServerSpec(
        name="reference",
        command=(sys.executable, str(Path(reference_mcp_server.__file__).resolve())),
        env=("PATH",), timeout_s=10.0,
    )})
    try:
        discovered = {str(d.extra["mcp_tool"]): d for d in tooldefs_from_mcp(pool, "reference")}
        sample = discovered["weather"]
    except Exception:
        pool.close()
        raise

    def make_def(name: str, impl_tool: str, effect: SideEffect) -> ToolDef:
        #: 合成定义**只用于断言本身**（慢工具与写工具）：参数按契约要求的 `text` 声明，
        #: 因为断言固定传 `{"text": ...}`。用 server 的真实 schema 反而会让断言在
        #: "缺少必填参数"这一层就失败，测不到它要测的超时与策略拒绝。
        #: 样本定义（weather）仍是真发现来的那份——它带的是 server 自己的 schema。
        return ToolDef(
            name=name, description="契约矩阵的真 stdio 列：验证超时与副作用拒绝",
            parameters=text_schema(), kind=ToolKind.MCP, side_effect=effect,
            impl_ref=f"mcp:reference:{impl_tool}",
            extra={"mcp_server": "reference", "mcp_tool": impl_tool},
        )

    def factory(definition: ToolDef) -> McpExecutor:
        return McpExecutor(definition, pool=pool)

    def synth(name: str, role: str) -> ToolDef:
        if role == "slow":
            return make_def(name, "slow", SideEffect.READ)
        if role == "write":
            return make_def(name, "send_email", SideEffect.WRITE)
        raise ValueError(f"未知的合成角色: {role!r}（可选 slow / write）")

    return sample, factory, {"city": "北京"}, synth, pool.close
