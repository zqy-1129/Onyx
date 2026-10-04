"""MCP（Model Context Protocol）客户端：stdio 传输 + JSON-RPC 2.0，只用 stdlib。

为什么手写而不是装 `mcp` SDK：`onyx.tools` 的可移植性锚点是"除 executors.http 外不依赖
三方网络库"，而 stdio 传输本质上是"往子进程的 stdin 写一行 JSON、从 stdout 读一行 JSON"。
用 stdlib 换来两件事：`import-linter` 的网络契约继续成立（MCP 不经过 httpx），
以及测试可以在**不起任何网络**的情况下穷举协议边界。

四条必须守住的安全/正确性约束：

1. **服务器命令来自配置文件，永远不来自模型。** 参数只能进 `tools/call` 的 `arguments`。
   这跟"不提供通用 fetch 工具"是同一条理由：一旦模型能决定跑什么命令，
   工具层就变成 RCE。`impl_ref` 还额外要过 `SandboxPolicy.allowed_impl_prefixes`。
2. **副作用默认取保守值。** MCP 的 `annotations.readOnlyHint` 是服务器**自报的提示**，
   不是保证；没标注一律按 `write` 处理（→ 默认策略直接拒绝，需显式放开）。
   "未标注就当成只读"是审计错误，而不是便利。
3. **stderr 不参与协议。** 很多 MCP server 把日志写到 stderr；把它当协议读会死锁
   （管道满），把它丢掉会丢诊断。这里持续排空 stderr 并只保留尾部若干行。
4. **超时要真的能退出。** `run_with_deadline` 到点只放弃等待，线程杀不掉；
   所以读循环本身带超时，并且超时后**立刻关掉子进程**，否则僵尸 server 会攒成句柄泄漏。

协议版本：优先 `2025-06-18`，服务器可以回它支持的版本；不支持的版本不做兼容猜测，
直接报错并说明——"能跑但语义不同"比"跑不了"更难查。
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from onyx.core.errors import (
    ToolRuntime,
    ToolTimeout,
    ToolUnknown,
)
from onyx.tools.spec import SideEffect, ToolDef, ToolKind

PROTOCOL_VERSION = "2025-06-18"
#: 只接受这一个版本：服务器回一个不同版本时不假装兼容
SUPPORTED_PROTOCOL_VERSIONS = frozenset({PROTOCOL_VERSION, "2024-11-05"})

DEFAULT_CALL_TIMEOUT_S = 30.0
#: 输出截断上限。工具输出会进上下文，一个 5MB 的 JSON 能直接把窗口吃光
MAX_OUTPUT_CHARS = 65_536
#: stderr 只保留尾部这些行，用于诊断
STDERR_TAIL_LINES = 40


class McpError(RuntimeError):
    """协议层错误。带 `code`（JSON-RPC error code）便于归因。"""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


# ── 配置 ──────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class ServerSpec:
    """一个 MCP server 的启动方式。全部字段都来自配置文件（人工审核）。"""

    name: str
    command: tuple[str, ...]
    args: tuple[str, ...] = ()
    env: tuple[str, ...] = ()          # 只转发**列出的**环境变量名，不整份继承
    cwd: str = ""
    timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    client_name: str = "onyx"

    @property
    def argv(self) -> tuple[str, ...]:
        return self.command + self.args

    @property
    def impl_prefix(self) -> str:
        return f"mcp:{self.name}:"


def load_servers(raw: Mapping[str, Any]) -> dict[str, ServerSpec]:
    """配置文件 → `ServerSpec` 表。

    校验很严，因为这里的每个字段最后都会变成 `Popen(argv)`：
    空命令、字符串命令（会被当成 shell 命令行）、空名字都直接拒绝。
    """
    out: dict[str, ServerSpec] = {}
    servers = raw.get("mcpServers") if isinstance(raw.get("mcpServers"), Mapping) else raw
    for name, item in (servers or {}).items():
        if not isinstance(item, Mapping):
            raise ValueError(f"mcp server {name!r} 的配置不是对象")
        command = item.get("command")
        if isinstance(command, str):
            command = [command]
        command = tuple(str(c) for c in (command or ()) if str(c))
        if not command:
            raise ValueError(f"mcp server {name!r} 缺少 command（必须是 argv 列表，不是 shell 字符串）")
        args = tuple(str(a) for a in (item.get("args") or ()))
        env = tuple(str(e) for e in (item.get("env") or ()))
        raw_timeout = item.get("timeout_s")
        # 0 / 负数要直接拒绝。写成 `or DEFAULT` 会把 0 悄悄变成默认值，
        # 而"我以为超时是 0（不超时）"与实际 30s 的行为差别，只有出事那天才看得见
        timeout = DEFAULT_CALL_TIMEOUT_S if raw_timeout is None else float(raw_timeout)
        if timeout <= 0:
            raise ValueError(f"mcp server {name!r} 的 timeout_s 必须 > 0（收到 {raw_timeout}）")
        out[str(name)] = ServerSpec(
            name=str(name), command=command, args=args, env=env,
            cwd=str(item.get("cwd") or ""), timeout_s=timeout,
            client_name=str(item.get("client_name") or "onyx"),
        )
    if not out:
        raise ValueError("MCP 配置里没有 server：需要 {\"mcpServers\": {名字: {command: [...]}}}")
    return out


def load_servers_file(path: str | os.PathLike[str]) -> dict[str, ServerSpec]:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    try:
        return load_servers(json.loads(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"MCP 配置不是合法 JSON: {path}（行 {exc.lineno}）") from exc


# ── 传输 ──────────────────────────────────────────────────────────
def decode_line(raw: bytes) -> str:
    """按 UTF-8 解码一行，坏字节替换而不是抛异常。

    这不是宽容，是止血：真机第一次跑就撞上"中文 Windows 上子进程的 stdout 是 GBK"，
    `readline()` 在 text 模式里抛 UnicodeDecodeError，**读线程当场死掉**，
    于是父进程再也等不到回答——现象是"卡到超时"而不是"编码错了"，最难归因的那一类。
    协议规定 JSON 载荷是 UTF-8；不守规矩的 server 我们照样不会崩，
    解不出 JSON 的行按非协议行跳过。
    """
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


class StdioTransport:
    """子进程 + 换行分隔的 JSON。**用二进制管道自己解码**，理由见 `decode_line`。"""

    def __init__(self, spec: ServerSpec) -> None:
        self.spec = spec
        env = {key: os.environ[key] for key in spec.env if key in os.environ}
        # 刻意**不**整份继承环境：MCP server 通常不需要看到 onyx 进程里的密钥
        if env:
            env["PATH"] = os.environ.get("PATH", "")
        self.process = subprocess.Popen(
            list(spec.argv),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=spec.cwd or None, env=env or None, bufsize=0,
        )
        self.stderr_tail: list[str] = []
        self._drain_stderr()

    def _drain_stderr(self) -> None:
        """持续排空 stderr。不排空会在管道满时死锁——现象是 server 卡住而不是报错。"""

        def pump() -> None:
            handle = self.process.stderr
            if handle is None:
                return
            try:
                for raw in handle:
                    self.stderr_tail.append(decode_line(raw))
                    del self.stderr_tail[:-STDERR_TAIL_LINES]
            except (ValueError, OSError):
                # close() 会关掉管道，正在迭代的线程随之抛"I/O operation on closed file"。
                # 不接住它，pytest 就会报"未处理的线程异常"——一个正常的关闭
                # 看起来像一次故障，久而久之人就习惯忽略这类告警了
                return

        threading.Thread(target=pump, daemon=True, name=f"mcp-stderr-{self.spec.name}").start()

    def write(self, payload: dict[str, Any]) -> None:
        if self.process.poll() is not None:
            raise ToolRuntime(
                f"MCP server {self.spec.name} 已退出（code={self.process.returncode}）",
                detail={"stderr": self.stderr_tail[-5:]},
            )
        stdin = self.process.stdin
        assert stdin is not None
        body = json.dumps(payload, ensure_ascii=False) + "\n"
        try:
            stdin.write(body.encode("utf-8"))
            stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            # 管道断了必须说清楚是哪条请求：否则上层只看到"超时"，
            # 而修法是"这个 server 起不来"，不是"给长一点的超时"
            raise ToolRuntime(
                f"写入 MCP server {self.spec.name} 的 stdin 失败：{type(exc).__name__}: {exc}",
                detail={"kind": "broken_pipe", "method": payload.get("method", ""),
                        "stderr": self.stderr_tail[-5:]},
            ) from exc

    def read(self, timeout: float) -> dict[str, Any] | None:
        """读一行 JSON，带超时。解析不出来的行返回 None（跳过并继续读）。"""
        box: queue.Queue[Any] = queue.Queue(maxsize=1)

        def worker() -> None:
            stdout = self.process.stdout
            if stdout is None:
                box.put(("eof", None))
                return
            try:
                box.put(("line", stdout.readline()))
            except Exception as exc:  # noqa: BLE001 - 读线程不能死，否则整个会话挂死
                box.put(("error", f"{type(exc).__name__}: {exc}"))

        threading.Thread(target=worker, daemon=True).start()
        try:
            kind, raw = box.get(timeout=timeout)
        except queue.Empty as exc:
            raise ToolTimeout(
                f"MCP server {self.spec.name} 在 {timeout:.1f}s 内没有响应",
                detail={"timeout_s": timeout, "stderr": self.stderr_tail[-5:]},
            ) from exc
        if kind == "error":
            raise ToolRuntime(
                f"读取 MCP server {self.spec.name} 的 stdout 失败: {raw}",
                detail={"kind": "stdout_read_error"},
            )
        if kind == "eof" or not raw:
            raise ToolRuntime(
                f"MCP server {self.spec.name} 关闭了 stdout（进程 code={self.process.poll()}）",
                detail={"stderr": self.stderr_tail[-5:]},
            )
        text = decode_line(raw).strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None  # 非协议行（某些 server 会往 stdout 打日志）：跳过而不是假装成功

    def close(self) -> None:
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            with contextlib.suppress(Exception):
                if stream is not None:
                    stream.close()
        if self.process.poll() is None:
            self.process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=2)
            if self.process.poll() is None:
                self.process.kill()


# ── 会话 ──────────────────────────────────────────────────────────
class McpSession:
    """一个 server 一条连接：握手、`tools/list`、`tools/call`。

    请求按 JSON-RPC id 配对，读循环在锁内串行——同一个子进程的 stdin 不能并发写。
    """

    def __init__(self, spec: ServerSpec, transport: Any | None = None) -> None:
        self.spec = spec
        self.transport = transport or StdioTransport(spec)
        self._next_id = 0
        self._lock = threading.Lock()
        self.server_info: dict[str, Any] = {}
        self.protocol_version = ""
        self.calls = 0

    def _request(self, method: str, params: dict[str, Any] | None = None,
                 *, timeout: float | None = None, expect_result: bool = True) -> Any:
        self._next_id += 1
        rid = self._next_id
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": rid, "method": method}
        if params is not None:
            message["params"] = params
        budget = timeout if timeout is not None else self.spec.timeout_s
        self.transport.write(message)
        deadline = time.monotonic() + budget
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ToolTimeout(f"MCP {method} 超时（{budget:.1f}s）",
                                  detail={"method": method})
            received = self.transport.read(remaining)
            if received is None:
                continue
            if received.get("method") and "id" not in received:
                continue  # 服务器发来的通知（notifications/*），不是对本次请求的回答
            if received.get("id") != rid:
                continue  # 迟到的回答：丢弃并继续找自己的那条
            if "error" in received:
                err = received.get("error") or {}
                raise McpError(
                    f"MCP {method} 返回错误: {err.get('message', err)}",
                    code=err.get("code"), data=err.get("data"),
                )
            return received.get("result") if expect_result else None

    def initialize(self, *, timeout: float | None = None) -> dict[str, Any]:
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": self.spec.client_name, "version": "0.1.0"},
            "capabilities": {},
        }, timeout=timeout) or {}
        version = str(result.get("protocolVersion") or "")
        if version not in SUPPORTED_PROTOCOL_VERSIONS:
            raise McpError(
                f"MCP server {self.spec.name} 报协议版本 {version!r}，onyx 只实现 "
                f"{sorted(SUPPORTED_PROTOCOL_VERSIONS)}；不做跨版本猜测",
                data={"server": result},
            )
        self.protocol_version = version
        self.server_info = dict(result.get("serverInfo") or {})
        # initialized 是通知（没有 id）：协议规定服务器收到后才会开始正常工作
        self.transport.write({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list") or {}
        tools = result.get("tools") or []
        if not isinstance(tools, list):
            raise McpError(f"MCP server {self.spec.name} 的 tools/list 返回了非列表")
        return [t for t in tools if isinstance(t, dict)]

    def call_tool(self, tool: str, arguments: dict[str, Any], *,
                  timeout: float | None = None) -> dict[str, Any]:
        self.calls += 1
        result = self._request("tools/call", {"name": tool, "arguments": arguments},
                               timeout=timeout)
        return dict(result or {})

    def close(self) -> None:
        close = getattr(self.transport, "close", None)
        if callable(close):
            close()


class McpPool:
    """按 server 名复用会话。一个 server 一个子进程——启动握手有几百毫秒，
    每次工具调用重开会让"工具开销"变成主要成本，而且失败更难归因。"""

    def __init__(self, servers: Mapping[str, ServerSpec]) -> None:
        self.servers = dict(servers)
        self._sessions: dict[str, McpSession] = {}
        self._lock = threading.Lock()
        self.started: list[str] = []      # 供测试断言"评测期一个进程都没起"

    def session(self, name: str) -> McpSession:
        spec = self.servers.get(name)
        if spec is None:
            raise ToolUnknown(
                f"未配置的 MCP server: {name!r}",
                detail={"known": sorted(self.servers),
                        "hint": "在 MCP 配置文件里加一节，或用 --config 指定文件"},
            )
        with self._lock:
            existing = self._sessions.get(name)
            if existing is not None:
                return existing
            session = McpSession(spec)
            try:
                session.initialize()
            except Exception:
                session.close()
                raise
            self._sessions[name] = session
            self.started.append(name)
            return session

    def close(self) -> None:
        with self._lock:
            for session in self._sessions.values():
                with contextlib.suppress(Exception):
                    session.close()
            self._sessions.clear()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured": sorted(self.servers),
                "running": sorted(self._sessions),
                "started": list(self.started),
                "calls": {k: v.calls for k, v in self._sessions.items()},
            }


# ── 发现 → ToolDef ────────────────────────────────────────────────
def _side_effect_of(tool: Mapping[str, Any]) -> tuple[SideEffect, str]:
    """MCP 的 annotations 是**提示**，不是保证。默认取保守值。"""
    annotations = tool.get("annotations") or {}
    if not isinstance(annotations, Mapping):
        return SideEffect.WRITE, "annotations 不是对象，按 write 处理"
    if annotations.get("destructiveHint") is True:
        return SideEffect.WRITE, "服务器自报 destructiveHint=true"
    if annotations.get("readOnlyHint") is True:
        return SideEffect.READ, "服务器自报 readOnlyHint=true"
    return SideEffect.WRITE, "服务器未标注 readOnlyHint，按最小权限原则当成有副作用"


def tooldefs_from_mcp(pool: McpPool, server: str, *, enabled: bool = True) -> list[ToolDef]:
    """发现一个 server 的工具并转成注册表定义。

    命名规则 `{server}__{tool}`：不同 server 会有同名工具（`fetch`、`search`），
    裸名注册会让两份定义互相覆盖，而"这个工具是谁提供的"恰好是排障时要问的第一个问题。
    """
    session = pool.session(server)
    out: list[ToolDef] = []
    for tool in session.list_tools():
        raw_name = str(tool.get("name") or "")
        if not raw_name:
            continue
        effect, reason = _side_effect_of(tool)
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        out.append(ToolDef(
            name=f"{server}__{raw_name}"[:64],
            description=str(tool.get("description") or ""),
            parameters=dict(schema) if isinstance(schema, Mapping) else {},
            kind=ToolKind.MCP,
            side_effect=effect,
            impl_ref=f"mcp:{server}:{raw_name}",
            version=str(tool.get("version") or "1"),
            enabled=enabled,
            doc=str(tool.get("title") or ""),
            extra={
                "mcp_server": server,
                "mcp_tool": raw_name,
                "side_effect_reason": reason,
                "annotations": dict(tool.get("annotations") or {})
                if isinstance(tool.get("annotations"), Mapping) else {},
            },
        ))
    if not out:
        raise McpError(f"MCP server {server!r} 没有报告任何工具")
    return out


# ── 结果映射 ──────────────────────────────────────────────────────
def _flatten_content(result: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """把 content blocks 摊平成文本；非文本块只留**摘要**。

    base64 图片绝不内联进文本：那会把几百 KB 的编码塞进模型上下文，
    而"上下文开销"正是这个看板要测量的东西。
    """
    blocks = result.get("content") or []
    texts: list[str] = []
    others: list[dict[str, Any]] = []
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, Mapping):
            others.append({"type": "unknown", "preview": str(block)[:120]})
            continue
        kind = str(block.get("type") or "")
        if kind == "text":
            texts.append(str(block.get("text") or ""))
        elif kind == "image":
            others.append({"type": "image", "mime": str(block.get("mimeType") or ""),
                           "bytes": len(str(block.get("data") or ""))})
        elif kind == "resource":
            resource = block.get("resource") or {}
            others.append({"type": "resource", "uri": str(resource.get("uri") or ""),
                           "text_chars": len(str(resource.get("text") or ""))})
        else:
            others.append({"type": kind or "unknown",
                           "preview": json.dumps(dict(block), ensure_ascii=False)[:200]})
    text = "\n".join(t for t in texts if t)
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + f"\n…（截断，原长 {len(text)} 字符）"
    return text, others


def payload_from_mcp(raw: Mapping[str, Any]) -> Any:
    """`tools/call` 的回答 → 工具输出。

    - `isError=true` 抛 `ToolRuntime` ⇒ `guarded_call` 会归到 `error` 档。
      把它记成 `arg_error` 会把 server 实现的 bug 算到模型头上，
      而两者的修法完全相反（改 server vs 改工具描述/提示词）。
    - base64 图片只留字节数，绝不内联：那会把几百 KB 编码塞进模型上下文，
      而"上下文开销"正是这个看板要测量的东西。
    """
    text, others = _flatten_content(raw)
    structured = raw.get("structuredContent")
    if raw.get("isError"):
        raise ToolRuntime(
            (text or f"MCP server 报告 isError=true（{json.dumps(others, ensure_ascii=False)[:200]}）")
            [:500],
            detail={"kind": "mcp_tool_error", "non_text_blocks": len(others),
                    "blocks": others[:5]},
        )
    if isinstance(structured, (Mapping, list)):
        return dict(structured) if isinstance(structured, Mapping) else structured
    if text:
        return text
    if others:
        return {"blocks": others}
    return ""


__all__ = [
    "PROTOCOL_VERSION",
    "McpError",
    "McpPool",
    "McpSession",
    "ServerSpec",
    "StdioTransport",
    "load_servers",
    "load_servers_file",
    "payload_from_mcp",
    "tooldefs_from_mcp",
]
