"""一个最小但**真实**的 MCP server（stdio + JSON-RPC 2.0，只用 stdlib）。

它存在的目的是让 `tests/integration/test_mcp_stdio.py` 验证"过 OS 管道"这件事本身：
伪造传输能测协议语义，但测不到**帧**（换行分隔、缓冲区、stderr 管道满时死锁）。

它刻意模仿真实 server 的三种坏毛病，每一种都会被断言：
- 往 stdout 打日志（非 JSON 行必须被跳过，而不是当成回答）；
- 往 stderr 打很多行（客户端不持续排空就会在管道满时死锁，现象是"卡住"而不是报错）；
- 有一个慢工具和一个 `isError` 工具（超时与失败的归因要走对档位）。
"""

from __future__ import annotations

import contextlib
import json
import sys
import time

TOOLS = [
    {
        "name": "weather",
        "title": "天气",
        "description": "查询指定城市的当前天气",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "send_email",
        "description": "发送一封邮件",
        "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}},
                        "required": ["to"]},
        # 故意不标注 readOnlyHint：客户端必须把它当成有副作用
    },
    {
        "name": "slow",
        "description": "永远比 deadline 慢一点的工具",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "noisy",
        "description": "先灌 2000 行 stderr 再回答",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "boom",
        "description": "工具侧失败",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True},
    },
]


def reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def answer(rid, result: dict) -> None:
    reply({"jsonrpc": "2.0", "id": rid, "result": result})


def main() -> int:
    # MCP 规定 JSON 载荷是 UTF-8。中文 Windows 上 Python 的 stdio 默认跟随控制台
    # 代码页（GBK），不显式改就会写出父进程解不了的字节——所以"合规"要自己声明。
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError):
            stream.reconfigure(encoding="utf-8")
    sys.stderr.write("demo server starting\n")
    sys.stderr.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        rid = message.get("id")
        if rid is None:                     # 通知：不需要回答
            continue

        if method == "initialize":
            sys.stdout.write("[log] 一行不属于协议的文字\n")   # 真实 server 常这么干
            sys.stdout.flush()
            answer(rid, {
                "protocolVersion": "2025-06-18",
                "serverInfo": {"name": "onyx-demo-mcp", "version": "0.1.0"},
                "capabilities": {"tools": {"listChanged": False}},
            })
        elif method == "tools/list":
            answer(rid, {"tools": TOOLS})
        elif method == "tools/call":
            name = (message.get("params") or {}).get("name")
            args = (message.get("params") or {}).get("arguments") or {}
            if name == "weather":
                answer(rid, {"content": [{"type": "text",
                                          "text": f"{args.get('city')}：晴，21 度"}]})
            elif name == "send_email":
                answer(rid, {"content": [{"type": "text",
                                          "text": f"已发给 {args.get('to')}"}]})
            elif name == "slow":
                time.sleep(2.0)
                answer(rid, {"content": [{"type": "text", "text": "太晚了"}]})
            elif name == "noisy":
                for i in range(1200):       # 不排空 stderr 就会在这里死锁
                    sys.stderr.write(f"log line {i}\n")
                sys.stderr.flush()
                answer(rid, {"content": [{"type": "text", "text": "灌完了"}]})
            elif name == "boom":
                answer(rid, {"content": [{"type": "text", "text": "SMTP 550 relay denied"}],
                             "isError": True})
            else:
                reply({"jsonrpc": "2.0", "id": rid,
                       "error": {"code": -32601, "message": f"unknown tool {name}"}})
        elif method == "ping":
            answer(rid, {})
        else:
            reply({"jsonrpc": "2.0", "id": rid,
                   "error": {"code": -32601, "message": f"method not found: {method}"}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
