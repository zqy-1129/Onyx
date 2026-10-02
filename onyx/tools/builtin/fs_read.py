"""只读文件工具：路径穿越必须被挡住。

参数来自**模型输出**，所以 `path` 是不可信输入：`../../.ssh/id_rsa`、
绝对路径、符号链接指到 root 外面，这三种都必须拒绝。

关键设计：允许根目录 `root` **不由参数传入**，而由 `ToolDef.extra.constants`
在执行器里注入（见 python_fn 执行器）。若 root 能被参数覆盖，
模型只要多写一个字段就能把整台机器读穿——白名单就成了摆设。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from onyx.core.errors import ToolArgError, ToolSandboxDenied

MAX_BYTES = 262_144
#: 拒绝读取的扩展名：二进制文件读出来是噪声，还会白白吃掉上下文
BLOCKED_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".mp3", ".mp4", ".avi", ".mov", ".wav", ".zip", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".bin", ".pdf", ".sqlite", ".db",
})


def read_file(
    path: str,
    root: str,
    max_bytes: int = MAX_BYTES,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    """读取 `root` 下的一个文本文件。

    `root` 是可信注入参数（不在 schema 的 properties 里），模型无法覆盖它。
    """
    if not isinstance(path, str) or not path.strip():
        raise ToolArgError("path 必须是非空字符串", detail={"kind": "bad_path"})
    if not isinstance(root, str) or not root.strip():
        # root 缺失是**部署配置**问题，不是模型的问题：报 rejected 而不是 arg_error
        raise ToolSandboxDenied(
            "fs_read 没有配置 extra.constants.root，拒绝读取任何路径",
            detail={"kind": "root_not_configured"},
        )
    if Path(path).suffix.lower() in BLOCKED_SUFFIXES:
        raise ToolArgError(
            f"拒绝读取二进制文件（{Path(path).suffix}）", detail={"kind": "binary_file"}
        )

    base = Path(root).resolve()
    candidate = Path(path).expanduser()
    target = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    # resolve() 会展开符号链接，所以这一条同时挡住了 `link -> /etc` 这类绕过
    if not target.is_relative_to(base):
        raise ToolSandboxDenied(
            f"路径 {path!r} 越出允许根目录 {str(base)!r}",
            detail={"kind": "path_traversal", "root": str(base), "resolved": str(target)},
        )
    if not target.is_file():
        raise ToolArgError(
            f"文件不存在或不是普通文件: {target.relative_to(base).as_posix()}",
            detail={"kind": "not_a_file", "path": target.relative_to(base).as_posix()},
        )

    raw = target.read_bytes()
    truncated = len(raw) > max_bytes
    text = raw[:max_bytes].decode(encoding, "replace")
    return {
        "path": target.relative_to(base).as_posix(),
        "bytes": len(raw),
        "returned_bytes": min(len(raw), max_bytes),
        "truncated": truncated,
        "lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
        "text": text,
    }
