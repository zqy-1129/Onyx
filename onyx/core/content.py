"""内容寻址 blob 存储（原则 3：原始证据不可变）。

存什么：原始 HTTP body、渲染后的 prompt、模型原文（含畸形 JSON）、工具返回值、图片。
为什么内容寻址：同一个长 system prompt 只落一份盘；引用即校验（sha256）；
重放请求时能证明"用的就是当时那份原文"。

安全：ref 必须严格匹配 `sha256:<64 hex>` 才参与路径拼接，杜绝目录穿越。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import BlobNotFound, InvalidBlobRef

_REF_RE = re.compile(r"^sha256:([0-9a-f]{64})$")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True, slots=True)
class BlobStat:
    ref: str
    size: int
    media: str
    path: Path


@runtime_checkable
class BlobStore(Protocol):
    """可替换：本地目录 / 只读快照 / 对象存储都实现这四个方法即可。"""

    def put(self, data: bytes | str, *, media: str = "application/octet-stream") -> str: ...
    def get(self, ref: str) -> bytes: ...
    def exists(self, ref: str) -> bool: ...
    def stat(self, ref: str) -> BlobStat: ...


class FileBlobStore:
    """两级分片目录：blobs/ab/cd/<full-hex>，避免单目录百万文件。"""

    def __init__(self, root: Path | str, *, shard_depth: int = 2) -> None:
        self.root = Path(root)
        self.shard_depth = shard_depth
        self.root.mkdir(parents=True, exist_ok=True)

    # ── 写 ────────────────────────────────────────────────────────
    def put(self, data: bytes | str, *, media: str = "application/octet-stream") -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        ref = _digest(raw)
        path = self._path_for(ref)
        if path.exists() and path.stat().st_size == len(raw):
            return ref  # 去重：同内容同 ref
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(raw)
        tmp.replace(path)  # 原子落盘，避免半截文件被当成证据
        if media != "application/octet-stream":
            self._sidecar(path).write_text(
                json.dumps({"media": media, "size": len(raw)}, ensure_ascii=False), encoding="utf-8"
            )
        return ref

    def put_json(self, obj: Any) -> str:
        return self.put(
            json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str),
            media="application/json",
        )

    def put_text(self, text: str) -> str:
        return self.put(text, media="text/plain; charset=utf-8")

    # ── 读 ────────────────────────────────────────────────────────
    def get(self, ref: str) -> bytes:
        path = self._path_for(ref)
        if not path.exists():
            raise BlobNotFound(f"blob 不存在: {ref}", detail={"ref": ref, "path": str(path)})
        return path.read_bytes()

    def get_text(self, ref: str) -> str:
        return self.get(ref).decode("utf-8", errors="replace")

    def get_json(self, ref: str) -> Any:
        return json.loads(self.get_text(ref))

    def exists(self, ref: str) -> bool:
        try:
            return self._path_for(ref).exists()
        except InvalidBlobRef:
            return False

    def stat(self, ref: str) -> BlobStat:
        path = self._path_for(ref)
        if not path.exists():
            raise BlobNotFound(f"blob 不存在: {ref}", detail={"ref": ref})
        media = "application/octet-stream"
        sidecar = self._sidecar(path)
        if sidecar.exists():
            with contextlib.suppress(json.JSONDecodeError):
                media = str(json.loads(sidecar.read_text(encoding="utf-8")).get("media", media))
        return BlobStat(ref=ref, size=path.stat().st_size, media=media, path=path)

    # ── 维护 ──────────────────────────────────────────────────────
    def iter_refs(self) -> list[str]:
        out: list[str] = []
        for path in self.root.rglob("*"):
            if path.is_file() and not path.name.endswith((".tmp", ".meta.json")):
                out.append(f"sha256:{path.name}")
        return out

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    # ── 内部 ──────────────────────────────────────────────────────
    def _path_for(self, ref: str) -> Path:
        m = _REF_RE.match(ref or "")
        if not m:
            raise InvalidBlobRef(
                f"非法 blob ref（必须形如 sha256:<64 hex>）: {ref!r}", detail={"ref": ref}
            )
        hexdig = m.group(1)
        parts = [hexdig[i : i + 2] for i in range(0, self.shard_depth * 2, 2)]
        return self.root.joinpath(*parts, hexdig)

    @staticmethod
    def _sidecar(path: Path) -> Path:
        return path.with_suffix(".meta.json")


class MemoryBlobStore:
    """测试与 dry-run 用：不落盘，接口一致。"""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}
        self._media: dict[str, str] = {}

    def put(self, data: bytes | str, *, media: str = "application/octet-stream") -> str:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
        ref = _digest(raw)
        self._data[ref] = raw
        self._media[ref] = media
        return ref

    def put_json(self, obj: Any) -> str:
        return self.put(
            json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str),
            media="application/json",
        )

    def put_text(self, text: str) -> str:
        return self.put(text, media="text/plain; charset=utf-8")

    def get(self, ref: str) -> bytes:
        if not _REF_RE.match(ref or ""):
            raise InvalidBlobRef(f"非法 blob ref: {ref!r}")
        if ref not in self._data:
            raise BlobNotFound(f"blob 不存在: {ref}", detail={"ref": ref})
        return self._data[ref]

    def get_text(self, ref: str) -> str:
        return self.get(ref).decode("utf-8", errors="replace")

    def get_json(self, ref: str) -> Any:
        return json.loads(self.get_text(ref))

    def exists(self, ref: str) -> bool:
        return ref in self._data

    def stat(self, ref: str) -> BlobStat:
        raw = self.get(ref)
        return BlobStat(ref=ref, size=len(raw), media=self._media.get(ref, ""), path=Path(ref))

    def __len__(self) -> int:
        return len(self._data)
