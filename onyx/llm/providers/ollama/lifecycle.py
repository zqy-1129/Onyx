"""Ollama 控制面：/api/version /api/tags /api/show /api/ps /api/delete /api/pull /api/copy。

解析原则：**只认实测到的字段，未知字段进 extra**。
Ollama 0.35 的 /api/tags 实际返回 capabilities 与 details.context_length，
这两项官方文档并未列出——所以任何"按文档写死"的解析都会漏掉看板最需要的信息。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from onyx.core.errors import ProviderRejected
from onyx.core.types import AdminResult, LoadedModel, ModelCard, ModelDetail
from onyx.llm.providers.ollama.client import OllamaClient

EventEmitter = Callable[[dict[str, Any]], None]

_KNOWN_TAG_KEYS = {"name", "model", "remote_model", "remote_host", "modified_at", "size", "digest",
                   "details", "capabilities"}
_KNOWN_DETAIL_KEYS = {"parent_model", "format", "family", "families", "parameter_size",
                      "quantization_level", "context_length", "embedding_length"}


def version(client: OllamaClient) -> str:
    raw = client.get_json("/api/version")
    return str(raw.get("version", "")) if isinstance(raw, dict) else ""


def list_models(client: OllamaClient, provider_id: str) -> list[ModelCard]:
    raw = client.get_json("/api/tags")
    models = raw.get("models") or [] if isinstance(raw, dict) else []
    out: list[ModelCard] = []
    for item in models:
        details = item.get("details") or {}
        extra = {k: v for k, v in item.items() if k not in _KNOWN_TAG_KEYS}
        extra.update({f"details.{k}": v for k, v in details.items() if k not in _KNOWN_DETAIL_KEYS})
        out.append(ModelCard(
            provider_id=provider_id,
            name=str(item.get("name", "")),
            model=str(item.get("model", item.get("name", ""))),
            remote_model=str(item.get("remote_model", "") or ""),
            remote_host=str(item.get("remote_host", "") or ""),
            modified_at=str(item.get("modified_at", "") or ""),
            bytes=int(item.get("size") or 0),
            digest=str(item.get("digest", "") or ""),
            family=str(details.get("family", "") or ""),
            families=tuple(details.get("families") or ()),
            parameter_size=str(details.get("parameter_size", "") or ""),
            quantization=str(details.get("quantization_level", "") or ""),
            format=str(details.get("format", "") or ""),
            capabilities=tuple(item.get("capabilities") or ()),
            context_length=_int_or_none(details.get("context_length")),
            embedding_length=_int_or_none(details.get("embedding_length")),
            extra=extra,
        ))
    return out


def running_models(client: OllamaClient) -> list[LoadedModel]:
    raw = client.get_json("/api/ps")
    models = raw.get("models") or [] if isinstance(raw, dict) else []
    out: list[LoadedModel] = []
    for item in models:
        details = item.get("details") or {}
        out.append(LoadedModel(
            name=str(item.get("name", "")),
            model=str(item.get("model", item.get("name", ""))),
            digest=str(item.get("digest", "") or ""),
            size=int(item.get("size") or 0),
            size_vram=int(item.get("size_vram") or 0),
            context_length=_int_or_none(item.get("context_length")),
            expires_at=str(item.get("expires_at", "") or ""),
            parameter_size=str(details.get("parameter_size", "") or ""),
            quantization=str(details.get("quantization_level", "") or ""),
            family=str(details.get("family", "") or ""),
            parent_model=str(details.get("parent_model", "") or ""),
            extra={k: v for k, v in item.items()
                   if k not in {"name", "model", "digest", "size", "size_vram", "context_length",
                                "expires_at", "details"}},
        ))
    return out


def show_model(client: OllamaClient, name: str) -> ModelDetail:
    # /api/show 是 POST（GET 会返回 405）——实测得到，文档未明确
    raw = client.post_json("/api/show", {"name": name})
    if not isinstance(raw, dict):
        raise ProviderRejected(f"/api/show 返回非对象: {raw!r}")
    details = raw.get("details") or {}
    return ModelDetail(
        name=name,
        template=str(raw.get("template", "") or ""),
        capabilities=tuple(raw.get("capabilities") or ()),
        model_info=dict(raw.get("model_info") or {}),
        parameters=str(raw.get("parameters", "") or ""),
        license=_license_text(raw.get("license")),
        system=str(raw.get("system", "") or ""),
        thinking=raw.get("thinking") if isinstance(raw.get("thinking"), bool) else None,
        modified_at=str(raw.get("modified_at", "") or ""),
        details=details,
        extra={k: v for k, v in raw.items()
               if k not in {"template", "capabilities", "model_info", "parameters", "license",
                            "system", "thinking", "modified_at", "details"}},
    )


def delete_model(client: OllamaClient, name: str) -> AdminResult:
    try:
        client.delete_json("/api/delete", {"name": name})
        return AdminResult(ok=True, action="delete", detail={"name": name})
    except ProviderRejected as exc:
        return AdminResult(ok=False, action="delete", error=exc.message, detail=exc.detail)


def copy_model(client: OllamaClient, source: str, destination: str) -> AdminResult:
    try:
        client.post_json("/api/copy", {"source": source, "destination": destination})
        return AdminResult(ok=True, action="copy", detail={"source": source, "destination": destination})
    except ProviderRejected as exc:
        return AdminResult(ok=False, action="copy", error=exc.message, detail=exc.detail)


def pull_model(
    client: OllamaClient, name: str, *, on_progress: EventEmitter | None = None
) -> Iterator[dict[str, Any]]:
    """拉取模型，逐条产出进度事件（Ollama 的 /api/pull 是 ndjson 流）。"""
    for chunk in client.post_ndjson("/api/pull", {"name": name, "stream": True}):
        if on_progress is not None:
            on_progress(chunk)
        yield chunk


def pull_outcome(chunks: Iterable[dict[str, Any]]) -> tuple[bool, str, dict[str, Any]]:
    """按 ndjson 流的**收尾形状**判定拉取成败：返回 (ok, error, 判据所在的那条 chunk)。

    Ollama 0.35.1 的 `/api/pull` 以 `{"status":"success"}` 结尾，**没有 `done` 字段**。
    早先按 `last["done"]` 判 ⇒ 每一次真拉都被报成失败（模型其实已经在盘上，
    用户接着会遇到"我说没拉到、你再拉一次又说已经有了"）。现在两代形状都认，
    并且**任何一条带 error 就算失败**：成功的操作被判成失败，与失败被判成成功，
    同样是把用户的判断带偏。
    """
    last: dict[str, Any] = {}
    for chunk in chunks:
        last = chunk
        if chunk.get("error"):
            return False, str(chunk["error"]), chunk
    status = str(last.get("status") or "")
    if last.get("done") or status == "success":
        return True, "", last
    return False, f"拉流没有正常收尾（最后一条是 {status or '空'}）", last


def unload_model(client: OllamaClient, name: str) -> AdminResult:
    """卸载：发一个 keep_alive=0 的空 generate 请求。不依赖任何未文档化端点。"""
    try:
        client.post_json("/api/generate", {"model": name, "keep_alive": 0})
        return AdminResult(ok=True, action="unload", detail={"name": name})
    except ProviderRejected as exc:
        return AdminResult(ok=False, action="unload", error=exc.message, detail=exc.detail)


def _license_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(v) for v in value)
    return str(value or "")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
