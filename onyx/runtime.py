"""运行时装配：把 provider / gateway / 观测 / 存储接成一个可用整体。

CLI、REST API、评测 runner 都从这里拿实例——**只允许有一个装配点**，
否则三处各自接线，口径迟早分裂（DESIGN 原则 1 的延伸）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onyx.core.content import BlobStore, FileBlobStore
from onyx.llm.gateway import Gateway
from onyx.llm.measurement.fidelity import CounterContext
from onyx.llm.providers.base import LlmProvider
from onyx.llm.registry import build_provider
from onyx.obs.engine import ObserverEngine
from onyx.settings import Settings, load_settings
from onyx.store.db import Database
from onyx.store.records import ProviderRecord
from onyx.store.repos import ModelRepo
from onyx.store.sinks import EventFanout, JsonlEventSink, RecordSink, SqliteRecordSink


@dataclass
class Runtime:
    settings: Settings
    db: Database
    sink: RecordSink
    observer: ObserverEngine
    blobs: BlobStore
    provider: LlmProvider
    gateway: Gateway
    events: EventFanout

    def flush(self, timeout: float = 5.0) -> None:
        self.sink.flush(timeout)
        self.events.flush(timeout)

    def close(self) -> None:
        self.flush()
        self.events.close()
        self.sink.close()
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()
        self.db.close()

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def build_runtime(
    *,
    provider_kind: str = "ollama",
    provider_id: str = "ollama-local",
    base_url: str = "http://127.0.0.1:11434",
    settings: Settings | None = None,
    db_path: Path | str | None = None,
    counter_ctx: CounterContext | None = None,
    sample_gpu: bool = False,
    event_log: bool = True,
    provider_kwargs: dict[str, Any] | None = None,
) -> Runtime:
    resolved = settings or load_settings()
    resolved.ensure_dirs()
    db = Database(db_path or resolved.db_path)
    sink = SqliteRecordSink(db)
    observer = ObserverEngine(record_sink=sink)
    blobs = FileBlobStore(resolved.blob_dir)

    events = EventFanout()
    if event_log:
        events.add(JsonlEventSink(resolved.data_dir / "events.ndjson"))

    provider = build_provider(
        provider_kind, id=provider_id, base_url=base_url, **(provider_kwargs or {})
    )
    gateway = Gateway(
        provider,
        observer=observer,
        blobs=blobs,
        counter_ctx=counter_ctx or CounterContext(),
        sample_gpu=sample_gpu,
        event_sink=events.emit,
    )
    return Runtime(
        settings=resolved, db=db, sink=sink, observer=observer, blobs=blobs,
        provider=provider, gateway=gateway, events=events,
    )


def register_provider(runtime: Runtime) -> None:
    """把 provider 与它的模型清单落库（`onyx models sync` 的核心）。"""
    info = runtime.provider.info()
    ModelRepo(runtime.db).upsert_provider(ProviderRecord(
        id=info.id, kind=str(info.kind), base_url=info.base_url, api_style=str(info.api_style),
        caps=tuple(sorted(str(c) for c in info.caps)), version=info.version,
    ))


def sync_models(runtime: Runtime) -> int:
    """从引擎拉模型清单并 upsert。返回同步到的模型数。"""
    from onyx.core.clock import utc_now_iso
    from onyx.store.records import ModelRecord

    register_provider(runtime)
    repo = ModelRepo(runtime.db)
    info = runtime.provider.info()
    now = utc_now_iso()
    count = 0
    for card in runtime.provider.list_models():
        repo.upsert_model(ModelRecord(
            id=f"{info.id}/{card.name}", provider_id=info.id, name=card.name,
            remote_model=card.remote_model, remote_host=card.remote_host,
            digest=card.digest, bytes=card.bytes or None, modified_at=card.modified_at,
            family=card.family, families=card.families, parameter_size=card.parameter_size,
            quantization=card.quantization, format=card.format,
            capabilities=card.capabilities, ctx_train=card.context_length,
            last_seen_at=now,
            extra={"embedding_length": card.embedding_length, **card.extra},
        ))
        count += 1
    return count
