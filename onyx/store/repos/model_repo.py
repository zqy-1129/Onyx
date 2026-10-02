from __future__ import annotations

from onyx.core.clock import utc_now_iso
from onyx.store.codec import dumps, loads_dict, loads_list
from onyx.store.db import Database
from onyx.store.records import MODEL_UPDATABLE_COLUMNS, ModelRecord, ProviderRecord


class ModelRepo:
    def __init__(self, db: Database) -> None:
        self.db = db

    # ── provider ──────────────────────────────────────────────────
    def upsert_provider(self, rec: ProviderRecord) -> None:
        now = rec.created_at or utc_now_iso()
        self.db.execute(
            """INSERT INTO provider(id, kind, base_url, api_style, enabled, caps_json, version,
                                    config_json, created_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 kind=excluded.kind, base_url=excluded.base_url, api_style=excluded.api_style,
                 enabled=excluded.enabled, caps_json=excluded.caps_json, version=excluded.version,
                 config_json=excluded.config_json""",
            (
                rec.id,
                rec.kind,
                rec.base_url,
                rec.api_style,
                int(rec.enabled),
                dumps(list(rec.caps)),
                rec.version,
                dumps(rec.config),
                now,
            ),
        )

    def list_providers(self, *, enabled_only: bool = False) -> list[ProviderRecord]:
        sql = "SELECT * FROM provider"
        if enabled_only:
            sql += " WHERE enabled=1"
        return [self._provider_from_row(r) for r in self.db.query(sql + " ORDER BY id")]

    def _provider_from_row(self, row) -> ProviderRecord:
        return ProviderRecord(
            id=row["id"],
            kind=row["kind"],
            base_url=row["base_url"],
            api_style=row["api_style"],
            enabled=bool(row["enabled"]),
            caps=tuple(loads_list(row["caps_json"])),
            version=row["version"] or "",
            config=loads_dict(row["config_json"]),
            created_at=row["created_at"],
        )

    # ── model ─────────────────────────────────────────────────────
    def upsert_model(self, rec: ModelRecord) -> str:
        now = utc_now_iso()
        first = rec.first_seen_at or now
        last = rec.last_seen_at or now
        self.db.execute(
            """INSERT INTO model(id, provider_id, name, remote_model, remote_host, digest, bytes,
                                 modified_at, family, families_json, parameter_size, quantization,
                                 format, parent_model, ctx_train, capabilities_json, template,
                                 model_info_json, tool_format, tokenizer_source, tokenizer_ref,
                                 usage_ratio, usage_ratio_n, probe_json, first_seen_at, last_seen_at,
                                 extra_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(provider_id, name) DO UPDATE SET
                 remote_model=excluded.remote_model, remote_host=excluded.remote_host,
                 digest=excluded.digest, bytes=excluded.bytes, modified_at=excluded.modified_at,
                 family=excluded.family, families_json=excluded.families_json,
                 parameter_size=excluded.parameter_size, quantization=excluded.quantization,
                 format=excluded.format, parent_model=excluded.parent_model,
                 last_seen_at=excluded.last_seen_at""",
            (
                rec.id,
                rec.provider_id,
                rec.name,
                rec.remote_model,
                rec.remote_host,
                rec.digest,
                rec.bytes,
                rec.modified_at,
                rec.family,
                dumps(list(rec.families)),
                rec.parameter_size,
                rec.quantization,
                rec.format,
                rec.parent_model,
                rec.ctx_train,
                dumps(list(rec.capabilities)),
                rec.template,
                dumps(rec.model_info),
                rec.tool_format,
                rec.tokenizer_source,
                rec.tokenizer_ref,
                rec.usage_ratio,
                rec.usage_ratio_n,
                dumps(rec.probe),
                first,
                last,
                dumps(rec.extra),
            ),
        )
        row = self.db.query_one("SELECT id FROM model WHERE provider_id=? AND name=?", (rec.provider_id, rec.name))
        return str(row["id"])

    def update_model(self, model_id: str, **fields) -> None:
        """回灌探针结论 / tokenizer 标定。列名走白名单，防注入也防误写。

        `*_json` 列接受 dict/list 并自动序列化——调用方不该关心存储表示。
        """
        bad = set(fields) - MODEL_UPDATABLE_COLUMNS
        if bad:
            raise KeyError(f"不允许更新的列: {sorted(bad)}（白名单见 MODEL_UPDATABLE_COLUMNS）")
        if not fields:
            return
        values = tuple(
            dumps(v) if col.endswith("_json") and not isinstance(v, str | type(None)) else v
            for col, v in fields.items()
        )
        assignments = ", ".join(f"{col}=?" for col in fields)
        self.db.execute(
            f"UPDATE model SET {assignments} WHERE id=?",
            (*values, model_id),
        )

    def get_model(self, model_id: str) -> ModelRecord | None:
        row = self.db.query_one("SELECT * FROM model WHERE id=?", (model_id,))
        return self._model_from_row(row) if row else None

    def find_by_name(self, provider_id: str, name: str) -> ModelRecord | None:
        row = self.db.query_one(
            "SELECT * FROM model WHERE provider_id=? AND name=?", (provider_id, name)
        )
        return self._model_from_row(row) if row else None

    def list_models(self, provider_id: str | None = None) -> list[ModelRecord]:
        if provider_id:
            rows = self.db.query("SELECT * FROM model WHERE provider_id=? ORDER BY name", (provider_id,))
        else:
            rows = self.db.query("SELECT * FROM model ORDER BY provider_id, name")
        return [self._model_from_row(r) for r in rows]

    def count(self) -> int:
        return int(self.db.scalar("SELECT COUNT(*) FROM model", default=0))

    def _model_from_row(self, row) -> ModelRecord:
        return ModelRecord(
            id=row["id"],
            provider_id=row["provider_id"],
            name=row["name"],
            remote_model=row["remote_model"] or "",
            remote_host=row["remote_host"] or "",
            digest=row["digest"] or "",
            bytes=row["bytes"],
            modified_at=row["modified_at"] or "",
            family=row["family"] or "",
            families=tuple(loads_list(row["families_json"])),
            parameter_size=row["parameter_size"] or "",
            quantization=row["quantization"] or "",
            format=row["format"] or "",
            parent_model=row["parent_model"] or "",
            ctx_train=row["ctx_train"],
            capabilities=tuple(loads_list(row["capabilities_json"])),
            template=row["template"] or "",
            model_info=loads_dict(row["model_info_json"]),
            tool_format=row["tool_format"] or "unknown",
            tokenizer_source=row["tokenizer_source"] or "none",
            tokenizer_ref=row["tokenizer_ref"] or "",
            usage_ratio=row["usage_ratio"],
            usage_ratio_n=row["usage_ratio_n"],
            probe=loads_dict(row["probe_json"]),
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            extra=loads_dict(row["extra_json"]),
        )
