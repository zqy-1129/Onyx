"""Fleet 与模型档案：总览页的数据源。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from onyx import __version__
from onyx.api.app import iso_before
from onyx.api.deps import AppState, get_state
from onyx.api.schemas import FleetView, HealthView, LoadedModelView, ModelView
from onyx.core.types import LoadedModel, ModelCard
from onyx.llm.caps import CapReport, infer_caps

router = APIRouter(prefix="/api", tags=["fleet"])

WINDOW_SECONDS = 3600


def _keep_alive_seconds(expires_at: str) -> float | None:
    if not expires_at:
        return None
    try:
        target = datetime.fromisoformat(expires_at)
    except ValueError:
        return None
    if target.tzinfo is None:
        return None
    return round((target - datetime.now(target.tzinfo)).total_seconds(), 1)


def _loaded_view(item: LoadedModel) -> LoadedModelView:
    return LoadedModelView(
        name=item.name, size=item.size, size_vram=item.size_vram, vram_share=item.vram_share,
        offloaded=item.offloaded, context_length=item.context_length, expires_at=item.expires_at,
        keep_alive_seconds=_keep_alive_seconds(item.expires_at),
        quantization=item.quantization, parameter_size=item.parameter_size,
    )


def _caps_for(state: AppState, card: ModelCard) -> tuple[CapReport, dict[str, Any]]:
    """优先用已回灌的探针结论；没跑过探针就只用引擎自报，并把未测项标为 unknown。"""
    record = state.models.find_by_name(card.provider_id, card.name)
    stored = (record.extra or {}).get("caps") if record else None
    if stored:
        return CapReport.from_dict(stored), stored
    findings = (record.probe if record else {}) or {}
    info = state.runtime.provider.info()
    report = infer_caps(
        engine_capabilities=card.capabilities, probe_findings=findings,
        api_style=info.api_style, provider_kind=info.kind,
    )
    return report, report.as_dict()


def _model_view(state: AppState, card: ModelCard, loaded: dict[str, LoadedModel]) -> ModelView:
    record = state.models.find_by_name(card.provider_id, card.name)
    _, caps_dict = _caps_for(state, card)
    extra = (record.extra or {}) if record else {}
    live = loaded.get(card.name)
    calibration = {
        "cjk_ratio": extra.get("fitted_cjk_ratio"),
        "other_ratio": extra.get("fitted_other_ratio"),
        "intercept": extra.get("fitted_intercept"),
        "r2": extra.get("fitted_r2"),
        "max_rel_error": extra.get("fitted_max_rel_error"),
        "n": record.usage_ratio_n if record else 0,
        "cold_ms_per_token": extra.get("cold_ms_per_token"),
        "warm_ms_per_token": extra.get("warm_ms_per_token"),
    }
    return ModelView(
        id=f"{card.provider_id}/{card.name}", name=card.name, provider_id=card.provider_id,
        parameter_size=card.parameter_size, quantization=card.quantization, size_gb=card.size_gb,
        capabilities=list(card.capabilities), caps=caps_dict,
        tool_format=(record.tool_format if record else "unknown"),
        ctx_train=card.context_length, ctx_loaded=live.context_length if live else None,
        tokenizer_source=(record.tokenizer_source if record else "none"),
        calibrated=bool(record and record.usage_ratio and (record.usage_ratio_n or 0) >= 30),
        calibration=calibration, probed=bool(record and record.probe),
        loaded=live is not None,
    )


@router.get("/health", response_model=HealthView)
def health(state: AppState = Depends(get_state)) -> HealthView:
    info = state.runtime.provider.info()
    return HealthView(
        ok=info.reachable, version=__version__,
        schema_version=state.runtime.db.version(),
        provider_reachable=info.reachable, engine_version=info.version,
    )


@router.get("/fleet", response_model=FleetView)
def fleet(state: AppState = Depends(get_state)) -> FleetView:
    """总览：服务状态 + 已载入模型 + 近 1 小时窗口指标 + 异常分布。"""
    provider = state.runtime.provider
    info = provider.info()
    loaded = provider.running()
    since = iso_before(WINDOW_SECONDS)
    summary = state.usage.summarize(since=since)

    traces_in_window = state.traces.count(since=since)
    errors = state.traces.count(since=since, status="error")
    decode_values = [
        row["decode_tps"] for row in state.runtime.db.query(
            "SELECT u.decode_tps FROM usage u JOIN trace t ON t.id=u.trace_id "
            "WHERE t.started_at>=? AND u.decode_tps IS NOT NULL", (since,)
        )
    ]
    window: dict[str, Any] = {
        "seconds": WINDOW_SECONDS,
        "traces": traces_in_window,
        "errors": errors,
        "error_rate": round(errors / traces_in_window, 4) if traces_in_window else 0.0,
        "in_tokens": summary.in_tokens,
        "out_tokens": summary.out_tokens,
        "thinking_tokens": summary.thinking_tokens,
        "decode_tps_avg": round(sum(decode_values) / len(decode_values), 2) if decode_values else None,
        "by_prefill_mode": state.usage.by_prefill_mode(since=since),
    }
    return FleetView(
        ok=info.reachable, app_version=__version__, provider_id=info.id,
        provider_kind=str(info.kind), provider_reachable=info.reachable,
        engine_version=info.version, base_url=info.base_url,
        loaded_models=[_loaded_view(item) for item in loaded],
        installed_models=len(provider.list_models()),
        window=window, anomalies=state.traces.anomaly_counts(since=since),
    )


@router.get("/models", response_model=list[ModelView])
def models(state: AppState = Depends(get_state)) -> list[ModelView]:
    provider = state.runtime.provider
    loaded = {item.name: item for item in provider.running()}
    return [_model_view(state, card, loaded) for card in provider.list_models()]


@router.get("/models/detail", response_model=ModelView)
def model_detail(
    name: str = Query(..., description="模型名，如 qwen3.5:9b"),
    state: AppState = Depends(get_state),
) -> ModelView:
    """模型详情。用 query 参数而不是路径参数：模型名可能含 `/`（命名空间）。"""
    provider = state.runtime.provider
    card = next((c for c in provider.list_models() if c.name == name), None)
    if card is None:
        raise HTTPException(status_code=404, detail=f"模型不存在: {name}")
    loaded = {item.name: item for item in provider.running()}
    return _model_view(state, card, loaded)


@router.get("/models/probe")
def model_probe(name: str = Query(...), state: AppState = Depends(get_state)) -> dict[str, Any]:
    """探针结论与 chat template / GGUF 元数据摘要。

    只回传 tokenizer 的**键名与规模**，不回传整个 vocab（可能几 MB）。
    """
    provider = state.runtime.provider
    detail = provider.show_model(name)
    record = state.models.find_by_name(provider.id, name)
    tokenizer_keys = sorted(k for k in detail.model_info if k.startswith("tokenizer."))
    return {
        "name": name,
        "capabilities": list(detail.capabilities),
        "thinking": detail.thinking,
        "template_chars": len(detail.template),
        "template_head": detail.template[:200],
        "has_real_template": len(detail.template) > 100,
        "tokenizer_family": detail.tokenizer_family,
        "tokenizer_keys": tokenizer_keys,
        "has_vocab": "tokenizer.ggml.tokens" in detail.model_info,
        "has_merges": "tokenizer.ggml.merges" in detail.model_info,
        "has_chat_template": "tokenizer.chat_template" in detail.model_info,
        "model_info_keys": len(detail.model_info),
        "probe_findings": (record.probe if record else {}) or {},
        "tool_format": record.tool_format if record else "unknown",
        "license": detail.license[:200],
        "parameters": detail.parameters[:400],
    }

