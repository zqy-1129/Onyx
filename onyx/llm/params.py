"""归一化参数 ↔ 各引擎参数映射。

纪律：**None 表示"不传"**，绝不填默认值。引擎默认值随版本变化，
我们填一个 0.8 进去就等于永久冻结了一个假设，还会让评测结果无法解释。
"""

from __future__ import annotations

import dataclasses
from typing import Any

from onyx.core.types import GenParams

#: GenParams 字段 → Ollama `options.*` 键名
_TO_OLLAMA_OPTIONS: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "num_predict",
    "num_ctx": "num_ctx",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
    "repeat_penalty": "repeat_penalty",
}

#: GenParams 字段 → OpenAI 兼容层顶层键名
_TO_OPENAI: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "max_tokens": "max_tokens",
    "seed": "seed",
    "stop": "stop",
    "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty",
}

#: Ollama 的 OpenAI 兼容层明确不支持的参数（官方文档列出）。
#: 命中时必须走原生通道或显式报 CapabilityMissing，不许静默丢弃。
OLLAMA_OPENAI_UNSUPPORTED: frozenset[str] = frozenset({"tool_choice", "logit_bias", "user", "n"})


def to_ollama_options(params: GenParams) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for field_name, key in _TO_OLLAMA_OPTIONS.items():
        value = getattr(params, field_name)
        if _unset(value):
            continue
        options[key] = list(value) if isinstance(value, tuple) else value
    # 引擎特有参数原样透传（如 min_p / typical_p / numa）
    options.update(params.extra)
    return options


def to_openai_params(params: GenParams) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field_name, key in _TO_OPENAI.items():
        value = getattr(params, field_name)
        if _unset(value):
            continue
        out[key] = list(value) if isinstance(value, tuple) else value
    dropped = dropped_by_openai(params)
    if dropped:
        # 不静默丢：把丢弃项记进返回值，调用方负责落进 trace.params 以便复盘
        out["_dropped_params"] = dropped
    return out


def dropped_by_openai(params: GenParams) -> list[str]:
    """走 OpenAI 兼容通道时会失效的参数名。

    无等价物的字段列表**从 dataclass 字段自动推导**，不从硬编码清单来：
    以后给 GenParams 加字段时，这里会自动跟上，不会出现"新参数被静默丢弃"。
    """
    dropped = [
        name for name in _NO_OPENAI_EQUIVALENT
        if not _unset(getattr(params, name))
    ]
    dropped += sorted(set(params.extra) - set(_TO_OPENAI.values()))
    return dropped


#: 与 dropped_by_openai 共用；`ollama_options_dropped_by_openai` 是它的别名语义
_NO_OPENAI_EQUIVALENT: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(GenParams)
    if f.name not in _TO_OPENAI and f.name not in {"json_schema", "extra"}
)


def _unset(value: Any) -> bool:
    return value is None or value == () or value == {} or value == []


def ollama_options_dropped_by_openai(params: GenParams) -> list[str]:
    """`dropped_by_openai` 的别名，保留给调用方语义化命名。"""
    return dropped_by_openai(params)
