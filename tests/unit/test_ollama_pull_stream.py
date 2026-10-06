"""Ollama `/api/pull` 的收尾形状（S33 前提实测顺手钉住的一处真缺陷）。

判"拉取成功"的依据本来只有一处：`last["done"]`。而 2026-10-06 实测 Ollama 0.35.1 的流是这样收尾的：

    {"status":"verifying sha256 digest"}
    {"status":"writing manifest"}
    {"status":"success"}            ← **没有 done 字段**

于是每一次真拉都被判成失败（模型明明已经在盘上，用户接着会遇到"我说没拉到、你再拉又说有了"）。
本文件把两代形状都钉住，并且把三种误判分开断言：
中途报错后补一条 success、流被截断、以及"什么也没产出"。
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from onyx.llm.providers.ollama import lifecycle
from onyx.llm.providers.ollama.client import OllamaClient
from onyx.llm.providers.ollama.provider import OllamaProvider

# 实测到的 0.35.1 收尾（没有 done）
REAL_TAIL = [
    {"status": "pulling manifest"},
    {"status": "pulling 06507c7b4268", "digest": "sha256:06507c7b4268",
     "total": 639_150_592, "completed": 639_150_592},
    {"status": "verifying sha256 digest"},
    {"status": "writing manifest"},
    {"status": "success"},
]
LEGACY_TAIL = [{"status": "downloading", "digest": "sha256:aaa"},
               {"status": "success", "done": True, "digest": "sha256:aaa"}]
ERROR_TAIL = [{"status": "pulling manifest"},
              {"error": "toomanyrequests: too many requests"}]
TRUNCATED_TAIL = [{"status": "pulling manifest"}, {"status": "writing manifest"}]

_TAILS = {"real": REAL_TAIL, "legacy": LEGACY_TAIL, "error": ERROR_TAIL,
          "truncated": TRUNCATED_TAIL, "empty": []}


def _tail() -> list[dict[str, Any]]:
    return _TAILS[os.environ.get("OLLAMA_PULL_TAIL", "real")]


def _handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/pull", request.url.path
    body = "\n".join(json.dumps(chunk) for chunk in _tail()) + "\n"
    return httpx.Response(200, content=body.encode(),
                          headers={"content-type": "application/x-ndjson"})


def _provider() -> OllamaProvider:
    return OllamaProvider(
        id="ollama-stub", base_url="http://stub.local",
        client=OllamaClient("http://stub.local", transport=httpx.MockTransport(_handler)),
    )


# ── 判据本体 ──────────────────────────────────────────────────────
def test_engine_tail_without_done_is_success():
    ok, error, last = lifecycle.pull_outcome(REAL_TAIL)
    assert ok is True and error == ""
    assert last["status"] == "success", "收尾那条要能被拿去做出处"


def test_legacy_tail_with_done_is_still_success():
    ok, error, last = lifecycle.pull_outcome(LEGACY_TAIL)
    assert ok is True and error == ""
    assert last["digest"] == "sha256:aaa"


def test_an_error_chunk_wins_even_if_a_success_line_follows():
    """中途报过错，后面再补一条 success 也不算成功：拉取是**一串**事实，不是最后一行。"""
    ok, error, _last = lifecycle.pull_outcome(
        [{"error": "404: model not found"}, {"status": "success"}]
    )
    assert ok is False and "404" in error


def test_a_truncated_stream_is_reported_as_failure_naming_the_last_status():
    """流停在 `writing manifest` ⇒ 必须说"没有正常收尾"，而不是沉默地判成功。"""
    ok, error, last = lifecycle.pull_outcome(TRUNCATED_TAIL)
    assert ok is False and last["status"] == "writing manifest"
    assert "writing manifest" in error


def test_an_empty_stream_is_a_failure_not_a_zero_sized_success():
    ok, error, _last = lifecycle.pull_outcome([])
    assert ok is False and error, "什么都没产出不能算成功（未知 ≠ 做完了）"


# ── provider 侧接线 ──────────────────────────────────────────────
def test_provider_pull_accepts_the_real_0_35_tail(monkeypatch):
    monkeypatch.setenv("OLLAMA_PULL_TAIL", "real")
    result = _provider().pull("qwen3-embedding:0.6b")
    assert result.ok is True and result.error == ""
    assert result.action == "pull"
    assert result.detail["name"] == "qwen3-embedding:0.6b"
    assert result.detail["status"] == "success"


def test_provider_pull_reports_the_engine_rejection(monkeypatch):
    monkeypatch.setenv("OLLAMA_PULL_TAIL", "error")
    result = _provider().pull("demo/missing")
    assert result.ok is False
    assert "toomanyrequests" in result.error
    assert result.detail["name"] == "demo/missing"


def test_provider_pull_keeps_the_legacy_done_shape_working(monkeypatch):
    monkeypatch.setenv("OLLAMA_PULL_TAIL", "legacy")
    result = _provider().pull("demo/old")
    assert result.ok is True and result.detail["digest"] == "sha256:aaa"


def test_provider_pull_of_a_truncated_stream_leaves_a_readable_reason(monkeypatch):
    monkeypatch.setenv("OLLAMA_PULL_TAIL", "truncated")
    result = _provider().pull("demo/half")
    assert result.ok is False and "收尾" in result.error
    # 判据落在哪一条上要能看见：这里停在 writing manifest，所以 digest 是空的
    assert result.detail["status"] == "writing manifest" and result.detail["digest"] == ""
