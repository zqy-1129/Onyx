"""探针运行器：上下文、注册表、报告渲染。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.core.clock import SYSTEM_CLOCK, Clock
from onyx.core.types import (
    Generation,
    GenerationRequest,
    GenParams,
    Message,
    ProbeFinding,
    ProbeReport,
    Role,
    TokenSource,
)

#: 一个探针 = 接收上下文、返回结论的函数。注册表让它可被外部插件扩展。
ProbeFn = Callable[["ProbeContext"], ProbeFinding]

_REGISTRY: dict[str, ProbeFn] = {}


def probe(name: str) -> Callable[[ProbeFn], ProbeFn]:
    """注册探针。名字即 CLI `--suite` 的取值，保持稳定。"""

    def decorator(fn: ProbeFn) -> ProbeFn:
        _REGISTRY[name] = fn
        fn.probe_name = name  # type: ignore[attr-defined]
        return fn

    return decorator


def registered_probes() -> list[str]:
    return sorted(_REGISTRY)


@dataclass
class ProbeContext:
    """探针的执行环境。

    `samples` 收集每一次真实调用的关键数字，作为结论的证据链——
    没有证据的结论一律标 `unknown=True`。
    """

    provider: Any
    model: str
    clock: Clock = SYSTEM_CLOCK
    samples: list[dict[str, Any]] = field(default_factory=list)
    provider_version: str = ""

    def ask(
        self,
        prompt: str | Sequence[Message],
        *,
        max_tokens: int = 128,
        thinking: bool | None = None,
        keep_alive: str | None = "5m",
        stream: bool = False,
        seed: int | None = None,
        temperature: float = 0.0,
        json_schema: dict[str, Any] | None = None,
        tag: str = "",
        **extra: Any,
    ) -> Generation:
        messages = (
            (Message(role=Role.USER, content=prompt),)
            if isinstance(prompt, str)
            else tuple(prompt)
        )
        req = GenerationRequest(
            model=self.model,
            messages=messages,
            params=GenParams(
                max_tokens=max_tokens, temperature=temperature, seed=seed, json_schema=json_schema
            ),
            thinking=thinking,
            keep_alive=keep_alive,
            stream=stream,
        )
        started = time.monotonic()
        gen = self.provider.generate(req, trace_id=f"probe:{tag or 'x'}", **extra)
        self.record(tag, gen, wall_ms=(time.monotonic() - started) * 1000)
        return gen

    def record(self, tag: str, gen: Generation, **extra: Any) -> dict[str, Any]:
        engine = gen.usage_from(TokenSource.ENGINE)
        sample = {
            "tag": tag,
            "in": engine.in_tokens if engine else None,
            "out": engine.out_tokens if engine else None,
            "thinking_out": engine.thinking_tokens if engine else None,
            "text_chars": len(gen.text),
            "thinking_chars": len(gen.thinking),
            "finish": str(gen.finish_reason),
            "prompt_eval_ms": gen.latency.ms("prompt_eval") if gen.latency else None,
            "eval_ms": gen.latency.ms("eval") if gen.latency else None,
            "load_ms": gen.latency.ms("load") if gen.latency else None,
            "ttft_ms": gen.ttft_ms,
            **extra,
        }
        self.samples.append(sample)
        return sample

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        """直接打引擎端点（探针专用，例如对比 /v1）。业务代码不许走这条路。"""
        return self.provider.client.post_json(path, payload)


class ProbeSuite:
    """按名字选择探针并顺序执行。单 GPU 独占 ⇒ 探针必须串行。"""

    def __init__(self, names: Iterable[str]) -> None:
        unknown = [n for n in names if n not in _REGISTRY]
        if unknown:
            raise KeyError(f"未注册的探针: {unknown}，可用: {registered_probes()}")
        self.names = list(names)

    def run(self, ctx: ProbeContext) -> ProbeReport:
        started = ctx.clock.wall_iso()
        findings: list[ProbeFinding] = []
        for name in self.names:
            findings.append(_REGISTRY[name](ctx))
        return ProbeReport(
            provider_id=getattr(ctx.provider, "id", "?"),
            findings=tuple(findings),
            started_at=started,
            finished_at=ctx.clock.wall_iso(),
        )


def run_suite(
    provider: Any,
    model: str,
    names: Sequence[str] = (),
    *,
    provider_version: str = "",
) -> tuple[ProbeReport, ProbeContext]:
    ctx = ProbeContext(provider=provider, model=model, provider_version=provider_version)
    suite = ProbeSuite(names or registered_probes())
    return suite.run(ctx), ctx


def render_markdown(report: ProbeReport, ctx: ProbeContext) -> str:
    """渲染成可直接追加进 docs/PROBES.md 的段落。"""
    lines = [
        f"## 探针运行 · {report.provider_id} · `{ctx.model}`",
        "",
        f"- 引擎版本：{ctx.provider_version or '未知'}",
        f"- 运行时间：{report.started_at} → {report.finished_at}",
        f"- 样本调用数：{len(ctx.samples)}",
        "",
        "| 探针 | 结论 | 未知 | 证据 |",
        "|---|---|---|---|",
    ]
    for f in report.findings:
        evidence = "; ".join(f"{k}={v}" for k, v in list(f.evidence.items())[:4])
        lines.append(
            f"| `{f.probe}` | {f.verdict} | {'是' if f.unknown else '否'} | {evidence} |"
        )
    lines += ["", "<details><summary>原始样本</summary>", "", "```json"]
    import json

    lines.append(json.dumps(ctx.samples, ensure_ascii=False, indent=2, default=str))
    lines += ["```", "", "</details>", ""]
    return "\n".join(lines)
