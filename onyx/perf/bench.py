"""采集与聚合：一次 perf 运行的执行侧。

数字只有两个来源，而且必须标着来源：引擎回报的 token 数与纳秒分段（复用
`llm.measurement.reconciler.latency_summary()` 那一套现成口径），或者墙钟。
把两者混进同一列，得到的是一个谁都没法复查的数——它既不是引擎自报的稳态吞吐，
也不是端到端体验，而"这个数从哪来"恰好是本项目最该有答案的问题。

三条不许漂的判据：
1. **引擎回报的 `in_tokens` 低于正文的汉字下限 ⇒ 这一格不记分**（记 `skipped` 带原因）。
   S32 实测过那个形状：16.8k tok 的正文被 Ollama 报成 `in_tokens=2050`。自称 8k 的格子
   实际量的是 2k 的吞吐，而数字看起来完全合理。
2. **预算用完 ⇒ 剩下的格子记 `skipped`，不补 0**。"没测到"与"测出来是 0"是两个相反的事实。
3. **冷启动只在真的能 `unload` 时才测**。做不到就明说，不许用"第一次请求"冒充冷启动——
   那个数里可能混着别人留下的缓存，读的人会以为它更冷。

请求一律经 `onyx.llm.gateway`（唯一咽喉点）：每发都留一条真 trace，基线上的每个数字点得回去。
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from onyx.core.clock import SYSTEM_CLOCK
from onyx.core.types import Generation, GenerationRequest, TokenSource, TraceContext, TracePurpose
from onyx.llm.measurement.heuristic import min_prompt_tokens
from onyx.llm.measurement.stats import spread_or_none
from onyx.perf.corpus import prompt_for
from onyx.perf.spec import BenchPlan, Cell

MAX_REASON_CHARS = 300
#: 指纹字段。**刻意不含 `app_version` / `git_rev`**：换代码正是基线要对比的对象，
#: 把它当"不可比"就等于这条命令永远给不出结论。
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "provider_id", "engine_version", "model", "quantization", "device",
    "num_ctx", "keep_alive", "stream", "timing_source", "temperature", "seed", "grid",
)
#: 认不出这三个就没有"这是哪台引擎的哪个模型"，两次运行不许声称可比
IDENTITY_FIELDS: tuple[str, ...] = ("provider_id", "engine_version", "model")


@dataclass(frozen=True, slots=True)
class Sample:
    """一发请求的全部事实。`None` = 没测到，不是 0。"""

    cell: str
    phase: str
    wave: int
    trace_id: str = ""
    ok: bool = True
    error: str = ""
    in_tokens: int | None = None          # 引擎回报
    out_tokens: int | None = None         # 引擎回报
    ttft_ms: float | None = None
    wall_ms: float | None = None
    decode_tps: float | None = None
    prefill_tps: float | None = None
    prefill_mode: str = "unknown"
    load_ms: float | None = None
    #: 引擎有没有给出纳秒分段。它决定 `timing_source`，而后者进指纹。
    ns_timed: bool = False
    #: 正文被引擎裁过（见模块头的判据 1）：这条样本不进任何聚合值。
    truncated: bool = False

    @classmethod
    def of(cls, cell: Cell, wave: int, prompt: str, *, trace_id: str = "",
           generation: Generation | None = None, latency: dict[str, Any] | None = None,
           error: str = "") -> Sample:
        if generation is None:
            return cls(cell=cell.key, phase=cell.phase, wave=wave, trace_id=trace_id,
                       ok=False, error=error[:MAX_REASON_CHARS])
        engine = generation.usage_from(TokenSource.ENGINE)
        in_tokens = engine.in_tokens if engine else None
        latency = latency or {}
        floor = min_prompt_tokens(prompt)
        # TTFT 取**请求侧给的**那个值，而不是 latency 字典里的：观测层现在从来没有写入
        # `TraceState.ttft_ms`（真机实测：库里 2318 条 usage 行，ttft_ms 全为 NULL），
        # 所以字典里那一路永远是 None。这是已登记的观测缺陷（见 docs/STATUS.md），
        # 基线等不起它修好——`gen.ttft_ms` 是流式缝合当场算出来的原始值，出处更近。
        ttft = generation.ttft_ms if generation.ttft_ms is not None else latency.get("ttft_ms")
        return cls(
            cell=cell.key, phase=cell.phase, wave=wave, trace_id=trace_id,
            ok=generation.status.value == "ok", error=(generation.error or "")[:MAX_REASON_CHARS],
            in_tokens=in_tokens, out_tokens=engine.out_tokens if engine else None,
            ttft_ms=ttft, wall_ms=generation.wall_ms,
            decode_tps=latency.get("decode_tps"), prefill_tps=latency.get("prefill_tps"),
            prefill_mode=str(latency.get("prefill_mode") or "unknown"),
            load_ms=latency.get("load_ms"),
            ns_timed=bool(generation.latency and generation.latency.eval_ns),
            # 只有"引擎报了数"才谈得上被裁：没报数就是不知道，不武断判裁
            truncated=bool(floor and in_tokens is not None and in_tokens < floor),
        )


@dataclass(frozen=True, slots=True)
class CellResult:
    cell: Cell
    status: str                       # measured | skipped | error
    reason: str
    samples: tuple[Sample, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchOutcome:
    plan: BenchPlan
    status: str                       # done | partial | error
    conditions: dict[str, Any]
    env_hash: str
    comparable: bool
    cells: tuple[CellResult, ...]
    elapsed_s: float

    @property
    def n_requests(self) -> int:
        return sum(len(item.samples) for item in self.cells)

    @property
    def unmeasured(self) -> tuple[CellResult, ...]:
        return tuple(item for item in self.cells if item.status != "measured")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "env_hash": self.env_hash,
            "comparable": self.comparable,
            "conditions": dict(self.conditions),
            "elapsed_s": round(self.elapsed_s, 3),
            "n_requests": self.n_requests,
            "plan": self.plan.as_dict(),
            "cells": [
                {
                    "key": item.cell.key, "status": item.status, "reason": item.reason,
                    "n_samples": len(item.samples),
                    "trace_ids": [s.trace_id for s in item.samples if s.trace_id],
                    "metrics": item.metrics,
                }
                for item in self.cells
            ],
        }


Progress = Callable[[int, int, str], None]


# ── 采集 ──────────────────────────────────────────────────────────
def collect(
    gateway: Any,
    plan: BenchPlan,
    *,
    device: str = "",
    clock: Any = SYSTEM_CLOCK,
    on_progress: Progress | None = None,
    unload: Callable[[str], None] | None = None,
    engine_info: dict[str, Any] | None = None,
) -> BenchOutcome:
    """按网格发请求并聚合成格。

    `unload` / `engine_info` 是注入位：真机由 CLI 接 provider 的能力，测试给假的。
    刻意不在这里"探测不到就自己想办法"——做不到就记 `skipped` 并写清原因；
    静默替代会往基线里塞一个没有依据的数。

    计时走项目的 `Clock` 抽象（与 gateway/obs 同一个口径），所以测试里的 `FakeClock`
    能把"一批跑了多久"精确成可复现的值，而不是取决于机器快慢。
    """
    def now() -> float:
        return clock.monotonic_ns() / 1e9

    cells = plan.cells()
    total = sum(cell.n_requests for cell in cells)
    started = now()
    deadline = started + plan.budget_s
    done = 0
    results: list[CellResult] = []
    for position, cell in enumerate(cells):
        if now() >= deadline:
            # 预算是"这一格还发不发"的单位：已经开跑的批次跑完再收手，
            # 中途撤掉已发出的请求只会留下一条永远不结束的 trace
            for rest in cells[position:]:
                results.append(CellResult(rest, "skipped",
                                          f"预算 {plan.budget_s:.0f}s 用尽，这一格一条没发", ()))
                done += rest.n_requests
            break
        if cell.phase == "cold":
            skipped = _prepare_cold(plan, cell, unload)
            if skipped is not None:
                results.append(skipped)
                done += cell.n_requests
                continue
        cell_result, sent = _run_cell(gateway, plan, cell, now, deadline)
        results.append(cell_result)
        done += sent
        if on_progress is not None:
            on_progress(done, total, cell.key)

    conditions = conditions_of(plan, gateway=gateway, device=device,
                               engine_info=engine_info, cells=results)
    statuses = {item.status for item in results}
    status = ("done" if statuses == {"measured"}
              else "error" if statuses == {"error"}
              else "partial")
    elapsed = now() - started
    return BenchOutcome(
        plan=plan, status=status, conditions=conditions,
        env_hash=fingerprint(conditions), comparable=fingerprint_ok(conditions),
        cells=tuple(results), elapsed_s=elapsed,
    )


def _prepare_cold(plan: BenchPlan, cell: Cell,
                  unload: Callable[[str], None] | None) -> CellResult | None:
    """冷启动前先真的把模型卸掉。做不到就整格记 `skipped`，不降级成"warm 当 cold"。"""
    if unload is None:
        return CellResult(cell, "skipped", "这个通道没有卸载接口 ⇒ 冷启动这一格不作数", ())
    try:
        unload(plan.model)
    except Exception as exc:  # noqa: BLE001 - 见上：宁可缺一格，不要一个来路不明的数
        return CellResult(cell, "skipped",
                          f"没能卸载模型（{type(exc).__name__}: {str(exc)[:120]}）⇒ 冷启动这一格不作数",
                          ())
    return None


def _run_cell(gateway: Any, plan: BenchPlan, cell: Cell,
              now: Callable[[], float], deadline: float) -> tuple[CellResult, int]:
    prompt = prompt_for(cell.prompt_chars)
    floor = min_prompt_tokens(prompt)
    samples: list[Sample] = []
    walls: list[float] = []
    for wave in range(cell.repeat):
        if wave and now() >= deadline:
            break
        started = now()
        samples.extend(_fire_wave(gateway, plan, cell, prompt, wave))
        walls.append(now() - started)

    measured = [s for s in samples if s.ok and not s.truncated]
    truncated = [s for s in samples if s.truncated]
    errors = [s for s in samples if not s.ok and not s.truncated]
    if not samples:
        status, reason = "skipped", "一条没发（预算在上一格里用完了）"
    elif truncated and not measured:
        reported = next((s.in_tokens for s in truncated if s.in_tokens is not None), None)
        status = "skipped"
        reason = (f"引擎回报 in_tokens={reported}，而这串正文的汉字下限至少 {floor}"
                  " ⇒ 正文被裁，这一格量的不是声明的长度（要测它请加 --num-ctx）")
    elif errors and not measured:
        status, reason = "error", errors[0].error or "全部请求失败"
    else:
        status = "measured"
        reason = (f"{len(truncated)} 发因正文被裁剔除，其余仍计入聚合" if truncated else
                  f"{len(errors)} 发失败，其余仍计入聚合" if errors else "")
    return (CellResult(cell, status, reason, tuple(samples), _metrics(measured, samples, walls)),
            len(samples))


def _fire_wave(gateway: Any, plan: BenchPlan, cell: Cell, prompt: str, wave: int) -> list[Sample]:
    req = GenerationRequest.of(
        plan.model, prompt, stream=plan.stream, keep_alive=plan.keep_alive,
        context=TraceContext(extra={"perf_cell": cell.key, "perf_wave": wave}),
    ).with_params(temperature=plan.params.get("temperature", 0.0),
                  max_tokens=cell.target_tokens,
                  **({"num_ctx": plan.num_ctx} if plan.num_ctx else {}))

    def one() -> Sample:
        try:
            result = gateway.generate(req, purpose=TracePurpose.BENCH)
        except Exception as exc:  # noqa: BLE001 - 一发失败不该让整格消失，但要留下失败形状
            return Sample.of(cell, wave, prompt,
                             error=f"{type(exc).__name__}: {str(exc)[:MAX_REASON_CHARS]}")
        return Sample.of(cell, wave, prompt, trace_id=result.trace_id,
                         generation=result.generation, latency=result.latency)

    if cell.concurrency <= 1:
        return [one()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=cell.concurrency) as pool:
        return list(pool.map(lambda _: one(), range(cell.concurrency)))


# ── 聚合 ──────────────────────────────────────────────────────────
def _metrics(measured: Sequence[Sample], all_samples: Sequence[Sample],
             wave_walls: Sequence[float]) -> dict[str, Any]:
    """每列都带 n；没有值的列是 None（渲染成「—」），不是 0。"""
    return {
        "n_requests": len(all_samples),
        "n_measured": len(measured),
        "n_error": sum(1 for s in all_samples if not s.ok and not s.truncated),
        "n_truncated": sum(1 for s in all_samples if s.truncated),
        "prompt_tokens": spread_or_none([s.in_tokens for s in measured if s.in_tokens]),
        "out_tokens": spread_or_none([s.out_tokens for s in measured if s.out_tokens]),
        "ttft_ms": spread_or_none([s.ttft_ms for s in measured if s.ttft_ms is not None]),
        "wall_ms": spread_or_none([s.wall_ms for s in measured if s.wall_ms is not None]),
        "decode_tps": spread_or_none([s.decode_tps for s in measured if s.decode_tps]),
        "load_ms": spread_or_none([s.load_ms for s in measured if s.load_ms]),
        # prefill 吞吐按冷热分列：混在一起的 P50 既不代表冷启动也不代表稳态（口径来自 reconciler）
        "prefill_tps_warm": spread_or_none(
            [s.prefill_tps for s in measured if s.prefill_mode == "warm" and s.prefill_tps]),
        "prefill_tps_cold": spread_or_none(
            [s.prefill_tps for s in measured if s.prefill_mode == "cold" and s.prefill_tps]),
        "n_prefill_mode_unknown": sum(1 for s in measured if s.prefill_mode == "unknown"),
        "aggregate_tps": _aggregate_tps(measured, wave_walls),
    }


def _aggregate_tps(measured: Sequence[Sample], wave_walls: Sequence[float]) -> dict[str, Any] | None:
    """一批的合计吞吐 = 该批 out_tokens 之和 / 该批墙钟。

    与 `decode_tps` 回答的是两个问题：前者是"这张卡一秒能吐多少 token"，
    后者是"一个请求多快"。并发下它们会分叉，只报一个就把"batching 到底赚不赚"糊掉了。
    一批里任何一发没有引擎回报的 out_tokens 就整批判为未知——少一路也照样算得出数，
    而那个数会被读成"同条件下的合计"。
    """
    by_wave: dict[int, list[Sample]] = {}
    for sample in measured:
        by_wave.setdefault(sample.wave, []).append(sample)
    rates: list[float] = []
    no_tokens = 0
    no_wall = 0
    for index, wall in enumerate(wave_walls):
        batch = by_wave.get(index, [])
        if not batch or wall <= 0:
            # 批墙钟为 0 只可能是时钟精度不够（Windows 上 time.monotonic 的分辨率可到 15.6ms）：
            # 这时候合计吞吐是"测不出来"，不是"无穷快"，所以整批判为未知并说清为什么
            no_wall += 1
            continue
        if any(s.out_tokens is None for s in batch):
            no_tokens += 1
            continue
        total_tokens = sum(s.out_tokens or 0 for s in batch)
        if total_tokens:
            rates.append(total_tokens / wall)
    notes = []
    if no_tokens:
        notes.append(f"{no_tokens} 批缺引擎回报的 out_tokens")
    if no_wall:
        notes.append(f"{no_wall} 批的墙钟短于时钟分辨率")
    summary = dict(spread_or_none(rates) or {})
    if notes:
        summary["note"] = "（" + "；".join(notes) + " ⇒ 未计入）"
    return summary or None


# ── 条件与指纹 ────────────────────────────────────────────────────
def conditions_of(plan: BenchPlan, *, gateway: Any, device: str,
                  engine_info: dict[str, Any] | None,
                  cells: Sequence[CellResult]) -> dict[str, Any]:
    """把"这次到底在什么条件下测"写成一份可对比的数据。

    `engine_version` 拿不到就留空串而不是猜一个：空串让 `fingerprint_ok()` 变 False，
    于是 compare 会拒绝——**认不出引擎就不许声称可比**，这比默默放行安全。
    """
    info = engine_info if engine_info is not None else _engine_info(gateway, plan.model)
    flat = [s for cell in cells for s in cell.samples]
    timing = ("engine_ns" if any(s.ns_timed for s in flat)
              else "wall_only" if any(s.wall_ms for s in flat) else "unknown")
    return {
        "provider_id": str(info.get("provider_id") or ""),
        "engine_version": str(info.get("version") or ""),
        "model": plan.model,
        "quantization": str(info.get("quantization") or ""),
        "device": device,
        "num_ctx": plan.num_ctx,
        "keep_alive": plan.keep_alive,
        "stream": plan.stream,
        "timing_source": timing,
        "temperature": plan.params.get("temperature", 0.0),
        "seed": plan.params.get("seed"),
        "grid": plan.grid_signature(),
        "app_version": str(info.get("app_version") or ""),
        "git_rev": str(info.get("git_rev") or ""),
    }


def _engine_info(gateway: Any, model: str) -> dict[str, Any]:
    """尽力从 provider 取"这是哪台引擎的哪个模型"；拿不到就留空串（不编）。"""
    out: dict[str, Any] = {"provider_id": "", "version": "", "quantization": ""}
    provider = getattr(gateway, "provider", None)
    if provider is None:
        return out
    out["provider_id"] = str(getattr(provider, "id", "") or "")
    info = getattr(provider, "info", None)
    if callable(info):
        with contextlib.suppress(Exception):
            out["version"] = str(getattr(info(), "version", "") or "")
    listing = getattr(provider, "list_models", None)
    if callable(listing):
        with contextlib.suppress(Exception):
            for card in listing():
                names = {getattr(card, "name", ""), getattr(card, "model", "")}
                if model in names:
                    out["quantization"] = str(getattr(card, "quantization", "") or "")
                    break
    return out


def fingerprint(conditions: dict[str, Any]) -> str:
    """只对 `FINGERPRINT_FIELDS` 取哈希：这些字段一变，两次结果就不是同一个实验。"""
    canonical = json.dumps(
        {key: conditions.get(key) for key in FINGERPRINT_FIELDS},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def fingerprint_ok(conditions: dict[str, Any]) -> bool:
    """三个身份字段是否都真的知道。`seed=None` / `num_ctx=None` 是**已知为"不限"**，算知道；
    空串才是"不知道"。混在一起的话，每条拿不到版本号的兼容通道都会被误判成不可信。
    """
    return all(str(conditions.get(key) or "").strip() for key in IDENTITY_FIELDS)


def diff_conditions(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    """逐字段列出两次运行的条件差异，只比指纹字段。

    报"哪个字段不同"而不是"条件不同"：后者是一句让人去猜的话，
    而换模型、换 ctx、换并发网格这三种情况的处理方式完全不同。
    """
    return [
        {"field": key, "a": a.get(key), "b": b.get(key)}
        for key in FINGERPRINT_FIELDS
        if json.dumps(a.get(key), sort_keys=True, ensure_ascii=False, default=str)
        != json.dumps(b.get(key), sort_keys=True, ensure_ascii=False, default=str)
    ]
