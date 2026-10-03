"""两次运行的配对对比（IMPLEMENTATION S15 / DESIGN §9.3）。

要回答的问题只有一个：**换个模型到底值不值**。而这个问题最容易答错的版本是
"比较两个平均值"——平均值差 0.02 可能是 30 条变好、28 条变坏相互抵消的结果，
那根本不是"略好"，而是"在两类任务上方向相反"。

所以这里的核心是**配对**：
1. 只取两个 run 都考过的 case（同一 `case_id`），差值在 case 层面算；
2. 报"净改善 / 净劣化 / 无变化"三条计数，而不是只报均值差；
3. 置信区间用**配对 bootstrap**：重采样的是 case，每次重采样都重新算均值差。
   常见错误是把两个 run 各自 CI 摆在一起看是否重叠——两条独立区间的重叠检验
   远比配对检验保守，n 小的时候几乎永远"不显著"，于是真回归被读成噪声。

比较之前必须先问"这是同一份考卷吗"：数据集不同、k 不同、温度不同，
得到的差值都不说明模型能力。这类问题一律写成 `warnings` 并带到出口（CLI/API/UI），
静默比较是这一步最坏的失败模式。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from onyx.eval.metrics import CI, jsonable, mean_ci
from onyx.eval.task import headline_of
from onyx.store.records import GradeRecord, RunRecord
from onyx.store.repos import EvalRepo

#: 低于这个配对数就别拿结论当决策依据（IMPLEMENTATION S15：UI 必须标低样本）
LOW_CONFIDENCE_PAIRS = 30


class CompareError(ValueError):
    """根本没法比：缺 run、任务不同、交集为空。"""


@dataclass(frozen=True, slots=True)
class PairedCase:
    """一个 case 在两个 run 里的配对结果。"""

    case_id: str
    kind: str
    score_base: float
    score_target: float
    delta: float
    #: k>1 时是"k 次全过"（pass^k）；None 表示有一边压根没判定（skip / 引擎失败）
    passed_base: bool | None
    passed_target: bool | None
    verdict_base: str
    verdict_target: str
    trace_base: str | None
    trace_target: str | None
    instruction: str = ""

    @property
    def direction(self) -> str:
        if self.delta > 0:
            return "improved"
        if self.delta < 0:
            return "regressed"
        return "unchanged"


@dataclass(frozen=True, slots=True)
class Comparison:
    base: RunRecord
    target: RunRecord
    paired: tuple[PairedCase, ...] = ()
    only_base: tuple[str, ...] = ()
    only_target: tuple[str, ...] = ()
    #: 判为"变化"的最小差值：小于它算无变化（浮点噪声与 k>1 的均值抖动都靠它挡住）
    eps: float = 0.0
    delta_ci: CI | None = None
    warnings: tuple[str, ...] = ()

    # ── 派生统计 ──────────────────────────────────────────────────
    @property
    def mean_delta(self) -> float | None:
        if not self.paired:
            return None
        return sum(item.delta for item in self.paired) / len(self.paired)

    @property
    def improved(self) -> int:
        return sum(1 for item in self.paired if item.delta > self.eps)

    @property
    def regressed(self) -> int:
        return sum(1 for item in self.paired if item.delta < -self.eps)

    @property
    def unchanged(self) -> int:
        return len(self.paired) - self.improved - self.regressed

    @property
    def flips(self) -> dict[str, int]:
        """二元口径上的翻转（McNemar 的两个不和谐格）。

        分数均值会被"部分分"抹平，而"这道题从会答变成不会答"是离散事件：
        只有按 pass^k 逐条配对，才数得出它。两个数一起看才知道净变化从哪来。
        """
        up = down = 0
        for item in self.paired:
            if item.passed_base is None or item.passed_target is None:
                continue
            if not item.passed_base and item.passed_target:
                up += 1
            elif item.passed_base and not item.passed_target:
                down += 1
        return {"up": up, "down": down, "net": up - down}

    @property
    def n_paired(self) -> int:
        return len(self.paired)

    @property
    def coverage(self) -> float | None:
        """配对上的 case 占两边并集的比例。低覆盖率意味着两份考卷差别很大。"""
        union = self.n_paired + len(self.only_base) + len(self.only_target)
        return self.n_paired / union if union else None

    @property
    def low_confidence(self) -> bool:
        return self.n_paired < LOW_CONFIDENCE_PAIRS

    @property
    def comparable(self) -> bool:
        return self.n_paired > 0

    def regressions(self, *, limit: int = 20) -> tuple[PairedCase, ...]:
        """劣化清单，按变小的幅度降序（最惨的在前）。"""
        worse = [item for item in self.paired if item.delta < -self.eps]
        return tuple(sorted(worse, key=lambda item: item.delta)[:limit])

    def improvements(self, *, limit: int = 20) -> tuple[PairedCase, ...]:
        better = [item for item in self.paired if item.delta > self.eps]
        return tuple(sorted(better, key=lambda item: -item.delta)[:limit])

    def as_dict(self) -> dict[str, Any]:
        """给 CLI/API 的通用形状。CI 必须是 dict，否则读回来是 repr 字符串。"""
        ci = self.delta_ci
        payload = {
            "base": _run_brief(self.base),
            "target": _run_brief(self.target),
            "eps": self.eps,
            "n_paired": self.n_paired,
            "only_base": list(self.only_base),
            "only_target": list(self.only_target),
            "coverage": self.coverage,
            "mean_delta": self.mean_delta,
            "delta_ci": None if ci is None else {
                "low": ci.low, "high": ci.high, "point": ci.point,
                "iterations": ci.iterations, "n": ci.n, "method": ci.method,
                # 覆盖 CI 自带的阈值：那个用的是单 run 的 n<100（metrics.LOW_CONFIDENCE_N），
                # 配对口径的门槛是 30。同一份载荷里两个"低置信"标记互相矛盾，
                # 界面就不知道该显示哪个了
                "low_confidence": self.low_confidence,
            },
            "improved": self.improved,
            "regressed": self.regressed,
            "unchanged": self.unchanged,
            "flips": self.flips,
            "low_confidence": self.low_confidence,
            "warnings": list(self.warnings),
            "cases": [jsonable(asdict(item)) for item in self.paired],
        }
        return jsonable(payload)


def per_case(grades: Sequence[GradeRecord]) -> dict[str, dict[str, Any]]:
    """把一个 run 的 grade 折成"每个 case 一条"。

    k>1 时分数取均值（那是这个 case 的能力估计），`passed` 取"k 次全过"
    ——pass^k 才是"可靠可用"，把 3 次里过了 2 次算成过就等于换了一个口径。
    """
    buckets: dict[str, list[GradeRecord]] = {}
    for grade in grades:
        buckets.setdefault(grade.case_id, []).append(grade)

    out: dict[str, dict[str, Any]] = {}
    for case_id, rows in buckets.items():
        passed_values = [row.passed for row in rows]
        # 有一边没判定 ⇒ 这个 case 的 pass^k 是"不知道"，不能悄悄当成 False
        passed = None if any(value is None for value in passed_values) else all(
            bool(value) for value in passed_values
        )
        verdicts = {row.verdict for row in rows}
        metrics = rows[0].metrics or {}
        out[case_id] = {
            "score": sum(row.score for row in rows) / len(rows),
            "passed": passed,
            "verdict": next(iter(verdicts)) if len(verdicts) == 1 else "mixed",
            "kind": str(metrics.get("kind") or "single"),
            # 下钻用第一次采样的 trace：k>1 时其余 trace 在 grade 行里仍可查
            "trace": next((row.trace_id for row in rows if row.trace_id), None),
            "n_samples": len(rows),
        }
    return out


def compare_runs(
    repo: EvalRepo, base_id: str, target_id: str,
    *, eps: float = 0.0, seed: int = 0, iterations: int = 2000,
) -> Comparison:
    """配对比较两个 run。差值方向永远是 **target − base**。"""
    base = _require(repo, base_id)
    target = _require(repo, target_id)
    if base.task_id != target.task_id:
        raise CompareError(
            f"任务不同（{base.task_id} vs {target.task_id}）：分数口径都不一样，"
            "所谓差值只是两个指标的差"
        )

    # 全量读取：这里曾经写过 limit=100_000"保护一下"，但那正是最坏的写法——
    # 超过上限的 grade 会被静默丢掉，配出来的差值看着正常，只是少了一批题
    a_cases = per_case(repo.list_grades(base_id))
    b_cases = per_case(repo.list_grades(target_id))
    shared = sorted(set(a_cases) & set(b_cases))
    # 题干来自 eval_case，不是 grade.metrics：任务把"期望/实际"写进 metrics 是各自的约定，
    # 而对比页要展示的是"哪道题变了"，那是样本的内容，属于数据集
    texts = case_texts(repo, base, target)

    paired = tuple(
        PairedCase(
            case_id=case_id,
            kind=str(a_cases[case_id]["kind"]),
            score_base=float(a_cases[case_id]["score"]),
            score_target=float(b_cases[case_id]["score"]),
            delta=float(b_cases[case_id]["score"]) - float(a_cases[case_id]["score"]),
            passed_base=a_cases[case_id]["passed"],
            passed_target=b_cases[case_id]["passed"],
            verdict_base=str(a_cases[case_id]["verdict"]),
            verdict_target=str(b_cases[case_id]["verdict"]),
            trace_base=a_cases[case_id]["trace"],
            trace_target=b_cases[case_id]["trace"],
            instruction=texts.get(case_id, ""),
        )
        for case_id in shared
    )
    if not paired:
        raise CompareError(
            f"两个 run 没有任何共同的 case（base {len(a_cases)} 条 / target {len(b_cases)} 条）："
            "大概是在不同数据集上跑的，配对差值无从谈起"
        )

    return Comparison(
        base=base, target=target, paired=paired,
        only_base=tuple(sorted(set(a_cases) - set(b_cases))),
        only_target=tuple(sorted(set(b_cases) - set(a_cases))),
        eps=eps,
        delta_ci=mean_ci([item.delta for item in paired], seed=seed, iterations=iterations),
        warnings=_comparability(base, target, a_cases, b_cases, paired),
    )


def _comparability(
    base: RunRecord, target: RunRecord,
    a_cases: dict[str, Any], b_cases: dict[str, Any], paired: Sequence[PairedCase],
) -> tuple[str, ...]:
    """把"这两次真的可比吗"的所有已知疑点写成文字，跟着结果一路带到界面。

    只警告不拒绝：数据集换了一版但 90% 重叠，工程师往往就是想看这个 diff——
    前提是那句话他看得见。
    """
    notes: list[str] = []
    if base.dataset_id and target.dataset_id:
        if base.dataset_id != target.dataset_id:
            notes.append(f"数据集不同：{base.dataset_id} vs {target.dataset_id}")
        elif base.dataset_revision and target.dataset_revision and \
                base.dataset_revision != target.dataset_revision:
            notes.append(
                f"同一数据集但版本不同：{base.dataset_revision} vs {target.dataset_revision}"
            )
    elif base.dataset_id or target.dataset_id:
        notes.append("至少一个 run 没记数据集来历（0005 之前的老 run），无法确认可比性")

    union = len(paired) + len(set(a_cases) ^ set(b_cases))
    if union and len(paired) / union < 0.95:
        notes.append(
            f"只有 {len(paired)}/{union} 个 case 被两边同时考到，"
            "结论只覆盖这部分样本"
        )

    for field_name, label in (("k", "采样次数 k"), ("split", "子集 split")):
        left, right = base.config.get(field_name), target.config.get(field_name)
        if left != right:
            notes.append(f"{label} 不同：{left!r} vs {right!r}")

    params_diff = sorted(
        key for key in set(base.params_snapshot) | set(target.params_snapshot)
        if base.params_snapshot.get(key) != target.params_snapshot.get(key)
    )
    if params_diff:
        notes.append(
            f"生成参数不同（{', '.join(map(str, params_diff))}）："
            "差值里混着温度的影响，不全是能力"
        )
    if base.seed is not None and target.seed is not None and base.seed != target.seed:
        notes.append(f"seed 不同：{base.seed} vs {target.seed}")
    if base.app_version and target.app_version and base.app_version != target.app_version:
        notes.append(f"代码版本不同：{base.app_version} vs {target.app_version}")
    for label, record in (("base", base), ("target", target)):
        if record.status != "done":
            notes.append(f"{label} 运行状态是 {record.status}，样本可能不完整")
    if len(paired) < LOW_CONFIDENCE_PAIRS:
        notes.append(
            f"配对样本只有 {len(paired)} 条（<{LOW_CONFIDENCE_PAIRS}），"
            "差值与区间都不足以支撑决策"
        )
    return tuple(notes)


def _require(repo: EvalRepo, run_id: str) -> RunRecord:
    run = repo.get_run(run_id)
    if run is None:
        raise CompareError(f"找不到 run {run_id!r}")
    return run


def case_texts(repo: EvalRepo, *runs: RunRecord) -> dict[str, str]:
    """取出参与对比的样本题干（按 case_id）。

    对比页要能读出"是哪道题变了"：只有 case_id 的清单没法得出"劣化的 8 条都是
    带日期参数的那种"这类结论，而那才是下一步要改的东西。
    """
    out: dict[str, str] = {}
    for run in runs:
        if not run.dataset_id:
            continue
        for record in repo.list_cases(run.dataset_id):
            out.setdefault(record.id, _text_of(record.input))
    return out


def _text_of(input: dict[str, Any]) -> str:
    for key in ("instruction", "query", "prompt", "text"):
        value = input.get(key)
        if isinstance(value, str) and value:
            return value[:160]
    for value in input.values():
        if isinstance(value, str) and value:
            return value[:160]
    return ""


def _run_brief(run: RunRecord) -> dict[str, Any]:
    headline = headline_of(run.aggregate)
    return {
        "id": run.id, "task_id": run.task_id, "model_id": run.model_id,
        "status": run.status, "n_cases": run.n_cases, "n_done": run.n_done,
        "n_error": run.n_error, "seed": run.seed, "started_at": run.started_at,
        "dataset_id": run.dataset_id, "dataset_revision": run.dataset_revision,
        "params": run.params_snapshot, "k": run.config.get("k"),
        # 与列表页/报告同一个优先级：两处各挑各的"主分数"会让人怀疑所有数字
        "headline": None if headline is None else {"metric": headline[0], "value": headline[1]},
        "cost": run.cost,
    }
