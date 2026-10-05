"""评测任务契约（DESIGN §9.1）。

`runner` 只做四件事：`build` → **gateway 发（自动被记录）** → `grade` → `aggregate`，
外加调度/断点续跑/取消。任务本身**不发请求**，这是"评测绝不建立第二条调用路径"
在类型上的体现：`build` 返回 `GenerationRequest`，而不是返回一个分数。

两条必须守住的口径：
1. `Grade.invalid_format` 与"内容答错"是**两个维度**（DESIGN §9.4）。
   API-only 拿不到受约束的 logprob，只能生成式打分，于是模型会因为
   "输出格式不听话"额外掉分。把两者混成一个正确率，就会把格式问题误读成能力问题。
2. `requires` 不满足时**必须 skip 并写明原因**，禁止隐式降级。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from onyx.core.types import Cap, Generation, GenerationRequest, ToolSpec

if TYPE_CHECKING:
    from onyx.eval.datasets.loader import Dataset


class Verdict(StrEnum):
    """单条样本的判定。**落库即契约**，只增不改。"""

    CORRECT = "correct"
    PARTIAL = "partial"
    WRONG = "wrong"
    #: 调了工具但选错。与 `NO_CALL`（该调却不调）必须分开：
    #: 前者改工具之间的描述区分度，后者改提示词与"什么时候该用"
    WRONG_TOOL = "wrong_tool"
    #: 工具选对了但参数不对。修的是参数 description / required / enum 说明，
    #: 与 WRONG_TOOL 完全是两件事
    BAD_ARGS = "bad_args"
    #: 输出不是要求的格式（JSON 坏了、多了前后缀）——与"答错"分开计
    INVALID_FORMAT = "invalid_format"
    #: 输出了一个标签集里不存在的名字。这是幻觉，不是选错
    OUT_OF_LABEL = "out_of_label"
    #: 调用了工具集里不存在的名字
    HALLUCINATED_TOOL = "hallucinated_tool"
    NO_CALL = "no_call"
    TIMEOUT = "timeout"
    REFUSED = "refused"
    ERROR = "error"
    #: 能力不满足而跳过。必须带原因，禁止静默降级（DESIGN §9.1）
    SKIPPED = "skipped"


#: 这些判定算"模型答对了"，用于 pass^k / pass@k
PASSING = frozenset({Verdict.CORRECT})
#: 这些判定是**格式/环境问题**，不该计入模型能力分母
NON_ATTRIBUTABLE = frozenset({Verdict.ERROR, Verdict.TIMEOUT, Verdict.SKIPPED})


@dataclass(frozen=True, slots=True)
class Case:
    """一条评测样本。`input` / `expect` 的形状由各任务自己定义。"""

    id: str
    input: dict[str, Any] = field(default_factory=dict)
    expect: dict[str, Any] = field(default_factory=dict)
    dataset_id: str = ""
    ord: int = 0
    #: single | multi_turn | multi_step | parallel | no_call_needed
    kind: str = "single"
    tools: tuple[ToolSpec, ...] = ()
    #: 工具返回值桩：评测期默认真实工具不执行（DESIGN §8.4）
    fixture: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Grade:
    """一条样本的一次评分。`seq` 是同一 case 的第几次采样（pass^k 用）。"""

    case_id: str
    score: float
    verdict: Verdict
    seq: int = 0
    passed: bool | None = None
    #: 格式是否合法。**独立于对错**：格式坏了但内容对，和格式对了但内容错，
    #: 修法完全不同（前者改提示词/停止词，后者改模型或改任务难度）
    invalid_format: bool = False
    #: 输出是否落在允许集合之外（幻觉标签/幻觉工具名）
    out_of_set: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)
    #: 每个分数都能点进一条真实 trace（DESIGN §15）。没有 trace_id 的分数不该被展示
    trace_id: str = ""
    error: str = ""
    judge_model_id: str = ""
    #: judge 自己的开销：judge 也是本地模型时同样吃 GPU 与时间，必须计入成本
    judge_usage: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict in PASSING

    @property
    def attributable(self) -> bool:
        """这条判定能不能算进模型能力分母。"""
        return self.verdict not in NON_ATTRIBUTABLE


@dataclass(frozen=True, slots=True)
class Skip:
    """一次跳过。**reason 不许为空**：静默 skip 会让分数看起来正常却少考了题。"""

    case_id: str
    reason: str
    missing: tuple[str, ...] = ()

    def as_grade(self) -> Grade:
        return Grade(
            case_id=self.case_id, score=0.0, verdict=Verdict.SKIPPED, passed=None,
            error=self.reason, extra={"missing": list(self.missing)},
        )


@runtime_checkable
class EvalTask(Protocol):
    id: str
    name: str
    #: 需要的能力位。不满足 → skip 并写明原因
    requires: frozenset[Cap]
    #: 该任务会产出的指标名（看板据此建列，不许跑完才知道有哪些指标）
    metric_names: tuple[str, ...]

    def load(self, *, split: str = "default", limit: int | None = None) -> Iterator[Case]: ...
    def build(self, case: Case) -> GenerationRequest: ...
    def grade(self, case: Case, sample: Generation) -> Grade: ...
    def aggregate(self, grades: Sequence[Grade], *, seed: int = 0) -> dict[str, Any]: ...


def check_capabilities(task: EvalTask, caps: Iterable[Cap]) -> Skip | None:
    """能力不足则返回带原因的 Skip，否则 None。

    这是**任务级**的检查（整个任务跑不了）。样本级的跳过由任务自己在 `load`/`grade`
    里判断——例如 `no_call_needed` 子集在不支持 tool_choice 的引擎上照样能跑。
    """
    have = frozenset(caps)
    missing = tuple(sorted(str(cap) for cap in task.requires - have))
    if not missing:
        return None
    return Skip(
        case_id=f"task:{task.id}",
        reason=(
            f"任务 {task.id} 需要能力 {missing}，当前 provider/模型不具备；"
            "不做隐式降级（用提示词模拟工具调用得到的分数无法与原生支持比较）"
        ),
        missing=missing,
    )


#: 主分数（headline）候选，按优先级。列表页、矩阵、导出报告**共用这一份**：
#: 同一个 run 在不同界面显示不同的"头号分数"会让人怀疑所有数字。
#: `score`（任务自己声明的主分数）排在 `pass_hat_k` 之前：后者是**稳定性**指标，
#: 把它当主分数等于用"稳不稳"顶掉"对不对"——新任务一旦同时产出两者就会显示错那一列。
HEADLINE_METRICS = ("macro_f1", "accuracy", "must_call_acc", "score", "pass_hat_k")


def headline_of(aggregate: dict[str, Any]) -> tuple[str, Any] | None:
    """返回 (指标名, 值)——第一个在该任务的聚合里出现的候选指标。

    注意取的是"**出现过**"而不是"非 None"：值为 None 表示这个指标算不出来
    （没有可判定样本），它仍然是这个任务的主分数，必须原样交给上层去显示「—」。
    跳过 None 去选下一个候选，就等于把"没考到"显示成"另一个指标得了分"。
    """
    for key in HEADLINE_METRICS:
        if key in aggregate:
            return key, aggregate[key]
    return None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """任务登记项：怎么构造它，以及默认数据集从哪来。

    这是 `onyx.tasks` 扩展点的**交接形状**（DESIGN §13）：外部包既可以交一个
    `TaskSpec`（自带数据集载入器），也可以只交一个 `EvalTask` 实现类
    （此时 `dataset=None`，跑的时候必须显式 `--dataset`）。

    没有默认数据集时**必须报错**而不是猜一个：静默退回某个内置数据集，
    等于用另一份数据评测却报着同一个任务名，分数从此不可解释。
    """

    factory: Callable[..., EvalTask]
    dataset: Callable[[], Dataset] | None = None

    def default_dataset(self, task_id: str) -> Dataset:
        if self.dataset is None:
            raise KeyError(
                f"任务 {task_id!r} 没有默认数据集；请用 --dataset file:<路径> 指定，"
                "或先 onyx eval import 导入再按 id 引用"
            )
        return self.dataset()


def coerce_task_spec(name: str, obj: Any) -> TaskSpec:
    """把 entry point 加载到的对象规约成 `TaskSpec`。

    只接受两种形状：`TaskSpec`，或直接实现 EvalTask 协议的类。其它形状必须被拒
    并说清期望是什么——插件作者最需要的是"我交的东西为什么不算任务"，
    而不是一个消失在日志里的 warning。
    """
    if isinstance(obj, TaskSpec):
        return obj
    if isinstance(obj, type):
        missing = [attr for attr in ("load", "build", "grade", "aggregate")
                   if not hasattr(obj, attr)]
        if missing:
            raise TypeError(
                f"onyx.tasks 插件 {name!r} 不是 EvalTask：缺少 {', '.join(missing)}()。"
                "请交一个实现 EvalTask 协议的类，或 TaskSpec(factory=..., dataset=...)"
            )
        return TaskSpec(obj)
    raise TypeError(
        f"onyx.tasks 插件 {name!r} 加载得到 {type(obj).__name__}，"
        "期望 EvalTask 实现类或 TaskSpec 实例"
    )
