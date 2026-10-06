"""执行器契约矩阵的构建处（CLI 与 API 共用）。

放这里而不是留在 `cli.py`，是因为看板要显示同一张矩阵。两处各写一遍的话，
"CLI 说全过、界面说有一列挂了"这种分歧根本无法排查——而这张矩阵存在的意义
就是给出可比较的结论。

三个不许退回的老决定：
- **每一列的样本出处都要报出来**：各列测的定义可能不是同一个（http/mcp 自带离线样本），
  不写出处就会把"两个不同的东西都通过了"读成"执行器可替换"。
- **装不上的执行器是"未知"，不是"通过"**：`unavailable` 与 `pending` 各带原因。
- **"还剩哪些种类没实现"从注册表推导**：写死的清单会在实现完成之后继续宣称"未实现"，
  而那句话看起来总是合理的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from onyx.core.errors import ToolUnknown
from onyx.tools.contract import CONTRACT_NAMES, ContractResult, run_contracts, sample_args, summarize
from onyx.tools.spec import ToolDef


class MatrixSetupError(ValueError):
    """矩阵搭不起来（缺定义、参数不是 JSON 对象）。带可行动的修法，不给堆栈。"""


#: mock 执行器结构上不触达真实实现，deadline 无从生效——豁免必须写明原因（不许静默跳过）
MOCK_CONTRACT_EXEMPTIONS: dict[str, str] = {
    "timeout_is_reported": (
        "mock 执行器不导入也不调用真实实现，deadline 无从生效；"
        "这条保证由 python_fn 上的同名断言覆盖"
    )
}

#: 样本出处说明。列名 → 一句话，界面上直接显示，不重复推导
SAMPLE_SOURCE_NOTE: dict[str, str] = {
    "python_fn": "（注册表/内置）",
    "mock": "（注册表/内置）",
    "http": "（离线 MockTransport）",
    "mcp": "（离线假连接，不起子进程）",
    "mcp_stdio": "（真子进程 + 真管道）",
}


def http_contract_target() -> tuple[Any, ...] | None:
    """http 列的离线契约样本（MockTransport + `.invalid` 域名，零真实网络）。

    惰性导入：httpx 属于 `runtime` extra，没装时这一列必须显示"未安装"，
    而不是让整张矩阵崩掉。
    """
    try:
        from onyx.tools.executors.http import contract_target
    except ImportError:
        return None
    return contract_target()


def stdio_contract_target():
    """mcp_stdio 列：真子进程 + 真管道（惰性导入的理由与 mcp 列相同）。"""
    from onyx.tools.executors.mcp import stdio_contract_target as _target

    return _target()


def mcp_contract_target() -> tuple[Any, ...] | None:
    """mcp 列的离线契约样本（假连接：不起子进程、不碰任何真实 server）。

    惰性导入的理由与 http 一样：导入失败要显示"未知"，不能让整张矩阵崩掉。
    """
    try:
        from onyx.tools.executors.mcp import contract_target
    except ImportError:
        return None
    return contract_target()


def default_valid_args(definition: ToolDef, source: str = "builtin") -> dict[str, Any]:
    """没指定参数时构造一份合法参数。

    内置 echo 有专门的样本参数；但**注册表里的同名 echo 不走这条路**——
    用户显式导入的那份才是模型实际会看到的定义，参数必须从它自己的 examples 构造，
    否则矩阵测的是内置定义，而报告说的是注册版本。
    """
    from onyx.tools.builtin.defs import CONTRACT_SAMPLE_ARGS

    if definition.name == "echo" and source == "builtin":
        return dict(CONTRACT_SAMPLE_ARGS)
    return sample_args(definition)


@dataclass(frozen=True, slots=True)
class Matrix:
    tool: str
    source: str
    valid_args: dict[str, Any]
    columns: tuple[str, ...]
    samples: dict[str, str]
    results: dict[str, dict[str, ContractResult]]
    unavailable: dict[str, str] = field(default_factory=dict)
    pending: dict[str, str] = field(default_factory=dict)

    def counts(self) -> dict[str, dict[str, int]]:
        return {name: summarize(list(self.results[name].values())) for name in self.columns}

    @property
    def failed(self) -> int:
        return sum(counts["failed"] for counts in self.counts().values())

    def failures(self) -> list[dict[str, str]]:
        """所有没通过的格子（含"不适用"），每条带列名、断言与原因。"""
        out: list[dict[str, str]] = []
        for column in self.columns:
            for name in CONTRACT_NAMES:
                result = self.results[column].get(name)
                if result is None or result.passed:
                    continue
                out.append({
                    "executor": column, "assertion": name,
                    "status": "not_applicable" if not result.applicable else "failed",
                    "detail": result.detail,
                })
        return out

    def as_dict(self) -> dict[str, Any]:
        """机器可读形状。`onyx tools contract --json` 与 `GET /api/tools/matrix` 用的就是它，
        所以 CLI 与界面不可能报出两个版本的同一张矩阵。

        `samples` 保持"列名 → 样本定义名"这一原始形状（历史 --json 消费方依赖它），
        出处说明另放 `sample_notes`，不把结构变更伪装成"顺手整理"。
        """
        return {
            "tool": self.tool,
            "source": self.source,
            "assertions": list(CONTRACT_NAMES),
            "valid_args": self.valid_args,
            "samples": dict(self.samples),
            "sample_notes": {
                name: SAMPLE_SOURCE_NOTE.get(name, "（自带样本）") for name in self.columns
            },
            "executors": {
                name: {
                    item: {
                        "passed": self.results[name][item].passed,
                        "applicable": self.results[name][item].applicable,
                        "detail": self.results[name][item].detail,
                    }
                    for item in CONTRACT_NAMES if item in self.results[name]
                }
                for name in self.columns
            },
            "unavailable": self.unavailable,
            "pending": self.pending,
            "summary": self.counts(),
            "failed": self.failed,
        }


def build_matrix(
    definition: ToolDef | None,
    *,
    source: str = "builtin",
    valid_args: dict[str, Any] | None = None,
    tool_label: str = "",
) -> Matrix:
    """把同一套断言跑在所有**已实现**的执行器列上。"""
    from onyx.tools.executors import MockReplayExecutor, PythonFnExecutor

    if definition is None:
        from onyx.tools.builtin.defs import BUILTIN_DEFS

        names = ", ".join(item.name for item in BUILTIN_DEFS)
        raise ToolUnknown(
            f"找不到工具 {tool_label!r}；内置可选: {names}，或用 onyx tools import 先注册"
        )

    args = valid_args if valid_args is not None else default_valid_args(definition, source)

    # 每一列是 (标签, 样本定义, 工厂, 合法参数, fixtures, 豁免, synth, teardown)
    targets: list[tuple] = [
        ("python_fn", definition, PythonFnExecutor, args, None, {}, None, None),
        ("mock", definition, MockReplayExecutor, args,
         {definition.name: {"__contract_mock__": True, "tool": definition.name}},
         MOCK_CONTRACT_EXEMPTIONS, None, None),
    ]
    unavailable: dict[str, str] = {}
    http = http_contract_target()
    if http is None:
        unavailable["http"] = "未安装 httpx（uv sync --extra runtime）——未知，不是通过"
    else:
        http_sample, http_factory, http_args, http_synth = http
        targets.append(("http", http_sample, http_factory, http_args, None, {}, http_synth, None))
    mcp = mcp_contract_target()
    if mcp is None:
        unavailable["mcp"] = "MCP 执行器导入失败——未知，不是通过"
    else:
        mcp_sample, mcp_factory, mcp_args, mcp_synth = mcp
        targets.append(("mcp", mcp_sample, mcp_factory, mcp_args, None, {}, mcp_synth, None))
    #: 第五列：真子进程。它起不来时必须是"未知"，不能让整张矩阵崩掉——
    #: 人跑 `tools contract` 通常正在排查别的问题，再给他一个堆栈就是帮倒忙。
    try:
        stdio_sample, stdio_factory, stdio_args, stdio_synth, stdio_close = (
            stdio_contract_target())
    except ImportError:
        unavailable["mcp_stdio"] = "MCP 执行器不可导入——未知，不是通过"
    except Exception as exc:  # noqa: BLE001 - 起进程失败要变成"未知"，见上
        unavailable["mcp_stdio"] = (
            f"真 stdio server 起不来（{type(exc).__name__}: {str(exc)[:160]}）——未知，不是通过"
        )
    else:
        # 这一列**不需要豁免**：八条断言在真子进程上全部成立
        # （`mock_policy_makes_no_real_call` 在这里反而是最强的一档——执行器自己数真实调用）
        targets.append(("mcp_stdio", stdio_sample, stdio_factory, stdio_args,
                        None, {}, stdio_synth, stdio_close))

    # "还剩哪些种类没实现"从注册表推导，不在这里写死
    from onyx.tools.executors import PENDING_KINDS

    pending = {kind: milestone for kind, milestone in PENDING_KINDS.items()
               if kind not in unavailable}

    columns: list[str] = []
    samples: dict[str, str] = {}
    results: dict[str, dict[str, ContractResult]] = {}
    for name, sample, factory, column_args, fixtures, exemptions, synth, teardown in targets:
        try:
            column_results = run_contracts(
                factory, sample, valid_args=column_args,
                fixtures=fixtures, exemptions=exemptions, synth=synth,
            )
        finally:
            # 真 stdio 列带着一个子进程池：不管断言跑成什么样都要关掉。
            # 僵尸 server 是句柄泄漏，而 `tools contract` 正是人来查故障时第一个跑的命令。
            if teardown is not None:
                teardown()
        columns.append(name)
        samples[name] = sample.name
        results[name] = {item.name: item for item in column_results}

    return Matrix(
        tool=definition.name, source=source, valid_args=dict(args), columns=tuple(columns),
        samples=samples, results=results, unavailable=unavailable, pending=pending,
    )
