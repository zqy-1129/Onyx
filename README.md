# Onyx

本地大模型管理看板：**token 计量 · 工具调用观测 · 数据集评测 · 工具测试**。

单 GPU 自托管场景下，回答四类问题：

1. 每次调用到底花了多少输入/输出 token？这个数字是引擎报的、还是我自己复算的？两者差多少？
2. 模型的工具调用是真会调，还是格式坏了 / 幻觉了工具名 / 参数错了？
3. 换模型之后，意图识别与工具调用能力是涨了还是掉了，置信度多少？
4. 我的工具库每次请求偷走多少上下文？哪个工具最贵？

设计文档：[`docs/DESIGN.md`](docs/DESIGN.md) · 分步实现方案：[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md)

## 核心设计取舍

- **单一咽喉点**：所有模型调用必经 `llm/gateway.py`，观测/评测/Playground 都只是它的消费者。由 `import-linter` 的 `no-direct-http` 契约强制，不是口头约定。
- **测量必须带出处**：每个 token 数字都带 `source`（engine / hf_tokenizer / gguf_vocab / fitted / compat / heuristic）与 `confidence`。多来源并列入库、计算 drift，超阈值报异常。**没有出处的数字不上看板。**
- **不引入 LangChain**：本产品的价值在数据边界上（引擎的纳秒时序、原始文本、`/api/ps`、GGUF 元数据），而这正是框架抽象会抹掉的东西。详见 DESIGN §12。
- **core 零第三方依赖**：领域层只用 stdlib，由契约强制。换掉 FastAPI / 存储引擎不需要动领域模型。

## 快速开始

```bash
uv sync --extra dev --extra runtime
uv run pytest -q            # 单元 + 契约测试（无网络、无模型）
uv run lint-imports         # 架构边界检查
uv run onyx doctor          # 环境体检
uv run onyx db init         # 初始化 .data/onyx.sqlite
```

## 实现进度

| 里程碑 | 状态 | 内容 |
|---|---|---|
| M0 环境 | ✅ | uv + Python 3.12 + git（Ollama 0.35.0 / Node 24 已就绪） |
| M1 计量 | ✅ | `core/` 领域层 · `store/` 存储层 · Ollama 适配器 · token 保真阶梯与双特征标定 · gateway 单一咽喉点 · 观测引擎与 visitors · 能力矩阵 · CLI（chat / traces / models / probe / calibrate / doctor） |
| M2 看板 | ✅ | REST + SSE · Fleet / Models / Traces / TraceDetail / Token Ledger / Playground 六页，已在真实浏览器实测（真机发送到 qwen3.5:9b，引擎计数与冷启动标注齐全，零 console 错误） |
| M3 工具 | ✅ | 注册表（内容 hash 版本化 + 契约审计 + 上下文开销核算）· 执行层（python_fn / mock_replay / http 三种执行器 + 沙箱 + 契约矩阵）· 客户端工具循环（预算 / 熔断 / 孤儿补齐）· fire-and-verify 六种判定，真机 qwen3.5:9b 端到端 PASS |
| M4 评测 | 🚧 S13 完成 · S14 待做 | 评测内核（task/grade/runner + 手算可验的指标层 + bootstrap CI）· 5 个评分器 · 236 条中文意图数据集 · `intent_classification` 真机跑通 |
| M5 对比 | ⬜ | 矩阵、回归 diff、报告导出 |
| M6 扩展 | ⬜ | 插件 entry points、第二 provider、MCP 执行器 |

**M3 执行层的核心保证**（`onyx tools contract`，离线、零真实网络）：
8 条契约断言在 3 个执行器上全部适用并通过（各列的 n/a 都写明原因）；
失败被强制分成 6 种互不相同的 kind —— `arg_error`（模型的错）/ `rejected`（策略）/
`timeout` / `unknown_tool`（路由）/ `skipped`（mock 配置）/ `error`（工具坏了），
因为种类一旦混淆，"工具调不对"就再也无法归因。

**M3 模型侧的核心保证**（`onyx tools fire`）：判定分六种 —— `PASS` / `NO_CALL`（改提示词）/
`WRONG_TOOL`（改工具区分度）/ `BAD_ARGS`（改参数描述或加 max_tokens）/
**`TOOL_FAILED`（改工具，不是改模型）** / `LOOP_BROKEN`（改工具返回值）/ `ERROR`（跑 doctor）。
`pass_rate` 的分母排除 `TOOL_FAILED` 与 `ERROR`：否则模型要替坏掉的工具和挂掉的引擎背锅。
循环还守着一条不变式：**带 N 个 tool_calls 的 assistant 消息，后面必须紧跟恰好 N 条
tool 消息**——少一条，之后每次请求的上下文都永久错位，而引擎通常不报错，只是开始答非所问。

**M4 评测的两条口径纪律**：
- **内容与格式分开报**（DESIGN §9.4）。API-only 拿不到受约束 logprob，只能生成式打分，
  模型会因为"输出格式不听话"额外掉分。混成一个正确率就会把格式问题读成能力问题，
  而前者改提示词就能修、后者要换模型。
- **未知显示「—」，绝不显示 0**。没有可判定样本时 `macro_f1` 是未定义而不是 0 分；
  零除一律返回 `None`；`n<100` 必须标 ⚠低样本——20 条全对时 bootstrap 会给出
  `[1.000–1.000]` 的**退化区间**，那不是"置信度 100%"。

真机实测（qwen3.5:9b，236 条自建中文意图集，2026-10-03）：
`macro_f1 0.991 [95% CI 0.978–1.000]` · `acc 0.992` · `format_valid 1.000` ·
`out_of_label 0.000` · 236/236 完成、0 错误、38 秒、20,395 in / 521 out token。
`--limit 20` 与 `--limit 200` 的 CI 宽度分别为 0.000（退化，已标低样本）与 0.026，
证明区间真的在算而不是返回常数。

**M1 已在真机达成**：`onyx chat` 一次对话即落库完整 trace —— 引擎计数（in=19/out=47，
source=engine，confidence=high）、分段归因（`msg:0=8 + template_ctl=11 == 19`，
残差与标定截距 10.9958 互相验证）、prefill 冷/热判定、decode TPS、GPU 快照、原始 body 可 replay。

实测结论见 [`docs/PROBES.md`](docs/PROBES.md)（P1–P21，每条带证据与引擎版本）。

下一步（S9 收尾）：Playground 页（多模型并排、thinking 分栏、工具面板、SSE 实时增量）与 Token Ledger 页。

## 前端

设计语言见 [`docs/UI_DESIGN.md`](docs/UI_DESIGN.md)（对标 Grafana / Datadog）。七条硬规则里最关键的两条：
**R1 每个数字必须带出处徽标**（`engine/high` 与 `heuristic/low` 的 1842 是两个完全不同的东西）、
**R2 未知显示「—」绝不显示 0**（0 是测量值，「没测出来」不是）。

```bash
cd onyx/web && npm install
npm run dev        # http://localhost:5173（/api 代理到后端，默认 8787）
npm run test       # vitest：格式化与徽标语义
npm run build      # tsc + vite build
```
技术选型：Vite + React + TS + **手写 CSS 设计 token**，不引 Tailwind / 组件库 / 图表库。
高密度看板的价值在像素级控制（28px 行高、tabular-nums、1px 分隔线），
而图表形态固定，手写 SVG 比引 400KB 图表库更可控。产物 172KB JS / 12KB CSS。

## 目录

```
onyx/core/     L0 领域层：类型、可排序 id、时钟、事件契约、错误族、内容寻址存储（零三方依赖）
onyx/store/    L0 存储层：SQLite/WAL、幂等迁移、repo、异步批量 sink
onyx/cli.py    命令行入口
docs/          设计文档与分步实现方案
tests/unit/    无 IO、无网络、无模型的纯函数测试
```
