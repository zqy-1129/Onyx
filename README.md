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
| M4 评测 | ✅ | 评测内核（task/grade/runner + 指标层 + bootstrap CI）· 6 个评分器 + 类型感知参数比对 · 236 条中文意图集 + 97 条工具调用集 · `intent_classification` 与 `tool_selection` 各一次真机运行 · BFCL 导入器 · GPU 独占锁（跨进程 + 心跳 + ETA），eval/Playground/live 测试互相排队 |
| M5 对比 | ✅ | 模型 × 任务矩阵 + 配对回归 diff（净改善/净劣化 + 配对 bootstrap CI + 劣化清单）· Eval / 矩阵 / 回归三页已在真实浏览器实测 · md/csv/自包含 html 报告导出 · 每次运行带数据集来历 |
| M6 扩展 | ✅ | 六个扩展点接 entry points（一处实现语义：坏插件隔离 + 失败可见 + 同名覆盖可查）· 两个真外部插件样板（任务 / provider）· 第二 provider：OpenAI 兼容通道（vLLM / LM Studio / Ollama `/v1`）· MCP 执行器（stdio + JSON-RPC，纯 stdlib）· OTLP 导出 sink · provider/sink/插件三套契约测试 · `scripts/check_extension_boundary.py` 把"接实现不改内核"变成构建门禁 |

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

**M4 评测的三条口径纪律**：
- **内容与格式分开报**（DESIGN §9.4）。API-only 拿不到受约束 logprob，只能生成式打分，
  模型会因为"输出格式不听话"额外掉分。混成一个正确率就会把格式问题误读成能力问题，
  而前者改提示词就能修、后者要换模型。
- **「未知」与「0 分」是两个不同的事实，两边都不许互相冒充**。没有可判定样本时 `macro_f1` 是未定义
  而不是 0 分，零除一律返回 `None`，界面显示「—」（UI_DESIGN R2）；反过来，**算得出来的 0 必须写成 0**——
  `P=R=0` 是"全错"而不是"没考到"，早期把它判成未定义会让最差的类从宏平均里整个消失，
  模型越差 macro_f1 反而越高。`n<100` 必须标 ⚠低样本：20 条全对时 bootstrap 会给出
  `[1.000–1.000]` 的**退化区间**，那不是"置信度 100%"。
- **放宽规则与重采样单位都必须可见**。参数比对按类型走不同规则（数值容差/日期归一/集合/模糊），
  所以 `exact_rate`（字面就一致）要与 `relaxed_share`（靠放宽挣来的占比）一起报——只报前者会把
  "我们把比对放宽了"藏进分数里，看起来像模型变强了。CI 的重采样单位是 **case** 而不是 sample：
  temperature=0 下同一 case 的 k 次采样不是 k 个独立观测，按 sample 算会让区间窄得像"模型很确定"。

真机实测（qwen3.5:9b，2026-10-03；两个任务走同一个 gateway ⇒ 每个分数都能跳到真实 trace）：

| task | 数据 | 头号分数 | 95% CI | 成本 |
|---|---|---|---|---|
| `intent_classification` | 236 条自建中文意图集 · k=1 | `macro_f1 0.991` · acc 0.992 · format_valid 1.000 · out_of_label 0.000 | [0.978–1.000] | 20,395 in / 521 out · 37s · 0 错误 |
| `tool_selection` | 97 条自建工具调用集 · k=3 | `must_call_acc 0.639` · set_f1 0.986 · hallucinated 0.000 · **false_call 0.000** · args_exact 0.684（relaxed 0.000）· pass^3 0.732 | [0.528–0.736]（⚠ 97 case） | 401,739 in / 23,265 out · 13.1min · 0 错误 |

`--limit 20` 与 `--limit 200` 的意图 CI 宽度分别是 0.000（退化，已标低样本）与 0.026，
证明区间真的在算而不是返回常数。工具那行的结论是：**选工具几乎没错，掉分全在参数上**
（291 次采样里 `bad_args` 69 次，而 `hallucinated_tool` 与误调率都是 0——在场 97 次的
`send_email` 一次都没被用）。

**M5 对比的一条纪律**：**换模型要看配对差值，不看两个平均值。**
均值差 0.02 可能是 30 条变好、28 条变坏相互抵消——那不是"略好"，是"方向相反"。
所以 `eval compare` 的头一行是"净改善 / 净劣化 / 不变"，区间用**配对 bootstrap**
（先按 case 算差值，再重采样 case），比"看两个 CI 是否重叠"灵敏得多；
下面紧跟劣化清单，每条带**两个模型各自的 trace_id**——差在哪道题只有并排看才知道。

真机配对结论（同一份 236 条意图集，qwen3.5:9b → gpt-oss:20b）：
**改善 0 / 劣化 226 / 不变 10**，均值差 −0.958，95% CI [−0.983, −0.932]。
但 gpt-oss 那一格的 `macro_f1` 显示 **1.000**——因为 236 条里只有 8 条产出了正文
（`max_tokens=32` 全被它的 reasoning 吃光，P12），主分数是在这 8 条上算的。
矩阵因此必须把分母一起摆出来（`可判定 8/236` + 顶部警告）：数字是真的，读法是错的。
下钻 trace 才给出正确结论——该改的是 `max_tokens`/`thinking` 参数，不是换模型。

**M6 扩展点的一条纪律**：**"可替换"不是文档里写着可替换，而是有一个外部实现真的跑通，
并且有脚本判定"接它有没有改内核"。**
内置实现与内核一起过拟合是察觉不到的——照协议写的外部实现才会把泄漏顶出来。
这一条在 M6 一次挖出四处：`LlmProvider` 协议漏了 gateway 必传的 `trace_id`
（外部实现必崩 TypeError）、`StreamingProvider` 描述了一个不存在的机制、
`ToolKind` 是 closed enum 所以外部执行器种类无法被表示、
`--provider mock|echo` 跑出来的 trace 在库里自称 `ollama-local`。
`scripts/check_extension_boundary.py` 因此是**第四道质量门**，而且豁免的只有注册表本身：
需要改内核的口径修正必须先单独落地（M6 里真的先落了采信阶梯与三态推断），
再接实现——反过来自家门禁会被"顺手绕过"一次，它就不再是门禁。

顺带立了另一条：**隔离机制会掩盖故障**，所以坏插件不再只是"跳过"——
它进台账，`onyx plugins` / `onyx eval tasks` 会打印出来并以退出码 1 结束，
`onyx doctor` 也多了一项体检。只隔离不报告，等于把"插件没生效"伪装成"插件正常工作"。

**M1 已在真机达成**：`onyx chat` 一次对话即落库完整 trace —— 引擎计数（in=19/out=47，
source=engine，confidence=high）、分段归因（`msg:0=8 + template_ctl=11 == 19`，
残差与标定截距 10.9958 互相验证）、prefill 冷/热判定、decode TPS、GPU 快照、原始 body 可 replay。

实测结论见 [`docs/PROBES.md`](docs/PROBES.md)（P1–P24，每条带证据与引擎版本）。
评测怎么跑才不出错觉：[`docs/eval-recipes.md`](docs/eval-recipes.md)。
**现在到底有什么、还欠什么**：[`docs/STATUS.md`](docs/STATUS.md)（数字当场核对，含核对命令）。

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
