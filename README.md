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
| M4 评测 | ✅ | 评测内核（task/grade/runner + 指标层 + bootstrap CI）· 6 个评分器 + 类型感知参数比对 · 236 条中文意图集 + 97 条工具调用集 · `intent_classification` 与 `tool_selection` 各一次真机运行 · BFCL 导入器 · GPU 独占锁（跨进程 + 心跳 + ETA），eval/Playground/live 测试互相排队（第三个任务与任务契约测试见下面的 M11 行） |
| M5 对比 | ✅ | 模型 × 任务矩阵 + 配对回归 diff（净改善/净劣化 + 配对 bootstrap CI + 劣化清单）· Eval / 矩阵 / 回归三页已在真实浏览器实测 · md/csv/自包含 html 报告导出 · 每次运行带数据集来历 |
| M6 扩展 | ✅ | 六个扩展点接 entry points（一处实现语义：坏插件隔离 + 失败可见 + 同名覆盖可查）· 两个真外部插件样板（任务 / provider）· 第二 provider：OpenAI 兼容通道（vLLM / LM Studio / Ollama `/v1`）· MCP 执行器（stdio + JSON-RPC，纯 stdlib）· OTLP 导出 sink · provider/sink/插件三套契约测试 · `scripts/check_extension_boundary.py` 把"接实现不改内核"变成构建门禁 |
| M7 跑得住 | ✅ | **S17** `onyx rotate`：默认 dry-run、只摘五列重 payload（分数指向的 trace 行永不删）、回收无主 blob、每次运行落 `retention_run` 留痕、单次回收超 60% 直接拦住 · **S18** `onyx db backup` / `verify-backup`：WAL 一致快照（不是 cp）、只装被引用的 blob、逐字节重算 sha256、"引用能否在备份里解析"专抓只备库不备证据 · **S19** `doctor` 补磁盘余量与 token 计量档位（明说 `hf_tokenizer`/`gguf_vocab` 本版本未实现）、迁移前自动留 `backups/pre-migration-v*.sqlite`、`onyx db sizes` 报体积曲线（跨度不足一天就说"问不出来"）· 第五道质量门：离线套件分支覆盖率 ≥ 80%（落地时基线 88%，S21 后 89%）|

| M8 配置与发行 | ✅ | **S20** `onyx.toml` 部署配置（provider / GPU 锁 / 保留窗口 / sandbox / serve 绑定）：优先级只有一条规则 **flag > 环境 > 文件 > 默认**，为此每条命令的 flag 内建默认都改成 `None` —— 带着具体默认值的 flag 会永远赢过配置文件，让它当场变成摆设且不报错 · 格式用 TOML 而不是设计稿里的 YAML，因为 `config.py` 在 `settings → runtime/store` 这条核心管道上，用 YAML 就等于要求"只用库不用 CLI"的人也装 `runtime` extra；`tomllib` 是标准库（详见 DESIGN §13） · `onyx config show` 逐项标出生效值来自哪一层（token 只报"已设置"，值不落终端） · `doctor` 抓"写了不生效"：未知键与坏类型（含 `port = true` 这种被当成 1 号端口的手滑）会指名并报红 · **S21** 非回环绑定的姿态：`--host 0.0.0.0` 没有 token 就**拒绝启动**（退出码 2），共享时 `--read-only` 只让看不让操作 · **S22** 版本策略（`0.8.0`＝M8；版本号单点定义在 `onyx/__init__.py`）+ `CHANGELOG.md` + `.github/workflows/ci.yml` 跑五道门与安装冒烟 |

| M9 操作闭环 | ✅ | **S23** 评测可以在界面发起了：`POST /api/runs` 只入队并立刻返回 run_id（跑评测的是服务里的一个 worker 线程，不是请求线程），进程内单飞排队 + 与 Playground 共用**同一把**机器级 GPU 锁；界面上看得到逐条进度、排队位置、"谁在占 GPU + ETA"，三条路都能取消（排队中 / 等锁中 / 跑一半）；跑完自动选中那条 run 并下钻 grade。被中断的运行不再留 `running` 僵尸：服务启动时把上次进程发起、且锁已空闲的行标成 `error` 并写明原因，**但不碰别的进程（CLI）正在跑的行**。**S24** 数据集也在界面导入了：`POST /api/datasets` 收 JSONL **文本**（不收路径——路径来自请求体就是任意文件读），来历 `upstream/revision/license` 一起落库，覆盖同名 id 要显式勾选确认，并且补上了断掉的闭环：以前 `onyx eval import --id x` 之后 `onyx eval run --dataset x` 会报"未知数据集"，而样本明明就在同一张库里。**S25** 工具库也能在界面审计了（Tool Bench）：注册表（内容 hash 版本 / 副作用 / tokens / 启用）、契约审计（逐条规则带修法，文案与 CLI 同源）、上下文开销（JSON 与模板脚手架分两笔，没传模板开销时占比是「—」而不是 0%）、执行器契约矩阵（点一次才真跑：✓ / ✗ / n/a / "没这列"四态互不冒充）、运行历史。做这一页时发现 `tool_run` 表与它的保留规则存在很久但**从来没有写入方** —— 现在 `onyx tools run` 会落一行，而确定性/幂等两列留「—」（跑一次测不出这两件事）。**S26** 模型治理有了出口：模型页可以拉取 / 卸载 / 删除权重（与 `onyx models pull|rm` 同一个 `AdminProvider`，未勾选确认时按钮点不动，通道不暴露控制面时报 501 并说清"为什么不做个假的"——兼容层没有统一端点，编一个会让"显存已经让出来了"这种判断建立在谎话上），删除只释放权重而**历史 trace 与分数一行都不动**；被中断的运行也能在界面上续跑（「续跑这条」→ 预填表单 → 同一个 `POST /api/runs` 带 `resume_run_id`，**写回原来那条 id**，实测中断在 12 条的 run 续到 236 且请求数只增 224；`dataset` 省略时继承原 run 那份而不是任务当前默认）。这一步顺带修掉两处只有真点才会露出来的口径：`n_cases` 被收尾缩成已评条数（`12/12` 与跑完那行完全同形，⚠ 未跑完 badge 因此永不亮），以及 Windows 上删不掉的锁文件（看板每秒读锁 ⇒ `unlink` 拿到 `ACCESS_DENIED`，原先被静默吞掉，留下一条心跳新鲜的假持有让后面的人白等 600 秒）|

| M10 观测触达 | ✅ | **S27** 出事会主动通知你了：`[alerts]` 配"盯哪些级别/哪些码、窗口内几次、cooldown 多久、发到哪"，判定是 `obs/alerts/rules.py` 里的**纯函数**（窗口/阈值/cooldown 全在离线单测里穷举），窗口计数**读库**而不是读进程内计数——所以重启既不会重复轰炸也不会忘记刚才发过什么。默认只盯 error 级：warn 级（低置信计数、缓存命中）在本地是常态，推给人会训练出"忽略通知"的习惯。**S28** 通用 webhook 出口：重试有上限、4xx 不重试、载荷只带结论与定位（**不含 trace 正文**），URL 优先环境变量且任何输出都去掉 query（那里通常挂着 secret）。它是第三条**显式登记**的网络例外——用 stdlib `urllib` 能悄悄绕过门禁，而"新增一个会发请求的模块必须写明是谁、为什么"正是这道门存在的理由。**S29** 触达看得见：Fleet 顶部一行同时说"近 1 小时有 N 条 error 级异常"与"通知系统自己好不好"（没装配 / 没出口 / 轮询出错 / 在跑，四句话互不冒充），「告警触发」面板里每条都能下钻到真实 trace；异常 chip 的**级别改为由后端给**（之前前端写死 warn，error 级于是显示成"提醒"）。`alert_trigger` 表记"命中 + 尝试投递"的合取（渠道失败也留一行带原因，被 cooldown 抑制的不写），`onyx alerts ls` / `doctor` 的「告警」项都读它。**多引擎形态就此定案**：一进程一引擎，多实例各配自己的 `ONYX_DATA_DIR`（理由与代价见 DESIGN §8.6） |

| M11 评测资产 | 🟡 | **S30** 第三个任务落地：`structured_extraction`（45 条中文结构化抽取 = 模板 36 + 人工难例 4 + 「无可抽取信息」负样本 5，生成器可复现、期望值自己过自己的 schema）。判定拆成**三层正交**——`json_valid_rate`（听不听话）/ `schema_valid_rate`（结构对不对）/ `field_em`+`score`（内容准不准），复用现有评分器不新写第四套；负样本走 `none_correct_rate` **不进主分数**（混进均值等于奖励"什么都不抽"）；`requires` 刻意不含 `STRUCTURED_OUTPUT`（那是被测对象，当前探针里大多是「未实测」）。同时把**"新任务的验收形状"变成一条契约测试**：`tests/contract/test_task_contract.py` 对 `specs()` 里每个任务（内置 + 插件）断言"声明的指标 == 产出的指标 / 每个区间跟着它那个数 / 主分数不许是稳定性指标 / 引擎故障只留「没考到」"，当场查出两个内置任务少声明 9–10 个指标。**真机三轮可归因**：qwen3.5:9b × 45，`2181720` ⇒ `score 0.129`、`date` 字段 EM **0.000**；查出来是**考卷自相矛盾**（提示词写"不要改写"而期望值要 ISO 日期与纯数值）+ 数值按字符串比 + 封闭词表没给模型 ⇒ `68d4f30` ⇒ **`score 0.893 [0.786–1.000]（n=28 case）`、`field_em 0.973`、`exact_object_rate 0.667`（差出来的 12 条全是多抽一个空占位字段）**。矩阵长到 3 列，导出报告的雷达图（阈值 3 个任务）真的出现了。S31 的 `instruction_following` 又把这一课教了一遍：39 条题、10 种**可机械检查**的约束，**约束由一句必然满足它的参考回答派生**，于是「这道题有解」从好话变成可跑的断言 （`unsatisfiable()` 对 39 条返回空，而收紧某题字数上限时它立刻点名那道题）。真机 qwen3.5:9b ⇒ `score 0.920 [0.884–0.952]`、`micro_rate 0.916`（152/166 条约束）、`all_satisfied_rate 0.641`——三个口径同时报，因为它们会分叉，只引用一个就是三种不同的故事；`by_kind` 指出薄弱环节在结构约束（`items_between 0.600`、`line_count 0.667`）而不是必含词（0.974）。空正文**不给任何约束记分**：逐条判的话 `max_chars` 与 `forbids` 会对「什么都没写」判通过。顺带把自己刚写下的两份同名 `_looks_like_refusal` 挖了出来，「不许同名顶层定义」那条结构门禁因此从只查 `cli.py` 升级为查整个 `onyx/`。矩阵现在 4 列；**S32** 交出第五列并把「分数太好」也当成缺陷来查：`long_context`（9 条组合式中文长文 = 4k/8k/16k 三档 × 3 变体，每条埋 3 个可精确匹配的事实，正文一个数字都不许出现）。第一版**没有干扰项**时真机跑出 9/9 全对、`needle_rate 1.000`——那不是模型强，是「扫到任意一个数字就能得分」，于是每个埋点配一个同句式、另一实体的干扰值，新增 `needle_confused` / `confusion_rate` 把「认错实体」与「没读到」分开量（分母只用带干扰项且答错的埋点，数据没带干扰项时写「—」而不是 0），并补三条考卷自检（正文无数字 / 值之间不许互为子串 / 题面只点名答案实体），每条都配注入缺陷测试证明会响。更贵的一课来自负控制：`--split 16k --num-ctx 4096` 时 Ollama 把 16.8k tok 的正文裁到 `in_tokens=2050`，**比窗口还小**，所以「`in_tokens ≥ num_ctx` 才记 skip」这条永远不响，三条被切的样本被判成 `partial`、`score 0.000`，而 `per_needle_ok` 只有结尾那条活下来——正是「切掉开头」的形状，该背锅的是窗口配置不是模型。判据改为**引擎给的数 vs 正文自己的 token 下限**（0.5 tok/汉字，实测 0.68）后，同一参数重跑得到 `score —` + 3 条 skipped，每条 grade 自带那句原因。正向那一次（run `01M4637FQZ…`，`git_rev 3f732a7`）是 `score 1.000 [1.000–1.000]（n=9）`、`needle_rate 1.000（27/27）`、`max_ctx_util 0.8201`、first/middle/last 各 n=9 全对 ⇒ **这台机器在 ≤16.8k tok 上没有中部塌陷，这一档对 9B 没有区分度**（要拉开得加 32k+，而 16GB 卡上 20480 已占 5.73G 显存）。第三处真缺陷在这次改动里露头：runner 只在「库里没有这份数据集」时才写样本行，而 case id 是内容哈希——加干扰项后 9 条全换了 id 而 `dataset_id` 不变，于是那一轮 **9 条 grade 反查样本命中 0 条**、`dataset.revision` 停在旧串；现在 revision 不同就重写，且旧样本只删没被任何 grade 引用过的（全留着会让「这份考卷几条」数不清楚，照 id 删则撕断历史分数）。矩阵现在 5 列（`qwen3.5:9b` 行 0.920 / 0.991 / 1.000 / 0.893 / 0.639）；S33 embedding 仍待做 |

| M12 防倒退 | 🟡 | **S34** 六页取数通路 e2e（10 条）+ SSE 在真 HTTP 连接上被消费（真 uvicorn + 真 httpx）+ "摘掉 broker 就只剩 hello"的断链自检，`-m e2e` 成为 CI 独立一步并被门禁清单钉住。**真浏览器驱动仍为 0**（Playwright 那一层待做）；S35 覆盖率基线与契约矩阵真 stdio 列、S36 `onyx perf` 基线待做 |
> CI 与安装冒烟都在本机验证过命令本身（干净环境里 `uv sync --extra dev,runtime,api,bench` → 1128 passed / 覆盖率 89%；
> `uv build` + `uv tool install` 隔离装起来后 `onyx version / db init / chat --provider mock / doctor` 全通），
> 但**这个仓库还没有远端**，所以 workflow 尚未真正跑过一次。推上 GitHub 才算"CI 已绿"。

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

**第五道质量门是覆盖率**（`make coverage`）：离线套件分支覆盖 **≥ 80%**，基线实测 90%（M10 后；1348 项）。
另有一档 **e2e**（`uv run pytest -m e2e`，12 项）：六个页面各取一次数并互相核对同源，
SSE 用真 uvicorn + 真 httpx 读帧，并额外钉住"把 broker 摘掉这条断言就会红"的自检；
它在 CI 里是独立一步（默认套件把它 deselect 掉了），也被 `test_release_surface.py` 钉进门禁清单。
门禁挂在门禁上而不是文档里——`fail_under` 让 `coverage report` 直接以非 0 退出。
用分支覆盖而不是语句覆盖，是因为这个项目的正确性大量住在 `if x is None` 的分岔上
（"未知"与"没有"必须走两条路），只数语句会让"两岔只走过一岔"的文件显得很像样。

**M1 已在真机达成**：`onyx chat` 一次对话即落库完整 trace —— 引擎计数（in=19/out=47，
source=engine，confidence=high）、分段归因（`msg:0=8 + template_ctl=11 == 19`，
残差与标定截距 10.9958 互相验证）、prefill 冷/热判定、decode TPS、GPU 快照、原始 body 可 replay。

实测结论见 [`docs/PROBES.md`](docs/PROBES.md)（P1–P24，每条带证据与引擎版本）。
版本与变更：[`CHANGELOG.md`](CHANGELOG.md)（含版本策略——MAJOR 停在 0 直到决定对外发行）。
评测怎么跑才不出错觉：[`docs/eval-recipes.md`](docs/eval-recipes.md)。
**现在到底有什么、还欠什么**：[`docs/STATUS.md`](docs/STATUS.md)（数字当场核对，含核对命令）。

下一步见 [`docs/ROADMAP.md`](docs/ROADMAP.md)（M7–M10 已全部达成；接下来是 M11 评测资产与 M12 防倒退，含三个需要决策的问题）。

## 共享给同事看（非回环绑定）

`onyx serve` 默认绑 `127.0.0.1` —— 那**不是**鉴权，只是碰巧没人连得上。
要局域网共享就必须带 token，否则**拒绝启动**（退出码 2），因为看板里有你全部的 prompt 与原始 body，
而且它能往你的 GPU 上打请求、unload 你正在用的模型：

```bash
ONYX_SERVE_TOKEN=… uv run onyx serve --host 0.0.0.0 --read-only
#   浏览器打开 http://<本机IP>:8787/?token=…      （token 只报"已设置"，不会被打进 config show 的输出）
```

- `--read-only`：GET 放行、写操作 403 并说明怎么放开 —— 共享"看一眼"与共享"操作台"是两件事。
- 优先用环境变量而不是配置文件写 token：文件会跟着备份、截图和 `git status` 漂走。
- `--allow-insecure-local` 可以裸跑，但**故意不可写进配置文件**：每次都要显式说一遍。
- SSE 只能走 `?token=`（`EventSource` 设不了请求头），所以这个 token 会出现在 URL、
  浏览器历史与任何中间层日志里 —— 能走 header 就走 header。

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
而图表形态固定，手写 SVG 比引 400KB 图表库更可控。产物 204KB JS / 13KB CSS（gzip 66KB / 3.5KB）。

## 目录

```
onyx/core/     L0 领域层：类型、可排序 id、时钟、事件契约、错误族、内容寻址存储（零三方依赖）
onyx/store/    L0 存储层：SQLite/WAL、幂等迁移、repo、异步批量 sink
onyx/cli.py    命令行入口
docs/          设计文档与分步实现方案
tests/unit/    无 IO、无网络、无模型的纯函数测试
```
