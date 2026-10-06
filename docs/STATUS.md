# 项目现状（Onyx · v0.8.0）

核对时间 **2026-10-05**，HEAD = M9 四步（S23 界面发起评测、S24 数据集闭环、S25 Tool Bench、
S26 模型治理出口与续跑）+ M10 三步（S27 告警内核与文件出口、S28 通用 webhook、S29 界面可见性与多引擎定案）
+ M12 第一步（S34 六页 e2e + SSE 上线消费）+ **M11 全部四步**（S30 结构化抽取与任务契约测试、
S31 指令遵循与考卷有解的自检、S32 长上下文检索与截断判据的实测修正、
S33 语义检索与第一条 embedding 通路）之后
⇒ **M0–M11 全部达成，只剩 M12 的 S35/S36**。
本文所有数字都是当场跑出来的（命令附在每节末尾），不是从旧文档抄的；
`README.md` 的进度表、`docs/IMPLEMENTATION.md` 的分步档案是历史沿革，
**这里回答的是"现在有什么、能干什么、还欠什么"**。

一句话：**不发一句命令就能完成"导入数据 → 拉模型 → 跑评测 → 续跑中断 → 看矩阵 → 下钻 trace →
审计工具库"，而且出事会在 1 分钟内主动通知你**（本地文件 + 通用 webhook 两个出口，触发历史落库可查）。
欠的是 S7 剩下的 `report usage` / `token explain` 两个入口、M12 剩下的 S35/S36，以及一批"带原因推迟"的项。

---

## 1. 有的功能（按层）

### L0 领域层 `onyx/core/`（零三方依赖，import-linter 强制）
11 个能力位、23 种异常码、6 种工具失败 kind、13 种评测判定、18 种事件类型、
sha256 内容寻址的 blob 存储、单调钟 + 墙钟双时间、事件契约带必填键校验（`strict=True` 会拒绝缺键事件）。

### L1 存储 `onyx/store/`
19 张表、schema v7（迁移可重复执行、`0005` 回灌了数据集来历、`0006` 是保留策略的留痕表、
`0007` 是告警触发历史 `alert_trigger`——审计表，永不参与清理）、
WAL + 批量 sink（队列满丢样本但 `dropped` 计数可见）、
三种事件 sink：`jsonl` / `null` / `otlp`（OTLP/HTTP JSON 编码）、
**数据生命周期 `store/retention.py`**：摘重引用 / 删无主 blob / 每次运行落 `retention_run`、
**可验证备份 `store/backup.py`**：WAL 一致快照 + 被引用的 blob + 逐字节 sha256 校验。

### L2 引擎接入 `onyx/llm/`
- **咽喉点只有一个**：所有模型调用必经 `gateway.py`（评测也不例外，所以每个分数能下钻到真实 trace）
- 3 个 provider：`ollama`（原生 `/api/chat`）、`openai-compat`（vLLM / LM Studio / Xinference / Ollama `/v1`）、`mock`（脚本化假引擎，离线跑通全链路）
- token 保真阶梯：`engine > hf_tokenizer > gguf_vocab > fitted > compat > heuristic`，
  每个数字带 `source` + `confidence`；**未知显示「—」，绝不显示 0**。
  **本版本实际只产出四档**：`engine` / `fitted` / `heuristic` / `compat`。
  `hf_tokenizer`（T1）与 `gguf_vocab`（T2）有类型、有采信优先级、有插拔位
  （`CounterContext.tokenizer`），但没有实现——P9 判定 T1 不能当主路径（3 个模型里 2 个
  根本没有 chat template），T2 的 GGUF 自建 BPE 没做。所以 `tokens` extra 装了也不生效，
  `onyx doctor` 会把这句话写在明面上。
- 分段归因：`Σ(各分段) + template_ctl == 引擎计数`，残差与标定截距互验（P20/P21）
- 冷/热 prefill 分列（P11：合并聚合是谎话）、keep-alive 剩余、显存 offload 判定
- 流式与非流式共用同一个 assembler；OpenAI 兼容分支同时吃 `delta` 与 `message`

### L3 观测 `onyx/obs/`
5 个内建 visitor（token / tool / gpu / cost / anomaly，**顺序即契约**）+ 外部插件 visitor；
23 种异常码统一码表（前端文案与 CLI 同源）；
有界状态 + TRACE_END 缺失时按容量淘汰（宁可丢一条观测也不 OOM）。
- **告警 `obs/alerts/`（M10）**：`rules` 是纯函数（窗口 / 阈值 / cooldown / 计数下限），
  `channels` 把投递失败收成结果而不抛异常，`service` 是 serve 里的轮询线程。
  判据**读库**（`anomaly` + `alert_trigger`）而不是读进程内计数 ⇒ 重启不重复轰炸也不忘记已发；
  默认只盯 error 级（warn 级在本地是常态，推给人会训练出忽略通知的习惯）；
  通知文案带 SPECS 的"建议：…"，不在前端或渠道里重写第二套
- **两个出口**：本地文件 `alerts-YYYYMMDD.jsonl`（零依赖一定能落）+ 通用 webhook
  （httpx 惰性导入 = 第三条**显式登记**的网络例外；重试有上限、4xx 不重试、URL 去 query 掩码、
  载荷不含 trace 正文）。装配的唯一落点是 `obs/alerts/service.build_channels`
- **触发历史 `alert_trigger`**：记"命中 + 尝试投递"的合取，每 (命中, 渠道) 一行，
  渠道失败也留一行带原因；cooldown 抑制**不写行**；`rule_json` 存命中当时的判据快照

### L4 工具子系统 `onyx/tools/`
- 注册表：内容 hash 版本化、契约审计（每条规则对应一个可行动修法）、上下文开销核算（与 trace 的 `part=tool_defs` 同源）
- 4 种执行器：`python_fn`（AST 白名单，不用 eval）/ `http`（URL 只来自人工审核定义，不做通用 fetch）/ `mcp`（stdio + JSON-RPC，纯 stdlib）/ `fixture`（评测零副作用通道）
- **8 条契约断言 × 4 列执行器矩阵**（`onyx tools contract`，离线零网络）：
  失败强制分成 `arg_error / rejected / timeout / unknown_tool / skipped / error` 六种，
  每个 n/a 必须写明原因（静默跳过等于让最关键的保证消失）
- 客户端工具循环：预算 / 熔断 / 孤儿补齐 / fire-and-verify 六种判定
- 沙箱：副作用白名单、dry-run、审批回调、`impl_ref` 白名单、deadline 传到 socket（P22）
- **契约矩阵的构建处只有一个（`tools/matrix.py`，S25）**：CLI 的 `--json` 与
  `GET /api/tools/matrix` 返回同一个形状；每列的样本定义与出处都报出来
- **`tool_run` 从此有写入方**：`onyx tools run` 落一行（状态/延迟/定义 hash/参数键/mock 档位）。
  这张表与它的保留规则存在了很久却没有一条路径写过它，所以"运行历史"面板会永远是空的

### L5 评测 `onyx/eval/`
- 6 个内建任务：`intent_classification`（236 条中文意图集）、`tool_selection`（97 条工具调用集）、
  `structured_extraction`（**S30**，45 条中文结构化抽取：模板 36 + 人工难例 4 + 「无可抽取信息」负样本 5）、
  `instruction_following`（**S31**，39 条中文指令遵循 = 6 话题 × 3 形态 × 2 变体 + 3 难例，
  每题 2–5 条**可机械检查**的约束，共 166 条约束判定）、
  `long_context`（**S32**，9 条组合式中文长文 = 4k/8k/16k 三档 × 3 变体，每条埋 3 个可精确匹配的事实
  在 first/middle/last，另各配一个**同句式、另一实体**的干扰值 ⇒ 27 个埋点、27 个干扰值）、
  `semantic_similarity`（**S33**，12 个语义框 = 12 条题，每题 1 query + 4 候选：同义改写 / 反义 /
  两个无关话题；query、gold、anti 全部由模板派生，关系是**构造**出来的不是人标的）
- **考卷有解是断言出来的（S31）**：每条样本带一句必然满足它全部约束的参考回答，
  约束参数由参考回答实测派生；`unsatisfiable()` 把"无解的题"变成可跑的失败，
  而且**注入一条收紧的约束它必须点名**（`test_the_satisfiability_detector_actually_fires`）。
  提示语再由约束渲染回来（`phrase_for`），凡是考模型的都必须说得出
- **任务契约测试（S30）**：`tests/contract/test_task_contract.py` 把"声明的指标 == 产出的指标、
  区间跟着它那个数、主分数不许是稳定性指标、引擎故障只留「没考到」不留 0 分"
  对 `specs()` 里**每一个**任务参数化跑（内置与插件同一套断言，新增任务不必改这个文件就会被覆盖）
- 结构化抽取判定的三层各自成指标：`json_valid_rate`（听不听话）/ `schema_valid_rate`（结构对不对）/
  `field_em` 与 `score`（内容准不准）；负样本走 `none_correct_rate` **不进主分数**——
  混进均值等于奖励"什么都不抽"
- 指令遵循同时报**三个会分叉的口径**：`score`（每条样本满足率的平均，样本等权）、
  `micro_rate`（Σ满足 / Σ约束，约束等权）、`all_satisfied_rate`（全照做的题占比，最严）。
  真机上它们是 0.920 / 0.916 / 0.641——只引用其中一个会讲出三种不同的故事。
  `by_kind` 每种约束各带分母，且**没考到的类型不出现在字典里**（不是填 0）
- **空正文不给任何约束记分（S31）**：逐条判的话 `max_chars` 与 `forbids` 会对"什么都没写"判通过，
  于是最省 token 的答案能拿到 2/6 分。拒答反过来按产出实测，只单独占 `refusal_rate`；
  并且"全部约束都满足"的答案永远不被拒答启发式改判
- **长上下文里"分数最好看"就是缺陷信号（S32）**：第一版数据真机跑出 **9/9 全对、`needle_rate 1.000`**——
  那不是模型强，是"扫到任意一个数字"就能得分。于是每个埋点配一个同句式、另一实体的干扰值，
  并新增 `needle_confused` / `confusion_rate`（分母只用**带干扰项且答错**的埋点，
  数据没带干扰项时写「—」而不是 0）把"认错实体"与"没读到"分开量
- **截断判据不能拿窗口当尺子（S32 最贵的一条实测）**：`--split 16k --num-ctx 4096` 时 Ollama 把
  16.8k tok 的正文裁到 `in_tokens=2050`——**比窗口还小**，所以"`in_tokens ≥ num_ctx` 才记 skip"
  这条永远不响，三条被切的样本被判成 `partial`、`score 0.000`（而 `per_needle_ok` 只有结尾那条活下来，
  正是"切掉开头"的形状）。现在判据是**引擎给的数 vs 正文自己的 token 下限**（0.5 tok/汉字），
  连下限都不到 ⇒ `SKIPPED` + 可行动的原因 + `shrink`；同一参数重跑得到 `score —` 与 3 条 skipped
- **语义检索判据是排序不是阈值（S33）**：实测同义对 cos 0.749–0.909 与反义对 0.649–0.796
  **区间重叠** ⇒ "离得近"完全可能是"意思相反"。指标因此是 `recall@1`（主分数）/ `recall@k` /
  `mrr` / `anti_first_rate` / `anti_above_gold_rate` + 三个平均相似度与 `sim_gap`；
  **并列按池内序号裁决**（不写死规则的话同一份数据两次能跑出不同名次，而向量本身有 ≤3e-4 抖动）
- **embedding 通路是 S33 新建的**（`Cap.EMBED` 与 `TraceKind.EMBED` 自 S3 起只是枚举位）：
  `EmbedRequest` / `Embedding`（L0）→ 可选协议 `EmbeddingProvider` → `gateway.embed()` →
  `/api/embed`。provider 级能力**不再**是闸门的依据：能力位现在取
  `provider.capabilities() ∪ /api/show 自报`，因为 embedding 是模型级事实
- 向量**不落库**（没有消费者读它，只会撑大 `.data`），落的是输入证据 + 引擎计数 + 耗时；
  `usage` 里 `out_tokens=None` 而不是 0（向量没有输出 token），runner 的成本口径按调用种类分开
- 长上下文的位置与档位各是一个分桶且**每格带分母**：`by_position{matched,confused,n,rate}`、
  `by_bucket{cases,all_correct,matched,needles,…}`；漏答也算进它所属位置的分母，
  而 `max_ctx_util` 刻意把 skip 的那些也算进来（这个数的用途就是"该不该调 `--num-ctx`"）
- **考卷自检同样跑在长上下文上（S32）**：`digits_only_in_needles`（正文不许有数字）、
  `value_string_collisions`（值之间不许互为子串——`12.5` 里含 `2.5`，"全文只出现一次"才会数对）、
  `ambiguous_questions`（问句只点名答案实体）——三条都配了注入缺陷测试证明会响
- 8 个评分器模块 + 类型感知参数比对（数值容差 / 日期归一 / 集合等价 / 严格档与宽松档分开报）。
  S30 修正了 `field_em`：数值字段按**数值**比，`500` 与 `500.0` 是同一笔钱
- 指标层：macro_f1 / accuracy / balanced_accuracy / P-R-F1 / 混淆对 /
  pass^k 与 pass@k / stability_gap / bootstrap CI（**按 case 重采样**，95% 区间带出处）
- 配对比较：净改善 / 净劣化 / 不变 + McNemar 翻转表 + 配对 bootstrap + 劣化清单（每条带两个 trace_id）
- 模型 × 任务矩阵（每格取**最新一次 done**，薄覆盖率强制显示分母）
- 报告导出：md / csv / 自包含 html（含手写 SVG 雷达图，任务数 ≥3 才画）
- 调度：GPU **机器级**独占锁 + 心跳 + ETA + `--unload-others`、断点续跑（成本与 n_done 不被后续片段清零）
- **`n_cases` 是考卷大小，不是已评条数（S26 修正）**：中断的 run 显示 `12/236` 并带 ⚠，
  而不是与跑完那行同形的 `12/12`；续跑段的 `max(原计划, 已评)` 也不许把考卷改小
- **提交服务 `eval/service.py`（S23）**：进程内单飞队列（`max_pending=8`，满了报 429 而不是默默排队）、
  取消（排队中 / 等锁中 / 跑一半三条路都真能停）、逐条进度快照、启动时回收上次进程留下的 `running` 僵尸行
- **界面上能续跑被中断的 run（S26）**：运行页「续跑这条」（只有 `cancelled`/`error` 可点）→ 预填表单 →
  同一个 `POST /api/runs` 带 `resume_run_id`，**写回原来那条 id**；四条拒绝各说自己的修法
  （id 不存在 / 任务或模型变了 / 已经跑完 / 状态还是 running 说明有别的进程在写）；
  `dataset` 省略时**继承原 run 那份**而不是任务当前默认——"省略"的本意是"照旧"
- 能力不满足 ⇒ 整任务 skip 且**留下带原因的记录**（禁止隐式降级）
- 数据集导入：`--source bfcl`、JSONL、`file:<路径>`；来历（upstream/revision/license）参与可比性判定
- **导入 → 读回 → 跑评测 是通的（S24）**：`load_registered` 把已登记的 id 读回成 Dataset，
  `load_dataset(..., db=)` 按"内置 → file: → 库里"解析，CLI 与界面同一个顺序；
  `register_dataset` 是唯一的落库口，覆盖旧 revision / 条数变化会返回警告（CLI 打印、API 一起回给界面）

### L6 接口
- **CLI 43 条命令**：顶层 7（`chat` / `serve` / `doctor` / `plugins` / `version` / `calibrate` / `rotate`）
  + 分组 36（`db 5` / `probe 4` / `models 4` / `traces 3` / `tools 9` / `eval 8` / `config 1` / `alerts 2`）
  ——`models pull|rm` 是 S26 补的出口（且 `sync`/`ls` 一起补上 `--provider`：
  pull/rm 能指到别的通道而它们不能，就会出现「拉得下来、同步不上」），`alerts ls|test` 是 S27 补的
- **部署配置 `onyx.toml`**（S20）：provider / GPU 锁 / 保留窗口 / sandbox / serve 绑定 / **告警** 收在一处。
  优先级只有一条规则 **flag > 环境 > 文件 > 默认**；`onyx config show` 逐项标出它来自哪一层，
  `onyx doctor` 把"写了不生效"的未知键与坏类型报成红项（模板见 `onyx.example.toml`）。
  `[alerts]` 的 9 个键 + `ONYX_ALERT_WEBHOOK_URL` 全部被真的消费；
  URL 与 serve.token 同样只报"已设置（N 字符，值不打印）"
- **doctor 9 项体检**（`--skip-network`；带网络时 10 项）：配置 / Python / 可写 / 磁盘 / 迁移 /
  blob 引用 / 计量档位 / **告警** / 插件（+ 引擎可达）。
  「告警」只看配置与留痕（文件出口写不下去、或最近 5 次投递全失败才红），
  **不去探 serve 在不在跑**——探别人的端口会把体检变成网络攻击面探测
- **API 34 个操作 / 32 条路径**（33 REST + `GET /api/stream` SSE）。写操作 7 个：
  Playground chat、admin unload、**拉取模型**、**删除权重**（后两个 S26，都要求 `confirm=1`，
  通道没有控制面时 501）、**发起评测**、**取消评测**、**导入数据集**
  （S21 的 `--read-only` 全部挡 403）；
  Tool Bench 的五个端点全是 GET，所以只读看板也能完整审计工具库
- **非回环绑定强制 token**（S21）：`serve --host 0.0.0.0` 没有 token 就拒绝启动；
  `--read-only` 让共享看板不变成共享操作台；错误体统一 `{error:{code,message,detail.hint}}`
- **Web 11 页**：Fleet（S29 起顶部有"error 级异常 + 通知系统状态"一行与「告警触发」面板）/
  Models（含 S26 的治理面板：拉取 / 卸载 / 删除，勾选才可用）/ Traces /
  TraceDetail / Token Ledger / Playground / **Tool Bench** /
  评测（运行与发起，S26 起可「续跑这条」）/ 矩阵 / 回归 / 数据集
  ——手写 CSS token 与手写 SVG，无组件库无图表库；每页都实测过（零 console 错误）。
  评测的四个子页从 S24 起有 subnav——之前矩阵与回归**只能手敲 hash** 才到得了

### 扩展点（DESIGN §13，六个 group 共用一套发现语义）
| group | 状态 | 现在有什么 |
|---|---|---|
| `onyx.providers` | ✅ | mock / ollama / openai-compat + 外部插件 |
| `onyx.tasks` | ✅ | 2 个内建 + 外部插件（`TaskSpec` 或实现类） |
| `onyx.sinks` | ✅ | jsonl / null / otlp + 外部插件（`--sink NAME` 装配） |
| `onyx.tool_executors` | ✅ | 4 内建 + 外部插件（`fixture` 通道抢不走） |
| `onyx.observers` | ✅ | 5 内建 + 外部插件（一律排在内置之后） |
| `onyx.graders` | ⏸ 刻意未接线 | 没有"按名字分发判据"的消费点，先建注册表就是死代码（理由写在 DESIGN §13 与 `onyx plugins` 表里） |

语义：坏插件隔离 + **失败进台账**（`onyx plugins` / `eval tasks` 打印并退出码 1，`doctor` 有一项）、
同名覆盖可查（`↻内置`）、按 group 缓存但失败不缓存。
两个样板包在 `plugins_example/`，`scripts/check_extension_boundary.py` 是第四道质量门。

### 实测知识（`docs/PROBES.md`，P1–P24）
24 条真机结论：缓存语义、thinking 计数、工具格式三态、标定方法、
`/v1` 与原生计数差异、`stream_options.include_usage`、httpx deadline 传播、
`/v1` 关不掉 thinking（P23）、兼容层计数在"只有它"时是唯一可信自报（P24）。
未决：U7（27B 的 offload 落差）、U8（图片 token）、U9（`cached_tokens` 字段是否存在）。

---

## 2. 质量门与规模

```
uv run pytest            # 1637 passed, 1 skipped, 36 deselected（默认档就是离线套件，CI 用它量覆盖率。S32 三个文件 58 条 + S33 六个文件 60 条 + 样本重写与能力闸门 6 条；`test_task_contract.py` 41 → 49 条：多一个任务就多 8 条断言，一行任务专属逻辑都不用加。S33 只逼它泛化了一次——原来写死用 `Generation` 造样本，现在按任务自己 `build()` 的返回类型造）
uv run pytest -m e2e     # 12 passed（S34：六页取数同源 10 条 + SSE 真 HTTP 消费与"断链自检"2 条。默认档把它 deselect 了，CI 里是独立一步）
uv run pytest -m live     # 20 passed（真打 qwen3.5:9b，与评测共用机器级 GPU 锁）
uv run pytest -m probe     # 4 passed（P 系列实验的可重跑版本）
uv run ruff check .         # All checks passed（`ruff format` 不是门禁）
uv run lint-imports          # 3 contracts kept（网络例外 2 条：executors.http + sinks.otlp/alerts.webhook 合并在契约 2，每条写明是谁与为什么）
uv run coverage run -m pytest -q && uv run coverage report   # 91% ≥ 80%（分支覆盖，离线套件，13,785 句；S33 新代码：任务 94% / 数据生成器 89%，S32 那两个仍 99% / 100%）
uv run python scripts/check_extension_boundary.py   # 接入实现未触碰受保护内核文件（S30 因此把 RunReport 那条修复单独成一个提交）
uv run onyx doctor            # 9 项体检（带网络 10 项）：配置 / Python / 可写 / 磁盘 / 迁移 / blob / 档位 / 告警 / 插件
uv run onyx config show       # 每一项生效值标出来自 flag/环境/文件/默认哪一层（token 与 webhook URL 只报"已设置"）
uv build && uv tool install --from dist/*.whl …  # 干净环境装起来：version / db init / chat(mock) / doctor 全通
前端：tsc --noEmit / vitest 106 / vite build（235.89KB js）+ 浏览器 take_snapshot
API：openapi 32 paths / 34 operations（含 `GET /api/stream` SSE）
CI：.github/workflows/ci.yml 跑上面这些（本机已验证命令本身可跑通；仓库尚无远端 ⇒ 还没真跑过一次）
```

真机跑过的证据（可复查，都在 git 里）：
S33 的语义检索是**第一条 embedding 通路**，正例与负例都在真机上跑过：
正向 ⇒ run `01M47MYTYE1MAB2N28171DV3YR`（`git_rev 37c70b3`，模型 `qwen3-embedding:0.6b`，
12/12 done，1.39s 热态 / 首次含载入 18.0s，444 in tok，12 个请求）
**`score 0.833 [0.583–1.000]（n=12）· recall@k 1.000（top_k=3）· mrr 0.917 ·
anti_first_rate 0.167 · mean_gold_sim 0.9659 vs mean_antonym_sim 0.9200（gap 0.0459）vs
无关 0.3567 · dimension 1024 · reported_in_tokens 12/12`**。
这一列**有区分度**（不是 S32 那种 1.000）：掉的两条指名道姓——
`密封圈没有转入B 号货架`、`无关人员没有进入配电间` 都排在了真同义句之前，
即"反义抢占"；`by_topic` 里仓储与安全各 2 条都是 0.5，其余四格 1.0（每格带 cases）。
负例 ⇒ `--model qwen3.5:9b` 整场 `SKIPPED · 需要能力 ('embed',)`，一条请求都不发
（`/api/show` 对 9B 确实不报 embedding）——注意这条判据本身是这一步修出来的：
原先只看 `provider.capabilities()`（Ollama 那是通道基线，永远不含 embed），
所以向量任务在真机上会**永远被 skip**，而那句 skip 读起来像"这个模型不行"。
通路侧另有一处实测纠正：`onyx models pull` 用 `last["done"]` 判成败，
而 0.35.1 的收尾是 `{"status":"success"}` ⇒ 每次真拉都被判成失败（`507f02c`）。
S32 的长上下文在 qwen3.5:9b 上留下**一正一负两个运行**：
正向 ⇒ run `01M4637FQZXQAEPZ0X4R8VFGE4`（`git_rev 3f732a7`，9/9 done，72.1s，89,079 in / 331 out）
**`score 1.000 [1.000–1.000]（n=9 case）· needle_rate 1.000（27/27）· confusion_rate 「—」·
n_truncated 0 · window 20480 · max_ctx_util 0.8201 · reported_in_tokens 9/9**，
`by_bucket` 三档各 3 case 全对（实测 in_tokens：4k 4,361–4,432 / 8k 8,508–8,570 / 16k 16,740–16,796），
`by_position` 的 first/middle/last 全是 9/9 ⇒ **这台机器在 ≤16.8k tok 上没有中部塌陷**，
反过来说这一档对 9B 没有区分度，要拉开得往 32k+ 走（而 16GB 卡上 20480 已占 5.73G 显存，
上限是卡决定的）。负向（故意把窗口设小）⇒ run `01M46097GT68W175SCFPR85MCV`
（`--split 16k --num-ctx 4096`）**3 条全部 SKIPPED、`score —` 而不是 0.000**，每条 grade 自带原因
「引擎只回报 2050 tok，而正文按汉字下限至少 13525 tok ⇒ 开头被切掉了」。
这条判据本身是被真机逼出来的：旧判据写的是"`in_tokens ≥ num_ctx` 才记 skip"，
而 Ollama 会先把正文裁到**比窗口还小**（2050 < 4096），于是它永远不响——
那一次的 `per_needle_ok` 是 `{q1:False, q2:False, q3:True}`，只有结尾埋点活下来，
正是"切掉开头"的形状，而分数显示的是 `score 0.000`（该背锅的是配置，不是模型）。
另一处真机逼出来的改动：第一版数据没有干扰项，跑出来 9/9 全对——**"全对"在这里是测量缺陷**
（扫到任意一个数字就能得分），所以补了同句式、另一实体的干扰值 + `needle_confused` /
`confusion_rate` 把"认错实体"与"没读到"分开量，revision 里留下 `distractors=yes`
（干扰项变了，同一个 id 的两次分数就不可比）。
顺带补了一处"查得到却看不见"的裂缝：`dataset_id` / `dataset_revision` 早就落了库，
而 `onyx eval show` 只印 seed/app/git，于是现在 `run` 与 `show` 都各有一行「考卷 … · …」。
**更深的一处是同一次改动挖出来的**：runner 只在"库里没有这份数据集"时才写样本行，而 case id
是内容哈希——加干扰项后 9 条全换了 id 而 `dataset_id` 不变，那一轮 **9 条 grade 反查样本命中 0 条**、
`dataset.revision` 停在旧串、`run_dataset_ids()` 返回空集。现在 revision 不同就重写，
旧样本只删没被任何 grade 引用过的（库里因此留着上一版那 9 条——历史 run 还靠它们回到当时的考卷）。
S31 的指令遵循在 qwen3.5:9b 上跑完 39 条（`git_rev a5f593f`，39.5s / 3,628 in / 734 out）：
**`score 0.920 [0.884–0.952]` 但 `all_satisfied_rate` 只有 0.641**——
"平均每条约束 92% 照做"与"只有 64% 的题全照做"是两个都真的数，引用哪一个就会讲哪一个故事，
所以三个口径（`score` / `micro_rate` 152/166 = 0.916 / `all_satisfied_rate`）一起出。
`by_kind` 指出薄弱环节在**结构约束**：`items_between 0.600（n=10）`、`line_count 0.667（n=3）`，
而 `contains 0.974（n=39）`、`forbids` / `no_markdown` / `prefix` 全 1.000 —— 不是不会抄词，是不会按条数排布。
考卷侧的自检也真跑过：`unsatisfiable()` 对全部 39 条返回空，
而把某题字数上限收到参考回答之下时它立刻点名那道题（注入缺陷的自检是一条独立测试）。
这一步顺带挖出自己刚写下的一个静默缺陷：`instruction_following.py` 里有**两份同名
`_looks_like_refusal`**，早期那份还写着 `len(_visible(text))`（而 `_visible` 返回整数）——
全量测试是绿的，因为 Python 取后一份。于是那条"不许同名顶层定义"的结构断言
从只查 `cli.py` 升级为对整个 `onyx/` 生效（`tests/unit/test_module_structure.py`）。
S30 的新任务在 qwen3.5:9b 上跑了三轮，**分数差异全部能归因到提交**（`git_rev` 逐条记录）：
`2181720` ⇒ `score 0.129 [0.032–0.258]`、`date` 字段级 EM **0.000（15 条全错）**；
查下去是**考卷自己有问题**——提示词写着"字段值必须来自原句，不要改写"，而期望值要 ISO 日期与纯数值，
模型照抄「3 月 4 号」于是被判不会抽取。同轮还有 `500` vs `500.0` 被算成抽错（数值按字符串比）、
以及 `event` 是五值封闭词表却没告诉模型（现场输出 `"event": "丢了钱包"`）。
修完 ⇒ `68d4f30`：**score 0.893 [0.786–1.000]（n=28 case）· field_em 0.973 · exact_object_rate 0.667 ·
json_valid 1.000 · schema_valid 0.733 · off_vocabulary 0.000 · 负样本 5/5 全对 · 45 请求 70.8s / 11,037 in / 1,400 out**。
`score` 与 `exact_object_rate` 差出来的 12 条全是"多抽一个空占位字段"（`"org": ""`），
那就是"内容对但下游不能直接用"的真实比例。浏览器侧（真 serve:8787 + vite:5173）：
矩阵长出第三列（`macro_f1 0.991 / score 0.893 / must_call_acc 0.639`，可比性警告点名三份数据），
运行列表显示 `score 0.893 …`（**改 HEADLINE 顺序之前这里显示的是 `pass_hat_k 0.667`**——
稳定性指标顶掉了任务自己的主分数），grade 表的「期望/预测」从整列 `[object Object]` 变成
`person=李娜 · place=杭州 · …`，判定筛选项改成读这次运行自己的分布
（硬编码清单里根本没有 `partial`，而 45 条里 27 条是 partial）。
`onyx eval report --format html` 里那张 SVG 雷达图的阈值是"任务数 ≥3"，第三个任务落地后它真的画了出来。
M10 把"记录"变成了"触达"，两条出口都在真机上验过，不是 MockTransport：
`ONYX_ALERT_WEBHOOK_URL=… onyx serve` 起来后往库里插一条 `CONTEXT_OVERFLOW`（error 级）⇒
一个轮询周期内本地假接收端**真收到 POST 200**，`alert_trigger` 落下两行 `sent`，
文件出口同刻多一行；库里 webhook 那行的 detail 是 `http://127.0.0.1:8799/hook ← 200`——
配置里那个 `?key=local-secret` **没有落库**（掩码在生产路径上成立）⇒ 停掉接收端再插一条
`TOOL_LOOP` ⇒ webhook 行变 `failed` 带完整 ConnectError 原因，而 file 行照样 `sent`
（两个出口互不拖累）⇒ 之后 12 秒两个周期历史不增长（cooldown 成立）。
浏览器 `#/` 侧：顶部一行读作"近 1 小时有 1 条 error 级异常：CONTEXT_OVERFLOW 1 · 告警在跑 ·
出口 file · 已轮询 13 次"，异常 chip 的**级别来自后端**（✕ error 与 ! warn 互不冒充——
改之前前端把两者都写死成 warn），「告警触发」面板那一行的样本按钮落到真实 trace。
`onyx doctor` 的「告警」项在真库上读作"判据：error 级 · 300s 内满 1 次 · cooldown 600s · 出口 file"。
验证用的两条假异常与它引出的通知行已从 `.data` 删除。
S26 在浏览器里把两条新出口都走到底：`#/eval/run` 真提交一条评测、立刻取消 ⇒ 该行显示
`12/236` 并带 ⚠ 未跑完（**修 `n_cases` 之前它显示 `12/12`，与跑完那行完全同形**）⇒
点「续跑这条」⇒ 面板出现「续跑」badge 与"正在续跑 1EGG17ZJYVP7"⇒ ✓ done `236/236`，
**run id 不变**、`cost.requests=236`（旧 12 + 新 224，已评过的没有重复计费）、
`aggregate.resumed=true / already_graded_before=12`。`#/models` 的治理面板：未勾选确认时
拉取/卸载/删除三个按钮实测都 disabled；勾选后拉取回执 `已拉取 mock/demo-pulled（digest —）`
（没有 digest 就印「—」而不是空）、卸载说清"下一次请求是冷启动"、删除用后端那句
`权重已释放；历史 trace 与分数保留`。这一步又挖出一处 Windows 真实缺陷：
`release()` 的 `unlink` 会因看板每秒读锁文件而拿到 `ACCESS_DENIED`，原先被
`contextlib.suppress` 吞掉 ⇒ 留下一条**心跳新鲜**的锁，后面所有人白等 `stale_after_s`
（默认 600 秒），现象只是"一直排队"看起来像死锁（我自己测续跑时卡住 19.5s 才发现，
持有者是我自己的 owner 串）。现在删不动就把心跳推到 `2000-01-01`——"已不在持有"正是事实。
S25 在浏览器里跑通了 Tool Bench：导入 `examples/tools.yaml`（5 个工具）后点「跑一次」，
矩阵显示 4 列 × 8 断言，`timeout_is_reported` 在 mock 列是 **n/a**、
`mock_policy_makes_no_real_call` 在 python_fn 列是 n/a，`ollama_builtin` 整列是「—」并写明"未实现"；
开销面板报 "JSON 本身 700 tok / 模板脚手架 — / 模板占比 —"（没传 overhead 就不编 0%）；
`onyx tools run calculator` 之后运行历史出现那一行，确定性与幂等两列都是「—」（跑一次测不出来）。
S24 走完整闭环：`#/eval/datasets` 导入 3 条 JSONL（`id=browser-mini-v1`、
手填 upstream/license）⇒ 自动跳回「运行与发起」，数据集下拉出现它 ⇒ 点开始 ⇒ ✓ done 3/3，
run 记录里 `dataset_id=browser-mini-v1`、`dataset_revision=sha256:75a32ae5246c2742`、`trigger=api`。
顺带修掉两处只有真点才会露出来的裂缝：评测四个子页之前没有导航（矩阵/回归只能手敲 hash），
以及子集下拉一直显示任务默认集的 splits（选别的数据集还能挑一个不存在的子集）。
意图集 `macro_f1 0.991 [0.978–1.000]`；工具集 `must_call_acc 0.639` 而
`hallucinated_tool`、误调率均为 0（**掉分全在参数上**）；
配对回归 `改善 0 / 劣化 226 / 不变 10`，同时暴露 `可判定 8/236` 的分母陷阱；
`--provider openai-compat` 打本机 `/v1` 得 `usage=compat/low in=16 out=400`（与 `finish=length` 自洽）；
`--provider echo`（外部插件）落库 `heuristic/low` 且无引擎计数的项显示「—」；
MCP 工具经 `tools fire --mock live` 判定 **PASS**（模型 → 循环 → stdio server → 回填 → final）。
S23 在浏览器里真跑过：`--provider mock` 的 serve 上点「开始评测」跑完 **944 条**（236×4）并自动下钻 grade；
另起进程占住机器级 GPU 锁时界面显示"等 GPU：当前由 cli-eval:holder 占用，预计还需 180s（已等 7s）"，
点取消后状态停在 `cancelled` 且 `/api/runs` 里**没有**那条 run（一条样本都没跑就不该有记录）。
这一步顺带暴露了一处 Windows 真实缺陷：看板每秒读锁文件之后，心跳的 `os.replace` 会拿到
`ACCESS_DENIED`，原先会把整轮评测判死——现在重试 + 失败计数进 `cost.gpu_heartbeat_errors`。

---

## 3. 还欠什么

### A. 计划里承诺、但确实没做的（S7 的运维交付物 + 一个页面）
| 缺什么 | 计划出处 | 为什么算事 |
|---|---|---|
| `onyx report usage --since 7d --format md` + `report/exporters/{csv,md,jsonl}` | S7 产出文件 | 只有 `eval report`（评测报告）；**用量周报**没有。今天要看一周吞吐只能自己写 SQL |
| `onyx token explain <trace>`（多源对比表） | S7 / 附录 A 的自测命令 | 功能其实存在但埋在 `traces show` 的对账表里，没有独立入口；对照"某个数字为什么被采信"这个高频问题，命令行入口是必要的 |
| `hf_tokenizer`（T1）与 `gguf_vocab`（T2）没有实现 | P9 结论：T2 应提前为 S4 主实现 | 阶梯、采信优先级、`CounterContext.tokenizer` 插拔位都在，缺的是实现本身。后果：分段归因最多到 `fitted/medium`，做不了"逐段完全归因"；`tokens` extra（minja/tokenizers/gguf）装了也不生效。`doctor` 现在会把这件事写在明面上，`onyx token explain` 入口也还没有 |
| `TOOL_EXEC_START` 事件的 `args_ref` 无处落库 | S17 真机 dry-run 查出来的 | 循环每次工具调用都写一个 args blob，但 schema 里没有任何列存它（`tool_call` 存的是内联 `args_json`）⇒ **每次工具循环泄漏一个小 blob**。`rotate` 能把孤儿回收掉，但正确的修法是别再写或者把它落库——别让"能删孤儿"掩盖"一直在造孤儿" |
| **真浏览器** e2e 仍为 0（S34 补的是数据通路那一半） | G6 / 附录 A | `-m e2e` 现在有 12 条：六页取数同源 + SSE 在真 HTTP 连接上被读到 + "摘掉 broker 会红"的自检。但本仓库**没有浏览器驱动**（实测：Python 侧无 playwright，web devDeps 只有 vitest + @testing-library + jsdom），所以**渲染本身**仍靠手工 `take_snapshot`。S23–S29 手工点出的八处缺陷里六处是数据通路（已被 e2e 覆盖），两处是前端映射（由 vitest 纯函数测试钉住）。要补的是驱动那一层（Playwright + 起 vite/serve），这属于"决定要分发/给别人用"之后才划算的投入 |
| `onyx serve --reload` | 附录 A 的自测命令 | 小：开发体验，非功能缺口（vite 的 HMR 已覆盖前端） |

### B. 明确推迟、带原因的（不是遗漏）
- **视觉任务仍推迟**：`/api/show` 之外还需要"图片 token 怎么计"与 `cached_tokens` 是否存在
  两个未决实测（U8/U9），否则分数会建在猜出来的 token 数上。
  评测任务本身已覆盖四类：`structured_extraction`（**S30**）、`instruction_following`（**S31**）、
  `long_context`（**S32**）、`semantic_similarity`（**S33**），加上 M4 的两个，共 6 个。
- **多轮/工具链长任务**：runner 支持 k 次采样，但样本的循环深度只有 2 步——
  要测"长链条工具使用"得先造那类数据，那是另一个量级的考卷工程。
- **长上下文的区分度到头了（S32 留下的口子）**：16k 档在 qwen3.5:9b 上是 9/9 全对、
  first/middle/last 无差异 ⇒ 这一档测不出模型间的高下。要拉开得加 32k+ 档，
  而本机 16GB 在 `num_ctx=20480` 已占 5.73G 显存，**窗口上限由卡决定**。
  加档需要一台窗口更大的机器，或者接受"这一列只能证明 ≤16k 可用"。
- `ollama_builtin` 执行器：必须先有 P21 的实测结论（模板里内建工具到底怎么渲染），
  否则做了也是猜的。`onyx tools contract` 那一列会显示"未实现，计划在 S16+"。
- `onyx.graders` 扩展点：见上表（没有消费点就先不建）。
- `onyx providers add/list`：多 provider 并存时的登记入口。现在 `--provider` +
  `plugins_example` + `.data` 里一张 `provider` 表就够用；等真出现"同一台机器常驻 3 个引擎"再做配置层。

### C. 已知边界（会被误读成 bug 的那些）
- `instruction_following` 的"拒答"是**启发式**：只认自我声明式短语（`抱歉`/`我无法`/`不能回答`…）
  且要求短于 60 个可见字符，并且**全约束满足时一律不改判**。仍然可能漏掉换了说法的拒答
  （"这个我做不到哦"），漏掉的就是普通的不遵循——分数仍按实测走，只是不会单独出现在 `refusal_rate` 里。
- `long_context` 的截断判据有一条**写在明面上的假设**：一个汉字至少值 0.5 token。
  实测 qwen3.5:9b 是 0.68（健康样本 16,740 tok vs 下限 13,525 ⇒ 1.24 倍余量），
  真被切时是 2,050（0.15 倍）。所以：换到比 0.5 tok/字更密的中文 tokenizer 会**误踢正常样本**，
  而纯英文/数字的 prompt 算不出下限（`min_prompt_tokens=0`）⇒ **不做截断判断**，
  这时"有没有被切"只能靠窗口配置自己保证。这两个方向都只会"少测一条"，不会把没测说成不会。
- `structured_extraction` 真机上 `date` 是 3/12 错，逐条看过都是**模型的错**，不是判据的错：
  两条把「二月二十八日」写成锚定日本身（`2026-03-01`），一条把「下周三」算成 `2026-03-08`。
  另有一处**已经露头但还没算进分数的考卷含糊**：难例"赵敏**昨天**下了一单，**后天**再退"
  模型答的是 `2026-03-03`（它取了后天），而期望值是昨天——本轮这条因为多抽了一个 `"event"`
  字段先落在 `schema_valid_rate` 那一层，日期判据没参与。规则"两个相对时间取第一个"
  目前只写在数据注释里，提示词没说。要么把它写进提示词，要么拆句，
  **不许悄悄改期望值让分数变好看**。
- **一个 `dataset_id` 下面可能留着上一版的样本行**（S32 之后库里 `longctx_zh-v1` 就是 18 行，
  而 `dataset.n_cases` 是 9）：case id 是内容哈希，生成器一改就是一批新 id，
  而**历史 grade 还靠旧行回答"当时考的是哪一条"**。所以重写样本时只删
  没被任何 grade 引用过的（`prune_stale_cases`）。读法：`n_cases` 是**当前这份考卷几条**，
  `list_cases` 里多出来的是历史；不是数字对不上。
- 打分口径只有 gen-based（API-only 拿不到受约束 logprob），评测输出会固定写"不可与公开 leaderboard 直接比较"。
- 兼容通道（vLLM / LM Studio / `/v1`）**没有分段时序** ⇒ TTFT/TPS 一律「—」；
  `structured_output` / `stream_usage` 长期是 `?`（未实测）而不是 `✗`——探针套件目前依赖 Ollama 原生端点。
- 非 ollama provider 的 `base_url` 目前仍记录 CLI `--url` 的默认值
  （修它要把散在 6 处的默认值提成常量并区分"用户没填"）。
- `RECONCILED` 事件在契约与 `PAYLOAD_REQUIRED` 里存在、token visitor 也消费它，
  **但 gateway 从不发**（采信在 observer 内部算完直接落库）⇒ OTLP 导出不带"采信来源"，
  只带各来源原始报告。这是事件契约与生产侧的一处不一致，还没修。
- 小样本的 CI 会很宽（97 条 case 的 must_call_acc 区间 0.528–0.736），
  `low_confidence` 阈值：单跑 <100 case、配对 <30 对。
- 评测/Playground/live 测试在 GPU 上互斥（机器级锁），并发起会排队而不是跑错。

---

## 4. 建议的下一步顺序

要成为"完整产品"的差距分析与 M7–M12 排期见 [`ROADMAP.md`](ROADMAP.md)（含三条需要决策的问题）。


1. ~~M7 跑得住~~（S17 保留策略 / S18 可验证备份 / S19 体检补齐 + 迁移前自动快照 + 体积曲线，全部达成）。
   ~~M8 配置与发行~~（S20 `onyx.toml` / S21 token 姿态 / S22 版本 + CHANGELOG + CI）。
   ~~M9 操作闭环~~（S23 界面发起评测 / S24 数据集导入与读回 / S25 Tool Bench 页 /
   S26 模型治理出口与被中断运行的续跑入口，**全部达成**）。
   ~~M10 观测触达~~（S27 告警内核 + 文件出口 + `alert_trigger` / S28 通用 webhook（第三条显式网络例外）/
   S29 Fleet 可见性 + doctor「告警」项 + **一进程一引擎定案**，**全部达成**）。
2. ~~`-m e2e` 用例~~（**S34 已交付数据通路那一半**：六页取数互相核对 + SSE 在真 HTTP 连接上被读到
   + "摘掉 broker 就会红"的自检 + CI 独立一步并钉进门禁清单）。
   剩下的部分是**真浏览器驱动**（Playwright 起 vite + serve），它仍欠着——见上表那一行。
3. ~~M11 评测资产（S30–S33）~~ **全部达成**：`structured_extraction` 45 条（真机 `score 0.893`）、
   `instruction_following` 39 条 / 10 种可机械检查的约束（`0.920 / 0.916 / 0.641` 三口径分叉）、
   `long_context` 9 条三档长文 + 干扰项（`1.000`，负控制 `--num-ctx 4096` 整场 `score —`）、
   `semantic_similarity` 12 题排序（`0.833`，`anti_first_rate 0.167`，并新建了整条 embedding 通路）。
   矩阵现在 **6 列**。剩下的两项都在 M12：**S35**（覆盖率"只防跌"基线 + 契约矩阵真 stdio 列）、
   **S36**（`onyx perf` 基线）；视觉任务仍等 U8/U9 的未决实测。
4. **`token explain` + `report usage`**：都属于"每天都在用但入口缺失"。
5. C 组三条一致性（`RECONCILED` 不发、`base_url` 默认值、兼容通道探针覆盖）适合凑成一次"口径一致性"清理。

核对方式（本文数字的来源）：

```bash
uv run onyx --help && for g in db probe models traces tools eval; do uv run onyx $g --help; done
uv run python -c "from onyx.api.app import create_app; print(len(create_app(db_path=':memory:', gpu_lock_path='x.lock').openapi()['paths']))"
uv run onyx plugins            # 六个扩展点的实际装配（内建/插件/覆盖/坏插件）
uv run python -c "from onyx.store.db import Database; d=Database(':memory:'); print(d.version(), d.table_names())"
grep -c 'AnomalySpec("' onyx/obs/anomalies.py        # 23
uv run onyx probe list           # 6 个探针
uv run pytest -m e2e --collect-only   # 0 个 e2e 用例
```
