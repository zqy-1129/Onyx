# 从"能用的内部工具"到"完整产品"——差距与计划

核对时间 **2026-10-04**（HEAD 在 M6 之后）。现状清单见 [`STATUS.md`](STATUS.md)，
本文只回答一个问题：**要成为一个别人也能装、能长期跑、敢拿去给别人看的产品，还差什么，按什么顺序补。**

文中的"证据"都是当场查出来的（命令附后），不是推测。

---

## 0. 先把"完整产品"定义清楚

Onyx 的目标形态不是 SaaS，也不是通用 LLM 应用框架，而是：

> **一台（或几台）自己管理的机器上，本地模型的观测台 + 评测台。**
> 使用者是模型/infra 工程师本人，或小团队内共享的一个只读看板。

由此推出三条产品级判据（缺任何一条都算不上"完整"）：

| 判据 | 含义 | 今天 |
|---|---|---|
| **跑得住** | 数据不会把磁盘吃掉，坏了能恢复，出问题能定位 | ⚠️ 保留策略（S17）与可验证备份（S18）已交付；仍缺磁盘与 tokenizer 体检 |
| **用得起来** | 日常动作不必背 CLI 参数，长任务能看见进度、能取消 | ⚠️ 观测/评测的**写操作全在 CLI**，界面只能读 + Playground |
| **给别人看** | 部署形态、权限边界、版本与升级是明确的，不靠口头知识 | ⚠️ 配置（S20）与非回环 token 姿态（S21）已交付；仍缺 CI/发行物/CHANGELOG |

**核心判断**：功能面其实已经很宽（34 条 CLI、19 个 API 端点、9 个页面、4 种执行器、
6 个扩展点、24 条实测结论），**缺的不是功能，是"运行多年"的外壳**。
所以下面六个缺口里，只有 G3/G5 是加功能，其余四项是把已有能力变成可交付产品。

---

## 1. 六个缺口域（含证据与后果）

### G1 数据生命周期与自愈（缺得最实在）
| 证据 | 现状 | 后果 |
|---|---|---|
| ~~`grep -rn "prune\|retention\|vacuum" onyx/` → 0~~ | **S17 已交付**：`onyx rotate` 默认 dry-run，摘五列重引用、回收无主 blob、每次运行落 `retention_run` 留痕 | 真机第一次 dry-run 就报出 6 个无人引用的 blob，并顺带查出 `args_ref` 泄漏（见 STATUS A 组）；`.data` 现在能自我约束 |
| ~~`onyx db` 只有 `init` / `info`~~ | **S18 已交付**：`db backup`（WAL 一致快照 + 只装被引用的 blob + manifest）与 `db verify-backup`（逐字节 sha256、行数、"引用能否在备份里解析"） | "备份存在"与"备份可用"从此是两件事，而 verify 负责后者；只备库不备证据的备份会被那条具名检查抓出来 |
| ~~`doctor` 6 项里没有磁盘余量、tokenizer 档位可用性~~ | **S19 已交付**：两项都在，且 blob 完整性项与 rotate 共用同一份引用清单 | 磁盘写满时现在会红（阈值 2 GiB）；档位项直接说明"`hf_tokenizer`/`gguf_vocab` 本版本未实现，`tokens` extra 装了也不生效"——这句话以前只藏在 PROBES P9 里 |
| 迁移只有 `0001`→`0005` 单向 | **S19 部分改善**：真要改 schema 前自动留 `backups/pre-migration-v{旧}.sqlite`（WAL 一致快照） | 仍然没有降级路径，但升级失败至少有一个明确的回滚点，不用手改库 |

**出口判据**：~~`onyx rotate` 报得出会删多少、释放多少字节~~（S17 达成：默认 dry-run，
真机报出 6 个可回收 blob / 955 B，且每次运行落 `retention_run`）；
~~`onyx db backup` + `verify-backup` 比对行数与 blob 摘要~~（S18 达成：真机 1231 个 blob 逐个重算
sha256 全对得上，行数/schema/引用可解析共 9 项检查全绿）；
~~故意删一个 blob 文件后 `doctor` 能指名道姓报出来（不是笼统 500）~~（S18 达成，副本实测退出码 1）；
~~`.data` 大小有上限曲线可查~~（S19 达成：`onyx db sizes` 从 `retention_run` 取点，跨度不足一天就说"问不出来"）；
~~磁盘余量与 tokenizer 档位体检 + 迁移前自动备份~~（S19 达成：升级前留
`backups/pre-migration-v{旧}.sqlite`，在 `.data` 副本上真升级 v6→v7 验过回滚点完好）。

### G2 配置、部署与发行（"给别人看"的前提）
| 证据 | 现状 |
|---|---|
| ~~`grep onyx.yaml` 只出现在 DESIGN §13~~ | **S20 已交付**（落地为 `onyx.toml`，理由见 DESIGN §13）：provider / GPU 锁 / 保留窗口 / sandbox / serve 绑定收进一份文件，优先级一条规则 flag > 环境 > 文件 > 默认，`onyx config show` 标出每一项来自哪一层 | 剩下的是发行面：鉴权姿态、CI、LICENSE/CHANGELOG 仍然没有 |
| ~~`grep APIKey\|Authorization onyx/api` → 0~~ | **S21 已交付**：`--host` 非回环且无 token ⇒ 拒绝启动（退出码 2）；`--read-only` 分开"共享看板"与"共享操作台"；SSE 走 `?token=`（`EventSource` 设不了头），代价写进 README 与错误 hint | 剩下的是团队共享那一档的完整形态（多 token、审计谁在操作）——只有真决定共享给同事时才做 |
| ~~无 `.github/`、无 `CHANGELOG.md`、`version = "0.1.0"` 从未升过~~ | **S22 已交付**：`ci.yml` 跑五道门 + 前端 + 安装冒烟；`CHANGELOG.md`（含版本策略）；版本号单点定义并升到 `0.8.0` | 仍未做：LICENSE / Docker / pipx 发行——它们都挂在"是否对外发行"那个未决问题上 |

**出口判据**：~~一份 `onyx.yaml` 能声明 provider、锁路径、保留策略、sandbox 白名单，
且 `doctor` 会报告"配置项写了但没生效"~~（S20 达成，落地为 `onyx.toml`；
doctor 现在会指名未知键与坏类型，真机 `serve.pprt` → 退出码 1）；
`--host` 非回环时**必须**显式给出 token 或 `--allow-insecure-local`，否则拒绝启动（S21 达成：
真机 `--host 0.0.0.0` 无 token ⇒ 端口根本没绑、退出码 2）；
CI 跑 ruff + lint-imports + 离线 + 边界 + 覆盖率 + 前端 + 安装冒烟（S22 达成；
`live` 两档明确不在 CI 跑——它要真引擎真 GPU）；一条 `uv tool install` 就装得上（已在本机用
干净环境验证：`uv build` → 隔离目录 `uv tool install --from dist/*.whl` → `version/db init/chat/doctor` 全通）。

### G3 操作闭环：从 CLI 工具到界面产品
证据（S23 之前）：`@router.post` 只有两个端点（`/api/playground/chat`、`/api/admin/models/unload`）。
即：**评测、数据集导入、工具注册、契约矩阵、矩阵报告导出，全都要离开浏览器用命令行做**。
另有两处能力"实现了但没出口"：`AdminProvider.pull/delete` 与 runner 的 `cancelled` 状态——
`onyx models` 只有 `sync`/`ls`（拉模型/删模型没有命令），跑了一半的评测**只能 Ctrl-C**，
库里留下 `running` 僵尸记录（矩阵会排除它，但没人知道那次跑的进度）。

**S23 已交付"发起评测"这一半**：写操作从 2 个变成 4 个
（`POST /api/runs`、`POST /api/runs/{id}/cancel` 加上原有两个），评测进程内单飞排队 +
机器级 GPU 锁同源，界面能看到逐条进度、排队位置、"谁在占 GPU + ETA"，三条路都能取消
（排队中 / 等锁中 / 跑一半）。僵尸回收也落了：服务启动时把上次进程留下的 `running` 行
标成 `error` 并写明原因，但**只动 `trigger=api` 且锁空闲的那些**。
真机验证：mock 引擎上点一次按钮跑完 944 条并自动下钻 grade；占住锁后取消 ⇒ 状态 cancelled
且库里没有那条 run。

**出口判据**（逐条核对）：
- ✅ 界面上能发起评测（选任务/模型/k/limit/seed/子集/卸掉其它模型）并看到实时进度与取消
- ✅ 能导入数据集并看到来历/revision/条数（S24：`POST /api/datasets` 收 JSONL **文本**——
  路径来自请求体就是任意文件读；本地文件由浏览器读成文本）。同时补上断掉的闭环：
  `eval import --id x` 之后 `--dataset x` 以前报"未知数据集"，而样本就在同一张库里
- ✅ Tool Bench 页：注册表 / 审计 / 开销 / 契约矩阵 / 运行历史（S25，五个只读 GET + 一页面板；
  矩阵是"点一次才跑"，因为它是真在各执行器上跑一遍断言）。
  顺带补上一个真实缺口：`tool_run` 表、它的保留规则与"运行历史"位置都存在很久，
  **但从来没有写入方** —— 现在 `onyx tools run` 会落一行，而确定性/幂等两列留「—」
  （跑一次测不出这两件事，填上就是猜）
- ✅ `onyx models pull/rm` 与界面同源（S26：模型页治理面板 + `POST /api/admin/models/pull`·`/rm`，
  和 CLI 同一个 `AdminProvider`。通道不实现控制面就 **501 并说清为什么不做个假的**；
  未勾选确认时三个按钮都点不动）
- ✅ 被中断的 run 显示为 `cancelled`/`error` 且**在界面上可续跑**（S26：运行页「续跑这条」→
  预填表单 → 同一条 `POST /api/runs` 带 `resume_run_id`。实测一条中断在 12 条的 run 续到 236，
  id 不变、请求数只增 224。同时修掉两处口径：`n_cases` 不再被收尾缩成已评条数（`12/236` 而不是
  `12/12`），Windows 上删不掉的锁文件改成把心跳推到过去，而不是留一条让后面人白等 600 秒的假持有）
这块最大的风险是"写操作把 GPU 抢了"：所有触发型端点必须走同一把机器级锁，
并在页面上显示"谁在占"（`/api/gpu` 已有数据）。S23 按这条做了：`AppState` 把同一把锁的
路径与 stale 阈值注入提交服务，测试 `test_submitted_runs_share_the_apps_gpu_lock` 钉住"锁路径不许分叉"。

### G4 观测的产品化闭环（只有记录，没有触达）✅ 已达成（M10 / S27–S29）
证据（改动前）：`grep webhook\|notify` → 0。23 种异常码全部只落库；`obs/visitors` 的设计目标是"新异常规则、成本模型、**告警**"，
但没有任何一条路径会在出问题时通知我。此外：
- 一个 serve 进程只绑一个 provider（`state.runtime.provider`）⇒ 同机多引擎要开多份服务，
  而 provider 表其实已经支持多行——**数据模型领先于使用路径**；
- 成本 visitor 只算 token，不算钱（本地无单价概念，但混合 offload/时间成本是可以算的"代价"）。

**出口判据**（逐条核对）：
- ✅ `onyx.toml` 的 `[alerts]` 能配"什么级别/哪些码、窗口内几次、cooldown 多久、发到哪"（S27）。
  两个出口都在：本地通知文件 `<数据目录>/alerts/alerts-YYYYMMDD.jsonl` + 通用 webhook（S28，
  URL 优先环境变量 `ONYX_ALERT_WEBHOOK_URL`，`config show` 与库里都只出去 query 的 URL）
- ✅ `onyx alerts ls` 看得到触发历史（`alert_trigger` 表，schema v7）。
  它记"命中 + 尝试投递"的合取，被 cooldown 挡的不写行；空表也打印当前生效判据与出处
- ✅ Fleet 页顶部有"近 1 小时有 N 条 error 级异常"那一行，并且**同时**说通知系统自己的状态
  （没装配 / 没出口 / 轮询出错 / 在跑，四种情况四句话）；页面还有「告警触发」面板，样本可下钻到真实 trace
- ✅ 多 provider 观测**定案**：写死一进程一引擎，多实例各配自己的 `ONYX_DATA_DIR`；
  理由与代价记在 `DESIGN §8.6`（含"共用一个 `.data` 未实测"这句实话）
- ✅ 判定读库轮询而不是在请求路径里发消息：观测者不许影响被测（S27 的核心决策）
- ⬜ 成本 visitor 仍然只算 token 不算钱——**这一条本次没做**，它属于"代价模型"而不是"触达"，
  留在 M11/M12 之后再说（本地没有单价，但 offload 比例与 GPU 分钟数是可以算的代价）

实测（可复查）：本地假接收端 + `ONYX_ALERT_WEBHOOK_URL=… onyx serve` ⇒ 插一条 `CONTEXT_OVERFLOW`
后一个轮询周期内真收到 POST，库里两行 `sent`；停掉接收端 ⇒ webhook 行 `failed` 带 ConnectError
而文件行照样 `sent`；随后 cooldown 内历史不再增长。

### G5 评测资产（护城河，但要小心变成清单收集）🟡 S30 已达成一条
现状：**3 个任务**（意图 236 条、工具调用 97 条、**结构化抽取 45 条（S30）**）、BFCL 导入器、
配对回归与矩阵，外加一条**所有任务共用**的契约测试（`tests/contract/test_task_contract.py`）。
剩下的评测面：`instruction_following`（S31）、长上下文中文集（S32，本机 16GB 能测出的最有价值维度）、
embedding（S33：能力位有 `EMBED`、此前无任务）、**视觉**（U8 未定：图片 token 怎么计）、
**多轮/工具链长任务**（runner 支持 k，但循环深度只有 2 步样本）。

**出口判据**（每个新任务都要求"三个同源"，逐条核对）：
- ✅ **`metric_names` 与 `aggregate` 双向同源**（S30）：契约测试对 `specs()` 的每个任务跑，
  空聚合与有样本两种都断言。它当场查出两个内置任务少声明了 9–10 个真的会产出的指标
  （分母与 verdict 分布），于是"看板提前建列"这句话第一次是真的
- ✅ **CI 分母同源**（S30）：每个 `X_ci` 必须有同名的 `X` 且 `point == X`、`n` 不超过参与聚合的样本数
- ✅ **下钻 trace 同源**（S30）：mock 真跑那条 run 的每条 grade 都带着能在 `trace` 表里查到的 trace_id
- ✅ 真机跑一次带分母与 CI（S30）：qwen3.5:9b × 45 条 ⇒ `score 0.893 [0.786–1.000]（n=28 case）`、
  `exact_object_rate 0.667`、负样本 5/5；三轮的 `git_rev` 不同，分数变化能归因到提交
- 🟡 **矩阵从 2 列长到 5–6 列**：现在 3 列（每格带 `n_judged/n_total` 与 coverage），S31–S33 补剩下的
- ✅ **雷达图出现**：`onyx eval report --format html` 那张 SVG 的条件是"任务数 ≥3"，第三个任务落地后真的画出来了
- 不加"任务数量"指标，因为一个能解释的 97 条比一个说不清的 3000 条有用。
  S30 的教训反而更值钱：**考卷没说清答题格式时，分数会看起来像能力问题**——
  同一台机器同一个模型，`score` 从 0.129 跳到 0.893 的全部原因就是提示词补齐了归一化口径

### G6 质量与可维护性（防倒退）🟡 S34 已达成一条
证据与后果（改动前）：
- **`-m e2e` 用例数 0**。目前浏览器验证是手工 `take_snapshot`——恰恰是它发现了 SSE 静默失效，
  说明这类问题真实存在，但也意味着**没有回归保护**：任何一次改版都可能把它改回去而全绿。
- 前端 `aria-*` / `role=` 一共只有 3 处；文案全部写死中文（内部用没问题，若要分发要抽 i18n）。
- 契约矩阵 4 列 × 8 断言，但 `mcp` 列用离线假连接；真子进程路径只在 `tests/unit/test_tools_mcp_stdio.py`，
  **没有进契约矩阵**（换 MCP SDK/换 server 时不会红）。
- 无覆盖率门禁、无性能基线（`tools cost`、`eval run` 的耗时目前靠人看）。

**出口判据**（逐条核对）：
- ✅ **6 页各一条 e2e，CI 跑通，且能抓到一次人为注入的 SSE 断链**（S34）：
  `tests/e2e/` 12 条 —— 六页取数互相核对（Fleet 的窗口 == Traces 页能数出的条数、
  Ledger 分桶求和 == 总数、每条 grade 的 trace_id 查得到真实 trace、矩阵每格的分母 == CI 的 n、
  劣化清单每条两个 trace 都可下钻）；SSE 用**真 uvicorn + 真 httpx** 读帧
  （TestClient 会缓冲流式响应：broker `published=9` 而客户端读到 0 行，那是装置的限制不是应用的 bug），
  外加一条"把 broker 从总线上摘掉 ⇒ 只剩 hello"的自检 ⇒ 守着 SSE 的那条断言本身是可用的。
  CI 里是独立一步（默认 addopts 把 `-m e2e` deselect 掉了），并被 `test_release_surface.py` 钉住
- ⬜ 覆盖率"只防跌"的基线（S35）：floor 80 / 实测 90 ⇒ 收紧到 85 并写明余量理由
- ⬜ 契约矩阵增加"真 stdio"变体一列（S35）
- ⬜ `onyx perf` 有一组可比的吞吐/延迟基线（S36）；基线的**条件**（引擎版本/模型/并发/冷热）必须一起落库
- ⬜ i18n 抽取：挂在"是否对外发行"上（ROADMAP §3 的并行/裁剪建议里明确它可以长期搁置）

**这条边界要说清，别让"e2e 全绿"被读成"界面没问题"**：本仓库没有浏览器驱动，
所以渲染本身仍靠手工 `take_snapshot`。S34 覆盖的是"页面取的那份数据"，
前端映射由 vitest 的纯函数测试钉住（五张页面各自的说法）。

---

## 2. 里程碑计划（M7–M12，按依赖排序）

约定沿用 `IMPLEMENTATION.md`：每步有产出文件、自测命令、验收 DoD、提交点；一步一 commit；
质量门（ruff / lint-imports / 离线+live / 边界脚本）每步都过。

| 里程碑 | 内容 | 步 | 出口判据 |
|---|---|---|---|
| **M7 跑得住** | G1 全部：`rotate`、`db backup/verify-backup`、`doctor` 补磁盘 + tokenizer 档位、迁移前自动备份、`.data` 体积报告 | **S17–S19 ✅** | 保留策略可 `--dry-run` 且落库审计 ✅ · 备份可验证恢复 ✅ · `doctor` 在人为破坏后报具体项 ✅（真机：删一个 blob ⇒ 具名 + 退出码 1）· 磁盘与档位两项体检 ✅ · `.data` 曲线 ✅ |
| **M8 配置与发行** | G2：`onyx.toml`（provider/锁/保留/白名单）+ 优先级与"写了不生效"检查；`--host` 非回环强制 token；CHANGELOG + 版本策略 + GitHub Actions；`uv tool install` 冒烟 | **S20–S22 ✅** | 配置项与 flag 冲突时有明确解释 ✅ · 非回环无 token 起不来 ✅ · 新机器一条命令装好并 `doctor` 全绿 ✅（本机干净环境验过）· CI 能挡住 lint-imports/边界脚本违规 ✅（workflow 就位；仓库尚无远端 ⇒ 待首次真跑）· LICENSE/Docker/pipx 挂在"是否对外发行"这个未决问题上 |
| **M9 操作闭环** ✅ | G3：界面发起评测 ✅、数据集导入 ✅、工具注册/审计页（Tool Bench）✅、`models pull/rm` ✅、被中断运行可续跑 ✅ | **S23–S26 ✅** | 不发一句命令就能完成"导入数据 → 选模型 → 跑评测 → 看矩阵 → 下钻 trace"（✅ 浏览器实测走通）并审计工具库（✅ 含契约矩阵三态）；被中断的 run 状态正确（✅ 含重启后的僵尸回收）且**在界面上可续跑**（✅ 实测中断在 12 条的 run 续到 236，id 与成本连续）；触发型端点全部过机器级锁（✅） |
| **M10 观测触达** ✅ | G4：告警规则 + 两个出口（本地文件 / 通用 webhook）+ 触发历史与界面可见性 + 多引擎形态定案 | **S27–S29 ✅** | 人为造一条 `CONTEXT_OVERFLOW` 在 1 分钟内收到通知 ✅（真机：本地假接收端真收到 POST，文件出口同刻落一行）· 界面看得见"为什么触发" ✅（`alert_trigger` 存命中当时的判据快照 + Fleet 顶部一行）· 一进程一引擎定案 ✅（DESIGN §8.6，含"共用 `.data` 未实测"的实话） |
| **M11 评测资产** 🟡 | G5：`structured_extraction` ✅、`instruction_following`、长上下文中文集、embedding 任务（先解决 U8/U9 未决实测再上视觉） | **S30 ✅** / S31–S33 | 每个新任务三条同源断言全绿 ✅（`tests/contract/test_task_contract.py` 对 `specs()` 每个任务参数化跑，顺出两处早就存在的漂移）· 真机跑一次带分母与 CI ✅（qwen3.5:9b × 45：`score 0.893 [0.786–1.000]（n=28）`、`exact_object_rate 0.667`、负样本 5/5，三轮 `git_rev` 可归因）· 矩阵从 2 列长到 5–6 列 🟡（现在 3 列，每格带 `28/45` 这种分母）· 雷达图出现 ✅（导出 HTML 那张 SVG，阈值 3 个任务已达到） |
| **M12 防倒退** 🟡 | G6：6 页 e2e ✅、覆盖率基线、契约矩阵真 stdio 变体、性能基线、i18n 抽取（仅在确定要分发时做） | S34 ✅ / S35–S36 | CI 上 e2e 跑通且能抓到一次人为注入的 SSE 断链 ✅（12 条：六页取数同源 + 真 uvicorn 读 SSE + 摘掉 broker 的自检）· 覆盖率有"只防跌"的基线 ⬜ S35 · `onyx perf` 有可比基线 ⬜ S36 |

**并行/裁剪建议**
- M7 与 M8 可部分并行（都碰 CLI，但文件不冲突），且都不依赖 M9。
- **M9 之前不要动 G5**：没有界面发起评测时，多加任务只会增加 CLI 记忆负担，护城河要等到"别人能自己点"才兑现。
- M12 的 i18n 是唯一可长期搁置的项——除非 M8 决定真要对外发行。
- 若只做一件事就停：**M7**（数据生命周期）。它是唯一会随时间必然恶化的缺口，其余都是"缺一块功能"，
  而它到点是"整个 `.data` 不可用"。

---

## 3. 明确不做（写下来，免得每次都重新讨论）

- 多租户 / SaaS / 账号体系：目标是自托管单实例，用户体系只有"局域网只读 token"这一档。
- 训练、微调、量化流水线：Onyx 观测与评测模型，不生产模型。
- 通用 agent 框架 / 编排语言：DESIGN §12 的论证仍然成立（框架会把计量口径藏起来）。
- 云端 API provider 的账单系统：`openai-compat` 只作为自托管服务器（vLLM / LM Studio / Xinference）的通道。
- GPU 编排 / 分布式：单机 16GB 显存是产品的物理边界，锁与 offload 检测就是为此存在的。
- 图表库与组件库：手写 SVG/CSS 是刻意的（高密度看板的像素控制），不因"产品化"而引入依赖。

---

## 4. 需要三个决策（会显著改变工作量）

1. **部署形态**：只给自己用（127.0.0.1，M8 只需 0.5 天做姿态声明）还是团队内共享看板
   （要 token、只读权限、README 部署章节，约 +2 天）。
2. **是否对外发行**：决定 CHANGELOG/版本策略/i18n/打包（pipx、独立可执行文件）值不值——不发行的话 M8 只做配置与 CI。
3. **评测是否要能在界面发起长任务**：这是 M9 最大的一块（含队列、进度、取消、锁排队）。
   若你接受"长任务继续用 CLI、界面只做观测与查看"，M9 可以砍到只剩 Tool Bench 页与 `models pull/rm`。

---

## 5. 核对方式

```bash
grep -rn "prune\|retention\|vacuum" --include=*.py onyx/ | wc -l        # 32：保留策略已落地（S17）
grep -rln "create_backup\|verify_backup" --include=*.py onyx/ | wc -l    # 3：备份与验证（S18）
onyx doctor | grep -c "^│[✓✗]"                                          # 9：体检项（S19 磁盘/档位、S20 配置）
grep -rn "@router.post" onyx/api/routes/*.py | wc -l                     # 2：界面只有两个写操作
grep -rn "APIKey\|Authorization" onyx/api --include=*.py | wc -l         # 2：token 闸已落地（S21，onyx/api/auth.py）
grep -rln "webhook\|notify" --include=*.py onyx/ | wc -l                 # 0：异常只落库
ls .github/workflows | wc -l ; ls LICENSE CHANGELOG.md 2>/dev/null | wc -l   # 1 / 1：CI 与 CHANGELOG 已就位；LICENSE 仍未选
ls onyx/config.py onyx.example.toml 2>/dev/null | wc -l                   # 2：部署配置已落地（S20，TOML 而非 YAML）
uv run pytest -m e2e --collect-only 2>&1 | grep collected        # "no tests collected" ⇒ e2e 用例 0 个
grep -c "aria-\|role=\"" -r onyx/web/src --include=*.tsx                 # 3 处
```
