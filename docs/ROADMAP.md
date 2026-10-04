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
| **跑得住** | 数据不会把磁盘吃掉，坏了能恢复，出问题能定位 | ❌ 无保留策略、无备份校验、磁盘不体检 |
| **用得起来** | 日常动作不必背 CLI 参数，长任务能看见进度、能取消 | ⚠️ 观测/评测的**写操作全在 CLI**，界面只能读 + Playground |
| **给别人看** | 部署形态、权限边界、版本与升级是明确的，不靠口头知识 | ❌ 无鉴权姿态、无 `onyx.yaml`、无 CI/发行物/CHANGELOG |

**核心判断**：功能面其实已经很宽（34 条 CLI、19 个 API 端点、9 个页面、4 种执行器、
6 个扩展点、24 条实测结论），**缺的不是功能，是"运行多年"的外壳**。
所以下面六个缺口里，只有 G3/G5 是加功能，其余四项是把已有能力变成可交付产品。

---

## 1. 六个缺口域（含证据与后果）

### G1 数据生命周期与自愈（缺得最实在）
| 证据 | 现状 | 后果 |
|---|---|---|
| `grep -rn "prune\|retention\|vacuum" onyx/` → 0 | 无任何保留策略，`raw_response_ref` 指向的 blob 永久保留 | 本机跑评测**最先撞的墙**：`.data` 无上限增长，写满那天是崩溃而不是提示 |
| `onyx db` 只有 `init` / `info` | 无 `backup` / `verify-backup`（S7 计划里有） | 观测数据是"证据"，但没有可验证的恢复手段；`doctor` 只能查引用完整，不能证明可恢复 |
| `doctor` 6 项里没有磁盘余量、tokenizer 档位可用性 | S7 检查项清单承诺过 | 磁盘写满是本地部署最典型故障；tokenizer 档位缺失会让 `hf_tokenizer/gguf_vocab` 两档静默不可用（只剩 heuristic/low） |
| 迁移只有 `0001`→`0005` 单向 | 无降级/校验路径 | 升级失败只能手改库 |

**出口判据**：`onyx rotate --trace-after 90d --raw-after 30d --dry-run` 报得出会删多少、
释放多少字节；`onyx db backup` + `verify-backup` 比对行数与 blob 摘要；
故意删一个 blob 文件后 `doctor` 能指名道姓报出来（不是笼统 500）；`.data` 大小有上限曲线可查。

### G2 配置、部署与发行（"给别人看"的前提）
| 证据 | 现状 |
|---|---|
| `grep onyx.yaml` 只出现在 DESIGN §13 | **配置系统不存在**。只有 `ONYX_DATA_DIR` + 每条命令各自的 flag；providers/keep_alive/GPU 锁/数据集缓存/sandbox 白名单全是散参数 |
| `grep APIKey\|Authorization onyx/api` → 0 | 无鉴权。默认绑 127.0.0.1 是"隐形安全边界"，一旦 `--host 0.0.0.0` 就变成任何局域网的人能 unload 你的模型、打你的 GPU |
| 无 `.github/`、无 `LICENSE`、无 `CHANGELOG.md`、无 `Dockerfile`；`version = "0.1.0"` 从未升过 | 没有 CI 门禁、没有发行物、没有升级说明；"别人怎么装"只能读 README 猜 |

**出口判据**：一份 `onyx.yaml` 能声明 provider、锁路径、保留策略、sandbox 白名单，
且 `doctor` 会报告"配置项写了但没生效"（这一条最容易烂，因为散落 flag 会悄悄覆盖配置）；
`--host` 非回环时**必须**显式给出 token 或 `--allow-insecure-local`，否则拒绝启动；
CI 跑 ruff + lint-imports + 离线 + live（有 runner 时）+ 边界脚本，README 顶部一条 `uv tool install` 就装得上。

### G3 操作闭环：从 CLI 工具到界面产品
证据：`@router.post` 只有两个端点（`/api/playground/chat`、`/api/admin/models/unload`）。
即：**评测、数据集导入、工具注册、契约矩阵、矩阵报告导出，全都要离开浏览器用命令行做**。
另有两处能力"实现了但没出口"：`AdminProvider.pull/delete` 与 runner 的 `cancelled` 状态——
`onyx models` 只有 `sync`/`ls`（拉模型/删模型没有命令），跑了一半的评测**只能 Ctrl-C**，
库里留下 `running` 僵尸记录（矩阵会排除它，但没人知道那次跑的进度）。

**出口判据**：界面上能发起评测（选任务/模型/k/limit）并看到实时进度与取消；
能导入数据集（上传 JSONL 或选内置生成器）并看到来历/revision/条数；
`onyx models pull/rm` 与界面同源；被中断的 run 显示为 `cancelled` 且可续跑（`--resume` 已有）。
这块最大的风险是"写操作把 GPU 抢了"：所有触发型端点必须走同一把机器级锁，
并在页面上显示"谁在占"（`/api/gpu` 已有数据）。

### G4 观测的产品化闭环（只有记录，没有触达）
证据：`grep webhook\|notify` → 0。23 种异常码全部只落库；`obs/visitors` 的设计目标是"新异常规则、成本模型、**告警**"，
但没有任何一条路径会在出问题时通知我。此外：
- 一个 serve 进程只绑一个 provider（`state.runtime.provider`）⇒ 同机多引擎要开多份服务，
  而 provider 表其实已经支持多行——**数据模型领先于使用路径**；
- 成本 visitor 只算 token，不算钱（本地无单价概念，但混合 offload/时间成本是可以算的"代价"）。

**出口判据**：`onyx.yaml` 里能配"什么异常、连续几次、发到哪"（先做两个出口：本地通知文件 + 一个通用 webhook），
`onyx alerts ls` 看得到触发历史；Fleet 页顶部有"当前有 N 条 error 级异常"的一行；
多 provider 观测（一个进程可绑多个 provider，或明确写死"一进程一引擎"并在文档里说明怎么起多个）。

### G5 评测资产（护城河，但要小心变成清单收集）
现状：2 个任务（意图 236 条、工具调用 97 条）、BFCL 导入器、配对回归与矩阵。
明确推迟的两个任务（`structured_extraction`、`instruction_following`）+ 完全没碰的评测面：
**视觉**（U8 未定：图片 token 怎么计）、**embedding**（能力位有 `EMBED`、无任务）、
**长上下文/中文长文**（本机 16GB 能测的最有价值维度）、**多轮/工具链长任务**（runner 支持 k，但循环深度只有 2 步样本）。

**出口判据**（每个新任务都要求"三个同源"）：`metric_names` 与 `aggregate` 同源、
CI 分母同源、下钻 trace 同源——这三条已有契约测试形状，直接复用。
不加"任务数量"指标，因为一个能解释的 97 条比一个说不清的 3000 条有用。

### G6 质量与可维护性（防倒退）
证据与后果：
- **`-m e2e` 用例数 0**。目前浏览器验证是手工 `take_snapshot`——恰恰是它发现了 SSE 静默失效，
  说明这类问题真实存在，但也意味着**没有回归保护**：任何一次改版都可能把它改回去而全绿。
- 前端 `aria-*` / `role=` 一共只有 3 处；文案全部写死中文（内部用没问题，若要分发要抽 i18n）。
- 契约矩阵 4 列 × 8 断言，但 `mcp` 列用离线假连接；真子进程路径只在 `tests/unit/test_tools_mcp_stdio.py`，
  **没有进契约矩阵**（换 MCP SDK/换 server 时不会红）。
- 无覆盖率门禁、无性能基线（`tools cost`、`eval run` 的耗时目前靠人看）。

**出口判据**：6 页各一条 e2e（含 SSE 联通、矩阵分母显示、劣化清单下钻三处最脆弱的）；
CI 上覆盖率有基线数字（不追高，只防跌）；契约矩阵增加"真 stdio"变体一列；
`onyx perf`（或 probe）留一组吞吐/延迟基线，改版能对比。

---

## 2. 里程碑计划（M7–M12，按依赖排序）

约定沿用 `IMPLEMENTATION.md`：每步有产出文件、自测命令、验收 DoD、提交点；一步一 commit；
质量门（ruff / lint-imports / 离线+live / 边界脚本）每步都过。

| 里程碑 | 内容 | 步 | 出口判据 |
|---|---|---|---|
| **M7 跑得住** | G1 全部：`rotate`、`db backup/verify-backup`、`doctor` 补磁盘 + tokenizer 档位、迁移前自动备份、`.data` 体积报告 | S17–S19 | 保留策略可 `--dry-run` 且落库审计（删了什么留痕）；备份可验证恢复；`doctor` 在人为破坏后报具体项 |
| **M8 配置与发行** | G2：`onyx.yaml`（provider/锁/保留/白名单）+ 优先级与"写了不生效"检查；`--host` 非回环强制 token；LICENSE + CHANGELOG + 版本策略；GitHub Actions 全门禁；`uv tool install .` 冒烟 | S20–S22 | 新机器一条命令装好并 `doctor` 全绿；CI 能挡住 lint-imports/边界脚本违规；配置项与 flag 冲突时有明确解释 |
| **M9 操作闭环** | G3：界面发起评测（锁排队 + 实时进度 + 取消）、数据集导入、工具注册/审计页（Tool Bench）、`models pull/rm` | S23–S26 | 不发一句命令就能完成"选模型 → 跑评测 → 看矩阵 → 下钻 trace"；被中断的 run 状态正确；触发型端点全部过机器级锁 |
| **M10 观测触达** | G4：告警规则 + 两个出口（本地文件 / 通用 webhook）+ 触发历史页；多引擎观测形态定案（要么一进程多 provider，要么文档化"多实例 + 汇总视图"） | S27–S29 | 人为造一条 `CONTEXT_OVERFLOW` 能在 1 分钟内收到通知并能在界面看到"为什么触发" |
| **M11 评测资产** | G5：`structured_extraction`、`instruction_following`、长上下文中文集、embedding 任务（先解决 U8/U9 未决实测再上视觉） | S30–S33 | 每个新任务三条同源断言全绿；真机跑一次带分母与 CI；矩阵从 2 列长到 5–6 列且雷达图出现 |
| **M12 防倒退** | G6：6 页 e2e、覆盖率基线、契约矩阵真 stdio 变体、性能基线、i18n 抽取（仅在确定要分发时做） | S34–S36 | CI 上 e2e 跑通且能抓到一次人为注入的 SSE 断链；`onyx perf` 有可比基线 |

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
grep -rn "prune\|retention\|vacuum" --include=*.py onyx/ | wc -l        # 0：无保留策略
grep -rn "@router.post" onyx/api/routes/*.py | wc -l                     # 2：界面只有两个写操作
grep -rn "APIKey\|Authorization" onyx/api --include=*.py | wc -l         # 0：无鉴权
grep -rln "webhook\|notify" --include=*.py onyx/ | wc -l                 # 0：异常只落库
ls .github 2>/dev/null | wc -l ; ls LICENSE CHANGELOG.md 2>/dev/null | wc -l   # 0 / 0
grep -n "onyx.yaml" docs/DESIGN.md | head -1                            # 承诺过，代码里没有
uv run pytest -m e2e --collect-only 2>&1 | grep -E "[0-9]+/1015 tests" # 0 个 e2e 用例
grep -c "aria-\|role=\"" -r onyx/web/src --include=*.tsx                 # 3 处
```
