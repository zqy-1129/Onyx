# 变更日志

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 的分组方式，
版本号遵循下面的**版本策略**。

## 版本策略

- **MAJOR 保持 0**：在项目对外发行之前不动。0.x 的语义是"接口会改，别假设稳定"，
  这与 Onyx 的现状一致（数据库迁移单向、CLI flag 仍在增删）。
- **MINOR = 里程碑**：`0.8.0` 对应 M8。里程碑是这套项目里唯一自然的"功能整包"单位，
  用它当次版本号，`git describe` 与看板上的 `app_version` 才互相对得上。
- **PATCH**：对外发行后用于修缺陷；在此之前不发（没有下游消费者就没有"兼容承诺"要维护）。
- 版本号只有 `onyx/__init__.py` 一处定义，`pyproject.toml` 通过 hatchling 的 dynamic version
  读它。`eval_run.app_version` 落的就是这个值——配对回归比的是"同一份代码跑出来的两次分数"，
  它必须是真话。

写法约定：每个条目要么写清**用户能感知的变化**，要么写清**曾经会出错的判断被修在哪里**；
不写"重构某某模块"这种只有作者关心的话。

## [Unreleased]

M9（操作闭环）进行中。这一版的主线是"不发一句命令就能完成一次评测"。

### Added

- **Tool Bench 页**：工具注册表（内容 hash 版本、副作用、tokens、启用位）、契约审计（逐条规则 + 修法，
  文案与 CLI 同源）、上下文开销、执行器契约矩阵、运行历史。五个端点全是 GET，
  所以只读看板也能完整审计工具库。矩阵是**点一次才跑**——它真的会在离线样本上把每个执行器
  调一遍（http 用 MockTransport、MCP 用假连接，零真实网络），不该被轮询。
- **`onyx tools run` 现在会留痕**。`tool_run` 表、它的引用列与保留规则存在了很久，
  但**从来没有写入方**：空面板看起来像"没人试过"，而真相是"试过的地方不记"，
  这比没有面板更坏（它会让人以为这条证据不存在）。确定性/幂等两列保持「—」，
  跑一次测不出这两件事。
- 契约矩阵的构建处抽到了 `onyx/tools/matrix.py`：`onyx tools contract --json` 与
  `GET /api/tools/matrix` 现在返回同一个形状。抽取时修正了一处真差异：内置 echo 的专用样本参数
  原本只看 `name == "echo"`，现在必须连着**出处**一起判——注册表里覆盖的同名工具要用它自己的
  examples，否则矩阵测的是内置定义、报告说的是注册版本。
- 在界面导入数据集。`POST /api/datasets` 收 JSONL **文本**而不是路径（请求体里带路径
  就是"服务器任意读文件"，而这个看板是可以带 token 共享的），本地文件由浏览器读成文本再发。
  来历 `upstream / revision / license` 一起落库；没填 revision 时按**内容 hash** 补——
  "没填"绝不能变成空串，空 revision 等于放弃"这两次跑的是不是同一份数据"的判据。
- **补上一条断掉的闭环**：`onyx eval import --id x` 把样本写进了库，但解析器只认内置别名与
  `file:<路径>`，于是紧接着 `onyx eval run --dataset x` 报"未知数据集"。现在
  `load_dataset(..., db=)` 按"内置 → file: → 库里登记"解析，CLI 与界面同一个顺序。
- **覆盖同名 id 必须显式确认**（409 → 人自己勾选 `允许覆盖同名 id`）。历史 grade 指向这个 id，
  悄悄换内容会让"这两次分数能不能比"从"能"变成"不知道"；替换成功后服务端返回警告
  （revision 变了 / 条数变了），CLI 打印它，界面显示它。
- 解析与落库各只有一份（`parse_jsonl` / `register_dataset`）：两个入口对"第 37 行坏了"
  必须给同一个行号，对"覆盖了旧 revision"必须说同一句警告。
- 只有 dataset 行、没有样本的登记：列表里标"不可选"，发起评测时 422 拒绝。
  否则跑出来是 `status=done / n_total=0` 这种看起来完全正常的空评测。
- 评测四个子页（运行与发起 / 矩阵 / 回归对比 / 数据集）之间终于有导航了：
  之前矩阵与回归**只能手敲 hash** 才到得了。
- **在界面发起评测**。`POST /api/runs` 只入队并**立刻**返回 run_id：一次 236 条的评测要占住
  GPU 几十分钟，让 HTTP 请求等结果，超时会让调用方以为失败了而 GPU 还在跑。跑评测的是服务里
  一个 worker 线程，用的仍是 `onyx eval run` 那套 `load_dataset` / `build_task` / `EvalRunner`
  ——界面与 CLI 之间不许有第二套评测语义。
- **取消，而且是三条路都能取消**：排队中（worker 取出时发现标记）、跑一半（runner 在两条样本
  之间检查）、**等 GPU 锁时**（锁的等待回调里抛出，否则取消按钮在"排队中"那一屏是死的）。
  取消不打断进行中的那条请求：半截请求的 trace 比没有 trace 更难解释。
- **进度与排队位置**。`GET /api/runs/{id}/progress` 合并内存快照与库里的 run，并用 `source`
  说清出处：CLI 发起的运行没有逐条进度也没有取消开关，显示成"可以取消"就是撒谎。
  `GET /api/queue` 报本进程的队列，`GET /api/tasks` 从任务注册表现读（前端不再硬编码任务名，
  否则装了插件的任务在界面上是隐形的）。
- **一条样本都没跑的取消不留 run 记录**。runner 是拿到 GPU 锁之后才写 run 行的，所以排队中/等锁中
  被取消的任务在库里没有任何痕迹——留一条空 run 会在历史里多出一个从没产生过 grade 的条目。
- **服务重启不再留下 `running` 僵尸**。启动时把上次进程发起（`config.trigger == "api"`）**且**
  GPU 锁已空闲的行标成 `error` 并写明原因；两个判据缺一不可，否则会去改别的进程正在跑的运行。
  `finished_at` 留空：只知道它不在了，不知道它什么时候没的。
- 状态词表保持 running/done/cancelled/skipped/error 不新增：worker 崩了记 error，僵尸也记 error。
- `--read-only` 的共享看板上这些写操作返回 403，未带 token 返回 401（与 S21 的姿态一致）。

### Fixed

- **没传模板开销时工具库占比报 0%**。`cost_report` 算的是 `overhead / (json + overhead)`，
  没传 overhead 时得到 `0.0`，会被读成"模板不花钱"——而 P17 的实测结论恰恰相反
  （大头就是模板注入的说明文本）。现在报 `None`，界面显示「—」。
- **Windows 上一轮评测会被"有人在看进度"判死**。心跳用 `os.replace` 原子替换锁文件，而看板开始
  每秒读那个文件来显示"谁在占 GPU"之后，Windows 会因目标被别的句柄打开而返回 `ACCESS_DENIED`——
  真机现象是 944 条的评测在前一轮就死掉，`status=error` 且一条 grade 都没落。现在 replace 带重试，
  重试仍失败时**只计数不打断评测**（`cost.gpu_heartbeat_errors` 与最后一次原因进 run 记录）：
  心跳是给排队者算 ETA 的优化，不是正确性前提；但心跳长期写不出去意味着锁可能已被别人判过期接管，
  那段数字要能被质疑，所以计数必须可见。

### 已知边界（本节随 M9 更新）

- `onyx models pull/rm` 仍只有能力没有出口，界面上也还没有被中断 run 的续跑入口（S26）。
- 被中断的 run 现在状态正确，但**续跑入口还在命令行**（`onyx eval run --resume <id>`）。
- 数据集导入有 4MB 上限且不流式：更大批次的导入走 CLI。删除数据集没有做——grade 有外键指向样本，
  真删会撕断分数历史，而"归档"需要先定义历史 run 在界面上的显示语义。
- `tools fire` 的结果不写 `tool_run`：模型侧调用的证据已经是 trace + `tool_call`
  （带 parse_status / result_status / 参数与返回值），再抄一份就是两个事实源。

---

## [0.8.0] - 2026-10-04

M7（跑得住）与 M8（配置与鉴权姿态）。这一版的主线不是加功能，而是
"这套东西能在自己机器上跑很多年而不坏、坏了能恢复、共享时不会不小心敞开"。

### Added

- **`onyx rotate` 数据生命周期**。默认 **dry-run**：只报"会删多少、能释放多少字节"，
  一个字节都不碰；确认后再 `--apply`。策略是"分数永久、证据有限期"——摘掉原始
  request/response、渲染后的 prompt、工具返回值这五列重 payload 的引用，trace 行与
  messages/output 都留着，所以每个评测分数仍然能点进一条真实 trace。被 `grade` /
  `tool_run` / `eval_run` 引用的 trace 永不删行（删行会让评测历史断线）。单次回收超过
  现有 blob 体积 60% 直接拦住，要人明确加 `--force`。
- **每次运行都落 `retention_run`（含 dry-run）**。事后问"三周前那次原始 body 怎么没了"，
  答案必须是一条记录而不是"大概跑了 rotate 吧"。审计列写**事实**（回滚了就是 0），
  估算只活在 `per_rule` 与 `detail_json` 里。
- **`onyx db backup` / `db verify-backup`**。备份用 SQLite 在线备份 API 而不是 `cp`：
  WAL 模式下最近的事务还在 `-wal` 里，只拷主文件会得到一个"打开不报错、行数看着合理"
  却少了最后一段数据的库。备份只装**被引用到的** blob；验证时逐字节重算每个 blob 的
  sha256 与文件名比对，并检查"备份库里的每个引用都能在备份里解析"——只备库不备证据
  的那种"备份"就是这样暴露的。
- **`onyx doctor` 补两项**：磁盘余量（`.data` 写满的表现是崩溃 + 半截 blob，不是优雅报错）、
  token 计量档位（报"这台机器上每个模型落到哪一档"，并明说 `hf_tokenizer`/`gguf_vocab`
  本版本未实现、`tokens` extra 装了也不生效）。blob 完整性项改为与 `rotate` 共用同一份
  引用清单，并把缺失的 ref **指名道姓**列出来。
- **迁移前自动快照**：真要改 schema 之前先留 `backups/pre-migration-v{旧}.sqlite`。
  它是回滚点，不是完整备份（不含 blob），所以与 `db backup` 分工明确。
- **`onyx db sizes`**：`.data` 现状 + 增长趋势 + 预计余量天数。采样点只来自 `retention_run`；
  两个点相隔不到一天时报"问不出来"而不是算一个噪声斜率，日均很小时也不报
  "还能写约 1.4 亿天"这种假精确。
- **`onyx.toml` 部署配置**：provider、GPU 锁、保留窗口、sandbox 白名单、serve 绑定收在一处。
  优先级只有一条规则 **显式 flag > 环境变量 > 配置文件 > 内建默认**，为此每条命令的 flag
  内建默认都改成 `None`（带着具体默认值的 flag 会永远赢过配置文件，让它当场变成摆设且不报错）。
  模板见 `onyx.example.toml`；`onyx config show` 逐项标出生效值来自哪一层。`doctor` 会抓
  "写了不生效"：schema 之外的键、类型不对的键（含 `port = true` 这种被当成 1 号端口的手滑）
  都会指名并报红。
- **非回环绑定强制 token**：`--host 0.0.0.0` 没有 token 时**拒绝启动**。默认绑 127.0.0.1
  从来不是鉴权，只是碰巧没人连得上；而看板里有你全部的 prompt 与原始 body，还能往 GPU 上
  打请求、unload 正在用的模型。`--read-only` 把"共享看板"与"共享操作台"分开。
  `--allow-insecure-local` 可以裸跑，但**故意不可写进配置文件**。前端能自助：URL 加一次
  `?token=` 就存进 sessionStorage（不是 localStorage），之后 fetch 走 header、SSE 走 query
  （`EventSource` 设不了请求头，这条路的代价写进 README 与错误提示）。
- **覆盖率成为第五道质量门**：`make coverage` 跑离线套件、算分支覆盖、`fail_under=80`。
- **GitHub Actions**：ruff / lint-imports / 边界脚本 / 离线套件 + 覆盖率门禁 / 前端三段 /
  打包安装冒烟。`-m live` 与 `-m probe` 需要真实引擎，明确不在 CI 里跑（见 workflow 注释）。

### Changed

- 版本号改为单点定义（`onyx/__init__.py`）并升到 `0.8.0`；此前 `pyproject.toml` 里写死的
  `0.1.0` 从未随里程碑更新，而 `eval_run.app_version` 一直在落这个假值。
- GPU 心跳阈值不再在两处各写一个字面量 `600.0`，统一为 `DEFAULT_GPU_STALE_AFTER_S`，
  并可用配置文件 `[gpu].stale_after_s` 覆盖。
- `serve` 默认端口与前端代理对齐（`8787`）：以前 CLI 默认 8000 而 vite 指向 8787，
  每条命令都得手动带 `--port`。
- `onyx doctor` 检查项从 6 项增加到 9 项。

### Fixed

- **体检门槛分叉**：`fitted` 档的最低样本量原本散在四处字面量 `30`（计数、标定写入、
  CLI 档位标签）。门槛一分叉，最坏的组合就是体检绿着而计数其实早已退回 `heuristic/low`。
  现在收敛到 `FITTED_MIN_SAMPLES`，并有测试钉住"体检与计数用的是同一个数"。
- **坏配置曾被静默忽略**：`--config` 指向不存在的文件时，只要 `ONYX_DATA_DIR` 设过，
  `db info` 就照常跑、退出码 0——因为每个值都能从更高优先级拿到，没有任何一条路径会去读
  那份文件。现在由根回调先验证再动手（`doctor` / `config` 例外：它们就是用来诊断这份文件的）。
- **抽象泄漏两处**：`iter_refs()` 曾把 blob 目录里任何人放的其它文件当成"一个可回收的 blob"；
  `delete()` / `total_bytes()` 与 `stat().size` 的字节口径不一致，导致"释放了多少"与
  "盘上小了多少"两个数互不相等。现在三处同口径，报告能 tie 得出来。
- **一条真实的泄漏**：`TOOL_EXEC_START` 事件写了一个 `args_ref` blob，但 schema 里没有任何列
  存它（`tool_call` 存的是内联 `args_json`）⇒ 每次工具循环泄漏一个小 blob。真机第一次
  `rotate` dry-run 报出的孤儿就是它。已记入 STATUS 的"还欠什么"——它该修在源头，
  而不是靠"rotate 能删孤儿"长期兜底。
- 备份验证不再把备份本身改成 WAL 模式（以 `mode=ro` 打开）：验证动作不该改动被验证的对象。

### 已知边界（写在这里，免得被当成"支持"）

- 单实例观测：一个 `serve` 进程绑一个 provider；多引擎要起多份服务。
- `hf_tokenizer` / `gguf_vocab` 两档计量未实现（只有插拔位与采信优先级）。
- 评测与工具子系统仍靠 CLI 驱动，界面只能读 + Playground（S23 起"发起评测"、S24 起"导入数据集"
  已经进界面，Tool Bench 与 `models pull/rm` 仍在这条边界里）。
- `-m e2e` 用例数为 0：浏览器验证目前是手工 `take_snapshot`。
- 未发行：没有 LICENSE（默认保留所有权利）、没有 Docker、没有 pipx 发布流程。

---

## 更早的里程碑（M0–M6）

这一版之前不按发布号切分（那段时间没有下游消费者），按里程碑记在
[`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) 的分步交付档案里：M1 计量（真机数字全部
可核对）、M2 看板六页、M3 工具子系统与契约矩阵、M4 评测内核与两个任务、M5 配对回归与矩阵、
M6 扩展点固化 + 第二 provider + MCP + OTLP。推翻过设计假设的实测结论在
[`docs/PROBES.md`](docs/PROBES.md)。
