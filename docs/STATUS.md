# 项目现状（Onyx · v0.1.0）

核对时间 **2026-10-04**，HEAD = M6 收口之后。
本文所有数字都是当场跑出来的（命令附在每节末尾），不是从旧文档抄的；
`README.md` 的进度表、`docs/IMPLEMENTATION.md` 的分步档案是历史沿革，
**这里回答的是"现在有什么、能干什么、还欠什么"**。

一句话：**M0–M6 全部达成，计划里的四个里程碑出口判据都有真机证据；
欠的是 S7 的运维交付物（保留策略 / 备份校验 / usage 报表）、Tool Bench 网页，
以及一批"带原因推迟"的项。**

---

## 1. 有的功能（按层）

### L0 领域层 `onyx/core/`（零三方依赖，import-linter 强制）
11 个能力位、23 种异常码、6 种工具失败 kind、13 种评测判定、18 种事件类型、
sha256 内容寻址的 blob 存储、单调钟 + 墙钟双时间、事件契约带必填键校验（`strict=True` 会拒绝缺键事件）。

### L1 存储 `onyx/store/`
16 张表、schema v5（迁移可重复执行、`0005` 回灌了数据集来历）、
WAL + 批量 sink（队列满丢样本但 `dropped` 计数可见）、
三种事件 sink：`jsonl` / `null` / `otlp`（OTLP/HTTP JSON 编码）。

### L2 引擎接入 `onyx/llm/`
- **咽喉点只有一个**：所有模型调用必经 `gateway.py`（评测也不例外，所以每个分数能下钻到真实 trace）
- 3 个 provider：`ollama`（原生 `/api/chat`）、`openai-compat`（vLLM / LM Studio / Xinference / Ollama `/v1`）、`mock`（脚本化假引擎，离线跑通全链路）
- token 保真阶梯：`engine > hf_tokenizer > gguf_vocab > fitted > compat > heuristic`，
  每个数字带 `source` + `confidence`；**未知显示「—」，绝不显示 0**
- 分段归因：`Σ(各分段) + template_ctl == 引擎计数`，残差与标定截距互验（P20/P21）
- 冷/热 prefill 分列（P11：合并聚合是谎话）、keep-alive 剩余、显存 offload 判定
- 流式与非流式共用同一个 assembler；OpenAI 兼容分支同时吃 `delta` 与 `message`

### L3 观测 `onyx/obs/`
5 个内建 visitor（token / tool / gpu / cost / anomaly，**顺序即契约**）+ 外部插件 visitor；
23 种异常码统一码表（前端文案与 CLI 同源）；
有界状态 + TRACE_END 缺失时按容量淘汰（宁可丢一条观测也不 OOM）。

### L4 工具子系统 `onyx/tools/`
- 注册表：内容 hash 版本化、契约审计（每条规则对应一个可行动修法）、上下文开销核算（与 trace 的 `part=tool_defs` 同源）
- 4 种执行器：`python_fn`（AST 白名单，不用 eval）/ `http`（URL 只来自人工审核定义，不做通用 fetch）/ `mcp`（stdio + JSON-RPC，纯 stdlib）/ `fixture`（评测零副作用通道）
- **8 条契约断言 × 4 列执行器矩阵**（`onyx tools contract`，离线零网络）：
  失败强制分成 `arg_error / rejected / timeout / unknown_tool / skipped / error` 六种，
  每个 n/a 必须写明原因（静默跳过等于让最关键的保证消失）
- 客户端工具循环：预算 / 熔断 / 孤儿补齐 / fire-and-verify 六种判定
- 沙箱：副作用白名单、dry-run、审批回调、`impl_ref` 白名单、deadline 传到 socket（P22）

### L5 评测 `onyx/eval/`
- 2 个内建任务：`intent_classification`（236 条中文意图集）、`tool_selection`（97 条工具调用集）
- 7 个评分器模块 + 类型感知参数比对（数值容差 / 日期归一 / 集合等价 / 严格档与宽松档分开报）
- 指标层：macro_f1 / accuracy / balanced_accuracy / P-R-F1 / 混淆对 /
  pass^k 与 pass@k / stability_gap / bootstrap CI（**按 case 重采样**，95% 区间带出处）
- 配对比较：净改善 / 净劣化 / 不变 + McNemar 翻转表 + 配对 bootstrap + 劣化清单（每条带两个 trace_id）
- 模型 × 任务矩阵（每格取**最新一次 done**，薄覆盖率强制显示分母）
- 报告导出：md / csv / 自包含 html（含手写 SVG 雷达图，任务数 ≥3 才画）
- 调度：GPU **机器级**独占锁 + 心跳 + ETA + `--unload-others`、断点续跑（成本与 n_done 不被后续片段清零）
- 能力不满足 ⇒ 整任务 skip 且**留下带原因的记录**（禁止隐式降级）
- 数据集导入：`--source bfcl`、JSONL、`file:<路径>`；来历（upstream/revision/license）参与可比性判定

### L6 接口
- **CLI 34 条命令**：顶层 6（`chat` / `serve` / `doctor` / `plugins` / `version` / `calibrate`）
  + 分组 28（`db 2` / `probe 4` / `models 2` / `traces 3` / `tools 9` / `eval 8`）
- **API 19 个端点**（18 REST + `GET /api/stream` SSE）
- **Web 9 页**：Fleet / Models / Traces / TraceDetail / Token Ledger / Playground / 评测 / 矩阵 / 回归
  ——手写 CSS token 与手写 SVG，无组件库无图表库；每页都实测过（零 console 错误）

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
uv run pytest            # 990 passed, 1 skipped（tests/unit 901 + tests/contract 90）
uv run pytest -m live     # 20 passed（真打 qwen3.5:9b，与评测共用机器级 GPU 锁）
uv run pytest -m probe     # 4 passed（P 系列实验的可重跑版本）
uv run ruff check .         # All checks passed（`ruff format` 不是门禁）
uv run lint-imports          # 3 contracts kept（两条网络例外显式登记）
uv run python scripts/check_extension_boundary.py   # 接入实现未触碰受保护内核文件
前端：tsc --noEmit / vitest 44 / vite build（203KB js）+ 浏览器 take_snapshot
```

真机跑过的证据（可复查，都在 git 里）：
意图集 `macro_f1 0.991 [0.978–1.000]`；工具集 `must_call_acc 0.639` 而
`hallucinated_tool`、误调率均为 0（**掉分全在参数上**）；
配对回归 `改善 0 / 劣化 226 / 不变 10`，同时暴露 `可判定 8/236` 的分母陷阱；
`--provider openai-compat` 打本机 `/v1` 得 `usage=compat/low in=16 out=400`（与 `finish=length` 自洽）；
`--provider echo`（外部插件）落库 `heuristic/low` 且无引擎计数的项显示「—」；
MCP 工具经 `tools fire --mock live` 判定 **PASS**（模型 → 循环 → stdio server → 回填 → final）。

---

## 3. 还欠什么

### A. 计划里承诺、但确实没做的（S7 的运维交付物 + 一个页面）
| 缺什么 | 计划出处 | 为什么算事 |
|---|---|---|
| `scripts/rotate.py`（trace 90d / 原始 body 30d / eval 记录永久） | IMPLEMENTATION S7 产出文件 | 现在**没有任何保留策略**：`raw_response_ref` 指向的 blob 只增不减，`.data` 会一直涨。这是本机跑评测最先撞到的现实问题 |
| `onyx db backup` / `db verify-backup` | S7 自测命令 | 观测数据没有备份与"备份可恢复"的验证手段，`doctor` 只能查引用完整、不能证明可恢复 |
| `onyx report usage --since 7d --format md` + `report/exporters/{csv,md,jsonl}` | S7 产出文件 | 只有 `eval report`（评测报告）；**用量周报**没有。今天要看一周吞吐只能自己写 SQL |
| `onyx token explain <trace>`（多源对比表） | S7 / 附录 A 的自测命令 | 功能其实存在但埋在 `traces show` 的对账表里，没有独立入口；对照"某个数字为什么被采信"这个高频问题，命令行入口是必要的 |
| `doctor` 的两项检查：磁盘余量、tokenizer 档位可用性 | S7 检查项清单 | 磁盘写满是本地部署最常见的故障；tokenizer 档位决定 `hf_tokenizer/gguf_vocab` 两档可不可用，缺了就只能出事后才发现 |
| Tool Bench 网页 | S10–S12 计划（后端已交付） | 工具审计、开销、契约矩阵、fire 结果目前只有 CLI；`GET /api/tools/demo` 也只服务 Playground 的下拉。看板上看"工具库上下文开销随版本变化"这件事没有界面 |
| `-m e2e` 用例（附录 A 的 S9 自测行） | 附录 A | 现在 **0 个 e2e 用例**：浏览器验证是手工做的（`take_snapshot` + 读 console）。手工是唯一一次 SSE 静默失效被发现的途径，但也意味着**没人记得住的那些路径**没有回归保护 |
| `onyx serve --reload` | 附录 A 的自测命令 | 小：开发体验，非功能缺口（vite 的 HMR 已覆盖前端） |

### B. 明确推迟、带原因的（不是遗漏）
- `structured_extraction`、`instruction_following` 两个任务：M4 出口只要求"两个 task 真机跑通"，
  先做深不如先做对（`docs/IMPLEMENTATION.md` S14 一节记了理由）。
- `ollama_builtin` 执行器：必须先有 P21 的实测结论（模板里内建工具到底怎么渲染），
  否则做了也是猜的。`onyx tools contract` 那一列会显示"未实现，计划在 S16+"。
- `onyx.graders` 扩展点：见上表（没有消费点就先不建）。
- `onyx providers add/list`：多 provider 并存时的登记入口。现在 `--provider` +
  `plugins_example` + `.data` 里一张 `provider` 表就够用；等真出现"同一台机器常驻 3 个引擎"再做配置层。

### C. 已知边界（会被误读成 bug 的那些）
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


1. **保留策略 + 备份校验**（A 组第 1、2 项）：本机长期运行的第一道墙，且没有它 `.data` 无法自我约束。
2. **`-m e2e` 用例**：把已经手工验证过的六页路径固化成回归（SSE 联通、矩阵分母显示、劣化清单下钻）。
3. **Tool Bench 页**：后端齐了（`tools ls/audit/cost/contract/fire` 的数据都在库里），缺的是把它摆上看板。
4. **`token explain` + `doctor` 两项检查 + `report usage`**：都属于"每天都在用但入口缺失"。
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
