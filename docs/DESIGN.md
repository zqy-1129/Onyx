# Onyx — 本地大模型管理看板 设计文档

> 状态：待评审 · 目标机器：Windows 11 / RTX 4060 Ti / 48GB RAM · 首个 Provider：Ollama
> 配套：`IMPLEMENTATION.md`（分步构建 + 每步自测）

---

## 1. 目标与非目标

### 1.1 目标
| # | 能力 | 完成判据 |
|---|---|---|
| G1 | 模型资产与运行时管理 | 列出已安装/已加载模型、量化、参数规模、能力位、显存占用、keep-alive 剩余时间；可加载/卸载/删除/拉取 |
| G2 | 输入输出 token 与延迟度量 | 每请求记录 input/output/reasoning token、TTFT、prefill/decode 吞吐；多来源对账并标注来源与置信度 |
| G3 | 工具调用观测 | 记录每次 tool_call 的名称/参数原文/解析结果/执行结果/耗时；能区分"模型不会调"与"工具坏了" |
| G4 | 模型评测 | 数据集驱动的 task：意图识别、工具选择、工具参数、结构化抽取、指令遵循、拒答、回归；输出 per-case 得分与聚合指标 |
| G5 | 工具测试 | 不依赖模型可做契约测试；依赖模型做 fire-and-verify（指令→是否调对工具、参数是否正确） |
| G6 | 可扩展/模块化 | 新增 provider / 评测任务 / grader / 观测器 = 实现一个接口 + 注册，不改内核；模块边界由工具强制校验 |

### 1.2 非目标
- 不做训练/微调/量化（但记录量化字段）
- 不做面向终端用户的 Agent 产品（工具循环只为观测与评测服务）
- 不做多租户/权限（单机自托管，仅 `--api-auth <token>` 占位）
- 不追求与公开 leaderboard 数值可比（API-only 无法做 loglikelihood 打分，见 §9.4）

---

## 2. 贯穿全局的设计原则

1. **单一咽喉点**：所有模型调用必经 `llm/gateway.py`。观测、计量、评测、Playground 都只是它的消费者。任何模块不得自己发 HTTP。*理由：只要存在第二条调用路径，统计口径就会分裂，而这个产品的全部价值在口径上。*
2. **观测与被观测解耦**：gateway 只产出标准化事件流（`core/event.py`，带 `CONTRACT_VERSION`）；token 归因、工具归因、异常检测各是一个独立 visitor。加一个指标 = 加一个 visitor。
3. **原始证据不可变**：原始 HTTP body、渲染后的 prompt、模型原文（含畸形 JSON）一律落内容寻址存储；派生记录必须写明由哪份原始证据推导。
4. **测量必须带出处**：所有数字带 `source ∈ {engine, hf_tokenizer, gguf_vocab, fitted, compat, heuristic}` + `confidence ∈ {high, medium, low}`。不同来源禁止静默混用（§6）。
5. **能力位显式声明**：Provider 报告支持什么（tools / tool_choice / structured_output / stream_usage / vision / embed / thinking / n_sampling / admin）。评测在 task 与能力不匹配时 **skip 并记录原因**，绝不隐式降级。
6. **持久化前向兼容**：未知字段进 `*_json` 影子列而非抛异常；列只增不减；迁移可在既有数据上重放。
7. **本地 GPU 是独占资源**：eval / benchmark / playground 互斥（GPU 锁 + 队列），否则所有延迟数字失去意义（§8.5）。

---

## 3. 分层架构

```
┌─ L6 呈现  api/(FastAPI+SSE)  web/(Vite+React+TS)  report/(md|csv|html 导出)
├─ L5 评测  eval/{task,datasets,tasks,graders,metrics,runner,compare}
├─ L4 工具  tools/{spec,registry,executors,sandbox,loop,contract}
├─ L3 观测  obs/visitors/{token,tool,anomaly,gpu,cost}   probe/(能力与行为实测)
├─ L2 适配  llm/providers/{ollama,openai_compat,mock}     ← 唯一允许发 HTTP 的层
├─ L1 内核  llm/{gateway,request,response,params,streaming,measurement}  ← 稳定契约
└─ L0 领域  core/{types,ids,clock,event,errors,content}   store/{db,migrations,repos,sinks}
```

**允许的依赖方向**（由 `import-linter` 在 `make lint` 强制）：

| 层 | 允许 import |
|---|---|
| `core` | 仅 stdlib（连 pydantic 都不引入，用 dataclass + Protocol） |
| `store` | `core` |
| `llm` | `core` `store` + `httpx/minja/tokenizers/gguf` |
| `obs` | `core` `llm` `store` |
| `tools` | `core` `llm`(仅接口) `store`；**只有 `tools/executors/http.py` 可以 import httpx** |
| `eval` | `core` `llm` `tools` `store` |
| `api/web/report` | 除 `core` 外全部（`core` 亦可） |

> `core` 无三方依赖是可移植性的锚点：换掉 FastAPI 不需要动领域模型。

---

## 4. 目录结构

```
Onyx/
  pyproject.toml  uv.lock  README.md  Makefile  .gitignore  onyx.example.yaml
  docs/  DESIGN.md  IMPLEMENTATION.md  PROBES.md(实测结论)  eval-recipes.md
  onyx/
    __init__.py  settings.py  cli.py            # Typer 入口
    core/  types.py  ids.py  clock.py  event.py  errors.py  content.py
    store/ db.py  migrations/0001_init.sql ...
           repos/{trace,usage,tool,eval,model}_repo.py   sinks/{sqlite,jsonl,null}.py
    llm/  gateway.py  request.py  response.py  params.py  streaming.py  registry.py
          providers/base.py  providers/ollama/{client,native,openai_compat,lifecycle,tokenizer,template}.py
          providers/openai_compat.py  providers/mock.py
          measurement/{reconciler,fidelity,parts,heuristic,calibrate}.py
    obs/  engine.py  visitors/{token,tool,anomaly,gpu,cost}.py  anomalies.py
    probe/ usage_fields.py  cache.py  stream_usage.py  think.py  tool_format.py  structured.py
    tools/ spec.py  registry.py  sandbox.py  loop.py  contract.py
          executors/{python_fn,http,mcp,ollama_builtin,mock_replay}.py
    eval/ task.py  datasets/{loader,sources,cache}.py  datasets/builtin/*.jsonl
          tasks/{intent_classification,tool_selection,tool_args,structured_extraction,
                 instruction_following,refusal,regression_replay}.py
          graders/{exact,set_match,args_match,regex,json_schema,fuzz,llm_judge,embedding}.py
          metrics.py  runner.py  compare.py  registry.py
    api/  app.py  deps.py  sse.py  routes/{fleet,models,traces,usage,tools,evals,playground,admin}.py
    web/  src/pages/{Fleet,Models,Playground,Traces,TraceDetail,Tools,Evals,Usage}.tsx
    report/exporters/{md,csv,html,jsonl}.py
    plugins/ example_task/ example_provider/    # 扩展点样板
  tests/ unit/ contract/ integration/ e2e/  fixtures/
  scripts/ dev.sh  datasets_import.py  bench/
  .data/ onyx.sqlite  blobs/  cache/datasets/  probe-results/
```

---

## 5. 数据模型（L0 契约，SQLite/WAL）

> 约定：**稳定且需聚合/索引 → 独立列；易变且仅展示 → `*_json`**。列只增不减。
> 大 payload 不入库，入库内容寻址指针 `sha256:...`（`.data/blobs/`）。DB 保持几十 MB 级、天然去重（同一长 system prompt 只存一份）、可按原始证据重放请求。

```sql
CREATE TABLE provider(
  id TEXT PRIMARY KEY,                 -- 'ollama-local'
  kind TEXT NOT NULL,                  -- ollama|openai_compat|vllm|lmstudio|mock
  base_url TEXT NOT NULL,
  api_style TEXT NOT NULL,             -- native|openai（决定能力位，§7）
  enabled INTEGER NOT NULL DEFAULT 1,
  config_json TEXT, created_at TEXT);

CREATE TABLE model(
  id TEXT PRIMARY KEY, provider_id TEXT NOT NULL REFERENCES provider(id),
  name TEXT NOT NULL,                  -- 'qwen3:8b'
  remote_model TEXT, remote_host TEXT, -- 用于取 HF tokenizer（T1 档）
  digest TEXT, bytes INTEGER, modified_at TEXT,
  family TEXT, parameter_size TEXT, quantization TEXT, parent_model TEXT, ctx_train INTEGER,
  capabilities_json TEXT,              -- /api/show.capabilities
  template TEXT,                       -- /api/show.template
  model_info_json TEXT,                -- /api/show.model_info = raw GGUF 元数据(tokenizer.ggml.*, arch.*)
  tool_format TEXT,                    -- chat_template|xml|json|native_head|unknown（probe 结论）
  tokenizer_source TEXT,               -- gguf|hf|tiktoken|none
  tokenizer_ref TEXT,
  usage_ratio REAL, usage_ratio_n INTEGER,  -- fitted 档：tokens/char 标定值与样本数
  first_seen_at TEXT, last_seen_at TEXT, extra_json TEXT,
  UNIQUE(provider_id, name));

CREATE TABLE trace(
  id TEXT PRIMARY KEY,                 -- 可排序 id（时间前缀，core/ids.py）
  parent_id TEXT REFERENCES trace(id), root_id TEXT,
  kind TEXT NOT NULL,                  -- generation|embed|rerank|health|admin|judge
  purpose TEXT NOT NULL,               -- chat|playground|eval:{task}|tool_test|probe|bench
  eval_run_id TEXT, case_id TEXT, sample_seq INTEGER,
  provider_id TEXT, model_id TEXT,
  started_at TEXT NOT NULL, first_token_at TEXT, finished_at TEXT,
  status TEXT NOT NULL,                -- ok|error|timeout|cancelled
  error TEXT,
  params_json TEXT, messages_json TEXT, tools_json TEXT,
  rendered_prompt_ref TEXT, output_ref TEXT, raw_request_ref TEXT, raw_response_ref TEXT,
  finish_reason TEXT,
  engine_latency_json TEXT,            -- {total,load,prompt_eval,eval}_duration（纳秒）
  gpu_json TEXT,                       -- {size_vram,size,context_length} 采样时刻
  keep_alive TEXT, extra_json TEXT);
CREATE INDEX idx_trace_started ON trace(started_at DESC);
CREATE INDEX idx_trace_purpose ON trace(purpose, eval_run_id, case_id);
CREATE INDEX idx_trace_model   ON trace(model_id, started_at DESC);

CREATE TABLE usage(                      -- 采信结果，看板只读这张
  trace_id TEXT PRIMARY KEY REFERENCES trace(id),
  in_tokens INTEGER, out_tokens INTEGER, thinking_tokens INTEGER, cached_tokens INTEGER,
  source TEXT NOT NULL, confidence TEXT NOT NULL,
  ttft_ms REAL, prefill_tps REAL, decode_tps REAL, wall_ms REAL,
  bytes_out INTEGER, drift_pct REAL,     -- |采信值-归因来源|/采信值
  extra_json TEXT);

CREATE TABLE usage_alt(                  -- 多来源明细：对账与异常检测的事实表
  trace_id TEXT NOT NULL REFERENCES trace(id),
  source TEXT NOT NULL,                  -- engine|compat|hf_tokenizer|gguf_vocab|fitted|heuristic
  in_tokens INTEGER, out_tokens INTEGER, thinking_tokens INTEGER, cached_tokens INTEGER,
  ok INTEGER NOT NULL DEFAULT 1, note TEXT,
  PRIMARY KEY(trace_id, source));

CREATE TABLE token_part(                 -- 归因（有渲染能力时才有值）
  trace_id TEXT NOT NULL REFERENCES trace(id),
  part TEXT NOT NULL,                    -- bos|system|tool_defs|msg:<idx>|gen_prompt|image|template_ctl
  ord INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL, bytes INTEGER,
  PRIMARY KEY(trace_id, part, ord));

CREATE TABLE tool_call(
  id TEXT PRIMARY KEY, trace_id TEXT NOT NULL REFERENCES trace(id), step INTEGER NOT NULL,
  call_id TEXT, name TEXT, args_json TEXT, args_raw TEXT,
  parse_status TEXT NOT NULL,            -- ok|json_error|unknown_tool|missing_required|type_mismatch|truncated|not_parsable
  parse_source TEXT,                     -- native_head|text_template|heuristic
  result_status TEXT,                    -- ok|error|timeout|rejected|mocked|skipped
  result_ref TEXT, result_bytes INTEGER,
  started_at TEXT, latency_ms REAL, tool_id TEXT, tool_def_hash TEXT,
  executed_by TEXT);                     -- client|server(ollama builtin)|mock|fixture
CREATE INDEX idx_toolcall_name ON tool_call(name, started_at);

CREATE TABLE tool_def(
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, version TEXT NOT NULL,
  kind TEXT NOT NULL,                    -- python_fn|http|mcp|ollama_builtin|fixture
  schema_json TEXT NOT NULL, impl_ref TEXT, hash TEXT NOT NULL,
  tokens INTEGER, bytes INTEGER,         -- 该工具注入上下文的成本（§6.2）
  tags TEXT, owner TEXT, enabled INTEGER DEFAULT 1,
  side_effect TEXT NOT NULL DEFAULT 'read',  -- read|write|network|exec → sandbox 决策
  timeout_ms INTEGER, doc TEXT, examples_json TEXT, extra_json TEXT,
  created_at TEXT, updated_at TEXT);
CREATE TABLE tool_test(
  id TEXT PRIMARY KEY, tool_id TEXT NOT NULL REFERENCES tool_def(id),
  name TEXT, args_json TEXT, expect_json TEXT, checks_json TEXT, live INTEGER DEFAULT 0,
  created_at TEXT);
CREATE TABLE tool_run(
  id TEXT PRIMARY KEY, tool_id TEXT, tool_def_hash TEXT, test_id TEXT, trace_id TEXT,
  started_at TEXT, latency_ms REAL, status TEXT, output_ref TEXT, error TEXT,
  deterministic INTEGER, idempotent INTEGER);

CREATE TABLE dataset(
  id TEXT PRIMARY KEY, upstream TEXT, revision TEXT, license TEXT,
  split_json TEXT, n_cases INTEGER, loader TEXT, notes TEXT, imported_at TEXT);
CREATE TABLE case(
  id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES dataset(id), ord INTEGER,
  kind TEXT,                             -- single|multi_turn|multi_step|parallel|no_call_needed
  input_json TEXT NOT NULL, tools_json TEXT, expect_json TEXT NOT NULL,
  fixture_json TEXT,                     -- 工具返回值桩：保证评测可复现（§8.4）
  meta_json TEXT, tags TEXT);
CREATE TABLE task(
  id TEXT PRIMARY KEY, name TEXT NOT NULL, dataset_id TEXT REFERENCES dataset(id),
  metrics_json TEXT NOT NULL, grader_json TEXT NOT NULL,
  sample_params_json TEXT, k INTEGER DEFAULT 1,   -- pass^k 次数（Ollama 不支持 n → 循环采样）
  budget_json TEXT, sandbox INTEGER DEFAULT 0, extra_json TEXT);
CREATE TABLE eval_run(
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES task(id),
  model_id TEXT NOT NULL, provider_id TEXT,
  started_at TEXT, finished_at TEXT, status TEXT,
  seed INTEGER, app_version TEXT, git_rev TEXT,
  params_snapshot_json TEXT, config_json TEXT,
  n_cases INTEGER, n_done INTEGER, n_error INTEGER, aggregate_json TEXT, notes TEXT);
CREATE TABLE grade(
  id TEXT PRIMARY KEY, eval_run_id TEXT NOT NULL REFERENCES eval_run(id),
  case_id TEXT NOT NULL, seq INTEGER NOT NULL DEFAULT 0, trace_id TEXT,
  score REAL NOT NULL, passed INTEGER,
  verdict TEXT,                          -- correct|partial|wrong|invalid_format|hallucinated_tool|timeout|refused
  metrics_json TEXT, error TEXT, judge_model_id TEXT, graded_at TEXT);
CREATE INDEX idx_grade_run ON grade(eval_run_id, case_id);

CREATE TABLE anomaly(
  id TEXT PRIMARY KEY, trace_id TEXT, code TEXT NOT NULL, severity TEXT NOT NULL,
  detail_json TEXT, created_at TEXT);   -- code 见 obs/anomalies.py 常量表
```

**分析层**：默认全部走 SQLite。行数上来（>1M trace）后用 DuckDB `ATTACH 'onyx.sqlite' (TYPE SQLITE)` 原地做列式聚合，不搬数据、不加 ETL，且只开只读连接。

---

## 6. Token 度量：多来源保真阶梯

本地 token 数的难点：引擎报的数未必准、未必全，缓存命中时甚至可能只统计未缓存的后缀。单来源方案必然自欺。

### 6.1 阶梯
| 档 | `source` | 来源 | 能得到 | 拿不到 | 采信 |
|---|---|---|---|---|---|
| T0 | `engine` | 原生 `/api/chat` 的 `prompt_eval_count`/`eval_count` | 引擎真实 tokenize 数 + 纳秒时序 | 归因分解；缓存语义需实测 | **首选** |
| T1 | `hf_tokenizer` | 由 `model.remote_model` 取 HF `tokenizer.json`，配合 `template` 用 minja 渲染 | **完全归因**：system / 每个历史 turn / **工具定义** / gen prompt 各占多少 | 与引擎模板实现存在 ±；需网络 | 归因首选 |
| T2 | `gguf_vocab` | 从 `~/.ollama/models/blobs/sha256-<digest>` 用 `gguf` 读 `tokenizer.ggml.*` 自建 BPE | 离线下较准复算 | 实现成本最高 | M2 后选做 |
| T3 | `fitted` | `onyx calibrate` 对该模型采 N 条样本拟合 tokens/char 比 | 无 tokenizer 时 ±2~5% | 归因；分布漂移 | 兜底 |
| T4 | `compat` | `/v1` 的 `usage` | 交叉验证 | 与 T0 可能不一致（**不一致本身即信号**） | 只记录不采信 |
| T5 | `heuristic` | `chars/4` + CJK 修正 | 永远可用 | 精度低 | `confidence=low` |

- **采信顺序** `engine > hf_tokenizer > gguf_vocab > fitted > heuristic`；同时记 `drift_pct = |engine − 归因来源| / engine`，超阈值写 `anomaly(TOKEN_DRIFT)` 并在 UI 高亮。
- **thinking**：reasoning token 是否计入 `eval_count` 必须由 `probe/think.py` 实测；未确认前 UI 显示"含推理，口径未定"，不要猜。

### 6.2 只有复算才能回答的问题
`prompt_eval_count` 是一个总数，无法告诉你"**24 个工具定义正在每次请求偷走 1380 个 token**"。而这恰是本地部署最贵且最易被忽视的一笔——工具 schema 被 chat template 注入**每一次**请求。所以 `token_part(part='tool_defs')` 是 Tools 页的一等公民指标，按 模型 × 模板 分别计算。

### 6.3 必须先实测、不许假设（`probe/` 的职责）
| 探针 | 待答问题 | 方法 |
|---|---|---|
| `usage_fields` | 原生与 `/v1` 的计数是否一致 | 同一请求走两条通道对比 |
| `cache` | **warm 缓存下 `prompt_eval_count` 是否只算未缓存后缀** | 同长 prompt 连发 3 次，观察 count/duration 变化 |
| `stream_usage` | 原生流是否只在最后一个 ndjson 事件给计数；`/v1` 是否必须 `stream_options.include_usage` | 流式并检查末事件 |
| `think` | thinking 是否计入 `eval_count`；`think:false` 是否真生效 | 同一推理模型三档对比 |
| `tool_format` | 工具走哪种序列化（ChatML 工具头/XML/JSON/文本模板） | 逐模型最小工具，存原文 |
| `structured` | `format:json_schema` 是否生效、失败是否静默降级 | 强制 schema 抽取 |

> 结论写入 `docs/PROBES.md` 并回灌 `model.extra_json`。**没有实测结论的字段一律 unknown，UI 显示"—"而非 0。**

### 6.4 派生指标
```
wall_ms     = finished - started
ttft_ms     = first_token - started          （cold 含 load_duration，必须分列）
prefill_tps = in_tokens / prompt_eval_duration
decode_tps  = out_tokens / eval_duration
ctx_util    = in_tokens / context_length     （来自 /api/ps.context_length）
```
冷/热永远分列：把冷启动 TTFT 混进 P50 是这类看板最常见的自欺。

---

## 7. Provider 抽象与 Ollama 映射

```python
class LlmProvider(Protocol):
    id: str
    kind: ProviderKind                        # ollama|openai_compat|vllm|lmstudio|mock
    def info(self) -> ProviderInfo: ...
    def capabilities(self) -> frozenset[Cap]: ...
    def list_models(self) -> list[ModelCard]: ...        # /api/tags
    def show_model(self, name: str) -> ModelDetail: ...  # /api/show
    def running(self) -> list[LoadedModel]: ...          # /api/ps
    def generate(self, req: GenerationRequest, *, on_event: EventCB | None = None) -> Generation: ...
    def embed(self, req: EmbedRequest) -> EmbedResponse: ...
    def pull(self, name: str, *, on_event: EventCB) -> AdminResult: ...
    def unload(self, name: str) -> AdminResult: ...       # keep_alive="0"
    def probe(self) -> ProbeReport: ...
```
`Cap = {chat, tools, tool_choice, structured_output, thinking, vision, embed, stream_usage, n_sampling, logprobs, admin}`

### 7.1 字段映射
| 内部 | Ollama 原生 `/api/chat` | Ollama `/v1/chat/completions` |
|---|---|---|
| temperature/top_p/top_k/max_tokens/seed/stop | `options.*` | 顶层同名 |
| thinking | `think: bool\|str` | `thinking`/`reasoning_effort`（版本相关，probe 确认） |
| keep_alive | `keep_alive: "5m"\|"0"` | ❌ |
| tools | `tools` | `tools` |
| tool_choice | 由客户端 loop 实现 | ❌ 官方文档列为不支持 |
| structured | `format: "json"` / `{"type":"object",...}` | `response_format` |
| usage | `prompt_eval_count`/`eval_count` + 纳秒时序 | `usage`（需 `stream_options.include_usage`） |
| n>1 | ❌ | ❌ 不支持 |
| 停止原因 | `done_reason` | `choices[].finish_reason` |

**结论：Ollama 适配器以原生 API 为主通道**（计数/时序/admin 更全），`/v1` 只作 T4 交叉验证与"仅暴露 OpenAI 兼容接口"的外部服务兜底。`api_style` 因此是一等配置项。

`tool_choice` 与 `n` 缺失的影响是具体的：强制调用类用例必须走客户端 loop；`pass^k` 只能循环发 k 次请求。这类事实落成 `Cap` 检查 + skip 原因，而不是运行时踩坑。

### 7.2 控制面映射
| 面板 | Ollama 字段 |
|---|---|
| 已安装 `/api/tags` | `models[].{name,model,remote_model,remote_host,modified_at,size,digest,details.{format,family,families,parameter_size,quantization_level}}` |
| 已加载 `/api/ps` | `models[].{name,model,size,digest,details.parent_model,expires_at,size_vram,context_length}` |
| 详情 `/api/show` | `{thinking,parameters,license,modified_at,details,template,capabilities,model_info}` ← `model_info` 即 raw GGUF 元数据，是 T1/T2 计数的原料 |
| 删除/拉取/复制 | `/api/delete` `/api/pull`(流式进度) `/api/copy`；错误体统一 `{error}` |

### 7.3 归一化与多模态
图片只存 `media_ref`（blob），provider 层按各自 API 序列化（原生 `images:[base64]` vs OpenAI `image_url`）。`request.normalize()` 是唯一入口，也是纯函数单测重点。

---

## 8. 工具子系统（L4）

### 8.1 三层，互不混淆
1. **契约层（无模型）**：`ToolDef` 是否合法可诊断 —— JSON Schema 合法性、命名规则、参数描述完备度、`side_effect` 标注、**schema 的 token 开销**。
2. **执行层（无模型）**：真实调用 + 边界 —— 超时、非法参数、幂等、确定性、错误传播。
3. **模型层（有模型）**：指令 → 是否选对工具、参数是否正确（与 §9 共用同一 harness）。

分层价值：工具调用得分低时先跑 1/2 层，**立刻判定是模型的问题还是工具的问题**。混在一起就是"工具坏了"和"模型不会用"互相甩锅。

### 8.2 接口
```python
class ToolExecutor(Protocol):
    kind: ToolKind
    def spec(self) -> ToolDef: ...
    def call(self, name: str, args: dict, ctx: ToolCtx) -> ToolResult: ...
# ToolCtx = trace_id, deadline, dry_run, mock_policy ∈ {live, fixture, replay, deny}
# ToolError 族: ToolUnknown / ToolArgError / ToolTimeout / ToolSandboxDenied / ToolSkipped / ToolRuntime
```

**执行器只有 `guarded_call` 一个入口**（`tools/executor.py`），顺序不可调换：
参数校验 → 沙箱策略 → mock 短路 → deadline 内执行 → 错误归一。
校验排在 mock 之前，否则给了桩就等于放过畸形参数；策略排在 mock 之前，
否则 mock 模式能绕过沙箱。每个执行器自己再实现一遍这两步，必然漏。

**失败必须分成 6 种互不相同的 kind**，这是整个执行层存在的理由：

| kind | 含义 | 归因 |
|---|---|---|
| `arg_error` | 参数缺字段/类型错/枚举越界/多余字段 | **模型**的输出问题 |
| `rejected` | 沙箱策略拒绝（副作用未放开、未审批、dry-run、impl_ref 不在白名单、路径越界） | **配置**问题 |
| `timeout` | 超过 deadline | 可重试；对 http 还必须证明 deadline 传到了 socket |
| `unknown_tool` | 请求的工具名与执行器绑定的定义不符 | **模型**选错工具（路由） |
| `skipped` | mock 策略主动不执行（deny / 缺桩） | 评测配置，不是失败 |
| `error` | 实现崩了、签名与 schema 脱节、上游 5xx | **工具**坏了 |

`ToolUnknown` 从 `resolve_impl`/签名不匹配抛出时归到 `error` 而**不是** `unknown_tool`：
那是定义与实现脱节，记成路由错会让评测把修法的方向指错（该改定义，不是改提示词）。

`extra.constants` 是定义级、**模型不可覆盖**的可信参数（如 `fs_read` 的允许根目录）。
它不出现在 `parameters.properties` 里；一旦同名出现，执行器直接报 `constant_exposed`，
而不是静默让常量覆盖参数——静默覆盖会把配置错误藏到运行时。

### 8.3 客户端工具循环
`tools/loop.py` 由**本地实现**（Ollama 只返回 `tool_calls`，不会替你执行工具；若某内置工具由服务端执行，需标 `executed_by='server'` 并单列延迟，待 probe 确认）。职责：
- 步数 / 时间 / token 预算上限，超限即 `anomaly(BUDGET_EXCEEDED)`
- **循环检测**：`(name, hash(args))` 重复 → 熔断 `anomaly(TOOL_LOOP)`
- **畸形参数保留原文**：`json.loads` 失败绝不丢弃，落 `args_raw` + `parse_status`
- **孤儿调用补齐**：`finish_reason=tool_calls` 但结果缺失/超时时补占位 tool 消息，否则后续上下文永久错位
- 停止词配置化：文本模板类模型常需注入 `<tool_response>` / `<|im_end|>` 这类标记，由 `model.tool_format` 驱动而非硬编码——原生 tool_calls 头的模型加了反而会截断正文

**上下文不变式（实现里由专门的断言函数守着）**：
> 一条带 N 个 `tool_calls` 的 assistant 消息，后面必须紧跟**恰好 N 条** `role=tool` 消息。

少一条，之后每次请求的上下文都永久错位，而引擎通常**不报错**——只是开始答非所问。
所以即使中途熔断，同一轮里剩下的调用也必须补占位结果。

**两个实现决定，都是踩出来的**：
1. 工具执行事件必须落在**发起该调用的那一步**的 trace 里，而 TRACE_END 由 gateway 发出，
   所以 `Gateway.generate` 提供 `before_trace_end` 钩子 + `TraceEmitter`（含 blob 写入口）。
   钩子抛异常时先关 trace 再抛，否则库里会留下永不结束的 trace，污染所有按 trace 聚合的统计。
2. 多步循环用 `TraceContext.root_trace_id` 分组，**不用** `parent_trace_id`：
   root 是循环自己造的分组键，没有对应的 trace 行，而 `trace.parent_id` 是外键。
   塞进 parent 会撞 `FOREIGN KEY constraint failed`，整条记录写不进去，
   而表面上只是日志里一行警告。

`tool_call_fingerprint` 放在 `core/types.py`：循环靠它熔断，tool visitor 靠它聚合重复调用，
两处必须算出同一个值。各写一份必然漂移，漂移之后一边报循环、一边报正常，
这种自相矛盾比没有检测更难查。

### 8.3.1 fire-and-verify 的六种判定
`tools/verify.py` 把"工具调用得分低"拆开。**判定顺序即归因优先级**：先看这一轮有没有收敛，
再看调没调、调得对不对——顺序反了会把"循环熔断"记成 PASS，因为第一次调用的参数往往是对的。

| verdict | 该改什么 |
|---|---|
| `NO_CALL` | 系统提示词 / 工具的 description（"什么时候该用"） |
| `WRONG_TOOL` | 工具之间的描述区分度；名字太像就改名 |
| `BAD_ARGS` | 参数 description、required、max_tokens（截断与填错字段修法不同，detail 里分开说） |
| `TOOL_FAILED` | **工具**，不是模型——先跑 `onyx tools contract` |
| `LOOP_BROKEN` | 工具返回值没给模型新信息，或提示词没让它换策略 |
| `ERROR` | 先跑 `onyx doctor` |

汇总的 `pass_rate` 分母**排除** `TOOL_FAILED` 与 `ERROR`：把工具缺陷和环境故障算进模型得分，
等于让模型替它们背锅，然后你会去调一个本来没问题的提示词。

### 8.4 可复现性：fixture 优先
评测**默认不执行真实工具**：`case.fixture_json` 提供工具返回值桩，产出 `tool_run(status='mocked')`。只有 `--live` 且工具 `side_effect='read'` 才真跑。理由：真实搜索结果/天气会让评测不可复现，把模型误差和数据漂移混为一谈。

### 8.5 资源调度（本地特有，不可省）
单 GPU 上，eval、benchmark、playground 会互相踩。规则：
- `runner` 拿 **GPU 独占锁**，一次只跑一个 model×task；playground 排队并显示 ETA。
- 运行前按策略 unload 上一个模型（`keep_alive="0"`），避免 `size_vram` 叠加导致加载失败或 CPU offload —— 后者会让吞吐数字差一个数量级而看起来"正常"。
- 每个 eval 样本记录采样时刻 `/api/ps` 的 `size_vram/context_length`，用于事后判定"这次慢是不是因为挤显存"。

---

## 9. 评测子系统（L5）

### 9.1 统一抽象
```python
class EvalTask(Protocol):
    id: str
    requires: frozenset[Cap]                       # 能力不满足 → skip 并记录原因
    def load(self, split: str) -> Iterator[Case]: ...
    def build(self, case: Case) -> GenerationRequest: ...   # 只产请求，不发请求
    def grade(self, case: Case, sample: Generation) -> Grade: ...
    def aggregate(self, grades: Sequence[Grade]) -> dict: ...
```
`runner` 只做四件事：`build` → **gateway 发（自动被记录）** → `grade` → `aggregate`，外加调度/断点续跑/取消。

> **评测绝不建立第二条调用路径**（原则 1）。于是每个分数都能点进一条真实 trace，带 token/延迟/工具记录；换 gateway 的观测实现不需要改任何评测代码。这是把"评测"和"可观测性"共享同一套数据的关键决定。

### 9.2 内置任务与指标
| task | 数据集来源 | 指标 |
|---|---|---|
| `intent_classification` | **自建中文集**（`datasets/builtin/intent_zh.jsonl`，模板生成 + 人工补充）；导入你已有语料 | 逐类 P/R/F1、macro-F1、混淆矩阵、**格式合法率**、**越界标签率**（输出不在标签集=幻觉）、拒答率 |
| `tool_selection` | BFCL（含 `no_call_needed` / `parallel` / `multi_step` 划分）+ 自建 | must-call 命中率、hit@1/hit@k、调用集合 P/R/F1、**误调率**、幻觉工具名率 |
| `tool_args` | BFCL + 自建 | 逐参数类型感知比对：数值容差、日期归一、枚举、集合语义、字符串模糊；schema 合法率 |
| `structured_extraction` | 自建（含 json_schema 变体） | JSON 合法率、schema 合规率、字段级 EM；若 `Cap.structured_output` → 额外跑"强制 vs 自由"对照 |
| `instruction_following` | IFEval 风格可验证规则（内置 checker 注册表） | 逐约束通过率（长度/语言/关键词/格式/禁用词） |
| `math_reasoning` | GSM8K 子集 | 答案提取+归一后 EM；无需 LLM judge |
| `refusal_safety` | 自建 | 过度拒答率、该拒不拒不率 |
| `regression_replay` | 你自己的黄金集（trace 一键转 case） | 配对对比 + bootstrap 置信区间、劣化用例清单 |
| `latency_bench` | 合成 | TTFT/TPS/吞吐/显存，cold/warm 分列 |

### 9.3 统计口径
- **程序化优先**；LLM judge 是可选 grader，须记录 `judge_model_id` 及其自身 usage（judge 也是本地模型时同样吃 GPU 与时间，必须计入成本）。
- 多次采样：`pass^k`（k 次全对，衡量稳定可用）与 `pass@k`（至少一次对）。Ollama 不支持 `n` → 循环 k 次，每次独立 trace。
- 聚合带 **bootstrap 95% CI**；样本数 <100 时在 UI 明确标注低置信。
- 参数快照 `params_snapshot_json` 落库，否则换 temperature 后的分数差异无法解释。

### 9.4 API-only 的硬约束（必须写进 UI 文案）
无法拿到受约束的 choice logprob ⇒ **loglikelihood 类基准（MMLU 式 ABCD 打分）做不了**，只能生成式打分。后果：分数与公开 leaderboard 不可直接比较，且会因"输出格式不听话"额外掉分。UI 上标注 `gen-based`，并把"格式合法率"与"内容正确率"分开展示，避免把格式问题误读成能力问题。

---

## 10. 事件契约（L3，跨版本兼容点）

```python
CONTRACT_VERSION = 1
# 每个事件: {"v":1, "type":..., "trace_id":..., "ts":..., "payload":{...}}
```
| type | payload 关键字段 | 产出者 |
|---|---|---|
| `trace_start` | kind, purpose, provider, model, params, messages_ref, tools_ref | gateway |
| `model_load` | load_duration_ns, cold(bool) | adapter |
| `first_token` | ttft_ms | streaming |
| `text_delta` / `thinking_delta` | text, seq | streaming |
| `tool_call_delta` | idx, name_fragment, args_fragment | streaming |
| `generation_end` | finish_reason, output_ref, raw_ref | adapter |
| `usage_engine` | in, out, thinking, cached, latency_ns | measurement(T0) |
| `usage_compat` | in, out | adapter(可选 T4) |
| `usage_local` | in, out, tool_defs, parts[] | measurement(T1/T2/T3) |
| `reconciled` | chosen_source, confidence, drift_pct | reconciler |
| `tool_exec_start/end` | name, args_ref, status, latency_ms, executed_by | loop |
| `gpu_sample` | size_vram, size, context_length | lifecycle |
| `anomaly` | code, severity, detail | visitors/anomaly |
| `trace_end` | status, wall_ms | gateway |

规则：新增字段=兼容变更；改名/删字段/改语义 = 递增 `CONTRACT_VERSION`，visitor 必须按 `v` 分派。UI 侧对未知事件类型必须**忽略而非崩溃**。

---

## 11. 看板信息架构

| 页面 | 内容 | 关键判据 |
|---|---|---|
| **Fleet 总览** | 服务存活/版本、已加载模型（`expires_at` 倒计时、`size_vram`、`context_length`）、近 1h 调用量/decode TPS/错误率、异常流 | 一眼看出"显存被谁占着" |
| **Models** | 能力位、量化/参数、chat template 原文、tokenizer 状态、`/api/show.model_info`、benchmark 卡 | 能力位与 probe 结论必须有出处链接 |
| **Playground** | 多模型并排、thinking 分栏、工具面板、每轮 usage 条、**渲染后 prompt 查看**、一键转 case | 并排必须串行过 GPU 锁 |
| **Traces** | 列表 + 详情：时间轴、消息、tool 循环树、per-part token 归因、缓存推断、原始 body | 任意派生值可回溯原始证据 |
| **Tool Bench** | 注册表健康、**工具库 token 开销排行**、契约测试结果、fire-and-verify 结果、MCP 工具发现 | 模型侧与工具侧结果分区展示 |
| **Eval** | task/benchmark 配置、model×task 矩阵、雷达图、per-case 钻取、回归 diff（两 run 并排 + 劣化清单） | 每个分数能跳到 trace |
| **Token Ledger** | input/output/thinking 时序、采信来源占比、drift 异常、prefill vs decode、缓存命中推断 | 明确区分 cold/warm |
| **Ops** | pull/delete/unload、keep_alive 策略、并发队列、DB 备份导出、数据集导入 | 危险操作二次确认 |

---

## 12. 框架选型分析：要不要引入 LangChain 一系

**结论：不引入 LangChain / LangGraph。核心（L0–L3）自己写，评测层留一个可选适配器。**

### 12.1 判断标准
本项目最难的三件事是：① **计量的真实性**（token/延迟/缓存语义与引擎严格对齐）② **评测 harness 设计**（case→metric→CI→回归）③ **本地运行时治理**（显存、keep-alive、GPU 独占）。三者都要求**贴近引擎的裸数据**。而"贴近裸数据"正是框架抽象层抹掉的东西。

### 12.2 LangChain 具体会吞掉什么
| 本项目需要 | LangChain 现状 | 后果 |
|---|---|---|
| Ollama 的 `prompt_eval_duration` / `eval_duration` / `load_duration` / `done_reason`（纳秒级） | `ChatOllama` 归一为 `usage_metadata`，其余散在 `response_metadata`，字段随版本漂移 | 时序与冷/热判定得靠扒私有字典，等于把稳定性押在别人版本上 |
| 引擎返回的**原始文本**（畸形工具 JSON 的原文） | tool calling 归一成 `tool_calls`，解析前的文本不可见 | 无法诊断"模型输出格式不对"这一类本地模型最高频故障 |
| 逐来源 token 对账（engine vs 本地复算 vs 兼容层） | 只有一个归一 usage | 必须**在框架之外**再写一套计量，框架变成纯开销 |
| `keep_alive`、`/api/ps`、`/api/show.model_info`、pull/delete 流式进度 | 无稳定映射 | 治理功能全部要绕过框架直接打 HTTP，形成"两套调用路径"——违反原则 1 |
| TTFT / 流式增量 / 延迟归因 | 多一层 Runnable/callback 间接 | 产品目标与框架抽象方向相反 |
| 工具 schema 的 token 开销（按模板渲染） | 抽象掉模板 | 拿不到 |

一句话：**用了 LangChain 之后，你仍然要自己写计量与治理，只是每件事都多绕一层。**

### 12.3 LangGraph
本项目工具循环是 ≤150 行的 `while` 循环（§8.3），且**必须自己掌握原始文本与每步预算**才能做观测，图抽象反而是负担。
唯一真正需要它的场景：弱工具调用模型走 **prompted 模式**（ReAct 文本循环，自己解析）。但那需要的是精细解析控制，手写循环比拆状态图更容易插桩。

### 12.4 该"买"的（不要写）
| 买什么 | 从哪买 | 理由 |
|---|---|---|
| **工具调用评测集** | BFCL（函数调用 leaderboard 数据 + v3 多步） | 自己造 case 成本高且容易被质疑不公 |
| **标准数据集加载** | `datasets` / `huggingface_hub` | 授权、版本、分片别人已解决；本地缓存 + 记录 `revision/license` |
| **chat template 渲染** | `minja`（llama.cpp 同源实现） | 自己写 Jinja 子集解释器是自找麻烦 |
| **GGUF 元数据读取** | `gguf` | 官方格式解析 |
| **schema 校验 / 模糊匹配** | `jsonschema`、`rapidfuzz` | 琐碎且已解决 |
| **整包评测 harness**（可选，M5 决策） | **Inspect AI**（UK AISI，HF 维护）：约 50+ 现成 benchmark、 scorer/扫描器/并发齐全 | 若你只缺"benchmark 覆盖度"而不缺"计量"，它比 LangChain 合适得多 —— 它的抽象边界正好停在"模型调用"，而**我们只需实现一个 `ModelAPI` 适配器转发到 gateway**（≈1 天），trace 仍在自家 DB 里 |

**Inspect AI 与 LangChain 的关键区别**：前者只站在"评测编排"这一层，不碰你的模型 IO；后者想站在所有层。这也是为什么"如果你要一个框架，应该是 Inspect 而不是 LangChain"。

### 12.5 其他同类工具定位
- **Langfuse / Phoenix（Arize）/ OpenLLMetry** —— 别人做好的观测半段。Langfuse 自托管很好，但它不懂**本地计量**（GGUF tokenizer 复算、工具 schema token 预算、显存/keep-alive、cold/warm 判定）。建议：把它们当**导出目标**（`sinks/` 加一个 OTLP/Langfire sink，几小时工作量），而不是底座。
- **DeepEval / promptfoo / OpenAI evals** —— grader 与配置格式可参考；`ToolCorrectness`、`RegexMatch` 等可当纯库摘出来复用。
- **LiteLLM** —— 路由/兜底/配额。当且仅当你想加云端模型做本地↔云对照基线时才需要。
- **vLLM / llama.cpp / TGI** —— 它们是**推理服务**，不是应用框架；本项目是它们的观测者，不是替代者。

### 12.6 什么时候反过来应该引入框架
触发条件（出现两条以上再评估 Inspect，出现三条以上引入 Pydantic AI/LangGraph 于应用侧）：
1. 要跑的公开 benchmark > 10 个；
2. 需要多子判官合议 / 复杂 rubric 评分；
3. 要做真正的多步 agent 产品（不只是评测）；
4. 需要分布式执行多 GPU/多机；
5. 需要给非开发者提供可视化流程编辑。

### 12.7 必须自写的部分（约 4k 行 Python 的量级）
| 组件 | 为何无法外包 |
|---|---|
| Provider 适配器（原生 API + 控制面） | 需要 `prompt_eval_*`、`/api/ps`、`/api/show.model_info`、pull 流式进度 |
| Token 对账与保真阶梯 | §6.3 每一项都是引擎特有语义 |
| 工具循环 + 畸形解析观测 | 需要原始文本与逐步预算 |
| 工具注册表与 token 开销核算 | 需绑定具体 chat template |
| 评测 runner（GPU 独占锁、断点续跑、回归 diff） | 本地单 GPU 资源治理是产品特性 |
| 看板 | 领域信息架构（显存、能力位、来源置信度）与通用 trace UI 形状不同 |

---

## 13. 扩展点（六个注册表，全部走 entry points）

| group | 契约 | 加一个实现即可支持 |
|---|---|---|
| `onyx.providers` | `LlmProvider` | vLLM / llama.cpp server / LM Studio / 云端基线 |
| `onyx.tasks` | `EvalTask` | 新评测（RAG 忠实度、代码生成、多语言） |
| `onyx.graders` | `Grader` | 新判据（自定义 rubric、执行式判分） |
| `onyx.sinks` | `Sink` | Langfuse / OTLP / Kafka / 只读导出 |
| `onyx.tool_executors` | `ToolExecutor` | MCP、HTTP、DB、内置工具 |
| `onyx.observers` | `EventVisitor` | 新异常规则、成本模型、告警 |

- 注册方式：`[project.entry-points."onyx.tasks"] intent_zh = "plugins.intent:Task"`；内核用 `importlib.metadata` 发现，**不扫描目录、不写死 if/else**。
- `plugins/` 提供两个样板实现，兼作扩展点的活体测试：内核若不能容纳一个外部实现，说明抽象错了。
- 所有契约带版本与能力协商；配置外置 `onyx.yaml`（providers、keep_alive 策略、GPU 锁、数据集缓存、sandbox 白名单）。

---

## 14. 风险与已知坑

| # | 风险 | 处置 |
|---|---|---|
| R1 | warm 缓存下 `prompt_eval_count` 可能只算未缓存后缀 ⇒ 输入 token 莫名偏低 | `probe/cache.py` 实测后决定语义；永远与本地复算并列显示 |
| R2 | 流式忘传 `stream_options.include_usage` ⇒ usage 恒为 0 | 兼容层强制注入；`probe/stream_usage.py` 断言 |
| R3 | `total_duration` 含 `load_duration` ⇒ 冷启动污染所有吞吐 | cold/warm 分列，`load_ms>50ms` 判 cold |
| R4 | 采样非确定性 ⇒ 评测不可复现 | 记录 `seed`（Ollama 支持 `options.seed`）+ 参数快照；不支持 seed 的模型标"低可复现" |
| R5 | keep_alive 常驻显存 ⇒ 下一个模型加载失败/静默 offload | GPU 锁 + 运行前 unload + 采样 `size_vram/size` 差值告警 |
| R6 | 图片 token 由引擎（mmproj）决定，文本 tokenizer 估不准 | 标 `vision_estimate_low_conf`，不参与成本汇总 |
| R7 | thinking 混进 content ⇒ 工具 JSON 解析失败 | 增量流分离 `thinking`；`probe/think.py` 确认计数口径 |
| R8 | 中文 `chars/4` 严重低估（CJK 每字 1.5–3 token） | heuristic 必须带 CJK 修正系数，并标 low confidence |
| R9 | 换 chat template ⇒ 同一工具库开销大变 | `tool_defs` token 按 模型×模板 版本化记录 |
| R10 | Ollama 不支持 `tool_choice`/`n` ⇒ 部分评测写法直接失效 | `Cap` 检查 + skip 原因，不静默降级 |
| R11 | SQLite 多写者锁 | WAL + 单写队列（gateway 内串行 flush）；分析用只读 DuckDB 连接 |
| R12 | 4060 Ti 显存小（8GB 档更甚），长上下文 KV 可能超过权重本身 | Ops 页显示 `ctx_util` 与 offload 判定；默认限制评测上下文长度 |
| R13 | judge 模型也是本地模型 ⇒ 评测被 judge 拖慢甚至互踩 | judge 走同一 GPU 锁，其 usage 单独计入 `eval_run` 成本 |
| R14 | 外部数据集授权/离线不可用 | 记录 `license/revision`；内置小型自建 JSONL 保证无网也能跑通 M4 |

---

## 15. 为什么是这个形状（一段话理由）

token 计量、工具观测、模型评测、工具测试**不是四个功能，而是同一个事实的四个投影**：一次请求 + 一次响应 + 上下文 + 时间。因此系统只需要一个"事实生产者"（gateway，产出 §10 的事件流）和四个"事件消费者"。这样做的三个直接收益：
1. 指标口径不可能漂移（只有一个生产者）；
2. 新增能力是加法不是改动（注册 visitor/task/executor）；
3. **每个评测分数都能下钻到一条可回放的真实 trace**——因为评测就是 gateway 的另一个调用方，没有第二套代码路径。

而 §12 的"不用 LangChain"结论其实是同一件事的推论：**产品的价值在数据边界上，任何把这层边界抽象掉的框架都在给你增加成本，而不是减少。**



