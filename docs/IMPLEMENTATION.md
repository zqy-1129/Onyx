# Onyx — 分步实现方案（每步含自测与验收）

> 用法：从 S0 顺序执行。每一步都是一个可独立提交、可独立验证的最小闭环。
> **纪律**：每步先写自测（`tests/`），跑到红，再写实现跑到绿，再执行该步"验收命令"，通过后 `git commit`。禁止跨步合并提交——出错时无法定位是哪一层契约破了。

---

## 0. 全局约定

### 0.1 里程碑与出口判据
| 里程碑 | 步骤 | 出口判据（人话验收） |
|---|---|---|
| M0 环境 | S0 | `uv run onyx doctor` 全绿，能列出本机 Ollama 模型 |
| M1 计量 | S1–S7 | **任意一次本地对话，DB 里能查到一条 trace，含 input/output token、TTFT、decode TPS、来源与置信度、工具调用若干，且原始 body 可回放** |
| M2 看板 | S8–S9 | 浏览器打开 Fleet 与 Traces 页，Playground 流式对话实时出现在 Traces |
| M3 工具 | S10–S12 | `onyx tool contract` 与 `onyx tool fire` 都能跑出通过/失败明细；工具库 token 开销有数字 |
| M4 评测 | S13–S14 | 意图识别与工具调用两个 task 各跑完一次，输出 macro-F1 / must-call 命中率 + 95% CI，每个分数可下钻 trace |
| M5 对比 | S15 | 两个模型同一 task 的矩阵与回归 diff 可导出 md/csv |
| M6 扩展 | S16 | 不改动 `core/`，仅加插件 entry point 就能新增一个 provider 与一个 task 并跑通 |

### 0.2 测试分层与命令
```
tests/unit/         纯函数，无 IO、无网络、无模型          → make test
tests/contract/     工具契约/接口实现检查，无模型           → make test
tests/integration/  需要 Ollama 在跑（@pytest.mark.live）  → make test-live
tests/probe/        语义实测实验，产出 docs/PROBES.md      → make probe
tests/e2e/          浏览器过看板（@pytest.mark.e2e）       → make test-e2e
```
`pyproject.toml`：
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["live: 需要运行中的 Ollama", "probe: 语义实测实验", "e2e: 浏览器"]
addopts = "-m 'not live and not probe and not e2e' --strict-markers -q"
filterwarnings = ["error::DeprecationWarning:onyx.*"]
```

### 0.3 依赖
```toml
[project]
requires-python = ">=3.12"
dependencies = [
  "fastapi", "uvicorn[standard]", "httpx", "pydantic>=2.7", "typer", "rich",
  "jsonschema", "minja",          # chat template 渲染（llama.cpp 同源）
  "rapidfuzz", "PyYAML",
]
[project.optional-dependencies]
tokens  = ["tokenizers", "gguf"]           # T1/T2 复算档位
bench   = ["pandas", "matplotlib"]         # 校准拟合与图
analysis= ["duckdb"]                        # 只读列式聚合
dev     = ["pytest", "pytest-asyncio", "respx", "import-linter", "ruff"]
```
> **`core/` 与 `store/` 禁止 import 上面任何三方库**（只有 stdlib）；这条由 §附B 的 import-linter 强制，不是口头约定。

---

## S0 — 环境引导

**目的**：这台机器目前没有 Python、没有 Node、没有 Ollama。先把地基立起来，否则后面每一步都无法自测。

**操作（PowerShell）**
```powershell
winget install astral-sh.uv
winget install Ollama.Ollama
winget install OpenJS.NodeJS.LTS
```
```bash
uv python install 3.12
ollama pull qwen3:8b        # 工具调用 + thinking，覆盖 M1/M4 需要
ollama pull nomic-embed-text
```

**产出文件**：`scripts/bootstrap.ps1`（幂等：已安装则跳过）

**自测**
```bash
uv --version && node --version && ollama --version
curl -s http://127.0.0.1:11434/api/version
ollama list | head -5
```
预期：三条版本行；`{"version":"0.x.y"}`；至少 1 个模型。

**验收 DoD**：`ollama list` 非空；`/api/ps` 能访问（空 models 也算通过）。

**常见阻塞**：服务未启动 → `Start-Service OllamaService` 或手动跑 `ollama serve`；8GB 显存则只保留 `qwen3:8b`（4B 量化档），并把评测默认上下文压到 4096。

**提交**：`chore: bootstrap toolchain (uv/python3.12/ollama/node)`

---

## S1 — 仓库骨架 + L0 契约（types / ids / clock / event / errors / content）

**目的**：把"数据形状"先钉死。后面所有层都依赖它，所以它必须最稳、且零依赖。

**产出文件**
```
pyproject.toml  Makefile  .gitignore  onyx.example.yaml  README.md
onyx/__init__.py  onyx/settings.py  onyx/cli.py
onyx/core/types.py   # Role, Message, ToolSpec, ToolCall, GenParams, GenerationRequest,
                     # EngineLatency, TokenSample, FinishReason, Generation（全 dataclass）
onyx/core/ids.py     # new_trace_id(): 时间前缀可排序 id（无三方依赖）
onyx/core/clock.py   # monotonic/wall 时钟注入点（测试用假时钟）
onyx/core/event.py   # CONTRACT_VERSION=1 + EventType 枚举 + 每类型 payload 字段表(§10 DESIGN)
onyx/core/errors.py  # OnyxError 族: ProviderUnreachable/SchemaInvalid/Tool*/Eval*/CapabilityMissing
onyx/core/content.py # BlobStore: put(bytes)->'sha256:..' / get(ref) / .data/blobs 分片目录
tests/unit/test_ids.py  test_event.py  test_content.py  test_types.py
```

**接口要点**
```python
@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model: str; messages: tuple[Message, ...]; params: GenParams
    tools: tuple[ToolSpec, ...] = (); tool_choice: str | None = None
    stream: bool = False; thinking: bool | None = None
    keep_alive: str | None = None
    kind: str = "generation"; purpose: str = "chat"
    context: dict[str, Any] = field(default_factory=dict)  # eval_run_id/case_id/seq 透传
```
`context` 是评测与观测的连接点：**trace 必须能反查它是哪个 case 的第几个样本**。

**自测**
```bash
uv run pytest tests/unit -q
uv run python -c "from onyx.core.ids import new_trace_id as n; a,b=[n() for _ in range(2)]; print(a<b)"
```
预期：`N passed`；打印 `True`（id 单调可排序）。必须覆盖的断言：
1. `new_trace_id()` 连续 1000 个严格递增且唯一；
2. `GenerationRequest` 未知字段不炸（extra 进 `context`）；
3. `BlobStore` 同内容两次 put 返回同一 ref（去重）；
4. `event.EventType` 每个成员都有 payload 声明（表驱动测试，防漏）。

**验收 DoD**：`uv run onyx --help` 出帮助；`grep -rE "^(import|from) (httpx|fastapi|pydantic)" onyx/core/` **无输出**（零依赖是真的）。

**提交**：`feat(core): domain types, sortable ids, event contract, blob store`

---

## S2 — 存储层（SQLite/WAL + 迁移 + repo + sink 抽象）

**产出文件**
```
onyx/store/db.py  onyx/store/migrations/0001_init.sql        # DESIGN §5 全部 DDL
onyx/store/repos/{model,trace,usage,tool,eval}_repo.py
onyx/store/sinks/{base,sqlite,jsonl,null}.py
tests/unit/test_migrations.py  test_repos.py  test_sink_jsonl.py
```

**接口要点**
```python
class Sink(Protocol):
    def emit(self, ev: TraceEvent) -> None: ...      # 非阻塞，内部队列
    def flush(self, timeout: float = 1.0) -> None: ...
    def close(self) -> None: ...
```
- 单写队列：gateway 线程安全投递，后台线程批量事务落库（避免 SQLite 写锁竞争，R11）。
- 迁移 = 编号 SQL 文件 + `schema_version` 表；启动时按序重放，**幂等**。

**自测**
```bash
uv run pytest tests/unit/test_migrations.py tests/unit/test_repos.py -q
uv run onyx db init --db .tmp/t.sqlite && uv run onyx db info
```
预期：`db info` 打印 `schema_version=1 tables=18`。必须覆盖：
1. 空库迁移 → 全部表存在；
2. 重复 init 不报错且版本不变；
3. `trace insert + usage_alt 多来源 upsert + 采信读回` 一致；
4. 未知 JSON 字段进 `extra_json` 不丢（前向兼容，原则 6）；
5. 500 并发 emit → flush 后行数恰为 500（不丢不重）。

**验收 DoD**：`onyx db info` 正确；`sqlite3` 手工 `.tables` 与 §5 一致；DB 文件在无数据时 < 200KB。

**提交**：`feat(store): sqlite wal schema, idempotent migrations, single-writer sinks`

---

## S3 — Ollama 适配器：控制面 + 原生数据面 + 流式增量

**目的**：拿到 T0（引擎真数）。**只用真实服务验证，不做 mock-only 开发**——mock 会把你没能力做的假设固化进代码。

**产出文件**
```
onyx/llm/providers/base.py            # LlmProvider Protocol + Cap + 重试/超时
onyx/llm/providers/ollama/client.py   # httpx client、错误体 {error} 解析、超时与取消
onyx/llm/providers/ollama/native.py   # /api/chat(/api/generate) 请求构造与响应解析
onyx/llm/providers/ollama/lifecycle.py# /api/tags /api/show /api/ps /api/version /api/delete /api/pull(流)
onyx/llm/providers/ollama/openai_compat.py  # /v1/*（T4 交叉验证用）
onyx/llm/streaming.py                 # 增量缝合：content/thinking/tool_calls 分片累积
onyx/llm/params.py                    # GenParams ↔ options 映射（含未支持项显式记录）
tests/contract/test_provider_contract.py   # 契约检查（任何 provider 都要过）
tests/integration/test_ollama_live.py      # @live
```

**接口要点**
- `generate(req, on_event=...)`：流式时只在**最后一个 ndjson 事件**读 `prompt_eval_count/eval_count/*_duration/done_reason`（DESIGN §6.3 的 `stream_usage` 探针会证实这条）。
- `capabilities()` 由 `/api/version` + 探测结果驱动，**硬编码要留 TODO 指向具体探针**。
- `unload(name)` = `keep_alive:"0"` 的空请求；不依赖任何未文档化端点。
- 所有 HTTP 失败必须转 `ProviderUnreachable`，带上 base_url 与耗时（看板要显示"服务没起"而不是 500 栈）。

**自测**
```bash
uv run pytest tests/contract -q
uv run onyx models sync --provider ollama-local     # 期望: 同步 N 个模型, M 个已加载
uv run onyx chat "用一句话介绍你自己" --model qwen3:8b --stream
```
预期最后一条打印：流式文本、`in/out tokens`、`TTFT`、`decode TPS`、`load_duration`、`done_reason`，且末尾给出 `trace_id`。
必须覆盖：
1. 非流式与流式对同一请求给出**相同** token 数（不一致即 `probe/stream_usage` 失败）；
2. `tool_calls` 分片累积能拼回完整 JSON（多分片断言）；
3. `thinking` 与 `content` 不串流（推理文本不混进正文）；
4. 服务停掉时 `onyx models sync` 报错信息含 base_url 而非堆栈；
5. `/api/ps` 的 `expires_at` 能算出剩余秒数且随时间递减。

**验收 DoD**：`onyx chat --stream` 输出的 4 个数字与 `curl -s localhost:11434/api/chat -d '{...}' | jq` 原始返回逐项一致。

**提交**：`feat(ollama): native provider, streaming reassembly, control-plane lifecycle`

---

## S4 — Token 计量：保真阶梯 + 归因 + 对账

**目的**：把 §6 的阶梯变成代码，并**在写第一行业务前完成语义实测**。

**产出文件**
```
onyx/llm/measurement/fidelity.py   # 各档 Counter: engine/compat/hf_tokenizer/gguf_vocab/fitted/heuristic
onyx/llm/measurement/parts.py      # 渲染后 prompt 的分段归因（system/tool_defs/msg_i/gen_prompt）
onyx/llm/measurement/reconciler.py # 采信顺序 + drift 计算 + confidence 标注
onyx/llm/measurement/heuristic.py  # chars/4 + CJK 修正系数（R8）
onyx/llm/providers/ollama/template.py  # minja 渲染；不可用时退化为内置 ChatML/简单模板并降 confidence
onyx/llm/providers/ollama/tokenizer.py # model_info → vocab；(可选) remote_model → HF tokenizer.json
onyx/probe/{usage_fields,cache,stream_usage,think,tool_format,structured}.py
docs/PROBES.md                     # 实测结论（含日期/Ollama 版本/模型）
tests/unit/{test_reconciler,test_parts,test_heuristic_cjk}.py
tests/probe/test_probe_cache.py    # @probe
```

**接口要点**
```python
@dataclass
class TokenSample: source: str; in_tokens: int|None; out_tokens: int|None
                   thinking: int|None; cached: int|None; ok: bool; note: str
class Counter(Protocol):
    name: ClassVar[str]; fidelity: ClassVar[int]
    def count(self, req: GenerationRequest, gen: Generation) -> TokenSample: ...
```
- 归因只有在能渲染模板时才产 `token_part`；不能渲染 → 该 trace 只有总量，UI 显示"归因不可用"。
- `fitted` 档由 `onyx calibrate` 生成，结果写回 `model.usage_ratio/usage_ratio_n`；样本 <50 时 `usage_ratio` 视为不可信。

**自测**
```bash
uv run pytest tests/unit -q
uv run onyx probe run --model qwen3:8b --suite usage,cache,think   # 写 docs/PROBES.md
uv run onyx calibrate --model qwen3:8b --n 200                     # 打印拟合值与残差
uv run onyx token explain <trace_id>                                # 逐来源对比表 + drift
```
`token explain` 预期输出（关键验收形态）：
```
source        in    out   thinking  cached  ok  note
engine        1842   213     —        0     ✓  /api/chat prompt_eval_count
compat        1842   213     —        —     ✓  /v1 usage
hf_tokenizer  1855   209     —        —     ✓  minja+qwen2, drift 0.7%
fitted        1796   205     —        —     ✓  ratio 0.62 tok/char (n=210)
chosen: engine (high)   drift(engine vs hf_tokenizer)=0.7%  ✓ 阈值内
```
必须覆盖：
1. 采信顺序单测（engine 缺失→降级 hf_tokenizer→fitted→heuristic，每级 confidence 正确）；
2. CJK 修正：纯中文样本的 heuristic 误差 < 纯英文样本误差的 2 倍（防 R8）；
3. `tool_defs` 归因 ≥ 0 且随工具数量单调增（拿 1/5/20 个工具各测一次）；
4. `drift_pct` 超阈值时 reconciler 产 `anomaly(TOKEN_DRIFT)` 事件；
5. 探针脚本在无模型时 **skip 并提示**，不产生假结论。

**验收 DoD（M1 前半）**：`docs/PROBES.md` 里 cache / think / stream_usage 三个语义有明确结论或标注"未定"，且代码里的假设与之逐条对应（能 grep 到引用）。

**提交**：`feat(measure): token fidelity ladder, template attribution, drift reconciliation + probes`

---

## S5 — Gateway 装配 + 观测引擎 + visitors

**目的**：把 S3/S4 接到 S2，形成"唯一调用路径"。这一步结束，任何一次对话都会留下可回溯的完整证据链。

**产出文件**
```
onyx/llm/gateway.py        # normalize → provider.generate → 事件产出 → fanout(sinks+visitors)
onyx/llm/registry.py       # entry points 'onyx.providers' 发现
onyx/obs/engine.py         # EventVisitor 链：异常隔离（一个 visitor 崩不影响主链路）
onyx/obs/visitors/{token,tool,anomaly,gpu,cost}.py
onyx/obs/anomalies.py      # code 常量表: TOKEN_DRIFT/TRUNCATED_JSON/ORPHAN_TOOL_CALL/MALFORMED_JSON/
                           # UNKNOWN_TOOL/BUDGET_EXCEEDED/TOOL_LOOP/CONTEXT_OVERFLOW/CACHE_MISS_ANOMALY/THINKING_LEAK
tests/unit/test_gateway_pipeline.py   # 用 providers/mock.py 做端到端
tests/unit/test_obs_isolation.py
onyx/llm/providers/mock.py  # 脚本化响应（固定 usage、可注入畸形 tool JSON、可注入超时）
```

**接口要点**
- visitor 协议：`def on(self, ev: TraceEvent) -> Iterable[TraceEvent]`（可产新事件，如 `reconciled`）。
- visitor 抛异常 → 记 `anomaly(OBSERVER_ERROR)` 并继续，**绝不影响用户请求**。
- gateway 是唯一允许 `provider.generate()` 的地方；`grep -rn "\.generate(" onyx/ | grep -v gateway.py` 必须为空（写进自测）。

**自测**
```bash
uv run pytest tests/unit -q
uv run onyx chat "hi" --model mock/echo --dry-run     # 不碰 GPU，纯管道验证
uv run onyx traces ls --limit 3
```
`traces ls` 预期：
```
id                 purpose    model        in    out  ttft  tps   status  tools
01J9X...QK7        chat       qwen3:8b     1842   213  142ms 31.2  ok      0
```
必须覆盖：
1. mock 注入 `prompt_eval_count=None` → 降级到 heuristic 且 `confidence=low`；
2. mock 注入截断 JSON 工具调用 → `parse_status=truncated` + `args_raw` 保住了原文；
3. 两个 sink（sqlite + jsonl）同写，JSONL 行数 == DB trace 行数；
4. 故意让 `visitors/cost` 抛异常 → 请求仍 `status=ok`，且有 `OBSERVER_ERROR`；
5. `context={'eval_run_id':..,'case_id':..,'sample_seq':..}` 正确透传进 trace 行。

**验收 DoD**：**M1 出口判据在此达成** —— 一次真实对话后，`onyx traces show <id>` 能显示 token(带来源/置信度)、TTFT、TPS、cold/warm、工具列表、原始 body 指针；`onyx traces replay <id> --dry-run` 能重建等价请求。

**提交**：`feat(llm): gateway single throat + observer chain with fault isolation`

---

## S6 — 能力探测与模型档案落库

**目的**：让"这个模型能不能测工具调用、走哪种格式"成为**查出来的事实**，而不是配置里抄来的说法。

**产出文件**
```
onyx/probe/runner.py               # 编排探针：逐模型跑，写 model 表
onyx/probe/report.py               # 输出 docs/PROBES.md 追加段 + 控制台矩阵
onyx/llm/caps.py                   # Cap 推断：capabilities + 版本 + 探测结果 → frozenset[Cap]
tests/unit/test_caps_inference.py  tests/probe/test_probe_matrix.py(@probe)
```

**探测矩阵**（对每个模型产出一行，全部落 `model.tool_format/capabilities_json/extra_json`）
| 探测 | 判定依据 |
|---|---|
| tools 能力 | `/api/show.capabilities` 含 `"tools"` |
| thinking 开关 | `think:false` 是否真减少 `eval_count`（未生效则 task 侧禁用 think 参数） |
| 工具格式 | 最小工具调用的**原文**：出现 XML 标签→`xml`；纯 JSON→`json`；有 native tool_calls 字段→`native_head`；否则 `unknown` |
| structured | `format:json_schema` 后输出是否 schema 合法 |
| stream_usage | 流式末事件是否含计数（决定 T0 是否走流式路径） |
| 上下文上限 | `context_length` 实测（超限时是截断还是报错 → `CONTEXT_OVERFLOW` 判定策略） |

**自测**
```bash
uv run onyx probe matrix --provider ollama-local        # 表格：模型 × 能力位
uv run pytest tests/unit/test_caps_inference.py -q
```
预期矩阵形态：
```
model            tools  tool_choice  think  structured  tool_format  stream_usage  max_ctx
qwen3:8b         ✓      ✗(cap)       ✓      ✓           native_head  ✓             32768
llama3.2:3b      ✓      ✗(cap)       ✗      ✗           json         ✓             8192
```
必须覆盖：`unknown` 与 `✗(cap)` 是不同状态（一个是"不支持"，一个是"没测出来"），UI 与评测 skip 原因都依赖这个区分。

**验收 DoD（M1 完整出口）**：`probe matrix` 全绿且无 `unknown` 的工具格式列；`docs/PROBES.md` 每个结论可追溯到一次真实 trace id。

**提交**：`feat(probe): capability matrix and per-model behavior profiling`

---

## S7 — M1 收口：CLI 报表 + 证据回放 + 数据保留

**产出文件**
```
onyx/cli.py（补齐子命令：models/chat/traces/token/tools/eval/probe/calibrate/doctor/db）
onyx/report/exporters/{csv,md,jsonl}.py
scripts/rotate.py   # blob 与 trace 保留策略（默认 trace 90d / 原始 body 30d / eval 记录永久）
tests/unit/test_exporters.py  tests/unit/test_doctor.py
```
`onyx doctor` 检查项：Ollama 可达性/版本、DB 与迁移版本、磁盘与 blob 一致性（引用计数）、tokenizer 档位可用性、**每个模型是否有 probe 结论**、GPU 锁是否被残留占用。

**自测**
```bash
uv run onyx doctor            # 全绿；任何红项有修复提示
uv run onyx report usage --since 7d --format md > reports/week.md
uv run onyx db backup --to .tmp/backup.sqlite && uv run onyx db verify-backup .tmp/backup.sqlite
```
必须覆盖：`doctor` 在故意停掉 Ollama / 删一个 blob 文件后能报出具体项（不是笼统 500）；`verify-backup` 比对行数与 blob 摘要。

**验收 DoD**：M1 出口判据 + `doctor` 全绿 + 备份可恢复。

**提交**：`feat(cli): doctor, reporting, retention, backup verify`

---

## S8 — API 层 + 前端骨架 + Fleet 页

**产出文件**
```
onyx/api/{app.py,deps.py,sse.py}
onyx/api/routes/{fleet,models,traces,usage,admin}.py
onyx/web/  (Vite+React+TS+Tailwind) src/{main.tsx,App.tsx,api/client.ts,api/events.ts}
          src/pages/Fleet.tsx  src/components/{StatCard.tsx,TraceTable.tsx,SourceBadge.tsx,ConfidenceBar.tsx}
```
**接口要点**
- REST 只读优先；写操作只留 `admin`（pull/unload/delete）且需 `?confirm=1`。
- SSE `/api/stream` 转发事件流（新 trace、eval 进度）；前端断线自动重连并补 `Last-Event-ID`。
- **`SourceBadge` 是全站必备组件**：任何数字旁边必须显示 `engine/hf/fitted/heuristic` + 置信度。没有它，看板会在骗人。
- 分页 + 服务端排序（trace 表会大到前端拉不完）。

**自测**
```bash
uv run pytest tests/unit -q && uv run onyx serve --port 8000
curl -s localhost:8000/api/fleet | jq '{providers,loaded_models,rate_1h}'
curl -s "localhost:8000/api/traces?limit=5" | jq '.items[].usage.source'
cd onyx/web && npm i && npm run build && npm run test -- --run
```
必须覆盖（前端 vitest）：
1. `SourceBadge` 对 unknown 源渲染 "—" 而非 0；
2. cold/warm 两列在延迟卡上不混算；
3. SSE 断开重连不重复渲染（按 trace id 去重）。

**验收 DoD（M2 前半）**：Fleet 页显示服务存活/版本、已加载模型（keep-alive 倒计时、`size_vram`、`ctx_util`）、近 1h 调用量与 TPS、异常列表；数据与 CLI `onyx models ls` 一致。

**提交**：`feat(api,web): read-only REST + SSE, fleet dashboard`

---

## S9 — Traces 详情 + Playground

**产出文件**
```
onyx/api/routes/{playground,traces_detail}.py
onyx/web/src/pages/{Traces,TraceDetail,Playground,Usage}.tsx
onyx/web/src/components/{Timeline.tsx,PromptBreakdown.tsx,ToolTree.tsx,RawBody.tsx,CompareModels.tsx}
```
- `TraceDetail`：时间轴（load → first_token → 每步 tool 调用 → end）、消息树、**prompt 分段条形图（tool_defs 高亮）**、原始 body（可复制成 curl）、"转成评测 case"按钮。
- `Playground`：多模型并排（走 GPU 锁串行执行，UI 显示排队状态）；thinking 分栏；工具面板实时渲染 `tool_call_delta`。
- `Usage`（Token Ledger）：时序、来源占比、drift 散点、prefill vs decode。

**自测**
```bash
uv run pytest tests/e2e -m e2e -q        # Playwright: 发一条对话 → Traces 里出现 → 详情数字非空
```
必须覆盖：
1. 一次带工具调用的对话，详情页里 tool 步骤顺序与 `step` 字段一致；
2. 截断 JSON 的 trace 显示原文 + `truncated` 徽标（不是"解析失败"一句话）；
3. 并排 3 个模型时，GPU 锁使请求串行（断言 ttft 区间不重叠）；
4. "转成 case" 后 `case.input_json` 与原 trace 的 messages 等价。

**验收 DoD（M2 完整出口）**：浏览器完成"对话 → 定位 trace → 看归因 → 转 case"全链路，无控制台报错。

**提交**：`feat(web): trace detail with token attribution, playground over gpu lock`

---

## S10 — 工具注册表：定义、校验、token 开销

**产出文件**
```
onyx/tools/spec.py       # ToolDef / ToolResult / ToolError 族 / OpenAI↔内部↔MCP 形状转换
onyx/tools/registry.py   # CRUD + 版本(hash) + 校验 + 开销核算 + 启停
onyx/api/routes/tools.py
tests/contract/test_toolspec.py
```
**校验规则**（每条一个 rule id，违规产 warning/error 列表，不阻塞保存但看板标红）
`NAME_PATTERN` `^[a-zA-Z0-9_-]{1,64}$` · `SCHEMA_VALID`(draft 2020-12) · `DESC_MISSING`(工具或参数无描述)
`DESC_TOO_SHORT`(<20 字符) · `DESCRIPTION_BUDGET`(单工具 >400 token 警告) · `DUPLICATE_NAME`
`SIDE_EFFECT_UNTAGGED`(未标 read/write/network/exec) · `NO_EXAMPLE`(无 examples 难以 fire-verify)
`REQUIRED_MISMATCH`(required 与 properties 不一致) · `ADDITIONAL_PROPS_UNSET`

**自测**
```bash
uv run onyx tools import examples/tools.yaml
uv run onyx tools audit          # 逐条列出 rule id 命中
uv run onyx tools cost --model qwen3:8b
```
`tools cost` 预期（Tools 页核心视图，DESIGN §6.2）：
```
tool               tokens  bytes  share
search_web           312    1186   14.1%
db_query             268    1042   12.1%
...
TOTAL (24 tools)    1380    5410   62.4% of system+tools context
```
必须覆盖：`tools audit` 对故意写坏的 5 个 schema 各命中对应 rule id；hash 变化即生成新版本且旧 trace 仍能定位到当时的 `tool_def_hash`。

**验收 DoD**：开销数字与 `onyx token explain <trace>` 里 `part=tool_defs` 一致（同一套渲染代码，不许两处各算一次）。

**提交**：`feat(tools): registry, schema audit, per-model context cost accounting`

---

## S11 — 执行器 + 沙箱 + 契约测试运行器（无模型）✅

**产出文件**
```
onyx/tools/executor.py            # guarded_call：所有执行器的唯一入口
onyx/tools/args.py                # 参数校验（6 种细分失败）+ diff_args
onyx/tools/sandbox.py             # 副作用白名单 / 审批 / dry-run / impl_ref 白名单 / deadline
onyx/tools/contract.py            # 契约运行器：8 条断言 + 豁免机制 + sample_args
onyx/tools/executors/{python_fn,mock_replay,http}.py
onyx/tools/builtin/{calculator,time_now,echo,fs_read,defs}.py
examples/tools.yaml               # 5 个示例定义（含 constants 与 http 两种写法）
tests/unit/test_tools_executors.py  tests/unit/test_tools_http_fs.py  tests/unit/test_cli_tools.py
```

**与原计划的偏差（记录在案，不是悄悄缩水）**
- `mcp` / `ollama_builtin` 两个执行器推迟到 S16：MCP 在 S16 本来就是交付项；
  `ollama_builtin` 需要先跑 P21 探针确认引擎内建工具是否真由服务端执行，
  在测出结论前写实现等于把假设编码进代码。矩阵里这两列显示 `—` 并写明里程碑，
  **不显示为通过**。
- 不提供计划里列的 `builtin/http_get.py`：一个"抓任意 URL"的工具，参数来自模型输出，
  等价于把 SSRF 开放给模型。改为通用 `http` 执行器——URL 写死在人工审核过的定义里，
  模型给的参数只能进 query/body，因此改不了目标主机。
- 契约断言从 7 条增加到 8 条：新增 `mock_policy_makes_no_real_call` 的**证据分级**
  （`counter` = 实测计数，`structural` = 结构性质 + monkeypatch canary 兜底），
  因为"零真实调用"是整个评测可复现性的地基，不能靠一句声明。

契约断言集（**所有 executor 必须全过，n/a 须写明原因**）：
1. `valid_args_ok` 合法参数 → ok；
2. `missing_required_is_arg_error` 缺 required → `arg_error`（**不能崩、不能返回 200**）；
3. `wrong_type_is_arg_error` 类型错 → `arg_error`；
4. `unknown_tool_is_distinguished` 工具名不符 → `unknown_tool`，与参数错分开；
5. `timeout_is_reported` 超 deadline → `timeout`；
6. `write_denied_without_approval` write 未审批 → `rejected`，且副作用未发生；
7. `read_is_idempotent` read 类连调两次结果一致（`non-deterministic` 标记的工具判 n/a，
   因为两次调用落在同一秒就会"通过"——报一个靠运气的 ✓ 比报 n/a 危险）；
8. `mock_policy_makes_no_real_call` FIXTURE/DENY 下零真实调用，且每个结果都标 `mocked`。

**自测**
```bash
uv run onyx tools contract                       # 期望: 3 列全绿，n/a 均带原因，exit 0
uv run onyx tools contract --json                # 机器可读，供看板直接消费
uv run onyx tools import --builtin && uv run onyx tools audit   # error=0 warn=0 info=0
uv run onyx tools run calculator --args '{"expr":"(12+8)*3"}'   # ✓ ok, value=60
uv run onyx tools run echo --args '{}'                          # ✗ arg_error, exit 1
uv run onyx tools run echo --mock deny                          # ✗ skipped（mocked）, exit 1
uv run onyx tools import examples/tools.yaml
uv run onyx tools run fs_read --args '{"path":"../pyproject.toml"}'   # ✗ rejected path_traversal
uv run onyx tools run http_demo --args '{}'                       # ✗ rejected（默认只放 read）
```
实测输出（节选，2026-10-02）：
```
┌─────────────────────────────┬───────────┬──────┬──────┬─────┬───────────────┐
│断言                         │ python_fn │ mock │ http │ mcp │ ollama_builtin│
│valid_args_ok                │     ✓     │  ✓   │  ✓   │  —  │       —       │
│timeout_is_reported          │     ✓     │ n/a  │  ✓   │  —  │       —       │
│mock_policy_makes_no_real_c… │    n/a    │  ✓   │  ✓   │  —  │       —       │
└─────────────────────────────┴───────────┴──────┴──────┴─────┴───────────────┘
python_fn    通过 7 · 失败 0 · 不适用 1
mock         通过 7 · 失败 0 · 不适用 1
http         通过 8 · 失败 0 · 不适用 0
```

**故意注入的缺陷必须能被检出**（`tests/unit/test_tools_http_fs.py`、`test_tools_executors.py`）：
```
DeadlineIgnoringHttp（无视 ctx.deadline_ms）
  → handler 从 request.extensions["timeout"] 读到 read=15.0 > 0.05 → 返回 500
  → error_kind=error 而非 timeout → 断言变红；对照组（正常执行器）报 timeout
DeadlineIgnoringExecutor（抹掉 deadline）      → timeout_is_reported 失败且 applicable=True
UnguardedExecutor（绕过 guarded_call 直接返回成功）→ 3 条断言同时变红
```
> httpx 会把有效超时放进 `request.extensions["timeout"]`（已实测，见下），
> 所以"deadline 有没有传到 socket"是**可观测的**，不必靠一个真实慢服务器去猜。
> 这一步不传播的后果不是报错，而是线程池 worker 被卡住的请求永久占用——
> 现象是"越来越慢"，属于最难查的那一类。

**验收 DoD**：8 条契约断言在 3 个已实现执行器上全部适用并通过（`mcp`/`ollama_builtin`
显示为未实现并写明 S16，`python_fn`/`mock` 各自的 1 条 n/a 都带原因）；
3 种注入缺陷全部被检出；`ruff` + 3 条 import-linter 契约 + 全量离线测试通过。

**提交**：`feat(tools): executors, sandbox, contract runner`

---

## S12 — 工具循环 + 模型侧 fire-and-verify（M3 出口）✅

**产出文件**
```
onyx/tools/loop.py       # 多步循环 / 预算 / 循环检测 / 孤儿调用补齐 / 停止词（DESIGN §8.3）
onyx/tools/verify.py     # 指令→期望工具→实际调用 的差异报告（六种判定）
tests/unit/test_loop.py         # 31 条，用 MockProvider 驱动，含全部边界
tests/unit/test_verify.py       # 25 条
tests/integration/test_fire_verify.py  (@live，8 条，真机 qwen3.5:9b)
```

**为此改动的既有代码（都是必要的，不是顺手重构）**
- `Gateway.generate(before_trace_end=...)` + `TraceEmitter`：工具执行必须记在**发起该调用的
  那一步**的 trace 里，而 TRACE_END 由 gateway 发出，所以需要一个在关 trace 之前的回调口子。
  顺带把一直没人发的 `TOOL_EXEC_START/END` 事件用起来——`tool_call` 表的
  `result_status / latency_ms / executed_by / result_ref / result_bytes / tool_def_hash`
  六列此前永远是空的。钩子抛异常时仍然先关 trace 再抛，否则库里会留下永不结束的 trace。
- `TraceContext.root_trace_id`（新）与 `parent_trace_id` 分开：循环的 root 是**分组键**，
  没有对应的 trace 行。一开始塞进 `parent_trace_id`，结果撞 `FOREIGN KEY constraint failed`，
  整条记录写不进去，而表面上只是日志里一行警告——这是本步最容易踩、也最难发现的坑。
- `tool_call_fingerprint` 上移到 `core/types.py`：循环靠它熔断 TOOL_LOOP，tool visitor
  靠它聚合重复调用，两处必须算出同一个值。各写一份必然漂移，漂移之后一边报循环、
  一边报正常，这种矛盾比没有检测更难查。
- `MockProvider.scripts` 支持**脚本序列**：多步循环的测试必须让同一个模型在连续几次调用里
  返回不同结果。否则只能靠"每步换一个模型名"来绕，那样测的就不是真实形态了。

**与原计划的偏差**
- `--mock weather`（按工具名选择性打桩）改为 `--mock <策略>` + `--fixture <JSON>`：
  一个参数同时表示"策略"和"工具名"会有歧义，而策略是全局的、桩是按工具的，两者不该挤在一起。
- DoD 写的"5 个模型"按本机实际装的 3 个跑（`ONYX_TEST_MODEL` 可换）。

**必须实现并单测的边界**（每条至少一个用例，全部落地）：
`max_steps` 截断 · token 预算耗尽 · 墙钟预算 · `(name,args)` 重复 → `TOOL_LOOP` 熔断 ·
模型返回不存在的工具名 → `UNKNOWN_TOOL` · `finish_reason=tool_calls` 但无 tool_calls →
`ORPHAN_TOOL_CALL` · 工具异常后能继续对话 · 孤儿调用补齐占位（断言下一条请求 messages
结构合法，无悬空 tool 消息）· 截断 JSON 进 `args_raw` 且**原文参与指纹** ·
并行多调用 · 引擎故障 · 工具超时不带走循环。

**核心不变式**（有专门的断言函数，出现在 6 个用例里）：
> 一条带 N 个 `tool_calls` 的 assistant 消息，后面必须紧跟**恰好 N 条** `role=tool` 消息。

少一条，之后每次请求的上下文都永久错位，而引擎通常不报错——只是开始答非所问。
所以即使中途熔断，剩下的调用也要补占位结果。

**六种判定的分工**（`verify.py` 文档里有完整的"该改什么"对照表）：
`PASS` / `NO_CALL`（改提示词与工具描述）/ `WRONG_TOOL`（改工具之间的区分度）/
`BAD_ARGS`（改参数描述或加 max_tokens）/ **`TOOL_FAILED`（改工具，不是改模型）** /
`LOOP_BROKEN`（改工具返回值）/ `ERROR`（先跑 doctor）。
判定顺序即归因优先级：先看这一轮**有没有收敛**，再看调没调、调得对不对——
顺序反了会把"循环熔断"记成 PASS，因为第一次调用的参数往往是对的。
汇总的 `pass_rate` 分母排除 `TOOL_FAILED` 与 `ERROR`：否则模型要替坏掉的工具和挂掉的引擎背锅。

**自测**
```bash
uv run pytest tests/unit/test_loop.py tests/unit/test_verify.py -q          # 56 passed
uv run pytest -m live tests/integration/test_fire_verify.py -q              # 8 passed
uv run onyx tools import --builtin
uv run onyx tools fire "把 hello 原样回显一次" --model mock/echo --provider mock --tools echo
uv run onyx tools fire "把 hello 原样回显一次" --model qwen3.5:9b --tools echo,calculator --mock live
uv run onyx traces ls --limit 3        # 每一步都是独立 trace，靠 root_id 聚合
```
真机实测输出（qwen3.5:9b，2026-10-03，`--mock live`）：
```
期望      : echo {"text": "hello", "times": 1}
实际      : echo {"text": "hello", "times": 1}
参数差异  : {"ok": true, "missing": [], "unexpected": [], "mismatched": {}, "subset_ok": true}
步数      : 2   停止原因: final
判定      : PASS      真实执行
```
> 同一次运行里 `TOKEN_DRIFT` 报 32–37%：因为用的是**全新的空数据目录**，模型没有标定
> （`usage_ratio IS NULL`），本地复算退回 heuristic 档，而它按 ~1.0 token/字估中文、
> 该模型实际 ~0.679（P21）。采信值仍是 `engine/high`，异常码报得对——这不是缺陷，
> 是"未标定"这件事被如实标出来了。跑一次 `onyx calibrate` 即归零（P21）。

**验收 DoD**：NO_CALL / WRONG_TOOL / BAD_ARGS / TOOL_FAILED / LOOP_BROKEN / ERROR
六种判定互不相同且各有用例；`--mock fixture|deny` 时零真实工具执行（canary 断言）；
上下文不变式在真机与离线两侧都被验证；真机端到端跑出 PASS。

**提交**：`feat(tools): client-side tool loop + fire-and-verify harness`

---

## S13 — 评测内核 + 意图识别任务 ✅

**产出文件**
```
onyx/eval/task.py                              # EvalTask/Case/Grade/Skip 契约 + 能力跳过
onyx/eval/metrics.py                           # prf1/macro/balanced-acc/confusion/hit@k/pass^k/bootstrap-ci/jsonable
onyx/eval/graders/{normalize,exact,set_match,regex,json_schema,fuzz}.py
onyx/eval/datasets/loader.py                   # JSONL 载入 + 来历 + 子集选择
onyx/eval/datasets/builtin/intent_zh.py        # 数据集**生成器**（不是来历不明的 JSONL）
onyx/eval/datasets/builtin/intent_zh.jsonl     # 236 条，由生成器产出
onyx/eval/tasks/intent_classification.py       # + tasks/__init__.py 的任务注册表
onyx/eval/runner.py                            # build → gateway → grade → aggregate
onyx/store/migrations/0004_eval.sql            # dataset/eval_case/eval_task/eval_run/grade
onyx/store/repos/eval_repo.py                  # + records.py 的 5 个记录类
tests/unit/{test_metrics,test_graders,test_intent_task,test_eval_runner,test_cli_eval}.py
```

**数据集画像**（`intent_zh.py` 的 `stats()`，seed=20261003）
```
n=236  hard=32  unique=236（无重复）
labels: 转账 73 / 投诉 62 / 其他 53 / 查余额 48   ← min/max = 0.66，不均衡但不至于让 accuracy 骗人
子集: default 236 · hard 32 · template 204
```
刻意包含 `其他` 兜底类：没有它，模型面对越界输入只能硬塞进四个类之一，
于是"越界标签率"永远测不出来——而它恰恰是幻觉的主要信号。
生成器打散顺序后再编 `ord`，所以 `--limit 20` 取到的前 20 条覆盖 4 个类（有测试断言）。

**指标的三条纪律**
1. **「未定义」与「0」分开，分界只有一条：这个数算不算得出来。**
   P 或 R 的分母是 0（这个类一条都没考到）⇒ F1 未定义 ⇒ 返回 `None`，宏平均**跳过它**；
   填 0 会让一个根本没考到的类把 macro 平均拖低，看起来像模型能力差。
   反过来 `P=R=0` 两边都算得出来，F1 就必须是 **0.0**——那是"全错"这个事实。
   （这条边界在 S13 写得太宽，S14 才修正，见下面的缺陷表。）宏平均**只对定义得出来的类**求平均。
2. **CI 必须真的在算**：重采样单位是 case，统计量每次重算。把算好的分数重排求均值，
   对 macro_f1 这类非线性统计量会得到一个"看起来合理但是错的"区间。
3. **零宽区间不等于确定**：20 条全对时每次重采样仍全对，区间就是 [1.0, 1.0]。
   这是退化 bootstrap，所以 `low_confidence`（n<100）必须跟着一起看。

**Grade 的两个正交维度**（DESIGN §9.4 的落地点）

| 模型输出 | verdict | invalid_format | 说明 |
|---|---|---|---|
| `转账` | correct | False | 干净且正确 |
| `查余额` | wrong | False | 格式对了内容错——真的分类错误 |
| `退款` | out_of_label | True | **标签集之外的幻觉**，不是"选错"，不进混淆矩阵 |
| `这个意图是转账。` | correct | **True** | 内容对但没遵守格式；仍可判，但格式合法率要扣 |
| `不是转账，是查余额` | invalid_format | True | 出现两个候选 ⇒ **不可判定**，绝不猜 |
| 空正文 + 有 thinking | invalid_format | True | P12：预算被 thinking 吃光，与"真的没输出"分开报 |
| 引擎失败 | error | — | `attributable=False`，不进模型能力分母 |

**自测**
```bash
uv run pytest tests/unit/test_metrics.py tests/unit/test_graders.py -q       # 72 passed
uv run pytest tests/unit/test_intent_task.py tests/unit/test_eval_runner.py -q  # 54 passed
uv run pytest tests/unit/test_cli_eval.py -q                                  # 23 passed
uv run onyx eval import --builtin intent_zh
uv run onyx eval run --task intent_classification --model mock/echo --provider mock --limit 20 --seed 42
uv run onyx eval run --task intent_classification --model qwen3.5:9b --limit 20  --seed 42 --json
uv run onyx eval run --task intent_classification --model qwen3.5:9b --limit 200 --seed 42 --json
uv run onyx eval show <run_id>          # 每条 grade 都带 trace_id，可直接跳进那条 trace
uv run onyx eval ls
```
真机实测（qwen3.5:9b，2026-10-03）：

| | n | macro_f1 | 95% CI | 宽度 | acc | format_valid | out_of_label | 错误 |
|---|---|---|---|---|---|---|---|---|
| `--limit 20` | 20 | 1.0000 | [1.000, 1.000] | 0.000（退化，已标 ⚠低样本） | 1.000 | 1.000 | 0.000 | 0 |
| `--limit 200` | 200 | 0.9898 | [0.974, 1.000] | 0.026 | 0.990 | 1.000 | 0.000 | 0 |
| 全量 | 236 | 0.9913 | [0.978, 1.000] | 0.022 | 0.992 | 1.000 | 0.000 | 0 |

全量那次：236/236 完成、0 错误、38 秒、20,395 in / 521 out token；
混淆仅 2 处（转账→查余额 ×1、投诉→其他 ×1）。
> 分数很高有一部分是数据集本身的性质：它由模板生成，分布与提示词风格一致。
> 真机验证的是**管道正确**，不是"这个模型意图识别很强"。要得出后者需要真实语料。

**这一步改掉的一个既有缺陷**：`eval ls` / `eval show` 的主分数在 `macro_f1=None` 时
会悄悄改显示 `pass_hat_k` 的 `0.000`——把"一条可判定样本都没有"显示成"模型得了 0 分"。
现在主分数**必须带指标名**，未知就是 `macro_f1 —（无可判定样本）`（UI_DESIGN R2）。
同类问题：`CI` 是 dataclass，走 `json.dumps(default=str)` 会落成 `"CI(low=…)"` 字符串，
写进去不报错、读出来取不到上下界，置信区间静默消失 ⇒ 新增 `metrics.jsonable()`，
在落库与 `--json` 两个出口统一转换，并有回归测试盯着。

**验收 DoD**：mock 与真实模型都能跑完；`--limit 20` 与 `--limit 200` 的 macro_f1 有可见差异
且 CI 变宽（0.000 → 0.026，证明区间真的在算）；每条 grade 都有 trace_id 且指向库里真实存在的
trace；`ruff` + 3 条 import-linter 契约（新增 `onyx.eval` 禁止网络 IO）+ 565 条离线测试全过。

**提交**：`feat(eval): task contract, metrics with CI, intent classification`

---

## S14 — 工具调用评测 + runner 调度（M4 出口）✅

**产出文件**
```
onyx/eval/graders/args_match.py                     # 类型感知参数比对，每条判定都带 kind
onyx/eval/tasks/tool_selection.py                   # 七个 verdict；选择维与参数维分开
onyx/eval/datasets/builtin/tool_calls_zh.{py,jsonl}  # 97 条手写样本，含 25 条 no_call_needed
onyx/eval/gpu_lock.py                               # 跨进程文件锁 + 心跳 + ETA + 死锁回收
onyx/eval/datasets/sources.py                       # BFCL 导入器（upstream/revision/license）
onyx/eval/runner.py                                 # GPU 锁、unload-others、resume、取消、进度检查点
onyx/api/routes/evals.py                            # /api/datasets /api/runs /api/runs/{id}/grades /api/gpu
onyx/api/deps.py                                    # AppState 持有 gpu_lock；serve 可 --gpu-lock 覆盖
tests/unit/{test_args_match,test_tool_selection_grade,test_gpu_lock,test_sources,test_api_evals}.py
tests/integration/conftest.py                       # live 套件也参与同一把 GPU 锁
```
未做：`structured_extraction`、`instruction_following`。M4 的 DoD 只要求"两个 task 各有一次真实运行"，
已满足；这两个任务需要的 `Cap.STRUCTURED_OUTPUT` 对照实验（强制 vs 自由）当前在 Ollama 上做不到，
留到 S15 之后与 IFEval 风格 checker 注册表一起做。

**数据集画像**（`tool_calls_zh.py`，seed=20261003）
```
n=97  kinds: single 46 · no_call_needed 25 · args 15 · parallel 11
8 个工具，其中 send_email 是 write 副作用 —— 它出现在**每一条**样本的工具集里，
但没有任何样本期望调用它
```
`send_email` 一直在场才有意义：只在场一次测不出"模型会不会因为工具存在就乱用"，
而误调率（`false_call_rate`）恰恰是工具评测里最贵的那个错误——它真的会发信。
有测试同时断言这两件事（每个 case 的工具集含它、且没有任何期望调用它）。

**`args_match` 的判定口径**（为什么不能用 `==`）

| kind | 触发条件 | 算不算"放宽" |
|---|---|---|
| `identical` | 字面相等 | 否 |
| `enum` | schema 声明了 enum 且归一化后命中 | **否**——大小写差异命中同一条规则，不是变宽松 |
| `normalized` | 去空白/全半角/前后缀后相等 | 是 |
| `numeric_tolerance` | 数值在 abs/rel 容差内，或一边是数字字符串 | 是 |
| `date_normalized` | `2026-10-03` ↔ `2026年10月3日` ↔ `2026-10` | 是 |
| `set_equal` | array + `uniqueItems` 时按集合比 | 是 |
| `fuzzy` | 仅对显式列入 `fuzzy_fields` 的字段生效，默认不开 | 是 |
| `type_mismatch`/`value_mismatch`/`missing`/`unexpected` | 判错 | — |

每条判定都把 `kind` 写进 grade，于是 `relaxed_share` 算得出来：
**只报"参数对了"会把"我们把比对放宽了"藏起来，分数变高就看起来像模型变强了。**
`exact_rate` 用的是 `strict_ok`（对了且没靠任何放宽规则），它与 `semantic_rate` 的差就是放宽的贡献。
`bool` 必须在数值之前判（Python 里 `True == 1`），期望值不在 schema 的 enum 里要**报错**而不是判错——
那说明用例写坏了。

**七个 verdict，因为七种的修法不同**：`correct` / `no_call`（改提示词）/ `wrong_tool`（改工具描述区分度）/
`bad_args`（改参数 description 与 required）/ `hallucinated_tool`（限工具命名）/ `invalid_format`（max_tokens 与模板，P20）/
`false_call`（不判成 verdict 而是独立指标 `false_call_rate`，因为它的修法与前六个都不同）。
`no_call_needed` 的误调绝不并进"没调对"：一个从不乱调的模型和一个不会调的模型，合并后分数一样，但它们相反。

**比率一律折算到 case**：bootstrap 的重采样单位是 case，同一个 case 的 k 次采样不是 k 个独立观测
（temperature=0 下它们几乎是同一个答案）。按 sample 算会让区间窄得像"模型很确定"，其实只是同一件事被数了三遍。
所以 `must_call_acc_ci.n`、`n_bootstrap_units` 与 `low_confidence` 说的是同一个数；
`by_kind` 同时给 `n`（sample）与 `cases`，否则 `n=138` 会被读成 138 个独立观测。

**GPU 锁的关键决定**（DESIGN §8.5 的落地）
1. **文件锁 + 心跳**，不用 OS advisory lock：Windows 没有 `flock`；而且锁只有两态时排队者只能干等，
   于是人去 kill 进程——那正好留下半截运行。锁文件里带 `done/total`，排队者能算出 ETA。
2. **判活用靠心跳过期，不靠 PID 存活**（`os.kill(pid, 0)` 在 Windows 上语义不同且可能误伤）。
   阈值必须明显大于单条样本耗时：CLI/serve 用 600s，live 套件用 900s。调小会误伤活着的持有者。
3. **锁路径是机器级全局的**（`tempfile.gettempdir()/onyx-gpu.lock`），不跟 `ONYX_DATA_DIR` 走：
   数据目录可以按实例覆盖，GPU 不行。跟着数据目录走的话两个实例各锁各的文件，然后照样同时塞显存。
4. **接管靠 `os.rename` + 内容复验**，不靠"覆盖后回读确认"：顺序执行的两个接管者用后者**都会成功**
   （后一个回读看到的是自己，前一个早已返回）。两个持有者比没有锁更危险，因为它看起来是安全的。
5. 坏文件（写入方崩在半路）当成没锁，让下一个进程接管；`release` 只删自己的锁。

**参与方**：`onyx eval run`（默认排队，`--no-queue` 立刻失败、`--lock-timeout` 限时、`--gpu-lock` 覆盖路径）、
`onyx serve` 的 Playground（忙时 HTTP 429，body 里带持有者与 ETA）、`GET /api/gpu`（只读，不参与竞争）、
以及 `pytest -m live`。**最后一条是补的缺陷**：live 套件原先完全不参与锁，与评测并发时
所有延迟与吞吐数字失真但不报错——我就是这么撞上 `test_unload_releases_model` 失败的。
反过来，**离线**测试绝不能碰这把真锁：`create_app(..., gpu_lock_path=...)` 给了覆盖口，
CLI/API 的离线测试全部指到 tmp，于是"有人在跑评测时 pytest 挂住"不会发生
（本次实测：真实评测正在跑时，62 个 CLI/API 测试照常通过）。

**runner 的调度语义**：锁在 `insert_run` **之前**拿（超时失败不留 `status=running` 的僵尸记录）；
`--unload-others` 在拿到锁之后卸掉其它已载入模型，结果记进 `cost.unloaded_models`；
心跳在每条 sample 后打；`progress_every`（默认 10）把 `n_done` 落库一次，
这样崩在中途时那一行也讲得出现在哪——只在收尾写的话，那一行会是 `n_done=0` 而库里已有几百条 grade。
续跑时 **cost 接续之前那一段**（`_carried_cost`）：本地评测最贵的就是 GPU 时间，
不接续就会出现"0 tok · 0 请求 · 0 ms"而分数齐全的正常运行记录。

**自测**
```bash
uv run pytest tests/unit/test_args_match.py tests/unit/test_tool_selection_grade.py -q
uv run pytest tests/unit/test_gpu_lock.py tests/unit/test_sources.py -q
uv run onyx eval import --builtin tool_calls_zh
uv run onyx eval run --task tool_selection --model mock/echo --provider mock --limit 10
uv run onyx eval run --task tool_selection --model qwen3.5:9b --k 3 --seed 7 --unload-others
uv run onyx eval run --task tool_selection --model qwen3.5:9b --k 3 --resume <run_id> --quiet  # 0 请求，只重算聚合
uv run onyx eval show <run_id> --verdict bad_args        # 每条 grade 带完整 trace_id 与下钻命令
uv run onyx eval import <questions.jsonl> --source bfcl --answers <answers.jsonl> --subset ast
```
并发起两个 `eval run` 的实测现象（第二个用 `--lock-timeout 2`，等过 2 秒后失败退出）：
```
[i] GPU 当前由 eval:intent_classification@qwen3.5:9b 占用（进度 60/240），排队中…
GPU 被 eval:intent_classification@qwen3.5:9b 占用，进度 60/240，预计还需 12s
[i] 详情: eval:intent_classification@qwen3.5:9b 进度 60/240 预计还需 12s；
    不想排队可以用 --no-queue 立刻失败，或 --gpu-lock 换一个锁文件          # exit 3
```
`--no-queue` 时第一行的措辞会变成"不排队，直接失败"——说"排队中"会让人一直盯着
一个已经退出的进程。ETA 拿不准时写「未知」而不是 0s：`0s` 会被读成"马上就轮到我"。

**真机实测（qwen3.5:9b，temperature=0，2026-10-03）**

| task | n | 头号分数 | 95% CI | 判定分布 | 成本 |
|---|---|---|---|---|---|
| `intent_classification` | 236 case × k1 | `macro_f1 0.991` | [0.978–1.000]（n=236） | correct 234 / wrong 2 | 20,395 in · 521 out · 37.3s |
| `tool_selection` | 97 case × k3 | `must_call_acc 0.639` | [0.528–0.736]（n=72 个该调的 case，⚠97 总样本） | correct 213 / bad_args 69 / no_call 6 / wrong_tool 3 | 401,739 in · 23,265 out · 13.1min |

`tool_selection` 全口径（run `01M3ZZPQC1…`）：
```
[内容] must_call_acc 0.639 [95% CI 0.528–0.736]（n=72 case）
[选择] hit_at_1 1.000  set_precision 0.986  set_recall 0.986  set_f1 0.986
       no_call_rate 0.028  wrong_tool_rate 0.014  false_call_rate 0.000
[参数] args_exact 0.684  subset 0.684  field 0.774  relaxed_share 0.000
[格式] hallucinated_tool 0.000  parse_fail 0.000
[稳定] pass^3 0.732 = pass@3 0.732（缺口 0.000）
[分类] no_call_needed 1.000 (25 case) · single 0.696 (46) · parallel 0.636 (11) · args 0.467 (15)
```
怎么读这份数字：
- **选工具几乎没错**（`hit_at_1 1.000`、`set_f1 0.986`、`hallucinated 0.000`——70 个可判定的 case 里
  只有 1 个选错），掉分几乎全在**参数**上（`bad_args 69/291`）。该改的是参数 description 与 required，
  不是提示词。`args_match_kinds` 给出细节：`identical 153 · enum 114 · value_mismatch 75 · unexpected 36 · missing 3`——
  多给字段（unexpected）与值不对几乎一样多。
- `relaxed_share 0.000`：0.684 的参数一致率里没有一条是靠放宽规则挣来的（命中只有 `identical` 与 `enum`）。
  这一条与上一条互相印证，也意味着这个数字不需要"会不会是比对器太宽容"的免责声明。
- `false_call_rate 0.000`：在场 97 次的 `send_email` 一次都没被误用——这是这份数据里最值钱的一个 0。
- `pass^3 == pass@3`：temperature=0 下同一 case 的三次采样几乎总是同一个答案，所以缺口 0 是**这个设置的下界**，
  不是"这个模型很稳"的结论。要谈稳定性必须升温度重测。
- ⚠低样本：97 个 case < 100，所以区间只说明"测过了"，不足以支撑模型之间的取舍决策。
  区间自身也印证了这点：修掉"按 sample 重采样"之后，同一个 0.639 的区间从 [0.579–0.704]
  变成 [0.528–0.736]——宽了约 40%，而那才是 97 个 case 该有的宽度。

**这一步修掉的缺陷**（共同点：都不崩溃，都只是"数字看着正常但含义错了"）

| 症状 | 根因 |
|---|---|
| `--resume` 之后 cost 变成 `0 tok · 0 请求`，而分数齐全 | 收尾直接写本段的 cost，没接续上一段 |
| `low_confidence=False`，而 CI 自带的 n 是 97 | 用 grade 条数（291）而不是 case 数判低样本 |
| `must_call_acc_ci` 的 n=216（应是 72 个 case） | CI 按 sample 重采样，同一个 case 被数了 k 遍 |
| 声明了 `set_precision/set_recall` 却没产出 | UI 读它们时永远显示「—」，与"这项 0 分"在界面上长得一样 |
| `exact_rate` 与"匹配率"是同一个数 | 没区分"对了"与"靠放宽规则才对" ⇒ 新增 `ArgMatch.strict_ok` |
| `set_f1 1.000` 却 `set_precision 0.986` | `P=R=0` 被判成"未定义"并从 F1 均值里剔除，而它照样进 P/R 的均值 ⇒ 三个数分母不同。**F1 的未定义只有一条边界：P 或 R 自己算不出来**；`P=R=0` 是"全错"这个事实，必须等于 0（`prf1` 同源修掉：否则错得最彻底的类不参与宏平均，模型越差 macro_f1 越高） |
| 报告与列表页只有 `macro_f1` 带区间 | CI 的查找写死了指标名 ⇒ 现在按「指标名 + `_ci`」通用查找，并把重采样单位一起打印（`（n=72 case）`） |
| live 套件与评测并发时延迟数字失真但不报错 | `pytest -m live` 完全不参与 GPU 锁 |
| 离线测试会在有人跑评测时挂住 | CLI/API 测试默认去抢机器级那把**真**锁 |
| `eval show` 的 trace 列只有 12 个字符 | 截断后跳不进去；补完整 id 的下钻提示 |
| `--builtin` 配 `--source bfcl` 时后者被静默忽略 | 参数组合错了要报错：使用者会以为自己导入的是 BFCL |
| 接管过期锁的两个进程可能都成功 | "覆盖 + 回读确认"对顺序执行不设防 ⇒ 改成 rename + 内容复验 |

**验收 DoD（M4 完整出口）**
- 两个 task 各有一次真实运行：`intent_classification`（236 条，`01M3ZWSQ…`）与
  `tool_selection`（97×3=291，`01M3ZZPQ…`），都在默认数据目录里，看板直接读得到
- `eval show` 任一分数能跳到 trace：291 条 grade **全部**带 `trace_id`，
  `onyx traces show <完整 id>` 直接可用；`GET /api/runs/{id}/grades` 同样暴露 trace_id 并返回 200
- 评测自身的耗时与 token 计入运行记录：`eval_run.cost_json` =
  `401,739 in / 23,265 out / 291 requests / 787,422 ms`；
  并且每条 grade 的 trace 都有 `usage.source=engine, confidence=high`
  ——评测同时是观测的数据来源（DESIGN §15），这条现在是可验证的事实而不是设计意图

**提交**：`feat(eval): tool calling tasks, args matcher, gpu-locked runner`

---

## S15 — 对比矩阵、配对回归、Eval UI 与报告导出（M5）✅

**产出文件**
```
onyx/eval/compare.py                     # 配对对比：per-case 折算 + 配对 bootstrap CI + 劣化清单
onyx/report/eval_report.py               # 模型×任务矩阵 + md/csv/自包含 html 导出（含内联 SVG 雷达）
onyx/store/migrations/0005_eval_provenance.sql  # eval_run 补 dataset_id/dataset_revision（含历史回填）
onyx/api/routes/evals.py                 # + /api/matrix、/api/compare
onyx/cli.py                              # eval compare / eval matrix / eval report
onyx/web/src/pages/{EvalRuns,EvalMatrix,Regression}.tsx
tests/unit/{test_compare_paired,test_eval_report,test_api_compare}.py
onyx/web/src/__tests__/eval.test.ts
```
**为什么"比较两个平均值"是错的**（这一步的全部立意）

均值差 0.02 可能是 30 条变好、28 条变坏相互抵消的结果——那不是"略好"，是"在两类任务上方向相反"。
所以 `Comparison` 的头一行是三个计数（净改善 / 净劣化 / 不变），均值差与区间跟在后面，
再下面必须是劣化清单，清单里每条都带**两个模型各自的 trace_id**：
差在哪道题、两边分别怎么想的，只有并排打开那两条 trace 才能判断。

配对 CI 的做法：先在 case 层面算差值，再**重采样 case 后重算均值差**。
常见错误是"把两个 run 各自的 CI 摆在一起看是否重叠"——两条独立区间的重叠检验远比配对检验保守，
n 小的时候几乎永远"不显著"，于是真回归被读成噪声。

另外报一对翻转数（McNemar 的两个不和谐格）：分数均值会被部分分抹平，
而"这道题从会答变成不会答"是离散事件，只有按 pass^k 逐条配对才数得出来。

**`eval_run` 现在记数据集来历**（0005）。这不是元数据装饰：M5 的每个结论都默认
"两次跑的是同一份考卷"，而原先只能靠 case_id 反推。迁移带历史回填（从 grade→eval_case 取多数），
并有测试用"升级前的库"验证回填真的执行——回填是迁移最容易糊弄过去的一步。

**矩阵的三条规则**
1. 每格取该 (模型, 任务) **最新一次 done** 运行，不取历史最好成绩；running/cancelled 不进网格。
2. 出现多于一份数据集 ⇒ 顶部警告，跨列比较被明说成无意义。
3. **覆盖率**：主分数只统计到不到一半样本时，格子里直接写"可判定 8/236"并计入警告。

第 3 条来自一次真实误读：`gpt-oss:20b` 的 236 条里有 226 条**没有正文**
（`max_tokens=32` 全被 reasoning 吃光，P12），于是它的 `macro_f1` 只在剩下 8 条上算出 **1.000**——
单看矩阵会得出"这个模型更强"，而它的 `format_valid_rate` 只有 3.4%。
数字是真的，读法是错的；界面必须把分母一起摆出来。

**自测**
```bash
uv run pytest tests/unit/test_compare_paired.py tests/unit/test_eval_report.py tests/unit/test_api_compare.py -q
uv run onyx eval matrix
uv run onyx eval compare <runA> <runB>                       # 终端：净变化 + 区间 + 劣化清单
uv run onyx eval compare <runA> <runB> --format md --out reports/diff.md
uv run onyx eval report --format html --out reports/matrix.html --compare <runA>:<runB>
cd onyx/web && npx tsc --noEmit && npx vitest run && npm run build
```
浏览器实测（`#/eval`、`#/eval/matrix`、`#/eval/regression`，真数据，零 console 错误）：
运行页点行→grade 面板→点 trace 进 TraceDetail；矩阵点格子→`/eval/run/<id>` 并选中那次运行；
回归页选任务+两次运行后给出 `qwen3.5:9b → gpt-oss:20b：劣化 226 / 不变 10，均值差 −0.958，
95% CI [−0.983, −0.932]（n=236）`，逐题表带 base/target 两个 trace 链接。

**真机配对结论（intent_classification，同一份 intent_zh-v1@seed=20261003）**

| | qwen3.5:9b | gpt-oss:20b |
|---|---|---|
| 主分数 | `macro_f1 0.991 [0.978–1.000]`（n=236） | `macro_f1 1.000 [1.000–1.000]`（n=8，**可判定 8/236**） |
| 格式合法率 | 1.000 | 0.034 |
| 判定分布 | correct 234 / wrong 2 | invalid_format 226 / correct 8 / out_of_label 2 |
| 成本 | 20,395 in / 521 out · 37.3s | 35,856 in / 7,549 out · 3.9min |

配对（qwen → gpt-oss）：**改善 0 / 劣化 226 / 不变 10**，均值差 **−0.9576**，
配对 95% CI **[−0.9831, −0.9322]**（n=236 case，覆盖率 100%）。
劣化清单里 target 侧判定齐一全是 `invalid_format`。下钻进那条 trace 才看到真正的原因：
`正文为空但产出了推理内容：预算被 thinking 吃光（P12），提高 max_tokens 或确认 thinking=False`
——`max_tokens=32` 对 gpt-oss 的 reasoning 来说整个预算都被 thinking 用掉了，正文一个字没剩。
所以结论**不是**"gpt-oss 意图识别差"，而是"这个提示词预算对它不成立"，该改的是
`max_tokens` / `thinking` 参数，而不是换模型。这个判断只有"分数 → trace"这条路走得通才做得到，
也正是 §15 那条主张的第一个真实回报点。

**这一步修掉的缺陷**
| 症状 | 根因 |
|---|---|
| `eval_run` 不记数据集来历 | 跨版本的两次运行看起来可比；回填 + 强制在写入时记录 |
| `RunView` 声明了 `dataset_id` 却没填 | 界面显示「—」，看起来像"数据缺失"而不是"API 漏传" |
| `StatusBadge('done')` 渲染成 `? done` | 映射表里没有评测的状态；而 "?" 在本项目里专指"未实测" |
| 矩阵只报 `macro_f1 1.000` 不报分母 | 见上面 8/236 的真实误读 |
| 报告 CI 用单 run 的 n<100 阈值 | 配对口径是 n<30；同一载荷里两个"低置信"标记互相矛盾 ⇒ 对比自己覆盖 |
| 差值 0 有被渲染成「—」的风险 | 0 是"没变化"的结论，未知是"没配对上"；`fmtDelta(0)` 必须是 `±0.000` |

**验收 DoD（M5 出口）**
- "该用哪个"：`eval compare` / 回归页给出净改善、净劣化、配对 CI 与劣化清单（真机一组见上）
- "在哪些题上更差"：逐题 Δ 表 + 题干（从 `eval_case` 取，不靠 grade.metrics 的私有约定）
  + 两侧 trace_id 可并排打开
- "报告可脱离看板阅读"：`eval report --format md|csv|html` 三种产物都带数据集来历、
  可比性警告与每次运行成本；html 自包含（内联样式，无外部依赖），可直接发给别人
- 雷达图只在任务数 ≥3 时画；两个任务时明说"两点的形状没有信息量"而不是硬画

**提交**：`feat(eval): paired regression diff, matrix UI, exportable reports`

---

## S16 — 扩展点固化 + 第二 Provider + MCP（M6）

**产出文件**
```
onyx/plugins_example/{example_task,example_provider}/   # 独立可 pip -e 安装的小包
onyx/llm/providers/openai_compat.py                     # vLLM / LM Studio / Xinference（验证抽象）
onyx/tools/executors/mcp.py                             # 任意 MCP server 的工具纳入注册表与测试
onyx/store/sinks/otlp.py（或 langfuse.py）               # 事件流导出，验证 Sink 抽象
docs/eval-recipes.md
tests/contract/test_plugin_discovery.py
```
这一步是**抽象的验收测试**：
- 第二 provider 必须**只实现 `LlmProvider`** 就能让全部看板与评测工作；若需要改 `core/` 或 `gateway.py`，说明抽象泄漏 → 记 issue 并在 DESIGN §13 补契约，不许在 gateway 里加 `if kind == ...`。
- 插件发现：外部包注册一个 `EvalTask` 并跑通一次，`onyx eval tasks` 能看到它。
- MCP：发现工具 → 自动进 `tool_def`（带 `kind='mcp'`）→ 能跑 contract 与 fire-verify。

**自测**
```bash
uv add -e onyx/plugins_example/example_task && uv run onyx eval tasks | grep example
uv run onyx providers add openai-compat --base-url http://127.0.0.1:1234/v1 --name lmstudio
uv run onyx probe matrix --provider lmstudio        # 能力位与 skip 原因要合理
uv run pytest tests/contract -q
```
**验收 DoD（M6 出口）**：`git diff` 显示接入新 provider **未修改** `onyx/core/**`、`onyx/llm/gateway.py`、`onyx/obs/**`；新 task 未修改 `onyx/eval/runner.py`。这条用脚本断言（`scripts/check_extension_boundary.sh`），不靠人review。

**提交**：`feat(plugins): entry-point discovery, openai-compatible provider, mcp executor`

---

## 附录 A — 每步自测速查

| 步 | 命令 | 绿的条件 |
|---|---|---|
| S1 | `pytest tests/unit -q` | 全过 + `grep` 证明 core 零三方依赖 |
| S2 | `onyx db init && onyx db info` | `schema_version=1`，迁移可重复执行 |
| S3 | `onyx chat --stream` | 4 项数字与 `curl+jq` 原始返回一致 |
| S4 | `onyx probe run --suite usage,cache,think` | `docs/PROBES.md` 有结论；`token explain` 多源对比表 |
| S5 | `onyx traces show <id>` | token/延迟/工具/原始 body 齐备，来源与置信度可见 |
| S6 | `onyx probe matrix` | 无 `unknown` 工具格式；`✗(cap)` 与 `unknown` 可区分 |
| S7 | `onyx doctor` | 全绿；破坏性测试能报出具体项 |
| S8 | `curl /api/fleet` + `npm run test` | 与 CLI 数字一致；unknown 渲染为 — |
| S9 | `pytest -m e2e` | 对话→trace→归因→转 case 全链路 |
| S10 | `onyx tools cost` | 与 `token explain` 的 `tool_defs` 同源一致 |
| S11 | `onyx tools contract` | 8 条断言 × 3 executor 全过；n/a 均带原因；注入缺陷能检出 |
| S12 | `onyx tools fire --provider mock` | 六种判定可区分；fixture/deny 档零真实执行；上下文不变式成立 |
| S13 | `onyx eval run --task intent_classification` | macro_f1 + CI + 混淆对；mock 也能跑；grade 均带 trace_id |
| S14 | `onyx eval run --task tool_selection --k 3` | skip 带原因；resume 不重复计费 |
| S15 | `onyx eval compare A B` | 配对净变化 + CI；n 小有警告 |
| S16 | `scripts/check_extension_boundary.sh` | 接入新 provider/task 未碰内核 |

## 附录 B — 架构自测（让"模块化"可验证，而非口号）

`Makefile` 关键 target：
```makefile
test:        ; uv run pytest -q
test-live:   ; uv run pytest -m live -q
probe:       ; uv run pytest -m probe -q && uv run onyx probe matrix
lint:        ; uv run ruff check . && uv run lint-imports
e2e:         ; uv run pytest -m e2e -q
dev:         ; uv run onyx serve --reload & cd onyx/web && npm run dev
```
`.importlinter`（DESIGN §3 那张表的机器版）：
```ini
[importlinter]
root_packages = onyx
[importlinter:contract:layers]
name = Onyx layering
type = layers
layers =
  onyx.api | onyx.web | onyx.report
  onyx.eval
  onyx.tools
  onyx.obs | onyx.probe
  onyx.llm
  onyx.store
  onyx.core
[importlinter:contract:core-purity]
name = core uses stdlib only
type = forbidden
source_modules = onyx.core
forbidden_modules = httpx fastapi pydantic duckdb tokenizers minja gguf
[importlinter:contract:no-direct-http]
name = only llm layer performs network IO
type = forbidden
source_modules = onyx.eval onyx.tools onyx.obs onyx.api
forbidden_modules = httpx
```
> `no-direct-http` 是原则 1（单一咽喉点）的**强制实现**：评测/工具/API 想绕过 gateway 直接发 HTTP，`make lint` 就红。
> 另加一条 grep 断言进 S5 自测：`.generate(` 只出现在 `gateway.py` 与 `providers/`。

## 附录 C — 提交点与回滚

每一步一个 commit（见各步末尾）。任何步骤验收不过 → **不要继续下一步**，回退到该步起点重做；因为后续步骤的正确性建立在前序契约上，跳步会让"哪一层破了"不可判定。

## 附录 D — 需要人工介入 / 无法自动验收的点

| 项 | 为何不能自动 | 建议 |
|---|---|---|
| `intent_zh.jsonl` 标注质量 | 需要业务口径 | 先 200 条，跑一次后按混淆对补最难样本；错例走"trace 转 case"回灌 |
| 中文工具调用指令的自然度 | 自动指标只看结构 | 每 task 留 20 条人工阅读集，`onyx eval report` 单列 |
| 探针结论是否随 Ollama 升级失效 | 版本相关 | `probe matrix` 记录 `provider_version`；版本变化时 doctor 提示重跑 |
| 视觉/交互 | — | S9/S15 的 e2e 只能验证不报错，布局需你实际用一轮后反馈 |

---

## 实施顺序与工作量（RTX 4060 Ti 单卡的现实排期）

| 里程碑 | 步骤 | 说明 |
|---|---|---|
| M0 | S0 | 装 uv/python3.12/Ollama/Node，拉 2 个模型 |
| M1 计量 | S1→S7 | 骨架与契约 → 存储 → Ollama 适配 → token 阶梯与实测探针 → gateway 装配 → 能力矩阵 → CLI/报表 |
| M2 看板 | S8→S9 | REST+SSE+Fleet → Traces/Playground/Ledger |
| M3 工具 | S10→S12 | 注册表与开销 → 执行器与契约测试 → 循环与 fire-and-verify |
| M4 评测 | S13→S14 | 内核与意图 → 工具调用与调度 |
| M5 对比 | S15 | 矩阵、回归 diff、报告 |
| M6 扩展 | S16 | 插件边界、第二 provider、MCP、导出 sink |

> M1 是唯一有"研究性质"的阶段（S4/S6 的语义实测决定后续所有数字的可信度），其余都是常规工程。
> 建议 M1 完成后先用一周（真实使用），再决定是否值得做 M4 的公开数据集接入——很多情况下你自建的回灌 case 集比 BFCL 更贴合用途。


