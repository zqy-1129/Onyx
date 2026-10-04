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

## S16 — 扩展点固化 + 第二 Provider + MCP（M6）✅

**产出文件**
```
plugins_example/{example_task,example_provider}/      # 独立可 pip -e 安装的小包（放仓库根，理由见 S16a 记录）
onyx/discovery.py                                     # 六个 group 共用的发现语义
onyx/llm/providers/openai_compat.py                   # vLLM / LM Studio / Xinference（验证抽象）
onyx/tools/executors/mcp.py                           # 任意 MCP server 的工具纳入注册表与测试
onyx/store/sinks/otlp.py（或 langfuse.py）             # 事件流导出，验证 Sink 抽象
docs/eval-recipes.md
tests/contract/{test_plugin_discovery,test_provider_contract}.py
scripts/check_extension_boundary.{py,sh}
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

**提交**：分四次，每次一个可独立验证的出口
`feat(plugins): entry-point discovery for all registries, example plugins, boundary gate`（S16a）→
`feat(llm): openai-compatible provider + provider contract suite`（S16b）→
`feat(store): export sink validating the EventSink abstraction`（S16c）→
`feat(tools): mcp executor + tool discovery into the registry`（S16d）

---

### S16a 已交付（扩展点固化 + 样板插件 + 边界门禁）

**实际产出**
```
onyx/discovery.py                                   # 六个 group 共用的发现语义（隔离/可见/覆盖/缓存）
onyx/llm/registry.py                                # 改用共用语义（原来自己实现了一份）
onyx/eval/tasks/__init__.py                         # onyx.tasks 接线：BUILTIN_TASKS + specs()
onyx/eval/task.py                                   # TaskSpec / coerce_task_spec；删掉没人用的 TaskRegistry
onyx/store/sinks/registry.py                        # onyx.sinks 接线 + build_event_sink(name, **opts)
onyx/tools/executors/__init__.py                    # onyx.tool_executors 接线；fixture 通道抢不走
onyx/obs/visitors/__init__.py                       # onyx.observers 接线：插件一律排在内置之后
onyx/tools/spec.py, onyx/tools/registry.py          # ToolKind 之外允许插件 slug（见下"改到的内核"）
onyx/runtime.py, onyx/api/app.py, onyx/cli.py       # event_sinks 装配点 + --sink/--provider 出口 + plugins/eval tasks 命令
plugins_example/example_task/                       # 外部 EvalTask 插件（自带数据集与指标）
plugins_example/example_provider/                   # 外部 LlmProvider 插件（只实现协议）
scripts/check_extension_boundary.py(.sh)            # M6 出口 DoD 的机器断言
tests/contract/{conftest,test_provider_contract,test_plugin_discovery}.py
tests/unit/{test_cli_plugins,test_cli_traces,test_extension_boundary}.py
```
样板包放在**仓库根的 `plugins_example/`** 而不是计划里写的 `onyx/plugins_example/`：
`[tool.hatch.build.targets.wheel] packages = ["onyx"]` 会把后者一起打进 onyx 的 wheel，
而"外部包"必须是**另一个可独立安装的 distribution**，否则 entry point 根本不成立。

**接线之后各注册表的语义（一处实现，六个 group 共用）**
- 坏插件：跳过 + 记进进程级台账（`failures()`，重复尝试累加 attempts），内置实现不受影响；
- 失败可见：`onyx plugins` / `onyx eval tasks` 会打印台账并以退出码 1 结束，`onyx doctor` 多一条体检项；
- 同名覆盖：插件覆盖内置，覆盖关系在 `onyx plugins` 里标 `↻内置`，不悄悄发生；
- 例外：`kind="fixture"` 永远走桩实现——评测的零副作用保证不能被外部实现劫走；
- 缓存：`entry_points()` 单次约 6ms 且注册表在热路径上被反复问，所以按 group 缓存原始声明、
  按 `(group, name)` 缓存已加载值；**失败不缓存**，所以坏插件的 attempts 会持续增长（保持可见）。

**改到的内核与为什么**（这些都不是"为某个实现开小灶"，而是把契约补全）
1. `LlmProvider.generate` 的协议里**没有 `trace_id`**，而 gateway 一直按
   `generate(req, trace_id=..., on_event=...)` 调用 → 照协议写的外部 provider 必然 TypeError。
   协议补上 `trace_id: str = ""`，并由 `test_generate_signature_matches_how_gateway_calls_it` 钉住。
   同时删掉从未被任何调用点使用的 `StreamingProvider`：它的文档声称"不支持时 gateway 退化为
   一次性返回"，而真实机制是 `req.stream` —— 留在契约里就是一条会被照抄的假话。
2. `ToolKind` 是 closed enum，插件种类（`"shout"` / 未来的 `"mcp_xxx"`）**无法被表示**，
   `onyx.tool_executors` 就只剩"覆盖内建"一种用法。改法：`ToolDef.kind: ToolKind | str`，
   构造时用 `coerce_kind` 归一（内建仍是枚举成员，口径不变），外部值限制成小写 slug；
   **导入边界**（`defs_from_payload`）仍然拒绝不认识的种类——拼错的 kind 必须当场炸，
   而不是躺进库里等某次调用才发现。`_PENDING` 的 `mcp`/`ollama_builtin` 保持可导入：先定义后实现是允许的。
3. CLI `_runtime` 之前把非 ollama 的 provider 也记成 `provider_id="ollama-local"`，
   于是 `--provider mock|echo` 跑出来的 trace 在库里自称来自 ollama —— 这是"数字来自哪个引擎"的假信息。
   现在按 kind 生成 `echo-local` / `mock-local`，ollama 保持原 id 以兼容既有标定数据。

**顺带修掉的既有缺陷**（都是"测试没覆盖到才活到现在"的那一类）
- `onyx/cli.py` 里有**两份同名 `_fmt`**：S3 的那份第二参数是 `suffix`，S13 加的那份是 `digits`，
  后定义者把前者遮蔽 → `onyx traces show` 在**任何 ttft 为数字的 trace** 上直接
  `ValueError: Format specifier missing precision`（值为 None 时提前返回「—」，所以 mock-only 测试看不见；
  ruff 的 F811 因"前一份被使用过"也不报）。合并成一份 `_fmt(value, digits=3, suffix="")`，
  补 `tests/unit/test_cli_traces.py`：行为回归 + **顶层不许有同名定义**的结构断言。
- `onyx eval tasks` 这个计划里点名的自测命令**此前不存在**（只有 `eval ls` 顺带列一行）。
- `onyx plugins` 表格里任务 id 被 rich 截断成 `example_char_co…`，`| grep example` 会查不到 →
  id 列改成 `overflow="fold"`。
- GPU 锁接管在 Windows 上会被"文件正被别的句柄打开"短暂拒绝：一次失败就放弃等于
  **没人能接管死锁**（评测无限排队等一个已死的持有者）。改成有界重试（3 次），
  并补 `test_transient_rename_denial_still_allows_takeover`。"实在不行就覆盖"依然禁止。

**自测（真机，全部实际执行）**
```bash
uv run pytest                              # 852 passed, 1 skipped（含 tests/contract 57 项）
uv run ruff check .                         # All checks passed
uv run lint-imports                         # 3 contracts kept
uv run python scripts/check_extension_boundary.py --files $(git status --porcelain ...)
                                            # ✓ 接入 1 个实现未触碰受保护的内核文件
uv run --with-editable plugins_example/example_task onyx eval tasks
                                            # example_char_count | 数汉字（扩展点样板）| 插件 | chat | 3 | 自带
ONYX_DATA_DIR=/tmp/... uv run --with-editable plugins_example/example_task \
  onyx eval run --task example_char_count --provider mock --model mock/echo
                                            # 8/8 · accuracy 0.000 · format_valid_rate 0.000 · in 960 / out 192 · 78ms
uv run --with-editable plugins_example/example_provider \
  onyx chat "测试出处" --provider echo --model echo/static
                                            # provider_id=echo-local · usage=heuristic/low(8/6)
                                            # 异常=LOW_CONFIDENCE_USAGE, NO_ENGINE_COUNT · 无引擎计数的项显示「—」而不是 0
```
外部任务跑通一次评测时 `onyx/eval/runner.py` 未被修改；外部 provider 被完整记录时
`onyx/core/**`、`onyx/llm/gateway.py`、`onyx/obs/**` 未被修改 —— 由脚本判定，不靠人 review。

**已知小缺口（记在这里，不藏）**
- ~~`onyx.serve --provider X` 还没有对应 flag~~ → S16b 已补；
- 非 ollama provider 的 `base_url` 仍记录 CLI `--url` 的默认值（`http://127.0.0.1:11434`），
  修它需要把散在 6 处的 url 默认值提成常量并区分"用户没填"，属于独立一次改动；
- ~~`onyx.providers add/list`（计划自测里提到的命令）尚未存在~~ → 仍未做，见 S16b 的"遗留"。

---

### S16b 已交付（第二个 provider：openai-compat）

**实际产出**
```
onyx/llm/providers/openai_compat.py     # 只实现 LlmProvider：/v1/models + /v1/chat/completions
onyx/llm/registry.py                    # BUILTIN 增加 openai-compat
onyx/llm/streaming.py                   # openai 分支同时吃 delta（流式）与 message（非流式）
onyx/core/types.py                      # SOURCE_PRIORITY 加入 COMPAT；删掉零消费者的 CROSSCHECK_SOURCES
onyx/llm/caps.py                        # 空 capabilities 清单 ⇒ unknown，不是 missing
onyx/llm/measurement/{reconciler,fidelity}.py
onyx/api/{schemas.py,routes/fleet.py,app.py}   # loaded / size_gb / loaded_known 的"未知"
onyx/cli.py                             # models ls 状态三态；serve --provider/--sink；chat --provider
onyx/web/src/{api/types.ts,format.ts,pages/Models.tsx,pages/Fleet.tsx}
tests/contract/test_provider_contract.py（4 个实现跑同一套断言）+ tests/unit/test_provider_openai_compat.py
docs/PROBES.md P23/P24
```
**三次提交的顺序是有意的**：需要改内核的口径修正**先落地**（`fix(measurement)`），
再接入实现（`feat(llm)`），最后才是实现暴露出来的显示问题（`fix(web,api)`）。
反过来的话 `check_extension_boundary.py` 会把"改内核"与"加实现"混在一次提交里判失败，
而那条门禁从此就会被人对付性绕过——一次性的绕过比没有门禁更糟。

**抽象泄漏的判定与处置**（计划里那条"记 issue + 补契约，不许在 gateway 加 if"）
- 接入过程中确实需要动 `core/types.py`（采信阶梯）与 `llm/caps.py`（三态推断）。
  两处都不是"为 openai_compat 开小灶"，而是**原口径在第二个通道上不成立**：
  前者把 compat 排除在采信之外（真机对照：compat 400 与 finish=length 自洽，
  heuristic 估成 602）；后者把"不上报"读成"不支持"（看板对正在服务 chat 的通道显示 ✗）。
  所以按 §13 的规矩处理：改契约 + 写进 PROBES（P23/P24），`gateway.py` 一行未动。
- **没有**按 `base_url`/服务器名字猜行为。专有参数走 `req.extra`，
  thinking 键位走显式声明的 `thinking_via`，不支持的输入（图片、`keep_alive`、
  无法表达的 `thinking`）一律 `CapabilityMissing` 并给出修法。

**真机自测（全部实际执行，零外部网络）**
```bash
uv run pytest                     # 899 passed, 1 skipped
uv run pytest -m live              # 20 passed
uv run ruff check . && uv run lint-imports    # clean / 3 contracts kept
uv run python scripts/check_extension_boundary.py --staged
                                  # ✓ 接入 1 个实现未触碰受保护的内核文件
onyx chat --provider openai-compat --url http://127.0.0.1:11434/v1 -m qwen3.5:9b
                                  # provider_id=openai-compat-local · usage=compat/low in=16 out=400
                                  # drift 6.25%（对照 heuristic）· 吞吐/TTFT 显示「—」而不是 0
onyx eval run --task intent_classification --provider openai-compat … --limit 3
                                  # 未声明 thinking_via ⇒ 逐条 CapabilityMissing + 修法
                                  # （主分数全部「—」而不是 0，run 记录仍可下钻）
onyx serve --provider openai-compat --url … --port 8791  +  vite dev
                                  # /api/fleet loaded_known=false · installed_models=3
                                  # /api/models loaded=null, size_gb=null, caps.missing=[]
                                  # 模型页：参数/量化/磁盘/驻留全「—」，能力位是 ? 而不是 ✗
                                  # 控制台 0 条错误
```
**遗留（S16 内继续处理）**：`onyx providers add/list` 仍未做（多 provider 并存时的登记入口）；
兼容通道的探针套件（`onyx probe`）目前大量依赖 ollama 原生端点，
因此 compat 通道上 structured_output/stream_usage 会长期停在「未实测」——这是事实而不是缺陷，
但要在 `docs/eval-recipes.md` 里写清楚，否则用户会以为是 bug。

---

### S16c 已交付（OTLP 导出 sink，验证 Sink 抽象）

**实际产出**
```
onyx/store/sinks/otlp.py                     # EventSink 的第二个真实实现：OTLP/HTTP JSON
onyx/store/sinks/registry.py                 # _LAZY_SINKS（otlp 需要 httpx ⇒ 惰性）+ builtin_sink_names()
onyx/llm/providers/…、pyproject.toml          # import-linter 契约登记例外（见下）
tests/contract/test_sink_contract.py          # 4 个实现（jsonl/null/otlp/插件）同一套断言
tests/unit/test_sink_otlp.py                  # OTLP JSON 映射细节
```
**三条刻意的取舍**（都写在模块 docstring 里，不藏在代码里）
- 编码用 **OTLP/HTTP JSON** 而不是 protobuf：不新增依赖，代价是不能声称"任何 collector 都能吃"。
  所以把 `onyx.encoding=json` 导成资源属性，接收端看一眼就知道面对的是什么。
- **一个 trace 只导结构 span**（根 span + 每次工具执行一个子 span），其余事件折成计数属性
  `onyx.events={"text_delta": 2, …}`。逐 token 变成逐 span 会把 collector 打满，
  而且那等于在 collector 里重建第二个 Onyx。
- **默认不导工具参数值**，只导键名与数量（`onyx.tool.arg_keys` / `arg_count`）；
  参数里常有地址、身份、内部 ID，把它们原样发到外部可观测栈不是"导出 trace"而是数据出境。
  确实需要时显式 `include_args=True`。

**契约测试挖出来的实现问题**（都有断言钉住）
- `int64` 在 OTLP JSON 里必须是**字符串**，发数字会被多数 collector 拒收；
- dict 属性必须 `json.dumps`，不能用 `str(dict)`——Python repr 是单引号，
  对面 `loads` 失败而我们这边"看起来导出成功了"；
- OTLP 没有嵌套 span：中间形状里的 `children` 必须摊平成同级 span 才能外发；
- id 宽度：onyx 的 26 字符 ULID 不是 W3C 的 128-bit trace id ⇒ **派生**（sha256）而不是截断，
  截断会让两条无关调用链有概率画成同一条；同时把 `onyx.trace_id` 留在属性里保证可回跳；
- 时间只从 `wall_iso` 换算，**不用 `ts_ns`**（单调钟换台机器就没有意义，
  导出成 1970 年附近的数比直接报错更难发现）；
- `close()` 不许抛：它站在 `finally` 里，抛出会盖掉真正让进程出错的那条异常，
  而数据并没有丢（SQLite 才是权威存储），欠着的 span 仍能在 `stats` 里看到；
- 发送失败时把 **span 原样退回队列**（编码是纯函数，重试不需要伪造数据），
  只"记一笔欠账"的重试第二次就没东西可发，那是把故障伪装成"已经尽力"；
- 队列上限在**入队时**生效（不是等 flush），否则一次长评测会把一整夜的 span 堆在内存里；
  丢最老的但 `dropped` 必须计数（与 sqlite sink 同一纪律）；
- 落在"没有开着的 trace"上的事件计入 `orphans`：它增长说明装配或事件顺序错了，
  而不是网络问题。没有这个计数，"接上了但一条都没导出去"就看不出来
  ——SSE broker 事故的那个形状。

**隔离不许掩盖故障，这条在 sink 上有两处具体体现**
`EventFanout` 每次 flush 记一次错误（不是每个事件），而 sink 自己记累计 `failures` 与
`pending`；契约测试同时断言两个数，防止"看起来只失败了一次"其实一直在失败。

**新增了一条 lint 例外**：`onyx.store.sinks.otlp -> httpx`。
`import-linter` 立刻把"store 层不许碰网络"判为 BROKEN，这是设计意图而不是噪音——
处置方式与 `executors.http` 一致：**惰性导入**（`_LAZY_SINKS`，只有 `--sink otlp` 才 import 到
httpx）+ 在 `pyproject.toml` 里显式登记并写明理由，契约名同步改成
`network IO confined to llm, executors.http and sinks.otlp`。
`onyx.store` 其余部分在零三方依赖下仍必须能 import（有契约守着）。

**自测（全部实际执行）**
```bash
uv run pytest                    # 939 passed, 1 skipped
uv run pytest -m live             # 20 passed
uv run ruff check . && uv run lint-imports   # clean / 3 kept（1 ignored import ×2）
uv run onyx plugins               # onyx.sinks 内建 = jsonl, null, otlp
onyx chat --sink otlp             # 未配端点 ⇒ 直接报错退出，说清该设 OTEL_EXPORTER_OTLP_ENDPOINT
# 端到端（同进程起一个假 collector，收 POST）：
#   exit 0 · 收到 1 次 POST /v1/traces · content-type application/json
#   span 名 "chat mock/echo" · traceId 32hex / spanId 16hex
#   时间 1791089873739949056 → …742945792（2026-10-04，不是 1970）
#   资源属性 service.name=onyx · onyx.encoding=json · onyx.app_version=0.1.0
#   onyx.trace_id 原样保留（可从 collector 跳回 onyx）· provider_id=mock-local
#   onyx.events 折成计数 JSON 字符串（first_token/generation_end/model_load/usage_* 各计数）
```
**顺手发现并如实记录**：`RECONCILED` 事件在契约与 `PAYLOAD_REQUIRED` 里都存在，
`obs/visitors/token.py` 也消费它，但 **gateway 从不发它**（采信结果算完直接落库）。
sink 因此不编造 `onyx.usage.source`，只导出事件流里真出现过的 `onyx.usage.engine.*` 原始报告。
这个"契约里有、生产侧不发"的空洞留给 S16 之后的观测一致性检查处理，记在这里而不是留在代码里。

---

### S16d 已交付（MCP 执行器：发现 → tool_def → contract → fire-verify）

**实际产出**
```
onyx/tools/mcp.py                 # stdio JSON-RPC 2.0 客户端（纯 stdlib）+ 配置 + 发现 + 结果映射
onyx/tools/executors/mcp.py       # ToolKind.MCP 的执行器（走同一条 guarded_call）+ 离线契约样本
onyx/tools/executors/__init__.py  # mcp 从 _PENDING 移到 _LAZY；新增 PENDING_KINDS 导出
onyx/tools/sandbox.py             # 默认 impl_ref 白名单加 "mcp:"（理由写在代码里）
onyx/cli.py                       # tools mcp-ls / mcp-import；contract 的 mcp 列；pending 改为推导
tests/fixtures/mcp_demo_server.py # 最小但真实的 stdio MCP server（故意带三种坏毛病）
tests/unit/test_tools_mcp.py      # 协议边界 42 项（假传输）
tests/unit/test_tools_mcp_stdio.py # 真子进程/真管道 7 项
```
**为什么手写客户端而不装 `mcp` SDK**：`onyx.tools` 的可移植性锚点是"除 executors.http 外不依赖
三方网络库"，而 stdio 传输本质就是"往子进程 stdin 写一行 JSON、从 stdout 读一行 JSON"。
用 stdlib 换来两件事：`import-linter` 的网络契约继续成立（MCP 不经 httpx，3 条契约零改动全绿），
以及测试能离线穷举协议边界。**但因此必须自己承担帧的正确性**——见下面真机挖出的两条。

**闸门放在哪里（安全设计，三条）**
1. server 的**命令只来自配置文件**（`$ONYX_MCP_CONFIG` 或 `<data_dir>/mcp.json`），
   参数永远只能进 `tools/call` 的 `arguments`；`impl_ref` 还额外过 `check_impl_ref`。
   与"不提供通用 fetch 工具"是同一条理由：模型能决定跑什么命令 = RCE。
2. **副作用取保守默认**：`annotations.readOnlyHint` 是服务器自报的提示，
   没标注一律 `write` ⇒ 默认策略直接拒绝执行。`mcp-import` 会把这批工具单独警告出来。
3. **工具输出里的非文本块不内联**：image 只记 `mime` 与字节数。
   把 base64 塞进上下文等于让一次工具调用吃掉几千 token，而上下文开销正是被测对象。
   另外**默认不导出参数值**（见 S16c 的同一条理由）。

**真机才暴露出来的两个 bug（都有断言）**
- **子进程 stdout 不是 UTF-8**：中文 Windows 上 Python 子进程按控制台代码页（GBK）输出，
  `text=True` 的 `readline()` 抛 UnicodeDecodeError，**读线程当场死掉**，
  父进程此后只能等到超时——现象是"卡住"而不是"编码错了"。
  改成二进制管道 + `decode_line()`（坏字节替换、解不出 JSON 当非协议行跳过），
  并给读线程加兜底：它一死整个会话就挂死，比报错危险得多。
- **stderr 排空线程在 close() 后抛"I/O operation on closed file"**：留下
  "未处理的线程异常"告警。一次正常的关闭不该长得像故障，否则人就学会忽略这类告警了。
- 顺带一条测试装置自身的坑：假传输的**回答 id 与请求 id 必须对齐**，
  错开时"握手响应"会被当成 `tools/call` 的回答，测试就绿在一次根本没发生的调用上。

**契约矩阵现在有四列**（`python_fn / mock / http / mcp`，mcp 用离线假连接，8 条断言全过）。
`pending` 一列改为从 `PENDING_KINDS` 推导并加断言：写死在 CLI 里的版本会在实现完成后
继续宣称"未实现"，而那句话看起来永远合理。

**自测（全部实际执行）**
```bash
uv run pytest            # 990 passed, 1 skipped（mcp 单元 42 + 真子进程 7 + 契约矩阵断言）
uv run pytest -m live     # 见下（与 fire-verify 一起跑）
uv run ruff check . && uv run lint-imports   # clean / 3 kept，零新增例外
uv run python scripts/check_extension_boundary.py --staged   # ✓ 未触碰受保护内核文件

onyx tools mcp-ls / mcp-import        # 真子进程发现 5 个工具；1 个按 write 登记并被警告出来
onyx tools contract                   # mcp 列 通过 8 · 失败 0 · 不适用 0
onyx tools fire --tools demo__weather --mock fixture   # 零真实调用：模型选对工具、参数缺 text ⇒ BAD_ARGS
onyx tools fire --tools demo__weather --mock live --expect-args '{"city":"北京"}'
                                      # 判定 PASS · 真实执行：模型 → loop → stdio server → 回填 → final
```
`--mock live` 那一条是 M6 想要的形状：**换一种执行器不需要改循环、不需要改评测、
不需要改看板**，contract 矩阵与 fire 六种判定照样把它测得下来。

---

## S17 — 数据生命周期：`onyx rotate` + 覆盖率门禁（M7 第一步）✅

**产出文件**
```
onyx/store/retention.py                    # parse_window / sweep / history / disk_report
onyx/store/migrations/0006_retention.sql   # retention_run：保留策略自己的留痕表
onyx/core/content.py                       # BlobStore 维护面：delete / iter_refs / total_bytes
onyx/cli.py                                # onyx rotate（默认 dry-run）+ db info 体积现状
pyproject.toml / Makefile                  # 第五道门禁：coverage fail_under=80（branch）
tests/unit/test_retention.py（22）+ test_content.py（+5）
```

**设计取舍（都是"删错了回不来"逼出来的）**
- **分数永久、证据有限期**：摘的是 `raw_request_ref / raw_response_ref / rendered_prompt_ref /
  tool_call.result_ref / tool_run.output_ref` 五列重 payload；trace 行与 messages/tools/output 留着，
  下钻与 `traces replay` 不受影响。删行只在 `--purge-traces` 时发生，且被
  `grade.trace_id / tool_run.trace_id / trace.eval_run_id` 引用的行永不删除。
- **dry-run 与 apply 共用一段代码**：dry-run 真的执行删除，只在事务末尾回滚，文件一个都不碰。
  估算与删除各写一套，迟早会对不上——而"报得出会删多少"是这个命令的全部价值。
- **事实与估算分列**：审计表的 `refs_cleared / traces_deleted / blobs_deleted` 写**事实**
  （dry-run 与被拦下时都是 0），估算只活在 `per_rule` 与 `detail_json`。混在一起等于让审计表说谎。
- **回收上限**：单次回收超过现有 blob 体积 60% 直接拦住（退出码非 0，且真的什么都没动），
  要人明确加 `--force`。文件删除严格排在事务提交之后——反过来的顺序会留下
  "引用还在、文件已没了"的证据空洞。
- **字节口径统一**：`stat().size`、`delete()` 返回值、`total_bytes()` 都只算内容，不含 media 侧车与
  `.tmp`。曾经"释放了多少"与"盘上小了多少"两个数对不上，而它们都自称 blob 体积。
- `iter_refs()` 只承认 `sha256:<64 hex>` 的文件名：目录里被人放一个 README 不该变成"一个可回收的 blob"。

**覆盖率门禁（本步同时落地）**：`coverage run -m pytest`（离线套件，branch 覆盖）+ `fail_under=80`，
实测基线 **88%**（1018 项离线用例；最低的是 `probe/checks.py` 17%，那些只在 `-m live/probe` 里跑）。
留 8 个点余量：贴线的门禁会在下一次正常改动时被绕过。

**自测（全部实际执行）**
```bash
uv run pytest                         # 1018 passed, 1 skipped
uv run ruff check . && uv run lint-imports     # clean / 3 kept
uv run coverage run -m pytest -q && uv run coverage report    # 88% ≥ 80%
uv run python scripts/check_extension_boundary.py --staged    # ✓ 未触碰受保护内核

uv run onyx rotate                    # 真机 .data：dry-run，报出 6 个无人引用的 blob / 955 B
uv run onyx rotate --json             # 机器可读，dry_run=true、refs_cleared=0
uv run onyx db info                   # .data 合计 + oldest trace + dangling 0 + rotate 留痕一行
```
`--apply` 端到端在 `tests/unit/test_retention.py` 里用**真实 FileBlobStore（临时目录 + 真文件）**验的，
没有拿仓库的 `.data` 做删除实验。

**顺带查出来的一个真实泄漏**：`TOOL_EXEC_START` 事件带 `args_ref`（loop 写了 blob），
但 schema 里没有任何列存它（`tool_call` 存的是内联 `args_json`）⇒ 每次工具循环泄漏一个小 blob。
真机 dry-run 报出的 6 个孤儿就是它。保留策略能把它们回收掉，但**正确的修法是别再写或者把它落库**，
记在这里，别让它在"rotate 能删孤儿"的表象下变成永久行为。

**验收 DoD**：M7 出口判据的 rotate 一半（可 dry-run、落库审计、报得出释放多少字节）。

**提交**：`feat(store): 数据生命周期 onyx rotate —— 分数永久证据有限期 + 覆盖率门禁 80%`

---

## S18 — 可验证备份 + doctor 具名报告（M7 第二步）✅

**产出文件**
```
onyx/store/backup.py            # create_backup / verify_backup / manifest
onyx/store/db.py                # Database.backup_to —— 在线备份 API，不是 cp
onyx/cli.py                     # onyx db backup --to / db verify-backup [+ --json]
                                # doctor 的 blob 项改用统一引用清单并指名道姓
tests/unit/test_backup.py（18）
```

**为什么不是 `cp .data/onyx.sqlite`**
- **WAL 下直接拷主文件会安静地少一段**：最近的事务可能还在 `-wal` 里，拷出来的库
  打开不报错、行数还"看着合理"。必须用 `Connection.backup()` 取一致快照，
  且整个过程持有单写者锁。测试 `test_backup_contains_rows_still_in_the_wal` 盯的就是这条。
- **备份必须一个文件就能恢复**：源库的头一页写着 WAL，复制出来的目标也是 WAL——
  那等于宣称可恢复却还依赖一个没被拷走的 `-wal`。落地时切回 DELETE 日志模式。
- **只备库不算备过**：证据在 blob 目录里。备份装的是**被引用到的** blob（孤儿不占体积），
  verify 检查"备份库里的每个引用都能在备份里解析"——只备库不备证据的备份就是这样暴露的。
- **备份会在没人看的时候坏**：verify 逐字节重算每个 blob 的 sha256 与文件名比对
  （内容寻址在这里免费提供了一个完整性校验器），并对「ref+大小」集合取摘要，
  于是"备份目录被动过 / 备份没做完"和"少了一个文件"是两条不同的具名检查。
- **备份之后库又长了数据不是错误**：`drift` 单独报"备份点之后当前库多出多少行"，
  而 `manifest.unresolved_refs` 会留下"备份当时就有 N 个引用解析不了"——
  证据在备份之前就丢了，备份修不回它，只能如实带上这件事。
- 目标目录已有备份 ⇒ 拒绝覆盖并退出 1：覆盖一个恢复点通常要等到真需要恢复时才发现。

**顺带补上 S7 承诺的另一半**：`doctor` 的 blob 完整性项以前只看 `raw_response_ref`，
现在与 `rotate` 共用同一份引用清单（`ALL_REF_COLUMNS`），并把缺的 ref **指名道姓**列出来。
"3/5 可解析"会让人去猜是哪两个没了。

**自测（全部实际执行）**
```bash
uv run pytest                        # 1036 passed, 1 skipped
uv run ruff check . && uv run lint-imports    # clean / 3 kept
uv run onyx db backup --to .tmp/backup-s18    # 真机：1231 个 blob / 469.6 KiB / 17 张表 16619 行
uv run onyx db verify-backup .tmp/backup-s18  # 9 项检查全 ✓，退出码 0
uv run onyx db verify-backup ... --json       # ok=true / checks=8 / drift.current_refs=1231

cp -r .data .tmp/data-copy && rm .tmp/data-copy/blobs/00/20/... # 在副本上人为删一个 blob
ONYX_DATA_DIR=.tmp/data-copy uv run onyx doctor                 # ✗ blob 引用完整 1230/1231
                                                                #   缺：sha256:00…  退出码 1
```
真机那次备份正好验证了 S17 报出的 6 个孤儿：盘上 1237 个文件，备份只装 1231 个被引用的。

**验收 DoD**：M7 出口判据的备份一半（备份可验证恢复；人为破坏后 doctor 报出具体 ref，不是笼统 500）。

**提交**：`feat(store): 可验证备份 db backup/verify-backup —— WAL 一致快照 + blob 逐字节校验`

---

## S19 — doctor 补齐两项 + 迁移前自动快照 + `.data` 曲线（M7 收口）✅

**产出文件**
```
onyx/cli.py                     # doctor: 磁盘余量 / token 计量档位；db sizes（现状+趋势）
onyx/store/db.py                # BACKUP_DIR_NAME + _snapshot_before_migration（升级前先复制）
onyx/store/retention.py         # Trend + footprint_trend（日均增速）
onyx/llm/measurement/fidelity.py # FITTED_MIN_SAMPLES：门槛只承认这一处
onyx/llm/measurement/calibrate.py # usable 改用同一个门槛
tests/unit/test_doctor.py（9，S7 承诺的文件名）+ test_retention.py（+7）+ test_db_migrations.py（+3）
```

**这步挖出来的一件事最重要：`FITTED_MIN_SAMPLES` 以前散在 4 处字面量 30**
（`FittedCounter`、`text_counter`、`Calibration.usable`、CLI 的档位标签）。
门槛一旦分叉，最坏的组合就是体检绿着而计数其实早已退回 heuristic/low。
现在四处共用一个常量，并由 `test_tier_check_boundary_is_the_same_number_the_counter_uses`
把"体检与计数用同一个门槛"钉成断言。

**两条新检查的立场**
- **磁盘余量**：低于 2 GiB 就红。`.data` 写满的表现不是优雅报错，而是崩溃 + 半截 blob。
  修法必须具体（先 `rotate --apply` 回收，还是把 `ONYX_DATA_DIR` 挪盘），余量问不出来时
  写"问不出来"而不是当作充足。
- **token 计量档位**：报的是"**这台机器上每个模型落到哪一档**"，不是"设计稿上有几档"。
  所以它明说 `hf_tokenizer` / `gguf_vocab` 本版本没有实现、`tokens` extra 装了也不生效
  （P9：3 个模型里 2 个没有 chat template，T2 的 GGUF 自建 BPE 没做）。
  不写这句话，看板就会一直显得比实际更能精确复算——同一个数字谎的两种写法而已。

**迁移前自动快照**：真要改 schema 之前先 `backup_to(backups/pre-migration-v{旧}.sqlite)`。
只备库不备 blob（那是回滚点，不是完整备份——完整备份是 `onyx db backup`）。
同名已存在就不再复制，空库与 `:memory:` 直接跳过。

**`.data` 曲线（`onyx db sizes`）**：采样点只来自 `retention_run`。
两个点相隔 72 秒也能算出"每天多少字节"，但那是噪声除以时间——所以跨度不足 1 天时
`footprint_trend` 返回 `per_day_bytes=None`，CLI 说"问不出来 · 最近两个点只跨 1 分钟"。
日均很小时也别报"还能写约 1.4 亿天"：超过十年就只说"余量还很充裕"。

**自测（全部实际执行）**
```bash
uv run pytest                     # 1057 passed, 1 skipped（doctor 9 + 曲线 7 + 快照 3）
uv run pytest -m live             # 20 passed
uv run ruff check . && uv run lint-imports    # clean / 3 kept
uv run coverage report            # 88%（retention.py 与 backup.py 均 100%）
uv run onyx doctor --skip-network # 8 项：新增 磁盘余量 416.2 GiB / token 计量档位（明说 T1/T2 未实现）
uv run onyx db sizes              # 现在 6.5 MiB · 趋势问不出来（两个点只跨 1 分钟）
```
真机验证自动快照：把 `.data/onyx.sqlite` 拷到 `.tmp/mig/`，加一个 `0007_note.sql` 后升级 ⇒
`backups/pre-migration-v6.sqlite` 生成，里面 version=6、trace 1345 行完好（没拿真库做实验）。

**验收 DoD**：M7 出口判据全部达成——保留策略可 dry-run 且落库审计、备份可验证恢复、
`doctor` 在人为破坏后报具体项（blob 具名 + 磁盘 + 档位）、`.data` 体积有曲线可查。

**提交**：`feat(cli): doctor 补磁盘与计量档位 + 迁移前自动快照 + db sizes 曲线（S19）`

---

## S20 — 部署配置 `onyx.toml`（M8 第一步）✅

**产出文件**
```
onyx/config.py                     # Config + schema + load_config / config_path / pick / effective
onyx/settings.py                   # data_dir 三层优先级；project_root 迁到 config（一份实现）
onyx/example 模板 → onyx.example.toml # 进版本库的模板；onyx.toml 本身进 .gitignore
onyx/cli.py                        # 根回调 --config + 验证；每条命令的 flag 默认改 None
onyx/api/{app,deps}.py             # gpu_stale_after_s 注入，不在装配层读全局配置
tests/conftest.py                  # _isolated_config：测试永不读开发者本地的 onyx.toml
tests/unit/test_config.py（25）
```

**格式换成了 TOML，理由写进 DESIGN §13**：`tomllib` 是标准库，为一配置文件引入 PyYAML
会把依赖面扩大在最不该扩的地方。承诺（配置外置 + 一条优先级规则）没变。
`[provider].keep_alive` 这一项**刻意没进 schema**：内核里没有任何请求路径消费它
（只有 calibrate 预热时写死一个 10m），"配置里有但代码不读"正是这一步要防的失败。

**优先级只有一条规则，且实现方式很具体**：flag > 环境变量 > 配置文件 > 内建默认。
要让"flag 赢"成立，flag 的内建默认必须一律 `None`——否则 `--url http://127.0.0.1:11434`
这种写法会永远赢过配置文件，配置文件当场变成摆设且不报错。
收口放在 `_runtime()` / `_engine_url()` / `_policy_from_cli()` 三处，
而不是 38 条命令各写一遍回落。

**这一步真机试出来的 bug（已修 + 已钉住）**：`--config` 指到一个不存在的文件时，
`onyx db info` **照常跑并且退出码 0**——因为 data_dir 从 `ONYX_DATA_DIR` 拿到了值，
`load_settings()` 的短路让 `load_config()` 根本没被执行。坏配置被静默忽略，
比没有配置危险得多：每个键都能从更高优先级拿到，于是没有任何一条命令会去读那份文件，
而人以为 `[serve].port` 生效了。
修法是**在根回调里先验证再动手**（`_CONFIG_TOLERANT = {doctor, config}` 例外——
它们就是用来诊断这份文件的），回归测试是
`test_broken_config_stops_ordinary_commands` + `test_doctor_survives_a_broken_config_because_it_is_the_diagnosis`。

**"写了不生效"的机械检出**：`_SCHEMA` 是唯一事实来源，文件里出现 schema 之外的键/节、
或类型不对的键（含 `port = true` —— bool 是 int 的子类，会被当成 1 号端口），
都记进 `unknown` / `problems`，由 `onyx doctor` 的"配置文件"一项指名并让退出码非 0；
`onyx config show` 则把每一项**生效值 + 它来自哪一层**摊开（报告与命令共用 `effective()`，
不允许两套回落逻辑）。

**自测（全部实际执行）**
```bash
uv run pytest                        # 1082 passed, 1 skipped
uv run ruff check . && uv run lint-imports     # clean / 3 kept
uv run coverage report               # 88%（config.py 98%）
uv run pytest -m live                # 20 passed（gateway/serve/eval 的装配路径都改了）

uv run onyx config show              # 无文件时逐项标 [默认]
ONYX_CONFIG=.tmp/onyx.toml uv run onyx rotate --json        # raw_after 取自文件 = "7d"
ONYX_CONFIG=... uv run onyx rotate --raw-after 3d --json    # flag 赢过文件 = "3d"
ONYX_CONFIG=... uv run onyx chat "只回一句话" --provider mock  # 模型来自 [provider].model
ONYX_CONFIG=... uv run onyx doctor --skip-network           # ✗ 配置文件 · 不认识的键：serve.pprt
uv run onyx --config .tmp/nope.toml db info                 # 退出码 2 + 一句人话（以前是 0）
```
真机跑配置验证时用 `.tmp/cfgdata` 当 `ONYX_DATA_DIR`，没往仓库的 `.data` 里写测试数据。

**验收 DoD**：M8 出口判据的配置一半——一份文件能声明 provider / 锁 / 保留 / sandbox，
且"配置项写了但没生效"会被 doctor 报出来。

**提交**：`feat(config): onyx.toml 部署配置 —— 一条优先级规则 + 写了不生效可检出（S20）`

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
| S17 | `onyx rotate && onyx rotate --apply` | 默认不删任何东西；两次数字一致；每次运行有留痕 |
| S18 | `onyx db backup --to D && onyx db verify-backup D` | 9 项检查全绿；人为删一个 blob 后 verify 与 `doctor` 都报出具体 ref |
| S19 | `onyx doctor && onyx db sizes` | 磁盘与计量档位两项可见；升级前自动留 `backups/pre-migration-v*.sqlite` |
| S20 | `onyx config show && onyx doctor` | 每一项标出来自 flag/环境/文件/默认哪一层；未知键与坏类型让 doctor 变红 |
| 门禁 | `make coverage`（`coverage run -m pytest -q`） | 离线套件分支覆盖率 ≥ 80%（基线 88%） |

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


