# PROBES — 实测结论

> **纪律**：没有实测结论的字段一律 `unknown`，UI 显示「—」而不是 0（DESIGN §6.3）。
> 每条结论标注引擎版本、模型、日期，因为这些都随版本漂移。
> 复现：`uv run pytest -m live -q -s`（S3 阶段）；S4 起由 `onyx probe run` 自动写回本文件。

环境：Ollama **0.35.0** · Windows 11 · RTX 4060 Ti **16GB** · 48GB RAM · 实测日期 2026-10-02
已安装模型：`qwen3.8:27b`(27.3B Q4_K_M, 17.7GB) · `gpt-oss:20b`(20.9B MXFP4, 13.8GB) · `qwen3.5:9b`(9.7B Q4_K_M, 6.6GB)

---

## 1. 已确证的事实

### P1 · `/api/tags` 返回的字段比官方文档多 ✅
文档只列了 `name/model/remote_model/remote_host/modified_at/size/digest/details`。
实测**额外**返回：
- `capabilities`：如 `("completion","vision","tools","thinking")` —— 能力位可直接从这里推，不必逐个 `/api/show`
- `details.context_length`：如 `262144` —— 训练上下文
- `details.embedding_length`：如 `4096`
- 本地导入的模型**没有** `remote_model` / `remote_host` 字段（不是空串，是键不存在）

**影响**：`ModelCard` 增加了 `capabilities/context_length/embedding_length` 字段；`model_caps()` 由此推导。
**注意**：`details.context_length` 是**训练**上下文，与 `/api/ps` 的**实际载入**上下文是两个不同的数（见 P2）。

### P2 · `/api/show` 是 POST，GET 返回 405 ✅
`GET /api/show?name=x` → `405 method not allowed`。必须 `POST {"name": x}`。

### P3 · 载入上下文远小于训练上下文 ✅
| 来源 | 值 | 含义 |
|---|---|---|
| `/api/tags` → `details.context_length` | 262144 | 模型**训练**支持的上限 |
| `/api/ps` → `context_length` | **4096** | 本次**实际载入**的上下文（Ollama 默认值） |

**影响（重要）**：`ctx_util = in_tokens / context_length` 必须用 `/api/ps` 的值。
若误用 262144，一个已经吃掉 3000 token（73% 满载、即将截断）的请求会显示成 1.1% 占用——
看板会完全掩盖上下文溢出风险。UI 必须同时显示两个数并标注含义。

### P4 · 返回 tool_calls 时 `done_reason` 可能是 `stop` ✅
实测：`get_weather` 被正确调用、参数完整（`{"city":"北京","unit":"celsius"}`），
但 `done_reason` 报 **`stop`** 而不是 `tool_calls`。

**影响（设计决定）**：`Generation.finish_reason` **原样保留引擎值**，另设派生属性
`Generation.wants_tool_call = bool(tool_calls) or finish_reason is TOOL_CALLS`。
工具循环按派生信号决策。这样既不篡改原始事实（可回放），也不会漏执行工具。
对应实现：`onyx/core/types.py::Generation.wants_tool_call`、`onyx/llm/streaming.py`。

### P5 · 推理模型会把整个 token 预算吃光，正文返回空 ✅
`qwen3.5:9b` 默认开 thinking：
| max_tokens | thinking 字符 | 正文字符 | finish_reason | eval_count |
|---|---|---|---|---|
| 64 | 223 | **0** | `length` | 64 |
| 256 | 835 | **0** | `length` | 256 |
| 512 | 1903 | **0** | `length` | — |
| 256 + `think:false` | 0 | 265 | `stop` | **169** |

**影响（对评测是致命的）**：模型不是"答错"，是"没预算答"。
- 评测/压测的默认参数必须显式设置 `thinking`，否则分数反映的是预算分配而不是能力。
- S5 的 anomaly visitor 需要 `EMPTY_CONTENT_WITH_THINKING` 规则。
- `think:false` 确认**真的生效**（thinking 字符归零，输出 token 从 256 降到 169）。

### P6 · 流式与非流式的输入计数一致 ✅
同一 prompt、`temperature=0` + `seed=42`：`in=19` 两条路径完全相同，输出文本一致。
非流式路径**没有真实 TTFT**（实现里保持 `None`，不用 prompt_eval 时长伪造）。
流式实测 `ttft=125.0ms`、`decode_tps=41.8`。

### P7 · 显存与载入状态 ✅
`/api/ps` 对 `qwen3.5:9b`：`size=5.49GB`、`size_vram=5.49GB`、`offloaded=False`、
`expires_at=2026-10-02T21:54:58+08:00`（keep-alive 倒计时可用）。
注意：磁盘上 6.6GB（`/api/tags.size`）vs 载入 5.49GB（`/api/ps.size`）——**两者不是一回事**，
前者含未载入部分，UI 不能混用。

### P8 · 卸载方式 ✅
`POST /api/generate {"model": name, "keep_alive": 0}` 即可卸载，`/api/ps` 随后不含该模型。
不需要任何未文档化端点。

---

## 2. 未定 / 待测（S4 探针负责回答）

| 编号 | 待答问题 | 为什么关键 |
|---|---|---|
| U1 | warm 缓存下 `prompt_eval_count` 数的是**整个 prompt** 还是**只有未缓存后缀** | 决定输入 token 汇总是否系统性偏低（DESIGN R1） |
| U2 | thinking token 是否计入 `eval_count` | 决定输出成本口径；P5 显示 256 上限被 thinking 占满，强烈提示**是计入的**，但需专门实验 |
| U3 | ~~各模型能否离线复算 token~~ **已测，见 P9** | — |
| U4 | 原生流是否**只在**最后一个 ndjson 事件给计数 | 决定流式 TTFT/计数的采集点 |
| U5 | `/v1` 的 `usage` 与原生 `prompt_eval_count` 是否一致 | T4 交叉验证是否有意义 |
| U6 | `format: json_schema` 是否强制生效、失败是否静默降级 | 结构化抽取评测的前提 |
| U7 | `qwen3.8:27b`(17.7GB > 16GB VRAM) 的 offload 行为与吞吐落差 | 验证 `offloaded` 判定与"必须分开聚合"的规则（DESIGN R5） |

---

## 3. P9 · tokenizer 元数据与 chat template 的可用性（决定 S4 实现顺序）✅

逐模型实测 `/api/show`：

| 模型 | `template` | `tokenizer.chat_template` | `tokenizer.ggml.tokens` | `.merges` | family |
|---|---|---|---|---|---|
| `qwen3.8:27b` | `{{ .Prompt }}`（13 字符，占位） | **无** | ✅ | ✅ | gpt2 |
| `gpt-oss:20b` | **7216 字符真模板**（`<|start|>system<|message|>…`） | 无 | ✅ | ✅ | gpt2 |
| `qwen3.5:9b` | `{{ .Prompt }}`（13 字符，占位） | **无** | ✅ | ✅ | gpt2 |

### 结论与对 S4 的影响
1. **T1 档（HF tokenizer + minja 渲染）不能作为主路径**：3 个模型里 2 个根本没有 chat template
   （`template` 只是 Go 占位符 `{{ .Prompt }}`，`tokenizer.chat_template` 键不存在）。
   没有模板就无法复现引擎真正喂进去的字符串。
2. **T2 档（用 GGUF 的 tokens+merges 自建 BPE）对 3 个模型全部可行** ⇒ 应把它提前为 S4 的主实现，
   而不是 DESIGN 原计划的"M2 后选做"。离线、无需网络、无需 HF 仓库对应关系。
3. **归因方式随之改变**（重要的设计修正）：不要求"渲染出完整 prompt 再数"，而是
   `Σ(各分段真实 token 数) + template_ctl 残差`，其中
   `template_ctl = engine.prompt_eval_count − Σ分段`。
   残差是**可测量且诚实**的：它正是模板控制符（角色标记、生成提示符、工具 schema 包装）的成本。
   有真模板时（如 `gpt-oss:20b`）残差应趋近 0，这本身就是一个校验信号。
4. **工具定义开销（DESIGN §6.2 的核心指标）在 qwen 系上只能给出"分段和 + 残差"口径**，
   UI 必须标明该模型的归因是 `medium` 置信度而非 `high`，不能显示成精确值。

> 这条发现推翻了原计划的一个假设，也正好验证了"保真阶梯 + 置信度标注"这个设计的必要性：
> 如果没有它，看板会对 2/3 的模型显示一个看起来精确、实际无从复算的数字。

