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

## 2. 探针运行结论（S4，`uv run pytest -m probe -q -s`）

环境：Ollama 0.35.0 · `qwen3.5:9b` · 2026-10-02

### P10 · 缓存命中时 `prompt_eval_count` 数的是**整个 prompt** ✅（U1 已答）
同一 644-token 长 prompt 连发 3 次（`keep_alive=5m` 保持载入）：

| 次数 | `prompt_eval_count` | `prompt_eval_duration` |
|---|---|---|
| 1（冷） | **644** | 384.5 ms |
| 2 | **644** | 83.4 ms |
| 3 | **644** | 82.8 ms |

计数三次完全相同，耗时降 **4.65×**。结论：**输入 token 汇总不会系统性偏低**（DESIGN R1 解除）。

### P11 · ⚠️ 缓存命中会让 `prefill_tps` 虚高 4.65 倍（新发现，高危）
由 P10 直接推出：
- 冷：`644 / 0.3845s` = **1675 t/s**
- 热：`644 / 0.0834s` = **7722 t/s**

`prefill_tps = in_tokens / prompt_eval_duration` 这个公式在缓存命中时算出的是
"**等效吞吐**"而不是"**真实计算吞吐**"。两者混进同一个 P50 会得到一个
既不代表冷启动、也不代表稳态的数字，而且**看起来完全合理**——这是最难发现的一类错误。

**处置**（已进 DESIGN §6.4 的实现要求）：
1. `usage` 表新增派生列 `prefill_mode ∈ {cold, warm}`，由 `prompt_eval_ms / in_tokens` 的比值判定；
2. 吞吐类指标一律按 cold/warm 分列聚合，UI 不给合并视图；
3. 冷启动判据从"load_duration > 50ms"扩展为"load_duration > 50ms **或** prompt 未命中缓存"。

### P12 · thinking token **计入** `eval_count` ✅（U2 已答）
同一问题、`max_tokens=256`：

| 设置 | `eval_count` | thinking 字符 | 正文字符 |
|---|---|---|---|
| `think:true` | **256**（撞上限） | 663 | **0** |
| `think:false` | **5** | 0 | 4 |

比值 51.2×。结论：**"输出 token"实际是"正文 + 推理"**。
成本与吞吐口径必须写明这一点，否则同一模型开/关推理的 token 数差 50 倍而无法解释。
另：`think:true` 时正文为空（预算被推理吃光，见 P5），`think:false` 确实生效。

### P13 · 流式与非流式计数完全一致 ✅（U4 已答）
同一 prompt：`in=20, out=42` 两条路径逐位相同；流式末事件确实携带计数（`saw_done=True`）；
流式 `ttft=125.0ms`，非流式 `ttft=None`（实现里刻意不伪造）。

### P14 · `/v1` 兼容层的 usage 与原生**不一致** ✅（U5 已答）
| | 原生 `/api/chat` | `/v1/chat/completions` | 差 |
|---|---|---|---|
| 输入 token | 22 | 20 | **−2** |
| 输出 token | 48 | 64 | +16 |

**结论：T4（compat 档）只能记录、永不采信**（DESIGN §6.1 的决定被实测支持）。
输入差 −2 说明两条通道的 chat template 包装不完全相同；输出差 +16 有一个已知混淆因素：
本次 `/v1` 调用未传 `think:false`，模型可能进入推理并撞上 `max_tokens=64`。
即便如此，**输入侧的 −2 与"两通道模板不同"这一结论成立**。

### P15 · `format: json_schema` 真的强制生效 ✅（U6 已答）
给定含 `required` 与 `enum` 的 schema，输出为
`{"city":"北京","unit":"celsius","temperature":21}` —— 合法 JSON、字段齐、枚举正确。
结构化抽取评测可以直接依赖它，但**仍须记录 JSON 合法率**（换模型就不一定）。

### P16 · 工具调用走原生 tool_calls 头 ✅
`get_weather` 被正确调用，`arguments={"city":"北京"}` 直接是结构化字段（非正文里的 XML/JSON），
`parse_status=ok`。同时**再次确认 P4**：`done_reason` 报 `stop` 而非 `tool_calls`，
`wants_tool_call` 派生信号为 True。工具定义使该请求 `in` 从 ~20 涨到 **280** token
（≈260 token 的工具 schema 开销，这正是 DESIGN §6.2 要量化的东西）。

### P17 · 工具定义的上下文成本，大部分是**模板脚手架**而不是 JSON 本身 ⚠️
真机跑通 gateway 后实测（`qwen3.5:9b`，一个 `get_weather` 工具）：

| 项 | token |
|---|---|
| 引擎计数 `prompt_eval_count` | **301** |
| 用户消息 `msg:0` | 12 |
| 工具 JSON 文本 `tool_defs` | 78 |
| **残差 `template_ctl`** | **213** |

即：一个只有 78 token JSON 的工具，实际吃掉约 **291 token** 上下文，
其中 **73% 是模板注入的说明性脚手架**（"You have access to the following functions…" 之类）。

**影响（会直接误导优化决策）**：如果看板只报 `tool_defs=78`，
用户会以为"精简工具描述能省下大头"，而实际上真正的大头是模板固定开销——
**换模型/换模板比删描述有效得多**。所以 UI 必须把工具成本报成
`tool_defs + 该请求的 template_ctl`，并注明后者是"每请求固定成本"。

**同时暴露的口径要求**：`template_ctl` 残差是用当前档位的 count_fn 算出来的，
本例是 heuristic（该模型没有可用 chat template，见 P9），所以 78/213 这个**拆分**是低置信的，
但 301 这个**总量**来自引擎、是高置信的。UI 必须分别标注，不能整体打一个置信度。

### P18 · 短 prompt 上的 drift 百分比是噪声，报警需要绝对差门槛 ✅
同一次真机运行：19 token 的请求，启发式估 16 → drift 15.8% 超阈值报警。
但 3 个 token 的差毫无意义。**报警条件改为「相对偏差 > 10% 且绝对差 ≥ 24 token」**，
偏差值本身仍然记录（`usage.drift_pct`），只是不触发告警——否则告警疲劳会淹掉真正的口径分裂。

另：19 token 时 `prefill_ms_per_token = 6.98`（≈143 t/s），远低于 644 token 时的 0.60 ms/token（1675 t/s）。
**极短 prompt 的 prefill 吞吐没有意义**（固定开销占主导），UI 在 in_tokens 过小时应显示「—」而不是一个看起来很慢的数字。

### P19 · 标定必须按字符语系双特征拟合，且 `max_rel_error` 要进可用门槛 ✅
`onyx calibrate --model qwen3.5:9b --n 30` 两轮对比：

| 拟合方式 | 参数 | R² | 最大相对误差 |
|---|---|---|---|
| 单比值 + 截距（混合语料按长度截断） | ratio=0.253, intercept=26.19 | 0.9388 | **66.0%** |
| **双特征（中文/其他分开）** | cjk=0.679, other=0.178, intercept=11.00 | **0.9999** | **2.11%** |

两个独立的方法论错误：
1. **中文与拉丁字符的 token 密度差 3.8 倍**（0.679 vs 0.178），单比值把它们混成一个数；
2. **采样方式污染了拟合**：把中文段与英文段分开排布再按长度截断，会让短样本几乎纯中文、
   长样本含大量英文 —— 拟合出的"比值"反映的是采样方式而不是模型属性。
   修正为中英文交替拼接，任何长度的前缀都保持相同语系比例。

**门槛修正**：`usable` 原为 `n≥30 且 R²≥0.90`。第一轮标定的 R²=0.94 能通过，
但最大相对误差 66% —— 这种标定**有害**，因为它看起来可信。现改为
`n≥30 且 R²≥0.90 且 max_rel_error≤20%`。

### P20 · 归因残差与标定截距互相验证 ✅（强一致性证据）
两条完全独立的测量路径给出同一个数：

| 路径 | 方法 | 结果 |
|---|---|---|
| 标定回归 | 30 个不同长度 prompt 的最小二乘**截距** | **10.9958** token/请求 |
| 单次归因 | 一条 19-token 请求的 `engine_in − Σ分段` **残差** | **11** token |

这说明：① 双特征标定的截距确实对应"每请求固定的模板控制符开销"；
② 分段归因没有系统性偏差。

**修正过程中的一个陷阱**（已修）：分段计数函数一开始仍用单比值折算中文，
把中文的低估误差全部推进残差（`template_ctl` 从 7 虚增到 16）。
**残差因此变成了误差垃圾桶**，"模板开销"这个指标失去物理意义。
修正后残差回落到 11，与标定截距吻合。规则：**分段计数一律不含截距**
（截距是每请求一次的量，摊到每段会重复计算）。

### P21 · 标定后 drift 归零 ✅
同一条请求（19 in / 47 out）标定前后对比：

| | 采信 | drift | 异常 |
|---|---|---|---|
| 标定前 | engine/high | 15.79% | TOKEN_DRIFT（后被绝对差门槛抑制）、COLD_LOAD |
| 标定后 | engine/high | **0.00%** | 无 |

heuristic 档对该模型输出估 71 token（引擎 47），因为它按 1.0 token/字 估中文，
而该模型实际是 0.679 —— **这正是 heuristic 必须标 `confidence=low` 的原因**。

### P22 · httpx 把有效超时暴露在 `request.extensions["timeout"]` ✅（S11 实测）

```python
with httpx.Client(transport=httpx.MockTransport(handler), timeout=httpx.Timeout(1.234)) as c:
    c.request("GET", "http://x.invalid/y")
# handler 里读到：
# request.extensions == {'timeout': {'connect': 1.234, 'read': 1.234,
#                                    'write': 1.234, 'pool': 1.234}}
```

**为什么这条值得单独记**：它让"deadline 有没有传到 socket"变成**可断言的**，
不必起一个真实慢服务器去猜。`tools/executors/http.py` 的契约 handler 就是靠它自检的——
若执行器无视 `ctx.deadline_ms`，handler 读到 `read=15.0 > 0.05` 就返回 500，
契约断言 `timeout_is_reported` 随即变红。

不传播的后果不是报错，而是 `run_with_deadline` 放弃等待后底层请求继续阻塞，
永久占用 `ThreadPoolExecutor` 的一个 worker；多来几次工具子系统整体变慢，
现象是"越来越慢"而不是"失败"——最难归因的那一类。

> 附带确认：`MockTransport` **不会**因为 handler 里 `time.sleep()` 而触发超时
> （它不看真实耗时），所以模拟超时只能由 handler 主动 `raise httpx.ReadTimeout`。
> 这也说明"用 sleep 测超时"在 mock 传输下是无效测试。

---

## 3. 仍未定 / 待测

| 编号 | 待答问题 | 为什么关键 |
|---|---|---|
| U7 | `qwen3.8:27b`(17.7GB > 16GB VRAM) 的 offload 行为与吞吐落差 | 验证 `offloaded` 判定与"必须分开聚合"（DESIGN R5） |
| U8 | 图片输入的 token 如何计（`vision` 能力模型） | 文本 tokenizer 估不准，需标 `vision_estimate_low_conf`（R6） |
| U9 | `prompt_eval_cached_count` 之类字段是否存在（本次实测 `cached_tokens` 恒为 None） | 若引擎不报缓存量，`prefill_mode` 只能用 P11 的比值推断 |


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


### P23 · Ollama 的 `/v1` 无法关闭 thinking，推理字段名是 `reasoning` ✅（S16b 实测）

```bash
# qwen3.5:9b，max_tokens=64
curl -s http://127.0.0.1:11434/v1/chat/completions   -d '{"model":"qwen3.5:9b","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":64,"think":false}'
curl -s http://127.0.0.1:11434/v1/chat/completions   -d '{"…","chat_template_kwargs":{"thinking":false}}'
# 两次都返回：{"content": "", "reasoning": "Thinking Process:

1. **Analyze the Req…"}
```

| 假设（来自文档/直觉） | 实测 |
|---|---|
| `think:false` 能关推理（原生通道的键名） | ✗ 无效，照样产出推理内容 |
| `chat_template_kwargs.thinking:false` 能关（vLLM 的键名） | ✗ Ollama `/v1` 不认 |
| 推理内容在 `delta.reasoning_content`（DeepSeek 系形状） | 非流式实际是 `message.reasoning` |
| `/v1` 报 `total_duration` 等分段时序 | ✗ 只有 `usage`，没有纳秒分段 |

**处置**：`openai-compat` provider 在 `req.thinking` 被设值且未声明 `thinking_via` 时
**显式拒绝**（`CapabilityMissing`），而不是"发一个可能被忽略的字段"。
理由是可比性：thinking 开着跑出来的分数与关着跑出来的分数不是同一个量，
静默忽略会让两次评测看起来同构而实际不同（P12 的教训在另一个通道上重演）。
服务器确实支持时，用 `thinking_via="chat_template_kwargs.thinking"` 声明键位即可。
解析侧同时吃 `reasoning` 与 `reasoning_content`，`streaming.py` 的 openai 分支
`delta` 与 `message` 都接受（只认 `delta` 时非流式解析成空正文，
而空正文与"模型真的没输出"在评测里长得一模一样）。

### P24 · 兼容层的 `usage` 是该通道唯一可信的引擎自报 ✅（S16b 实测）

同一请求（prompt「用一句话解释什么是 KV cache」，`max_tokens=400`）：

| 出处 | in | out | 备注 |
|---|---|---|---|
| `compat`（服务器 `usage`） | 16 | 400 | `finish_reason=length` ⇒ 400 与预算**自洽** |
| `heuristic`（本地字符加权） | 15 | 602 | 估高了 20% |
| `fitted` | 未标定 | 未标定 | 该通道没有模板，标定档不可用 |

P14 的结论（`/v1` 与原生计数口径不同，差 −2/+16）没有被推翻：它约束的是
**有原生数字时该信谁**。而 vLLM / LM Studio / Ollama `/v1` 只有 compat 这个数字，
把它排除、改用字符估计，等于放着卡尺不用去拃。所以
`SOURCE_PRIORITY = (ENGINE, HF_TOKENIZER, GGUF_VOCAB, FITTED, COMPAT, HEURISTIC)`
——compat 在所有本地复算档之后、heuristic 之前，并带 LOW 置信度。

**顺带实测到**：`/v1/models` **不返回** per-model `capabilities`、体积与上下文长度
（`{"id","created","owned_by"}` 而已）。所以"空 capabilities 清单"的含义是
**该通道不上报**，不是"上报了且什么都不支持"。看板原先会因此显示一排 ✗，
而 ✗ 的处置是"评测直接 skip"、? 的处置才是"先跑探针"——两个相反的结论被压成了同一个符号。
