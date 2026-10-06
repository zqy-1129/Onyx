# 评测配方（eval recipes）

这份文档回答的是"**我该怎么跑一次能解释的评测**"，不是 API 参考。
每条配方后面都跟着它会骗人的地方——在本项目里，一个不带口径的分数比没有分数更糟。

相关文档：口径与结论在 `PROBES.md`，架构约束在 `DESIGN.md`，
界面显示规则在 `UI_DESIGN.md`（R2：未知显示「—」，绝不显示 0）。

---

## 0. 三条前置认知

1. **打分口径只有 gen-based**。API-only 拿不到受约束的 logprob，所以
   `onyx eval run` 的输出会固定写着"分数不可与公开 leaderboard 直接比较"。
   内容与格式是**两个正交维度**（`verdict` 与 `invalid_format`）：
   混成一个"正确率"就会把格式问题误读成能力问题，而两者修法相反。
2. **分数必须有分母**。`macro_f1 1.000` 在 8/236 可判定的时候是真的，但它是
   "那 8 条都对"，不是"这个模型很强"。任何出现在界面/报告/导出里的数字都要带上
   `可判定 n/N`、CI 的 case 数、覆盖率。
3. **未知 ≠ 0**。任务被能力位跳过会留一条 `status=skipped` 的记录并写明原因；
   指标算不出来是 `None`（显示「—」）；`P=R=0` 才是 0.0（全错，不是未定义）。

---

## 1. 最小一次评测

```bash
onyx db init
onyx eval tasks                       # 有哪些任务、来源（内建/插件）、需要哪些能力
onyx eval run --task intent_classification --model qwen3.5:9b --limit 20
onyx eval show <run_id>               # 每个分数都能下钻到真实 trace
```

离线跑通整条管道（不起引擎）：`--provider mock`。它产出的是**管道正确性**证据，
不是模型能力证据——mock 会回显提示词，分数必然是 0，这没问题；
把它当成"这个模型 0 分"才是问题。

k>1 才有的指标（pass^k / stability_gap）在 k=1 时是 `None`，不是 0。

## 2. 数据集：从哪来、什么时候不可比

| 场景 | 命令 | 注意 |
|---|---|---|
| 用任务自带的内建数据集 | 省略 `--dataset` | 报告里会写 `dataset_id` 与 `revision` |
| 直接读一份 JSONL | `--dataset file:cases.jsonl` | 不落库，快，但 `revision` 由文件内容决定不了 ⇒ 自己保证 |
| 进库并登记来历 | `onyx eval import --id my_set --file cases.jsonl --upstream … --revision …` | 来历缺失的导入会被拒绝 |
| 第三方基准 | `onyx eval import --source bfcl --answers …` | 只导入**题干与期望**，打分仍走 onyx 自己的口径 |

**改语料必须改 `revision`**。`onyx eval compare` 会在 revision 不同时直接判"不可比"，
而不是给你一个大差值——两次不同数据的 diff 没有意义。

## 3. 工具调用评测（tool_selection）

```bash
onyx eval run --task tool_selection --model qwen3.5:9b --k 3
```

看点在**判定分布**而不是一个总分：`WRONG_TOOL`（工具之间的描述区分度不够）、
`BAD_ARGS`（参数描述/required/枚举说明不够）、`HALLUCINATED_TOOL`（工具集外编名字）、
`NO_CALL`（该调不调）、`FALSE_CALL`（不该调却调）——修法各不相同。
真机第一版结论就是"选工具几乎没错，掉分全在参数上"。

评测期默认 `mock_policy=fixture`：**一个真实副作用都不发生**。
case 里没预置桩时该次调用记 `skipped`，不会退回真跑（那样"可复现"就没了）。

### 用 MCP 工具做这件事

```bash
$EDITOR .data/mcp.json                 # {"mcpServers": {"demo": {"command": ["python","server.py"]}}}
onyx tools mcp-ls                      # 看副作用与"为什么这么定"
onyx tools mcp-import                  # 注册为 kind=mcp，名字 demo__<tool>
onyx tools contract                    # 五列矩阵：mcp（假连接）与 mcp_stdio（真子进程）都要 8/8
onyx eval run --task tool_selection --model … --k 3     # 评测期照旧走桩
```

服务器没标注 `readOnlyHint` 的工具会按 `write` 登记，默认策略直接拒绝执行——
这是有意的（提示不是保证）。`onyx tools fire … --mock live` 才会真起子进程。

## 4. 比较与回归

```bash
onyx eval compare --run-a <a> --run-b <b> --metric macro_f1
onyx eval matrix --tasks intent_classification,tool_selection
onyx eval report --run <id> --format html --out report.html
```

比较是**按 case 配对**的（improved/regressed/unchanged + McNemar 翻转表 + 配对 bootstrap），
不是"比两个平均分"，也不是"看两个 CI 是否重叠"。
配对数 < 30 会标 `low_confidence`；单跑 case 数 < 100 同理。
CI 是对 **case** 重采样得到的，不是对样本——同一个 0.639 按 sample 采样会把区间窄掉 40%。

## 5. 换引擎跑同一套任务

```bash
# vLLM / LM Studio / Xinference / Ollama 的 /v1
onyx eval run --task intent_classification --provider openai-compat \
  --url http://127.0.0.1:8000/v1 --model qwen3.5:9b
```

三条必须知道的差别（不是 bug，是通道的物理限制）：
- **计数出处是 `compat`**：没有原生通道数字时它是唯一可信的服务器自报；
  两条通道对同一 prompt 的计数本来就不同（P14）。
- **没有分段时序**：TTFT、prefill/decode TPS 一律「—」，不是 0。
- **thinking 关不掉就别装能关掉**。Ollama `/v1` 实测既不认 `think` 也不认
  `chat_template_kwargs.thinking`（P23），所以 provider 在被要求 `thinking=False` 时
  **直接报错**而不是假装设置过。服务器确实支持的（vLLM 一类）在构造时声明键位：
  `thinking_via="chat_template_kwargs.thinking"`。
- 探针套件目前大量依赖 Ollama 原生端点 ⇒ 兼容通道上 `structured_output` /
  `stream_usage` 会长期停在「未实测」(`?`)。**`?` 与 `✗` 的处置相反**：
  前者该去补测，后者该 skip 并写明原因。别把 `?` 读成"不支持"。

外部 provider 也可以走插件（`onyx.providers`），见 `plugins_example/example_provider/`；
`onyx plugins` 会把每个扩展点的"内建 / 插件 / 覆盖 / 坏插件"一次列全，有坏插件时退出码非 0。

## 6. 导出与接线外部可观测栈

```bash
onyx chat "你好" --sink jsonl                     # <data_dir>/events.ndjson
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318 \
  onyx serve --provider openai-compat --url http://127.0.0.1:8000/v1 --sink otlp
```

OTLP 走 **JSON over HTTP**（不是 protobuf），所以个别只实现 protobuf 的接收端会拒绝；
导出里带 `onyx.encoding=json` 与 `onyx.trace_id`，两边可以互相跳转。
一个 trace 只导结构 span（根 + 每次工具执行），逐 token 增量折成计数属性。

## 7. 写一个自己的任务（不改内核）

```toml
# pyproject.toml
[project.entry-points."onyx.tasks"]
my_task = "my_pkg.task:spec"
```

```python
from onyx.eval.task import Case, Grade, TaskSpec, Verdict

class MyTask:
    id = "my_task"                       # 必须等于 entry point 的名字
    name = "…"
    requires = frozenset()               # 不满足的能力位 ⇒ 整任务 skip 并留记录
    metric_names = ("accuracy", "n_judged")

    def __init__(self, dataset, *, model, **kw): …
    def load(self, *, split="default", limit=None): …    # yield Case
    def build(self, case) -> GenerationRequest: …        # 只构造请求，不发请求
    def grade(self, case, sample) -> Grade: …            # 内容 verdict 与 invalid_format 分开
    def aggregate(self, grades, *, seed=0) -> dict: …    # 键必须与 metric_names 同源

spec = TaskSpec(MyTask, my_dataset_loader)
```

三条会被断言拦住的规矩：
- **任务不许自己发请求**：`build` 返回 `GenerationRequest`，评分永远经过同一个 gateway，
  这样每个分数都能下钻到真实 trace；
- `metric_names` 与 `aggregate` 的键同源（声明了产不出 = 界面上永远「—」，和 0 分长得一样）；
- 没有默认数据集时 `TaskSpec.dataset=None`，跑的时候必须显式 `--dataset`（不许猜一份数据）。

装好验证：`uv run --with-editable ./my_plugin onyx eval tasks`。
`onyx eval tasks` 会显示 `来源=插件`；插件坏了会进台账并让命令退出码 1。

## 8. 出问题时先看这几处

| 现象 | 大概率原因 | 怎么确认 |
|---|---|---|
| 分数全是 `—` | 任务被能力位 skip，或没有可判定样本 | `onyx eval show <run>` 顶部的 skip 原因 / `n_judged` |
| 正文为空 + `invalid_format` 高 | thinking 吃光预算（P12/P23） | 提高 `--max-tokens`，或换能关 thinking 的通道 |
| 两次分数差很多但 CI 很窄 | 样本量小、k 太小 | 看 `n` 与 `low_confidence`，加 `--k` |
| 模型选对工具但参数总错 | 参数描述/required 不够 | `onyx tools audit`、`onyx tools contract` |
| 评测很慢/GPU 被占 | 别的进程持锁（锁是**机器级**的） | `curl :8000/api/gpu` 或 `onyx doctor` |
| 接了 sink/插件但没效果 | 它加载失败被隔离了 | `onyx plugins`（坏插件会列出来且退出码 1） |
