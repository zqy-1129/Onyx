/** 后端 API 契约的镜像（onyx/api/schemas.py）。
 *  只描述前端会用到的字段；后端多给的字段会被忽略，少给的用可选类型兜住。 */

export type Confidence = 'high' | 'medium' | 'low'
export type PrefillMode = 'cold' | 'warm' | 'unknown'
export type TokenSource =
  | 'engine'
  | 'compat'
  | 'hf_tokenizer'
  | 'gguf_vocab'
  | 'fitted'
  | 'heuristic'

export interface HealthView {
  ok: boolean
  version: string
  schema_version: number
  provider_reachable: boolean
  engine_version: string
}

export interface LoadedModelView {
  name: string
  size: number
  size_vram: number
  vram_share: number | null
  offloaded: boolean
  context_length: number | null
  expires_at: string
  keep_alive_seconds: number | null
  quantization: string
  parameter_size: string
}

export interface FleetWindow {
  seconds: number
  traces: number
  errors: number
  error_rate: number
  in_tokens: number
  out_tokens: number
  thinking_tokens: number
  decode_tps_avg: number | null
  by_prefill_mode: Record<string, number>
}

export interface FleetView {
  ok: boolean
  app_version: string
  provider_id: string
  provider_kind: string
  provider_reachable: boolean
  engine_version: string
  base_url: string
  loaded_models: LoadedModelView[]
  /** false 时 loaded_models 是空且无含义：面板必须显示「驻留状态未知」而不是"当前没有载入任何模型" */
  loaded_known: boolean
  installed_models: number
  window: FleetWindow
  anomalies: Record<string, number>
}

export interface CapReportDto {
  confirmed: string[]
  missing: string[]
  unknown: string[]
  reasons: Record<string, string>
}

export interface ModelView {
  id: string
  name: string
  provider_id: string
  parameter_size: string
  quantization: string
  size_gb: number | null
  capabilities: string[]
  caps: CapReportDto
  tool_format: string
  ctx_train: number | null
  ctx_loaded: number | null
  tokenizer_source: string
  calibrated: boolean
  calibration: Record<string, number | null>
  probed: boolean
  /** null = 该通道不报告驻留状态（例如 OpenAI 兼容层）。
   *  必须与 false（问了，答案是"没载入"）区分：把未知画成"未载入"会引着人去查一个不存在的问题 */
  loaded: boolean | null
}

export interface UsageAlt {
  source: string
  in_tokens: number | null
  out_tokens: number | null
  thinking_tokens: number | null
  cached_tokens: number | null
  ok: boolean
  confidence: string | null
  note: string
}

export interface TokenPartView {
  part: string
  ord: number
  tokens: number
  bytes: number | null
}

export interface LatencyView {
  ttft_ms: number | null
  wall_ms: number | null
  prefill_mode: PrefillMode
  prefill_ms_per_token: number | null
  prefill_tps: number | null
  decode_tps: number | null
  load_ms: number | null
  cold_load: boolean
}

export interface ToolCallView {
  id: string
  step: number
  name: string | null
  parse_status: string
  parse_source: string | null
  args: Record<string, unknown> | null
  args_raw: string | null
  result_status: string | null
  latency_ms: number | null
  executed_by: string | null
}

export interface AnomalyView {
  id: string
  code: string
  severity: string
  meaning: string
  action: string
  detail: Record<string, unknown>
}

export interface TraceSummary {
  id: string
  purpose: string
  kind: string
  model_name: string | null
  provider_id: string | null
  started_at: string
  status: string
  finish_reason: string | null
  in_tokens: number | null
  out_tokens: number | null
  source: string | null
  confidence: string | null
  ttft_ms: number | null
  decode_tps: number | null
  prefill_mode: string | null
  tool_calls: number
  anomalies: number
  eval_run_id: string | null
  case_id: string | null
}

export interface TracePage {
  items: TraceSummary[]
  next_cursor: string | null
  total: number
}

export interface TraceDetail {
  trace: TraceSummary
  params: Record<string, unknown>
  latency: LatencyView
  usage: UsageAlt | null
  alts: UsageAlt[]
  parts: TokenPartView[]
  tool_calls: ToolCallView[]
  anomalies: AnomalyView[]
  gpu: Record<string, unknown>
  engine_latency: Record<string, number | null>
  refs: Record<string, string>
  messages: Array<Record<string, unknown>>
  output: Record<string, unknown>
  attribution: Record<string, unknown>
}

export interface UsageSummaryView {
  traces: number
  in_tokens: number
  out_tokens: number
  thinking_tokens: number
  by_source: Record<string, number>
  by_confidence: Record<string, number>
  by_prefill_mode: Record<string, number>
  drift: { n: number; max: number | null; p50: number | null; over_threshold: number }
  timeseries: Array<Record<string, number | string>>
}

export interface ChatResponse {
  trace_id: string
  text: string
  thinking: string
  finish_reason: string
  tool_calls: ToolCallView[]
  usage: UsageAlt | null
  latency: LatencyView
  anomalies: AnomalyView[]
  parts: TokenPartView[]
}

/* ── 评测（S13–S15）───────────────────────────────────────────── */

/** 置信区间。后端在落库前已把 CI dataclass 转成 dict，所以这里永远是对象。 */
/** 一次运行的开销。unloaded_models 是字符串数组，所以这里不用 Record<string, number> 糊过去。 */
export interface EvalCost {
  in_tokens?: number
  out_tokens?: number
  requests?: number
  in_tokens_unknown?: number
  wall_ms?: number
  unloaded_models?: string[]
}

export interface CIView {
  low: number | null
  high: number | null
  point: number | null
  n: number
  iterations?: number
  method?: string
  low_confidence?: boolean
}

export interface DatasetView {
  id: string
  n_cases: number | null
  upstream: string
  revision: string
  license: string
  loader: string
  splits: Record<string, number>
  imported_at: string
  notes: string
  /** 界面能不能直接选它跑评测：内置写法，或已导入且真有样本的登记。
   *  只有 dataset 行没有样本的（导入被中断）选了只会跑出一个 n_total=0 的"正常"评测 */
  selectable: boolean
}

/** 数据集导入：JSONL **文本**而不是路径——请求体里带路径等于让服务器读任意文件 */
export interface DatasetImportBody {
  jsonl: string
  name?: string
  id?: string | null
  upstream?: string
  revision?: string
  license?: string
  notes?: string
  allow_replace?: boolean
}

export interface ImportView {
  id: string
  n_cases: number
  upstream: string
  revision: string
  license: string
  splits: Record<string, number>
  replaced: boolean
  warnings: string[]
}

/** 评测任务清单：来自后端注册表，不在前端硬编码，否则装了插件的任务界面看不到 */
export interface TaskView {
  id: string
  name: string
  requires: string[]
  metrics: string[]
  labels: string[]
  default_dataset: string
  dataset_revision: string
  n_cases: number | null
  splits: Record<string, number>
  max_tokens: number | null
  temperature: number | null
  error: string
}

export interface SubmitView {
  run_id: string
  state: string
  position: number
  task: string
  model: string
}

/** 进度：source 说清出处。CLI 发起的运行没有内存快照，也就没有取消开关 */
export interface ProgressView {
  run_id: string
  source: 'service' | 'db' | string
  state: string
  task: string
  model: string
  done: number
  total: number
  case_id: string
  verdict: string
  position: number
  holder: string
  eta_s: number | null
  waited_s: number
  error: string
  reason: string
  queued_at: string
  started_at: string
  finished_at: string | null
  cancellable: boolean
  n_error: number
  dataset_id: string | null
}

export interface QueueView {
  max_pending: number
  jobs: ProgressView[]
}

export interface RunView {
  id: string
  task_id: string
  model_id: string
  status: string
  started_at: string
  finished_at: string | null
  seed: number | null
  app_version: string
  git_rev: string
  params_snapshot: Record<string, unknown>
  config: Record<string, unknown>
  n_cases: number
  n_done: number
  n_error: number
  n_skipped: number
  dataset_id: string | null
  dataset_revision: string
  /** 汇总指标。值可能是 null（=未定义），CI 是 CIView */
  aggregate: Record<string, unknown>
  cost: EvalCost
}

export interface GradeView {
  case_id: string
  seq: number
  score: number
  verdict: string
  passed: boolean | null
  invalid_format: boolean
  out_of_set: boolean
  /** 每个分数都能点进一条真实 trace；为空表示这条没发出请求（skip） */
  trace_id: string | null
  error: string | null
  metrics: Record<string, unknown>
}

export interface MatrixCell {
  run_id: string
  model_id: string
  task_id: string
  metric: string
  value: number | null
  ci: CIView | null
  n: number | null
  n_judged: number | null
  n_total: number | null
  coverage: number | null
  low_confidence: boolean
  status: string
  started_at: string
  dataset_id: string
  dataset_revision: string
  cost: EvalCost
}

export interface MatrixView {
  models: string[]
  tasks: string[]
  cells: MatrixCell[]
  provenance: string[]
  warnings: string[]
}

export interface PairedCase {
  case_id: string
  kind: string
  score_base: number
  score_target: number
  delta: number
  passed_base: boolean | null
  passed_target: boolean | null
  verdict_base: string
  verdict_target: string
  trace_base: string | null
  trace_target: string | null
  instruction: string
}

export interface RunBrief {
  id: string
  task_id: string
  model_id: string
  status: string
  n_cases: number
  n_done: number
  n_error: number
  seed: number | null
  started_at: string
  dataset_id: string | null
  dataset_revision: string
  params: Record<string, unknown>
  k: number | null
  headline: { metric: string; value: number | null } | null
  cost: EvalCost
}

export interface ComparisonView {
  base: RunBrief
  target: RunBrief
  eps: number
  n_paired: number
  only_base: string[]
  only_target: string[]
  coverage: number | null
  mean_delta: number | null
  delta_ci: CIView | null
  improved: number
  regressed: number
  unchanged: number
  flips: { up: number; down: number; net: number }
  low_confidence: boolean
  warnings: string[]
  cases: PairedCase[]
}

export interface GpuStatusView {
  busy: boolean
  owner: string | null
  progress: string | null
  eta_s: number | null
  holder_host: string | null
}
