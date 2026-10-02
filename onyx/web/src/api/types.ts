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
  size_gb: number
  capabilities: string[]
  caps: CapReportDto
  tool_format: string
  ctx_train: number | null
  ctx_loaded: number | null
  tokenizer_source: string
  calibrated: boolean
  calibration: Record<string, number | null>
  probed: boolean
  loaded: boolean
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
