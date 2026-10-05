/** API 客户端。
 *  错误统一成 ApiError{code,message,detail}——后端保证不返回堆栈，前端也不该把
 *  原始 Response 到处传。 */

import { authHeaders } from './token'

export class ApiError extends Error {
  code: string
  status: number
  detail: Record<string, unknown>

  constructor(status: number, code: string, message: string, detail: Record<string, unknown> = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.detail = detail
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(path, {
      ...init,
      // 每个请求都要带 Authorization（非回环部署时）；content-type 由本文件统一给
      headers: authHeaders({ 'content-type': 'application/json' }),
    })
  } catch (err) {
    // 网络层失败（后端没起）：给出可行动的提示，而不是 "Failed to fetch"
    throw new ApiError(0, 'NETWORK', `无法连接 Onyx API：${(err as Error).message}`, {
      hint: '确认已运行 onyx serve（默认 http://127.0.0.1:8787）',
    })
  }
  const text = await resp.text()
  const body = text ? safeJson(text) : null
  if (!resp.ok) {
    const error = (body as { error?: { code?: string; message?: string; detail?: object } })?.error
    throw new ApiError(
      resp.status,
      error?.code ?? 'HTTP_' + resp.status,
      error?.message ?? resp.statusText,
      (error?.detail as Record<string, unknown>) ??
        (resp.status === 401 ? { hint: '这个看板要 token：在地址后加 ?token=…（只读共享时最省事）' } : {}),
    )
  }
  return body as T
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return { raw: text }
  }
}

function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined || value === '') continue
    search.set(key, String(value))
  }
  const s = search.toString()
  return s ? `?${s}` : ''
}

import type {
  AdminResultView,
  AlertRuntime,
  AlertTriggerView,
  ChatResponse,
  ComparisonView,
  DatasetImportBody,
  DatasetView,
  FleetView,
  GradeView,
  GpuStatusView,
  HealthView,
  ImportView,
  MatrixView,
  ModelView,
  ProgressView,
  QueueView,
  RunView,
  SubmitView,
  TaskView,
  ToolAuditView,
  ToolCostView,
  ToolDefView,
  ToolMatrixView,
  ToolRunView,
  TraceDetail,
  TracePage,
  UsageSummaryView,
} from './types'

export const api = {
  health: () => request<HealthView>('/api/health'),
  fleet: () => request<FleetView>('/api/fleet'),
  models: () => request<ModelView[]>('/api/models'),
  modelDetail: (name: string) => request<ModelView>(`/api/models/detail${qs({ name })}`),
  modelProbe: (name: string) => request<Record<string, unknown>>(`/api/models/probe${qs({ name })}`),
  traces: (params: {
    limit?: number
    cursor?: string | null
    purpose?: string | null
    model?: string | null
    status?: string | null
    since?: string | null
  } = {}) => request<TracePage>(`/api/traces${qs(params)}`),
  trace: (id: string, includeBlobs = true) =>
    request<TraceDetail>(`/api/traces/${encodeURIComponent(id)}${qs({ include_blobs: includeBlobs })}`),
  usageSummary: (params: { since?: string | null; model?: string | null; bucket_minutes?: number } = {}) =>
    request<UsageSummaryView>(`/api/usage/summary${qs(params)}`),
  chat: (body: {
    model: string
    prompt: string
    max_tokens?: number
    temperature?: number
    thinking?: boolean | null
    stream?: boolean
    tools?: string[]
    system?: string
    /** 客户端关联键：SSE 事件流靠它对上这次请求（trace_id 仍由服务端生成以保持可排序） */
    client_key?: string
  }) =>
    request<ChatResponse>('/api/playground/chat', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  demoTools: () => request<{ tools: Array<{ key: string; name: string; description: string }> }>('/api/tools/demo'),
  /** Tool Bench（S25）：注册表 / 审计 / 开销 / 契约矩阵 / 运行历史。全部只读，与 CLI 同源 */
  toolDefs: (params: { enabled_only?: boolean; kind?: string | null } = {}) =>
    request<ToolDefView[]>(`/api/tools${qs(params)}`),
  toolAudit: (model?: string | null) =>
    request<ToolAuditView>(`/api/tools/audit${qs({ model })}`),
  toolCost: (params: { model?: string | null; overhead?: number } = {}) =>
    request<ToolCostView>(`/api/tools/cost${qs(params)}`),
  /** 矩阵会在离线样本上真跑一遍各执行器：给人点一次，不给界面轮询 */
  toolMatrix: (params: { tool?: string; args?: string | null } = {}) =>
    request<ToolMatrixView>(`/api/tools/matrix${qs(params)}`),
  toolRuns: (params: { tool_id?: string | null; limit?: number } = {}) =>
    request<ToolRunView[]>(`/api/tools/runs${qs(params)}`),
  unload: (name: string) =>
    request<AdminResultView>(`/api/admin/models/unload${qs({ name, confirm: 1 })}`, {
      method: 'POST',
    }),
  /** 拉取是一次长请求：几 GB 的下载会占住这条连接几分钟，中间层可能先掐断。
   *  界面上必须写这句，否则超时看起来像"Onyx 拉取失败"。大下载请走 onyx models pull。 */
  pullModel: (name: string) =>
    request<AdminResultView>(`/api/admin/models/pull${qs({ name, confirm: 1 })}`, { method: 'POST' }),
  /** 只删权重：历史 trace 与分数一行都不动（那次测量已经发生了） */
  removeModel: (name: string) =>
    request<AdminResultView>(`/api/admin/models/rm${qs({ name, confirm: 1 })}`, { method: 'POST' }),
  /** GPU 锁状态是**只读**的：看板轮询它不能把锁抢了 */
  gpu: () => request<GpuStatusView>('/api/gpu'),
  /** 触发历史：回答"某天到底通知没通知"。测试行默认不混进来 */
  alertTriggers: (params: { code?: string | null; status?: string | null; limit?: number } = {}) =>
    request<AlertTriggerView[]>(`/api/alerts${qs(params)}`),
  /** 告警系统自己的状态：没装配时 reason 会说清，而不是留一片空白 */
  alertStatus: () => request<AlertRuntime>('/api/alerts/status'),
  evalDatasets: () => request<DatasetView[]>('/api/datasets'),
  /** 导入 JSONL 数据集（文本而不是路径）。覆盖已有 id 需要 allow_replace=true */
  importDataset: (body: DatasetImportBody) =>
    request<ImportView>('/api/datasets', { method: 'POST', body: JSON.stringify(body) }),
  /** 任务清单现读后端注册表：前端不硬编码任务名，否则插件任务在界面上是隐形的 */
  evalTasks: () => request<TaskView[]>('/api/tasks'),
  /** 发起评测：只入队，立刻返回 run_id。跑评测的是服务里的 worker 线程，不是这个请求 */
  startRun: (body: {
    task: string
    model: string
    k?: number
    limit?: number | null
    split?: string
    seed?: number | null
    dataset?: string | null
    max_tokens?: number | null
    unload_others?: boolean
    notes?: string
    /** 续跑：新样本写回这条 run，已评过的 case 不重复计费 */
    resume_run_id?: string | null
  }) => request<SubmitView>('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  runProgress: (runId: string) => request<ProgressView>(`/api/runs/${encodeURIComponent(runId)}/progress`),
  cancelRun: (runId: string) =>
    request<{ run_id: string; state: string; cancelled: boolean; message: string }>(
      `/api/runs/${encodeURIComponent(runId)}/cancel`,
      { method: 'POST' },
    ),
  evalQueue: () => request<QueueView>('/api/queue'),
  evalRuns: (params: { task?: string | null; model?: string | null; limit?: number } = {}) =>
    request<RunView[]>(`/api/runs${qs(params)}`),
  evalGrades: (runId: string, params: { verdict?: string | null; limit?: number } = {}) =>
    request<GradeView[]>(`/api/runs/${encodeURIComponent(runId)}/grades${qs(params)}`),
  matrix: (params: { task?: string | null; model?: string | null; dataset?: string | null } = {}) =>
    request<MatrixView>(`/api/matrix${qs(params)}`),
  /** 配对对比在后端算：CI 与净变化只有一处实现，界面不会算出第二个版本 */
  compare: (base: string, target: string, params: { eps?: number; with_cases?: boolean } = {}) =>
    request<ComparisonView>(`/api/compare${qs({ base, target, ...params })}`),
}
