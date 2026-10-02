/** API 客户端。
 *  错误统一成 ApiError{code,message,detail}——后端保证不返回堆栈，前端也不该把
 *  原始 Response 到处传。 */

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
      headers: { 'content-type': 'application/json' },
      ...init,
    })
  } catch (err) {
    // 网络层失败（后端没起）：给出可行动的提示，而不是 "Failed to fetch"
    throw new ApiError(0, 'NETWORK', `无法连接 Onyx API：${(err as Error).message}`, {
      hint: '确认已运行 onyx serve（默认 http://127.0.0.1:8000）',
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
      (error?.detail as Record<string, unknown>) ?? {},
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
  ChatResponse,
  FleetView,
  HealthView,
  ModelView,
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
  unload: (name: string) =>
    request<{ ok: boolean }>(`/api/admin/models/unload${qs({ name, confirm: 1 })}`, {
      method: 'POST',
    }),
}
