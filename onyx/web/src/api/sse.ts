/** SSE 客户端：订阅 /api/stream 并按需过滤。
 *  断线自动重连；事件按 trace/client_key 关联，避免多模型并排时串台。 */
import { useEffect, useRef, useState } from 'react'

import { withToken } from './token'

export interface TraceEventDto {
  v: number
  type: string
  trace_id: string
  ts_ns: number
  wall_iso: string
  payload: Record<string, unknown>
}

export interface SseState {
  connected: boolean
  /** 最近 N 条事件（环形缓冲，避免长时间运行把内存吃满） */
  events: TraceEventDto[]
  error: string | null
}

const BUFFER = 400
const RECONNECT_MS = 1500

export function useSse(filter?: (event: TraceEventDto) => boolean): SseState {
  const [state, setState] = useState<SseState>({ connected: false, events: [], error: null })
  const filterRef = useRef(filter)
  filterRef.current = filter

  useEffect(() => {
    let source: EventSource | null = null
    let stopped = false
    let timer = 0

    const connect = () => {
      if (stopped) return
      // token 只能走 query：EventSource 没有设请求头的办法
      source = new EventSource(withToken('/api/stream'))
      source.onopen = () => setState((s) => ({ ...s, connected: true, error: null }))
      source.onerror = () => {
        setState((s) => ({ ...s, connected: false, error: 'SSE 连接中断，重连中' }))
        source?.close()
        if (!stopped) timer = window.setTimeout(connect, RECONNECT_MS)
      }
      source.onmessage = (message) => {
        let event: TraceEventDto
        try {
          event = JSON.parse(message.data) as TraceEventDto
        } catch {
          return // 非法帧直接丢弃：未知/坏数据不许让页面崩（事件契约的前向兼容要求）
        }
        if (filterRef.current && !filterRef.current(event)) return
        setState((s) => ({
          ...s,
          events: [...s.events, event].slice(-BUFFER),
        }))
      }
    }

    connect()
    return () => {
      stopped = true
      window.clearTimeout(timer)
      source?.close()
    }
  }, [])

  return state
}

/** 按 client_key 过滤：TRACE_START 的 context.extra.client_key 由前端自己生成。 */
export function byClientKey(key: string): (event: TraceEventDto) => boolean {
  if (!key) return () => true
  let matchedTrace: string | null = null
  return (event) => {
    if (matchedTrace) return event.trace_id === matchedTrace
    if (event.type !== 'trace_start') return false
    const context = event.payload.context as { extra?: { client_key?: string } } | undefined
    if (context?.extra?.client_key === key) {
      matchedTrace = event.trace_id
      return true
    }
    return false
  }
}
