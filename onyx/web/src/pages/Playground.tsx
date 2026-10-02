/** Playground：多模型并排、thinking 分栏、工具调用实时渲染。
 *  并排请求在服务端被 GPU 锁串行化（单卡独占），UI 明确显示"排队中"而不是假装并行——
 *  否则用户会把排队时间读成模型延迟。 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import { useSse, type TraceEventDto } from '../api/sse'
import type { ChatResponse, ModelView } from '../api/types'
import { PromptBreakdown } from '../components/PromptBreakdown'
import {
  AnomalyChip,
  ErrorState,
  Panel,
  PrefillTag,
  SourceBadge,
  StatCard,
  StatusBadge,
} from '../components/primitives'
import { fmtFloat, fmtInt, fmtMs, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

type RunStatus = 'idle' | 'queued' | 'streaming' | 'done' | 'error'

interface Run {
  clientKey: string
  model: string
  status: RunStatus
  text: string
  thinking: string
  toolCalls: Array<{ idx: number; name: string; args: string }>
  firstTokenMs: number | null
  result: ChatResponse | null
  error: string | null
  startedAt: number
}

function newKey(): string {
  return `pg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

export function PlaygroundPage() {
  const models = useApi<ModelView[]>(() => api.models())
  const tools = useApi<{ tools: Array<{ key: string; name: string; description: string }> }>(() =>
    api.demoTools(),
  )

  const [prompt, setPrompt] = useState('北京现在天气怎么样？')
  const [system, setSystem] = useState('')
  const [maxTokens, setMaxTokens] = useState(512)
  const [temperature, setTemperature] = useState(0)
  const [thinking, setThinking] = useState(false)
  const [selected, setSelected] = useState<string[]>([])
  const [selectedTools, setSelectedTools] = useState<string[]>([])
  const [runs, setRuns] = useState<Record<string, Run>>({})
  const [busy, setBusy] = useState(false)

  const keysRef = useRef<Set<string>>(new Set())
  const keyToModel = useRef<Map<string, string>>(new Map())
  const traceToModel = useRef<Map<string, string>>(new Map())
  const processed = useRef(0)
  const initialized = useRef(false)

  const sseFilter = useCallback((event: TraceEventDto) => {
    if (event.type === 'trace_start') {
      const context = event.payload.context as { extra?: { client_key?: string } } | undefined
      const key = context?.extra?.client_key
      if (key && keysRef.current.has(key)) {
        const model = keyToModel.current.get(key)
        if (model) traceToModel.current.set(event.trace_id, model)
        return true
      }
      return false
    }
    return traceToModel.current.has(event.trace_id)
  }, [])
  const sse = useSse(sseFilter)

  // 把增量事件折叠进对应模型的 run
  useEffect(() => {
    if (processed.current >= sse.events.length) return
    const fresh = sse.events.slice(processed.current)
    processed.current = sse.events.length
    if (!fresh.length) return
    setRuns((prev) => {
      const next = { ...prev }
      for (const event of fresh) {
        const model = traceToModel.current.get(event.trace_id)
        if (!model) continue
        const run = next[model]
        if (!run) continue
        const payload = event.payload
        if (event.type === 'trace_start') {
          next[model] = { ...run, status: 'streaming', startedAt: Date.now() }
        } else if (event.type === 'text_delta') {
          next[model] = { ...run, text: run.text + String(payload.text ?? '') }
        } else if (event.type === 'thinking_delta') {
          next[model] = { ...run, thinking: run.thinking + String(payload.text ?? '') }
        } else if (event.type === 'first_token') {
          next[model] = { ...run, firstTokenMs: Number(payload.ttft_ms ?? 0) || null }
        } else if (event.type === 'tool_call_delta') {
          const idx = Number(payload.idx ?? 0)
          const name = String(payload.name_fragment ?? '')
          const args = payload.args_fragment
          const list = [...run.toolCalls]
          const existing = list.findIndex((c) => c.idx === idx)
          const merged = {
            idx,
            name: existing >= 0 ? list[existing].name + name : name,
            args:
              existing >= 0
                ? list[existing].args + (typeof args === 'string' ? args : args ? JSON.stringify(args) : '')
                : typeof args === 'string'
                  ? args
                  : args
                    ? JSON.stringify(args)
                    : '',
          }
          if (existing >= 0) list[existing] = merged
          else list.push(merged)
          next[model] = { ...run, toolCalls: list }
        }
      }
      return next
    })
  }, [sse.events])

  const toggle = (name: string) =>
    setSelected((prev) => (prev.includes(name) ? prev.filter((n) => n !== name) : [...prev, name]))

  const send = async () => {
    if (!selected.length || !prompt.trim()) return
    setBusy(true)
    const batch: Record<string, Run> = {}
    for (const model of selected) {
      const key = newKey()
      keysRef.current.add(key)
      keyToModel.current.set(key, model)
      batch[model] = {
        clientKey: key, model, status: 'queued', text: '', thinking: '', toolCalls: [],
        firstTokenMs: null, result: null, error: null, startedAt: Date.now(),
      }
    }
    setRuns(batch)
    processed.current = sse.events.length

    await Promise.allSettled(
      selected.map(async (model) => {
        const key = batch[model].clientKey
        try {
          const result = await api.chat({
            model, prompt, system, max_tokens: maxTokens, temperature,
            thinking, stream: true, tools: selectedTools, client_key: key,
          })
          setRuns((prev) => ({
            ...prev,
            [model]: { ...prev[model], status: 'done', result },
          }))
        } catch (err) {
          const e = err as { message?: string; code?: string }
          setRuns((prev) => ({
            ...prev,
            [model]: {
              ...prev[model], status: 'error',
              error: `${e?.message ?? String(err)}${e?.code ? ` (${e.code})` : ''}`,
            },
          }))
        }
      }),
    )
    setBusy(false)
  }

  if (models.error && !models.data) return <ErrorState error={models.error} />

  const available = models.data ?? []
  // 只在首次拿到模型列表时给一个默认选择；之后完全尊重用户的勾选。
  // 曾经写成 `effective = selected.length ? selected : available.slice(0,1)`，
  // 结果是"取消勾选最后一个模型"会被立刻复活成默认模型，选择状态像幽灵一样。
  useEffect(() => {
    if (!initialized.current && available.length) {
      initialized.current = true
      setSelected([available[0].name])
    }
  }, [available])
  const effective = selected

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <Panel title="请求" note="并排请求在服务端按 GPU 锁串行执行">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
          <div className="row-wrap">
            <span className="field-label">模型（可多选并排）</span>
            {available.map((m) => (
              <label key={m.id} className="row small" style={{ cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={effective.includes(m.name)}
                  onChange={() => toggle(m.name)}
                />
                <span className="mono">{m.name}</span>
                <span className="muted">{m.parameter_size}</span>
                {m.calibrated ? <span className="badge badge-ok" title="已标定 token 密度">cal</span> : null}
              </label>
            ))}
          </div>

          <label className="field">
            <span className="field-label">system</span>
            <input className="input" value={system} placeholder="（可选）"
              onChange={(e) => setSystem(e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">prompt</span>
            <textarea className="input" rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          </label>

          <div className="row-wrap">
            <label className="field">
              <span className="field-label">max_tokens</span>
              <input className="input" type="number" min={1} max={8192} style={{ width: 90 }}
                value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
            </label>
            <label className="field">
              <span className="field-label">temperature</span>
              <input className="input" type="number" step={0.1} min={0} max={2} style={{ width: 80 }}
                value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
            </label>
            <label className="row small" style={{ cursor: 'pointer' }}>
              <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} />
              thinking
            </label>
            {(tools.data?.tools ?? []).map((tool) => (
              <label key={tool.key} className="row small" style={{ cursor: 'pointer' }} title={tool.description}>
                <input
                  type="checkbox"
                  checked={selectedTools.includes(tool.key)}
                  onChange={() =>
                    setSelectedTools((prev) =>
                      prev.includes(tool.key) ? prev.filter((t) => t !== tool.key) : [...prev, tool.key],
                    )
                  }
                />
                <span className="mono">{tool.name}</span>
              </label>
            ))}
            <span className="panel-head-spacer" />
            <span className="small muted">
              SSE {sse.connected ? <span className="badge badge-ok">● 已连接</span> : <span className="badge badge-warn">○ 未连接</span>}
            </span>
            <button
              className="btn btn-primary"
              onClick={send}
              disabled={busy || !prompt.trim() || !effective.length}
              title={!effective.length ? '请先勾选至少一个模型' : undefined}
            >
              {busy ? '运行中…' : '发送'}
            </button>
          </div>
          {thinking ? (
            <p className="small muted">
              提示：P5/P12 —— 推理模型开 thinking 时可能把整个 max_tokens 吃光导致正文为空；
              这时看到 <code>EMPTY_CONTENT_WITH_THINKING</code> 不是模型答错，是没预算答。
            </p>
          ) : null}
        </div>
      </Panel>

      <div className="grid">
        {effective.map((model) => {
          const run = runs[model]
          return (
            <div key={model} className={effective.length > 1 ? 'col-6' : 'col-12'}>
              <RunPanel model={model} run={run} />
            </div>
          )
        })}
      </div>
    </div>
  )
}

function RunPanel({ model, run }: { model: string; run?: Run }) {
  if (!run) {
    return (
      <Panel title={<span className="mono">{model}</span>} note="尚未运行">
        <div className="empty">
          <div className="empty-title">点「发送」开始</div>
        </div>
      </Panel>
    )
  }
  const result = run.result
  const usage = result?.usage ?? null
  const latency = result?.latency ?? null

  return (
    <Panel
      title={<span className="mono">{model}</span>}
      note={
        run.status === 'queued'
          ? '排队中（GPU 独占锁）'
          : run.status === 'streaming'
            ? '生成中…'
            : result
              ? `trace ${result.trace_id.slice(-10)}`
              : run.status === 'error'
                ? '失败'
                : ''
      }
      actions={
        result ? (
          <button className="btn" onClick={() => navigate(`/traces/${result.trace_id}`)}>
            查看 trace →
          </button>
        ) : null
      }
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
        {run.status === 'error' ? (
          <div className="badge badge-error" style={{ padding: 'var(--space-2)' }}>✕ {run.error}</div>
        ) : null}

        {/* thinking 分栏：推理内容混进正文会破坏工具 JSON 解析，必须分开显示 */}
        {run.thinking ? (
          <div>
            <div className="field-label">thinking（{run.thinking.length} 字符）</div>
            <div className="code" style={{ maxHeight: 160, opacity: 0.85 }}>{run.thinking}</div>
          </div>
        ) : null}

        <div>
          <div className="field-label">输出</div>
          {run.text || run.status === 'streaming' || run.status === 'queued' ? (
            <div className="code" style={{ minHeight: 60 }}>{run.text || '…'}</div>
          ) : (
            <div className="code" style={{ minHeight: 60 }}>
              {result?.text || <span className="muted">（正文为空）</span>}
            </div>
          )}
        </div>

        {run.toolCalls.length || result?.tool_calls.length ? (
          <div>
            <div className="field-label">工具调用</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {(result?.tool_calls.length ? result.tool_calls : []).map((call) => (
                <div key={call.id} className="row small">
                  <span className={`badge badge-${call.parse_status === 'ok' ? 'ok' : 'warn'}`}>
                    {call.parse_status}
                  </span>
                  <b className="mono">{call.name}</b>
                  <code className="muted">{JSON.stringify(call.args ?? call.args_raw)}</code>
                </div>
              ))}
              {!result?.tool_calls.length
                ? run.toolCalls.map((call) => (
                    <div key={call.idx} className="row small">
                      <span className="badge badge-neutral">streaming</span>
                      <b className="mono">{call.name || '…'}</b>
                      <code className="muted">{call.args || '…'}</code>
                    </div>
                  ))
                : null}
            </div>
          </div>
        ) : null}

        {result ? (
          <>
            <div className="grid">
              <div className="col-6">
                <StatCard
                  label="输入 / 输出 token"
                  value={`${fmtInt(usage?.in_tokens ?? null)} / ${fmtInt(usage?.out_tokens ?? null)}`}
                  unknown={!usage}
                  badge={<SourceBadge source={usage?.source ?? null} confidence={usage?.confidence ?? null} note={usage?.note} />}
                />
              </div>
              <div className="col-6">
                <StatCard
                  label="TTFT / decode"
                  value={`${latency?.ttft_ms != null ? fmtMs(latency.ttft_ms) : UNKNOWN} / ${fmtFloat(latency?.decode_tps ?? null)} t/s`}
                  badge={latency ? <PrefillTag mode={latency.prefill_mode} msPerToken={latency.prefill_ms_per_token} /> : undefined}
                  sub={<>wall {fmtMs(latency?.wall_ms ?? null)} · finish {result.finish_reason}</>}
                />
              </div>
            </div>

            {result.parts.length ? (
              <div>
                <div className="field-label">本次 prompt 分段</div>
                <PromptBreakdown parts={result.parts} engineIn={usage?.in_tokens ?? null} showOutput />
              </div>
            ) : null}

            {result.anomalies.length ? (
              <div className="chips">
                {result.anomalies.map((anomaly) => (
                  <AnomalyChip key={anomaly.id} anomaly={anomaly} />
                ))}
              </div>
            ) : null}

            <div className="row small muted">
              <StatusBadge status={result.trace_id ? 'ok' : 'error'} />
              <span>已落库，点右上角可查看完整证据链</span>
            </div>
          </>
        ) : null}
      </div>
    </Panel>
  )
}

