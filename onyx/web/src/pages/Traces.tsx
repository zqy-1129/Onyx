/** Traces 列表：游标分页（id 本身按时间排序，直接走主键索引）。 */
import { useCallback, useState } from 'react'
import { api } from '../api/client'
import type { TracePage, TraceSummary } from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { EmptyState, ErrorState, PrefillTag, SourceBadge, StatusBadge } from '../components/primitives'
import { fmtClock, fmtFloat, fmtInt, shortId, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

const PAGE = 50

const columns: Array<Column<TraceSummary>> = [
  { key: 'id', header: 'trace', mono: true, render: (r) => shortId(r.id, 10), sortValue: (r) => r.id },
  { key: 'time', header: '时间', mono: true, render: (r) => fmtClock(r.started_at), sortValue: (r) => r.started_at },
  { key: 'purpose', header: 'purpose', render: (r) => r.purpose, sortValue: (r) => r.purpose },
  { key: 'model', header: '模型', mono: true, render: (r) => r.model_name ?? UNKNOWN, sortValue: (r) => r.model_name },
  {
    key: 'in',
    header: 'in',
    align: 'right',
    mono: true,
    render: (r) => fmtInt(r.in_tokens),
    sortValue: (r) => r.in_tokens,
  },
  {
    key: 'out',
    header: 'out',
    align: 'right',
    mono: true,
    render: (r) => fmtInt(r.out_tokens),
    sortValue: (r) => r.out_tokens,
  },
  {
    key: 'source',
    header: '出处',
    render: (r) => <SourceBadge source={r.source} confidence={r.confidence} />,
  },
  {
    key: 'prefill',
    header: 'prefill',
    render: (r) => <PrefillTag mode={r.prefill_mode} />,
  },
  {
    key: 'ttft',
    header: 'TTFT',
    align: 'right',
    mono: true,
    // 非流式没有真实 TTFT：显示「—」而不是拿 prompt_eval 冒充
    render: (r) => (r.ttft_ms == null ? <span className="muted">{UNKNOWN}</span> : fmtFloat(r.ttft_ms, 0)),
    sortValue: (r) => r.ttft_ms,
  },
  {
    key: 'tps',
    header: 'decode',
    align: 'right',
    mono: true,
    render: (r) => fmtFloat(r.decode_tps),
    sortValue: (r) => r.decode_tps,
  },
  {
    key: 'tools',
    header: '工具',
    align: 'right',
    mono: true,
    render: (r) => (r.tool_calls ? <b>{r.tool_calls}</b> : <span className="cell-dim">0</span>),
    sortValue: (r) => r.tool_calls,
  },
  {
    key: 'anomalies',
    header: '异常',
    align: 'right',
    mono: true,
    render: (r) =>
      r.anomalies ? <span className="badge badge-warn">! {r.anomalies}</span> : <span className="cell-dim">—</span>,
    sortValue: (r) => r.anomalies,
  },
  { key: 'status', header: '状态', render: (r) => <StatusBadge status={r.status} /> },
]

export function TracesPage() {
  const [cursor, setCursor] = useState<string | null>(null)
  const [purpose, setPurpose] = useState('')
  const [status, setStatus] = useState('')
  const [extra, setExtra] = useState<TraceSummary[]>([])

  const first = useApi<TracePage>(
    () => api.traces({ limit: PAGE, purpose: purpose || null, status: status || null }),
    { deps: [purpose, status] },
  )

  const loadMore = useCallback(async () => {
    if (!first.data?.next_cursor) return
    const page = await api.traces({
      limit: PAGE,
      cursor: first.data.next_cursor,
      purpose: purpose || null,
      status: status || null,
    })
    setExtra((prev) => [...prev, ...page.items])
    setCursor(page.next_cursor)
  }, [first.data?.next_cursor, purpose, status])

  if (first.error && !first.data) return <ErrorState error={first.error} />

  const rows = [...(first.data?.items ?? []), ...extra]
  const nextCursor = cursor ?? first.data?.next_cursor ?? null

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div className="row-wrap">
        <span className="field">
          <span className="field-label">purpose</span>
          <select className="select" value={purpose} onChange={(e) => { setPurpose(e.target.value); setExtra([]); setCursor(null) }}>
            <option value="">全部</option>
            <option value="chat">chat</option>
            <option value="playground">playground</option>
            <option value="probe">probe</option>
            <option value="tool_test">tool_test</option>
            <option value="bench">bench</option>
          </select>
        </span>
        <span className="field">
          <span className="field-label">状态</span>
          <select className="select" value={status} onChange={(e) => { setStatus(e.target.value); setExtra([]); setCursor(null) }}>
            <option value="">全部</option>
            <option value="ok">ok</option>
            <option value="error">error</option>
            <option value="timeout">timeout</option>
          </select>
        </span>
        <span className="panel-head-spacer" />
        <span className="small muted">
          共 {fmtInt(first.data?.total ?? null)} 条 · 已加载 {rows.length}
        </span>
        <button className="btn" onClick={first.refresh}>刷新</button>
      </div>

      <div className="panel">
        <div className="panel-body flush">
          {rows.length ? (
            <DataTable
              columns={columns}
              rows={rows}
              rowKey={(r) => r.id}
              onRowClick={(r) => navigate(`/traces/${r.id}`)}
              maxHeight="calc(100vh - 210px)"
            />
          ) : first.loading ? (
            <div className="panel-body">加载中…</div>
          ) : (
            <EmptyState
              title="还没有 trace"
              hint="onyx chat &quot;你好&quot; -m <模型> --no-thinking   或在 Playground 发一条对话"
            />
          )}
        </div>
      </div>

      {nextCursor ? (
        <div className="row">
          <button className="btn" onClick={loadMore}>加载更多（游标 {shortId(nextCursor, 8)}）</button>
        </div>
      ) : null}
    </div>
  )
}
