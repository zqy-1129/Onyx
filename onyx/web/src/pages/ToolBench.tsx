/** Tool Bench（S25）：注册表、契约审计、上下文开销、执行器契约矩阵、运行历史。
 *
 * 这一页存在的理由是把"这套工具能不能信"变成不用开终端就能回答的问题。
 * 三条口径：
 * - **矩阵的"不适用"与"未知"都不许显示成 ✓**。装不上的执行器（没装 httpx）是未知，
 *   mock 列结构上无法证明零真实调用是 n/a——把它们画成通过就等于把缺依赖写成质量保证。
 * - **开销要分两笔报**：JSON 本身 + 模板脚手架。P17 实测大头来自模板注入的说明文本，
 *   只报 JSON 会把优化方向引到"精简描述"上去。占比问不出来就显示「—」，不是 0%。
 * - **每个 run 结果都要点得回它那次真实请求**（trace_id），否则"工具跑成什么样"又是一句无据之言。
 */
import { useState } from 'react'
import { api } from '../api/client'
import type {
  ToolAuditFinding, ToolAuditView, ToolCostView, ToolDefView, ToolMatrixView, ToolRunView,
} from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { EmptyState, ErrorState, Panel, Skeleton, StatusBadge } from '../components/primitives'
import { fmtClock, fmtInt, shortId, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

export type CellState = { symbol: string; tone: string; title: string }

/** 矩阵一格的三态。`undefined` = 这一列根本没跑（未知），绝不是通过。 */
export function matrixCell(
  cell: { passed: boolean; applicable: boolean; detail: string } | undefined,
  column: string,
): CellState {
  if (!cell) {
    return { symbol: '?', tone: 'unknown', title: `${column}：这一列没有结果（未知，不是通过）` }
  }
  if (cell.passed) return { symbol: '✓', tone: 'ok', title: cell.detail || `${column} 通过` }
  if (!cell.applicable) {
    return { symbol: 'n/a', tone: 'unknown', title: `不适用：${cell.detail}` }
  }
  return { symbol: '✗', tone: 'error', title: `失败：${cell.detail}` }
}

/** 开销的两笔账 + 占比。没传模板开销时占比是"问不出来"，不是 0%。 */
export function costLines(cost: ToolCostView | null): Array<{ label: string; value: string; hint: string }> {
  if (!cost) return []
  const share = cost.template_share === null || cost.template_share === undefined
    ? UNKNOWN
    : `${(cost.template_share * 100).toFixed(1)}%`
  return [
    { label: 'JSON 本身', value: `${fmtInt(cost.json_tokens)} tok · ${fmtInt(cost.json_bytes)} B`,
      hint: '工具定义序列化进请求的那部分' },
    { label: '模板脚手架', value: cost.template_overhead_tokens
        ? `${fmtInt(cost.template_overhead_tokens)} tok` : UNKNOWN,
      hint: '取 trace 归因的 template_ctl 传进 --overhead / 表单；P17：实测这才是大头' },
    { label: '每次请求实付', value: `${fmtInt(cost.effective_tokens)} tok`,
      hint: 'JSON + 模板脚手架' },
    { label: '模板占比', value: share,
      hint: cost.template_share === null || cost.template_share === undefined
        ? '没传模板开销 ⇒ 问不出来，不是 0%'
        : '占比高说明该换模板/换模型，而不是精简描述' },
    { label: '计数档位', value: cost.count_source || UNKNOWN,
      hint: cost.hint || '按指定模型的标定档位核算' },
  ]
}

export function severityTone(severity: string): string {
  return severity === 'error' ? 'error' : severity === 'warn' ? 'warn' : 'info'
}

/** 只有 error 才算"这库不能直接用"；warn/info 是待办，不是否决 */
export function auditVerdict(counts: Record<string, number> | undefined): { tone: string; text: string } {
  const error = counts?.error ?? 0
  const warn = counts?.warn ?? 0
  if (error) return { tone: 'error', text: `${error} 条 error 级问题` }
  if (warn) return { tone: 'warn', text: `${warn} 条 warn 级问题` }
  return { tone: 'ok', text: '没有 error / warn 级问题' }
}

const defColumns: Array<Column<ToolDefView>> = [
  { key: 'name', header: '工具', mono: true, render: (t) => t.name, sortValue: (t) => t.name },
  { key: 'v', header: '版本', align: 'right', mono: true, render: (t) => `v${t.version}`, sortValue: (t) => t.version },
  { key: 'kind', header: '类型', render: (t) => t.kind, sortValue: (t) => t.kind },
  {
    key: 'side', header: '副作用',
    render: (t) => (
      <span className={`badge badge-${t.side_effect === 'read' ? 'ok' : t.side_effect ? 'warn' : 'unknown'}`}>
        {t.side_effect || '未标注'}
      </span>
    ),
    sortValue: (t) => t.side_effect,
  },
  {
    key: 'tokens', header: 'tokens', align: 'right', mono: true,
    // null 是"还没核算过"，画成 0 会让人以为这个工具免费
    render: (t) => (t.tokens === null ? <span className="cell-dim">{UNKNOWN}</span> : fmtInt(t.tokens)),
    sortValue: (t) => t.tokens ?? -1,
  },
  {
    key: 'bytes', header: 'bytes', align: 'right', mono: true,
    render: (t) => (t.bytes === null ? <span className="cell-dim">{UNKNOWN}</span> : fmtInt(t.bytes)),
    sortValue: (t) => t.bytes ?? -1,
  },
  { key: 'ex', header: '样本', align: 'right', mono: true, render: (t) => fmtInt(t.n_examples) },
  {
    key: 'enabled', header: '启用',
    render: (t) => (t.enabled ? <span className="badge badge-ok">✓</span> : <span className="badge badge-neutral">✗</span>),
    sortValue: (t) => (t.enabled ? 1 : 0),
  },
  {
    key: 'hash', header: 'hash', mono: true,
    render: (t) => <span className="cell-dim small" title={t.hash}>{shortId(t.hash, 10)}</span>,
  },
]

const findingColumns: Array<Column<ToolAuditFinding>> = [
  { key: 'tool', header: '工具', mono: true, render: (f) => f.tool, sortValue: (f) => f.tool },
  {
    key: 'rule', header: '规则', mono: true,
    render: (f) => <span className="badge badge-neutral" title={f.meaning}>{f.rule}</span>,
    sortValue: (f) => f.rule,
  },
  {
    key: 'sev', header: '级别',
    render: (f) => <span className={`badge badge-${severityTone(f.severity)}`}>{f.severity}</span>,
    sortValue: (f) => f.severity,
  },
  { key: 'msg', header: '问题', render: (f) => <span className="small">{f.message}</span> },
  {
    key: 'path', header: '位置', mono: true,
    render: (f) => <span className="cell-dim small">{f.path || UNKNOWN}</span>,
  },
  { key: 'fix', header: '修法', render: (f) => <span className="small">{f.fix || UNKNOWN}</span> },
]

const runColumns: Array<Column<ToolRunView>> = [
  { key: 'id', header: 'run', mono: true, render: (r) => shortId(r.id, 10) },
  { key: 'status', header: '状态', render: (r) => <StatusBadge status={r.status} /> },
  {
    key: 'tool', header: '工具', mono: true,
    // 工具名是人起的短词，不是可截断的 id：按 hash 那样截尾会把 "calculator" 显示成 "lculator"
    render: (r) => r.tool_id ?? UNKNOWN,
  },
  {
    key: 'def', header: '定义 hash', mono: true,
    render: (r) => <span className="cell-dim small" title={r.tool_def_hash ?? ''}>{shortId(r.tool_def_hash ?? '', 8)}</span>,
  },
  {
    key: 'lat', header: '延迟', align: 'right', mono: true,
    render: (r) => (r.latency_ms === null ? UNKNOWN : `${r.latency_ms.toFixed(1)}ms`),
    sortValue: (r) => r.latency_ms ?? -1,
  },
  {
    key: 'det', header: '确定性',
    render: (r) => (r.deterministic === null
      ? <span className="cell-dim">{UNKNOWN}</span>
      : <span className={`badge badge-${r.deterministic ? 'ok' : 'warn'}`}>{r.deterministic ? '✓' : '✗'}</span>),
  },
  {
    key: 'idem', header: '幂等',
    render: (r) => (r.idempotent === null
      ? <span className="cell-dim">{UNKNOWN}</span>
      : <span className={`badge badge-${r.idempotent ? 'ok' : 'warn'}`}>{r.idempotent ? '✓' : '✗'}</span>),
  },
  {
    key: 'err', header: '说明',
    render: (r) => (r.error ? <span className="small" title={r.error}>{r.error.slice(0, 52)}</span> : null),
  },
  {
    key: 'trace', header: 'trace',
    render: (r) => (r.trace_id ? (
      <button className="linklike mono" onClick={() => navigate(`/traces/${r.trace_id}`)}
        title="这次调用真正发生过">
        {shortId(r.trace_id, 10)}
      </button>
    ) : <span className="cell-dim">{UNKNOWN}</span>),
  },
  { key: 'at', header: '时间', mono: true, render: (r) => fmtClock(r.started_at), sortValue: (r) => r.started_at },
]

function MatrixPanel() {
  const [tool, setTool] = useState('echo')
  const [matrix, setMatrix] = useState<ToolMatrixView | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)

  const run = async () => {
    setBusy(true)
    setError(null)
    try {
      setMatrix(await api.toolMatrix({ tool }))
    } catch (err) {
      setError(err)
    } finally {
      setBusy(false)
    }
  }

  const columns = matrix ? Object.keys(matrix.executors) : []

  return (
    <Panel
      title="执行器契约矩阵"
      note="同一套断言跑在所有已实现的执行器上；n/a 与 ? 都不算通过"
      actions={(
        <span className="row">
          <input className="input" style={{ width: 130 }} value={tool}
            onChange={(e) => setTool(e.target.value)} placeholder="样本工具" />
          <button className="btn" onClick={run} disabled={busy}>{busy ? '跑…' : '跑一次'}</button>
        </span>
      )}
      flush
    >
      {error ? <ErrorState error={error} /> : null}
      {!matrix && !error ? (
        <EmptyState
          title="还没跑过"
          hint="这一格会真的跑一遍各执行器（http 用 MockTransport、mcp 用假连接、"
          >
          <div className="empty-hint">
            mcp_stdio 起一个真子进程走真管道，所以是秒级；全程零真实网络，但仍由人点一次，不轮询。
          </div>
        </EmptyState>
      ) : null}
      {matrix ? (
        <>
          <div className="row-wrap small" style={{ marginBottom: 'var(--space-2)' }}>
            <span className="cell-dim">样本：</span>
            {columns.map((name) => (
              <span key={name} className="badge badge-neutral" title={matrix.sample_notes[name] || ''}>
                {name} → {matrix.samples[name]}
              </span>
            ))}
            <span className="cell-dim">出处 {matrix.source} · 参数 {JSON.stringify(matrix.valid_args)}</span>
          </div>
          <table className="data matrix">
            <thead>
              <tr>
                <th>断言</th>
                {columns.map((name) => <th key={name}>{name}</th>)}
                {Object.keys(matrix.unavailable).map((name) => <th key={name}>{name}</th>)}
                {Object.keys(matrix.pending).map((name) => <th key={name}>{name}</th>)}
              </tr>
            </thead>
            <tbody>
              {matrix.assertions.map((assertion) => (
                <tr key={assertion}>
                  <td className="mono">{assertion}</td>
                  {columns.map((name) => {
                    const cellState = matrixCell(matrix.executors[name]?.[assertion], name)
                    return (
                      <td key={name} title={cellState.title} style={{ textAlign: 'center' }}>
                        <span className={`badge badge-${cellState.tone}`}>{cellState.symbol}</span>
                      </td>
                    )
                  })}
                  {Object.entries(matrix.unavailable).map(([name, reason]) => (
                    <td key={name} title={reason} style={{ textAlign: 'center' }}>
                      <span className="badge badge-unknown">?</span>
                    </td>
                  ))}
                  {Object.entries(matrix.pending).map(([name, milestone]) => (
                    <td key={name} title={`未实现，计划在 ${milestone}`} style={{ textAlign: 'center' }}>
                      <span className="cell-dim">—</span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <div className="row-wrap small" style={{ marginTop: 'var(--space-2)' }}>
            {columns.map((name) => {
              const counts = matrix.summary[name]
              return (
                <span key={name} className={`badge badge-${counts.failed ? 'error' : 'ok'}`}>
                  {name} 通过 {counts.passed} · 失败 {counts.failed} · 不适用 {counts.not_applicable}
                </span>
              )
            })}
            {Object.entries(matrix.unavailable).map(([name, reason]) => (
              <span key={name} className="badge badge-unknown" title={reason}>{name} 未知</span>
            ))}
            {Object.entries(matrix.pending).map(([name, milestone]) => (
              <span key={name} className="badge badge-neutral" title={milestone}>{name} 未实现</span>
            ))}
          </div>
        </>
      ) : null}
    </Panel>
  )
}

export function ToolBenchPage() {
  const [model, setModel] = useState('')
  const [overhead, setOverhead] = useState('')
  const defs = useApi<ToolDefView[]>(() => api.toolDefs(), { intervalMs: 60_000 })
  const audit = useApi<ToolAuditView>(() => api.toolAudit(model || null), { deps: [model], intervalMs: 120_000 })
  const cost = useApi<ToolCostView>(() => api.toolCost({
    model: model || null, overhead: Number(overhead || 0) || 0,
  }), { deps: [model, overhead], intervalMs: 120_000 })
  const runs = useApi<ToolRunView[]>(() => api.toolRuns({ limit: 50 }), { intervalMs: 30_000 })

  const verdict = auditVerdict(audit.data?.counts)
  const lines = costLines(cost.data)

  return (
    <>
      <div className="grid">
        <div className="col-8">
          <Panel
            title="工具注册表"
            note="内容 hash 版本化；模型实际看到的是启用中的那一份"
            actions={(
              <span className="row">
                <input className="input" style={{ width: 150 }} value={model} placeholder="模型（开销按它核算）"
                  onChange={(e) => setModel(e.target.value)} />
                <input className="input" style={{ width: 110 }} value={overhead} placeholder="模板开销 token"
                  onChange={(e) => setOverhead(e.target.value)} inputMode="numeric" />
              </span>
            )}
            flush
          >
            {defs.error ? <ErrorState error={defs.error} /> : null}
            {defs.loading && !defs.data ? <Skeleton rows={5} /> : null}
            {defs.data && defs.data.length === 0 ? (
              <EmptyState title="注册表是空的" hint="onyx tools import defs.yaml，或 onyx tools mcp-import" />
            ) : null}
            {defs.data && defs.data.length > 0 ? (
              <DataTable columns={defColumns} rows={defs.data} rowKey={(t) => `${t.name}-${t.version}`} maxHeight="36vh" />
            ) : null}
          </Panel>
        </div>
        <div className="col-4">
          <Panel title="上下文开销" note="P17：大头通常在模板脚手架，不在描述">
            {cost.error ? <ErrorState error={cost.error} /> : null}
            {cost.loading && !cost.data ? <Skeleton rows={5} /> : null}
            {lines.map((line) => (
              <div className="stat-foot" key={line.label} title={line.hint}>
                <span className="cell-dim">{line.label}</span>
                <span className="mono">{line.value}</span>
              </div>
            ))}
            {cost.data?.hint ? <div className="note small">{cost.data.hint}</div> : null}
          </Panel>
        </div>
      </div>

      <Panel
        title="契约审计"
        note="逐条规则列出问题与修法；文案与 CLI 同源"
        actions={<span className={`badge badge-${verdict.tone}`}>{verdict.text}</span>}
        flush
      >
        {audit.error ? <ErrorState error={audit.error} /> : null}
        {audit.loading && !audit.data ? <Skeleton rows={4} /> : null}
        {audit.data && audit.data.findings.length === 0 ? (
          <EmptyState title="没有发现问题" hint={audit.data.note} />
        ) : null}
        {audit.data && audit.data.findings.length > 0 ? (
          <DataTable
            columns={findingColumns} rows={audit.data.findings}
            rowKey={(f) => `${f.tool}-${f.rule}-${f.path}`} maxHeight="32vh"
          />
        ) : null}
      </Panel>

      <MatrixPanel />

      <Panel title="运行历史" note="onyx tools run 的留痕；模型触发的调用在 Traces 里（每条都带 trace_id）" flush>
        {runs.error ? <ErrorState error={runs.error} /> : null}
        {runs.loading && !runs.data ? <Skeleton rows={4} /> : null}
        {runs.data && runs.data.length === 0 ? (
          <EmptyState title="还没有运行记录" hint="onyx tools fire 「把 hello 回显一次」 --model … --provider mock" />
        ) : null}
        {runs.data && runs.data.length > 0 ? (
          <DataTable columns={runColumns} rows={runs.data} rowKey={(r) => r.id} maxHeight="32vh" />
        ) : null}
      </Panel>
    </>
  )
}
