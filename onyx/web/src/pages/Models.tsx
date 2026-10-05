/** Models：模型资产与三态能力位。
 *  ✗（确认不支持）与 ?（未实测）必须视觉可区分——前者评测 skip，后者要先跑探针。 */
import { api } from '../api/client'
import type { CapReportDto, ModelView } from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { CapSymbol, ErrorState, Panel, Skeleton } from '../components/primitives'
import { fmtFloat, fmtGb, fmtInt, fmtPct, residencyOf, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { ModelGovernancePanel } from './ModelGovernance'

const CAP_COLUMNS: Array<{ key: string; cap: string; label: string }> = [
  { key: 'tools', cap: 'tools', label: 'tools' },
  { key: 'tool_choice', cap: 'tool_choice', label: 'tool_ch' },
  { key: 'thinking', cap: 'thinking', label: 'think' },
  { key: 'structured_output', cap: 'structured_output', label: 'struct' },
  { key: 'stream_usage', cap: 'stream_usage', label: 'stream' },
  { key: 'vision', cap: 'vision', label: 'vision' },
  { key: 'embed', cap: 'embed', label: 'embed' },
  { key: 'n_sampling', cap: 'n_sampling', label: 'n>1' },
]

function capState(caps: CapReportDto | undefined, cap: string): { state: string; reason: string } {
  if (!caps) return { state: 'unknown', reason: '无能力报告' }
  if (caps.confirmed?.includes(cap)) return { state: 'confirmed', reason: caps.reasons?.[cap] ?? '' }
  if (caps.missing?.includes(cap)) return { state: 'missing', reason: caps.reasons?.[cap] ?? '' }
  return { state: 'unknown', reason: caps.reasons?.[cap] ?? '' }
}

const columns: Array<Column<ModelView>> = [
  {
    key: 'name',
    header: '模型',
    mono: true,
    sortValue: (r) => r.name,
    render: (r) => {
      const res = residencyOf(r.loaded)
      return (
        <span>
          {/* 三态必须可区分：已载入 / 未载入 / 该通道不报告（未知）。
              把"未知"画成"未载入"会引着人去查一个不存在的问题（R2） */}
          {res.marker === 'loaded' ? (
            <span className="status-dot status-ok" title={res.title} />
          ) : (
            <span className="muted" title={res.title}>{res.text}</span>
          )}{' '}
          {r.name}
        </span>
      )
    },
  },
  { key: 'params', header: '参数', align: 'right', render: (r) => r.parameter_size || UNKNOWN, sortValue: (r) => r.parameter_size },
  { key: 'quant', header: '量化', render: (r) => r.quantization || UNKNOWN },
  {
    key: 'size',
    header: '磁盘',
    align: 'right',
    mono: true,
    // 兼容通道不报体积 ⇒ 「—」，不是 0.00GB（0 会被当成一个测量值）
    render: (r) => fmtGb(r.size_gb),
    sortValue: (r) => r.size_gb ?? -1,
  },
  ...CAP_COLUMNS.map((entry): Column<ModelView> => ({
    key: entry.key,
    header: entry.label,
    align: 'right',
    render: (r) => {
      const { state, reason } = capState(r.caps, entry.cap)
      return <CapSymbol state={state} cap={entry.cap} reason={reason} />
    },
  })),
  { key: 'tool_format', header: 'tool_format', render: (r) => (r.probed ? r.tool_format : <span className="cell-dim">{r.tool_format}</span>) },
  {
    // P3：载入上下文 4096 与训练上下文 262144 差 64 倍，必须并列显示
    key: 'ctx',
    header: 'ctx 载入/训练',
    align: 'right',
    mono: true,
    render: (r) => `${r.ctx_loaded ?? UNKNOWN} / ${r.ctx_train ?? UNKNOWN}`,
    sortValue: (r) => r.ctx_loaded,
  },
  {
    key: 'calib',
    header: '标定',
    align: 'right',
    render: (r) =>
      r.calibrated ? (
        <span title={`R²=${fmtFloat(Number(r.calibration.r2), 4)} 最大相对误差=${fmtPct(Number(r.calibration.max_rel_error))} n=${fmtInt(Number(r.calibration.n))}`}>
          <span className="badge badge-ok">✓</span> 中{fmtFloat(Number(r.calibration.cjk_ratio), 3)} /
          其他{fmtFloat(Number(r.calibration.other_ratio), 3)} +{fmtFloat(Number(r.calibration.intercept), 1)}
        </span>
      ) : (
        <span className="badge badge-unknown" title="未标定：fitted 档不可用，只能用 heuristic（low 置信）">
          ? 未标定
        </span>
      ),
  },
]

export function ModelsPage() {
  const state = useApi<ModelView[]>(() => api.models(), { intervalMs: 30_000 })
  if (state.error && !state.data) return <ErrorState error={state.error} />
  if (!state.data) return <Skeleton rows={5} />

  const unprobed = state.data.filter((m) => !m.probed)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <Panel
        title={`模型（${state.data.length}）`}
        note="✓ 确认 · ✗ 不支持（评测 skip）· ? 未实测（≠不支持）"
        flush
        actions={<button className="btn" onClick={state.refresh}>刷新</button>}
      >
        <DataTable columns={columns} rows={state.data} rowKey={(r) => r.id} maxHeight="calc(100vh - 220px)" />
      </Panel>
      <ModelGovernancePanel
        models={state.data}
        providerId={state.data[0]?.provider_id ?? ''}
        onChanged={state.refresh}
      />
      {unprobed.length ? (
        <Panel title="待实测">
          <p className="small muted">
            以下模型还没跑过探针，其 <code>structured_output</code> / <code>stream_usage</code> /{' '}
            <code>tool_format</code> 均为「未实测」而非「不支持」：
          </p>
          <div className="chips mt-3">
            {unprobed.map((m) => (
              <code key={m.id} className="badge badge-neutral">{m.name}</code>
            ))}
          </div>
          <p className="small mt-3">
            执行：<code>onyx probe write-back --model {unprobed[0].name}</code>
          </p>
        </Panel>
      ) : null}
    </div>
  )
}
