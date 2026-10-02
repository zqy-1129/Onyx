/** Trace 详情：证据链的终点。
 *  R7：每个派生值都能回溯到原始 body；归因不闭合时直接说，不掩盖。 */
import { api } from '../api/client'
import type { ToolCallView, TraceDetail as Detail, UsageAlt } from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { PromptBreakdown } from '../components/PromptBreakdown'
import {
  AnomalyChip,
  EmptyState,
  ErrorState,
  Panel,
  PrefillTag,
  Skeleton,
  SourceBadge,
  StatCard,
  StatusBadge,
} from '../components/primitives'
import { fmtBytes, fmtFloat, fmtInt, fmtMs, fmtPct, shortRef, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

const altColumns: Array<Column<UsageAlt>> = [
  { key: 'source', header: '来源', mono: true, render: (r) => r.source, sortValue: (r) => r.source },
  { key: 'in', header: 'in', align: 'right', mono: true, render: (r) => fmtInt(r.in_tokens), sortValue: (r) => r.in_tokens },
  { key: 'out', header: 'out', align: 'right', mono: true, render: (r) => fmtInt(r.out_tokens), sortValue: (r) => r.out_tokens },
  {
    key: 'thinking',
    header: 'thinking',
    align: 'right',
    mono: true,
    render: (r) => fmtInt(r.thinking_tokens),
    sortValue: (r) => r.thinking_tokens,
  },
  {
    key: 'ok',
    header: '可用',
    render: (r) => (r.ok ? <span className="badge badge-ok">✓</span> : <span className="badge badge-error">✕</span>),
  },
  { key: 'confidence', header: '置信', render: (r) => r.confidence ?? UNKNOWN },
  { key: 'note', header: '说明', render: (r) => <span className="cell-dim">{r.note || '—'}</span> },
]

const toolColumns: Array<Column<ToolCallView>> = [
  { key: 'step', header: '#', align: 'right', mono: true, render: (r) => r.step, sortValue: (r) => r.step },
  { key: 'name', header: '工具', mono: true, render: (r) => r.name ?? UNKNOWN },
  {
    key: 'parse',
    header: '解析',
    render: (r) => {
      const tone = r.parse_status === 'ok' ? 'ok' : r.parse_status === 'truncated' ? 'warn' : 'error'
      return <span className={`badge badge-${tone}`}>{r.parse_status}</span>
    },
  },
  {
    key: 'args',
    header: '参数',
    mono: true,
    render: (r) => (
      <span title={r.args_raw ?? ''}>
        {r.args ? JSON.stringify(r.args) : <span className="cell-err">{r.args_raw ?? '—'}</span>}
      </span>
    ),
  },
  {
    key: 'result',
    header: '执行',
    render: (r) => r.result_status ?? <span className="cell-dim">未执行</span>,
  },
  { key: 'latency', header: '耗时', align: 'right', mono: true, render: (r) => fmtMs(r.latency_ms), sortValue: (r) => r.latency_ms },
]

function Timeline({ detail }: { detail: Detail }) {
  const engine = detail.engine_latency ?? {}
  const load = Number(engine.load ?? 0) / 1e6
  const prompt = Number(engine.prompt_eval ?? 0) / 1e6
  const decode = Number(engine.eval ?? 0) / 1e6
  const total = load + prompt + decode
  if (!total) return <EmptyState title="引擎未返回分段时序" hint="该通道可能不报 *_duration 字段" />
  const seg = (label: string, ms: number, cls: string) =>
    ms > 0 ? (
      <div className={`tl-seg ${cls}`} style={{ width: `${(ms / total) * 100}%` }} title={`${label}: ${fmtMs(ms)}`}>
        {(ms / total) * 100 > 12 ? `${label} ${fmtMs(ms)}` : ''}
      </div>
    ) : null
  return (
    <div>
      <div className="timeline">
        {seg('load', load, 'tl-load')}
        {seg('prompt_eval', prompt, 'tl-prompt')}
        {seg('decode', decode, 'tl-decode')}
      </div>
      <div className="legend">
        <span className="legend-item"><span className="legend-swatch tl-load" />load <b className="num">{fmtMs(load)}</b></span>
        <span className="legend-item"><span className="legend-swatch tl-prompt" />prompt_eval <b className="num">{fmtMs(prompt)}</b></span>
        <span className="legend-item"><span className="legend-swatch tl-decode" />decode <b className="num">{fmtMs(decode)}</b></span>
        <span className="legend-item">合计 <b className="num">{fmtMs(total)}</b></span>
        {detail.latency.cold_load ? <span className="badge badge-info">❄ 含冷启动载入</span> : null}
      </div>
    </div>
  )
}

export function TraceDetailPage({ traceId }: { traceId: string }) {
  const state = useApi<Detail>(() => api.trace(traceId), { deps: [traceId] })

  if (state.error) return <ErrorState error={state.error} />
  if (!state.data) return <Skeleton rows={8} />

  const detail = state.data
  const { trace, usage, latency } = detail
  const attribution = detail.attribution as Record<string, unknown>

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div className="row-wrap">
        <button className="btn" onClick={() => navigate('/traces')}>← 列表</button>
        <b className="mono">{trace.id}</b>
        <StatusBadge status={trace.status} />
        <span className="badge badge-neutral">{trace.purpose}</span>
        <span className="mono small muted">{trace.model_name ?? UNKNOWN}</span>
        <span className="small muted">{trace.started_at}</span>
        <span className="panel-head-spacer" />
        <button className="btn" onClick={state.refresh}>刷新</button>
      </div>

      <div className="grid">
        <div className="col-3">
          <StatCard
            label="输入 token"
            value={fmtInt(usage?.in_tokens ?? null)}
            unknown={usage?.in_tokens == null}
            badge={<SourceBadge source={usage?.source ?? null} confidence={usage?.confidence ?? null} note={usage?.note} />}
          />
        </div>
        <div className="col-3">
          <StatCard
            label="输出 token（含 thinking）"
            value={fmtInt(usage?.out_tokens ?? null)}
            unknown={usage?.out_tokens == null}
            sub={
              usage?.thinking_tokens ? (
                <span>其中推理 {fmtInt(usage.thinking_tokens)}（P12）</span>
              ) : (
                'P12：thinking 计入 eval_count'
              )
            }
          />
        </div>
        <div className="col-3">
          <StatCard
            label="TTFT"
            value={fmtMs(latency.ttft_ms)}
            unknown={latency.ttft_ms == null}
            sub={latency.ttft_ms == null ? '非流式无真实 TTFT，不伪造' : undefined}
            badge={<PrefillTag mode={latency.prefill_mode} msPerToken={latency.prefill_ms_per_token} />}
          />
        </div>
        <div className="col-3">
          <StatCard
            label="decode 吞吐"
            value={fmtFloat(latency.decode_tps)}
            unit="t/s"
            unknown={latency.decode_tps == null}
            sub={
              <>
                prefill {fmtFloat(latency.prefill_tps)} t/s · wall {fmtMs(latency.wall_ms)}
              </>
            }
          />
        </div>
      </div>

      <div className="grid">
        <div className="col-7">
          <Panel title="时间轴" note="引擎自报分段耗时（纳秒）">
            <Timeline detail={detail} />
          </Panel>
        </div>
        <div className="col-5">
          <Panel
            title="分段归因"
            note={attribution.count_source ? `count_source=${String(attribution.count_source)}` : 'Σ分段 + template_ctl = 引擎计数'}
          >
            <PromptBreakdown parts={detail.parts} engineIn={usage?.in_tokens ?? null} />
            {attribution.clamped ? (
              <p className="small" style={{ color: 'var(--warn)', marginTop: 'var(--space-2)' }}>
                ! 分段和超过引擎计数（残差 {String(attribution.residual_raw ?? UNKNOWN)}）：
                计数档位高估或引擎发生截断，归因不可信。
              </p>
            ) : null}
            {detail.gpu?.size_vram != null ? (
              <p className="small muted" style={{ marginTop: 'var(--space-2)' }}>
                GPU 快照：显存 {fmtBytes(Number(detail.gpu.size_vram))} / 权重 {fmtBytes(Number(detail.gpu.size ?? 0))} ·
                载入 ctx {String(detail.gpu.context_length ?? UNKNOWN)} ·
                占用 {usage?.in_tokens && detail.gpu.context_length
                  ? fmtPct(usage.in_tokens / Number(detail.gpu.context_length))
                  : UNKNOWN}
              </p>
            ) : null}
          </Panel>
        </div>
      </div>

      <Panel title="各来源计数对账" note="compat 永不采信，仅作交叉验证（P14）" flush>
        {detail.alts.length ? (
          <DataTable columns={altColumns} rows={detail.alts} rowKey={(r) => r.source} />
        ) : (
          <EmptyState title="没有多来源记录" />
        )}
      </Panel>

      {detail.tool_calls.length ? (
        <Panel title={`工具调用（${detail.tool_calls.length}）`} note="畸形参数保留原文，这是唯一诊断证据" flush>
          <DataTable columns={toolColumns} rows={detail.tool_calls} rowKey={(r) => r.id} />
        </Panel>
      ) : null}

      {detail.anomalies.length ? (
        <Panel title={`异常（${detail.anomalies.length}）`}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
            {detail.anomalies.map((anomaly) => (
              <div key={anomaly.id} className="row-wrap">
                <AnomalyChip anomaly={anomaly} />
                <span className="small">{anomaly.meaning}</span>
                {anomaly.action ? <span className="small muted">→ {anomaly.action}</span> : null}
                {Object.keys(anomaly.detail).length ? (
                  <code className="small muted">{JSON.stringify(anomaly.detail)}</code>
                ) : null}
              </div>
            ))}
          </div>
        </Panel>
      ) : null}

      <div className="grid">
        <div className="col-6">
          <Panel title="请求消息" note={shortRef(detail.refs.messages)} flush>
            <div className="code">{JSON.stringify(detail.messages, null, 2)}</div>
          </Panel>
        </div>
        <div className="col-6">
          <Panel title="模型输出" note={shortRef(detail.refs.output)} flush>
            <div className="code">{JSON.stringify(detail.output, null, 2)}</div>
          </Panel>
        </div>
      </div>

      <Panel title="参数快照" note="换参数后的分数差异必须可解释">
        <div className="code">{JSON.stringify(detail.params, null, 2)}</div>
      </Panel>
    </div>
  )
}
