/** Fleet 总览：一眼回答"服务活着吗 / 谁占着显存 / 最近一小时什么水平 / 有没有异常 / 通知系统好不好"。 */
import { api } from '../api/client'
import type {
  AlertRuntime, AlertTriggerView, ErrorAnomalySummary, FleetView, LoadedModelView,
} from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import {
  AnomalyChip,
  EmptyState,
  ErrorState,
  Panel,
  PrefillTag,
  Skeleton,
  StatCard,
  StatusDot,
} from '../components/primitives'
import { fmtBytes, fmtClock, fmtCompact, fmtFloat, fmtInt, fmtPct, fmtSeconds, timeAgo, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

const REFRESH_MS = 10_000
/** 触发历史变化慢得多（ cooldown 分钟级），30s 足够；轮询更密只会和评测抢同一个库 */
const ALERTS_REFRESH_MS = 30_000

export interface AnomalyChipRow {
  code: string
  n: number
  severity: string
}

/** chip 的级别来自后端，不自己猜。
 *  同一个码在一小时里同时以两种级别出现过时取更严重的那个——取轻的会把 error 洗白。
 */
export function anomalyChips(anomalies: FleetView['anomalies']): AnomalyChipRow[] {
  const rank = (s: string) => (s === 'error' ? 2 : s === 'warn' ? 1 : 0)
  return Object.entries(anomalies ?? {})
    .map(([code, stat]) => ({
      code,
      n: stat.n,
      severity: [...(stat.severities ?? ['warn'])].sort((a, b) => rank(b) - rank(a))[0],
    }))
    .sort((a, b) => b.n - a.n || a.code.localeCompare(b.code))
}

/** 顶部那一行说什么。`''` = 无话可说（没有 error 级异常）。 */
export function errorBannerText(summary: ErrorAnomalySummary | undefined): string {
  if (!summary || !summary.n) return ''
  const codes = Object.entries(summary.by_code).sort((a, b) => b[1] - a[1])
    .map(([code, n]) => `${code} ${n}`).join(' · ')
  return `近 1 小时有 ${summary.n} 条 error 级异常：${codes}`
}

/** 告警系统自己的状态。三种"不能发"分开说，因为修法不同。 */
export function alertBanner(alerts: AlertRuntime | undefined): { tone: string; text: string } {
  if (!alerts) return { tone: 'banner-warn', text: '告警状态未知（/api/fleet 没带回这一段，别当成"没有告警"）' }
  if (!alerts.enabled) {
    return { tone: 'banner-warn', text: `告警未装配：${alerts.reason || '规则没启用'}` }
  }
  if (alerts.last_error) {
    return { tone: 'banner-error', text: `告警轮询出错：${alerts.last_error}（通知可能没发出去）` }
  }
  if (!alerts.channels.length) {
    return { tone: 'banner-warn', text: '告警已启用但没有出口' }
  }
  return { tone: '', text: `告警在跑 · 出口 ${alerts.channels.join('/')} · 已轮询 ${fmtInt(alerts.ticks)} 次` }
}

const alertColumns: Array<Column<AlertTriggerView>> = [
  { key: 'time', header: '时间', mono: true, sortValue: (r) => r.created_at, render: (r) => fmtClock(r.created_at) },
  { key: 'code', header: '码', mono: true, sortValue: (r) => r.code, render: (r) => r.code },
  {
    key: 'n',
    header: '次数',
    align: 'right',
    sortValue: (r) => r.n_in_window,
    // 命中时窗口里的计数是判据的一部分：只显示"发过"会丢掉"为什么发"
    render: (r) => (r.is_test ? <span className="badge badge-neutral">测试</span> : `${r.n_in_window}/${r.window_s}s`),
  },
  { key: 'channel', header: '出口', sortValue: (r) => r.channel, render: (r) => r.channel },
  {
    key: 'status',
    header: '结果',
    sortValue: (r) => r.status,
    render: (r) => (
      <span title={r.detail}>
        <span className={r.status === 'sent' ? 'badge badge-ok' : 'badge badge-error'}>
          {r.status === 'sent' ? '✓ 已发' : '✗ 失败'}
        </span>
        <span className="cell-dim small">　{r.detail.length > 46 ? `${r.detail.slice(0, 46)}…` : r.detail}</span>
      </span>
    ),
  },
  {
    key: 'trace',
    header: '样本',
    render: (r) => (r.trace_ids.length
      ? (
        <button className="linklike mono" onClick={() => navigate(`/traces/${r.trace_ids[0]}`)}
          title={`这条通知来自 ${r.trace_ids.length} 个样本中的第一个`}>
          {r.trace_ids[0].slice(0, 10)}
        </button>
      )
      : <span className="cell-dim">{UNKNOWN}</span>),
  },
]

const loadedColumns: Array<Column<LoadedModelView>> = [
  { key: 'name', header: '模型', mono: true, sortValue: (r) => r.name, render: (r) => r.name },
  {
    key: 'vram',
    header: '显存',
    align: 'right',
    sortValue: (r) => r.size_vram,
    render: (r) => (
      <span title={`权重总大小 ${fmtBytes(r.size)}，其中显存 ${fmtBytes(r.size_vram)}`}>
        {fmtBytes(r.size_vram)}
        {r.offloaded ? (
          <span className="badge badge-warn" title="部分权重在 CPU 上：吞吐会低一个数量级，不可与全量载入的结果混算">
            ! offload {fmtPct(1 - (r.vram_share ?? 1), 0)}
          </span>
        ) : null}
      </span>
    ),
  },
  {
    key: 'ctx',
    header: 'ctx',
    align: 'right',
    sortValue: (r) => r.context_length,
    // 载入上下文 ≠ 训练上下文（PROBES P3：4096 vs 262144，差 64 倍）
    render: (r) => r.context_length ?? <span className="muted">{UNKNOWN}</span>,
  },
  {
    key: 'keepalive',
    header: 'keep-alive',
    align: 'right',
    sortValue: (r) => r.keep_alive_seconds,
    render: (r) => fmtSeconds(r.keep_alive_seconds),
  },
  { key: 'quant', header: '量化', sortValue: (r) => r.quantization, render: (r) => r.quantization || UNKNOWN },
]

/** 触发历史。它回答的是"某天到底通知没通知"，所以空表也要能被读成"没通知过任何事"。 */
function AlertHistoryPanel({
  rows, loading, error, onRefresh,
}: {
  rows: AlertTriggerView[] | null
  loading: boolean
  error: unknown
  onRefresh: () => void
}) {
  return (
    <Panel
      title="告警触发"
      note="只记「命中并尝试投递」的：被 cooldown 挡住的不会出现在这里"
      actions={<button className="btn" onClick={onRefresh}>刷新</button>}
      flush
    >
      {error ? <ErrorState error={error} /> : null}
      {loading ? <Skeleton rows={4} /> : null}
      {rows && rows.length ? (
        <DataTable columns={alertColumns} rows={rows} rowKey={(r) => r.id} maxHeight={260} />
      ) : null}
      {rows && !rows.length ? (
        <EmptyState
          title="还没有触发记录"
          hint="自检一条出口：onyx alerts test（会落一行标明 is_test，不影响真实 cooldown）"
        />
      ) : null}
    </Panel>
  )
}

export function FleetPage() {
  const fleet = useApi<FleetView>(() => api.fleet(), { intervalMs: REFRESH_MS })
  const triggers = useApi<AlertTriggerView[]>(() => api.alertTriggers({ limit: 30 }),
    { intervalMs: ALERTS_REFRESH_MS })

  if (fleet.error && !fleet.data) return <ErrorState error={fleet.error} />
  if (!fleet.data) return <Skeleton rows={6} />

  const { data } = fleet
  const window_ = data.window
  const coldCount = window_.by_prefill_mode?.cold ?? 0
  const warmCount = window_.by_prefill_mode?.warm ?? 0
  const chips = anomalyChips(data.anomalies)
  const errorText = errorBannerText(data.error_anomalies)
  const alertLine = alertBanner(data.alerts)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      {errorText || alertLine.tone ? (
        <div className={`banner ${errorText ? 'banner-error' : ''} ${alertLine.tone}`.trim()}>
          <b>{errorText || '异常通知状态'}</b>
          <span className="cell-dim small">　{alertLine.text}</span>
          <span className="panel-head-spacer" />
          <button className="linklike" onClick={() => navigate('/traces')} title="近 1 小时的异常在 trace 详情里">
            看现场
          </button>
        </div>
      ) : null}
      <div className="grid">
        <div className="col-3">
          <StatCard
            label="请求数 / 近 1h"
            value={fmtInt(window_.traces)}
            sub={
              <>
                错误 {fmtInt(window_.errors)}（{fmtPct(window_.error_rate)}）
              </>
            }
          />
        </div>
        <div className="col-3">
          <StatCard
            label="输入 token / 近 1h"
            value={fmtCompact(window_.in_tokens)}
            sub={`输出 ${fmtCompact(window_.out_tokens)}`}
          />
        </div>
        <div className="col-3">
          <StatCard
            label="decode 吞吐均值"
            value={fmtFloat(window_.decode_tps_avg)}
            unit="t/s"
            unknown={window_.decode_tps_avg === null}
            sub="cold/warm 分列见下"
          />
        </div>
        <div className="col-3">
          <StatCard
            label="推理 token（含 thinking）"
            value={fmtCompact(window_.thinking_tokens)}
            sub="P12：thinking 计入 eval_count"
          />
        </div>
      </div>

      <div className="grid">
        <div className="col-8">
          <Panel
            title="已载入模型"
            note={data.loaded_known
              ? `${data.loaded_models.length} / ${data.installed_models} 已安装`
              : '驻留状态未知（该通道不报告）'}
            flush
          >
            {data.loaded_known && data.loaded_models.length ? (
              <DataTable
                columns={loadedColumns}
                rows={data.loaded_models}
                rowKey={(r) => r.name}
                maxHeight={260}
              />
            ) : (
              <EmptyState
                title={data.loaded_known ? '当前没有载入任何模型' : '这个通道不报告驻留状态'}
                hint={data.loaded_known
                  ? '发一次请求即会载入；或在 Playground 里选一个模型'
                  : 'OpenAI 兼容层（vLLM / LM Studio 等）没有"哪些模型在显存里"的统一端点；'
                    + '驻留策略由服务器自己决定，Onyx 不猜'}
              />
            )}
          </Panel>
        </div>
        <div className="col-4">
          <Panel title="prefill 冷 / 热" note="不可合并聚合（P11）">
            <div className="row gap-3">
              <PrefillTag mode="cold" />
              <b className="num">{fmtInt(coldCount)}</b>
              <PrefillTag mode="warm" />
              <b className="num">{fmtInt(warmCount)}</b>
            </div>
            <p className="small muted" style={{ marginTop: 'var(--space-2)' }}>
              热缓存下 prefill 吞吐是<b>等效值</b>：实测同一 prompt 冷 1675 t/s、热 7722 t/s，
              差 4.65 倍。合并后的 P50 既不代表冷启动也不代表稳态，而且看起来完全合理。
            </p>
          </Panel>
        </div>
      </div>

      <Panel title="异常分布" note="近 1h · 级别与文案都来自后端统一码表" flush>
        {chips.length ? (
          <div className="chips" style={{ padding: 'var(--space-3)' }}>
            {chips.map((row) => (
              <span key={row.code} className="row">
                <AnomalyChip
                  anomaly={{ id: row.code, code: row.code, severity: row.severity,
                             meaning: '', action: '', detail: {} }}
                />
                <b className="num">{fmtInt(row.n)}</b>
              </span>
            ))}
          </div>
        ) : (
          <EmptyState title="近 1 小时无异常" />
        )}
      </Panel>

      <AlertHistoryPanel rows={triggers.data} loading={triggers.loading && !triggers.data}
        error={triggers.error} onRefresh={triggers.refresh} />

      <div className="row small muted">
        <StatusDot ok={data.provider_reachable} />
        <span>
          {data.provider_id} · {data.provider_kind} · 引擎 v{data.engine_version || UNKNOWN} ·{' '}
          {data.base_url}
        </span>
        <span className="panel-head-spacer" />
        <span>数据 {fleet.fetchedAt ? timeAgo(new Date(fleet.fetchedAt).toISOString()) : UNKNOWN} · 每 10s 刷新</span>
      </div>
    </div>
  )
}
