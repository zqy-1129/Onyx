/** Fleet 总览：一眼回答"服务活着吗 / 谁占着显存 / 最近一小时什么水平 / 有没有异常"。 */
import { api } from '../api/client'
import type { FleetView, LoadedModelView } from '../api/types'
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
import { fmtBytes, fmtCompact, fmtFloat, fmtInt, fmtPct, fmtSeconds, timeAgo, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'

const REFRESH_MS = 10_000

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

export function FleetPage() {
  const fleet = useApi<FleetView>(() => api.fleet(), { intervalMs: REFRESH_MS })

  if (fleet.error && !fleet.data) return <ErrorState error={fleet.error} />
  if (!fleet.data) return <Skeleton rows={6} />

  const { data } = fleet
  const window_ = data.window
  const coldCount = window_.by_prefill_mode?.cold ?? 0
  const warmCount = window_.by_prefill_mode?.warm ?? 0
  const anomalyEntries = Object.entries(data.anomalies).sort((a, b) => b[1] - a[1])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
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

      <Panel title="异常分布" note="近 1h · 文案来自后端统一码表" flush>
        {anomalyEntries.length ? (
          <div className="chips" style={{ padding: 'var(--space-3)' }}>
            {anomalyEntries.map(([code, count]) => (
              <span key={code} className="row">
                <AnomalyChip
                  anomaly={{ id: code, code, severity: 'warn', meaning: '', action: '', detail: {} }}
                />
                <b className="num">{fmtInt(count)}</b>
              </span>
            ))}
          </div>
        ) : (
          <EmptyState title="近 1 小时无异常" />
        )}
      </Panel>

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
