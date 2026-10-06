/** Token Ledger：输入/输出/推理 token 的时序、来源占比、drift 分布、冷热分列。
 *  R3 在这里最要命：把 cold 与 warm 的 prefill 吞吐画进同一条线，
 *  得到的曲线既不代表冷启动也不代表稳态，而且看起来完全合理。 */
import { useState } from 'react'
import { api } from '../api/client'
import type { UsageSummaryView } from '../api/types'
import { Sparkline } from '../components/primitives'
import { EmptyState, ErrorState, Panel, Skeleton, StatCard } from '../components/primitives'
import { fmtCompact, fmtFloat, fmtInt, fmtPct, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'

const RANGES = [
  { label: '近 1h', seconds: 3600 },
  { label: '近 24h', seconds: 86_400 },
  { label: '近 7d', seconds: 604_800 },
  { label: '全部', seconds: null },
] as const

function sinceIso(seconds: number | null): string | null {
  if (seconds === null) return null
  return new Date(Date.now() - seconds * 1000).toISOString()
}

function BarList({
  data,
  colorFor,
  emptyText,
}: {
  data: Record<string, number>
  colorFor?: (key: string) => string
  emptyText: string
}) {
  const entries = Object.entries(data).sort((a, b) => b[1] - a[1])
  const total = entries.reduce((sum, [, v]) => sum + v, 0)
  if (!entries.length || !total) return <EmptyState title={emptyText} />
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {entries.map(([key, value]) => (
        <div key={key} className="row" style={{ gap: 'var(--space-2)' }}>
          <span className="mono small" style={{ width: 96, flex: '0 0 96px' }}>{key}</span>
          <span style={{ flex: 1, height: 10, background: 'var(--bg-inset)', borderRadius: 2 }}>
            <span
              style={{
                display: 'block', height: '100%', width: `${(value / total) * 100}%`,
                background: colorFor?.(key) ?? 'var(--info)', borderRadius: 2,
              }}
            />
          </span>
          <b className="num small" style={{ width: 56, textAlign: 'right' }}>{fmtInt(value)}</b>
          <span className="muted small" style={{ width: 44, textAlign: 'right' }}>
            {fmtPct(value / total, 0)}
          </span>
        </div>
      ))}
    </div>
  )
}

function SeriesChart({ rows }: { rows: Array<Record<string, number | string>> }) {
  if (rows.length < 2) return <EmptyState title="样本不足，画不出趋势" hint="至少需要 2 个时间桶" />
  const inTokens = rows.map((r) => Number(r.in_tokens ?? 0))
  const outTokens = rows.map((r) => Number(r.out_tokens ?? 0))
  const cold = rows.map((r) => Number(r.cold_prefill_tps ?? 0))
  const warm = rows.map((r) => Number(r.warm_prefill_tps ?? 0))
  const decode = rows.map((r) => Number(r.decode_tps ?? 0))
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-2)' }}>
      <div className="row small">
        <span className="muted" style={{ width: 120, flex: '0 0 120px' }}>输入 token</span>
        <Sparkline values={inTokens} width={220} color="var(--info)" />
        <b className="num">{fmtCompact(Math.max(...inTokens))}</b>
        <span className="muted nowrap">峰值</span>
      </div>
      <div className="row small">
        <span className="muted" style={{ width: 120, flex: '0 0 120px' }}>输出 token</span>
        <Sparkline values={outTokens} width={220} color="var(--ok)" />
        <b className="num">{fmtCompact(Math.max(...outTokens))}</b>
        <span className="muted nowrap">峰值</span>
      </div>
      <div className="row small">
        <span className="muted" style={{ width: 120, flex: '0 0 120px' }}>decode t/s</span>
        <Sparkline values={decode} width={220} color="var(--accent)" />
        <b className="num">{fmtFloat(decode[decode.length - 1])}</b>
        <span className="muted nowrap">最新</span>
      </div>
      {/* 冷/热两条独立系列，绝不合并（R3） */}
      <div className="row small">
        <span className="muted" style={{ width: 120, flex: '0 0 120px' }}>❄ cold prefill</span>
        <Sparkline values={cold} width={220} color="var(--cold)" />
        <b className="num">{fmtFloat(avg(cold))}</b>
        <span className="muted nowrap">均值 t/s</span>
      </div>
      <div className="row small">
        <span className="muted" style={{ width: 120, flex: '0 0 120px' }}>♨ warm prefill</span>
        <Sparkline values={warm} width={220} color="var(--warm)" />
        <b className="num">{fmtFloat(avg(warm))}</b>
        <span className="muted nowrap">均值 t/s</span>
      </div>
    </div>
  )
}

function avg(values: number[]): number | null {
  const positive = values.filter((v) => v > 0)
  if (!positive.length) return null
  return positive.reduce((a, b) => a + b, 0) / positive.length
}

const SOURCE_COLORS: Record<string, string> = {
  engine: 'var(--src-engine)',
  fitted: 'var(--src-fitted)',
  heuristic: 'var(--src-heuristic)',
  compat: 'var(--src-compat)',
  hf_tokenizer: 'var(--src-hf)',
  gguf_vocab: 'var(--src-gguf)',
}

export function UsagePage() {
  const [rangeIndex, setRangeIndex] = useState(1)
  const range = RANGES[rangeIndex]
  const since = sinceIso(range.seconds)
  const state = useApi<UsageSummaryView>(
    () => api.usageSummary({ since, bucket_minutes: range.seconds && range.seconds <= 86_400 ? 60 : 360 }),
    { deps: [rangeIndex], intervalMs: 30_000 },
  )

  if (state.error && !state.data) return <ErrorState error={state.error} />
  if (!state.data) return <Skeleton rows={6} />
  const data = state.data

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
      <div className="row-wrap">
        <span className="field-label">时间范围</span>
        {RANGES.map((item, index) => (
          <button
            key={item.label}
            className={`btn${index === rangeIndex ? ' btn-primary' : ''}`}
            onClick={() => setRangeIndex(index)}
          >
            {item.label}
          </button>
        ))}
        <span className="panel-head-spacer" />
        <button className="btn" onClick={state.refresh}>刷新</button>
      </div>

      <div className="grid">
        <div className="col-3">
          <StatCard label="请求数" value={fmtInt(data.traces)} sub={range.label} />
        </div>
        <div className="col-3">
          <StatCard label="输入 token" value={fmtCompact(data.in_tokens)} />
        </div>
        <div className="col-3">
          <StatCard label="输出 token" value={fmtCompact(data.out_tokens)} />
        </div>
        <div className="col-3">
          <StatCard
            label="推理 token"
            value={fmtCompact(data.thinking_tokens)}
            sub="P12：thinking 计入 eval_count"
          />
        </div>
      </div>

      <div className="grid">
        <div className="col-4">
          <Panel title="采信来源分布" note="每个数字的出处（R1）">
            <BarList
              data={data.by_source}
              colorFor={(key) => SOURCE_COLORS[key] ?? 'var(--text-muted)'}
              emptyText="该时间范围内没有记录"
            />
          </Panel>
        </div>
        <div className="col-4">
          <Panel title="置信度分布" note="low 的数字不可用于容量决策">
            <BarList
              data={data.by_confidence}
              colorFor={(key) =>
                key === 'high' ? 'var(--ok)' : key === 'medium' ? 'var(--warn)' : 'var(--text-muted)'
              }
              emptyText="该时间范围内没有记录"
            />
          </Panel>
        </div>
        <div className="col-4">
          <Panel title="prefill 冷 / 热" note="不可合并聚合（P11）">
            <BarList
              data={data.by_prefill_mode}
              colorFor={(key) => (key === 'warm' ? 'var(--warm)' : key === 'cold' ? 'var(--cold)' : 'var(--text-muted)')}
              emptyText="没有 prefill 记录"
            />
            <p className="small muted" style={{ marginTop: 'var(--space-2)' }}>
              实测同一 644-token prompt：冷 1675 t/s、热 7722 t/s，差 4.65×。
              合并后的均值两边都不代表。
            </p>
          </Panel>
        </div>
      </div>

      <div className="grid">
        <div className="col-8">
          <Panel title="时序" note={range.label}>
            <SeriesChart rows={data.timeseries} />
          </Panel>
        </div>
        <div className="col-4">
          <Panel title="计量偏差（drift）" note="采信值 vs 独立复算值">
            <div className="grid">
              <div className="col-6">
                <StatCard label="样本数" value={fmtInt(data.drift.n)} />
              </div>
              <div className="col-6">
                <StatCard
                  label="超阈值(10%)"
                  value={fmtInt(data.drift.over_threshold)}
                  unknown={data.drift.n === 0}
                />
              </div>
              <div className="col-6">
                <StatCard label="中位数" value={fmtPct(data.drift.p50)} unknown={data.drift.p50 == null} />
              </div>
              <div className="col-6">
                <StatCard label="最大值" value={fmtPct(data.drift.max)} unknown={data.drift.max == null} />
              </div>
            </div>
            <p className="small muted mt-3">
              短 prompt 上百分比是噪声，因此报警还需满足绝对差 ≥ 24 token（P18）。
              drift 持续偏高通常意味着 tokenizer 档位与引擎模板不一致。
              {data.drift.n === 0 ? ` 当前范围无样本：${UNKNOWN}` : ''}
            </p>
          </Panel>
        </div>
      </div>
    </div>
  )
}
