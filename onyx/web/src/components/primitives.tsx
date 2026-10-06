/** 设计系统基元。见 docs/UI_DESIGN.md §4。
 *  规则：R1 数字必须带出处 · R2 未知显示「—」 · R4 色+符号+文字三重编码 · R5 红只表示错误 */
import type { ReactNode } from 'react'
import { UNKNOWN } from '../format'
import type { AnomalyView, Confidence, PrefillMode } from '../api/types'

/* ── Panel ─────────────────────────────────────────────── */
export function Panel({
  title,
  note,
  actions,
  children,
  flush,
  className = '',
}: {
  title: ReactNode
  note?: ReactNode
  actions?: ReactNode
  children: ReactNode
  flush?: boolean
  className?: string
}) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-head">
        <span className="panel-title">{title}</span>
        {/* 口径说明常驻标题栏：面板必须自述"这个数字怎么来的" */}
        {note ? <span className="panel-note">{note}</span> : null}
        <span className="panel-head-spacer" />
        {actions}
      </header>
      <div className={`panel-body${flush ? ' flush' : ''}`}>{children}</div>
    </section>
  )
}

/* ── StatCard ──────────────────────────────────────────── */
export function StatCard({
  label,
  value,
  unit,
  sub,
  badge,
  unknown,
}: {
  label: string
  value: ReactNode
  unit?: string
  sub?: ReactNode
  badge?: ReactNode
  /** true ⇒ 明确表达"没测出来"，值渲染为「—」并置灰（R2） */
  unknown?: boolean
}) {
  return (
    <div className="stat">
      <div className="stat-label" title={label}>{label}</div>
      <div className={`stat-value num${unknown ? ' is-unknown' : ''}`}>
        {unknown ? UNKNOWN : value}
        {unit && !unknown ? <span className="stat-unit">{unit}</span> : null}
      </div>
      <div className="stat-foot">
        {sub ? <span className="stat-sub">{sub}</span> : null}
        {badge}
      </div>
    </div>
  )
}

/* ── 出处与置信度（R1）────────────────────────────────── */
export function ConfidenceDot({ level }: { level: Confidence | string | null }) {
  const normalized = (level ?? 'low') as string
  const label = { high: '高置信', medium: '中等置信', low: '低置信（估计值）' }[normalized] ?? normalized
  return <span className={`conf-dot conf-${normalized}`} title={`置信度：${label}`} aria-label={label} />
}

export function SourceBadge({
  source,
  confidence,
  note,
}: {
  source: string | null
  confidence?: Confidence | string | null
  note?: string
}) {
  if (!source) {
    // 没有出处的数字不上看板（R1）——这里明确显示"无来源"而不是留白
    return <span className="badge badge-unknown" title="该数字没有计量来源">no source</span>
  }
  const conf = (confidence ?? 'low') as string
  const tip = [
    `计量来源：${source}`,
    `置信度：${conf}`,
    note ? `说明：${note}` : '',
    conf === 'low' ? '低置信度：估计值，不可用于计费或容量决策' : '',
  ]
    .filter(Boolean)
    .join('\n')
  return (
    <span className={conf === 'low' ? 'conf-low-wrap' : undefined} title={tip}>
      <span className={`badge badge-src-${source}`}>
        <ConfidenceDot level={conf} />
        {source}
      </span>
    </span>
  )
}

/* ── prefill 冷/热（R3）───────────────────────────────── */
export function PrefillTag({ mode, msPerToken }: { mode: PrefillMode | string | null; msPerToken?: number | null }) {
  const normalized = (mode ?? 'unknown') as string
  const symbol = { cold: '❄', warm: '♨', unknown: '?' }[normalized] ?? '?'
  const tip =
    normalized === 'warm'
      ? 'KV 缓存命中：prefill 吞吐是等效值而非真实计算吞吐，实测比冷启动高约 4.65 倍，不可与 cold 合并聚合'
      : normalized === 'cold'
        ? '冷 prefill：完整计算了 prompt'
        : '无法判定冷/热（缺少 in_tokens 或 prompt_eval 时长）'
  return (
    <span
      className={`badge badge-${normalized}`}
      title={msPerToken ? `${tip}\n${msPerToken.toFixed(3)} ms/token` : tip}
    >
      {symbol} {normalized}
    </span>
  )
}

/* ── 严重度 / 状态 ─────────────────────────────────────── */
export function SeverityBadge({ severity, children }: { severity: string; children?: ReactNode }) {
  const symbol = { error: '✕', warn: '!', info: 'i' }[severity] ?? '?'
  return <span className={`badge badge-${severity}`}>{symbol} {children}</span>
}

/** 状态 → 符号。导出来是为了能被测到：R4 要求"形状互不相同"，
 *  而 "?" 在本项目里专指"未实测"，所以任何真实状态都不许落到 "?"。 */
export const STATUS_BADGE_MAP: Record<string, string> = {
  ok: 'ok', done: 'ok', error: 'error', timeout: 'warn', cancelled: 'neutral',
  running: 'info', skipped: 'unknown', queued: 'neutral',
}

export const STATUS_SYMBOL: Record<string, string> = {
  ok: '✓', done: '✓', error: '✕', timeout: '⏱', cancelled: '⊘',
  running: '▶', skipped: '⊝', queued: '⏳',
}

export function statusSymbol(status: string): string {
  return STATUS_SYMBOL[status] ?? '?'
}

export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`badge badge-${STATUS_BADGE_MAP[status] ?? 'neutral'}`}>
      {statusSymbol(status)} {status}
    </span>
  )
}

export function AnomalyChip({ anomaly }: { anomaly: AnomalyView }) {
  const tip = [anomaly.meaning, anomaly.action ? `处置：${anomaly.action}` : '']
    .filter(Boolean)
    .join('\n')
  return (
    <span title={tip}>
      <SeverityBadge severity={anomaly.severity}>{anomaly.code}</SeverityBadge>
    </span>
  )
}

/* ── 能力位三态（✗ 与 ? 必须可区分）────────────────────── */
export function CapSymbol({ state, cap, reason }: { state: string; cap: string; reason?: string }) {
  const symbol = { confirmed: '✓', missing: '✗', unknown: '?' }[state] ?? '?'
  const label = { confirmed: '确认支持', missing: '确认不支持（评测应 skip）', unknown: '未实测（≠不支持，先跑探针）' }[
    state
  ] ?? state
  return (
    <span className={`cap cap-${state}`} title={`${cap}: ${label}${reason ? `\n依据：${reason}` : ''}`}>
      {symbol}
    </span>
  )
}

/** `ok === null` 是"还不知道"：画成灰点，不能画成红色的不可达（R2/R5）。 */
export function StatusDot({ ok }: { ok: boolean | null }) {
  const cls = ok === null ? 'status-unknown' : ok ? 'status-ok' : 'status-err'
  const label = ok === null ? '状态未知' : ok ? '可达' : '不可达'
  return <span className={`status-dot ${cls}`} title={label} />
}

/* ── 空态 / 骨架 ───────────────────────────────────────── */
export function EmptyState({ title, hint, children }: { title: string; hint?: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-title">{title}</div>
      {children}
      {hint ? <div className="empty-hint">{hint}</div> : null}
    </div>
  )
}

export function Skeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="skeleton" style={{ width: `${70 + ((i * 13) % 30)}%` }} />
      ))}
    </div>
  )
}

export function ErrorState({ error }: { error: unknown }) {
  const err = error as { code?: string; message?: string; detail?: { hint?: string } }
  return (
    <div className="empty">
      <div className="empty-title" style={{ color: 'var(--error)' }}>
        ✕ {err?.message ?? String(error)}
      </div>
      {err?.code ? <div className="empty-hint">code: {err.code}</div> : null}
      {err?.detail?.hint ? <div className="empty-hint">{err.detail.hint}</div> : null}
    </div>
  )
}

/* ── Sparkline：纯 SVG，无图表库 ───────────────────────── */
export function Sparkline({
  values,
  width = 96,
  height = 20,
  color = 'var(--info)',
}: {
  /** `null` = 这一格**没有测到**（不是 0）。折线在此断线，绝不补值。 */
  values: Array<number | null>
  width?: number
  height?: number
  color?: string
}) {
  const known = values.filter((v): v is number => v != null && Number.isFinite(v))
  if (known.length < 2) return <span className="muted small">{UNKNOWN}</span>
  const max = Math.max(...known)
  const min = Math.min(...known)
  const span = max - min || 1
  // x 按**原始下标**算，所以断点后的点不会挤到一起；补 0 会让空桶看起来像一次真实测量
  const step = width / Math.max(1, values.length - 1)
  const segments: string[] = []
  let current = ''
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) {
      if (current) segments.push(current.trim())
      current = ''
      return
    }
    const x = (index * step).toFixed(1)
    const y = (height - ((value - min) / span) * height).toFixed(1)
    current += `${current ? 'L' : 'M'}${x},${y} `
  })
  if (current) segments.push(current.trim())
  return (
    <svg width={width} height={height} role="img" aria-label="趋势">
      {segments.map((d, i) => (
        <path key={i} d={d} fill="none" stroke={color} strokeWidth={1.5} />
      ))}
    </svg>
  )
}
