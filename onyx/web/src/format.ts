/** 格式化工具。
 *  R2 是硬规则：未知一律显示「—」，绝不显示 0。
 *  0 是一个测量值，「没测出来」不是；混淆二者会让人基于空数据做决策。 */

export const UNKNOWN = '—'

function isMissing(value: number | null | undefined): value is null | undefined {
  return value === null || value === undefined || Number.isNaN(value)
}

export function fmtInt(value: number | null | undefined): string {
  if (isMissing(value)) return UNKNOWN
  return Math.round(value).toLocaleString('en-US')
}

export function fmtFloat(value: number | null | undefined, digits = 1): string {
  if (isMissing(value)) return UNKNOWN
  return value.toFixed(digits)
}

/** 大数字用 k/M 缩写，但保留一位小数——看板要的是量级感，不是假精度。 */
export function fmtCompact(value: number | null | undefined): string {
  if (isMissing(value)) return UNKNOWN
  const abs = Math.abs(value)
  if (abs >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`
  if (abs >= 10_000) return `${(value / 1000).toFixed(1)}k`
  return Math.round(value).toLocaleString('en-US')
}

export function fmtMs(value: number | null | undefined): string {
  if (isMissing(value)) return UNKNOWN
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`
  return `${value.toFixed(value < 10 ? 1 : 0)}ms`
}

export function fmtPct(value: number | null | undefined, digits = 1): string {
  if (isMissing(value)) return UNKNOWN
  return `${(value * 100).toFixed(digits)}%`
}

export function fmtBytes(value: number | null | undefined): string {
  if (isMissing(value)) return UNKNOWN
  const units = ['B', 'KiB', 'MiB', 'GiB']
  let v = value
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`
}

export function fmtSeconds(value: number | null | undefined): string {
  if (isMissing(value)) return UNKNOWN
  if (value < 0) return '已过期'
  if (value < 60) return `${Math.round(value)}s`
  if (value < 3600) return `${Math.floor(value / 60)}m${Math.round(value % 60)}s`
  return `${Math.floor(value / 3600)}h${Math.floor((value % 3600) / 60)}m`
}

export function fmtClock(iso: string | null | undefined): string {
  if (!iso) return UNKNOWN
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return UNKNOWN
  return date.toLocaleTimeString('zh-CN', { hour12: false })
}

export function fmtDateTime(iso: string | null | undefined): string {
  if (!iso) return UNKNOWN
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return UNKNOWN
  return date.toLocaleString('zh-CN', { hour12: false })
}

export function timeAgo(iso: string | null | undefined): string {
  if (!iso) return UNKNOWN
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return UNKNOWN
  const seconds = (Date.now() - then) / 1000
  if (seconds < 5) return '刚刚'
  if (seconds < 60) return `${Math.round(seconds)}s 前`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m 前`
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h 前`
  return `${Math.round(seconds / 86400)}d 前`
}

export function shortId(id: string | null | undefined, keep = 8): string {
  if (!id) return UNKNOWN
  return id.length > keep ? id.slice(-keep) : id
}

/** 把 sha256:<hex> 缩成可显示的短引用。 */
export function shortRef(ref: string | null | undefined): string {
  if (!ref) return UNKNOWN
  const [scheme, digest] = ref.split(':')
  if (!digest) return ref
  return `${scheme}:${digest.slice(0, 10)}`
}
