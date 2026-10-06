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

/* ── 评测数字（S15）─────────────────────────────────────────────
 * 规则与 R2 一致，但多一条：**带符号的差值不能把 0 显示成「—」**。
 * 差值 0 是"没变化"这个结论，未知是"没配对上"，两者混了就等于把结论抹掉。 */

/** 0–1 的分数保留三位；未定义显示「—」 */
export function fmtScore(value: number | null | undefined): string {
  return isMissing(value) ? UNKNOWN : value.toFixed(3)
}

/** 带符号的差值，用于回归/改善列表。 */
export function fmtDelta(value: number | null | undefined): string {
  if (isMissing(value)) return UNKNOWN
  const sign = value > 0 ? '+' : value < 0 ? '−' : '±'
  return `${sign}${Math.abs(value).toFixed(3)}`
}

/** CI 的紧凑写法：`[0.528–0.736]`，缺一端就只写知道的那端。 */
export function fmtCi(ci: { low: number | null; high: number | null } | null | undefined): string {
  if (!ci || (ci.low === null && ci.high === null)) return ''
  const low = ci.low === null ? '?' : ci.low.toFixed(3)
  const high = ci.high === null ? '?' : ci.high.toFixed(3)
  return `[${low}–${high}]`
}

/** 比例（覆盖率、误调率）用百分数；分母未知时不猜。 */
export function fmtRate(value: number | null | undefined, digits = 0): string {
  return isMissing(value) ? UNKNOWN : `${(value * 100).toFixed(digits)}%`
}

/** 差值该染成"变好"还是"变坏"：方向由调用方给，颜色不由数值大小自己决定。 */
export function deltaTone(value: number | null | undefined, eps = 0): 'good' | 'bad' | 'flat' {
  if (isMissing(value) || Math.abs(value) <= eps) return 'flat'
  return value > 0 ? 'good' : 'bad'
}

/** 体积按 GB 显示；未知（该通道不上报）显示「—」而不是 0.00GB。 */
export function fmtGb(value: number | null | undefined): string {
  return isMissing(value) ? UNKNOWN : `${value.toFixed(2)}GB`
}

/** 驻留状态的三态。
 *
 * `null`（这个通道不报告）与 `false`（问了，答案是没有）必须长得不一样：
 * 把"未知"画成"未载入"会让人去查一个不存在的问题（R2）。 */
export type Residency = { marker: 'loaded' | 'idle' | 'unknown'; text: string; title: string }

export function residencyOf(loaded: boolean | null): Residency {
  if (loaded === true) return { marker: 'loaded', text: '●', title: '已载入' }
  if (loaded === false) return { marker: 'idle', text: '·', title: '未载入' }
  return {
    marker: 'unknown',
    text: UNKNOWN,
    title: '驻留状态未知：这个通道不报告（OpenAI 兼容层没有该端点）',
  }
}

/** 分段闭合的**唯一**判据：Σ非 output 分段恰好等于引擎计数。
 *
 * 与后端 `llm/measurement/explain._closure` 是同一条（那里也是 `total == in_tokens`）。
 * 这里原先留了 ±2% 容差，注释写的是"计数档位本身有舍入"——但 `template_ctl` 就是那条残差，
 * 没被 clamp 时相等**由构造保证**，所以差 1 个 token 恰恰说明"两边不是同一次计算"。
 * 把它包进容差等于把唯一能发现不自洽的机会抹掉，而且会让看板与 `onyx token explain`
 * 对同一条 trace 给出不同结论。 */
export function isClosed(inputTotal: number, engineIn: number | null | undefined): boolean {
  return engineIn != null && inputTotal === engineIn
}

/** 分段归因那一行说明：等式是有条件的，条件就是"没被 clamp"。
 *  档位（count_source）与残差来自 `trace.extra.attribution`；老行没记就得说"没记"，
 *  不能假装有档位。 */
export function attributionNote(attribution: Record<string, unknown> | null | undefined): string {
  const base = 'Σ分段 + template_ctl = 引擎计数（仅未 clamp 时成立）'
  const tier = attribution?.count_source
  if (!tier) return `${base} · 这条没记归因档位`
  return `${base} · 分段按 ${String(tier)} 数`
}
