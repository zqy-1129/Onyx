/** Prompt 分段归因：堆叠条（不用饼图——占比接近时饼图分不出来）。
 *  核心价值：让"工具定义每次请求偷走多少上下文"和"模板控制符成本"可见。 */
import type { TokenPartView } from '../api/types'
import { fmtInt, fmtPct, isClosed, UNKNOWN } from '../format'

const PART_LABELS: Record<string, string> = {
  system: 'system',
  tool_defs: '工具定义',
  template_ctl: '模板控制符',
  output: '输出',
  image: '图片',
  bos: 'BOS',
  gen_prompt: '生成提示符',
}

function partClass(part: string): string {
  if (part.startsWith('msg:')) return 'seg-msg'
  if (part === 'tool_defs') return 'seg-tool_defs'
  if (part === 'template_ctl') return 'seg-template_ctl'
  if (part === 'output') return 'seg-output'
  if (part === 'image') return 'seg-image'
  return 'seg-system'
}

function label(part: string): string {
  if (part.startsWith('msg:')) return `消息 ${part.slice(4)}`
  return PART_LABELS[part] ?? part
}

export function PromptBreakdown({
  parts,
  engineIn,
  showOutput = false,
  attribution,
  model,
}: {
  parts: TokenPartView[]
  engineIn?: number | null
  showOutput?: boolean
  /** `trace.extra.attribution`：这一条的分段是按哪一档计数数的、残差有没有被 clamp */
  attribution?: Record<string, unknown> | null
  /** 模型名。只有拿到真名才敢写 `onyx calibrate --model …`——占位符是抄不动的命令 */
  model?: string | null
}) {
  const visible = parts.filter((p) => p.tokens > 0 && (showOutput || p.part !== 'output'))
  if (!visible.length) {
    return (
      <div className="empty">
        <div className="empty-title">归因不可用</div>
        <div className="empty-hint">
          这条 trace 没有分段归因记录（分段全为 0 或缺失）。总量仍由引擎给出；
          各段的档位与残差见 docs/PROBES.md P9 与 onyx token explain。
        </div>
      </div>
    )
  }
  const total = visible.reduce((sum, p) => sum + p.tokens, 0)
  // 闭合校验只对**输入侧**分段做：引擎的 prompt_eval_count 不含生成内容。
  // 曾经把 output 段也算进总和，于是 Playground 上一律显示"未闭合（差 = 输出 token 数）"。
  const inputTotal = visible
    .filter((p) => p.part !== 'output')
    .reduce((sum, p) => sum + p.tokens, 0)
  const outputTotal = total - inputTotal
  const closed = isClosed(inputTotal, engineIn)
  const clamped = attribution?.clamped === true
  const tier = attribution?.count_source ? String(attribution.count_source) : null

  return (
    <div>
      <div className="stack" role="img" aria-label="prompt 分段归因">
        {visible.map((part) => (
          <div
            key={`${part.part}-${part.ord}`}
            className={`stack-seg ${partClass(part.part)}`}
            style={{ width: `${(part.tokens / total) * 100}%` }}
            title={`${label(part.part)}：${fmtInt(part.tokens)} token（${fmtPct(part.tokens / total)}）`}
          />
        ))}
      </div>
      <div className="legend">
        {visible.map((part) => (
          <span key={`${part.part}-${part.ord}`} className="legend-item">
            <span className={`legend-swatch ${partClass(part.part)}`} />
            {label(part.part)}
            <b className="num">{fmtInt(part.tokens)}</b>
            <span className="muted">{fmtPct(part.tokens / total, 0)}</span>
          </span>
        ))}
      </div>
      <div className="row mt-3 small">
        <span className="muted">
          Σ输入分段 = <b className="num">{fmtInt(inputTotal)}</b>
        </span>
        <span className="muted">
          引擎计数 = <b className="num">{engineIn == null ? UNKNOWN : fmtInt(engineIn)}</b>
        </span>
        {outputTotal > 0 ? (
          <span className="muted">
            输出 = <b className="num">{fmtInt(outputTotal)}</b>
            <span title="引擎的 prompt_eval_count 不含生成内容，因此输出段不参与闭合校验">（不计入闭合）</span>
          </span>
        ) : null}
        {engineIn != null ? (
          closed ? (
            <span className="badge badge-ok" title={tier ? `分段按 ${tier} 数，残差没有被 clamp` : undefined}>
              ✓ 归因闭合
            </span>
          ) : clamped ? (
            <span
              className="badge badge-warn"
              title={`残差 ${String(attribution?.residual_raw ?? UNKNOWN)} 为负 ⇒ template_ctl 记 0。`
                + '分段计数器比引擎高估，各段只能比相对占比。'
                + (tier === 'heuristic' && model ? `修法：onyx calibrate --model ${model}` : '')}
            >
              ! 计数器高估（差 {fmtInt(inputTotal - engineIn)}）
            </span>
          ) : (
            <span
              className="badge badge-warn"
              title="没被 clamp 时相等由构造保证，所以这个差值说明分段与采信总数不是同一次计算的结果"
            >
              ! 未闭合（差 {fmtInt(inputTotal - engineIn)}）
            </span>
          )
        ) : null}
      </div>
    </div>
  )
}
