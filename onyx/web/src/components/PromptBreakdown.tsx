/** Prompt 分段归因：堆叠条（不用饼图——占比接近时饼图分不出来）。
 *  核心价值：让"工具定义每次请求偷走多少上下文"和"模板控制符成本"可见。 */
import type { TokenPartView } from '../api/types'
import { fmtInt, fmtPct, UNKNOWN } from '../format'

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
}: {
  parts: TokenPartView[]
  engineIn?: number | null
  showOutput?: boolean
}) {
  const visible = parts.filter((p) => p.tokens > 0 && (showOutput || p.part !== 'output'))
  if (!visible.length) {
    return (
      <div className="empty">
        <div className="empty-title">归因不可用</div>
        <div className="empty-hint">
          该模型当前没有可用的分段计数档位（见 docs/PROBES.md P9）。总量仍由引擎给出，
          但无法拆到 system / 工具定义 / 各消息。
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
  const closed = engineIn != null && Math.abs(inputTotal - engineIn) <= Math.max(1, engineIn * 0.02)

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
            <span className="badge badge-ok">✓ 归因闭合</span>
          ) : (
            <span className="badge badge-warn" title="分段计数之和与引擎计数不一致：计数档位可能高估或引擎发生截断">
              ! 未闭合（差 {fmtInt(inputTotal - engineIn)}）
            </span>
          )
        ) : null}
      </div>
    </div>
  )
}
