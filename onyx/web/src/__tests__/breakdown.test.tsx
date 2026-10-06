/** PromptBreakdown 的闭合校验：回归测试。
 *  曾经把 output 段也算进"Σ分段"，于是 Playground 上一律显示「未闭合（差 = 输出 token 数）」——
 *  一个假的告警比没有告警更糟，它会让人不再相信真的告警。 */
import { describe, expect, it } from 'vitest'
import { render } from '@testing-library/react'
import { PromptBreakdown } from '../components/PromptBreakdown'
import type { TokenPartView } from '../api/types'

const part = (p: string, ord: number, tokens: number, bytes: number | null = null): TokenPartView => ({
  part: p,
  ord,
  tokens,
  bytes,
})

describe('PromptBreakdown 闭合校验', () => {
  it('输出段不参与闭合（引擎 prompt_eval_count 只数输入）', () => {
    const { container } = render(
      <PromptBreakdown
        parts={[part('msg:0', 0, 7, 30), part('template_ctl', 1, 10), part('output', 2, 133, 400)]}
        engineIn={17}
        showOutput
      />,
    )
    const text = container.textContent ?? ''
    expect(text).toContain('归因闭合')
    expect(text).not.toContain('未闭合')
    expect(text).toContain('不计入闭合')
    expect(text).toContain('133')
  })

  it('输入侧真的不闭合时必须报警，并给出差值', () => {
    const { container } = render(
      <PromptBreakdown parts={[part('msg:0', 0, 5, 10), part('template_ctl', 1, 2)]} engineIn={17} />,
    )
    const text = container.textContent ?? ''
    expect(text).toContain('未闭合')
    expect(text).toContain('-10')
    expect(text).not.toContain('归因闭合')
  })

  it('差 1 个 token 就是未闭合——容差会把唯一的不自洽信号抹掉（S39）', () => {
    /** 原先这里是 ±2%（注释写"计数档位本身有舍入"）。但 `template_ctl` 就是那条残差，
     *  没被 clamp 时相等由构造保证 ⇒ 差 1 恰恰说明两边不是同一次计算。
     *  而且后端 `onyx token explain` 用的是精确相等，两边留不同容差就是同一个量两个事实。 */
    const { container } = render(<PromptBreakdown parts={[part('msg:0', 0, 999), part('template_ctl', 1, 10)]} engineIn={1008} />)
    expect(container.textContent).toContain('未闭合（差 1）')
    expect(container.textContent).not.toContain('归因闭合')
  })

  it('clamp 的行报"计数器高估"而不是笼统的未闭合，并给出标定修法', () => {
    const { container } = render(
      <PromptBreakdown
        parts={[part('msg:0', 0, 621), part('template_ctl', 1, 0)]}
        engineIn={475}
        attribution={{ count_source: 'heuristic', clamped: true, residual_raw: -146 }}
        model="qwen3.5:9b"
      />,
    )
    expect(container.textContent).toContain('计数器高估（差 146）')
    expect(container.textContent).not.toContain('未闭合')
    const badge = container.querySelector('.badge-warn')
    expect(badge?.getAttribute('title')).toContain('onyx calibrate --model qwen3.5:9b')
    expect(badge?.getAttribute('title')).toContain('相对占比')
  })

  it('没有模型名时不写占位符命令——抄不动的命令等于没有命令', () => {
    const { container } = render(
      <PromptBreakdown
        parts={[part('msg:0', 0, 621), part('template_ctl', 1, 0)]}
        engineIn={475}
        attribution={{ count_source: 'heuristic', clamped: true, residual_raw: -146 }}
      />,
    )
    expect(container.querySelector('.badge-warn')?.getAttribute('title')).not.toContain('calibrate')
  })

  it('已标定还被 clamp 时不该再叫人生跑一次标定', () => {
    const { container } = render(
      <PromptBreakdown
        parts={[part('msg:0', 0, 621), part('template_ctl', 1, 0)]}
        engineIn={475}
        attribution={{ count_source: 'fitted', clamped: true, residual_raw: -146 }}
        model="qwen3.5:9b"
      />,
    )
    expect(container.querySelector('.badge-warn')?.getAttribute('title')).not.toContain('calibrate')
  })

  it('归因闭合的那句说明档位来自哪一条记录', () => {
    const { container } = render(
      <PromptBreakdown
        parts={[part('msg:0', 0, 16), part('template_ctl', 1, 9)]}
        engineIn={25}
        attribution={{ count_source: 'fitted', clamped: false, residual_raw: 9 }}
      />,
    )
    expect(container.textContent).toContain('归因闭合')
    expect(container.querySelector('.badge-ok')?.getAttribute('title')).toContain('fitted')
  })

  it('没有引擎计数时不做闭合判定（不猜）', () => {
    const { container } = render(
      <PromptBreakdown parts={[part('msg:0', 0, 7)]} engineIn={null} />,
    )
    const text = container.textContent ?? ''
    expect(text).not.toContain('归因闭合')
    expect(text).not.toContain('未闭合')
    expect(text).toContain('—')
  })

  it('没有任何分段时说明"归因不可用"而不是显示 0', () => {
    const { container } = render(<PromptBreakdown parts={[]} engineIn={17} />)
    expect(container.textContent).toContain('归因不可用')
    expect(container.textContent).toContain('P9')
  })

  it('工具定义段单独可见（DESIGN §6.2 的核心指标）', () => {
    const { container } = render(
      <PromptBreakdown
        parts={[part('system', 0, 120), part('tool_defs', 1, 1380), part('msg:1', 2, 340), part('template_ctl', 3, 15)]}
        engineIn={1855}
      />,
    )
    const text = container.textContent ?? ''
    expect(text).toContain('工具定义')
    expect(text).toContain('1,380')
    expect(text).toContain('归因闭合')
  })
})
