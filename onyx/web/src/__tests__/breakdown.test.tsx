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

  it('2% 容差内算闭合（计数档位本身有舍入）', () => {
    const { container } = render(
      <PromptBreakdown parts={[part('msg:0', 0, 1000), part('template_ctl', 1, 10)]} engineIn={1000} />,
    )
    // 差 10 / 1000 = 1% ≤ 2%
    expect(container.textContent).toContain('归因闭合')
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
