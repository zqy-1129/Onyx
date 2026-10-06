/** 前端单测：重点是 R2（未知显示「—」而不是 0）与 R1（数字必须带出处）。
 *  这两条一旦破功，看板就会用空数据骗人，而且看不出来。 */
import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import {
  attributionNote,
  fmtBytes,
  fmtGb,
  fmtCompact,
  fmtInt,
  fmtMs,
  fmtPct,
  fmtSeconds,
  shortId,
  shortRef,
  isClosed,
  residencyOf,
  timeAgo,
  UNKNOWN,
} from '../format'
import { CapSymbol, PrefillTag, SourceBadge, StatCard } from '../components/primitives'

describe('format：未知一律「—」，绝不显示 0（R2）', () => {
  it.each([
    ['fmtInt', () => fmtInt(null)],
    ['fmtInt(undefined)', () => fmtInt(undefined)],
    ['fmtInt(NaN)', () => fmtInt(Number.NaN)],
    ['fmtFloat', () => fmtMs(null)],
    ['fmtPct', () => fmtPct(null)],
    ['fmtCompact', () => fmtCompact(null)],
    ['fmtBytes', () => fmtBytes(undefined)],
    ['fmtSeconds', () => fmtSeconds(null)],
    ['timeAgo', () => timeAgo(null)],
    ['shortId', () => shortId(null)],
    ['shortRef', () => shortRef('')],
  ])('%s 对缺失值返回「—」', (_name, fn) => {
    expect(fn()).toBe(UNKNOWN)
  })

  it('0 是有效测量值，必须显示 0 而不是「—」', () => {
    expect(fmtInt(0)).toBe('0')
    expect(fmtPct(0)).toBe('0.0%')
    expect(fmtMs(0)).toBe('0.0ms')
    expect(fmtCompact(0)).toBe('0')
  })

  it('数值格式', () => {
    expect(fmtInt(1842)).toBe('1,842')
    expect(fmtCompact(18_420)).toBe('18.4k')
    expect(fmtCompact(2_400_000)).toBe('2.40M')
    expect(fmtCompact(9_999)).toBe('9,999')
    expect(fmtMs(1400)).toBe('1.40s')
    expect(fmtMs(94.5)).toBe('95ms')
    expect(fmtBytes(5_490_081_790)).toBe('5.1 GiB')
    expect(fmtPct(0.1579)).toBe('15.8%')
    expect(fmtSeconds(90)).toBe('1m30s')
    expect(fmtSeconds(-5)).toBe('已过期')
    expect(shortId('01M3YHC9Y2J1PJFGHCQBN11KPB', 8)).toBe('QBN11KPB')
    expect(shortRef('sha256:' + 'a'.repeat(64))).toBe('sha256:aaaaaaaaaa')
  })
})

describe('SourceBadge：没有出处的数字不上看板（R1）', () => {
  it('无来源时显式标注，而不是留白', () => {
    render(<SourceBadge source={null} />)
    expect(screen.getByText('no source')).toBeTruthy()
  })

  it('显示来源与置信度，低置信度带虚线下划线', () => {
    const { container } = render(<SourceBadge source="heuristic" confidence="low" note="估计值" />)
    expect(container.textContent).toContain('heuristic')
    expect(container.querySelector('.conf-low-wrap')).toBeTruthy()
    expect(container.querySelector('.conf-low')).toBeTruthy()
  })

  it('高置信度不加虚线（R5：不确定不等于错误，不该用告警样式）', () => {
    const { container } = render(<SourceBadge source="engine" confidence="high" />)
    expect(container.querySelector('.conf-low-wrap')).toBeNull()
    expect(container.querySelector('.badge-src-engine')).toBeTruthy()
  })
})

describe('StatCard：unknown 与 0 必须视觉可分', () => {
  it('unknown 时渲染「—」并置灰，忽略传入的 value', () => {
    const { container } = render(<StatCard label="TTFT" value="999" unknown />)
    expect(container.textContent).toContain(UNKNOWN)
    expect(container.textContent).not.toContain('999')
    expect(container.querySelector('.is-unknown')).toBeTruthy()
  })

  it('值为 0 时正常显示 0', () => {
    const { container } = render(<StatCard label="错误数" value={fmtInt(0)} />)
    expect(container.textContent).toContain('0')
    expect(container.querySelector('.is-unknown')).toBeNull()
  })
})

describe('PrefillTag：冷/热不可混淆（R3、R4）', () => {
  it('三态各有独立符号与样式', () => {
    const cold = render(<PrefillTag mode="cold" />)
    expect(cold.container.textContent).toContain('❄')
    expect(cold.container.querySelector('.badge-cold')).toBeTruthy()

    const warm = render(<PrefillTag mode="warm" />)
    expect(warm.container.textContent).toContain('♨')
    expect(warm.container.querySelector('.badge-warm')).toBeTruthy()

    const unknown = render(<PrefillTag mode={null} />)
    expect(unknown.container.textContent).toContain('?')
    expect(unknown.container.querySelector('.badge-unknown')).toBeTruthy()
  })

  it('tooltip 说明为何不可合并聚合', () => {
    const { container } = render(<PrefillTag mode="warm" msPerToken={0.129} />)
    const tip = container.querySelector('.badge-warm')?.getAttribute('title') ?? ''
    expect(tip).toContain('4.65')
    expect(tip).toContain('0.129')
  })
})

describe('CapSymbol：✗ 与 ? 必须可区分', () => {
  it.each([
    ['confirmed', '✓'],
    ['missing', '✗'],
    ['unknown', '?'],
  ])('%s 渲染为 %s', (state, symbol) => {
    const { container } = render(<CapSymbol state={state} cap="tools" reason="依据" />)
    expect(container.textContent).toBe(symbol)
    expect(container.querySelector(`.cap-${state}`)).toBeTruthy()
  })

  it('tooltip 说明未实测 ≠ 不支持', () => {
    const { container } = render(<CapSymbol state="unknown" cap="structured_output" />)
    expect(container.querySelector('.cap-unknown')?.getAttribute('title')).toContain('未实测')
  })
})
describe('驻留状态三态（R2 的又一个断面）', () => {
  it('未知(null) 与 未载入(false) 必须长得不一样', () => {
    const unknown = residencyOf(null)
    const idle = residencyOf(false)
    expect(unknown.marker).toBe('unknown')
    expect(idle.marker).toBe('idle')
    // 这条不是样式测试：把"问不出来"显示成"没载入"，人会去查一个不存在的问题
    expect(unknown.text).not.toBe(idle.text)
    expect(unknown.title).toContain('不报告')
    expect(idle.title).toBe('未载入')
  })

  it('已载入用实心点，未知用「—」而不是 0 或空格', () => {
    expect(residencyOf(true).marker).toBe('loaded')
    expect(residencyOf(null).text).toBe(UNKNOWN)
  })
})
describe('体积（GB）', () => {
  it('不上报体积的通道显示「—」而不是 0.00GB', () => {
    expect(fmtGb(null)).toBe(UNKNOWN)
    expect(fmtGb(undefined)).toBe(UNKNOWN)
    // 0.00GB 会被读成"这个模型不占磁盘"，而真相是我们没问出来
    expect(fmtGb(0)).toBe('0.00GB')
    expect(fmtGb(4.567)).toBe('4.57GB')
  })
})

describe('分段闭合（S39：等式是有条件的，且前后端只能有一份定义）', () => {
  it('精确相等才算闭合——没有容差', () => {
    expect(isClosed(1000, 1000)).toBe(true)
    expect(isClosed(1001, 1000)).toBe(false) // 差 1 也得响：未 clamp 时相等由构造保证
    expect(isClosed(621, 475)).toBe(false)
  })

  it('没有引擎计数就不判（未知不是闭合，也不是不闭合）', () => {
    expect(isClosed(10, null)).toBe(false)
    expect(isClosed(10, undefined)).toBe(false)
  })

  it('说明句把条件与档位一起说，老行没记档位就明说', () => {
    expect(attributionNote({ count_source: 'fitted' })).toContain('仅未 clamp 时成立')
    expect(attributionNote({ count_source: 'fitted' })).toContain('分段按 fitted 数')
    expect(attributionNote({})).toContain('这条没记归因档位')
    expect(attributionNote(null)).toContain('这条没记归因档位')
  })
})
