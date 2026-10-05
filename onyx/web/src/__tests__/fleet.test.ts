/** Fleet 顶部一行与异常 chip 的口径（S29）。
 *
 * 只测"界面会不会说错话"的那几个纯函数：级别不许前端自己猜、
 * "没装配告警"与"没有异常"必须是两句话、轮询出错时不能仍然显示一切正常。
 */
import { describe, expect, it } from 'vitest'
import { alertBanner, anomalyChips, errorBannerText } from '../pages/Fleet'
import type { AlertRuntime, ErrorAnomalySummary, FleetView } from '../api/types'

function runtime(over: Partial<AlertRuntime> = {}): AlertRuntime {
  return { enabled: true, channels: ['file'], ticks: 12, last_error: '', thread_alive: true, ...over }
}

function summary(over: Partial<ErrorAnomalySummary> = {}): ErrorAnomalySummary {
  return { n: 0, by_code: {}, latest_code: '', latest_at: '', latest_trace_id: '', ...over }
}

describe('anomalyChips', () => {
  const source: FleetView['anomalies'] = {
    TOKEN_DRIFT: { n: 2, severities: ['warn'] },
    CONTEXT_OVERFLOW: { n: 5, severities: ['error'] },
    COMPAT_DIVERGENCE: { n: 5, severities: ['info'] },
  }

  it('级别跟着后端给的那份走，不写死 warn', () => {
    const chips = anomalyChips(source)
    expect(chips.find((c) => c.code === 'CONTEXT_OVERFLOW')?.severity).toBe('error')
    expect(chips.find((c) => c.code === 'COMPAT_DIVERGENCE')?.severity).toBe('info')
  })

  it('按条数多的在前，同数按码名字典序', () => {
    expect(anomalyChips(source).map((c) => c.code)).toEqual(
      ['COMPAT_DIVERGENCE', 'CONTEXT_OVERFLOW', 'TOKEN_DRIFT'],
    )
  })

  it('同一个码同时出现过两种级别时取更严重的', () => {
    // 取轻的那个会把 error 级洗白成"提醒"，而这两种事情的紧急程度完全不同
    const mixed = anomalyChips({ X: { n: 3, severities: ['warn', 'error'] } })
    expect(mixed[0].severity).toBe('error')
  })

  it('缺字段的旧响应不会崩，也不会编出一个级别', () => {
    const loose = { Y: { n: 1 } } as unknown as FleetView['anomalies']
    expect(anomalyChips(loose)[0].severity).toBe('warn')
    expect(anomalyChips(undefined as unknown as FleetView['anomalies'])).toEqual([])
  })
})

describe('errorBannerText', () => {
  it('没有 error 级时返回空串，让调用方整行都不渲染', () => {
    expect(errorBannerText(summary())).toBe('')
    expect(errorBannerText(undefined)).toBe('')
  })

  it('有 error 级时说出条数与每个码各几条', () => {
    const text = errorBannerText(summary({
      n: 4, by_code: { CONTEXT_OVERFLOW: 3, PROVIDER_ERROR: 1 },
    }))
    expect(text).toContain('4 条 error 级异常')
    expect(text).toContain('CONTEXT_OVERFLOW 3')
    expect(text).toContain('PROVIDER_ERROR 1')
  })
})

describe('alertBanner', () => {
  it('没带回状态时说"未知"，不说"没有告警"', () => {
    // 这两句话在界面上看起来一样，修法完全不同：前者是接口缺字段，后者是配置没开
    expect(alertBanner(undefined).text).toContain('未知')
    expect(alertBanner(undefined).tone).toBe('banner-warn')
  })

  it('没装配时把原因说出来', () => {
    const line = alertBanner(runtime({ enabled: false, reason: '[alerts].enabled = false' }))
    expect(line.text).toContain('未装配')
    expect(line.text).toContain('enabled = false')
  })

  it('轮询出错时升到 error 级——这等于"通知可能没发出去"', () => {
    const line = alertBanner(runtime({ last_error: 'OSError: disk full' }))
    expect(line.tone).toBe('banner-error')
    expect(line.text).toContain('disk full')
    expect(line.text).toContain('可能没发出去')
  })

  it('启用了但一个出口都没有也要说', () => {
    expect(alertBanner(runtime({ channels: [] })).text).toContain('没有出口')
  })

  it('一切正常时给出出口与轮询次数', () => {
    const line = alertBanner(runtime({ channels: ['file', 'webhook'], ticks: 240 }))
    expect(line.tone).toBe('')
    expect(line.text).toContain('file/webhook')
    expect(line.text).toContain('240')
  })
})
