/** Tool Bench 页的派生逻辑（S25）。
 *
 * 只测"界面会不会说错话"的那几个纯函数：矩阵三态里 n/a 与"没有结果"都不许画成 ✓、
 * 模板开销没传时占比是「—」而不是 0%、审计只有 error 级才构成"这库不能直接用"。
 */
import { describe, expect, it } from 'vitest'
import { auditVerdict, costLines, matrixCell, severityTone } from '../pages/ToolBench'
import type { ToolCostView } from '../api/types'

function cell(passed: boolean, applicable = true, detail = '') {
  return { passed, applicable, detail }
}

function cost(over: Partial<ToolCostView> = {}): ToolCostView {
  return {
    tools: [{ name: 'echo', tokens: 78, bytes: 210, kind: 'python_fn', side_effect: 'read' }],
    json_tokens: 78, json_bytes: 210, template_overhead_tokens: 0, effective_tokens: 78,
    template_share: null, count_source: 'heuristic', model: '', hint: '未指定模型 ⇒ heuristic 档',
    ...over,
  }
}

describe('matrixCell', () => {
  it('通过、失败、不适用是三件事', () => {
    expect(matrixCell(cell(true), 'python_fn')).toMatchObject({ symbol: '✓', tone: 'ok' })
    expect(matrixCell(cell(false, true, '没报 arg_error'), 'mock')).toMatchObject({ symbol: '✗', tone: 'error' })
    expect(matrixCell(cell(false, false, 'deadline 无从生效'), 'mock')).toMatchObject({ symbol: 'n/a', tone: 'unknown' })
  })

  it('n/a 的理由要能看见', () => {
    expect(matrixCell(cell(false, false, '无法证明零真实调用'), 'mock').title).toContain('无法证明零真实调用')
  })

  it('这一列根本没跑是「未知」，不是通过也不是失败', () => {
    const state = matrixCell(undefined, 'http')
    expect(state.symbol).toBe('?')
    expect(state.tone).toBe('unknown')
    expect(state.title).toContain('未知，不是通过')
  })
})

describe('costLines', () => {
  it('没传模板开销时占比显示「—」', () => {
    // 0% 会被读成"模板不花钱"，而 P17 的实测结论恰恰相反
    const lines = costLines(cost())
    const share = lines.find((line) => line.label === '模板占比')
    expect(share?.value).toBe('—')
    expect(share?.hint).toContain('不是 0%')
    expect(lines.find((line) => line.label === '模板脚手架')?.value).toBe('—')
  })

  it('传了开销就把两笔账分开列出来', () => {
    const lines = costLines(cost({ template_overhead_tokens: 213, effective_tokens: 291, template_share: 0.732 }))
    expect(lines.find((line) => line.label === 'JSON 本身')?.value).toContain('78')
    expect(lines.find((line) => line.label === '模板脚手架')?.value).toContain('213')
    expect(lines.find((line) => line.label === '每次请求实付')?.value).toContain('291')
    expect(lines.find((line) => line.label === '模板占比')?.value).toBe('73.2%')
  })

  it('计数档位与它的提醒一起给：heuristic 的绝对值只能用来比较', () => {
    const lines = costLines(cost())
    const source = lines.find((line) => line.label === '计数档位')
    expect(source?.value).toBe('heuristic')
    expect(source?.hint).toContain('heuristic')
  })

  it('数据没到的时候不编任何一行', () => {
    expect(costLines(null)).toEqual([])
  })
})

describe('auditVerdict', () => {
  it('只有 error 级才构成"这库不能直接用"', () => {
    expect(auditVerdict({ error: 2, warn: 1, info: 0 })).toMatchObject({ tone: 'error' })
    expect(auditVerdict({ error: 0, warn: 3, info: 1 })).toMatchObject({ tone: 'warn' })
    expect(auditVerdict({ error: 0, warn: 0, info: 4 })).toMatchObject({ tone: 'ok' })
    expect(auditVerdict(undefined)).toMatchObject({ tone: 'ok' })
  })

  it('severity 到色调的映射不落在默认色上', () => {
    expect(severityTone('error')).toBe('error')
    expect(severityTone('warn')).toBe('warn')
    expect(severityTone('info')).toBe('info')
  })
})
