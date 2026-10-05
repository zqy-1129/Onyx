/** 评测页的派生逻辑。
 *
 * 只测"会决定界面怎么说"的那几个纯函数：主分数选谁、未知怎么显示、
 * 差值的 0 与"没配对上"怎么区分。渲染快照不测——那类测试改样式就一片红，
 * 却测不出口径错误。
 */
import { describe, expect, it } from 'vitest'
import { statusSymbol } from '../components/primitives'
import { fmtCi, fmtDelta, fmtScore, deltaTone, UNKNOWN } from '../format'
import { gradeValueCell, headlineOf, headlineText, thinNote, verdictBadge, verdictOptions } from '../pages/EvalRuns'
import { cellIsThin, THIN_COVERAGE } from '../pages/EvalMatrix'
import { pickRuns, verdictFlips, LOW_PAIRS } from '../pages/Regression'
import type { MatrixCell, PairedCase, RunView } from '../api/types'

function cell(over: Partial<MatrixCell> = {}): MatrixCell {
  return {
    run_id: 'r1', model_id: 'm', task_id: 't', metric: 'macro_f1', value: 0.9,
    ci: null, n: 10, n_judged: 10, n_total: 10, coverage: 1, low_confidence: false,
    status: 'done', started_at: '2026-10-03T00:00:00+00:00', dataset_id: 'ds',
    dataset_revision: 'r1', cost: {}, ...over,
  } as MatrixCell
}

function run(over: Partial<RunView> = {}): RunView {
  return {
    id: 'r1', task_id: 'intent_classification', model_id: 'm', status: 'done',
    started_at: '2026-10-03T00:00:00+00:00', finished_at: null, seed: 1, app_version: '0.1.0',
    git_rev: '', params_snapshot: {}, config: {}, n_cases: 4, n_done: 4, n_error: 0,
    n_skipped: 0, dataset_id: 'ds', dataset_revision: 'r1', aggregate: {}, cost: {}, ...over,
  } as RunView
}

function paired(over: Partial<PairedCase> = {}): PairedCase {
  return {
    case_id: 'c', kind: 'single', score_base: 1, score_target: 0, delta: -1,
    passed_base: true, passed_target: false, verdict_base: 'correct', verdict_target: 'wrong',
    trace_base: 't1', trace_target: 't2', instruction: '查余额', ...over,
  } as PairedCase
}

describe('主分数选择', () => {
  it('取第一个出现过的指标，而不是第一个非空的', () => {
    // macro_f1 未定义时如果顺手改显示 pass_hat_k，"根本没有可判定样本"就会被读成
    // "模型得了 0 分"——两者的修法完全相反
    expect(headlineOf({ macro_f1: null, pass_hat_k: 0.4 })).toEqual({ key: 'macro_f1', value: null })
    expect(headlineOf({ must_call_acc: 0.639 })).toEqual({ key: 'must_call_acc', value: 0.639 })
    // 主分数让位给稳定性指标，等于把"对不对"藏起来只报"稳不稳"（与后端 HEADLINE_METRICS 同序）
    expect(headlineOf({ score: 0.667, pass_hat_k: 0.5 })).toEqual({ key: 'score', value: 0.667 })
    expect(headlineOf({})).toBeNull()
  })

  it('未定义渲染成「—」并带上原因，不渲染成 0', () => {
    const text = headlineText({ macro_f1: null, pass_hat_k: 0 })
    expect(text).toContain(UNKNOWN)
    expect(text).not.toContain('0.000')
  })

  it('有值时把区间一起带上', () => {
    const text = headlineText({
      macro_f1: 0.9913,
      macro_f1_ci: { low: 0.9778, high: 1, point: 0.9913, n: 236 },
    })
    expect(text).toBe('macro_f1 0.991 [0.978–1.000]')
  })
})

describe('判定配色', () => {
  it('红只表示失败，模型答错是琥珀色', () => {
    expect(verdictBadge('correct')).toBe('badge badge-ok')
    expect(verdictBadge('error')).toBe('badge badge-error')
    expect(verdictBadge('wrong')).toBe('badge badge-warn')
    expect(verdictBadge('bad_args')).toBe('badge badge-warn')
    expect(verdictBadge('skipped')).toBe('badge badge-unknown')
  })
})

describe('grade 表的取值与筛选项', () => {
  it('对象值渲染成 k=v，而不是 [object Object]', () => {
    // 结构化抽取的期望/预测是字段集合；String(obj) 会让那一列什么都没说
    expect(gradeValueCell({ person: '张伟', amount: 500 })).toBe('person=张伟 · amount=500')
    expect(gradeValueCell('转账')).toBe('转账')
    expect(gradeValueCell(null)).toBe(UNKNOWN)
    expect(gradeValueCell(undefined)).toBe(UNKNOWN)
    // 空对象是"句子里没有可抽取信息"这个正确答案，不许显示成「—」
    expect(gradeValueCell({})).toBe('{}')
    // 空字符串必须显式画出来：`org=` 后面什么都没有会让人以为渲染坏了，
    // 而那恰恰是模型"多抽了一个空占位字段"的真实形态
    expect(gradeValueCell({ org: '' })).toBe('org=""')
    expect(gradeValueCell(['get_weather', 'send_mail'])).toBe('get_weather · send_mail')
    expect(gradeValueCell([])).toBe('（空）')
  })

  it('判定筛选项来自这次运行自己的分布', () => {
    // 硬编码清单曾经漏掉 partial：45 条里 27 条筛不出来，而界面看起来一切正常
    const options = verdictOptions({ verdicts: { correct: 30, wrong: 12, partial: 3 } })
    expect(options.map((o) => o.value)).toEqual(['', 'correct', 'wrong', 'partial'])
    expect(options[3].label).toBe('partial (3)')
    expect(verdictOptions(undefined).map((o) => o.value)).toEqual([''])
    expect(verdictOptions({ verdicts: { error: 0 } }).map((o) => o.value)).toEqual([''])
  })
})

describe('矩阵覆盖率点名', () => {
  it('主分数只覆盖少数样本的格子必须被点名', () => {
    // 真机形态：正文被 thinking 吃光的模型让 macro_f1 只在 8/236 上算出 1.000
    expect(cellIsThin(cell({ n_judged: 8, n_total: 236, coverage: 8 / 236 }))).toBe(true)
    expect(cellIsThin(cell())).toBe(false)
    // 任务没报可判定数 ⇒ 不猜
    expect(cellIsThin(cell({ n_judged: null, n_total: null, coverage: null }))).toBe(false)
    expect(THIN_COVERAGE).toBe(0.5)
  })

  it('运行列表用同一份口径，不各写一套', () => {
    expect(thinNote({ n_judged: 8, n_total: 236 })).toBe('可判定 8/236')
    expect(thinNote({ n_judged: 236, n_total: 236 })).toBeNull()
    expect(thinNote({ n_total: 236 })).toBeNull()
    // 0/236 是"一条都没判定"，比例上的 0 不该被当成"覆盖率为 0"来展示
    expect(thinNote({ n_judged: 0, n_total: 236 })).toBeNull()
  })
})

describe('差值与区间', () => {
  it('0 是"没变化"这个结论，不是未知', () => {
    expect(fmtDelta(0)).toBe('±0.000')
    expect(fmtDelta(null)).toBe(UNKNOWN)
    expect(fmtDelta(-0.5)).toBe('−0.500')
    expect(fmtDelta(0.25)).toBe('+0.250')
    expect(deltaTone(0)).toBe('flat')
    expect(deltaTone(null)).toBe('flat')
    expect(deltaTone(-0.01, 0.05)).toBe('flat')
  })

  it('分数与区间都遵守「—」规则', () => {
    expect(fmtScore(null)).toBe(UNKNOWN)
    expect(fmtScore(0)).toBe('0.000')
    expect(fmtCi({ low: 0.5, high: null })).toBe('[0.500–?]')
    expect(fmtCi(null)).toBe('')
  })
})

describe('状态符号', () => {
  it('每种运行状态都有自己的形状，done 不许渲染成「?」', () => {
    // 本项目里 "?" 专门表示"未实测"（CapSymbol / PROBES），
    // 把 done 也画成 "?" 就等于说"这次运行没测出来"
    const symbol = (status: string) => statusSymbol(status)
    expect(symbol('done')).toBe('✓')
    expect(symbol('running')).toBe('▶')
    expect(symbol('skipped')).toBe('⊝')
    expect(symbol('error')).toBe('✕')
    expect(new Set(['done', 'running', 'skipped', 'error', 'cancelled'].map(symbol)).size).toBe(5)
    expect(symbol('nonsense')).toBe('?')
  })
})

describe('配对口径', () => {
  it('未完成与取消的运行不参与配对', () => {
    const { done, byTask } = pickRuns([
      run({ id: 'a' }), run({ id: 'b', status: 'cancelled' }), run({ id: 'c', status: 'running' }),
      run({ id: 'd', task_id: 'tool_selection' }),
    ])
    expect(done.map((r) => r.id)).toEqual(['a', 'd'])
    expect(byTask.get('intent_classification')?.map((r) => r.id)).toEqual(['a'])
    expect(byTask.get('tool_selection')).toHaveLength(1)
  })

  it('未判定的样本不算翻转', () => {
    // 把"引擎挂了所以没判定"算成"从不会变成会"，会把基础设施故障报成模型进步
    const flips = verdictFlips([
      paired({ case_id: 'x', passed_base: null, passed_target: true }),
      paired({ case_id: 'y', passed_base: false, passed_target: true }),
      paired({ case_id: 'z', passed_base: true, passed_target: false }),
    ])
    expect(flips).toEqual({ up: 1, down: 1 })
    expect(LOW_PAIRS).toBe(30)
  })
})
