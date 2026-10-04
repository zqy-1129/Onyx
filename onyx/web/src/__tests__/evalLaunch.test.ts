/** 发起评测面板的派生逻辑。
 *
 * 只测"界面会不会说错话"的那几个纯函数：进度比例在总数未知时必须是 null（而不是 0%）、
 * 表单空值与 0 的区别、排队与等锁两种"还没在跑"的措辞。
 * 渲染快照不测——与评测页其余部分同一个理由。
 */
import { describe, expect, it } from 'vitest'
import {
  buildBody, isLive, keepSplit, progressRatio, splitChoices, toNumber, waitNote, type Form,
} from '../pages/EvalLaunch'
import type { ProgressView } from '../api/types'

function form(over: Partial<Form> = {}): Form {
  return {
    task: 'intent_classification', model: 'qwen3.5:9b', dataset: '', split: '', k: '1',
    limit: '', seed: '', notes: '', unloadOthers: false, ...over,
  }
}

function progress(over: Partial<ProgressView> = {}): ProgressView {
  return {
    run_id: 'r1', source: 'service', state: 'running', task: 'intent_classification',
    model: 'm', done: 1, total: 4, case_id: '', verdict: '', position: 0, holder: '',
    eta_s: null, waited_s: 0, error: '', reason: '', queued_at: '', started_at: '',
    finished_at: null, cancellable: true, n_error: 0, dataset_id: null, ...over,
  }
}

describe('buildBody', () => {
  it('空字符串一律发 null，让后端用任务默认', () => {
    const body = buildBody(form())
    expect(body.dataset).toBeNull()
    expect(body.limit).toBeNull()
    expect(body.seed).toBeNull()
    expect(body.split).toBe('default')
  })

  it('数据集下拉的「任务默认」这一项必须发 null 而不是空串', () => {
    // 空串会被后端当成一个未知的数据集名，报错信息也就指不到真正的原因
    expect(buildBody(form({ dataset: '' })).dataset).toBeNull()
    expect(buildBody(form({ dataset: 'intent_zh' })).dataset).toBe('intent_zh')
  })

  it('k 没填才是 1，填了 0 就发 0 给后端拒绝', () => {
    // 悄悄把 0 补成 1 会变成"按下按钮做的不是填的那件事"
    expect(buildBody(form({ k: '' })).k).toBe(1)
    expect(buildBody(form({ k: '0' })).k).toBe(0)
    expect(buildBody(form({ k: '3' })).k).toBe(3)
  })

  it('勾了就发 true', () => {
    expect(buildBody(form({ unloadOthers: true })).unload_others).toBe(true)
  })
})

describe('toNumber', () => {
  it('区分「没填」与「填了但不是数」', () => {
    expect(toNumber('')).toBeNull()
    expect(toNumber('  ')).toBeNull()
    expect(toNumber('abc')).toBeNull()
    expect(toNumber('0')).toBe(0)
    expect(toNumber('12')).toBe(12)
  })
})

describe('progressRatio', () => {
  it('总数未知时返回 null，不返回 0', () => {
    // 0% 会被读成"一条都没跑完"，而真相是"还不知道总共多少条"
    expect(progressRatio(null)).toBeNull()
    expect(progressRatio(progress({ state: 'queued', done: 0, total: 0 }))).toBeNull()
  })

  it('比例落在 0..1', () => {
    expect(progressRatio(progress({ done: 2, total: 4 }))).toBe(0.5)
    expect(progressRatio(progress({ done: 9, total: 4 }))).toBe(1)
    expect(progressRatio(progress({ done: -1, total: 4 }))).toBe(0)
  })
})

describe('isLive', () => {
  it('只有排队与运行中算活着', () => {
    expect(['queued', 'running'].every(isLive)).toBe(true)
    expect(['done', 'cancelled', 'skipped', 'error'].every(isLive)).toBe(false)
  })
})

describe('waitNote', () => {  it('排队要说前面还有几个', () => {
    expect(waitNote(progress({ state: 'queued', position: 3 }))).toContain('前面还有 3 个')
    expect(waitNote(progress({ state: 'queued', position: 0 }))).toBe('排队中')
  })

  it('等锁要说谁在占与还要多久；ETA 未知就说未知', () => {
    const waiting = waitNote(progress({ state: 'running', holder: 'cli-eval', waited_s: 12.4, eta_s: 30 })
    )
    expect(waiting).toContain('cli-eval')
    expect(waiting).toContain('30s')
    expect(waiting).toContain('已等 12s')
    expect(waitNote(progress({ state: 'running', holder: 'cli-eval', eta_s: null }))).toContain('进度未知')
  })

  it('正常在跑时不编造一句等待', () => {
    expect(waitNote(progress({ state: 'running', holder: '' }))).toBe('')
    expect(waitNote(null)).toBe('')
  })
})

describe('splitChoices / keepSplit', () => {
  const taskSource = { splits: { default: 236, hard: 32, 转账: 73 } }
  const miniSource = { splits: { default: 3, hard: 1 } }

  it('default 永远不重复出现在选项里', () => {
    expect(splitChoices(taskSource)).toEqual(['hard', '转账'])
    expect(splitChoices(null)).toEqual([])
  })

  it('换了数据集后不存在的子集要回到 default', () => {
    // 选了 3 条的小集合却还能选"转账（73）"，跑起来只会得到一条"没有这个子集"的 error
    expect(keepSplit('转账', miniSource)).toBe('')
    expect(keepSplit('hard', miniSource)).toBe('hard')
    expect(keepSplit('', taskSource)).toBe('')
    expect(keepSplit('hard', null)).toBe('')
  })

  it('选项跟着实际要用的那份数据走', () => {
    expect(splitChoices(miniSource)).toEqual(['hard'])
  })
})
