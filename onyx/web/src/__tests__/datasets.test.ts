/** 数据集导入表单的派生逻辑（S24）。
 *
 * 盯的是"界面会不会替人做决定"：空 id 必须发 null（空串会被当成一个真的 id 去登记）、
 * 覆盖确认位不能由前端替人勾上、文件名转 name 的规则要和 case id 前缀对得上。
 */
import { describe, expect, it } from 'vitest'
import { buildImportBody, defaultName, isConflict, EMPTY_FORM, type ImportForm } from '../pages/Datasets'

function form(over: Partial<ImportForm> = {}): ImportForm {
  return { ...EMPTY_FORM, ...over }
}

describe('buildImportBody', () => {
  it('留空的 id 发 null，而不是空字符串', () => {
    // 空串会被后端当成"真的要登记一个空 id"，那是个查不出来的键
    const body = buildImportBody(form({ jsonl: '{}', id: '   ' }))
    expect(body.id).toBeNull()
    expect(body.name).toBe('uploaded')
  })

  it('其余文本字段一律去空白', () => {
    const body = buildImportBody(form({
      jsonl: 'x', name: ' mini ', upstream: ' repo ', revision: ' r1 ', license: ' MIT ',
    }))
    expect(body).toMatchObject({ name: 'mini', upstream: 'repo', revision: 'r1', license: 'MIT' })
  })

  it('覆盖确认位照人勾的值发出去，前端不替人勾', () => {
    expect(buildImportBody(form({ jsonl: 'x' })).allow_replace).toBe(false)
    expect(buildImportBody(form({ jsonl: 'x', allowReplace: true })).allow_replace).toBe(true)
  })

  it('jsonl 原样发送（行内空格与中文都是数据）', () => {
    const text = '{"input": {"instruction": "帮我转 500 元"}, "expect": {"label": "转账"}}'
    expect(buildImportBody(form({ jsonl: text })).jsonl).toBe(text)
  })
})

describe('defaultName', () => {
  it('去掉数据文件扩展名', () => {
    expect(defaultName('intents.jsonl')).toBe('intents')
    expect(defaultName('a.ndjson')).toBe('a')
    expect(defaultName('b.JSON')).toBe('b')
  })

  it('没有可扩展名时不返回空字符串', () => {
    // name 决定未来自派生 case id 的前缀，空前缀会让 id 长得像乱码
    expect(defaultName('.jsonl')).toBe('uploaded')
    expect(defaultName('keep')).toBe('keep')
  })
})

describe('isConflict', () => {
  it('只有 409 算冲突', () => {
    expect(isConflict({ status: 409 })).toBe(true)
    expect(isConflict({ status: 422 })).toBe(false)
    expect(isConflict(null)).toBe(false)
    expect(isConflict(undefined)).toBe(false)
    expect(isConflict(new Error('boom'))).toBe(false)
  })
})
