/** token 在前端的落脚点：URL 收进 sessionStorage、fetch 走 header、SSE 走 query。
 *
 * 这三条路各有各的坑（EventSource 设不了头、隐私模式 sessionStorage 会抛、
 * 缓存会让"换了 token 不生效"），所以逐条钉住而不是只测"能拼出字符串"。
 */
import { beforeEach, describe, expect, it } from 'vitest'

import { apiToken, authHeaders, withToken } from '../api/token'

function at(url: string): void {
  window.history.replaceState({}, '', url)
}

describe('apiToken', () => {
  beforeEach(() => {
    window.sessionStorage.clear()
    at('/')
  })

  it('没有 token 时什么都不加，不塞空 header', () => {
    expect(apiToken()).toBeNull()
    expect(authHeaders({ 'content-type': 'application/json' })).toEqual({
      'content-type': 'application/json',
    })
    expect(withToken('/api/stream')).toBe('/api/stream')
  })

  it('URL 上的 ?token= 优先，并留到本会话', () => {
    at('/?token=abc')
    expect(apiToken()).toBe('abc')

    at('/')
    // 刷新后不该又要一次 token（sessionStorage 的作用就在这）
    expect(apiToken()).toBe('abc')
    expect(authHeaders()).toEqual({ authorization: 'Bearer abc' })
  })

  it('换 URL 上的 token 立刻生效（不做进程内缓存）', () => {
    at('/?token=old')
    expect(apiToken()).toBe('old')
    at('/?token=new')
    expect(apiToken()).toBe('new')
  })

  it('SSE 只能把 token 放 query，并转义已有参数', () => {
    at('/?token=a%2Fb')

    expect(withToken('/api/stream')).toBe('/api/stream?token=a%2Fb')
    expect(withToken('/api/x?b=1')).toBe('/api/x?b=1&token=a%2Fb')
  })

  it('sessionStorage 被禁用时仍然工作（隐私模式）', () => {
    at('/?token=abc')
    const original = Object.getOwnPropertyDescriptor(window, 'sessionStorage')
    Object.defineProperty(window, 'sessionStorage', {
      configurable: true,
      get() {
        throw new Error('denied')
      },
    })
    try {
      // 存不下不该让读取失败：隐私模式下 URL 上的值仍然可用
      expect(apiToken()).toBe('abc')
      expect(withToken('/api/stream')).toContain('token=abc')
    } finally {
      // 必须还原成真 Storage：后面的用例还要 sessionStorage.clear()
      if (original) Object.defineProperty(window, 'sessionStorage', original)
    }
  })
})
