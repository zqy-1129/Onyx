/** rail / 引擎状态块开合的持久化：'0' 是用户明确收起过，不能和"没存过"混成一档。 */
import { beforeEach, describe, expect, it } from 'vitest'
import { act, renderHook } from '@testing-library/react'
import { usePersistentFlag } from '../hooks/usePersistentFlag'

describe('usePersistentFlag', () => {
  beforeEach(() => localStorage.clear())

  it('没存过时取默认值并写回', () => {
    const { result } = renderHook(() => usePersistentFlag('onyx.rail-open', true))
    expect(result.current[0]).toBe(true)
    expect(localStorage.getItem('onyx.rail-open')).toBe('1')
  })

  it('存过 0 就是收起，不能回落到默认的展开', () => {
    localStorage.setItem('onyx.rail-open', '0')
    const { result } = renderHook(() => usePersistentFlag('onyx.rail-open', true))
    expect(result.current[0]).toBe(false)
    expect(localStorage.getItem('onyx.rail-open')).toBe('0')
  })

  it('切换后两侧同步', () => {
    const { result } = renderHook(() => usePersistentFlag('onyx.engine-open', true))
    act(() => result.current[1]())
    expect(result.current[0]).toBe(false)
    expect(localStorage.getItem('onyx.engine-open')).toBe('0')
  })
})
