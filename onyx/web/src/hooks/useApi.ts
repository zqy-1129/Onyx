/** 数据获取 hook：加载 / 错误 / 轮询 / 手动刷新。
 *  错误不吞掉：交给 ErrorState 渲染 code + hint，用户才知道该去启服务还是改查询。 */
import { useCallback, useEffect, useRef, useState } from 'react'

export interface AsyncState<T> {
  data: T | null
  error: unknown
  loading: boolean
  /** 上次成功刷新的时间戳，用于顶栏"数据新鲜度"提示 */
  fetchedAt: number | null
  refresh: () => void
}

export function useApi<T>(
  loader: () => Promise<T>,
  { intervalMs, deps = [] }: { intervalMs?: number; deps?: unknown[] } = {},
): AsyncState<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const [fetchedAt, setFetchedAt] = useState<number | null>(null)
  const [nonce, setNonce] = useState(0)
  const alive = useRef(true)

  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    loader()
      .then((result) => {
        if (cancelled || !alive.current) return
        setData(result)
        setError(null)
        setFetchedAt(Date.now())
      })
      .catch((err) => {
        if (cancelled || !alive.current) return
        setError(err)
      })
      .finally(() => {
        if (!cancelled && alive.current) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nonce, ...deps])

  useEffect(() => {
    if (!intervalMs) return
    const timer = window.setInterval(() => setNonce((n) => n + 1), intervalMs)
    return () => window.clearInterval(timer)
  }, [intervalMs])

  const refresh = useCallback(() => setNonce((n) => n + 1), [])
  return { data, error, loading, fetchedAt, refresh }
}
