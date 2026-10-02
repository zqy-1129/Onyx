/** 极简 hash 路由。
 *  不引 react-router：看板只有 7 个页面、无嵌套布局、无 loader，
 *  一个 60 行的 hash 路由足够，且少一个依赖。 */
import { useCallback, useEffect, useState } from 'react'

export interface Route {
  path: string
  segments: string[]
}

function parse(): Route {
  const raw = window.location.hash.replace(/^#/, '') || '/fleet'
  const path = raw.split('?')[0]
  return { path, segments: path.split('/').filter(Boolean) }
}

export function navigate(to: string): void {
  if (window.location.hash === `#${to}`) return
  window.location.hash = to
}

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(parse)
  useEffect(() => {
    const onChange = () => setRoute(parse())
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

/** 页面内跳转的稳定回调（避免每次渲染新建函数导致子组件重渲染）。 */
export function useNavigate(): (to: string) => void {
  return useCallback((to: string) => navigate(to), [])
}
