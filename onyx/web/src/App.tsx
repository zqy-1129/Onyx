/** 应用外壳：左 rail + 顶栏 + 路由。 */
import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { HealthView } from './api/types'
import { StatusDot } from './components/primitives'
import { UNKNOWN } from './format'
import { useApi } from './hooks/useApi'
import { navigate, useRoute } from './router'
import { FleetPage } from './pages/Fleet'
import { ModelsPage } from './pages/Models'
import { TracesPage } from './pages/Traces'
import { TraceDetailPage } from './pages/TraceDetail'

interface NavItem {
  path: string
  glyph: string
  label: string
}

const NAV: NavItem[] = [
  { path: '/fleet', glyph: '◫', label: 'Fleet 总览' },
  { path: '/models', glyph: '▤', label: '模型' },
  { path: '/traces', glyph: '≡', label: 'Traces' },
]

function useTheme(): [string, () => void] {
  const [theme, setTheme] = useState<string>(
    () => localStorage.getItem('onyx.theme') ?? 'dark',
  )
  useEffect(() => {
    document.documentElement.className = `onyx-theme-${theme}`
    localStorage.setItem('onyx.theme', theme)
  }, [theme])
  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), [])
  return [theme, toggle]
}

export function App() {
  const route = useRoute()
  const [theme, toggleTheme] = useTheme()
  const health = useApi<HealthView>(() => api.health(), { intervalMs: 15_000 })

  const segments = route.segments
  const page = segments[0] ?? 'fleet'
  const activePath = `/${page}`

  let content: React.ReactNode
  if (page === 'traces' && segments[1]) {
    content = <TraceDetailPage traceId={decodeURIComponent(segments[1])} />
  } else if (page === 'traces') {
    content = <TracesPage />
  } else if (page === 'models') {
    content = <ModelsPage />
  } else {
    content = <FleetPage />
  }

  return (
    <div className="shell">
      <nav className="rail">
        <div className="rail-logo">◈</div>
        {NAV.map((item) => (
          <button
            key={item.path}
            className={`rail-item${activePath === item.path ? ' active' : ''}`}
            onClick={() => navigate(item.path)}
            title={item.label}
          >
            <span className="rail-glyph">{item.glyph}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="main">
        <header className="topbar">
          <span className="topbar-title">Onyx</span>
          <span className="topbar-meta">本地大模型观测与评测</span>
          <span className="topbar-spacer" />
          <span className="row small">
            <StatusDot ok={Boolean(health.data?.provider_reachable)} />
            <span className="topbar-meta">
              {health.data
                ? `引擎 v${health.data.engine_version || UNKNOWN} · schema v${health.data.schema_version}`
                : health.error
                  ? 'API 不可达'
                  : '连接中…'}
            </span>
          </span>
          <button className="btn" onClick={toggleTheme} title="切换亮/暗主题">
            {theme === 'dark' ? '☾' : '☀'}
          </button>
        </header>
        <main className="content">{content}</main>
      </div>
    </div>
  )
}
