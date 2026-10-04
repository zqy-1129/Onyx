/** 应用外壳：左 rail + 顶栏 + 路由。 */
import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { HealthView } from './api/types'
import { StatusDot } from './components/primitives'
import { UNKNOWN } from './format'
import { useApi } from './hooks/useApi'
import { navigate, useRoute } from './router'
import { DatasetsPage } from './pages/Datasets'
import { EvalMatrixPage } from './pages/EvalMatrix'
import { EvalRunsPage } from './pages/EvalRuns'
import { FleetPage } from './pages/Fleet'
import { ModelsPage } from './pages/Models'
import { PlaygroundPage } from './pages/Playground'
import { RegressionPage } from './pages/Regression'
import { TracesPage } from './pages/Traces'
import { TraceDetailPage } from './pages/TraceDetail'
import { UsagePage } from './pages/Usage'

interface NavItem {
  path: string
  glyph: string
  label: string
}

const NAV: NavItem[] = [
  { path: '/fleet', glyph: '◫', label: 'Fleet 总览' },
  { path: '/models', glyph: '▤', label: '模型' },
  { path: '/playground', glyph: '▶', label: 'Playground' },
  { path: '/traces', glyph: '≡', label: 'Traces' },
  { path: '/usage', glyph: '∿', label: 'Token Ledger' },
  { path: '/eval', glyph: '◎', label: '评测' },
]

/** 评测子页。之前矩阵与回归只能手敲 hash 才到得了——四个页共用一条导航才说得过去 */
const EVAL_TABS: Array<{ path: string; label: string; note?: string }> = [
  { path: '/eval', label: '运行与发起' },
  { path: '/eval/matrix', label: '矩阵' },
  { path: '/eval/regression', label: '回归对比' },
  { path: '/eval/datasets', label: '数据集' },
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
  } else if (page === 'playground') {
    content = <PlaygroundPage />
  } else if (page === 'usage') {
    content = <UsagePage />
  } else if (page === 'eval' && segments[1] === 'matrix') {
    content = <EvalMatrixPage />
  } else if (page === 'eval' && segments[1] === 'regression') {
    content = <RegressionPage />
  } else if (page === 'eval' && segments[1] === 'datasets') {
    content = <DatasetsPage />
  } else if (page === 'eval') {
    // /eval/run/<id> 直接落到运行页并选中它：矩阵与 diff 都靠这个链接下钻
    content = (
      <EvalRunsPage
        selectedRunId={segments[1] === 'run' && segments[2] ? decodeURIComponent(segments[2]) : undefined}
      />
    )
  } else {
    content = <FleetPage />
  }

  const evalTab = page === 'eval'
    ? (segments[1] === 'matrix' || segments[1] === 'regression' || segments[1] === 'datasets'
        ? `/eval/${segments[1]}`
        : '/eval')
    : ''

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
        <main className="content">
          {evalTab ? (
            <nav className="subnav">
              {EVAL_TABS.map((tab) => (
                <button
                  key={tab.path}
                  className={`subnav-item${evalTab === tab.path ? ' active' : ''}`}
                  onClick={() => navigate(tab.path)}
                >
                  {tab.label}
                </button>
              ))}
              <span className="subnav-note">
                发起评测与导入数据集都在这里；被中断的运行不进矩阵，半截分数没有可比性
              </span>
            </nav>
          ) : null}
          {content}
        </main>
      </div>
    </div>
  )
}
