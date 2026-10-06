/** 应用外壳：左 rail（可开合）+ 顶栏 + 路由。 */
import { useCallback, useEffect, useState } from 'react'
import { api } from './api/client'
import type { HealthView } from './api/types'
import { StatusDot } from './components/primitives'
import { UNKNOWN, timeAgo } from './format'
import { useApi, type AsyncState } from './hooks/useApi'
import { usePersistentFlag } from './hooks/usePersistentFlag'
import { navigate, useRoute } from './router'
import { DatasetsPage } from './pages/Datasets'
import { EvalMatrixPage } from './pages/EvalMatrix'
import { EvalRunsPage } from './pages/EvalRuns'
import { FleetPage } from './pages/Fleet'
import { ModelsPage } from './pages/Models'
import { PlaygroundPage } from './pages/Playground'
import { ToolBenchPage } from './pages/ToolBench'
import { RegressionPage } from './pages/Regression'
import { TracesPage } from './pages/Traces'
import { TraceDetailPage } from './pages/TraceDetail'
import { UsagePage } from './pages/Usage'

/** 评测子页。之前矩阵与回归只能手敲 hash 才到得了——四个页共用一条导航才说得过去 */
const EVAL_TABS: Array<{ path: string; label: string }> = [
  { path: '/eval', label: '运行与发起' },
  { path: '/eval/matrix', label: '矩阵' },
  { path: '/eval/regression', label: '回归对比' },
  { path: '/eval/datasets', label: '数据集' },
]

const EVAL_NOTE = '发起评测与导入数据集都在这里；被中断的运行不进矩阵，半截分数没有可比性'

interface NavItem {
  path: string
  glyph: string
  label: string
  /** 子页在 rail 展开时挂在本项下面；收起态回落到内容区的 subnav，否则子页又只能手敲 hash 才到得了 */
  children?: Array<{ path: string; label: string }>
  note?: string
}

const NAV: NavItem[] = [
  { path: '/fleet', glyph: '◫', label: 'Fleet 总览' },
  { path: '/models', glyph: '▤', label: '模型' },
  { path: '/playground', glyph: '▶', label: 'Playground' },
  { path: '/traces', glyph: '≡', label: 'Traces' },
  { path: '/usage', glyph: '∿', label: 'Token Ledger' },
  { path: '/tools', glyph: '⚒', label: 'Tool Bench' },
  { path: '/eval', glyph: '◎', label: '评测', children: EVAL_TABS, note: EVAL_NOTE },
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
  const [railOpen, toggleRail] = usePersistentFlag('onyx.rail-open', true)
  const [engineOpen, toggleEngine] = usePersistentFlag('onyx.engine-open', true)
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
  } else if (page === 'tools') {
    content = <ToolBenchPage />
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

  // rail 收起时子页在 rail 里没有容身之处，退回内容区顶部——不能让它们又变成只能手敲 hash
  const subnavInContent = evalTab !== '' && !railOpen

  return (
    <div className="shell">
      <nav className={`rail${railOpen ? ' open' : ''}`} aria-label="主导航">
        <div className="rail-head">
          <span className="rail-logo" aria-hidden="true">◈</span>
          <button
            className="rail-toggle"
            onClick={toggleRail}
            aria-expanded={railOpen}
            title={railOpen ? '收起侧边栏（只留图标）' : '展开侧边栏'}
          >
            <span className="rail-glyph" aria-hidden="true">{railOpen ? '◀' : '▶'}</span>
            {railOpen ? <span className="rail-label">收起</span> : null}
          </button>
        </div>

        <div className="rail-nav">
          {NAV.map((item) => {
            const active = activePath === item.path
            return (
              <div className="rail-group" key={item.path}>
                <button
                  className={`rail-item${active ? ' active' : ''}`}
                  onClick={() => navigate(item.path)}
                  title={item.label}
                  aria-label={item.label}
                  aria-current={active ? 'page' : undefined}
                >
                  <span className="rail-glyph" aria-hidden="true">{item.glyph}</span>
                  {railOpen ? <span className="rail-label">{item.label}</span> : null}
                </button>
                {railOpen && item.children ? (
                  <div className="rail-sub">
                    {item.children.map((child) => (
                      <button
                        key={child.path}
                        className={`rail-sub-item${evalTab === child.path ? ' active' : ''}`}
                        onClick={() => navigate(child.path)}
                        aria-current={evalTab === child.path ? 'page' : undefined}
                      >
                        {child.label}
                      </button>
                    ))}
                    {item.note ? <p className="rail-sub-note">{item.note}</p> : null}
                  </div>
                ) : null}
              </div>
            )
          })}
        </div>

        <EngineStatus health={health} railOpen={railOpen} open={engineOpen}
          onToggle={toggleEngine} onExpandRail={toggleRail} />
      </nav>

      <div className="main">
        <header className="topbar">
          <span className="topbar-title">Onyx</span>
          <span className="topbar-meta">本地大模型观测与评测</span>
          <span className="topbar-spacer" />
          <button className="btn" onClick={toggleTheme} title="切换亮/暗主题">
            {theme === 'dark' ? '☾' : '☀'}
          </button>
        </header>
        <main className="content">
          {subnavInContent ? (
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
              <span className="subnav-note">{EVAL_NOTE}</span>
            </nav>
          ) : null}
          {content}
        </main>
      </div>
    </div>
  )
}

/** rail 底部常驻的引擎状态块：收起态只剩状态灯，展开侧栏后才有通道身份与数据新鲜度。
 *  结论词（可达/不可达/连接中…）跟着灯一起走，不藏在一跳之后（R4 颜色不是唯一通道）。 */
function EngineStatus({
  health, railOpen, open, onToggle, onExpandRail,
}: {
  health: AsyncState<HealthView>
  railOpen: boolean
  open: boolean
  onToggle: () => void
  onExpandRail: () => void
}) {
  const d = health.data
  /** null = 还没拿到数据。未知不能画成红色的「不可达」，那是两件事（R2） */
  const reachable = d ? d.provider_reachable : null
  const verdict = d
    ? (d.provider_reachable ? '可达' : '不可达')
    : health.error ? 'API 不可达' : '连接中…'

  if (!railOpen) {
    return (
      <button
        className="rail-engine-dot"
        onClick={onExpandRail}
        title={`引擎状态：${verdict} — 展开侧边栏看详情`}
      >
        <StatusDot ok={reachable} />
      </button>
    )
  }

  return (
    <div className="rail-engine">
      <button
        className="rail-engine-head"
        onClick={onToggle}
        aria-expanded={open}
        title={open ? '收起引擎状态' : '展开引擎状态'}
      >
        <StatusDot ok={reachable} />
        <span className="rail-label">引擎状态</span>
        <span className="rail-engine-verdict">· {verdict}</span>
        <span className="rail-engine-caret" aria-hidden="true">{open ? '▾' : '▸'}</span>
      </button>
      {open ? (
        <div className="rail-engine-body">
          <div className="rail-engine-line">
            {`${d?.provider_id || UNKNOWN} · ${d?.provider_kind || UNKNOWN}`}
          </div>
          <div className="rail-engine-line">
            {`引擎 v${d?.engine_version || UNKNOWN} · schema v${d?.schema_version ?? UNKNOWN}`}
          </div>
          <div className="rail-engine-line">{d?.base_url || UNKNOWN}</div>
          <div className="rail-engine-line">
            {`数据 ${health.fetchedAt ? timeAgo(new Date(health.fetchedAt).toISOString()) : UNKNOWN} · 每 15s 刷新`}
          </div>
        </div>
      ) : null}
    </div>
  )
}
