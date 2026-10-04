/** 评测运行列表 + 逐条 grade 下钻。
 *
 * 这一页的存在理由只有一条：**每个分数都能点进一条真实 trace**。
 * 因此 grade 表里的 trace 列必须可点，而不是把 id 印出来让人自己抄。
 */
import { useCallback, useEffect, useState } from 'react'
import { api } from '../api/client'
import type { CIView, GradeView, GpuStatusView, RunView } from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { EmptyState, ErrorState, Panel, Skeleton, StatusBadge } from '../components/primitives'
import { fmtClock, fmtInt, fmtScore, fmtCi, shortId, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'
import { EvalLaunchPanel } from './EvalLaunch'
import { THIN_COVERAGE } from './EvalMatrix'

/** 主分数：按与后端 HEADLINE_METRICS 相同的优先级挑第一个出现过的键。
 *  取"出现过"而不是"非空"——值为 null 表示未定义，它仍然是这个任务的主分数。 */
const HEADLINE = ['macro_f1', 'accuracy', 'must_call_acc', 'pass_hat_k', 'score']

export function headlineOf(aggregate: Record<string, unknown>): { key: string; value: number | null } | null {
  for (const key of HEADLINE) {
    if (key in aggregate) {
      const raw = aggregate[key]
      return { key, value: typeof raw === 'number' ? raw : null }
    }
  }
  return null
}

export function headlineCi(aggregate: Record<string, unknown>, key: string): CIView | null {
  const ci = aggregate[`${key}_ci`]
  return ci && typeof ci === 'object' ? (ci as CIView) : null
}

export function headlineText(aggregate: Record<string, unknown>): string {
  const picked = headlineOf(aggregate)
  if (!picked) return UNKNOWN
  if (picked.value === null) return `${picked.key} ${UNKNOWN}`
  const ci = headlineCi(aggregate, picked.key)
  const span = ci ? ` ${fmtCi(ci)}` : ''
  return `${picked.key} ${fmtScore(picked.value)}${span}`
}

/** 主分数只覆盖少数样本时给出一句说明。与矩阵用同一个阈值与同一份口径。 */
export function thinNote(aggregate: Record<string, unknown>): string | null {
  const judged = typeof aggregate.n_judged === 'number' ? aggregate.n_judged : null
  const total = typeof aggregate.n_total === 'number' ? aggregate.n_total : null
  if (!judged || !total || judged / total >= THIN_COVERAGE) return null
  return `可判定 ${judged}/${total}`
}

function HeadlineCell({ run }: { run: RunView }) {
  const picked = headlineOf(run.aggregate)
  const unknown = picked?.value === null || picked === undefined
  const ci = picked ? headlineCi(run.aggregate, picked.key) : null
  const thin = thinNote(run.aggregate)
  return (
    <span title={ci ? `区间重采样单位 n=${ci.n}` : undefined}>
      {headlineText(run.aggregate)}
      {/* 低样本必须常驻显示：一个没有区间的数字与一个"测得不够"的数字，修法完全不同 */}
      {run.aggregate.low_confidence === true && !unknown ? <span className="badge badge-warn">⚠低样本</span> : null}
      {thin ? (
        <span className="badge badge-warn" title="主分数只统计到这些样本；正文为空/格式不合规会把样本挤出可判定集合">
          {thin}
        </span>
      ) : null}
      {unknown ? <span className="cell-dim small">（无可判定样本）</span> : null}
    </span>
  )
}

const columns: Array<Column<RunView>> = [
  { key: 'id', header: 'run', mono: true, render: (r) => shortId(r.id, 10), sortValue: (r) => r.id },
  { key: 'task', header: '任务', render: (r) => r.task_id, sortValue: (r) => r.task_id },
  { key: 'model', header: '模型', mono: true, render: (r) => r.model_id, sortValue: (r) => r.model_id },
  { key: 'status', header: '状态', render: (r) => <StatusBadge status={r.status} /> },
  {
    key: 'n',
    header: '完成',
    align: 'right',
    mono: true,
    render: (r) => (
      <span className={r.n_done < r.n_cases ? 'badge badge-warn' : undefined}>
        {fmtInt(r.n_done)}/{fmtInt(r.n_cases)}
      </span>
    ),
    sortValue: (r) => r.n_done,
  },
  { key: 'headline', header: '主分数', render: (r) => <HeadlineCell run={r} />, sortValue: (r) => {
      const picked = headlineOf(r.aggregate)
      return picked?.value ?? null
    } },
  {
    key: 'cost',
    header: '成本',
    align: 'right',
    mono: true,
    render: (r) => (
      <span title="评测自身的开销：它同样是 GPU 时间">
        {fmtInt(r.cost.in_tokens)} in · {fmtInt(r.cost.out_tokens)} out ·{' '}
        {fmtInt(r.cost.requests)} 次
      </span>
    ),
  },
  {
    key: 'data',
    header: '数据集',
    render: (r) => (
      <span className="cell-dim small" title={r.dataset_revision}>
        {r.dataset_id ?? UNKNOWN}
      </span>
    ),
  },
  { key: 'time', header: '开始', mono: true, render: (r) => fmtClock(r.started_at), sortValue: (r) => r.started_at },
]

export function EvalRunsPage({ selectedRunId }: { selectedRunId?: string }) {
  const runs = useApi<RunView[]>(() => api.evalRuns({ limit: 50 }), { intervalMs: 20_000 })
  const gpu = useApi<GpuStatusView>(() => api.gpu(), { intervalMs: 5_000 })
  const [picked, setPicked] = useState<string | null>(selectedRunId ?? null)
  // 路由优先：矩阵与 diff 页都靠 /eval/run/<id> 下钻，进来就必须选中那一行。
  // 只把 selectedRunId 当初始值的话，从别的页面跳进来时会停在上一次选中的运行上。
  useEffect(() => {
    if (selectedRunId) setPicked(selectedRunId)
  }, [selectedRunId])
  const active = picked ?? selectedRunId ?? runs.data?.[0]?.id ?? null

  // 稳定的回调：轮询进度那个 effect 把它列在依赖里，
  // 每次渲染新建一个箭头函数会让轮询从头开始，进度条反而变成每帧重启
  const pickRun = useCallback((id: string) => setPicked(id), [])

  const list = runs.data ?? []

  return (
    <>
      {gpu.data?.busy ? (
        <div className="banner banner-warn">
          GPU 正被 <b>{gpu.data.owner}</b> 占用（进度 {gpu.data.progress ?? UNKNOWN}）
          {gpu.data.eta_s !== null ? `，预计还需 ${Math.round(gpu.data.eta_s)}s` : '，进度未知'}
          <span className="cell-dim small">
            　评测与 Playground 都排队，不并发跑：并发不会报错，只会让所有延迟数字失真
          </span>
        </div>
      ) : null}
      <EvalLaunchPanel onRefresh={runs.refresh} onFinished={pickRun} />
      <Panel
        title="运行"
        note="每行是一次 eval run；点行看它的 grade"
        actions={<button className="btn" onClick={runs.refresh}>刷新</button>}
        flush
      >
        {runs.error ? <ErrorState error={runs.error} /> : null}
        {runs.loading && !runs.data ? <Skeleton rows={6} /> : null}
        {runs.data && runs.data.length === 0 ? (
          <EmptyState
            title="还没有评测运行"
            hint="用上方「发起评测」填任务与模型点开始，或 onyx eval run --task intent_classification --model qwen3.5:9b --seed 42"
          />
        ) : null}
        {runs.data && runs.data.length > 0 ? (
          <DataTable
            columns={columns}
            rows={list}
            rowKey={(r) => r.id}
            onRowClick={(r) => setPicked(r.id)}
            selectedKey={active}
            maxHeight="calc(42vh)"
          />
        ) : null}
      </Panel>
      {active ? <GradesPanel runId={active} /> : null}
    </>
  )
}

/** 判定 → 颜色。R4：颜色不是唯一通道，文字本身就在 badge 里。
 *  R5：红只表示"这次请求失败了"（error/timeout），不表示"模型答错了"——
 *  后者是测量结果，不是故障，混成红色会让人以为评测本身坏了。 */
const WARN_VERDICTS = new Set([
  'wrong', 'no_call', 'wrong_tool', 'bad_args', 'invalid_format', 'out_of_label',
  'hallucinated_tool', 'false_call', 'partial',
])

export function verdictBadge(verdict: string): string {
  if (verdict === 'correct' || verdict === 'pass') return 'badge badge-ok'
  if (verdict === 'error' || verdict === 'timeout') return 'badge badge-error'
  if (verdict === 'skipped') return 'badge badge-unknown'
  if (WARN_VERDICTS.has(verdict)) return 'badge badge-warn'
  return 'badge badge-neutral'
}

const VERDICTS = ['', 'correct', 'wrong', 'no_call', 'wrong_tool', 'bad_args', 'invalid_format',
  'out_of_label', 'hallucinated_tool', 'error', 'skipped']

function GradesPanel({ runId }: { runId: string }) {
  const [verdict, setVerdict] = useState('')
  const grades = useApi<GradeView[]>(
    () => api.evalGrades(runId, { verdict: verdict || null, limit: 500 }),
    { deps: [runId, verdict] },
  )
  const cols: Array<Column<GradeView>> = [
    { key: 'case', header: 'case', mono: true, render: (g) => shortId(g.case_id, 16), sortValue: (g) => g.case_id },
    { key: 'seq', header: 'seq', align: 'right', mono: true, render: (g) => g.seq },
    { key: 'verdict', header: '判定', render: (g) => <span className={verdictBadge(g.verdict)}>{g.verdict}</span> },
    { key: 'score', header: '分', align: 'right', mono: true, render: (g) => g.score.toFixed(2), sortValue: (g) => g.score },
    {
      key: 'format',
      header: '格式',
      render: (g) => (g.invalid_format ? <span className="badge badge-warn">不合规</span> : <span className="cell-dim">合规</span>),
    },
    {
      key: 'expected',
      header: '期望',
      render: (g) => <span className="mono small">{String(g.metrics.expected ?? UNKNOWN)}</span>,
    },
    {
      key: 'actual',
      header: '预测',
      render: (g) => <span className="mono small">{String(g.metrics.predicted ?? g.metrics.actual ?? UNKNOWN)}</span>,
    },
    {
      key: 'error',
      header: '说明',
      render: (g) => (g.error ? <span className="small" title={g.error}>{g.error.slice(0, 60)}</span> : null),
    },
    {
      key: 'trace',
      header: 'trace',
      render: (g) =>
        g.trace_id ? (
          <button className="linklike mono" onClick={() => navigate(`/traces/${g.trace_id}`)}
            title="这条分数是从哪次请求来的">
            {shortId(g.trace_id, 10)}
          </button>
        ) : (
          <span className="cell-dim">{UNKNOWN}</span>
        ),
    },
  ]

  return (
    <Panel
      title={`grade · ${shortId(runId, 12)}`}
      note="每条都能跳到产生它的那次请求"
      actions={
        <label className="field">
          <span className="field-label">只看判定</span>
          <select className="select" value={verdict} onChange={(e) => setVerdict(e.target.value)}>
            {VERDICTS.map((v) => (
              <option key={v} value={v}>{v || '全部'}</option>
            ))}
          </select>
        </label>
      }
      flush
    >
      {grades.error ? <ErrorState error={grades.error} /> : null}
      {grades.loading ? <Skeleton rows={5} /> : null}
      {grades.data && grades.data.length === 0 ? (
        <EmptyState title="这个筛选条件下没有 grade" hint="换一个判定，或清空筛选" />
      ) : null}
      {grades.data && grades.data.length > 0 ? (
        <DataTable columns={cols} rows={grades.data} rowKey={(g) => `${g.case_id}-${g.seq}`} maxHeight="46vh" />
      ) : null}
    </Panel>
  )
}
