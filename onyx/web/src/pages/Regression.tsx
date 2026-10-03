/** 配对回归 diff：换一个模型，到底值不值。
 *
 * 这一页刻意**不把均值差当结论**。均值差 0.02 可能是 30 条变好、28 条变坏相互抵消，
 * 那不是"略好"，而是"在两类任务上方向相反"。所以头一行是三个计数，
 * 下面紧跟劣化清单——清单里每条都能并排打开两个模型的那两条 trace。
 */
import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { ComparisonView, PairedCase, RunView } from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { EmptyState, ErrorState, Panel, Skeleton, StatCard } from '../components/primitives'
import { fmtCi, fmtDelta, fmtInt, fmtRate, fmtScore, shortId, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

/** 与后端 LOW_CONFIDENCE_PAIRS 同一个阈值：配对数太少就别拿结论当决策。 */
export const LOW_PAIRS = 30

export function pickRuns(runs: RunView[]): { byTask: Map<string, RunView[]>; done: RunView[] } {
  const done = runs.filter((run) => run.status === 'done')
  const byTask = new Map<string, RunView[]>()
  for (const run of done) {
    const list = byTask.get(run.task_id) ?? []
    list.push(run)
    byTask.set(run.task_id, list)
  }
  return { byTask, done }
}

export function verdictFlips(cases: PairedCase[]): { up: number; down: number } {
  let up = 0
  let down = 0
  for (const item of cases) {
    if (item.passed_base === null || item.passed_target === null) continue
    if (!item.passed_base && item.passed_target) up += 1
    else if (item.passed_base && !item.passed_target) down += 1
  }
  return { up, down }
}

function Summary({ data }: { data: ComparisonView }) {
  const ci = data.delta_ci
  return (
    <div className="grid">
      <div className="col-3">
        <StatCard
          label="配对样本"
          value={fmtInt(data.n_paired)}
          sub={`覆盖率 ${data.coverage === null ? UNKNOWN : fmtRate(data.coverage, 0)}`}
          badge={data.n_paired < LOW_PAIRS ? <span className="badge badge-warn">⚠不足以决策</span> : undefined}
        />
      </div>
      <div className="col-3">
        <StatCard
          label="改善 / 劣化"
          value={<><span className="tone-good">{fmtInt(data.improved)}</span> / <span className="tone-bad">{fmtInt(data.regressed)}</span></>}
          sub={`不变 ${fmtInt(data.unchanged)} · pass^k 翻转 +${data.flips.up}/-${data.flips.down}`}
        />
      </div>
      <div className="col-4">
        <StatCard
          label="均值差（target − base）"
          value={data.mean_delta === null ? UNKNOWN : fmtDelta(data.mean_delta)}
          unknown={data.mean_delta === null}
          /* 区间必须带着它的 n：只看宽度不看样本数，等于没看区间 */
          sub={ci ? `95% CI ${fmtCi(ci)} · n=${fmtInt(ci.n)}` : '配对样本太少，给不出区间'}
        />
      </div>
      <div className="col-2">
        <StatCard
          label="重采样口径"
          value={<span className="small">按 case</span>}
          sub="同一 case 的 k 次采样不算多个独立观测"
        />
      </div>
    </div>
  )
}

function caseColumns(): Array<Column<PairedCase>> {
  const link = (id: string | null, label: string) =>
    id ? (
      <button className="linklike mono small" onClick={() => navigate(`/traces/${id}`)} title={label}>
        {shortId(id, 8)}
      </button>
    ) : (
      <span className="cell-dim">{UNKNOWN}</span>
    )
  return [
    { key: 'case', header: 'case', mono: true, render: (c) => shortId(c.case_id, 14), sortValue: (c) => c.case_id },
    {
      key: 'delta',
      header: 'Δ',
      align: 'right',
      mono: true,
      render: (c) => (
        <span className={c.delta < 0 ? 'tone-bad' : c.delta > 0 ? 'tone-good' : 'cell-dim'}>
          {fmtDelta(c.delta)}
        </span>
      ),
      sortValue: (c) => c.delta,
    },
    { key: 'kind', header: '类型', render: (c) => c.kind, sortValue: (c) => c.kind },
    { key: 'base', header: 'base', render: (c) => (
        <span className="small">{fmtScore(c.score_base)} <span className="cell-dim">{c.verdict_base}</span></span>
      ) },
    { key: 'target', header: 'target', render: (c) => (
        <span className="small">{fmtScore(c.score_target)} <span className="cell-dim">{c.verdict_target}</span></span>
      ) },
    { key: 'text', header: '题干', render: (c) => <span className="small" title={c.instruction}>{c.instruction.slice(0, 40) || UNKNOWN}</span> },
    { key: 'tb', header: 'base trace', render: (c) => link(c.trace_base, 'base 的那次请求') },
    { key: 'tt', header: 'target trace', render: (c) => link(c.trace_target, 'target 的那次请求') },
  ]
}

export function RegressionPage() {
  const runs = useApi<RunView[]>(() => api.evalRuns({ limit: 100 }), { intervalMs: 20_000 })
  const [base, setBase] = useState('')
  const [target, setTarget] = useState('')

  const { byTask, done } = useMemo(() => pickRuns(runs.data ?? []), [runs.data])
  const tasks = useMemo(() => [...byTask.keys()].sort(), [byTask])
  const [task, setTask] = useState('')

  // 任务一换，原来的选择就可能跨数据集了：清空比"留着错配"诚实
  useEffect(() => {
    setBase('')
    setTarget('')
  }, [task])

  const options = byTask.get(task) ?? []
  const ready = Boolean(task && base && target && base !== target)
  const diff = useApi<ComparisonView | null>(
    () => (ready ? api.compare(base, target, { with_cases: true }) : Promise.resolve(null)),
    { deps: [ready, base, target] },
  )

  return (
    <>
      <Panel title="选择要配对的两批运行" note="同一任务、同一数据集才有意义">
        {runs.error ? <ErrorState error={runs.error} /> : null}
        {runs.loading && !runs.data ? <Skeleton rows={2} /> : null}
        {runs.data && done.length < 2 ? (
          <EmptyState title="至少需要两次 done 运行" hint="onyx eval run --task … --model 第二个模型" />
        ) : null}
        <div className="row-wrap">
          <label className="field">
            <span className="field-label">任务</span>
            <select className="select" value={task} onChange={(e) => setTask(e.target.value)}>
              <option value="">（选任务）</option>
              {tasks.map((name) => (
                <option key={name} value={name}>{name}（{byTask.get(name)?.length ?? 0} 次）</option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">base（基准）</span>
            <select className="select" value={base} onChange={(e) => setBase(e.target.value)} disabled={!task}>
              <option value="">（选基准运行）</option>
              {options.map((run) => (
                <option key={run.id} value={run.id}>
                  {run.model_id} · {shortId(run.id, 8)} · {run.dataset_id ?? '无数据集记录'}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span className="field-label">target（对比）</span>
            <select className="select" value={target} onChange={(e) => setTarget(e.target.value)} disabled={!task}>
              <option value="">（选对比运行）</option>
              {options.filter((run) => run.id !== base).map((run) => (
                <option key={run.id} value={run.id}>
                  {run.model_id} · {shortId(run.id, 8)} · {run.dataset_id ?? '无数据集记录'}
                </option>
              ))}
            </select>
          </label>
        </div>
        {!ready ? <p className="note">差值方向是 target − base：负数表示换成 target 之后变差。</p> : null}
      </Panel>

      {ready && diff.loading ? <Panel title="配对中"><Skeleton rows={4} /></Panel> : null}
      {ready && diff.error ? <Panel title="无法比较"><ErrorState error={diff.error} /></Panel> : null}
      {diff.data ? (
        <>
          <Panel title={`${diff.data.base.model_id} → ${diff.data.target.model_id}`}
            note={`${diff.data.target.task_id} · k=${diff.data.base.k ?? 1}`}>
            <Summary data={diff.data} />
            {diff.data.warnings.length ? (
              <div className="warnbox">
                <b>可比性警告</b>
                <ul>{diff.data.warnings.map((note) => <li key={note}>{note}</li>)}</ul>
              </div>
            ) : null}
            {diff.data.only_base.length || diff.data.only_target.length ? (
              <p className="note">
                只有 base 考了 {fmtInt(diff.data.only_base.length)} 条、
                只有 target 考了 {fmtInt(diff.data.only_target.length)} 条——结论只覆盖交集部分。
              </p>
            ) : null}
          </Panel>
          <Panel title="逐题配对" note="按 Δ 排序可直接看到最惨的题" flush>
            <DataTable
              columns={caseColumns()}
              rows={[...diff.data.cases].sort((a, b) => a.delta - b.delta)}
              rowKey={(c) => c.case_id}
              maxHeight="calc(100vh - 430px)"
            />
          </Panel>
        </>
      ) : null}
    </>
  )
}
