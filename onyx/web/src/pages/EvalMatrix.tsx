/** 模型 × 任务矩阵。
 *
 * 矩阵最容易骗人的地方：它把不可比的格子画得像可比。
 * 所以这一页把"数据集来历"和"警告"放在网格上方，而不是折叠起来。
 */
import { useMemo } from 'react'
import { api } from '../api/client'
import type { MatrixCell, MatrixView } from '../api/types'
import { EmptyState, ErrorState, Panel, Skeleton } from '../components/primitives'
import { fmtCi, fmtInt, fmtRate, fmtScore, shortId, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

/** 覆盖率低于它的格子要点名——与后端 THIN_COVERAGE 保持同一个阈值。 */
export const THIN_COVERAGE = 0.5

export function cellReadout(cell: MatrixCell): string {
  if (cell.value === null) return UNKNOWN
  const span = cell.ci ? ` ${fmtCi(cell.ci)}` : ''
  return `${fmtScore(cell.value)}${span}`
}

export function cellIsThin(cell: MatrixCell): boolean {
  return cell.coverage !== null && cell.coverage < THIN_COVERAGE
}

function GridCell({ cell }: { cell: MatrixCell | null }) {
  if (!cell) return <td className="cell-empty">{UNKNOWN}</td>
  if (cell.value === null) {
    return (
      <td className="cell-unknown" title="没有可判定样本：这不是 0 分">
        <span className="cell-dim">{cell.metric} {UNKNOWN}</span>
      </td>
    )
  }
  return (
    <td className={cellIsThin(cell) ? 'cell-thin' : undefined}>
      <button className="linklike" onClick={() => navigate(`/eval/run/${cell.run_id}`)}
        title={`跳到这次运行：${cell.run_id}`}>
        <b className="num">{fmtScore(cell.value)}</b>
        {cell.ci ? <span className="cell-dim small num"> {fmtCi(cell.ci)}</span> : null}
      </button>
      <div className="row small cell-dim">
        <span className="mono">{cell.metric}</span>
        {cell.low_confidence ? <span className="badge badge-warn">⚠低样本</span> : null}
        {cellIsThin(cell) ? (
          <span className="badge badge-warn"
            title="主分数只统计到这些样本；正文为空或格式不合规会把样本挤出可判定集合">
            可判定 {fmtInt(cell.n_judged)}/{fmtInt(cell.n_total)}（{fmtRate(cell.coverage)}）
          </span>
        ) : null}
      </div>
    </td>
  )
}

export function EvalMatrixPage() {
  const matrix = useApi<MatrixView>(() => api.matrix(), { intervalMs: 30_000 })
  const view = matrix.data

  const grid = useMemo(() => {
    if (!view) return null
    const byKey = new Map(view.cells.map((c) => [`${c.model_id}\u0000${c.task_id}`, c]))
    return {
      tasks: view.tasks,
      rows: view.models.map((model) => ({
        model,
        cells: view.tasks.map((task) => byKey.get(`${model}\u0000${task}`) ?? null),
      })),
    }
  }, [view])

  return (
    <Panel
      title="模型 × 任务矩阵"
      note={view ? `${view.cells.length} 格 · 数据集 ${view.provenance.join('、') || UNKNOWN}` : '加载中'}
      actions={
        <span className="row">
          <a className="btn" href="/api/matrix" target="_blank" rel="noreferrer">JSON</a>
          <button className="btn" onClick={matrix.refresh}>刷新</button>
        </span>
      }
    >
      {matrix.error ? <ErrorState error={matrix.error} /> : null}
      {matrix.loading && !view ? <Skeleton rows={5} /> : null}
      {view && view.cells.length === 0 ? (
        <EmptyState title="没有 done 状态的运行" hint="先跑一次：onyx eval run --task intent_classification --model …" />
      ) : null}
      {grid ? (
        <div className="table-wrap">
          <table className="matrix">
            <thead>
              <tr>
                <th>模型</th>
                {grid.tasks.map((task) => (
                  <th key={task}>{task}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {grid.rows.map((row) => (
                <tr key={row.model}>
                  <th className="mono">{row.model}</th>
                  {row.cells.map((cell, index) => (
                    <GridCell key={grid.tasks[index]} cell={cell} />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {view?.warnings.length ? (
        <div className="warnbox">
          <b>可比性警告</b>
          <ul>
            {view.warnings.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      ) : null}
      <p className="note">
        每格是<b>该模型在该任务上最新一次 done 运行</b>的主分数（不是历史最好成绩）。
        未完成与取消的运行不进网格——半截分数没有可比性。点格子跳到那次运行。
      </p>
    </Panel>
  )
}

/** 供回归页复用的 run 摘要行。 */
export function runLabel(run: { id: string; model_id: string; task_id: string; started_at: string }): string {
  return `${run.model_id} · ${run.task_id} · ${shortId(run.id, 8)}`
}
