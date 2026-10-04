/** 发起评测（S23）：表单 → POST /api/runs → 轮询进度 → 可以取消。
 *
 * 三条口径：
 * - 选项全部来自后端（任务注册表 / 已同步模型 / 可选数据集）。前端硬编码任务名的话，
 *   插件任务在界面上就是隐形的，而 CLI 却能跑——同一个项目两种事实。
 * - 进度必须区分**出处**：`source=db` 的运行是别的进程发起的，这里没有它的取消开关，
 *   显示成"可以取消"是撒谎。
 * - 排队不是失败。等待时说出"谁在占 GPU、预计还要多久"，否则用户只会看到界面卡住，
 *   然后去 kill 一个正在正常排队的进程。
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type { DatasetView, ModelView, ProgressView, QueueView, SubmitView, TaskView } from '../api/types'
import { ErrorState, Panel, StatusBadge } from '../components/primitives'
import { fmtInt, shortId, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'

const LIVE = ['queued', 'running']
const POLL_MS = 1000

export type RunBody = {
  task: string
  model: string
  k?: number
  limit?: number | null
  split?: string
  seed?: number | null
  dataset?: string | null
  unload_others?: boolean
  notes?: string
}

export type Form = {
  task: string
  model: string
  dataset: string
  split: string
  k: string
  limit: string
  seed: string
  notes: string
  unloadOthers: boolean
}

export function isLive(state: string): boolean {
  return LIVE.includes(state)
}

/** 进度比例。total 为 0 时返回 null——画成 0% 会被读成"一条都没跑完"，
 *  而真相是"还不知道总共多少条"（排队中就是这样） */
export function progressRatio(p: ProgressView | null): number | null {
  if (!p || p.total <= 0) return null
  return Math.min(1, Math.max(0, p.done / p.total))
}

/** 空字符串一律当"没填"（后端会用任务默认）。
 *  0 **不是**没填：k=0 要交给后端拒绝并说明范围，悄悄补成 1 就变成"用户按了个和填的不一样的按钮" */
export function toNumber(value: string): number | null {
  const trimmed = (value ?? '').trim()
  if (trimmed === '') return null
  const parsed = Number(trimmed)
  return Number.isFinite(parsed) ? parsed : null
}

export function buildBody(form: Form): RunBody {
  const k = toNumber(form.k)
  return {
    task: form.task,
    model: form.model,
    k: k === null ? 1 : k,
    limit: toNumber(form.limit),
    seed: toNumber(form.seed),
    split: form.split === '' ? 'default' : form.split,
    // '' 表示"用任务默认的那份"，必须发 null 而不是空串：空串会被当成一个未知的数据集名
    dataset: form.dataset === '' ? null : form.dataset,
    unload_others: form.unloadOthers,
    notes: form.notes,
  }
}

/** 这一句要不要显示。等待与排队两种"还没在跑"要分开说：
 *  前者有人在占 GPU，后者只是排在我前面的任务还没轮到 */
export function waitNote(p: ProgressView | null): string {
  if (!p) return ''
  if (p.state === 'queued') {
    return p.position > 0 ? `排队中，前面还有 ${p.position} 个` : '排队中'
  }
  if (p.state === 'running' && p.holder) {
    const eta = p.eta_s === null ? '进度未知' : `预计还需 ${Math.round(p.eta_s)}s`
    return `等 GPU：当前由 ${p.holder} 占用，${eta}（已等 ${Math.round(p.waited_s)}s）`
  }
  return ''
}

/** 子集列表跟着**实际要用的那份数据**走。任务默认集的 splits 与导入集的 splits 通常不同，
 *  选了 A 集却把 B 集的子集留在下拉里，跑起来只会得到一条"没有这个子集"的 error。 */
export function splitChoices(source: { splits: Record<string, number> } | null): string[] {
  return Object.keys(source?.splits ?? {}).filter((name) => name !== 'default')
}

/** 换了数据集之后，原来选中的子集还在不在；不在就回到 default（空串） */
export function keepSplit(
  split: string,
  source: { splits: Record<string, number> } | null,
): string {
  if (split === '') return ''
  return source && split in source.splits ? split : ''
}

export function EvalLaunchPanel({
  onRefresh,
  onFinished,
}: {
  onRefresh: () => void
  onFinished: (runId: string) => void
}) {
  const tasks = useApi<TaskView[]>(() => api.evalTasks(), { intervalMs: 60_000 })
  const models = useApi<ModelView[]>(() => api.models(), { intervalMs: 60_000 })
  const datasets = useApi<DatasetView[]>(() => api.evalDatasets(), {
    intervalMs: 60_000,
  })
  const queue = useApi<QueueView>(() => api.evalQueue(), { intervalMs: 5_000 })

  const [form, setForm] = useState<Form>({
    task: '', model: '', dataset: '', split: '', k: '1', limit: '', seed: '', notes: '',
    unloadOthers: false,
  })
  const [submitted, setSubmitted] = useState<SubmitView | null>(null)
  const [progress, setProgress] = useState<ProgressView | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [note, setNote] = useState('')
  const timer = useRef<number | null>(null)
  const runId = submitted?.run_id ?? null

  const taskList = tasks.data ?? []
  const pickedTask = taskList.find((t) => t.id === form.task) ?? null

  // 任务和模型各只有一份来源，缺省时取第一个可用项：
  // 让界面一打开就能按提交，比让人先猜"任务名怎么填"有用
  useEffect(() => {
    const firstTask = taskList.length ? taskList[0].id : ''
    const firstModel = models.data?.[0]?.name ?? ''
    if (!form.task && firstTask) setForm((f) => ({ ...f, task: firstTask, split: '' }))
    if (!form.model && firstModel) setForm((f) => ({ ...f, model: firstModel }))
  }, [taskList, models.data, form.task, form.model])

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearTimeout(timer.current)
      timer.current = null
    }
  }, [])

  // 轮询进度：终态就停，不做无意义的每秒请求
  useEffect(() => {
    if (!runId) return
    let cancelled = false
    const tick = async () => {
      try {
        const next = await api.runProgress(runId)
        if (cancelled) return
        setProgress(next)
        setError(null)
        if (isLive(next.state)) {
          timer.current = window.setTimeout(tick, POLL_MS)
          return
        }
        stopPolling()
        // 只有真的产生过 run 记录（跑过至少一条）才把它选进表格：
        // 排队中就被取消的任务在库里没有任何痕迹，选它只会得到一个 404
        onRefresh()
        if (next.total > 0) onFinished(next.run_id)
      } catch (err) {
        if (cancelled) return
        setError(err)
        stopPolling()
      }
    }
    timer.current = window.setTimeout(tick, 0)
    return () => {
      cancelled = true
      stopPolling()
    }
  }, [runId, onRefresh, onFinished, stopPolling])

  const start = async () => {
    setError(null)
    setNote('')
    setSubmitted(null)
    setProgress(null)
    try {
      const view = await api.startRun(buildBody(form))
      setSubmitted(view)
      onRefresh()
    } catch (err) {
      setError(err)
    }
  }

  const cancel = async () => {
    if (!runId) return
    try {
      const resp = await api.cancelRun(runId)
      setNote(resp.message)
    } catch (err) {
      setError(err)
    }
  }

  const ratio = progressRatio(progress)
  const state = progress?.state ?? submitted?.state ?? ''
  const live = isLive(state)
  const datasetRows = (datasets.data ?? []).filter((d) => d.selectable)
  const chosenDataset = form.dataset
    ? datasetRows.find((item) => item.id === form.dataset) ?? null
    : null
  const splitSource = chosenDataset ?? pickedTask
  const splits = splitChoices(splitSource)

  return (
    <Panel
      title="发起评测"
      note="与 onyx eval run 同一条路径：同一个 runner、同一把 GPU 锁"
      actions={
        queue.data ? (
          <span className="small muted" title="本进程排队中的任务数 / 上限">
            队列 {queue.data.jobs.filter((j) => isLive(j.state)).length}/{queue.data.max_pending}
          </span>
        ) : null
      }
      flush
    >
      <div className="row-wrap gap-3">
        <label className="field">
          <span className="field-label">任务</span>
          <select
            className="select"
            value={form.task}
            onChange={(e) => setForm({ ...form, task: e.target.value, split: '' })}
          >
            {taskList.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name || t.id}
                {t.error ? '（构造失败）' : ''}
              </option>
            ))}
            {!taskList.length ? <option value="">（没有可用任务）</option> : null}
          </select>
        </label>
        <label className="field">
          <span className="field-label">模型</span>
          <select className="select" value={form.model} onChange={(e) => setForm({ ...form, model: e.target.value })}>
            {(models.data ?? []).map((m) => (
              <option key={m.id} value={m.name}>
                {m.name}
              </option>
            ))}
            {!models.data?.length ? <option value="">（先同步模型清单）</option> : null}
          </select>
        </label>
        <label className="field">
          <span className="field-label">数据集</span>
          <select
            className="select" value={form.dataset}
            onChange={(e) => {
              const id = e.target.value
              const next = id ? datasetRows.find((item) => item.id === id) ?? null : pickedTask
              setForm({ ...form, dataset: id, split: keepSplit(form.split, next) })
            }}
          >
            <option value="">
              任务默认{pickedTask ? `（${pickedTask.default_dataset}，${fmtInt(pickedTask.n_cases)} 条）` : ''}
            </option>
            {datasetRows.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}（{fmtInt(d.n_cases)} 条）
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field-label">子集</span>
          <select className="select" value={form.split} onChange={(e) => setForm({ ...form, split: e.target.value })}>
            <option value="">default</option>
            {splits.map((name) => (
              <option key={name} value={name}>
                {name}（{fmtInt(splitSource?.splits[name] ?? null)}）
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field-label">k</span>
          <input className="input" style={{ width: 64 }} value={form.k}
            onChange={(e) => setForm({ ...form, k: e.target.value })} inputMode="numeric" />
        </label>
        <label className="field">
          <span className="field-label">limit</span>
          <input className="input" style={{ width: 72 }} value={form.limit} placeholder="全部"
            onChange={(e) => setForm({ ...form, limit: e.target.value })} inputMode="numeric" />
        </label>
        <label className="field">
          <span className="field-label">seed</span>
          <input className="input" style={{ width: 72 }} value={form.seed} placeholder="不传"
            onChange={(e) => setForm({ ...form, seed: e.target.value })} inputMode="numeric" />
        </label>
        <label className="field">
          <span className="field-label">备注</span>
          <input className="input" style={{ width: 160 }} value={form.notes} placeholder="这次为什么跑"
            onChange={(e) => setForm({ ...form, notes: e.target.value })} />
        </label>
        <label className="field">
          <span className="field-label">其它</span>
          <label className="small" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={form.unloadOthers}
              onChange={(e) => setForm({ ...form, unloadOthers: e.target.checked })} />
            先卸掉其它模型
          </label>
        </label>
        <div className="field">
          <span className="field-label">&nbsp;</span>
          <button
            className="btn btn-primary"
            onClick={start}
            disabled={!form.task || !form.model || live}
            title={live ? '已经在跑一条了：本地一块 GPU，并发只会污染数字' : undefined}
          >
            {live ? '排队/运行中…' : '开始评测'}
          </button>
        </div>
      </div>

      {error ? <ErrorState error={error} /> : null}

      {state ? (
        <div className="row-wrap gap-3" style={{ marginTop: 'var(--space-2)' }}>
          <StatusBadge status={state} />
          <span className="mono small">{shortId(runId ?? '', 12)}</span>
          <span className="small">
            {/* 总数未知时不写 0/0：那读起来像"一条都没跑完"，而真相是"还没开始/还不知道总共多少条" */}
            {progress && progress.total > 0
              ? `${fmtInt(progress.done)}/${fmtInt(progress.total)}`
              : UNKNOWN}
            {progress?.case_id ? ` · ${progress.case_id}` : ''}
            {progress?.verdict ? ` · ${progress.verdict}` : ''}
          </span>
          {progress && progress.n_error > 0 ? (
            <span className="badge badge-warn" title="判定为 error 的样本数">
              ⚠ error {progress.n_error}
            </span>
          ) : null}
          {waitNote(progress) ? <span className="small muted">{waitNote(progress)}</span> : null}
          {note ? <span className="small muted">{note}</span> : null}
          {progress?.cancellable ? (
            <button className="btn btn-danger" onClick={cancel}>取消</button>
          ) : null}
          {progress && progress.source === 'db' && live ? (
            <span className="small muted" title="这条运行是别的进程发起的，本服务没有它的取消开关">
              由其它进程发起，取消要回到那边
            </span>
          ) : null}
        </div>
      ) : null}

      {state ? (
        <div className="meter" role="progressbar" aria-valuenow={ratio === null ? undefined : Math.round(ratio * 100)}
          aria-valuemin={0} aria-valuemax={100}>
          <div className="meter-fill" style={{ width: `${(ratio ?? 0) * 100}%` }} />
        </div>
      ) : null}

      {progress?.error ? (
        <div className="warnbox small">
          <b>服务侧失败：</b>
          {progress.error}
          <div className="muted">已经跑出来的样本仍然在库里，可以按 run 下钻看。</div>
        </div>
      ) : null}
      {progress?.reason ? (
        <div className="note small">
          <b>整轮被跳过：</b>
          {progress.reason}
        </div>
      ) : null}
    </Panel>
  )
}
