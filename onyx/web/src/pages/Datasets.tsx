/** 数据集页（S24）：列出已登记的数据集与来历，并把 JSONL 导进来。
 *
 * 三条口径：
 * - **提交的是文本不是路径**。看板可以带 token 共享出去（S21），请求体里的路径
 *   就是"服务器任意读文件"的入口；本地文件由浏览器读成文本再发。
 * - **来历不是装饰**。`revision` 空着就等于放弃"这两次分数是不是同一份考卷"的判据，
 *   所以没填时后端按内容 hash 补一个，界面把它显示回来让人核对。
 * - **覆盖必须显式**。历史 grade 指向这个 id，悄悄换内容会让老分数指向另一份考卷，
 *   所以 409 之后要人自己勾选确认，而不是前端替人勾上。
 */
import { useState } from 'react'
import { api } from '../api/client'
import type { DatasetImportBody, DatasetView, ImportView } from '../api/types'
import { DataTable, type Column } from '../components/DataTable'
import { EmptyState, ErrorState, Panel, Skeleton } from '../components/primitives'
import { fmtClock, fmtInt, UNKNOWN } from '../format'
import { useApi } from '../hooks/useApi'
import { navigate } from '../router'

export type ImportForm = {
  name: string
  id: string
  upstream: string
  revision: string
  license: string
  notes: string
  jsonl: string
  allowReplace: boolean
}

export const EMPTY_FORM: ImportForm = {
  name: '', id: '', upstream: '', revision: '', license: '', notes: '', jsonl: '',
  allowReplace: false,
}

/** 表单 → 请求体。空 id 发 null 让后端按 name 推导；空串会被当成一个真的 id 去登记 */
export function buildImportBody(form: ImportForm): DatasetImportBody {
  const trim = (value: string) => (value ?? '').trim()
  return {
    jsonl: form.jsonl,
    name: trim(form.name) || 'uploaded',
    id: trim(form.id) || null,
    upstream: trim(form.upstream),
    revision: trim(form.revision),
    license: trim(form.license),
    notes: trim(form.notes),
    // 确认位只在真的撞过 409 并且人自己勾了之后才是 true
    allow_replace: form.allowReplace,
  }
}

/** 文件名去掉扩展名当默认 name：它决定未来自派生 case id 的前缀 */
export function defaultName(filename: string): string {
  return filename.replace(/\.(jsonl|ndjson|json|txt)$/i, '') || 'uploaded'
}

export function isConflict(err: unknown): boolean {
  return (err as { status?: number } | null)?.status === 409
}

const columns: Array<Column<DatasetView>> = [
  { key: 'id', header: '数据集', mono: true, render: (d) => d.id, sortValue: (d) => d.id },
  {
    key: 'n', header: '条数', align: 'right', mono: true,
    render: (d) => (d.n_cases === null || d.n_cases === 0
      ? <span className="badge badge-warn" title="登记过但没有样本：导入被中断或库被动过">{fmtInt(d.n_cases)}</span>
      : fmtInt(d.n_cases)),
    sortValue: (d) => d.n_cases ?? 0,
  },
  {
    key: 'splits', header: '子集',
    render: (d) => (
      <span className="cell-dim small" title={Object.entries(d.splits).map(([k, v]) => `${k}:${v}`).join('  ')}>
        {Object.keys(d.splits).filter((k) => k !== 'default').slice(0, 4).join(' · ') || UNKNOWN}
      </span>
    ),
  },
  { key: 'upstream', header: '来源', render: (d) => <span className="small">{d.upstream || UNKNOWN}</span> },
  {
    key: 'revision', header: 'revision', mono: true,
    render: (d) => (d.revision
      ? <span className="small mono">{d.revision}</span>
      : <span className="badge badge-warn" title="没有 revision 就无法回答「这两次跑的是不是同一份数据」">空</span>),
  },
  { key: 'license', header: '许可', render: (d) => <span className="small">{d.license || UNKNOWN}</span> },
  {
    key: 'run', header: '可跑',
    render: (d) => (d.selectable
      ? <span className="badge badge-ok">✓ 可选</span>
      : <span className="badge badge-unknown" title="界面只能选内置数据集或已导入且有样本的数据集">✗ 不可选</span>),
    sortValue: (d) => (d.selectable ? 1 : 0),
  },
  { key: 'at', header: '导入', mono: true, render: (d) => fmtClock(d.imported_at), sortValue: (d) => d.imported_at },
]

/** 子页入口：导入成功就去运行页，那里现在能选到刚导入的数据集 */
export function DatasetsPage() {
  return <DatasetsPanel onImported={() => navigate('/eval')} />
}

export function DatasetsPanel({ onImported }: { onImported: () => void }) {
  const datasets = useApi<DatasetView[]>(() => api.evalDatasets(), { intervalMs: 60_000 })
  const [form, setForm] = useState<ImportForm>(EMPTY_FORM)
  const [result, setResult] = useState<ImportView | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const rows = datasets.data ?? []

  const set = <K extends keyof ImportForm>(key: K, value: ImportForm[K]) =>
    setForm((current) => ({ ...current, [key]: value }))

  const pickFile = async (file?: File) => {
    if (!file) return
    const text = await file.text()
    setForm((current) => ({ ...current, jsonl: text, name: current.name || defaultName(file.name) }))
  }

  const submit = async () => {
    setBusy(true)
    setError(null)
    setResult(null)
    try {
      const view = await api.importDataset(buildImportBody(form))
      setResult(view)
      setForm({ ...EMPTY_FORM, upstream: form.upstream, license: form.license })
      datasets.refresh()
      onImported()
    } catch (err) {
      setError(err)
      if (isConflict(err)) set('allowReplace', false)  // 确认位由人勾，前端不替人勾
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <Panel
        title="数据集"
        note="来历（upstream / revision / license）参与「两次分数能不能比」的判定"
        actions={<button className="btn" onClick={datasets.refresh}>刷新</button>}
        flush
      >
        {datasets.error ? <ErrorState error={datasets.error} /> : null}
        {datasets.loading && !datasets.data ? <Skeleton rows={4} /> : null}
        {datasets.data && rows.length === 0 ? (
          <EmptyState title="库里还没有数据集" hint="用下面的表单导入 JSONL，或 onyx eval import <file> --id xxx" />
        ) : null}
        {rows.length > 0 ? (
          <DataTable columns={columns} rows={rows} rowKey={(d) => d.id} maxHeight="32vh" />
        ) : null}
      </Panel>

      <Panel
        title="导入数据集"
        note="发的是 JSONL 文本，不是服务器上的路径"
      >
        <div className="row-wrap gap-3">
          <label className="field">
            <span className="field-label">本地文件</span>
            <input
              className="input" type="file" accept=".jsonl,.ndjson,.json,.txt"
              onChange={(e) => pickFile(e.target.files?.[0])}
            />
          </label>
          <label className="field">
            <span className="field-label">name</span>
            <input className="input" style={{ width: 150 }} value={form.name} placeholder="决定 case id 前缀"
              onChange={(e) => set('name', e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">数据集 id</span>
            <input className="input" style={{ width: 150 }} value={form.id} placeholder="留空＝name-v1"
              onChange={(e) => set('id', e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">upstream</span>
            <input className="input" style={{ width: 170 }} value={form.upstream} placeholder="来源仓库/文件"
              onChange={(e) => set('upstream', e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">revision</span>
            <input className="input" style={{ width: 130 }} value={form.revision} placeholder="留空＝内容 hash"
              onChange={(e) => set('revision', e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">license</span>
            <input className="input" style={{ width: 110 }} value={form.license} placeholder="转载必填"
              onChange={(e) => set('license', e.target.value)} />
          </label>
          <label className="field">
            <span className="field-label">备注</span>
            <input className="input" style={{ width: 170 }} value={form.notes} placeholder="这批样本怎么来的"
              onChange={(e) => set('notes', e.target.value)} />
          </label>
        </div>
        <label className="field" style={{ marginTop: 'var(--space-2)' }}>
          <span className="field-label">JSONL（每行一个样本，必须含 input 对象）</span>
          <textarea className="input" rows={7} value={form.jsonl}
            placeholder={'{"input": {"instruction": "帮我转 500"}, "expect": {"label": "转账"}, "tags": ["hard"]}'}
            onChange={(e) => set('jsonl', e.target.value)} />
        </label>
        <div className="row-wrap gap-3" style={{ marginTop: 'var(--space-2)' }}>
          <button className="btn btn-primary" onClick={submit}
            disabled={busy || !form.jsonl.trim()}>
            {busy ? '导入中…' : '导入'}
          </button>
          <label className="small" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={form.allowReplace}
              onChange={(e) => set('allowReplace', e.target.checked)} />
            允许覆盖同名 id
          </label>
          <span className="cell-dim small">
            覆盖会改变历史分数指向的考卷，所以要显式勾选
          </span>
        </div>

        {error ? <ErrorState error={error} /> : null}
        {isConflict(error) ? (
          <div className="warnbox small">
            这个 id 已经登记过。确认要用新内容替换它，就勾上左边的「允许覆盖同名 id」再点导入——
            替换之后指向它的历史分数与之后的分数<b>不可比</b>，换个新 id 才是通常正确的做法。
          </div>
        ) : null}
        {result ? (
          <div className="note small" style={{ marginTop: 'var(--space-2)' }}>
            已导入 <b>{result.id}</b>：{fmtInt(result.n_cases)} 条 ·
            revision {result.revision || UNKNOWN} · 来源 {result.upstream || UNKNOWN}
            {result.replaced ? ' · （覆盖了旧内容）' : ''}
            <div className="row-wrap" style={{ marginTop: 4 }}>
              {Object.entries(result.splits).map(([name, count]) => (
                <span className="badge badge-neutral" key={name}>{name} {fmtInt(count)}</span>
              ))}
            </div>
            {result.warnings.map((text) => (
              <div className="badge badge-warn" key={text} style={{ marginTop: 4 }}>⚠ {text}</div>
            ))}
          </div>
        ) : null}
      </Panel>
    </>
  )
}
