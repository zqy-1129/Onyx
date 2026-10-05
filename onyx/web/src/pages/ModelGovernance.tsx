/** 模型治理面板（S26）：拉取 / 卸载 / 删除，与 `onyx models pull|rm` 同源。
 *
 * 这三件事引擎侧早就支持（`AdminProvider`），Playground 也一直在用 unload，
 * 但界面上一个入口都没有 —— 于是"加个模型进来"必须开终端，而那条路径绕开了 Onyx：
 * 不刷清单、看板上看不到，人就以为拉取失败了。
 *
 * 三条规矩：
 * - **危险动作要二次确认**，和 API 的 `confirm=1` 是同一件事的两层，不是装饰：
 *   拉取写 GB 级磁盘，删除不可逆，卸载会让下一次请求变冷启动。
 * - **没有控制面的通道要提前说**。兼容层不实现 `AdminProvider`，硬做出来的是假动作；
 *   拿到 501 之后把这句话常驻显示，而不是让人反复点同一个按钮。
 * - **删除只删权重**。历史 trace 与分数一行都不动——那次测量已经发生了。
 *   这句话必须显示出来，否则人会以为连带着分数也没了。
 */
import { useState } from 'react'
import { api } from '../api/client'
import type { AdminResultView, ModelView } from '../api/types'
import { ErrorState, Panel } from '../components/primitives'
import { UNKNOWN } from '../format'

export type GovernanceAction = 'pull' | 'unload' | 'rm'

export function isGovernanceUnsupported(err: unknown): boolean {
  return (err as { status?: number } | null)?.status === 501
}

/** 一次动作之后界面上该留哪句话。三种动作的后果不同，措辞也不能共用一句"成功"。 */
export function governanceSummary(action: GovernanceAction, resp: AdminResultView, name: string): string {
  if (action === 'pull') {
    const digest = resp.digest || UNKNOWN
    return `已拉取 ${name}（digest ${digest}）· 清单已同步 ${resp.models_synced ?? 0} 个模型`
  }
  if (action === 'rm') {
    return resp.note || `已删除 ${name}；历史 trace 与分数保留`
  }
  return `已卸载 ${name}：显存让出来了，下一次请求会是冷启动（TTFT 会明显变高）`
}

/** 破坏性程度决定确认文案的措辞——"不可逆"和"会变慢"不是一回事 */
export function confirmText(action: GovernanceAction, name: string): string {
  if (action === 'rm') return `我确认删除 ${name} 的权重，这一步不可逆（历史分数与 trace 保留）`
  if (action === 'pull') return `我确认拉取 ${name}，这会写入 GB 级权重`
  return `我确认卸载 ${name}（会丢 KV 缓存，下一次请求变冷启动）`
}

export function ModelGovernancePanel({
  models, onChanged, providerId,
}: {
  models: ModelView[]
  onChanged: () => void
  providerId: string
}) {
  const [pullName, setPullName] = useState('')
  const [picked, setPicked] = useState('')
  const [confirmed, setConfirmed] = useState(false)
  const [note, setNote] = useState('')
  const [error, setError] = useState<unknown>(null)
  const [unsupported, setUnsupported] = useState(false)
  const [busy, setBusy] = useState<GovernanceAction | null>(null)
  const target = picked || models[0]?.name || ''

  const act = async (action: GovernanceAction) => {
    const name = action === 'pull' ? pullName.trim() : target
    if (!name || !confirmed) return
    setBusy(action)
    setError(null)
    setNote('')
    try {
      const resp = action === 'pull'
        ? await api.pullModel(name)
        : action === 'rm'
          ? await api.removeModel(name)
          : await api.unload(name)
      setNote(governanceSummary(action, resp, name))
      onChanged()
    } catch (err) {
      setError(err)
      setUnsupported(isGovernanceUnsupported(err))
    } finally {
      setBusy(null)
    }
  }

  return (
    <Panel
      title="模型治理"
      note={`与 onyx models pull / rm 同源 · 通道 ${providerId || UNKNOWN}`}
      flush
    >
      {unsupported ? (
        <div className="banner banner-warn">
          这个通道不暴露控制面，卸载/拉取/删除都做不了。
          <span className="cell-dim small">
            　兼容层没有统一端点，做一个假的会让"显存已经让出来了"这种判断建立在谎话上；
            换 ollama 通道或直接在引擎侧操作
          </span>
        </div>
      ) : null}

      <div className="row-wrap gap-3">
        <label className="field">
          <span className="field-label">拉取新模型</span>
          <input className="input" style={{ width: 220 }} value={pullName}
            placeholder="例如 qwen3:8b" onChange={(e) => setPullName(e.target.value)} />
        </label>
        <label className="field">
          <span className="field-label">目标模型</span>
          <select className="select" value={target} onChange={(e) => setPicked(e.target.value)}>
            {models.map((m) => <option key={m.id} value={m.name}>{m.name}</option>)}
            {!models.length ? <option value="">（清单为空）</option> : null}
          </select>
        </label>
        <div className="field">
          <span className="field-label">动作</span>
          <span className="row">
            <button className="btn" disabled={!pullName.trim() || !confirmed || busy !== null}
              onClick={() => act('pull')}
              title={`${confirmText('pull', pullName.trim() || '…')}；这是一次长请求，大下载请用 onyx models pull`}>
              {busy === 'pull' ? '拉取中…' : '拉取'}
            </button>
            <button className="btn" disabled={!target || !confirmed || busy !== null}
              onClick={() => act('unload')} title={confirmText('unload', target)}>
              {busy === 'unload' ? '卸载中…' : '卸载'}
            </button>
            <button className="btn btn-danger" disabled={!target || !confirmed || busy !== null}
              onClick={() => act('rm')} title={confirmText('rm', target)}>
              {busy === 'rm' ? '删除中…' : '删除权重'}
            </button>
          </span>
        </div>
      </div>

      <label className="small" style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 'var(--space-2)' }}>
        <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
        勾选后按钮才可用 —— 拉取会写 GB 级权重、卸载会丢 KV 缓存、删除不可逆（历史分数与 trace 保留）
      </label>
      <div className="note small">
        删除只释放权重：<b>历史 trace 与分数一行都不动</b>，那次测量已经发生了。
      </div>

      {error && !unsupported ? <ErrorState error={error} /> : null}
      {note ? <div className="note small">{note}</div> : null}
    </Panel>
  )
}
