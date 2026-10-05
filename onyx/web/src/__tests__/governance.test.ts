/** 模型治理面板的措辞逻辑（S26）。
 *
 * 拉取 / 卸载 / 删除是这台机器上唯一会丢东西的三个动作，界面话术属于产品口径：
 * 501 要常驻说"这个通道没有控制面"，而不是让人反复点同一个按钮；
 * 删除必须说清分数还在——否则人会以为一次测量被抹掉了。
 */
import { describe, expect, it } from 'vitest'
import {
  confirmText, governanceSummary, isGovernanceUnsupported, type GovernanceAction,
} from '../pages/ModelGovernance'
import type { AdminResultView } from '../api/types'

function resp(over: Partial<AdminResultView> = {}): AdminResultView {
  return { ok: true, action: 'pull', ...over }
}

describe('isGovernanceUnsupported', () => {
  it('只有 501 算「通道不暴露控制面」', () => {
    // 把 502/超时也说成不支持，会把一次网络抖动变成"这台机器不能治理模型"的假结论
    expect(isGovernanceUnsupported({ status: 501 })).toBe(true)
    expect(isGovernanceUnsupported({ status: 502 })).toBe(false)
    expect(isGovernanceUnsupported(new Error('boom'))).toBe(false)
    expect(isGovernanceUnsupported(null)).toBe(false)
    expect(isGovernanceUnsupported(undefined)).toBe(false)
  })
})

describe('governanceSummary', () => {
  it('拉取说清落了哪版权重、清单同步了几个', () => {
    const line = governanceSummary('pull', resp({ digest: 'sha256:ab', models_synced: 7 }), 'qwen3:8b')
    expect(line).toContain('qwen3:8b')
    expect(line).toContain('sha256:ab')
    expect(line).toContain('7')
  })

  it('拉取没有 digest 时不印一个空 digest', () => {
    expect(governanceSummary('pull', resp(), 'qwen3:8b')).toContain('—')
  })

  it('删除必须说分数还在', () => {
    // 后端给了话就用后端的（口径只有一份），没给也要自己补上这句
    expect(governanceSummary('rm', resp({ note: '权重已释放；历史 trace 与分数保留' }), 'qwen3:8b'))
      .toContain('分数保留')
    expect(governanceSummary('rm', resp(), 'qwen3:8b')).toContain('分数保留')
  })

  it('卸载要说下一次请求是冷启动', () => {
    expect(governanceSummary('unload', resp(), 'qwen3:8b')).toContain('冷启动')
  })
})

describe('confirmText', () => {
  const actions: GovernanceAction[] = ['pull', 'unload', 'rm']

  it('每个动作都点名目标，不写一句通用「确认操作」', () => {
    for (const action of actions) expect(confirmText(action, 'qwen3:8b')).toContain('qwen3:8b')
  })

  it('破坏性程度不同，措辞不同', () => {
    expect(confirmText('rm', 'qwen3:8b')).toContain('不可逆')
    expect(confirmText('rm', 'qwen3:8b')).toContain('分数')
    expect(confirmText('pull', 'qwen3:8b')).toContain('GB')
    expect(confirmText('unload', 'qwen3:8b')).toContain('冷启动')
    expect(confirmText('pull', 'qwen3:8b')).not.toContain('不可逆')
  })
})
