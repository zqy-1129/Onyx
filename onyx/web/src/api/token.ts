/** 看板 token 的浏览器侧落脚点。
 *
 * 后端在非回环绑定上要 token（`onyx serve --read-only` 共享看板）。前端能拿到 token 的
 * 唯一通用渠道是 URL：`EventSource` **设不了请求头**，所以 SSE 只能走 query 参数。
 * 因此这里把 `?token=` 收进 sessionStorage（关标签页即失效，不落 localStorage），
 * 之后 fetch 走 Authorization 头、SSE 继续走 query。
 *
 * 诚实的代价：query 参数会进浏览器历史、Referer 与任何中间层访问日志。
 * 这就是"只在需要共享时才设 token"而不是"默认就开"的原因。
 */

const KEY = 'onyx.token'

function readSession(): string | null {
  try {
    return window.sessionStorage.getItem(KEY)
  } catch {
    // 隐私模式/禁用存储下 sessionStorage 会抛：拿不到 token 就退回"每次从 URL 读"
    return null
  }
}

function writeSession(value: string): void {
  try {
    window.sessionStorage.setItem(KEY, value)
  } catch {
    /* 存不下就算了：本轮会话继续用 URL 上的值 */
  }
}

/** 当前 token：URL 上的 `?token=` 优先（它代表"这次访问想用的身份"）。 */
export function apiToken(): string | null {
  const fromUrl = new URLSearchParams(window.location.search).get('token')
  if (fromUrl) {
    // 不缓存：缓存会造出"改了 URL 上的 token 却不生效"——正是这个项目反复踩的那类失败
    writeSession(fromUrl)
    return fromUrl
  }
  return readSession()
}

/** fetch 用的请求头。没有 token 就什么都不加（回环部署是常态，别塞空 header）。 */
export function authHeaders(base: Record<string, string> = {}): Record<string, string> {
  const token = apiToken()
  return token ? { ...base, authorization: `Bearer ${token}` } : base
}

/** EventSource 用的 URL：token 只能挂在 query 上。 */
export function withToken(path: string): string {
  const token = apiToken()
  if (!token) return path
  return `${path}${path.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`
}
