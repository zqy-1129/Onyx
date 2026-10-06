/** 常驻的布尔偏好（侧栏开合、引擎状态块开合）：初值读 localStorage，变化即写回。
 *  和 useTheme 同一个套路，只是值是布尔，用 '1'/'0' 存。 */
import { useCallback, useEffect, useState } from 'react'

export function usePersistentFlag(key: string, defaultValue: boolean): [boolean, () => void] {
  const [value, setValue] = useState<boolean>(() => {
    const stored = localStorage.getItem(key)
    return stored === null ? defaultValue : stored === '1'
  })
  useEffect(() => {
    localStorage.setItem(key, value ? '1' : '0')
  }, [key, value])
  const toggle = useCallback(() => setValue((v) => !v), [])
  return [value, toggle]
}
