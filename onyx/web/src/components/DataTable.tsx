/** 密集数据表：28px 行高、1px 分隔线、数字右对齐、列头可排序。
 *  刻意不做斑马纹——高密度下分隔线比交替底色更易扫读。 */
import { useMemo, useState, type ReactNode } from 'react'

export interface Column<T> {
  key: string
  header: ReactNode
  render: (row: T) => ReactNode
  align?: 'left' | 'right'
  mono?: boolean
  width?: number
  title?: string
  /** 提供才可排序；没有稳定排序键的列不许假装能排 */
  sortValue?: (row: T) => number | string | null
}

export function DataTable<T>({
  columns,
  rows,
  rowKey,
  onRowClick,
  selectedKey,
  empty,
  maxHeight,
}: {
  columns: Array<Column<T>>
  rows: T[]
  rowKey: (row: T) => string
  onRowClick?: (row: T) => void
  selectedKey?: string | null
  empty?: ReactNode
  maxHeight?: number | string
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null)

  const sorted = useMemo(() => {
    if (!sort) return rows
    const column = columns.find((c) => c.key === sort.key)
    if (!column?.sortValue) return rows
    const pick = column.sortValue
    return [...rows].sort((a, b) => {
      const left = pick(a)
      const right = pick(b)
      // null（未知）永远排最后：把「没测出来」混进数值排序会得出错误的极值
      if (left === null && right === null) return 0
      if (left === null) return 1
      if (right === null) return -1
      if (typeof left === 'number' && typeof right === 'number') return (left - right) * sort.dir
      return String(left).localeCompare(String(right)) * sort.dir
    })
  }, [rows, sort, columns])

  if (!rows.length && empty) return <>{empty}</>

  return (
    <div className="table-wrap" style={maxHeight ? { maxHeight } : undefined}>
      <table className="data">
        <thead>
          <tr>
            {columns.map((column) => {
              const sortable = Boolean(column.sortValue)
              const active = sort?.key === column.key
              return (
                <th
                  key={column.key}
                  className={`${column.align === 'right' ? 'num ' : ''}${sortable ? '' : 'no-sort'}`}
                  style={column.width ? { width: column.width } : undefined}
                  title={sortable ? `点击按${column.key}排序` : column.title}
                  onClick={
                    sortable
                      ? () =>
                          setSort((prev) =>
                            prev?.key === column.key
                              ? { key: column.key, dir: prev.dir === 1 ? -1 : 1 }
                              : { key: column.key, dir: 1 },
                          )
                      : undefined
                  }
                >
                  {column.header}
                  {active ? (sort?.dir === 1 ? ' ▲' : ' ▼') : ''}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => {
            const key = rowKey(row)
            return (
              <tr
                key={key}
                className={`${onRowClick ? 'clickable' : ''}${selectedKey === key ? ' selected' : ''}`}
                onClick={onRowClick ? () => onRowClick(row) : undefined}
              >
                {columns.map((column) => (
                  <td
                    key={column.key}
                    className={`${column.align === 'right' ? 'num ' : ''}${column.mono ? 'mono' : ''}`}
                  >
                    {column.render(row)}
                  </td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
