-- 0006：数据生命周期的留痕表。
--
-- 保留策略会删掉证据，所以"什么时候按什么参数删了多少"必须自己成为一条记录：
-- 事后有人问"三周前那次原始 body 怎么没了"，答案不能是"大概跑了 rotate 吧"。
-- 注意这张表本身**永不参与清理**（它是审计日志，不是数据）。

CREATE TABLE IF NOT EXISTS retention_run(
  id             TEXT PRIMARY KEY,
  started_at     TEXT NOT NULL,
  finished_at    TEXT,
  dry_run        INTEGER NOT NULL,           -- 1 = 只算不删。默认所有调用都是 dry-run
  trace_after_d  INTEGER,                    -- 当时用的参数，复盘时能判断是不是有人写错了窗口
  raw_after_d    INTEGER,
  purge_traces   INTEGER NOT NULL DEFAULT 0,
  refs_cleared   INTEGER NOT NULL DEFAULT 0, -- 摘掉的原始证据引用数（行保留）
  traces_deleted INTEGER NOT NULL DEFAULT 0, -- 只有 --purge-traces 才可能非 0
  blobs_deleted  INTEGER NOT NULL DEFAULT 0,
  orphans_found  INTEGER NOT NULL DEFAULT 0, -- 没有行引用但躺在盘上的 blob（崩溃/半截写入的产物）
  bytes_before   INTEGER NOT NULL DEFAULT 0,
  bytes_after    INTEGER NOT NULL DEFAULT 0,
  detail_json    TEXT                        -- 被删对象的清单摘要（截断），足够指回具体一次调用
);

CREATE INDEX IF NOT EXISTS idx_retention_started ON retention_run(started_at DESC);
