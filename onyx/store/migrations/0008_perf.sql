-- 0008：性能基线（S36）。
--
-- 两张表的分工：`perf_run` 回答"这次是在什么条件下测的"，`perf_cell` 回答"每一格测出来多少"。
-- 拆开是因为条件属于整次运行：把它抄进每一格，就会出现"改了条件而某几格还是旧条件"这种
-- 看起来自洽其实错位的对比。
--
-- **数字抄进表里，而不是只留 trace 指针**：trace 会被保留策略摘走（`retention`），
-- 而一条基线存在的意义正是"半年后还能拿来比"。`trace_ids_json` 只用于下钻，
-- 它指不指得回来都不影响这张表里的数还成立。
--
-- 与 `retention_run` / `alert_trigger` 同一类：它是结论，不是数据，**永不参与清理**。
--
-- `env_hash` 是可比性的唯一依据，字段清单在 `onyx/perf/bench.py::FINGERPRINT_FIELDS`。
-- 刻意**不含** `app_version` / `git_rev`：换代码正是基线要对比的对象，
-- 把它当"不可比"就等于这条命令永远给不出结论。
-- `comparable=0` 表示连"这是哪台引擎的哪个模型"都没认出来（常见于兼容通道报不出版本），
-- 这时 compare 要拒绝——**认不出引擎就不许声称可比**。

CREATE TABLE IF NOT EXISTS perf_run(
  id               TEXT PRIMARY KEY,
  started_at       TEXT NOT NULL,
  finished_at      TEXT NOT NULL,
  status           TEXT NOT NULL,             -- done / partial / error
  provider_id      TEXT NOT NULL DEFAULT '',
  engine_version   TEXT NOT NULL DEFAULT '',  -- 空串 = 引擎报不出来（不是"版本为空"）
  model            TEXT NOT NULL,
  quantization     TEXT NOT NULL DEFAULT '',
  device           TEXT NOT NULL DEFAULT '',   -- 由 --device 人工标注；空 ⇒ 跨机器对比不成立
  num_ctx          INTEGER,                    -- NULL = 没传给引擎（用它的默认）
  keep_alive       TEXT NOT NULL DEFAULT '',
  stream           INTEGER NOT NULL DEFAULT 1,
  timing_source    TEXT NOT NULL DEFAULT 'unknown',  -- engine_ns / wall_only / unknown
  env_hash         TEXT NOT NULL,
  comparable       INTEGER NOT NULL DEFAULT 0,
  conditions_json  TEXT NOT NULL DEFAULT '{}', -- 全量条件快照（含 app_version/git_rev）
  grid_json        TEXT NOT NULL DEFAULT '{}', -- 网格与预算
  elapsed_s        REAL NOT NULL DEFAULT 0,
  n_requests       INTEGER NOT NULL DEFAULT 0,
  app_version      TEXT NOT NULL DEFAULT '',
  git_rev          TEXT NOT NULL DEFAULT '',
  note             TEXT NOT NULL DEFAULT '',
  error            TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_perf_run_started ON perf_run(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_perf_run_env ON perf_run(env_hash, started_at DESC);

CREATE TABLE IF NOT EXISTS perf_cell(
  run_id          TEXT NOT NULL REFERENCES perf_run(id) ON DELETE CASCADE,
  cell_key        TEXT NOT NULL,
  phase           TEXT NOT NULL,               -- warm / cold
  prompt_chars    INTEGER NOT NULL,            -- 声明长度（汉字数）
  target_tokens   INTEGER NOT NULL,
  concurrency     INTEGER NOT NULL,
  repeat          INTEGER NOT NULL,
  status          TEXT NOT NULL,               -- measured / skipped / error
  reason          TEXT NOT NULL DEFAULT '',    -- 没测到的原因写在人看得见的地方
  n_requests      INTEGER NOT NULL DEFAULT 0,
  n_measured      INTEGER NOT NULL DEFAULT 0,
  metrics_json    TEXT NOT NULL DEFAULT '{}',  -- 每列都带 n；缺的列是 null 不是 0
  trace_ids_json  TEXT NOT NULL DEFAULT '[]',
  PRIMARY KEY(run_id, cell_key)
);

CREATE INDEX IF NOT EXISTS idx_perf_cell_key ON perf_cell(cell_key, run_id);
