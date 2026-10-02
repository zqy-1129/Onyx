-- 0004: 评测子系统（L5）
-- 核心约束：grade.trace_id 指向一条**真实** trace。评测绝不建立第二条调用路径，
-- 所以每个分数都能点进去看 token / 延迟 / 工具记录（DESIGN §9.1、§15）。
-- grade.trace_id 刻意不设 REFERENCES：trace 可能因保留策略被清理，而分数必须留着
-- （清理 trace 不该让评测历史连带消失）。
--
-- NOT NULL 只加在"空值即 bug"的列上：store/codec.py 的 dumps() 会把 {} / [] 存成 NULL，
-- 所以给一个允许为空的容器列加 NOT NULL，等于让合法的空配置写不进去。

CREATE TABLE IF NOT EXISTS dataset(
  id          TEXT PRIMARY KEY,
  upstream    TEXT,                          -- 来源（builtin / bfcl / 自建）
  revision    TEXT,                          -- 上游版本；换版本后的分数不可直接比较
  license     TEXT,
  split_json  TEXT,                          -- {split 名: 条数}
  n_cases     INTEGER,
  loader      TEXT,                          -- 载入器标识，便于复现
  notes       TEXT,
  imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_case(
  id           TEXT PRIMARY KEY,
  dataset_id   TEXT NOT NULL REFERENCES dataset(id),
  ord          INTEGER NOT NULL DEFAULT 0,
  kind         TEXT NOT NULL DEFAULT 'single', -- single|multi_turn|multi_step|parallel|no_call_needed
  input_json   TEXT NOT NULL,               -- 没有输入的样本是坏样本，必须炸出来
  tools_json   TEXT,
  expect_json  TEXT,                         -- latency_bench 这类任务没有期望答案

  fixture_json TEXT,                          -- 工具返回值桩：保证评测可复现（§8.4）
  meta_json    TEXT,
  tags         TEXT
);
CREATE INDEX IF NOT EXISTS idx_case_dataset ON eval_case(dataset_id, ord);
CREATE INDEX IF NOT EXISTS idx_case_kind ON eval_case(kind);

CREATE TABLE IF NOT EXISTS eval_task(
  id               TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  dataset_id       TEXT REFERENCES dataset(id),
  metrics_json     TEXT NOT NULL,             -- 该任务会产出哪些指标（跑之前就知道）
  grader_json      TEXT,                      -- 用的哪些评分器与阈值；无额外配置时为 NULL

  sample_params_json TEXT,                    -- temperature/max_tokens 等；不落库就无法解释分数变化
  k                INTEGER NOT NULL DEFAULT 1,-- pass^k 采样次数（引擎不支持 n → 循环采样）
  budget_json      TEXT,
  sandbox          INTEGER NOT NULL DEFAULT 0,
  extra_json       TEXT
);

CREATE TABLE IF NOT EXISTS eval_run(
  id                  TEXT PRIMARY KEY,
  task_id             TEXT NOT NULL REFERENCES eval_task(id),
  model_id            TEXT NOT NULL,
  provider_id         TEXT,
  started_at          TEXT NOT NULL,
  finished_at         TEXT,
  status              TEXT NOT NULL DEFAULT 'running', -- running|done|cancelled|error
  seed                INTEGER,
  app_version         TEXT,
  git_rev             TEXT,
  params_snapshot_json TEXT,
  config_json         TEXT,
  n_cases             INTEGER NOT NULL DEFAULT 0,
  n_done              INTEGER NOT NULL DEFAULT 0,
  n_error             INTEGER NOT NULL DEFAULT 0,
  n_skipped           INTEGER NOT NULL DEFAULT 0,
  aggregate_json      TEXT,                   -- 汇总指标（含 CI 与 low_confidence 标记）
  cost_json           TEXT,                   -- 评测自身开销：token / 墙钟 / judge 成本
  notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_task ON eval_run(task_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_run_model ON eval_run(model_id, started_at DESC);

CREATE TABLE IF NOT EXISTS grade(
  id             TEXT PRIMARY KEY,
  eval_run_id    TEXT NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
  case_id        TEXT NOT NULL,
  seq            INTEGER NOT NULL DEFAULT 0,  -- 同一 case 的第几次采样
  trace_id       TEXT,                        -- 每个分数都能点进一条真实 trace
  score          REAL NOT NULL,
  passed         INTEGER,                     -- NULL = 未判定（skip / 不可归因）
  verdict        TEXT NOT NULL,               -- 见 eval/task.py: Verdict
  invalid_format INTEGER NOT NULL DEFAULT 0,  -- 与"内容答错"分开计（§9.4）
  out_of_set     INTEGER NOT NULL DEFAULT 0,  -- 幻觉标签/幻觉工具名
  metrics_json   TEXT,
  error          TEXT,
  judge_model_id TEXT,
  judge_usage_json TEXT,                      -- judge 自己也是本地模型，成本必须计入
  extra_json     TEXT,
  graded_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_grade_run ON grade(eval_run_id, case_id, seq);
CREATE UNIQUE INDEX IF NOT EXISTS uq_grade_run_case_seq ON grade(eval_run_id, case_id, seq);
CREATE INDEX IF NOT EXISTS idx_grade_trace ON grade(trace_id);
CREATE INDEX IF NOT EXISTS idx_grade_verdict ON grade(eval_run_id, verdict);
