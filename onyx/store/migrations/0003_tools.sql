-- 0003: 工具子系统（L4）
-- 三层分离的前提是"定义"可版本化：schema 变了就是新版本，
-- 旧 trace 仍能通过 tool_def_hash 定位到当时的定义（否则回归对比没有意义）。

CREATE TABLE IF NOT EXISTS tool_def(
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL UNIQUE,
  version       TEXT NOT NULL,
  kind          TEXT NOT NULL,              -- python_fn|http|mcp|ollama_builtin|fixture
  schema_json   TEXT NOT NULL,              -- OpenAI function 形状
  impl_ref      TEXT,
  hash          TEXT NOT NULL,              -- schema+描述 的内容 hash，变更即新版本
  tokens        INTEGER,                    -- 注入上下文的成本（按目标模型的计数档位）
  bytes         INTEGER,
  tags          TEXT,
  owner         TEXT,
  enabled       INTEGER NOT NULL DEFAULT 1,
  side_effect   TEXT NOT NULL DEFAULT 'read', -- read|write|network|exec → sandbox 决策
  timeout_ms    INTEGER,
  doc           TEXT,
  examples_json TEXT,
  extra_json    TEXT,
  created_at    TEXT NOT NULL,
  updated_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_def_kind ON tool_def(kind, enabled);

CREATE TABLE IF NOT EXISTS tool_test(
  id          TEXT PRIMARY KEY,
  tool_id     TEXT NOT NULL REFERENCES tool_def(id),
  name        TEXT NOT NULL,
  args_json   TEXT NOT NULL,
  expect_json TEXT,
  checks_json TEXT,                          -- type_of|path_exists|regex|subset_of|raises
  live        INTEGER NOT NULL DEFAULT 0,    -- 0=用 fixture，1=真跑（默认不碰真实副作用）
  created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tool_test_tool ON tool_test(tool_id);

CREATE TABLE IF NOT EXISTS tool_run(
  id            TEXT PRIMARY KEY,
  tool_id       TEXT,
  tool_def_hash TEXT,
  test_id       TEXT,
  trace_id      TEXT,                        -- 关联到产生这次调用的模型请求
  started_at    TEXT NOT NULL,
  latency_ms    REAL,
  status        TEXT NOT NULL,               -- ok|error|timeout|rejected|mocked|skipped
  output_ref    TEXT,
  error         TEXT,
  deterministic INTEGER,
  idempotent    INTEGER,
  extra_json    TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_run_tool ON tool_run(tool_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_tool_run_trace ON tool_run(trace_id);
