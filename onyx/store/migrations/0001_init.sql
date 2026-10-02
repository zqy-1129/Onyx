-- 0001: 观测核心表（provider / model / trace / usage / usage_alt / token_part / tool_call / anomaly）
-- 约定：稳定且需聚合的字段 → 独立列；易变字段 → *_json。列只增不减。
-- 大 payload 不入库，入库存 blob 指针（sha256:...）。

CREATE TABLE IF NOT EXISTS provider(
  id            TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,
  base_url      TEXT NOT NULL,
  api_style     TEXT NOT NULL,
  enabled       INTEGER NOT NULL DEFAULT 1,
  caps_json     TEXT,
  version       TEXT,
  config_json   TEXT,
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model(
  id               TEXT PRIMARY KEY,
  provider_id      TEXT NOT NULL REFERENCES provider(id),
  name             TEXT NOT NULL,
  remote_model     TEXT,
  remote_host      TEXT,
  digest           TEXT,
  bytes            INTEGER,
  modified_at      TEXT,
  family           TEXT,
  families_json    TEXT,
  parameter_size   TEXT,
  quantization     TEXT,
  format           TEXT,
  parent_model     TEXT,
  ctx_train        INTEGER,
  capabilities_json TEXT,
  template         TEXT,
  model_info_json  TEXT,
  tool_format      TEXT,
  tokenizer_source TEXT,
  tokenizer_ref    TEXT,
  usage_ratio      REAL,
  usage_ratio_n    INTEGER,
  probe_json       TEXT,
  first_seen_at    TEXT NOT NULL,
  last_seen_at     TEXT NOT NULL,
  extra_json       TEXT,
  UNIQUE(provider_id, name)
);
CREATE INDEX IF NOT EXISTS idx_model_provider ON model(provider_id);

CREATE TABLE IF NOT EXISTS trace(
  id                 TEXT PRIMARY KEY,
  parent_id          TEXT REFERENCES trace(id),
  root_id            TEXT,
  kind               TEXT NOT NULL,
  purpose            TEXT NOT NULL,
  eval_run_id        TEXT,
  case_id            TEXT,
  sample_seq         INTEGER,
  provider_id        TEXT,
  model_id           TEXT,
  model_name         TEXT,
  started_at         TEXT NOT NULL,
  first_token_at     TEXT,
  finished_at        TEXT,
  status             TEXT NOT NULL,
  error              TEXT,
  params_json        TEXT,
  messages_ref       TEXT,
  tools_ref          TEXT,
  rendered_prompt_ref TEXT,
  output_ref         TEXT,
  raw_request_ref    TEXT,
  raw_response_ref   TEXT,
  finish_reason      TEXT,
  engine_latency_json TEXT,
  gpu_json           TEXT,
  keep_alive         TEXT,
  contract_version   INTEGER NOT NULL DEFAULT 1,
  extra_json         TEXT
);
CREATE INDEX IF NOT EXISTS idx_trace_started ON trace(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_trace_id_desc ON trace(id DESC);
CREATE INDEX IF NOT EXISTS idx_trace_purpose ON trace(purpose, eval_run_id, case_id);
CREATE INDEX IF NOT EXISTS idx_trace_model   ON trace(model_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_trace_status  ON trace(status, started_at DESC);

CREATE TABLE IF NOT EXISTS usage(
  trace_id        TEXT PRIMARY KEY REFERENCES trace(id),
  in_tokens       INTEGER,
  out_tokens      INTEGER,
  thinking_tokens INTEGER,
  cached_tokens   INTEGER,
  source          TEXT NOT NULL,
  confidence      TEXT NOT NULL,
  ttft_ms         REAL,
  prefill_tps     REAL,
  decode_tps      REAL,
  wall_ms         REAL,
  bytes_out       INTEGER,
  drift_pct       REAL,
  extra_json      TEXT
);

-- 每个来源一行：对账与异常检测的事实表
CREATE TABLE IF NOT EXISTS usage_alt(
  trace_id        TEXT NOT NULL REFERENCES trace(id),
  source          TEXT NOT NULL,
  in_tokens       INTEGER,
  out_tokens      INTEGER,
  thinking_tokens INTEGER,
  cached_tokens   INTEGER,
  ok              INTEGER NOT NULL DEFAULT 1,
  confidence      TEXT,
  note            TEXT,
  PRIMARY KEY(trace_id, source)
);

CREATE TABLE IF NOT EXISTS token_part(
  trace_id TEXT NOT NULL REFERENCES trace(id),
  part     TEXT NOT NULL,
  ord      INTEGER NOT NULL DEFAULT 0,
  tokens   INTEGER NOT NULL,
  bytes    INTEGER,
  PRIMARY KEY(trace_id, part, ord)
);

CREATE TABLE IF NOT EXISTS tool_call(
  id             TEXT PRIMARY KEY,
  trace_id       TEXT NOT NULL REFERENCES trace(id),
  step           INTEGER NOT NULL,
  call_id        TEXT,
  name           TEXT,
  args_json      TEXT,
  args_raw       TEXT,
  parse_status   TEXT NOT NULL,
  parse_source   TEXT,
  result_status  TEXT,
  result_ref     TEXT,
  result_bytes   INTEGER,
  started_at     TEXT,
  latency_ms     REAL,
  tool_id        TEXT,
  tool_def_hash  TEXT,
  executed_by    TEXT,
  extra_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_toolcall_trace ON tool_call(trace_id, step);
CREATE INDEX IF NOT EXISTS idx_toolcall_name  ON tool_call(name, started_at);

CREATE TABLE IF NOT EXISTS anomaly(
  id         TEXT PRIMARY KEY,
  trace_id   TEXT,
  code       TEXT NOT NULL,
  severity   TEXT NOT NULL,
  detail_json TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_anomaly_code ON anomaly(code, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_trace ON anomaly(trace_id);
