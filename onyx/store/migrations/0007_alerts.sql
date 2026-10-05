-- 0007：告警触发历史。
--
-- 通知系统的第一个问题永远是"那天到底通知没通知"。答案必须是一条记录，
-- 而不是"大概是 cooldown 挡了吧"。所以这张表记的是**命中 + 尝试投递**这两件事的合取：
-- 每个 (命中, 渠道) 一行，渠道失败就留 status=failed 与原因。
--
-- cooldown 抑制**不写行**：只写"真的准备发出去"的。否则重复异常会把这张表淹没成心跳日志，
-- 而它存在的意义是回答某一次具体通知的下落。
--
-- 与 retention_run 同一类：它是审计，不是数据，**永不参与清理**。

CREATE TABLE IF NOT EXISTS alert_trigger(
  id               TEXT PRIMARY KEY,
  created_at       TEXT NOT NULL,
  code             TEXT NOT NULL,               -- anomaly 码（obs/anomalies.py 的持久契约）
  severity         TEXT NOT NULL,               -- 当时的规格，随码走
  rule_json        TEXT NOT NULL,               -- 生效的阈值/窗口/codes/cooldown + 出处（flag/env/文件/默认）
  n_in_window      INTEGER NOT NULL,            -- 命中时窗口里有多少条，"连续几次"的证据
  window_s         INTEGER NOT NULL,
  first_anomaly_id TEXT,
  last_anomaly_id  TEXT,
  trace_ids_json   TEXT,                        -- 样本 trace_id（下钻用；不是全量清单）
  channel          TEXT NOT NULL,               -- file / webhook / test
  status           TEXT NOT NULL,               -- sent / failed
  detail           TEXT,                        -- 渠道回执或失败原因（截断）
  is_test          INTEGER NOT NULL DEFAULT 0   -- `alerts test` 造的行走 1，不混进真实历史
);

CREATE INDEX IF NOT EXISTS idx_alert_trigger_created ON alert_trigger(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_trigger_code ON alert_trigger(code, created_at DESC);
