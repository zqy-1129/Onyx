-- 0002: prefill 冷/热分列（PROBES P11）
-- 实测：同一 644-token prompt，冷 384.5ms(1675 t/s) vs 热 83.4ms(7722 t/s)，差 4.65×。
-- 把两者混进同一个吞吐 P50 会得到一个既不代表冷启动也不代表稳态、且看起来完全合理的错误数字，
-- 所以 prefill 模式必须落库，让聚合能分列。
ALTER TABLE usage ADD COLUMN prefill_mode TEXT;
ALTER TABLE usage ADD COLUMN prefill_ms_per_token REAL;
CREATE INDEX IF NOT EXISTS idx_usage_prefill_mode ON usage(prefill_mode);
