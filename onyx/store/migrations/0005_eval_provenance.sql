-- 0005 — 运行记录必须带数据集来历
--
-- 对比与回归（M5）的全部结论都建立在一件事上：**两次跑的是不是同一份数据**。
-- 原先 `eval_run` 只记 task/model/params，数据集只能靠 grade 的 case_id 反推，
-- 于是"换了数据集版本之后的两次分数"看起来完全可比。分数不可比却显示成可比，
-- 比没有分数更糟（DESIGN §9.3、UI_DESIGN R2）。
ALTER TABLE eval_run ADD COLUMN dataset_id TEXT;
ALTER TABLE eval_run ADD COLUMN dataset_revision TEXT;

CREATE INDEX IF NOT EXISTS idx_run_dataset ON eval_run(dataset_id, started_at DESC);

-- 回填：历史 run 的数据集从它的 grade 落到 eval_case 上取多数那个。
-- 取多数而不是任取一条，是因为跨数据集的旧 run（理论上不该存在）在这里会暴露成
-- "只记了一个 id"——但那种情况本来就无从重建，记多数比记 NULL 更接近真相。
UPDATE eval_run
SET dataset_id = (
    SELECT c.dataset_id
    FROM grade g JOIN eval_case c ON c.id = g.case_id
    WHERE g.eval_run_id = eval_run.id AND c.dataset_id IS NOT NULL
    GROUP BY c.dataset_id
    ORDER BY COUNT(*) DESC
    LIMIT 1
)
WHERE eval_run.dataset_id IS NULL;

UPDATE eval_run
SET dataset_revision = (SELECT d.revision FROM dataset d WHERE d.id = eval_run.dataset_id)
WHERE eval_run.dataset_id IS NOT NULL AND eval_run.dataset_revision IS NULL;
