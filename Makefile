# Onyx 常用命令。Windows 无 make 时，直接用等价的 `uv run ...`（见每行注释）。
UV := uv run --extra dev --extra runtime

.PHONY: sync test test-live probe lint format typecheck dev doctor db-init clean

sync:            ## 安装依赖（含 dev/runtime extras）
	$(UV) python -c "print('deps ok')"

test:            ## 单元 + 契约测试（无网络、无模型）	→ uv run pytest -q
	$(UV) pytest -q

test-live:       ## 需要 Ollama 在跑			→ uv run pytest -m live -q
	$(UV) pytest -m live -q

probe:           ## 语义实测（写 docs/PROBES.md）	→ uv run pytest -m probe -q
	$(UV) pytest -m probe -q

lint:            ## 代码风格 + 架构边界契约		→ uv run ruff check . && uv run lint-imports
	$(UV) ruff check .
	$(UV) lint-imports

format:          ## 自动修复可修项			→ uv run ruff check --fix .
	$(UV) ruff check --fix .

doctor:          ## 体检				→ uv run onyx doctor
	$(UV) onyx doctor

db-init:         ## 初始化/迁移数据库			→ uv run onyx db init
	$(UV) onyx db init

clean:
	$(UV) python -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache','.ruff_cache','.tmp')]"
