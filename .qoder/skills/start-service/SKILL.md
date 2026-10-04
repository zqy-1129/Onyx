---
name: start-service
description: 启动 Onyx 看板服务（Ollama 引擎检查 + doctor 体检 + db init + 后端 onyx serve:8787 + 前端 vite:5173），并逐层探活到代理链打通。当用户要"起服务""启动看板""跑 dev""restart 后端/前端""看看板打不开"时使用。
argument-hint: "[ollama|openai-compat|mock]"
---

# 启动 Onyx 看板服务

## Overview

拉起看板的两个进程（后端 `onyx serve` + 前端 `vite dev`），起前确认引擎与数据库，起后按
「后端 → 前端 → 经 vite 代理的 /api」三层探活，确保看板真的能拿到数据而不只是两个进程活着。

## 事实（不要现场猜）

- **后端端口必须是 8787**：`onyx serve` 的 `--port` 默认是 8000，但本机 8000 是引擎
  （openai-compat/vLLM）的地址；`onyx/web/vite.config.ts` 的代理默认打
  `http://127.0.0.1:8787`。用默认 8000 会既撞引擎又让前端拿不到数据。
- **Vite 只监听 `[::1]:5173`**：探活用 `http://localhost:5173/`；用 `127.0.0.1` 恒返回 000，
  那是 IPv6-only 绑定，不是服务坏了。后端绑 127.0.0.1，用 IP 探活。
- **provider 三选一**（`onyx plugins` 的 `onyx.providers`）：`ollama`（默认，引擎
  `http://127.0.0.1:11434`，会采显存）、`openai-compat`（要 `--url http://host:port/v1`）、
  `mock`（不碰 GPU，纯看板/UI 开发）。
- **GPU 锁是机器级的** `%TEMP%/onyx-gpu.lock`（不在 `.data` 里）：`serve` 空闲时不持锁，
  只在 Playground/eval 请求期间持有，stale 阈值 600s。空闲态 `/api/gpu` 返回 `busy:false` 是正常。
- **没有 `--reload`**：`onyx serve` 未实现该 flag，附录 B 里 `make dev` 那行是设想，
  Makefile 的 `dev` 也只有 .PHONY 名没有规则。改后端代码要重启进程。

## 流程

1. **预检**（并行）：
   - `netstat -ano | grep -E ":(8787|5173)\s"` —— 已在 LISTENING 就别重复起，直接跳到第 4 步探活并汇报现状。
   - 引擎：`curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:11434/api/tags`。
     非 200 且用户要的是 ollama：停下来报告"Ollama 没在跑"，给修法（Windows 服务
     `Start-Service OllamaService`，或前台 `ollama serve`），**不要静默改用 mock 糊过去**。
2. **体检 + 建库**（首次，或 doctor 报「数据库已迁移 ✗」）：
   - `uv run --no-sync onyx doctor` —— 它查 Python/数据目录可写/schema 版本/blob 引用完整/插件可加载/引擎可达。有 ✗ 先修再起服务。
   - `uv run --no-sync onyx db init`。只想看环境不看网络时给 doctor 加 `--skip-network`。
3. **起后端**：`run_in_background` 执行 `uv run --no-sync onyx serve --port 8787`
   （换 provider 时补 `--provider/--url`，需要导出事件时补 `--sink jsonl`）。记下返回的 task id。
4. **后端就绪**：轮询到 200（≤30s），然后读 body 确认 `provider_reachable:true`——
   200 只证明 uvicorn 活着，`provider_reachable` 才证明看板有引擎。

   ```bash
   for i in $(seq 1 30); do
     [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:8787/api/health)" = 200 ] && break; sleep 1
   done; curl -s http://127.0.0.1:8787/api/health
   ```
5. **起前端**：`run_in_background` 在 `onyx/web` 执行 `npm run dev`（缺 `node_modules` 先 `npm install`）；
   用同样的轮询打 `http://localhost:5173/`。
6. **验代理链**：`curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/api/health` 必须 200。
   这一步区分"两个进程都活着"和"前端真的能拿到数据"；502/000 说明后端端口与 `ONYX_API` 不匹配。
7. **汇报**：看板 http://localhost:5173 · API 文档 http://127.0.0.1:8787/api/docs ·
   `/api/health` 里的 engine_version 与 schema_version · `/api/gpu` 的忙闲。附两个 task id 便于停止。

## 停止

**TaskStop 不够，必须按端口复核。** 实测：停 vite 的那次 TaskStop 报成功，但外层 shell 死了、
`node.exe` 仍占着 `[::1]:5173`（探活还是 200）；停后端那次报"无法终止部分子进程"，
而 8787 确实已释放。两个方向的错都出现过，所以停止的验收判据是端口，不是工具返回。

1. `netstat -ano | grep -E ":(8787|5173)\s+.*LISTENING"` —— 只剩 TIME_WAIT 是正常收尾，有 LISTENING 就没停干净。
2. 对残留 PID 先用 `powershell -NoProfile -Command "Get-Process -Id <pid> | Select-Object Id,ProcessName,Path"`
   确认归属（应为 uv/python 或 `D:\Softwares\NodeJS\node.exe`），再 `taskkill //PID <pid> //F`
   （Git Bash 里斜杠要 doubling，已实测可用）。**确认归属再起手，别照着端口就杀。**
3. 终端起的直接 Ctrl-C。

## 坑

- `uv run` 默认可能去同步依赖并卡网络：始终带 `--no-sync`（`.venv` 里 api/runtime 已装齐）。
  真需要新 extra 时 `uv sync --extra api` 并设 `UV_HTTP_TIMEOUT=120`。
- 单元/契约测试（`uv run pytest -q`）不需要任何服务；`-m live` 与 `-m e2e` 才要求 Ollama 在跑。
- 换 provider 会换 `provider_id`（ollama → `ollama-local`，其它 → `{provider}-local`），
  trace 按它归组，看板口径不延续上一次 —— 汇报时说清这次连的是哪个。
- 起了服务却什么都不显示，先分清三层：`/api/health`（后端）、`localhost:5173`（前端）、
  `localhost:5173/api/health`（代理）。异常码出现在哪一层就修哪一层。
