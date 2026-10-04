#!/usr/bin/env bash
# 扩展点边界检查的 shell 入口（计划里点名的就是这个文件名；实现是 Python，
# 这样 Windows 上也能跑，并且能被单元测试直接调用纯函数部分）。
#
#   ./scripts/check_extension_boundary.sh              # 检查 HEAD
#   ./scripts/check_extension_boundary.sh --base v0 --head HEAD
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python scripts/check_extension_boundary.py "$@"
