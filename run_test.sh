#!/usr/bin/env bash
# run_test.sh —— 一键测试脚本
# 调用 test/run_with_log.py 把 run.py 包起来，
# 把 stdout/stderr 落到 ../output/<时间戳>/ 下，便于改进对比。
#
# 用法
#   ./run_test.sh                       # 默认参数（runs=4, workers=20）
#   ./run_test.sh --runs 1 --workers 50 # 任意参数会原样透传给 run.py
#
# 兼容 Git Bash / WSL / Linux / macOS。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    echo "未找到 python，可执行：export PYTHON=/path/to/python 后重试" >&2
    exit 1
  fi
fi

exec "$PYTHON" "test/run_with_log.py" "$@"
