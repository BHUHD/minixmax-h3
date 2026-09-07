#!/usr/bin/env bash
# FastAPI Worker 启动（mock 或真实 serve 模式）
set -euo pipefail

WS="${WORKSPACE_ROOT:-/workspace}"
export PYTHONPATH="${WS}/src:${WS}/api:${WS}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 默认 mock 便于无 NPU 调试；容器生产设 H3_API_MOCK=0
export H3_API_MOCK="${H3_API_MOCK:-1}"
export H3_JOB_ROOT="${H3_JOB_ROOT:-/workspace/jobs}"
export H3_API_PORT="${H3_API_PORT:-8080}"
export H3_MOCK_STEP_DELAY="${H3_MOCK_STEP_DELAY:-0.02}"

if [[ -f /workspace/scripts/hdk/cann_env.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/scripts/hdk/cann_env.sh
elif [[ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.1.0/set_env.sh
fi

echo "=== H3 Ref2VA FastAPI Worker ==="
echo "MOCK=$H3_API_MOCK PORT=$H3_API_PORT JOB_ROOT=$H3_JOB_ROOT"

exec python3 -u /workspace/scripts/run_api.py
