#!/usr/bin/env bash
# 双进程 entrypoint：NPU serve（或 mock serve）+ FastAPI Worker
set -euo pipefail

WS="${WORKSPACE_ROOT:-/workspace}"
export PYTHONPATH="${WS}/src:${WS}/api:${WS}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export H3_JOB_ROOT="${H3_JOB_ROOT:-${WS}/jobs}"
export H3_SERVE_DIR="${H3_SERVE_DIR:-${WS}/out/h3_serve}"
export H3_API_PORT="${H3_API_PORT:-8080}"
export H3_API_MOCK="${H3_API_MOCK:-0}"
export H3_MOCK_SERVE="${H3_MOCK_SERVE:-0}"

mkdir -p "$H3_JOB_ROOT" "$H3_SERVE_DIR" "${H3_OUTPUT_DIR:-${WS}/output}"

# CANN 环境（容器内）
if [[ -f "${WS}/scripts/hdk/cann_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WS}/scripts/hdk/cann_env.sh"
elif [[ -f /usr/local/Ascend/cann-9.1.0/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann-9.1.0/set_env.sh
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/cann/set_env.sh
fi

SERVE_PID=""
API_PID=""

_cleanup() {
  echo "[entrypoint] shutdown..."
  [[ -n "$API_PID" ]] && kill "$API_PID" 2>/dev/null || true
  [[ -n "$SERVE_PID" ]] && kill "$SERVE_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap _cleanup EXIT INT TERM

_start_serve() {
  if [[ "$H3_API_MOCK" == "1" ]]; then
    echo "[entrypoint] H3_API_MOCK=1, skip serve"
    return 0
  fi
  if [[ "$H3_MOCK_SERVE" == "1" ]]; then
    echo "[entrypoint] starting mock serve (no NPU)"
    python3 -u "${WS}/scripts/mock_serve.py" &
    SERVE_PID=$!
    return 0
  fi
  echo "[entrypoint] starting NPU serve (16-die)"
  export H3_SERVE=1
  export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
  export H3_FA_BACKEND="${H3_FA_BACKEND:-infer_v2}"
  bash "${WS}/scripts/run_serve.sh" &
  SERVE_PID=$!
}

_start_api() {
  echo "[entrypoint] starting FastAPI :${H3_API_PORT} (MOCK=$H3_API_MOCK)"
  python3 -u "${WS}/scripts/run_api.py" &
  API_PID=$!
}

echo "=== H3 Ref2VA dual entrypoint WS=$WS ==="
_start_serve
_start_api

# serve 挂掉时：删 READY，让 FastAPI 任务失败而不是永久卡住
while true; do
  if [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" 2>/dev/null; then
    echo "[entrypoint] API exited"
    break
  fi
  if [[ -n "$SERVE_PID" ]] && ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "[entrypoint] NPU serve exited — clearing READY"
    rm -f "${H3_SERVE_DIR}/READY"
    # 不自动重启 serve（需人工介入）；等 API 超时/探测失败
    SERVE_PID=""
  fi
  sleep 2
done
wait "$API_PID" 2>/dev/null || true
