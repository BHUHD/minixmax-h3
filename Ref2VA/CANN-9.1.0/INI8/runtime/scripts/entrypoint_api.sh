#!/usr/bin/env bash
# 容器 entrypoint：清理后启动 FastAPI（NPU serve 需另进程或 compose 侧启动）
set -euo pipefail

export H3_JOB_ROOT="${H3_JOB_ROOT:-/workspace/jobs}"
mkdir -p "$H3_JOB_ROOT"

echo "[entrypoint] cleanup job cache under $H3_JOB_ROOT (via API startup)"
exec bash /workspace/scripts/run_api.sh
