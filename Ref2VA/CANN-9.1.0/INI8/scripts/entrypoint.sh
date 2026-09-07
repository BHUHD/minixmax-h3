#!/usr/bin/env bash
# INT8 容器入口：16-die serve + FastAPI Worker
set -euo pipefail
exec bash /workspace/scripts/entrypoint_serve_api.sh
