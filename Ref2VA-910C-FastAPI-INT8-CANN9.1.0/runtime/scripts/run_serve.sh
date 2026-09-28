#!/usr/bin/env bash
# Resident 16-die generate: HCCL + DiT stay up; jobs via submit_generate.py
set -euo pipefail
export PYTHONPATH="/workspace/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export H3_SERVE=1
export H3_SERVE_PREWARM="${H3_SERVE_PREWARM:-1}"
export H3_SKIP_WARMUP="${H3_SKIP_WARMUP:-1}"
exec /workspace/scripts/run_generate.sh "$@"
