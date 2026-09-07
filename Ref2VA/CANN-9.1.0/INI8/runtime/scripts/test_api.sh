#!/usr/bin/env bash
# 运行 FastAPI 模拟测试（无需 NPU）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 tests/test_api_simulation.py
