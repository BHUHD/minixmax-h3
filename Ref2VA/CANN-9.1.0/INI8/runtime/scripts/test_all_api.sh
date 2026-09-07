#!/usr/bin/env bash
# 全量实测：单元模拟 + 双进程 E2E + Gateway 客户端
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo ">>> [1/3] unit simulation"
python3 tests/test_api_simulation.py
echo ">>> [2/3] dual_mock e2e (serve + FastAPI)"
python3 tests/test_api_e2e.py --mode dual_mock
echo ">>> [3/3] api_mock e2e"
python3 tests/test_api_e2e.py --mode api_mock
echo ">>> ALL TESTS PASSED"
