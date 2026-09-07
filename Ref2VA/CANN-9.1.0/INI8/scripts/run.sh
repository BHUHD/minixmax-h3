#!/usr/bin/env bash
# 启动 INT8 16-die serve，并等待 READY
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 释放可能占卡的同类容器
for c in minimax-h3-quant-npu-serve minimax-h3-ref2va-bf16-npu8 minimax-h3-ref2va-serve minimax-h3-mindie-npu-skip4; do
  if docker ps -q -f "name=^${c}$" | grep -q .; then
    echo "停止占卡容器: $c"
    docker stop "$c" >/dev/null || true
  fi
done

# 确保镜像 tag（与 quant-cann910:v1 同 ID）
if ! docker image inspect minimax-h3:ref2va-cann9.1.0-bf16 >/dev/null 2>&1; then
  if docker image inspect minimax-h3-quant-cann910:v1 >/dev/null 2>&1; then
    docker tag minimax-h3-quant-cann910:v1 minimax-h3:ref2va-cann9.1.0-bf16
    echo "已 tag: minimax-h3-quant-cann910:v1 → minimax-h3:ref2va-cann9.1.0-bf16"
  else
    echo "ERROR: 缺少镜像 minimax-h3:ref2va-cann9.1.0-bf16 / minimax-h3-quant-cann910:v1" >&2
    exit 1
  fi
fi

mkdir -p "$ROOT/debug/out" "$ROOT/debug/input"
if [[ ! -f "$ROOT/debug/input/ref_golden_retriever.png" ]]; then
  echo "ERROR: 缺少参考图 debug/input/ref_golden_retriever.png" >&2
  exit 1
fi

docker compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true
docker compose -f docker-compose.yml up -d --force-recreate

echo "等待 serve READY（最多 1800s）..."
READY="$ROOT/debug/out/h3_serve/READY"
for i in $(seq 1 360); do
  if [[ -f "$READY" ]]; then
    echo "READY (${i}x5s)"
    docker compose -f docker-compose.yml ps
    exit 0
  fi
  st=$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}}' minimax-h3-ref2va-int8-npu16 2>/dev/null || echo missing)
  if [[ "$st" != running* ]]; then
    echo "容器异常: $st" >&2
    docker logs --tail 80 minimax-h3-ref2va-int8-npu16 2>&1 | tail -80
    exit 1
  fi
  if (( i % 12 == 0 )); then
    echo "  …仍在启动 (${i}x5s) status=$st"
    docker logs --tail 5 minimax-h3-ref2va-int8-npu16 2>&1 | tail -5 || true
  fi
  sleep 5
done
echo "READY 超时" >&2
docker logs --tail 100 minimax-h3-ref2va-int8-npu16 2>&1 | tail -100
exit 1
