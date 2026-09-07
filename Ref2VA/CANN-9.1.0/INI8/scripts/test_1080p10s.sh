#!/usr/bin/env bash
# 对齐 bench_hdk2611：10s / 1920×1088 / 20 step / 同款 prompt，并落盘分阶段耗时
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CNAME="${CNAME:-minimax-h3-ref2va-int8-npu16}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$ROOT/debug/out/bench_hdk2611_replay_${STAMP}}"
mkdir -p "$OUT_DIR/metrics" "$OUT_DIR/logs"

PROMPT='Use <Picture 1> as the character. A golden retriever running along a sunny beach, waves in the background, cinematic lighting'
echo "$PROMPT" >"$OUT_DIR/prompt.txt"

# 等待 READY
READY="$ROOT/debug/out/h3_serve/READY"
if [[ ! -f "$READY" ]]; then
  echo "serve 未 READY，先跑 ./scripts/run.sh" >&2
  exit 1
fi

OUT_MP4="/workspace/out/bench_hdk2611_replay_${STAMP}.mp4"
HOST_MP4="$ROOT/debug/out/bench_hdk2611_replay_${STAMP}.mp4"

echo "=== INT8 16die 1080P 10s 20step ($(date -Iseconds)) ===" | tee "$OUT_DIR/run.log"
echo "container=$CNAME out=$OUT_MP4" | tee -a "$OUT_DIR/run.log"

# 提交任务（容器内）
set +e
docker exec \
  -e H3_STEPS=20 \
  -e H3_SECONDS=10 \
  -e H3_HEIGHT=1088 \
  -e H3_WIDTH=1920 \
  -e H3_PROGRESSIVE=0 \
  -e H3_FA_BACKEND=infer_v2 \
  -e H3_OUT="$OUT_MP4" \
  -e H3_PROMPT="$PROMPT" \
  -e H3_SUBMIT_TIMEOUT=1800 \
  -e H3_SERVE_DIR=/workspace/out/h3_serve \
  "$CNAME" \
  python /workspace/scripts/submit_generate.py \
  2>&1 | tee -a "$OUT_DIR/run.log" | tee "$OUT_DIR/logs/submit.log"
RC=${PIPESTATUS[0]}
set -e

# 收集 metrics：worker 会把同名 .metrics.json 写到 out 旁
METRICS_CANDIDATES=(
  "$ROOT/debug/out/bench_hdk2611_replay_${STAMP}.metrics.json"
  "$HOST_MP4.metrics.json"
)
# 也搜最近生成的 metrics
shopt -s nullglob
for f in "$ROOT/debug/out/"*.metrics.json; do
  METRICS_CANDIDATES+=("$f")
done
shopt -u nullglob

METRICS=""
for f in "${METRICS_CANDIDATES[@]}"; do
  if [[ -f "$f" ]] && rg -q "bench_hdk2611_replay_${STAMP}|video_written" "$f" 2>/dev/null; then
    # prefer matching stamp in out path inside json
    if rg -q "bench_hdk2611_replay_${STAMP}" "$f" 2>/dev/null; then
      METRICS="$f"
      break
    fi
  fi
done
if [[ -z "$METRICS" ]]; then
  # fallback: newest metrics under debug/out
  METRICS=$(ls -t "$ROOT/debug/out/"*.metrics.json 2>/dev/null | head -1 || true)
fi

if [[ -n "$METRICS" && -f "$METRICS" ]]; then
  cp -f "$METRICS" "$OUT_DIR/metrics/raw.metrics.json"
  python3 "$ROOT/scripts/analyze_metrics.py" \
    --metrics "$OUT_DIR/metrics/raw.metrics.json" \
    --out-dir "$OUT_DIR" \
    2>&1 | tee -a "$OUT_DIR/run.log"
else
  echo "WARN: 未找到 metrics.json，仅保留 submit 日志" | tee -a "$OUT_DIR/run.log"
fi

# 保存 docker 尾日志
docker logs --tail 200 "$CNAME" >"$OUT_DIR/logs/docker.tail.log" 2>&1 || true

if [[ -f "$HOST_MP4" ]]; then
  ln -sfn "$(basename "$HOST_MP4")" "$OUT_DIR/output.mp4" 2>/dev/null || cp -f "$HOST_MP4" "$OUT_DIR/output.mp4"
  ls -lh "$HOST_MP4" | tee -a "$OUT_DIR/run.log"
fi

echo "rc=$RC out_dir=$OUT_DIR" | tee -a "$OUT_DIR/run.log"
exit "$RC"
