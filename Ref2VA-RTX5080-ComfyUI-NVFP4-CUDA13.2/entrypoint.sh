#!/usr/bin/env bash
# MiniMax-H3 Ref2VA ComfyUI 入口：与 FL2VA NVFP4 镜像一致的启动参数
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility,video}"

cd /app/ComfyUI

# 将镜像内捆绑的 UI 工作流种子到 user/（不覆盖已有同名文件）
if [ -d /opt/comfyui/bundled_workflows ]; then
  mkdir -p /app/ComfyUI/user/default/workflows
  for f in /opt/comfyui/bundled_workflows/*; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    dest="/app/ComfyUI/user/default/workflows/${base}"
    if [ ! -f "$dest" ]; then
      cp -a "$f" "$dest"
      echo "[entrypoint] seeded workflow: ${base}"
    fi
  done
fi

echo "[entrypoint] torch/cuda check..."
python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device0={torch.cuda.get_device_name(0)}")
PY

exec python main.py \
  --listen "${COMFY_LISTEN:-0.0.0.0}" \
  --port "${COMFY_PORT:-8188}" \
  --disable-cuda-malloc \
  "$@"
