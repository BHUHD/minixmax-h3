#!/usr/bin/env bash
# 准备 FL2VA 模型目录：TE/VAE 软链 Ref2VA，DiT 从 HuggingFace 下载（约 21GB）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
REF="${ROOT}/models/ComfyUI/INI8/Ref2VA"
FL2="${ROOT}/models/ComfyUI/INI8/FL2VA"
DIT="minimax_h3_fl2va_pruned_int8_convrot.safetensors"
HF_REPO="Comfy-Org/MiniMax-H3"
HF_FILE="diffusion_models/${DIT}"

mkdir -p "${FL2}/diffusion_models" "${FL2}/text_encoders" "${FL2}/vae"

link_if_missing() {
  local src="$1" dst="$2"
  if [[ -e "$dst" ]]; then
    return 0
  fi
  if [[ ! -e "$src" ]]; then
    echo "[setup] 缺少 Ref2VA 源: $src" >&2
    exit 1
  fi
  ln -s "$(realpath "$src")" "$dst"
  echo "[setup] ln -s -> $dst"
}

link_if_missing "${REF}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
  "${FL2}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
link_if_missing "${REF}/vae/minimax_h3_video_vae_fp16.safetensors" \
  "${FL2}/vae/minimax_h3_video_vae_fp16.safetensors"
link_if_missing "${REF}/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "${FL2}/vae/minimax_h3_audio_vae_fp32.safetensors"

DEST="${FL2}/diffusion_models/${DIT}"
if [[ -f "$DEST" ]]; then
  echo "[setup] DiT 已存在: $DEST ($(du -h "$DEST" | cut -f1))"
  exit 0
fi

echo "[setup] 下载 FL2VA DiT: ${HF_REPO}/${HF_FILE}"
echo "[setup] 目标: $DEST"
echo "[setup] 若 huggingface.co 不可达，可: export HF_ENDPOINT=https://hf-mirror.com"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_REPO" "$HF_FILE" --local-dir "$FL2" --local-dir-use-symlinks False
elif python3 -c "import huggingface_hub" 2>/dev/null; then
  python3 - <<PY
import os
from huggingface_hub import hf_hub_download
from pathlib import Path
p = hf_hub_download(
    repo_id="${HF_REPO}",
    filename="${HF_FILE}",
    local_dir="${FL2}",
    resume_download=True,
)
print("[setup] saved", p)
PY
else
  echo "[setup] 请安装 huggingface-cli 或 huggingface_hub 后重试" >&2
  echo "  pip install -U huggingface_hub" >&2
  echo "  huggingface-cli download ${HF_REPO} ${HF_FILE} --local-dir ${FL2}" >&2
  exit 1
fi

if [[ -f "$DEST" ]]; then
  echo "[setup] 完成: $DEST"
else
  echo "[setup] 下载后未找到 $DEST，请检查 huggingface 输出路径" >&2
  exit 1
fi
