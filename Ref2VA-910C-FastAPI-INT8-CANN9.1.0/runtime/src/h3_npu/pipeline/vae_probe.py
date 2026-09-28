"""Decode zero / random latents with the Video VAE to isolate DiT vs VAE garbage."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")

import torch
from PIL import Image

from h3_npu.model.video_vae import load_video_vae
from h3_npu.runtime.h2d import warmup_npu


def _save(path, video):
    x = video[0].detach().float().clamp(-1, 1)
    x = ((x + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    Image.fromarray(x[0]).save(path)
    print(f"wrote {path} shape={tuple(video.shape)} min={float(video.min()):.3f} max={float(video.max()):.3f}", flush=True)


def main():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    warmup_npu(device)
    vae = load_video_vae(
        Path(os.environ.get("H3_MODELS", "/models/h3_quant")) / "vae" / "minimax_h3_video_vae_fp16.safetensors",
        device=device,
        dtype=torch.float16,
    )
    z0 = torch.zeros(1, 24, 17, 16, 16, device=device, dtype=torch.float16)
    zr = torch.randn(1, 24, 17, 16, 16, device=device, dtype=torch.float16)
    out = Path("/workspace/out")
    _save(out / "vae_zero.png", vae.decode(z0))
    _save(out / "vae_randn.png", vae.decode(zr))
    print("vae probe ok", flush=True)


if __name__ == "__main__":
    main()
