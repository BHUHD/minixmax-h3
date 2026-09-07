"""5s low-res DiT smoke + NVFP4 layer probe (container-only)."""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

# Allow `python -m h3_npu.pipeline.smoke` when src is on PYTHONPATH
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from h3_npu.load.safetensors_quant import load_nvfp4_linear_smoke, load_pruned_dit
from h3_npu.runtime.device import empty_cache, init_npu, sync
from h3_npu.runtime.accel import maybe_torchair_compile


DEFAULT_MODELS = Path(os.environ.get("H3_MODELS", "/models/h3_quant"))


def _latent_shapes(seconds: float, height: int, width: int, fps: float = 24.0):
    # Video VAE: ~ temporal 4x, spatial 8x (Comfy MiniMax H3); audio 40Hz stereo pack.
    frames = max(1, int(round(seconds * fps)))
    # latent T ≈ frames/4 with patch t=1
    lt = max(1, (frames + 3) // 4)
    lh = max(2, height // 8)
    lw = max(2, width // 8)
    # make even for patch 2x2
    lh += lh % 2
    lw += lw % 2
    audio_t = max(1, int(round(seconds * 40)))
    return lt, lh, lw, audio_t


@torch.inference_mode()
def smoke_dit(
    models_dir: Path,
    *,
    device: torch.device,
    seconds: float,
    height: int,
    width: int,
    steps: int,
    text_len: int,
) -> None:
    dit_path = models_dir / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    print(f"[smoke] load DiT {dit_path}", flush=True)
    t0 = time.time()
    model = load_pruned_dit(dit_path, device=device, dtype=torch.bfloat16)
    model = maybe_torchair_compile(model)
    print(f"[smoke] DiT ready in {time.time() - t0:.1f}s", flush=True)

    lt, lh, lw, audio_t = _latent_shapes(seconds, height, width)
    print(f"[smoke] latent video=1x24x{lt}x{lh}x{lw} audio=1x32x2x{audio_t} text_len={text_len}", flush=True)

    video = torch.randn(1, 24, lt, lh, lw, device=device, dtype=torch.bfloat16)
    audio = torch.randn(1, 32, 2, audio_t, device=device, dtype=torch.bfloat16)
    # Fake Qwen layer-50 states (real TE path: encode separately)
    context = torch.randn(1, text_len, 5120, device=device, dtype=torch.bfloat16)

    # warmup
    ts = torch.tensor([700.0], device=device)
    _ = model([video, audio], ts, context)
    sync()
    empty_cache()

    step_times = []
    for i in range(steps):
        ts = torch.tensor([1000.0 * (1.0 - (i + 1) / (steps + 1))], device=device)
        sync()
        t1 = time.time()
        out = model([video, audio], ts, context)
        sync()
        dt = time.time() - t1
        step_times.append(dt)
        v_norm = float(out[0][0].float().norm())
        print(f"[smoke] step {i+1}/{steps} {dt:.3f}s |v|={v_norm:.4f}", flush=True)
        # Euler-ish update on video only for smoke
        video = video + 0.05 * out[0][0].to(video.dtype)

    print(
        f"[smoke] DiT OK avg_step={sum(step_times)/len(step_times):.3f}s "
        f"min={min(step_times):.3f}s max={max(step_times):.3f}s",
        flush=True,
    )
    del model
    empty_cache()


@torch.inference_mode()
def smoke_nvfp4(models_dir: Path, *, device: torch.device) -> None:
    te_path = models_dir / "text_encoders" / "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    print(f"[smoke] NVFP4 layer probe {te_path}", flush=True)
    layer = load_nvfp4_linear_smoke(te_path, device=device)
    x = torch.randn(4, layer.in_features, device=device, dtype=torch.bfloat16)
    y = layer(x)
    print(f"[smoke] NVFP4 OK out={tuple(y.shape)} dtype={y.dtype}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="H3 quant NPU smoke (no ComfyUI)")
    p.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--text-len", type=int, default=64)
    p.add_argument("--skip-dit", action="store_true")
    p.add_argument("--skip-nvfp4", action="store_true")
    args = p.parse_args(argv)

    device = init_npu(args.device_id)
    print(f"[smoke] device={device} models={args.models}", flush=True)

    if not args.skip_nvfp4:
        smoke_nvfp4(args.models, device=device)
        empty_cache()
    if not args.skip_dit:
        smoke_dit(
            args.models,
            device=device,
            seconds=args.seconds,
            height=args.height,
            width=args.width,
            steps=args.steps,
            text_len=args.text_len,
        )
    print("[smoke] done", flush=True)


if __name__ == "__main__":
    main()
