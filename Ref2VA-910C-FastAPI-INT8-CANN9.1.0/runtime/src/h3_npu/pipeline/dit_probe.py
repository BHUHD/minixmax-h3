"""Load DiT and print whether residual blocks actually move hidden states."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/src")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "5")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
os.environ["H3_NPU_LAYERWISE_OFFLOAD"] = "0"
os.environ["H3_NPU_INT8_FORCE_DEQUANT"] = "1"

import torch

from h3_npu.load.safetensors_quant import load_pruned_dit_streaming
from h3_npu.modules.linear import QuantLinear


def _stats(name, t):
    if t is None or t.numel() == 0:
        print(f"  {name}: EMPTY", flush=True)
        return
    x = t.detach().float()
    print(
        f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
        f"mean={float(x.mean()):.5f} std={float(x.std()):.5f} "
        f"min={float(x.min()):.5f} max={float(x.max()):.5f}",
        flush=True,
    )


def main():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    path = Path(os.environ.get("H3_MODELS", "/models/h3_quant")) / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    model = load_pruned_dit_streaming(path, device=device, dtype=torch.bfloat16)
    print("[probe] key tensors", flush=True)
    _stats("adaln_t_table", model.adaln_t_table)
    _stats("rope.inv_freq", model.rope.inv_freq)
    _stats("video_patch_proj.weight", model.video_patch_proj.weight)
    _stats("condition_proj.weight", model.condition_proj.weight)
    blk = model.blocks[0]
    _stats("block0.adaln_proj.linear.weight", blk.adaln_proj.linear.weight)
    _stats("block0.adaln_proj.linear.bias", blk.adaln_proj.linear.bias)
    qkv = blk.attn.qkv_proj
    print(f"  qkv type={type(qkv).__name__} quant={getattr(qkv, 'quant_format', None)} convrot={getattr(qkv, 'convrot', None)}", flush=True)
    if isinstance(qkv, QuantLinear):
        _stats("block0.qkv.weight_i8", qkv.weight_i8)
        _stats("block0.qkv.weight_scale", qkv.weight_scale)
    _stats("block0.attn.q_norm.weight", blk.attn.q_norm.weight)
    _stats("block0.norm1.weight", blk.norm1.weight)

    # tiny packed forward: 1 video frame-token grid + dummy text/audio
    torch.manual_seed(0)
    video = torch.randn(1, 24, 2, 4, 4, device=device, dtype=torch.bfloat16)
    audio = torch.randn(1, 32, 2, 4, device=device, dtype=torch.bfloat16)
    context = torch.randn(1, 4, 5120, device=device, dtype=torch.bfloat16)
    tags = torch.ones(4, device=device, dtype=torch.long)
    ts = torch.tensor([1000.0], device=device)
    out = model([video, audio], ts, context, minimax_payload={"text_token_tags": tags})
    _stats("out_video", out[0])
    _stats("out_audio", out[1])
    print("[probe] dit ok", flush=True)


if __name__ == "__main__":
    main()
