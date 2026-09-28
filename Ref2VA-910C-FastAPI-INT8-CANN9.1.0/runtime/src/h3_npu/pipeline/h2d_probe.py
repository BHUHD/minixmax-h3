"""Single-die resident H2D probe (no torchrun). Pin vis before import via env."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/workspace/src")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "1")
os.environ.setdefault("ASCEND_VISIBLE_DEVICES", os.environ["ASCEND_RT_VISIBLE_DEVICES"])
os.environ.setdefault("ASCEND_DEVICE_ID", "0")
os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:False")
os.environ["H3_NPU_LAYERWISE_OFFLOAD"] = "0"

import torch  # noqa: E402

from h3_npu.load.safetensors_quant import load_pruned_dit_streaming  # noqa: E402
from h3_npu.runtime.h2d import warmup_npu  # noqa: E402


def main() -> None:
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    warmup_npu(device)
    models = Path(os.environ.get("H3_MODELS", "/models/h3_quant"))
    dit = models / "diffusion_models" / "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    t0 = time.time()
    model = load_pruned_dit_streaming(dit, device=device, dtype=torch.bfloat16)
    cpu = [
        n
        for n, t in list(model.named_parameters()) + list(model.named_buffers())
        if t is not None and t.numel() > 0 and t.device.type != "npu"
    ]
    if cpu:
        raise RuntimeError(f"not resident on NPU: {cpu[:8]}")
    print(f"[h2d_probe] DiT resident in {time.time() - t0:.1f}s", flush=True)
    x = torch.zeros(1, device=device, dtype=torch.bfloat16)
    _ = x + 1
    torch.npu.synchronize()
    print("[h2d_probe] ok", flush=True)


if __name__ == "__main__":
    main()
