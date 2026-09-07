"""Serialized host→NPU copies that avoid torch_npu H2D hangs.

Do not initialize NPU in a parent that sees many dies. Warm up int8/bf16 on the
pinned die before moving 20G INT8 packs.

Empty Parameter(0) `.to(npu)` hangs on this 910C runtime — leave those on CPU
(generate's resident check already skips numel==0).
"""
from __future__ import annotations

import gc
import os

import torch
import torch.nn as nn

_LOG_LEFT = 0
_WARMED = False


def warmup_npu(device: torch.device) -> None:
    """Force allocator + int8 H2D path before the first QuantLinear pack."""
    global _WARMED
    if _WARMED:
        return
    print(f"[h3_npu] NPU warmup on {device} vis={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    import torch_npu  # noqa: F401

    torch.npu.set_device(device.index if device.index is not None else 0)
    x = torch.zeros(256, 256, device=device, dtype=torch.bfloat16)
    y = torch.arange(256 * 256, dtype=torch.int8).reshape(256, 256).to(device)
    z = (x + y.to(torch.bfloat16)).sum()
    torch.npu.synchronize()
    print(f"[h3_npu] NPU warmup done checksum={float(z):.1f} i8={int(y.float().sum())}", flush=True)
    for mb in (16,):
        n = mb * 1024 * 1024
        print(f"[h3_npu] warmup one-shot {mb}MiB int8 H2D", flush=True)
        t = torch.zeros(n, dtype=torch.int8).to(device)
        torch.npu.synchronize()
        print(f"[h3_npu] warmup {mb}MiB ok device={t.device}", flush=True)
        del t
        torch.npu.empty_cache()
    del x, y, z
    torch.npu.empty_cache()
    _WARMED = True


def _copy_to_npu(src: torch.Tensor, device: torch.device) -> torch.Tensor:
    global _LOG_LEFT
    if src.device == device:
        return src
    if src.device.type == "meta" or src.numel() == 0:
        return src
    # Clone off safetensors mmap; DMA from file-backed storage hangs on some shapes.
    cpu = src.detach().contiguous().clone()
    nbytes = cpu.numel() * cpu.element_size()
    log = _LOG_LEFT > 0
    if log:
        print(f"[h3_npu] H2D {tuple(cpu.shape)} {cpu.dtype} {nbytes}B", flush=True)
        _LOG_LEFT = max(0, _LOG_LEFT - 1)
    out = cpu.to(device, non_blocking=True)
    del cpu
    if log:
        print(f"[h3_npu] H2D done → {out.device}", flush=True)
    return out


def module_to_npu(module: nn.Module, device: torch.device, *, label: str = "") -> None:
    """Move parameters/buffers to NPU. Skip empty shells (numel=0 hangs on 910C)."""
    for name, param in list(module.named_parameters(recurse=False)):
        if param is None or param.numel() == 0:
            continue
        module._parameters[name] = nn.Parameter(_copy_to_npu(param, device), requires_grad=False)
    for name, buf in list(module.named_buffers(recurse=False)):
        if buf is None or buf.numel() == 0:
            continue
        module._buffers[name] = _copy_to_npu(buf, device)
    for child in module.children():
        module_to_npu(child, device, label=label)
    if label:
        print(f"[h3_npu] H2D {label} → {device}", flush=True)


def dit_blocks_to_npu(model: nn.Module, device: torch.device) -> None:
    n_blocks = len(model.blocks)
    for i, block in enumerate(model.blocks):
        if i == 0 or i + 1 == n_blocks or (i + 1) % 10 == 0:
            print(f"[h3_npu] H2D block {i + 1}/{n_blocks}", flush=True)
        module_to_npu(block, device)
        if i == 0:
            print("[h3_npu] H2D block 1 done", flush=True)
        if (i + 1) % 10 == 0:
            torch.npu.synchronize()
            gc.collect()
    print("[h3_npu] H2D token_refiner + final_layer + dense", flush=True)
    module_to_npu(model.token_refiner, device)
    module_to_npu(model.final_layer, device)
    for name in ("video_patch_proj", "audio_patch_proj", "condition_proj"):
        module_to_npu(getattr(model, name), device)
    if hasattr(model, "adaln_t_table") and model.adaln_t_table is not None:
        model.adaln_t_table = _copy_to_npu(model.adaln_t_table, device)
    if hasattr(model, "rope") and hasattr(model.rope, "inv_freq"):
        model.rope.inv_freq = _copy_to_npu(model.rope.inv_freq, device)
    gc.collect()
    torch.npu.synchronize()
    print(f"[h3_npu] DiT resident on {device} (sync deferred to first forward)", flush=True)
    n_cube = 0
    from h3_npu.modules.linear import QuantLinear

    for m in model.modules():
        if isinstance(m, QuantLinear):
            m.prepare_cube()
            if getattr(m, "weight_layout", "nk") != "nk":
                n_cube += 1
    print(f"[h3_npu] INT8 Cube packed {n_cube} linears (layout kn/nz)", flush=True)
